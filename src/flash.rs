//! Tiled Online-Softmax FlashAttention-2 CPU Engine.
//!
//! Computes exact scaled dot-product attention (single-token decode and
//! multi-token prefill) using the FlashAttention-2 online-softmax recurrence
//! (`m_new = max(m_old, s_block)`, `l_new = l_old * exp(m_old - m_new) + sum(exp(s - m_new))`).
//!
//! Key advantages over naive two-pass attention:
//! - Never materializes the `[n_heads, seq_len, seq_len]` attention score matrix.
//! - Keeps working tiles inside L1 data cache (`TILE_KV = 32` or `64`).
//! - Supports Grouped-Query Attention (GQA), causal masking, sliding-window
//!   attention, and Gemma-2 style logit softcapping (`cap * tanh(score / cap)`).
//! - Supports reading directly from `Q8_0` quantized KV caches without full
//!   FP32 materialization.

use crate::kv::BlockQ8_0;
use crate::ops;

/// Configuration for a FlashAttention-2 forward pass.
#[derive(Clone, Copy, Debug)]
pub struct FlashAttentionConfig {
    /// Number of query attention heads.
    pub n_heads: usize,
    /// Number of key/value attention heads (`n_heads % n_kv_heads == 0`).
    pub n_kv_heads: usize,
    /// Dimension of each attention head (`head_dim`).
    pub head_dim: usize,
    /// Softmax temperature scale (typically `1.0 / (head_dim as f32).sqrt()`).
    pub scale: f32,
    /// Whether to apply a causal mask (`kv_pos <= q_pos`).
    pub causal: bool,
    /// Optional sliding window limit (`q_pos - kv_pos < window`).
    pub sliding_window: Option<usize>,
    /// Optional Gemma-2 style attention logit softcapping (`cap * tanh(x / cap)`).
    pub softcap: Option<f32>,
    /// KV tile size for online softmax blocking (defaults to 32).
    pub kv_tile_size: usize,
}

impl FlashAttentionConfig {
    /// Construct a standard causal GQA configuration.
    pub fn new(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        assert!(n_heads > 0 && n_kv_heads > 0 && n_heads % n_kv_heads == 0);
        assert!(head_dim > 0);
        Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            causal: true,
            sliding_window: None,
            softcap: None,
            kv_tile_size: 32,
        }
    }
}

#[inline]
fn apply_score_transform(raw_dot: f32, scale: f32, softcap: Option<f32>) -> f32 {
    let mut s = raw_dot * scale;
    if let Some(cap) = softcap {
        if cap > 0.0 {
            s = cap * (s / cap).tanh();
        }
    }
    s
}

/// Compute single-query FlashAttention-2 over an FP32 KV cache of length `kv_len`.
///
/// - `q`: `[n_heads * head_dim]`
/// - `k_cache`: `[kv_len * n_kv_heads * head_dim]`
/// - `v_cache`: `[kv_len * n_kv_heads * head_dim]`
/// - `q_pos`: logical sequence position of this query token
/// - `out`: `[n_heads * head_dim]`
pub fn flash_attention_f32(
    cfg: &FlashAttentionConfig,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_len: usize,
    q_pos: usize,
    out: &mut [f32],
) {
    let h_dim = cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * h_dim;
    let gqa_ratio = cfg.n_heads / cfg.n_kv_heads;
    let tile = cfg.kv_tile_size.max(1);

    assert!(q.len() >= cfg.n_heads * h_dim);
    assert!(out.len() >= cfg.n_heads * h_dim);
    assert!(k_cache.len() >= kv_len * kv_dim);
    assert!(v_cache.len() >= kv_len * kv_dim);

    let max_kv = if cfg.causal {
        (q_pos + 1).min(kv_len)
    } else {
        kv_len
    };
    let min_kv = match cfg.sliding_window {
        Some(win) if win > 0 => max_kv.saturating_sub(win),
        _ => 0,
    };

    for h in 0..cfg.n_heads {
        let kv_h = h / gqa_ratio;
        let q_head = &q[h * h_dim..(h + 1) * h_dim];
        let out_head = &mut out[h * h_dim..(h + 1) * h_dim];
        out_head.fill(0.0);

        if min_kv >= max_kv {
            continue;
        }

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;

        let mut t_start = min_kv;
        while t_start < max_kv {
            let t_end = (t_start + tile).min(max_kv);

            for pos in t_start..t_end {
                let kv_offset = pos * kv_dim + kv_h * h_dim;
                let k_vec = &k_cache[kv_offset..kv_offset + h_dim];
                let v_vec = &v_cache[kv_offset..kv_offset + h_dim];

                let raw = ops::dot(q_head, k_vec);
                let score = apply_score_transform(raw, cfg.scale, cfg.softcap);

                if score > running_max {
                    let correction = if running_max == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (running_max - score).exp()
                    };
                    if correction != 1.0 {
                        for o in out_head.iter_mut() {
                            *o *= correction;
                        }
                    }
                    running_sum = running_sum * correction + 1.0;
                    running_max = score;
                    for d in 0..h_dim {
                        out_head[d] += v_vec[d];
                    }
                } else {
                    let w = (score - running_max).exp();
                    running_sum += w;
                    for d in 0..h_dim {
                        out_head[d] += w * v_vec[d];
                    }
                }
            }

            t_start = t_end;
        }

        if running_sum > 0.0 {
            let inv = 1.0 / running_sum;
            for o in out_head.iter_mut() {
                *o *= inv;
            }
        }
    }
}

/// Multi-token prefill FlashAttention-2 over an FP32 KV cache.
///
/// - `q_batch`: `[n_tokens * n_heads * head_dim]`
/// - `k_cache`: `[kv_len * n_kv_heads * head_dim]`
/// - `v_cache`: `[kv_len * n_kv_heads * head_dim]`
/// - `start_pos`: logical sequence position of the first query token in `q_batch`
/// - `out_batch`: `[n_tokens * n_heads * head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_prefill(
    cfg: &FlashAttentionConfig,
    q_batch: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_tokens: usize,
    kv_len: usize,
    start_pos: usize,
    out_batch: &mut [f32],
) {
    let q_dim = cfg.n_heads * cfg.head_dim;
    assert!(q_batch.len() >= n_tokens * q_dim);
    assert!(out_batch.len() >= n_tokens * q_dim);

    for t in 0..n_tokens {
        let q_slice = &q_batch[t * q_dim..(t + 1) * q_dim];
        let out_slice = &mut out_batch[t * q_dim..(t + 1) * q_dim];
        flash_attention_f32(
            cfg,
            q_slice,
            k_cache,
            v_cache,
            kv_len,
            start_pos + t,
            out_slice,
        );
    }
}

/// Compute FlashAttention-2 reading directly from a `Q8_0` quantized KV cache.
///
/// Each KV head vector of length `head_dim` (multiple of 32) is stored as
/// `head_dim / 32` contiguous [`BlockQ8_0`] blocks.
pub fn flash_attention_q8_kv(
    cfg: &FlashAttentionConfig,
    q: &[f32],
    k_blocks: &[BlockQ8_0],
    v_blocks: &[BlockQ8_0],
    kv_len: usize,
    q_pos: usize,
    out: &mut [f32],
) {
    let h_dim = cfg.head_dim;
    assert!(
        h_dim % 32 == 0,
        "Q8_0 FlashAttention requires head_dim to be a multiple of 32"
    );
    let blocks_per_head = h_dim / 32;
    let blocks_per_token = cfg.n_kv_heads * blocks_per_head;
    let gqa_ratio = cfg.n_heads / cfg.n_kv_heads;
    let tile = cfg.kv_tile_size.max(1);

    assert!(k_blocks.len() >= kv_len * blocks_per_token);
    assert!(v_blocks.len() >= kv_len * blocks_per_token);

    let max_kv = if cfg.causal {
        (q_pos + 1).min(kv_len)
    } else {
        kv_len
    };
    let min_kv = match cfg.sliding_window {
        Some(win) if win > 0 => max_kv.saturating_sub(win),
        _ => 0,
    };

    for h in 0..cfg.n_heads {
        let kv_h = h / gqa_ratio;
        let q_head = &q[h * h_dim..(h + 1) * h_dim];
        let out_head = &mut out[h * h_dim..(h + 1) * h_dim];
        out_head.fill(0.0);

        if min_kv >= max_kv {
            continue;
        }

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;

        let mut t_start = min_kv;
        while t_start < max_kv {
            let t_end = (t_start + tile).min(max_kv);
            for pos in t_start..t_end {
                let base_blk = pos * blocks_per_token + kv_h * blocks_per_head;
                let k_head_blks = &k_blocks[base_blk..base_blk + blocks_per_head];
                let v_head_blks = &v_blocks[base_blk..base_blk + blocks_per_head];

                let mut raw = 0.0f32;
                for (b, blk) in k_head_blks.iter().enumerate() {
                    raw += blk.dot_f32(&q_head[b * 32..(b + 1) * 32]);
                }
                let score = apply_score_transform(raw, cfg.scale, cfg.softcap);

                if score > running_max {
                    let correction = if running_max == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (running_max - score).exp()
                    };
                    if correction != 1.0 {
                        for o in out_head.iter_mut() {
                            *o *= correction;
                        }
                    }
                    running_sum = running_sum * correction + 1.0;
                    running_max = score;
                    for (b, blk) in v_head_blks.iter().enumerate() {
                        blk.accumulate_weighted(1.0, &mut out_head[b * 32..(b + 1) * 32]);
                    }
                } else {
                    let w = (score - running_max).exp();
                    running_sum += w;
                    for (b, blk) in v_head_blks.iter().enumerate() {
                        blk.accumulate_weighted(w, &mut out_head[b * 32..(b + 1) * 32]);
                    }
                }
            }
            t_start = t_end;
        }

        if running_sum > 0.0 {
            let inv = 1.0 / running_sum;
            for o in out_head.iter_mut() {
                *o *= inv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_attention(
        cfg: &FlashAttentionConfig,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        kv_len: usize,
        q_pos: usize,
    ) -> Vec<f32> {
        let h_dim = cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * h_dim;
        let gqa = cfg.n_heads / cfg.n_kv_heads;
        let max_kv = if cfg.causal {
            (q_pos + 1).min(kv_len)
        } else {
            kv_len
        };
        let min_kv = match cfg.sliding_window {
            Some(w) if w > 0 => max_kv.saturating_sub(w),
            _ => 0,
        };

        let mut out = vec![0.0f32; cfg.n_heads * h_dim];
        for h in 0..cfg.n_heads {
            let kv_h = h / gqa;
            let q_h = &q[h * h_dim..(h + 1) * h_dim];
            let mut scores = Vec::with_capacity(max_kv - min_kv);
            for pos in min_kv..max_kv {
                let off = pos * kv_dim + kv_h * h_dim;
                let dot = ops::dot(q_h, &k_cache[off..off + h_dim]);
                scores.push(apply_score_transform(dot, cfg.scale, cfg.softcap));
            }
            ops::softmax(&mut scores);
            for (i, pos) in (min_kv..max_kv).enumerate() {
                let off = pos * kv_dim + kv_h * h_dim;
                let w = scores[i];
                for d in 0..h_dim {
                    out[h * h_dim + d] += w * v_cache[off + d];
                }
            }
        }
        out
    }

    #[test]
    fn flash_attention_matches_naive_exact_gqa_and_softcap() {
        let mut cfg = FlashAttentionConfig::new(4, 2, 32);
        cfg.kv_tile_size = 7; // test non-power-of-two tile boundary
        cfg.softcap = Some(30.0);
        cfg.sliding_window = Some(25);

        let kv_len = 40;
        let q_pos = 33;
        let q: Vec<f32> = (0..4 * 32).map(|i| ((i as f32) * 0.13).sin()).collect();
        let k: Vec<f32> = (0..kv_len * 2 * 32)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();
        let v: Vec<f32> = (0..kv_len * 2 * 32)
            .map(|i| ((i as f32) * 0.11).sin())
            .collect();

        let expected = naive_attention(&cfg, &q, &k, &v, kv_len, q_pos);
        let mut actual = vec![0.0f32; 4 * 32];
        flash_attention_f32(&cfg, &q, &k, &v, kv_len, q_pos, &mut actual);

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "mismatch: actual={a} expected={e}");
        }
    }

    #[test]
    fn flash_attention_q8_kv_matches_f32_closely() {
        let cfg = FlashAttentionConfig::new(2, 2, 32);
        let kv_len = 16;
        let q_pos = 15;
        let q: Vec<f32> = (0..2 * 32).map(|i| ((i as f32) * 0.09).sin()).collect();
        let k_f32: Vec<f32> = (0..kv_len * 2 * 32)
            .map(|i| ((i as f32) * 0.05).cos())
            .collect();
        let v_f32: Vec<f32> = (0..kv_len * 2 * 32)
            .map(|i| ((i as f32) * 0.08).sin())
            .collect();

        let k_blks: Vec<BlockQ8_0> = k_f32.chunks_exact(32).map(BlockQ8_0::quantize).collect();
        let v_blks: Vec<BlockQ8_0> = v_f32.chunks_exact(32).map(BlockQ8_0::quantize).collect();

        let mut out_f32 = vec![0.0f32; 2 * 32];
        let mut out_q8 = vec![0.0f32; 2 * 32];
        flash_attention_f32(&cfg, &q, &k_f32, &v_f32, kv_len, q_pos, &mut out_f32);
        flash_attention_q8_kv(&cfg, &q, &k_blks, &v_blks, kv_len, q_pos, &mut out_q8);

        for (a, e) in out_q8.iter().zip(out_f32.iter()) {
            assert!((a - e).abs() < 0.02, "q8 mismatch: {a} vs {e}");
        }
    }
}
