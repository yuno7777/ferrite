//! Sparse attention masking engine (Local, Strided, and BigBird block-sparse patterns).
//!
//! Replaces dense O(N^2) attention with structured sparse connectivity graphs:
//! - Local window attention: preserves near-neighbor linguistic context.
//! - Strided / dilated attention: captures periodic and long-distance dependencies.
//! - Global tokens: designated tokens (e.g. BOS or summary tokens) attend to all tokens and vice versa.
//! - BigBird random block attention: pseudo-random block routing guaranteeing O(1) graph diameter.

use std::collections::BTreeSet;

use crate::ops::{dot, AttentionParams};

/// Specification of a structured sparse attention pattern.
#[derive(Clone, Debug, PartialEq)]
pub enum SparsePattern {
    /// Local sliding window attention.
    LocalWindow { window: usize },
    /// Combined local and strided/dilated attention.
    Strided { window: usize, stride: usize },
    /// BigBird sparse pattern combining local window, strided dilation, global tokens, and random blocks.
    BigBird {
        window: usize,
        stride: usize,
        num_random_blocks: usize,
        block_size: usize,
        global_indices: Vec<usize>,
    },
    /// Explicit block-sparse coordinate list of active `(query_block, key_block)` pairs.
    BlockSparse {
        block_size: usize,
        active_blocks: BTreeSet<(usize, usize)>,
    },
}

/// Sparse attention mask evaluator.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseAttentionMask {
    pub num_tokens: usize,
    pub pattern: SparsePattern,
}

impl SparseAttentionMask {
    /// Constructs a local sliding window attention mask.
    pub fn new_local(num_tokens: usize, window: usize) -> Self {
        Self {
            num_tokens,
            pattern: SparsePattern::LocalWindow {
                window: window.max(1),
            },
        }
    }

    /// Constructs a strided attention mask with local window support.
    pub fn new_strided(num_tokens: usize, window: usize, stride: usize) -> Self {
        Self {
            num_tokens,
            pattern: SparsePattern::Strided {
                window: window.max(1),
                stride: stride.max(1),
            },
        }
    }

    /// Constructs a full BigBird sparse attention mask.
    pub fn new_bigbird(
        num_tokens: usize,
        window: usize,
        stride: usize,
        num_random_blocks: usize,
        block_size: usize,
        global_indices: Vec<usize>,
    ) -> Self {
        Self {
            num_tokens,
            pattern: SparsePattern::BigBird {
                window: window.max(1),
                stride: stride.max(1),
                num_random_blocks,
                block_size: block_size.max(1),
                global_indices,
            },
        }
    }

    /// Constructs a custom block-sparse attention mask from active block coordinate pairs.
    pub fn new_block_sparse(
        num_tokens: usize,
        block_size: usize,
        active_pairs: &[(usize, usize)],
    ) -> Self {
        let active_blocks: BTreeSet<(usize, usize)> = active_pairs.iter().copied().collect();
        Self {
            num_tokens,
            pattern: SparsePattern::BlockSparse {
                block_size: block_size.max(1),
                active_blocks,
            },
        }
    }

    /// Checks if a causal attention connection exists between `query_pos` and historical `key_pos`.
    ///
    /// Always enforces causal constraint: `key_pos <= query_pos`.
    pub fn is_allowed(&self, query_pos: usize, key_pos: usize) -> bool {
        if key_pos > query_pos || query_pos >= self.num_tokens {
            return false;
        }

        match &self.pattern {
            SparsePattern::LocalWindow { window } => query_pos - key_pos < *window,
            SparsePattern::Strided { window, stride } => {
                if query_pos - key_pos < *window {
                    true
                } else {
                    (query_pos - key_pos) % *stride == 0
                }
            }
            SparsePattern::BigBird {
                window,
                stride,
                num_random_blocks,
                block_size,
                global_indices,
            } => {
                // 1. Global token check
                if global_indices.contains(&query_pos) || global_indices.contains(&key_pos) {
                    return true;
                }
                // 2. Local window check
                if query_pos - key_pos < *window {
                    return true;
                }
                // 3. Strided check
                if *stride > 1 && (query_pos - key_pos) % *stride == 0 {
                    return true;
                }
                // 4. Random block routing check
                if *num_random_blocks > 0 && *block_size > 0 {
                    let q_block = query_pos / *block_size;
                    let k_block = key_pos / *block_size;
                    if q_block == k_block {
                        return true;
                    }
                    let total_blocks = q_block + 1;
                    for r in 0..*num_random_blocks {
                        // Deterministic LCG hash for reproducible pseudo-random block connectivity
                        let hash = (q_block as u64)
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add((r as u64).wrapping_mul(1442695040888963407))
                            .wrapping_add(1);
                        let chosen_block = (hash as usize) % total_blocks;
                        if chosen_block == k_block {
                            return true;
                        }
                    }
                }
                false
            }
            SparsePattern::BlockSparse {
                block_size,
                active_blocks,
            } => {
                let q_block = query_pos / *block_size;
                let k_block = key_pos / *block_size;
                active_blocks.contains(&(q_block, k_block))
            }
        }
    }

    /// Returns sorted list of all active `key_pos` indices that `query_pos` attends to.
    pub fn active_keys_for_query(&self, query_pos: usize) -> Vec<usize> {
        let max_k = (query_pos + 1).min(self.num_tokens);
        let mut active = Vec::new();
        for k in 0..max_k {
            if self.is_allowed(query_pos, k) {
                active.push(k);
            }
        }
        active
    }

    /// Applies the sparse attention mask directly to an array of raw attention scores.
    /// Disallowed entries are set to `f32::NEG_INFINITY`.
    pub fn apply_mask(&self, query_pos: usize, scores: &mut [f32]) {
        for (k_pos, score) in scores.iter_mut().enumerate() {
            if !self.is_allowed(query_pos, k_pos) {
                *score = f32::NEG_INFINITY;
            }
        }
    }

    /// Evaluates sparse attention over KV-cache rows, computing query attention
    /// strictly across active keys and values.
    pub fn sparse_attention(
        &self,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        query_pos: usize,
        params: AttentionParams,
        out: &mut [f32],
    ) {
        let AttentionParams {
            row_width,
            offset,
            scale,
            logit_cap,
            v_row_width,
            v_offset,
            ..
        } = params;
        let val_row_width = v_row_width.unwrap_or(row_width);
        let val_offset = v_offset.unwrap_or(offset);
        let q_len = query.len();
        let v_dim = out.len();

        let active_keys = self.active_keys_for_query(query_pos);
        if active_keys.is_empty() {
            out.fill(0.0);
            return;
        }

        let mut scores = Vec::with_capacity(active_keys.len());
        for &past in &active_keys {
            let row = past * row_width + offset;
            let mut score = dot(query, &keys[row..row + q_len]) * scale;
            if let Some(cap) = logit_cap {
                score = cap * (score / cap).tanh();
            }
            scores.push(score);
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if !max_score.is_finite() {
            out.fill(0.0);
            return;
        }

        let mut denom = 0.0f32;
        out.fill(0.0);

        for (&past, &score) in active_keys.iter().zip(&scores) {
            let p = (score - max_score).exp();
            denom += p;
            let v_row = past * val_row_width + val_offset;
            let v_slice = &values[v_row..v_row + v_dim];
            for (out_slot, &v) in out.iter_mut().zip(v_slice) {
                *out_slot += p * v;
            }
        }

        if denom > 0.0 {
            let inv_denom = 1.0 / denom;
            for val in out.iter_mut() {
                *val *= inv_denom;
            }
        }
    }

    /// Computes the overall sparsity ratio in `[0.0, 1.0]`.
    /// 1.0 indicates completely masked out, 0.0 indicates fully dense attention.
    pub fn sparsity(&self) -> f64 {
        if self.num_tokens == 0 {
            return 0.0;
        }
        let total_causal_pairs = (self.num_tokens * (self.num_tokens + 1)) / 2;
        let mut active_pairs = 0;
        for q in 0..self.num_tokens {
            active_pairs += self.active_keys_for_query(q).len();
        }
        1.0 - (active_pairs as f64 / total_causal_pairs as f64)
    }
}
