//! The handful of kernels a transformer decode step needs.
//!
//! All f32, all single-threaded, all obvious. This is the reference the
//! quantized and threaded versions get checked against — when output turns to
//! fluent nonsense, the question is always "does it still match the reference",
//! and that only works if the reference is too simple to be wrong.

/// Root-mean-square normalization, then a per-channel weight.
///
/// The sum is accumulated in f64. In f32 it drifts badly for long vectors, and
/// a normalization that is slightly off poisons everything downstream in a way
/// that reads as a bad model rather than a bug.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());

    let sum: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let scale = 1.0 / ((sum / x.len() as f64) + eps as f64).sqrt();
    for ((slot, value), w) in out.iter_mut().zip(x).zip(weight) {
        *slot = (*value as f64 * scale) as f32 * w;
    }
}

/// In-place softmax. Subtracting the maximum first is what keeps `exp` from
/// overflowing on a confident logit.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let mut total = 0.0;
    for value in x.iter_mut() {
        *value = (*value - max).exp();
        total += *value;
    }
    if total > 0.0 {
        let inverse = 1.0 / total;
        for value in x.iter_mut() {
            *value *= inverse;
        }
    }
}

/// Parameters for [`attention_online`].
#[derive(Clone, Copy, Debug)]
pub struct AttentionParams {
    pub row_width: usize,
    pub offset: usize,
    pub len: usize,
    pub scale: f32,
    pub logit_cap: Option<f32>,
    pub window: Option<usize>,
    pub position: usize,
    pub tile_size: usize,
    pub v_row_width: Option<usize>,
    pub v_offset: Option<usize>,
}

/// Attention over strided KV-cache rows using tiled online softmax.
/// `keys` and `values` contain rows of `row_width` elements; `offset..offset+dim`
/// selects one KV head from each row. Only one tile of scores is held at once.
pub fn attention_online(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    params: AttentionParams,
    out: &mut [f32],
) {
    let AttentionParams {
        row_width,
        offset,
        len,
        scale,
        logit_cap,
        window,
        position,
        tile_size,
        v_row_width,
        v_offset,
    } = params;
    let val_row_width = v_row_width.unwrap_or(row_width);
    let val_offset = v_offset.unwrap_or(offset);
    debug_assert!(tile_size > 0 && offset + query.len() <= row_width);
    debug_assert!(keys.len() >= len * row_width && values.len() >= len * val_row_width);
    debug_assert!(val_offset + out.len() <= val_row_width);
    out.fill(0.0);
    let mut max_score = f32::NEG_INFINITY;
    let mut denominator = 0.0f32;
    let mut stack_scores = [0.0f32; 64];
    let mut heap_scores = if tile_size > 64 {
        vec![0.0f32; tile_size]
    } else {
        Vec::new()
    };
    let scores: &mut [f32] = if tile_size <= 64 {
        &mut stack_scores[..tile_size]
    } else {
        &mut heap_scores[..tile_size]
    };
    for start in (0..len).step_by(tile_size) {
        let end = (start + tile_size).min(len);
        for (i, past) in (start..end).enumerate() {
            if window.is_some_and(|w| position.saturating_sub(past) >= w) {
                scores[i] = f32::NEG_INFINITY;
                continue;
            }
            let row = past * row_width + offset;
            let mut score = dot(query, &keys[row..row + query.len()]) * scale;
            if let Some(cap) = logit_cap {
                score = cap * (score / cap).tanh();
            }
            scores[i] = score;
        }
        let tile_max = scores[..end - start]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if !tile_max.is_finite() {
            continue;
        }
        let next_max = max_score.max(tile_max);
        let old_scale = if max_score.is_finite() {
            (max_score - next_max).exp()
        } else {
            0.0
        };
        denominator *= old_scale;
        for v in out.iter_mut() {
            *v *= old_scale;
        }
        let out_len = out.len();
        for (i, past) in (start..end).enumerate() {
            if !scores[i].is_finite() {
                continue;
            }
            let weight = (scores[i] - next_max).exp();
            denominator += weight;
            let row = past * val_row_width + val_offset;
            for (slot, value) in out.iter_mut().zip(&values[row..row + out_len]) {
                *slot += weight * value;
            }
        }
        max_score = next_max;
    }
    if denominator > 0.0 {
        for value in out {
            *value /= denominator;
        }
    }
}

/// High-throughput tiled online attention with dynamic tile selection and 4-lane unrolled accumulation.
pub fn attention_online_dynamic(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    params: AttentionParams,
    out: &mut [f32],
) {
    let AttentionParams {
        row_width,
        offset,
        len,
        scale,
        logit_cap,
        window,
        position,
        tile_size,
        v_row_width,
        v_offset,
    } = params;
    let val_row_width = v_row_width.unwrap_or(row_width);
    let val_offset = v_offset.unwrap_or(offset);
    debug_assert!(tile_size > 0 && offset + query.len() <= row_width);
    debug_assert!(keys.len() >= len * row_width && values.len() >= len * val_row_width);
    debug_assert!(val_offset + out.len() <= val_row_width);
    out.fill(0.0);
    let mut max_score = f32::NEG_INFINITY;
    let mut denominator = 0.0f32;
    let mut stack_scores = [0.0f32; 128];
    let mut heap_scores = if tile_size > 128 {
        vec![0.0f32; tile_size]
    } else {
        Vec::new()
    };
    let scores: &mut [f32] = if tile_size <= 128 {
        &mut stack_scores[..tile_size]
    } else {
        &mut heap_scores[..tile_size]
    };

    let q_len = query.len();
    let out_len = out.len();

    for start in (0..len).step_by(tile_size) {
        let end = (start + tile_size).min(len);
        for (i, past) in (start..end).enumerate() {
            if window.is_some_and(|w| position.saturating_sub(past) >= w) {
                scores[i] = f32::NEG_INFINITY;
                continue;
            }
            let row = past * row_width + offset;
            let mut score = dot(query, &keys[row..row + q_len]) * scale;
            if let Some(cap) = logit_cap {
                score = cap * (score / cap).tanh();
            }
            scores[i] = score;
        }

        let tile_max = scores[..end - start]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if !tile_max.is_finite() {
            continue;
        }

        let next_max = max_score.max(tile_max);
        let old_scale = if max_score.is_finite() {
            (max_score - next_max).exp()
        } else {
            0.0
        };

        denominator *= old_scale;
        for v in out.chunks_exact_mut(4) {
            v[0] *= old_scale;
            v[1] *= old_scale;
            v[2] *= old_scale;
            v[3] *= old_scale;
        }
        let full_out = out_len - out_len % 4;
        for v in &mut out[full_out..] {
            *v *= old_scale;
        }

        for (i, past) in (start..end).enumerate() {
            if !scores[i].is_finite() {
                continue;
            }
            let weight = (scores[i] - next_max).exp();
            denominator += weight;
            let row = past * val_row_width + val_offset;
            let val_slice = &values[row..row + out_len];

            for (slot, v_chunk) in out.chunks_exact_mut(4).zip(val_slice.chunks_exact(4)) {
                slot[0] += weight * v_chunk[0];
                slot[1] += weight * v_chunk[1];
                slot[2] += weight * v_chunk[2];
                slot[3] += weight * v_chunk[3];
            }
            for (slot, &v) in out[full_out..].iter_mut().zip(&val_slice[full_out..]) {
                *slot += weight * v;
            }
        }
        max_score = next_max;
    }

    if denominator > 0.0 {
        let inv_den = 1.0 / denominator;
        for v in out.chunks_exact_mut(4) {
            v[0] *= inv_den;
            v[1] *= inv_den;
            v[2] *= inv_den;
            v[3] *= inv_den;
        }
        let full_out = out_len - out_len % 4;
        for v in &mut out[full_out..] {
            *v *= inv_den;
        }
    }
}

/// Single context partition state for FlashDecoding parallel attention.
#[derive(Clone, Debug, PartialEq)]
pub struct FlashDecodingSplit {
    pub max_score: f32,
    pub denominator: f32,
    pub output: Vec<f32>,
}

impl FlashDecodingSplit {
    pub fn new(dim: usize) -> Self {
        Self {
            max_score: f32::NEG_INFINITY,
            denominator: 0.0,
            output: vec![0.0; dim],
        }
    }

    /// Combines two split states using online softmax rescaling.
    pub fn reduce(&self, other: &Self) -> Self {
        if !self.max_score.is_finite() && !other.max_score.is_finite() {
            return self.clone();
        }
        if !self.max_score.is_finite() {
            return other.clone();
        }
        if !other.max_score.is_finite() {
            return self.clone();
        }

        let new_max = self.max_score.max(other.max_score);
        let alpha = (self.max_score - new_max).exp();
        let beta = (other.max_score - new_max).exp();

        let new_denom = alpha * self.denominator + beta * other.denominator;
        let mut new_out = vec![0.0f32; self.output.len()];
        for (i, slot) in new_out.iter_mut().enumerate() {
            *slot = alpha * self.output[i] + beta * other.output[i];
        }

        Self {
            max_score: new_max,
            denominator: new_denom,
            output: new_out,
        }
    }

    /// Finalizes the combined split into the normalized attention vector.
    pub fn finalize(&self, out: &mut [f32]) {
        if self.denominator > 0.0 {
            let inv_denom = 1.0 / self.denominator;
            for (slot, &val) in out.iter_mut().zip(&self.output) {
                *slot = val * inv_denom;
            }
        } else {
            out.fill(0.0);
        }
    }
}

/// Logarithmic tree reduction combining S splits in O(log S) rounds.
pub fn tree_reduce_splits(mut splits: Vec<FlashDecodingSplit>) -> FlashDecodingSplit {
    if splits.is_empty() {
        return FlashDecodingSplit {
            max_score: f32::NEG_INFINITY,
            denominator: 0.0,
            output: Vec::new(),
        };
    }
    while splits.len() > 1 {
        let mut next_level = Vec::with_capacity(splits.len().div_ceil(2));
        for chunk in splits.chunks(2) {
            if chunk.len() == 2 {
                next_level.push(chunk[0].reduce(&chunk[1]));
            } else {
                next_level.push(chunk[0].clone());
            }
        }
        splits = next_level;
    }
    splits.pop().unwrap()
}

/// Compute partial attention for a split slice of the KV cache in `[split_start..split_end]`.
pub fn compute_flash_decoding_split(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    params: &AttentionParams,
    split_start: usize,
    split_end: usize,
) -> FlashDecodingSplit {
    let AttentionParams {
        row_width,
        offset,
        scale,
        logit_cap,
        window,
        position,
        v_row_width,
        v_offset,
        ..
    } = *params;
    let val_row_width = v_row_width.unwrap_or(row_width);
    let val_offset = v_offset.unwrap_or(offset);
    let q_len = query.len();
    let v_dim = q_len;

    let split_len = split_end.saturating_sub(split_start);
    if split_len == 0 {
        return FlashDecodingSplit::new(v_dim);
    }

    let mut scores = vec![0.0f32; split_len];
    for (i, past) in (split_start..split_end).enumerate() {
        if window.is_some_and(|w| position.saturating_sub(past) >= w) {
            scores[i] = f32::NEG_INFINITY;
            continue;
        }
        let row = past * row_width + offset;
        let mut score = dot(query, &keys[row..row + q_len]) * scale;
        if let Some(cap) = logit_cap {
            score = cap * (score / cap).tanh();
        }
        scores[i] = score;
    }

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if !max_score.is_finite() {
        return FlashDecodingSplit::new(v_dim);
    }

    let mut denom = 0.0f32;
    let mut out = vec![0.0f32; v_dim];

    for (i, past) in (split_start..split_end).enumerate() {
        let s = scores[i];
        if s > f32::NEG_INFINITY {
            let p = (s - max_score).exp();
            denom += p;
            let v_row = past * val_row_width + val_offset;
            let val_slice = &values[v_row..v_row + v_dim];
            for (out_slot, &v) in out.iter_mut().zip(val_slice) {
                *out_slot += p * v;
            }
        }
    }

    FlashDecodingSplit {
        max_score,
        denominator: denom,
        output: out,
    }
}

/// FlashDecoding parallel attention over partitioned KV cache context splits
/// with logarithmic tree reduction.
pub fn flash_decoding_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    params: AttentionParams,
    threads: usize,
    out: &mut [f32],
) {
    let len = params.len;
    if threads <= 1 || len <= 128 {
        attention_online(query, keys, values, params, out);
        return;
    }

    let num_splits = threads.min(len.div_ceil(64));
    if num_splits <= 1 {
        attention_online(query, keys, values, params, out);
        return;
    }

    let chunk_size = len.div_ceil(num_splits);

    let splits = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..num_splits)
            .map(|split_idx| {
                let split_start = split_idx * chunk_size;
                let split_end = ((split_idx + 1) * chunk_size).min(len);
                let params_ref = &params;
                scope.spawn(move || {
                    compute_flash_decoding_split(
                        query,
                        keys,
                        values,
                        params_ref,
                        split_start,
                        split_end,
                    )
                })
            })
            .collect();

        workers
            .into_iter()
            .map(|w| {
                w.join()
                    .unwrap_or_else(|_| FlashDecodingSplit::new(out.len()))
            })
            .collect::<Vec<_>>()
    });

    let reduced = tree_reduce_splits(splits);
    reduced.finalize(out);
}

/// Parameters for Multi-Head Latent Attention direct latent evaluation.
#[derive(Clone, Copy, Debug)]
pub struct MlaAttentionParams {
    pub latent_dim: usize,
    pub rope_dim: usize,
    pub len: usize,
    pub scale: f32,
    pub logit_cap: Option<f32>,
}

/// Multi-Head Latent Attention (MLA) direct latent cache attention kernel.
/// Evaluates attention scores directly between the query and cached latent KV vectors (c_KV),
/// along with decoupled RoPE keys (k_R), accumulating into an attention-weighted latent vector sum
/// without ever expanding historical per-head keys and values in memory.
pub fn attention_mla_latent(
    q_nope_projected: &[f32],
    q_rope: &[f32],
    latents: &[f32],
    rope_keys: &[f32],
    params: MlaAttentionParams,
    out_latent_sum: &mut [f32],
) {
    let MlaAttentionParams {
        latent_dim,
        rope_dim,
        len,
        scale,
        logit_cap,
    } = params;
    debug_assert_eq!(q_nope_projected.len(), latent_dim);
    debug_assert_eq!(q_rope.len(), rope_dim);
    debug_assert!(latents.len() >= len * latent_dim);
    debug_assert!(rope_keys.len() >= len * rope_dim);
    debug_assert_eq!(out_latent_sum.len(), latent_dim);

    out_latent_sum.fill(0.0);
    if len == 0 {
        return;
    }

    let mut max_score = f32::NEG_INFINITY;
    let mut denominator = 0.0f32;
    let tile_size = 64.min(len);
    let mut scores = [0.0f32; 64];

    for start in (0..len).step_by(tile_size) {
        let end = (start + tile_size).min(len);
        for (i, past) in (start..end).enumerate() {
            let lat_slice = &latents[past * latent_dim..(past + 1) * latent_dim];
            let rope_slice = &rope_keys[past * rope_dim..(past + 1) * rope_dim];
            let dot_nope = dot(q_nope_projected, lat_slice);
            let dot_rope = dot(q_rope, rope_slice);
            let mut score = (dot_nope + dot_rope) * scale;
            if let Some(cap) = logit_cap {
                score = cap * (score / cap).tanh();
            }
            scores[i] = score;
        }

        let tile_max = scores[..end - start]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if !tile_max.is_finite() {
            continue;
        }

        let next_max = max_score.max(tile_max);
        let old_scale = if max_score.is_finite() {
            (max_score - next_max).exp()
        } else {
            0.0
        };

        denominator *= old_scale;
        for v in out_latent_sum.iter_mut() {
            *v *= old_scale;
        }

        for (i, past) in (start..end).enumerate() {
            if !scores[i].is_finite() {
                continue;
            }
            let weight = (scores[i] - next_max).exp();
            denominator += weight;
            let lat_slice = &latents[past * latent_dim..(past + 1) * latent_dim];
            for (slot, &val) in out_latent_sum.iter_mut().zip(lat_slice) {
                *slot += weight * val;
            }
        }
        max_score = next_max;
    }

    if denominator > 0.0 {
        for v in out_latent_sum.iter_mut() {
            *v /= denominator;
        }
    }
}

/// `out[r] = dot(matrix row r, x)`, row-major with rows contiguous.
///
/// Rows are contiguous because that is how GGUF stores them, and walking them
/// in order is the difference between streaming memory and thrashing it.
pub fn matvec(matrix: &[f32], x: &[f32], out: &mut [f32]) {
    let cols = x.len();
    debug_assert_eq!(matrix.len(), cols * out.len());

    for (slot, row) in out.iter_mut().zip(matrix.chunks_exact(cols)) {
        *slot = dot(row, x);
    }
}

/// Batched matrix-vector multiplication: `out[b, r] = dot(matrix row r, x[b])`.
///
/// `matrix`: row-major with shape `[rows, cols]`.
/// `x`: row-major batch with shape `[batch, cols]`.
/// `out`: row-major batch with shape `[batch, rows]`.
///
/// Reuses each weight row across all `batch` tokens to transform memory-bound
/// GEMV into compute-bound GEMM during prompt prefill.
pub fn matmul(matrix: &[f32], x: &[f32], out: &mut [f32], batch: usize) {
    if batch == 0 {
        return;
    }
    let cols = x.len() / batch;
    let rows = out.len() / batch;
    debug_assert_eq!(matrix.len(), rows * cols);
    debug_assert_eq!(x.len(), batch * cols);
    debug_assert_eq!(out.len(), batch * rows);

    for (r, row) in matrix.chunks_exact(cols).enumerate() {
        for b in 0..batch {
            let x_tok = &x[b * cols..(b + 1) * cols];
            out[b * rows + r] = dot(row, x_tok);
        }
    }
}

/// Dynamically quantizes an activation vector to signed INT8 with symmetric scaling.
pub fn quantize_int8_dynamic(x: &[f32], q_out: &mut [i8]) -> f32 {
    debug_assert_eq!(x.len(), q_out.len());
    let mut max_abs = 0.0f32;
    for &val in x {
        let abs = val.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    if max_abs == 0.0 {
        q_out.fill(0);
        return 0.0;
    }
    let scale = max_abs / 127.0;
    let inv_scale = 127.0 / max_abs;
    for (src, dst) in x.iter().zip(q_out.iter_mut()) {
        let q = (*src * inv_scale).round().clamp(-128.0, 127.0) as i8;
        *dst = q;
    }
    scale
}

/// Evaluates integer dot product between two signed INT8 slices using 4-lane 32-bit accumulation.
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0i32; 4];
    for (chunk_a, chunk_b) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        acc[0] += (chunk_a[0] as i32) * (chunk_b[0] as i32);
        acc[1] += (chunk_a[1] as i32) * (chunk_b[1] as i32);
        acc[2] += (chunk_a[2] as i32) * (chunk_b[2] as i32);
        acc[3] += (chunk_a[3] as i32) * (chunk_b[3] as i32);
    }
    let full = a.len() - (a.len() % 4);
    let mut tail = 0i32;
    for (x, y) in a[full..].iter().zip(&b[full..]) {
        tail += (*x as i32) * (*y as i32);
    }
    acc[0] + acc[1] + acc[2] + acc[3] + tail
}

/// High-performance dynamic INT8 fused matrix-matrix multiplication (GEMM) for batched prefill.
///
/// Multiplies weight matrix `w_q` (INT8 with per-row scales `w_scales`) by activation batch `x` (F32),
/// writing the result to `out` with shape `[batch, rows]`.
pub fn gemm_int8_fused(
    w_q: &[i8],
    w_scales: &[f32],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    batch: usize,
) {
    if batch == 0 || rows == 0 || cols == 0 {
        return;
    }
    debug_assert_eq!(w_q.len(), rows * cols);
    debug_assert_eq!(w_scales.len(), rows);
    debug_assert_eq!(x.len(), batch * cols);
    debug_assert_eq!(out.len(), batch * rows);

    let mut q_x = vec![0i8; batch * cols];
    let mut x_scales = vec![0.0f32; batch];

    for b in 0..batch {
        let x_slice = &x[b * cols..(b + 1) * cols];
        let q_slice = &mut q_x[b * cols..(b + 1) * cols];
        x_scales[b] = quantize_int8_dynamic(x_slice, q_slice);
    }

    for (r, w_row) in w_q.chunks_exact(cols).enumerate() {
        let w_scale = w_scales[r];
        for b in 0..batch {
            let x_tok = &q_x[b * cols..(b + 1) * cols];
            let i32_dot = dot_i8(w_row, x_tok);
            let combined_scale = w_scale * x_scales[b];
            out[b * rows + r] = (i32_dot as f32) * combined_scale;
        }
    }
}

/// Plain dot product, accumulated in four lanes.
///
/// The lanes exist so the compiler can vectorize: a single accumulator forces a
/// serial dependency chain, and f32 addition is not associative so it will not
/// split one on its own.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut lanes = [0.0f32; 4];
    // `chunks_exact(4)` rather than indexed stepping: it hands the compiler a
    // provably-4-element slice, which is what lets it emit a single vector
    // multiply-add instead of four scalar ones.
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for lane in 0..4 {
            lanes[lane] += x[lane] * y[lane];
        }
    }
    let full = a.len() - a.len() % 4;
    let tail: f32 = a[full..].iter().zip(&b[full..]).map(|(x, y)| x * y).sum();
    lanes[0] + lanes[1] + lanes[2] + lanes[3] + tail
}

/// Standard sigmoid activation: `1 / (1 + exp(-x))`.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Adds bias vector in-place to vector `x`.
pub fn vector_add_bias(x: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(x.len(), bias.len());
    for (val, b) in x.iter_mut().zip(bias) {
        *val += b;
    }
}

/// Top-k expert selection and weight normalization with auxiliary-loss-free bias routing.
///
/// If `bias` is provided, expert selection ranks by `raw_scores[i] + bias[i]`.
/// The actual routing weights are computed from the original `raw_scores[i]` (without bias)
/// across the selected top-k experts and normalized (optionally scaled by `weights_scale`).
pub fn topk_moe_routing(
    raw_scores: &[f32],
    bias: Option<&[f32]>,
    top_k: usize,
    weights_scale: Option<f32>,
) -> Vec<(usize, f32)> {
    let n = raw_scores.len();
    let k = top_k.min(n);
    if k == 0 {
        return Vec::new();
    }

    let mut ranked: Vec<(usize, f32)> = if let Some(b) = bias {
        debug_assert_eq!(raw_scores.len(), b.len());
        raw_scores
            .iter()
            .zip(b.iter())
            .enumerate()
            .map(|(idx, (&s, &bias_val))| (idx, s + bias_val))
            .collect()
    } else {
        raw_scores.iter().copied().enumerate().collect()
    };

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let selected = &ranked[..k];
    let sum_raw: f32 = selected.iter().map(|&(idx, _)| raw_scores[idx]).sum();
    let norm_factor = if sum_raw > 0.0 { 1.0 / sum_raw } else { 0.0 };
    let scale = weights_scale.unwrap_or(1.0);

    selected
        .iter()
        .map(|&(idx, _)| {
            let weight = (raw_scores[idx] * norm_factor) * scale;
            (idx, weight)
        })
        .collect()
}

/// SiLU, also called swish: `x * sigmoid(x)`. The gate half of SwiGLU.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Rotary position embedding over adjacent pairs — the layout GGUF llama
/// models are converted into.
///
/// Rotating pair `i` by `position * base^(-2i/dim)` makes the dot product
/// between a query at one position and a key at another depend only on their
/// distance, which is the entire point.
pub fn rope(vector: &mut [f32], position: usize, base: f32) {
    let dim = vector.len();
    for pair in 0..dim / 2 {
        let frequency = 1.0 / base.powf(2.0 * pair as f32 / dim as f32);
        let angle = position as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        let (x, y) = (vector[pair * 2], vector[pair * 2 + 1]);
        vector[pair * 2] = x * cos - y * sin;
        vector[pair * 2 + 1] = x * sin + y * cos;
    }
}

/// RoPE scaling configuration for extended context windows.
#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    Linear {
        factor: f32,
    },
    Yarn {
        factor: f32,
        orig_ctx: usize,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    },
}

/// Dynamic and scaled RoPE for extended context lengths.
///
/// Applies frequency adjustments (Linear or YaRN) and restricts rotation to
/// `rotary_dim` (dimensions beyond `rotary_dim` pass through unrotated).
pub fn rope_scaled(
    vector: &mut [f32],
    position: usize,
    base: f32,
    rotary_dim: usize,
    scaling: Option<&RopeScaling>,
) {
    let rot_dim = rotary_dim.min(vector.len());
    let sub = &mut vector[..rot_dim];

    match scaling {
        None => {
            rope(sub, position, base);
        }
        Some(RopeScaling::Linear { factor }) => {
            let eff_factor = if *factor > 0.0 { *factor } else { 1.0 };
            let dim = sub.len();
            for pair in 0..dim / 2 {
                let frequency = (1.0 / eff_factor) / base.powf(2.0 * pair as f32 / dim as f32);
                let angle = position as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                let (x, y) = (sub[pair * 2], sub[pair * 2 + 1]);
                sub[pair * 2] = x * cos - y * sin;
                sub[pair * 2 + 1] = x * sin + y * cos;
            }
        }
        Some(RopeScaling::Yarn {
            factor,
            orig_ctx,
            ext_factor: _,
            attn_factor,
            beta_fast,
            beta_slow,
        }) => {
            let eff_factor = if *factor > 0.0 { *factor } else { 1.0 };
            let dim = sub.len();
            let orig_ctx = *orig_ctx as f32;
            let (b_fast, b_slow) = (*beta_fast, *beta_slow);

            let mscale = if *attn_factor > 0.0 {
                *attn_factor
            } else {
                0.1 * eff_factor.ln() + 1.0
            };

            for pair in 0..dim / 2 {
                let base_freq = 1.0 / base.powf(2.0 * pair as f32 / dim as f32);
                let wavelength = 2.0 * std::f32::consts::PI / base_freq;
                let r = orig_ctx / wavelength;
                let gamma = if b_fast == b_slow {
                    0.0
                } else {
                    ((r - b_slow) / (b_fast - b_slow)).clamp(0.0, 1.0)
                };

                let interp_freq = base_freq / eff_factor;
                let freq = (1.0 - gamma) * interp_freq + gamma * base_freq;

                let angle = position as f32 * freq;
                let (sin, cos) = angle.sin_cos();
                let (x, y) = (sub[pair * 2] * mscale, sub[pair * 2 + 1] * mscale);
                sub[pair * 2] = x * cos - y * sin;
                sub[pair * 2 + 1] = x * sin + y * cos;
            }
        }
    }
}

/// The GPT-NeoX pairing: element `i` rotates against `i + dim/2` rather than
/// its neighbour. Some architectures store weights expecting this.
pub fn rope_neox(vector: &mut [f32], position: usize, base: f32) {
    let dim = vector.len();
    let half = dim / 2;
    for pair in 0..half {
        let frequency = 1.0 / base.powf(2.0 * pair as f32 / dim as f32);
        let angle = position as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        let (x, y) = (vector[pair], vector[pair + half]);
        vector[pair] = x * cos - y * sin;
        vector[pair + half] = x * sin + y * cos;
    }
}

/// RoPE pair layout mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoPEType {
    /// Adjacent pair layout (LLaMA style): (x_0, x_1), (x_2, x_3), ...
    Normal,
    /// Half-split pair layout (GPT-NeoX / Qwen2-VL style): (x_i, x_{i + dim/2})
    Neox,
}

/// Multimodal Rotary Position Embedding (M-RoPE).
///
/// Partitions rotary channels across up to 4 coordinate axes (e.g., temporal/sequence,
/// height, width for vision-language models such as Qwen2-VL).
pub fn mrope(
    vector: &mut [f32],
    positions: [usize; 4],
    sections: [usize; 4],
    base: f32,
    rope_type: RoPEType,
) {
    let total_pairs = sections[0] + sections[1] + sections[2] + sections[3];
    let rot_dim = (total_pairs * 2).min(vector.len());
    let half = rot_dim / 2;
    let s0 = sections[0];
    let s1 = s0 + sections[1];
    let s2 = s1 + sections[2];

    for i in 0..half {
        let pos = if i < s0 {
            positions[0]
        } else if i < s1 {
            positions[1]
        } else if i < s2 {
            positions[2]
        } else {
            positions[3]
        };

        let frequency = 1.0 / base.powf(2.0 * i as f32 / rot_dim as f32);
        let angle = pos as f32 * frequency;
        let (sin, cos) = angle.sin_cos();

        match rope_type {
            RoPEType::Neox => {
                let (x, y) = (vector[i], vector[i + half]);
                vector[i] = x * cos - y * sin;
                vector[i + half] = x * sin + y * cos;
            }
            RoPEType::Normal => {
                let (x, y) = (vector[i * 2], vector[i * 2 + 1]);
                vector[i * 2] = x * cos - y * sin;
                vector[i * 2 + 1] = x * sin + y * cos;
            }
        }
    }
}

/// Precomputed Rotary Position Embedding (RoPE) frequency and trigonometric cache
/// with Su-YaRN interpolation and dynamic context extension support.
#[derive(Clone, Debug)]
pub struct RopeCache {
    dim: usize,
    rotary_dim: usize,
    base: f32,
    scaling: Option<RopeScaling>,
    inv_freq: Vec<f32>,
    cos_sin_table: Vec<(f32, f32)>,
    max_cached_positions: usize,
    mscale: f32,
}

impl RopeCache {
    /// Creates a new precomputed RoPE cache.
    pub fn new(
        dim: usize,
        rotary_dim: usize,
        base: f32,
        initial_positions: usize,
        scaling: Option<RopeScaling>,
    ) -> Self {
        let eff_rot_dim = rotary_dim.min(dim);
        let pairs = eff_rot_dim / 2;
        let (inv_freq, mscale) = Self::compute_inv_freq(eff_rot_dim, base, scaling.as_ref());
        let mut cache = Self {
            dim,
            rotary_dim: eff_rot_dim,
            base,
            scaling,
            inv_freq,
            cos_sin_table: Vec::with_capacity(initial_positions * pairs),
            max_cached_positions: 0,
            mscale,
        };
        cache.ensure_capacity(initial_positions);
        cache
    }

    /// Computes the inverse frequency array and attention multiplier according to RoPE scaling.
    pub fn compute_inv_freq(
        rotary_dim: usize,
        base: f32,
        scaling: Option<&RopeScaling>,
    ) -> (Vec<f32>, f32) {
        let pairs = rotary_dim / 2;
        let mut inv_freq = Vec::with_capacity(pairs);
        let mut mscale = 1.0f32;

        match scaling {
            None => {
                for pair in 0..pairs {
                    let freq = 1.0 / base.powf(2.0 * pair as f32 / rotary_dim as f32);
                    inv_freq.push(freq);
                }
            }
            Some(RopeScaling::Linear { factor }) => {
                let eff_factor = if *factor > 0.0 { *factor } else { 1.0 };
                for pair in 0..pairs {
                    let freq =
                        (1.0 / eff_factor) / base.powf(2.0 * pair as f32 / rotary_dim as f32);
                    inv_freq.push(freq);
                }
            }
            Some(RopeScaling::Yarn {
                factor,
                orig_ctx,
                ext_factor: _,
                attn_factor,
                beta_fast,
                beta_slow,
            }) => {
                let eff_factor = if *factor > 0.0 { *factor } else { 1.0 };
                let orig_ctx_f = *orig_ctx as f32;
                let (b_fast, b_slow) = (*beta_fast, *beta_slow);

                mscale = if *attn_factor > 0.0 {
                    *attn_factor
                } else {
                    0.1 * eff_factor.ln() + 1.0
                };

                for pair in 0..pairs {
                    let base_freq = 1.0 / base.powf(2.0 * pair as f32 / rotary_dim as f32);
                    let wavelength = 2.0 * std::f32::consts::PI / base_freq;
                    let r = orig_ctx_f / wavelength;
                    let gamma = if b_fast == b_slow {
                        0.0
                    } else {
                        ((r - b_slow) / (b_fast - b_slow)).clamp(0.0, 1.0)
                    };

                    let interp_freq = base_freq / eff_factor;
                    let freq = (1.0 - gamma) * interp_freq + gamma * base_freq;
                    inv_freq.push(freq);
                }
            }
        }

        (inv_freq, mscale)
    }

    /// Precomputes trigonometric values up to `positions`.
    pub fn ensure_capacity(&mut self, positions: usize) {
        if positions <= self.max_cached_positions {
            return;
        }
        let pairs = self.rotary_dim / 2;
        let start = self.max_cached_positions;
        let additional = (positions - start) * pairs;
        self.cos_sin_table.reserve(additional);

        for pos in start..positions {
            for &freq in &self.inv_freq {
                let angle = pos as f32 * freq;
                let (sin, cos) = angle.sin_cos();
                self.cos_sin_table.push((cos, sin));
            }
        }
        self.max_cached_positions = positions;
    }

    /// Gets precomputed cos and sin for a given position and pair index.
    #[inline]
    pub fn get_cos_sin(&self, position: usize, pair: usize) -> (f32, f32) {
        let pairs = self.rotary_dim / 2;
        if position < self.max_cached_positions {
            self.cos_sin_table[position * pairs + pair]
        } else {
            let freq = self.inv_freq[pair];
            let angle = position as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            (cos, sin)
        }
    }

    /// Returns the number of precomputed positions currently resident in cache.
    pub fn cached_positions(&self) -> usize {
        self.max_cached_positions
    }

    /// Returns the total head dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the rotary dimension.
    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    /// Returns the RoPE base frequency.
    pub fn base(&self) -> f32 {
        self.base
    }

    /// Returns the active RoPE scaling configuration, if any.
    pub fn scaling(&self) -> Option<&RopeScaling> {
        self.scaling.as_ref()
    }

    /// Applies cached RoPE rotation to a vector in-place.
    pub fn apply(&self, vector: &mut [f32], position: usize, rope_type: RoPEType) {
        let rot_dim = self.rotary_dim.min(vector.len());
        let half = rot_dim / 2;
        let mscale = self.mscale;

        match rope_type {
            RoPEType::Normal => {
                for pair in 0..half {
                    let (cos, sin) = self.get_cos_sin(position, pair);
                    let (x, y) = (vector[pair * 2] * mscale, vector[pair * 2 + 1] * mscale);
                    vector[pair * 2] = x * cos - y * sin;
                    vector[pair * 2 + 1] = x * sin + y * cos;
                }
            }
            RoPEType::Neox => {
                for pair in 0..half {
                    let (cos, sin) = self.get_cos_sin(position, pair);
                    let (x, y) = (vector[pair] * mscale, vector[pair + half] * mscale);
                    vector[pair] = x * cos - y * sin;
                    vector[pair + half] = x * sin + y * cos;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn rms_norm_scales_to_unit_rms() {
        let x = [3.0, 4.0, 0.0, 0.0];
        let weight = [1.0; 4];
        let mut out = [0.0; 4];
        rms_norm(&x, &weight, 0.0, &mut out);

        // rms = sqrt((9+16)/4) = 2.5
        assert!(close(out[0], 1.2), "{out:?}");
        assert!(close(out[1], 1.6), "{out:?}");
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!(close(rms, 1.0), "normalized rms should be 1, got {rms}");
    }

    #[test]
    fn rms_norm_applies_the_weight() {
        let x = [2.0, 2.0];
        let mut out = [0.0; 2];
        rms_norm(&x, &[3.0, 0.5], 0.0, &mut out);
        assert!(close(out[0], 3.0) && close(out[1], 0.5), "{out:?}");
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_inputs() {
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        assert!(close(x.iter().sum::<f32>(), 1.0));
        assert!(x[2] > x[1] && x[1] > x[0]);

        // Without the max subtraction this overflows to NaN.
        let mut x = [1000.0, 1001.0];
        softmax(&mut x);
        assert!(close(x.iter().sum::<f32>(), 1.0), "{x:?}");
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn softmax_of_equal_logits_is_uniform() {
        let mut x = [5.0; 8];
        softmax(&mut x);
        assert!(x.iter().all(|v| close(*v, 0.125)), "{x:?}");
    }

    #[test]
    fn tiled_online_attention_matches_materialized_reference() {
        let q = [0.2, -0.7, 1.1];
        let keys: Vec<f32> = (0..39).map(|i| (i as f32 * 0.17).sin()).collect();
        let values: Vec<f32> = (0..39).map(|i| (i as f32 * 0.31).cos()).collect();
        let mut scores: Vec<f32> = keys
            .chunks_exact(3)
            .map(|k| dot(&q, k) / 3.0f32.sqrt())
            .collect();
        softmax(&mut scores);
        let mut expected = [0.0; 3];
        for (w, v) in scores.iter().zip(values.chunks_exact(3)) {
            for (o, value) in expected.iter_mut().zip(v) {
                *o += w * value;
            }
        }
        let mut actual = [0.0; 3];
        attention_online(
            &q,
            &keys,
            &values,
            AttentionParams {
                row_width: 3,
                offset: 0,
                len: 13,
                scale: 1.0 / 3.0f32.sqrt(),
                logit_cap: None,
                window: None,
                position: 12,
                tile_size: 4,
                v_row_width: None,
                v_offset: None,
            },
            &mut actual,
        );
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn matvec_multiplies_rows() {
        // [[1, 2], [3, 4], [0, -1]] * [1, 10]
        let matrix = [1.0, 2.0, 3.0, 4.0, 0.0, -1.0];
        let mut out = [0.0; 3];
        matvec(&matrix, &[1.0, 10.0], &mut out);
        assert_eq!(out, [21.0, 43.0, -10.0]);
    }

    #[test]
    fn matmul_matches_batched_matvec() {
        // 3 rows, 2 cols matrix
        let matrix = [1.0, 2.0, 3.0, 4.0, 0.0, -1.0];
        // 2 tokens: [1.0, 10.0] and [-2.0, 5.0]
        let x_batch = [1.0, 10.0, -2.0, 5.0];
        let mut out = [0.0; 6];
        matmul(&matrix, &x_batch, &mut out, 2);

        let mut expected0 = [0.0; 3];
        let mut expected1 = [0.0; 3];
        matvec(&matrix, &[1.0, 10.0], &mut expected0);
        matvec(&matrix, &[-2.0, 5.0], &mut expected1);

        assert_eq!(&out[0..3], &expected0);
        assert_eq!(&out[3..6], &expected1);
    }

    #[test]
    fn dot_handles_lengths_that_are_not_multiples_of_four() {
        for len in 1..20 {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (i * 2) as f32).collect();
            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!(close(dot(&a, &b), expected), "length {len}");
        }
    }

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        let mut v = [1.0, 2.0, 3.0, 4.0];
        rope(&mut v, 0, 10000.0);
        assert_eq!(v, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_preserves_length_and_relative_angle() {
        let before = [0.6, 0.8, -1.0, 2.0];
        let mut v = before;
        rope(&mut v, 7, 10000.0);
        // Rotation is norm-preserving, per pair.
        for pair in 0..2 {
            let original = before[pair * 2].hypot(before[pair * 2 + 1]);
            let rotated = v[pair * 2].hypot(v[pair * 2 + 1]);
            assert!(
                close(original, rotated),
                "pair {pair}: {original} {rotated}"
            );
        }
        assert!(v != before, "position 7 should actually rotate");
    }

    #[test]
    fn neox_rotates_across_the_halves() {
        let mut v = [1.0, 0.0, 0.0, 0.0];
        rope_neox(&mut v, 1, 10000.0);
        // Element 0 pairs with element 2, so both move and the others do not.
        assert!(v[0] != 1.0 && v[2] != 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[3], 0.0);
    }

    #[test]
    fn silu_matches_its_definition() {
        assert!(close(silu(0.0), 0.0));
        assert!(close(silu(1.0), 1.0 / (1.0 + (-1.0f32).exp())));
        // Saturates toward the identity for large positive input.
        assert!(close(silu(20.0), 20.0));
        assert!(silu(-20.0).abs() < 1e-6);
    }

    #[test]
    fn rope_scaled_none_matches_unscaled() {
        let mut v1 = [0.6, 0.8, -1.0, 2.0];
        let mut v2 = v1;
        rope(&mut v1, 5, 10000.0);
        rope_scaled(&mut v2, 5, 10000.0, 4, None);
        assert_eq!(v1, v2);
    }

    #[test]
    fn rope_scaled_linear_at_double_pos_matches_unscaled() {
        let mut v1 = [0.6, 0.8, -1.0, 2.0];
        let mut v2 = v1;
        rope(&mut v1, 5, 10000.0);
        rope_scaled(
            &mut v2,
            10,
            10000.0,
            4,
            Some(&RopeScaling::Linear { factor: 2.0 }),
        );
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!(close(*a, *b), "{v1:?} != {v2:?}");
        }
    }

    #[test]
    fn rope_scaled_partial_dim_preserves_tail() {
        let mut v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        rope_scaled(&mut v, 3, 10000.0, 4, None);
        assert_eq!(v[4], 5.0);
        assert_eq!(v[5], 6.0);
        assert!(v[0] != 1.0 || v[1] != 2.0);
    }

    #[test]
    fn mrope_uniform_positions_matches_rope_neox() {
        let mut v1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut v2 = v1;
        rope_neox(&mut v1, 7, 10000.0);
        mrope(&mut v2, [7, 7, 7, 7], [1, 1, 2, 0], 10000.0, RoPEType::Neox);
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!(close(*a, *b), "{v1:?} != {v2:?}");
        }
    }

    #[test]
    fn mrope_multimodal_sections_rotate_independent_coordinates() {
        let mut v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // 4 pairs: section 0 (1 pair, T=0), section 1 (1 pair, H=2), section 2 (2 pairs, W=5)
        mrope(&mut v, [0, 2, 5, 0], [1, 1, 2, 0], 10000.0, RoPEType::Neox);
        // At T=0, section 0 (pair 0: elements 0 and 4) should stay unchanged because angle = 0!
        assert!(close(v[0], 1.0));
        assert!(close(v[4], 5.0));
        // Sections 1 and 2 (H=2, W=5) should be rotated
        assert!(!close(v[1], 2.0) || !close(v[5], 6.0));
        assert!(!close(v[2], 3.0) || !close(v[6], 7.0));
    }

    #[test]
    fn sigmoid_and_bias_ops() {
        assert!(close(sigmoid(0.0), 0.5));
        assert!(sigmoid(20.0) > 0.9999);
        assert!(sigmoid(-20.0) < 0.0001);

        let mut v = [1.0, 2.0, 3.0];
        vector_add_bias(&mut v, &[0.5, -1.0, 2.0]);
        assert_eq!(v, [1.5, 1.0, 5.0]);
    }

    #[test]
    fn topk_moe_routing_with_bias() {
        // Raw scores: expert 0 and 1 are highest without bias
        let raw = [0.8, 0.7, 0.4, 0.3];
        // Without bias, top-2 should be 0 and 1
        let unmasked = topk_moe_routing(&raw, None, 2, None);
        assert_eq!(unmasked.len(), 2);
        assert_eq!(unmasked[0].0, 0);
        assert_eq!(unmasked[1].0, 1);
        let sum_unmasked: f32 = unmasked.iter().map(|(_, w)| *w).sum();
        assert!(close(sum_unmasked, 1.0));

        // With bias, boost experts 2 and 3 so they win top-2
        let bias = [0.0, 0.0, 1.0, 1.0];
        // Biased scores: 0.8, 0.7, 1.4, 1.3 -> top 2 are expert 2 (1.4) and expert 3 (1.3)
        let biased = topk_moe_routing(&raw, Some(&bias), 2, Some(2.5));
        assert_eq!(biased.len(), 2);
        assert_eq!(biased[0].0, 2);
        assert_eq!(biased[1].0, 3);
        // Weights are calculated from raw scores without bias!
        // raw[2] = 0.4, raw[3] = 0.3, sum = 0.7.
        // weight[2] = (0.4 / 0.7) * 2.5
        // weight[3] = (0.3 / 0.7) * 2.5
        let total_weight: f32 = biased.iter().map(|(_, w)| *w).sum();
        assert!(close(total_weight, 2.5));
        assert!(close(biased[0].1, (0.4 / 0.7) * 2.5));
        assert!(close(biased[1].1, (0.3 / 0.7) * 2.5));
    }

    #[test]
    fn attention_online_dynamic_matches_reference_across_tile_sizes() {
        let q = [0.2, -0.7, 1.1, 0.4];
        let len = 40;
        let keys: Vec<f32> = (0..len * 4).map(|i| (i as f32 * 0.17).sin()).collect();
        let values: Vec<f32> = (0..len * 4).map(|i| (i as f32 * 0.31).cos()).collect();

        let mut expected = [0.0f32; 4];
        attention_online(
            &q,
            &keys,
            &values,
            AttentionParams {
                row_width: 4,
                offset: 0,
                len,
                scale: 0.5,
                logit_cap: Some(30.0),
                window: None,
                position: len - 1,
                tile_size: 64,
                v_row_width: None,
                v_offset: None,
            },
            &mut expected,
        );

        for tile in [8, 16, 32, 64, 128] {
            let mut actual = [0.0f32; 4];
            attention_online_dynamic(
                &q,
                &keys,
                &values,
                AttentionParams {
                    row_width: 4,
                    offset: 0,
                    len,
                    scale: 0.5,
                    logit_cap: Some(30.0),
                    window: None,
                    position: len - 1,
                    tile_size: tile,
                    v_row_width: None,
                    v_offset: None,
                },
                &mut actual,
            );

            for (a, e) in actual.iter().zip(&expected) {
                assert!(
                    (a - e).abs() < 1e-5,
                    "tile {tile}: actual {actual:?} != expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn rope_cache_normal_matches_rope_scaled() {
        let base = 10000.0;
        let dim = 16;
        let rotary_dim = 16;

        let cache = RopeCache::new(dim, rotary_dim, base, 32, None);

        for pos in 0..20 {
            let mut v1 = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.5, 0.4, 0.3, 0.2, 0.1, -1.0, -2.0, -3.0,
            ];
            let mut v2 = v1;

            rope_scaled(&mut v1, pos, base, rotary_dim, None);
            cache.apply(&mut v2, pos, RoPEType::Normal);

            for (a, b) in v1.iter().zip(&v2) {
                assert!((a - b).abs() < 1e-5, "pos {pos}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn rope_cache_neox_matches_rope_neox() {
        let base = 10000.0;
        let dim = 16;
        let rotary_dim = 16;

        let cache = RopeCache::new(dim, rotary_dim, base, 16, None);

        for pos in 0..10 {
            let mut v1 = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.5, 0.4, 0.3, 0.2, 0.1, -1.0, -2.0, -3.0,
            ];
            let mut v2 = v1;

            rope_neox(&mut v1, pos, base);
            cache.apply(&mut v2, pos, RoPEType::Neox);

            for (a, b) in v1.iter().zip(&v2) {
                assert!((a - b).abs() < 1e-5, "pos {pos}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn rope_cache_yarn_matches_rope_scaled_yarn() {
        let base = 10000.0;
        let dim = 16;
        let rotary_dim = 16;
        let scaling = RopeScaling::Yarn {
            factor: 2.0,
            orig_ctx: 1024,
            ext_factor: 1.0,
            attn_factor: 1.1,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };

        let cache = RopeCache::new(dim, rotary_dim, base, 16, Some(scaling.clone()));

        for pos in 0..10 {
            let mut v1 = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.5, 0.4, 0.3, 0.2, 0.1, -1.0, -2.0, -3.0,
            ];
            let mut v2 = v1;

            rope_scaled(&mut v1, pos, base, rotary_dim, Some(&scaling));
            cache.apply(&mut v2, pos, RoPEType::Normal);

            for (a, b) in v1.iter().zip(&v2) {
                assert!((a - b).abs() < 1e-5, "yarn pos {pos}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn rope_cache_dynamic_capacity_expansion() {
        let base = 10000.0;
        let dim = 8;
        let rotary_dim = 8;
        let mut cache = RopeCache::new(dim, rotary_dim, base, 4, None);
        assert_eq!(cache.cached_positions(), 4);

        // Position within cache
        let mut v1 = [1.0; 8];
        let mut v2 = [1.0; 8];
        rope(&mut v1, 2, base);
        cache.apply(&mut v2, 2, RoPEType::Normal);
        assert_eq!(v1, v2);

        // Position beyond current cache: auto fallback and dynamic capacity expansion
        let mut v3 = [1.0; 8];
        let mut v4 = [1.0; 8];
        rope(&mut v3, 10, base);
        cache.apply(&mut v4, 10, RoPEType::Normal);
        assert_eq!(v3, v4);

        cache.ensure_capacity(16);
        assert_eq!(cache.cached_positions(), 16);
        let mut v5 = [1.0; 8];
        cache.apply(&mut v5, 10, RoPEType::Normal);
        assert_eq!(v3, v5);
    }

    #[test]
    fn gemm_int8_fused_agrees_with_float_reference() {
        let rows = 4;
        let cols = 8;
        let batch = 3;

        // Weights: signed int8 values and per-row scales
        let mut w_q = vec![0i8; rows * cols];
        for (i, val) in w_q.iter_mut().enumerate() {
            *val = ((i as i32 * 17) % 255 - 127) as i8;
        }
        let w_scales = vec![0.02f32, 0.05f32, 0.01f32, 0.03f32];

        // Dequantize weights to float for reference matmul
        let mut w_f32 = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                w_f32[r * cols + c] = (w_q[r * cols + c] as f32) * w_scales[r];
            }
        }

        // Activations: float values
        let mut x = vec![0.0f32; batch * cols];
        for b in 0..batch {
            for c in 0..cols {
                x[b * cols + c] = ((b as f32 + 1.0) * 0.3) + ((c as f32) * 0.1) - 0.4;
            }
        }

        let mut expected = vec![0.0f32; batch * rows];
        matmul(&w_f32, &x, &mut expected, batch);

        let mut actual = vec![0.0f32; batch * rows];
        gemm_int8_fused(&w_q, &w_scales, &x, &mut actual, rows, cols, batch);

        // Dynamic int8 quantization error is bounded by quantization step size
        for b in 0..batch {
            for r in 0..rows {
                let diff = (expected[b * rows + r] - actual[b * rows + r]).abs();
                let mag = expected[b * rows + r].abs().max(1.0);
                let rel_err = diff / mag;
                assert!(
                    rel_err < 0.08,
                    "batch {b}, row {r}: expected {}, got {}, rel_err {}",
                    expected[b * rows + r],
                    actual[b * rows + r],
                    rel_err
                );
            }
        }
    }
}
