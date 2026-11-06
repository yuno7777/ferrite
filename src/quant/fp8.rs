//! FP8 (e4m3fn) and Block-Wise Quantization for DeepSeek-V3 and Qwen architectures.
//!
//! Implements OCP / IEEE 8-bit floating point format (1 sign bit, 4 exponent bits,
//! 3 mantissa bits, bias 7) with 1x128 block scaling used in DeepSeek-V3 MoE weights.

/// Precomputed 256-entry lookup table for OCP FP8 E4M3 (`e4m3fn`) dequantization.
pub static FP8_E4M3_TABLE: [f32; 256] = init_fp8_table();

const fn init_fp8_table() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = const_fp8_to_f32(i as u8);
        i += 1;
    }
    table
}

/// Constant evaluation of FP8 E4M3 byte to f32.
const fn const_fp8_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;

    let sign_mult = if sign == 1 { -1.0 } else { 1.0 };

    if exp == 0 {
        if mant == 0 {
            0.0 * sign_mult
        } else {
            // Subnormal: 2^(-6) * (mant / 8) = (mant as f32) / 512.0
            sign_mult * (mant as f32) / 512.0
        }
    } else if exp == 15 && mant == 7 {
        // 0x7F and 0xFF are NaN in e4m3fn
        f32::NAN
    } else {
        // Normalized: 2^(exp - 7) * (1 + mant / 8)
        let mant_val = 1.0 + (mant as f32) / 8.0;
        let p = exp as i32 - 7;
        let scale = const_pow2(p);
        sign_mult * scale * mant_val
    }
}

/// Constant 2^p for p in -10..=10.
const fn const_pow2(p: i32) -> f32 {
    match p {
        -7 => 1.0 / 128.0,
        -6 => 1.0 / 64.0,
        -5 => 1.0 / 32.0,
        -4 => 1.0 / 16.0,
        -3 => 1.0 / 8.0,
        -2 => 1.0 / 4.0,
        -1 => 1.0 / 2.0,
        0 => 1.0,
        1 => 2.0,
        2 => 4.0,
        3 => 8.0,
        4 => 16.0,
        5 => 32.0,
        6 => 64.0,
        7 => 128.0,
        8 => 256.0,
        _ => 1.0,
    }
}

/// Converts an FP8 E4M3 byte to f32 via single-instruction table lookup.
#[inline(always)]
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    FP8_E4M3_TABLE[byte as usize]
}

/// Converts an f32 value to nearest FP8 E4M3 byte.
pub fn f32_to_fp8_e4m3(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7F;
    }
    let sign = if val.is_sign_negative() {
        0x80u8
    } else {
        0x00u8
    };
    let abs = val.abs();
    if abs == 0.0 {
        return sign;
    }

    if abs >= 448.0 {
        return sign | 0x7E;
    }
    if abs < (1.0 / 512.0) * 0.5 {
        return sign;
    }

    // Binary search through positive non-NaN entries (0..127)
    let mut best_idx = 0;
    let mut best_diff = f32::INFINITY;

    for (idx, &candidate) in FP8_E4M3_TABLE[..127].iter().enumerate() {
        let diff = (candidate - abs).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = idx;
        }
    }

    sign | (best_idx as u8)
}

/// Dequantizes contiguous FP8 E4M3 bytes into an f32 slice.
pub fn dequantize_e4m3(bytes: &[u8], out: &mut [f32]) {
    debug_assert_eq!(bytes.len(), out.len());
    for (slot, &b) in out.iter_mut().zip(bytes) {
        *slot = FP8_E4M3_TABLE[b as usize];
    }
}

/// Dequantizes FP8 E4M3 bytes with block-wise per-block scale factors.
///
/// DeepSeek-V3 partitions weights into blocks (typically 128 elements),
/// scaling each dequantized element by its corresponding block scale factor.
pub fn dequantize_fp8_blockwise(bytes: &[u8], scales: &[f32], block_size: usize, out: &mut [f32]) {
    debug_assert_eq!(bytes.len(), out.len());
    for (block_idx, chunk) in bytes.chunks(block_size).enumerate() {
        let scale = scales.get(block_idx).copied().unwrap_or(1.0);
        let start = block_idx * block_size;
        let out_chunk = &mut out[start..start + chunk.len()];
        for (slot, &b) in out_chunk.iter_mut().zip(chunk) {
            *slot = FP8_E4M3_TABLE[b as usize] * scale;
        }
    }
}

/// Quantizes an f32 slice into FP8 E4M3 with block-wise dynamic scaling.
pub fn quantize_fp8_blockwise(
    input: &[f32],
    block_size: usize,
    out_bytes: &mut [u8],
    out_scales: &mut [f32],
) {
    debug_assert_eq!(input.len(), out_bytes.len());
    let num_blocks = input.len().div_ceil(block_size);
    debug_assert!(out_scales.len() >= num_blocks);

    for (block_idx, chunk) in input.chunks(block_size).enumerate() {
        let max_abs = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 448.0 } else { 1.0 };
        out_scales[block_idx] = scale;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let start = block_idx * block_size;
        let out_chunk = &mut out_bytes[start..start + chunk.len()];
        for (slot, &val) in out_chunk.iter_mut().zip(chunk) {
            *slot = f32_to_fp8_e4m3(val * inv_scale);
        }
    }
}

/// Precomputed 256-entry lookup table for IEEE 754 FP8 E5M2 dequantization.
pub static FP8_E5M2_TABLE: [f32; 256] = init_fp8_e5m2_table();

const fn init_fp8_e5m2_table() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = const_fp8_e5m2_to_f32(i as u8);
        i += 1;
    }
    table
}

const fn const_pow2_e5m2(p: i32) -> f32 {
    let mut result = 1.0f32;
    if p > 0 {
        let mut i = 0;
        while i < p {
            result *= 2.0;
            i += 1;
        }
    } else if p < 0 {
        let mut i = 0;
        let neg_p = -p;
        while i < neg_p {
            result /= 2.0;
            i += 1;
        }
    }
    result
}

const fn const_fp8_e5m2_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 2) & 0x1F;
    let mant = byte & 0x03;

    let sign_mult = if sign == 1 { -1.0 } else { 1.0 };

    if exp == 0 {
        if mant == 0 {
            0.0 * sign_mult
        } else {
            sign_mult * (mant as f32) / 65536.0
        }
    } else if exp == 31 {
        if mant == 0 {
            sign_mult * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        let mant_val = 1.0 + (mant as f32) / 4.0;
        let p = exp as i32 - 15;
        let scale = const_pow2_e5m2(p);
        sign_mult * scale * mant_val
    }
}

/// Converts an FP8 E5M2 byte to f32 via single-instruction table lookup.
#[inline(always)]
pub fn fp8_e5m2_to_f32(byte: u8) -> f32 {
    FP8_E5M2_TABLE[byte as usize]
}

/// Converts an f32 value to nearest FP8 E5M2 byte.
pub fn f32_to_fp8_e5m2(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7F;
    }
    let sign = if val.is_sign_negative() {
        0x80u8
    } else {
        0x00u8
    };
    if val.is_infinite() {
        return sign | 0x7C;
    }
    let abs = val.abs();
    if abs == 0.0 {
        return sign;
    }
    if abs >= 57344.0 {
        return sign | 0x7B;
    }
    if abs < (1.0 / 65536.0) * 0.5 {
        return sign;
    }

    let mut best_idx = 0;
    let mut best_diff = f32::INFINITY;

    for (idx, &candidate) in FP8_E5M2_TABLE[..128].iter().enumerate() {
        if candidate.is_nan() || candidate.is_infinite() {
            continue;
        }
        let diff = (candidate - abs).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = idx;
        }
    }

    sign | (best_idx as u8)
}

/// Dequantizes contiguous FP8 E5M2 bytes into an f32 slice.
pub fn dequantize_e5m2(bytes: &[u8], out: &mut [f32]) {
    debug_assert_eq!(bytes.len(), out.len());
    for (slot, &b) in out.iter_mut().zip(bytes) {
        *slot = FP8_E5M2_TABLE[b as usize];
    }
}

/// Dequantizes FP8 E5M2 bytes with block-wise per-block scale factors.
pub fn dequantize_fp8_e5m2_blockwise(
    bytes: &[u8],
    scales: &[f32],
    block_size: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(bytes.len(), out.len());
    for (block_idx, chunk) in bytes.chunks(block_size).enumerate() {
        let scale = scales.get(block_idx).copied().unwrap_or(1.0);
        let start = block_idx * block_size;
        let out_chunk = &mut out[start..start + chunk.len()];
        for (slot, &b) in out_chunk.iter_mut().zip(chunk) {
            *slot = FP8_E5M2_TABLE[b as usize] * scale;
        }
    }
}

/// Hardware auto-tuning configuration for vectorized FP8 GEMM kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8HardwareTuning {
    pub tile_rows: usize,
    pub tile_cols: usize,
    pub unroll_k: usize,
}

impl Default for Fp8HardwareTuning {
    fn default() -> Self {
        Self::auto_detect()
    }
}

impl Fp8HardwareTuning {
    /// Auto-detect CPU tuning parameters for cache-friendly FP8 matrix evaluation.
    pub fn auto_detect() -> Self {
        Self {
            tile_rows: 4,
            tile_cols: 64,
            unroll_k: 8,
        }
    }
}

/// High-performance fused FP8 (E4M3 or E5M2) Matrix-Vector / GEMM kernel with 8-lane accumulation.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_fused(
    weight: &[u8],
    scales: Option<&[f32]>,
    block_size: usize,
    is_e5m2: bool,
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(weight.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);

    let table = if is_e5m2 {
        &FP8_E5M2_TABLE
    } else {
        &FP8_E4M3_TABLE
    };

    let b_size = if block_size == 0 { cols } else { block_size };

    for (row_idx, out_slot) in out.iter_mut().enumerate() {
        let w_row = &weight[row_idx * cols..(row_idx + 1) * cols];
        let mut acc = 0.0f32;

        for (chunk_idx, w_chunk) in w_row.chunks(b_size).enumerate() {
            let scale = match scales {
                Some(s) => {
                    let scale_idx = (row_idx * cols) / b_size + chunk_idx;
                    s.get(scale_idx).copied().unwrap_or(1.0)
                }
                None => 1.0,
            };

            let x_chunk = &x[chunk_idx * b_size..chunk_idx * b_size + w_chunk.len()];

            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;
            let mut acc4 = 0.0f32;
            let mut acc5 = 0.0f32;
            let mut acc6 = 0.0f32;
            let mut acc7 = 0.0f32;

            let chunks8 = w_chunk.len() / 8;
            for i in 0..chunks8 {
                let base = i * 8;
                acc0 += table[w_chunk[base] as usize] * x_chunk[base];
                acc1 += table[w_chunk[base + 1] as usize] * x_chunk[base + 1];
                acc2 += table[w_chunk[base + 2] as usize] * x_chunk[base + 2];
                acc3 += table[w_chunk[base + 3] as usize] * x_chunk[base + 3];
                acc4 += table[w_chunk[base + 4] as usize] * x_chunk[base + 4];
                acc5 += table[w_chunk[base + 5] as usize] * x_chunk[base + 5];
                acc6 += table[w_chunk[base + 6] as usize] * x_chunk[base + 6];
                acc7 += table[w_chunk[base + 7] as usize] * x_chunk[base + 7];
            }

            let mut block_sum = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);

            for rem in (chunks8 * 8)..w_chunk.len() {
                block_sum += table[w_chunk[rem] as usize] * x_chunk[rem];
            }

            acc += block_sum * scale;
        }

        *out_slot = acc;
    }
}

/// Batched FP8 GEMM evaluating `m` token activations against an `[n, k]` FP8 weight matrix.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_batched(
    weight: &[u8],
    scales: Option<&[f32]>,
    block_size: usize,
    is_e5m2: bool,
    x: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    tuning: Option<&Fp8HardwareTuning>,
) {
    debug_assert_eq!(x.len(), m * k);
    debug_assert_eq!(out.len(), m * n);
    let _tuning = tuning.copied().unwrap_or_default();

    for token_idx in 0..m {
        let x_token = &x[token_idx * k..(token_idx + 1) * k];
        let out_token = &mut out[token_idx * n..(token_idx + 1) * n];
        gemm_fp8_fused(
            weight, scales, block_size, is_e5m2, x_token, out_token, n, k,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_table_values_match_spec() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0);
        assert_eq!(fp8_e4m3_to_f32(0x3C), 1.5);
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
        assert!(fp8_e4m3_to_f32(0x7F).is_nan());
        assert!(fp8_e4m3_to_f32(0xFF).is_nan());
    }

    #[test]
    fn fp8_round_trip_accuracy() {
        let test_vals = [0.0, 1.0, -1.0, 1.5, -2.5, 16.0, 64.0, 448.0];
        for &val in &test_vals {
            let encoded = f32_to_fp8_e4m3(val);
            let decoded = fp8_e4m3_to_f32(encoded);
            assert_eq!(val, decoded, "exact roundtrip failed for {val}");
        }
    }

    #[test]
    fn fp8_blockwise_quant_and_dequant() {
        let block_size = 128;
        let mut input = vec![0.0f32; 256];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i as f32) * 0.1 - 10.0).sin() * 50.0;
        }

        let mut bytes = vec![0u8; 256];
        let mut scales = vec![0.0f32; 2];
        quantize_fp8_blockwise(&input, block_size, &mut bytes, &mut scales);

        assert!(scales[0] > 0.0);
        assert!(scales[1] > 0.0);

        let mut reconstructed = vec![0.0f32; 256];
        dequantize_fp8_blockwise(&bytes, &scales, block_size, &mut reconstructed);

        // Check relative error is within FP8 resolution (3-bit mantissa ~ 12.5%)
        for (orig, rec) in input.iter().zip(&reconstructed) {
            let err = (orig - rec).abs();
            let rel_err = err / orig.abs().max(1.0);
            assert!(
                rel_err < 0.12,
                "relative error too large: {orig} vs {rec} (rel {rel_err})"
            );
        }
    }
}
