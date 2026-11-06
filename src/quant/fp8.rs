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
