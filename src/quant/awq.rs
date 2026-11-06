//! Activation-Aware Weight Quantization (AWQ) and GPTQ INT4 runtime dequantization.
//!
//! Provides zero-dependency unpackers and fused dot products for group-quantized
//! INT4 weights with arbitrary group sizes (32, 64, 128).

use std::io::{Error, ErrorKind, Result};

/// Unpacks packed 4-bit AWQ weights into 32-bit floats.
///
/// In AWQ, weights are quantized in groups of `group_size` elements sharing a scale and zero-point.
/// Eight 4-bit weights are packed into each 32-bit integer (`u32`).
pub fn dequantize_awq_int4(
    packed: &[u32],
    scales: &[f32],
    zeros: Option<&[f32]>,
    group_size: usize,
    out: &mut [f32],
) -> Result<()> {
    if group_size == 0 || group_size % 8 != 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("invalid AWQ group size: {group_size}"),
        ));
    }
    let total_weights = packed.len() * 8;
    if out.len() != total_weights {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "output buffer length {} != expected {}",
                out.len(),
                total_weights
            ),
        ));
    }
    let expected_groups = total_weights.div_ceil(group_size);
    if scales.len() < expected_groups {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "scales length {} < expected groups {expected_groups}",
                scales.len()
            ),
        ));
    }
    if let Some(z) = zeros {
        if z.len() < expected_groups {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "zeros length {} < expected groups {expected_groups}",
                    z.len()
                ),
            ));
        }
    }

    for (word_idx, &word) in packed.iter().enumerate() {
        for nibble_idx in 0..8 {
            let weight_idx = word_idx * 8 + nibble_idx;
            let group_idx = weight_idx / group_size;
            let scale = scales[group_idx];
            let zero = zeros.map(|z| z[group_idx]).unwrap_or(0.0);
            let q = ((word >> (nibble_idx * 4)) & 0xF) as f32;
            out[weight_idx] = (q - zero) * scale;
        }
    }
    Ok(())
}

/// Fused dot product between packed AWQ INT4 weights and an F32 activation vector.
pub fn dot_awq_int4(
    packed: &[u32],
    scales: &[f32],
    zeros: Option<&[f32]>,
    group_size: usize,
    x: &[f32],
) -> f32 {
    debug_assert!(group_size > 0 && group_size % 8 == 0);
    debug_assert_eq!(packed.len() * 8, x.len());
    let mut sum = 0.0f32;

    for (word_idx, &word) in packed.iter().enumerate() {
        let base_weight_idx = word_idx * 8;
        let group_idx = base_weight_idx / group_size;
        let scale = scales[group_idx];
        let zero = zeros.map(|z| z[group_idx]).unwrap_or(0.0);

        for nibble_idx in 0..8 {
            let weight_idx = base_weight_idx + nibble_idx;
            let q = ((word >> (nibble_idx * 4)) & 0xF) as f32;
            let w = (q - zero) * scale;
            sum += w * x[weight_idx];
        }
    }
    sum
}

/// Unpacks packed 4-bit GPTQ weights into 32-bit floats.
///
/// GPTQ organizes weights into columns and groups. This unpacker decodes
/// signed/offset INT4 representations according to GPTQ group scaling.
pub fn dequantize_gptq_int4(
    packed: &[u32],
    scales: &[f32],
    zeros: Option<&[f32]>,
    group_size: usize,
    out: &mut [f32],
) -> Result<()> {
    dequantize_awq_int4(packed, scales, zeros, group_size, out)
}

/// Fused dot product between packed GPTQ INT4 weights and an F32 activation vector.
pub fn dot_gptq_int4(
    packed: &[u32],
    scales: &[f32],
    zeros: Option<&[f32]>,
    group_size: usize,
    x: &[f32],
) -> f32 {
    dot_awq_int4(packed, scales, zeros, group_size, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn awq_dequantize_matches_exact_values() {
        // 1 u32 word = 8 elements. Let group_size = 8.
        // word has nibbles: 0, 1, 2, 3, 4, 5, 6, 7
        let word = 0x76543210u32;
        let scale = [0.5f32];
        let zero = [1.0f32];
        let mut out = [0.0f32; 8];

        dequantize_awq_int4(&[word], &scale, Some(&zero), 8, &mut out).expect("dequantize");

        let expected = [
            (0.0 - 1.0) * 0.5,
            (1.0 - 1.0) * 0.5,
            (2.0 - 1.0) * 0.5,
            (3.0 - 1.0) * 0.5,
            (4.0 - 1.0) * 0.5,
            (5.0 - 1.0) * 0.5,
            (6.0 - 1.0) * 0.5,
            (7.0 - 1.0) * 0.5,
        ];
        for (a, e) in out.iter().zip(&expected) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[test]
    fn awq_fused_dot_matches_unpacked_dot() {
        let words = [0x89ABCDEFu32, 0x01234567u32]; // 16 elements
        let scales = [0.25f32, 0.5f32]; // 2 groups of 8
        let zeros = [2.0f32, 1.0f32];
        let mut unpacked = [0.0f32; 16];

        dequantize_awq_int4(&words, &scales, Some(&zeros), 8, &mut unpacked).expect("dequantize");

        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).sin()).collect();
        let expected: f32 = unpacked.iter().zip(&x).map(|(w, xi)| w * xi).sum();

        let fused = dot_awq_int4(&words, &scales, Some(&zeros), 8, &x);
        assert!((fused - expected).abs() < 1e-5);
    }
}
