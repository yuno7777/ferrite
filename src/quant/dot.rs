//! Fused dequantize-and-dot.
//!
//! The unfused path expands a weight row into an f32 scratch buffer and then
//! dots it. That writes the whole row to memory and reads it straight back —
//! for a 4096-wide row, 16 KB out and 16 KB in, per row, per token. Fusing the
//! expansion into the inner loop deletes both.
//!
//! Every kernel here accumulates into the same four lanes, in the same index
//! order, as [`crate::ops::dot`]. That is not decoration: it makes the fused
//! result **bit-identical** to dequantizing first, so the fast path can be
//! diffed against the reference rather than merely spot-checked. Block sizes
//! are all multiples of four, so `index & 3` picks the same lane the unfused
//! loop would have.

use super::{half, k};
use crate::gguf::GgmlType;

/// Sum the lanes exactly as `ops::dot` does — left to right, no `iter().sum()`,
/// which would fold in a leading zero and disturb signed zeros.
#[inline]
fn total(lanes: [f32; 4]) -> f32 {
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

/// Whether a fused kernel exists, so callers can skip allocating the scratch
/// row they would only need for the fallback.
pub fn supports(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::F32
            | GgmlType::Q8_0
            | GgmlType::Q4_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
    )
}

/// Dot one weight row, still quantized, against an activation vector.
///
/// `None` for types without a fused kernel; the caller falls back to
/// dequantize-then-dot, which is slower but never wrong.
pub fn fused(ty: GgmlType, row: &[u8], x: &[f32]) -> Option<f32> {
    Some(match ty {
        GgmlType::F32 => f32_row(row, x),
        GgmlType::Q8_0 => q8_0(row, x),
        GgmlType::Q4_0 => q4_0(row, x),
        GgmlType::Q4_K => q4_k(row, x),
        GgmlType::Q5_K => q5_k(row, x),
        GgmlType::Q6_K => q6_k(row, x),
        _ => return None,
    })
}

pub fn f32_row(row: &[u8], x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 4];
    for (index, (chunk, activation)) in row.chunks_exact(4).zip(x).enumerate() {
        let weight = f32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)"));
        lanes[index & 3] += weight * activation;
    }
    total(lanes)
}

pub fn q8_0(row: &[u8], x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 4];
    for (block, activations) in row.chunks_exact(34).zip(x.chunks_exact(32)) {
        let d = half::read_f16(block);
        for (index, (quant, activation)) in block[2..34].iter().zip(activations).enumerate() {
            lanes[index & 3] += d * (*quant as i8) as f32 * activation;
        }
    }
    total(lanes)
}

pub fn q4_0(row: &[u8], x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 4];
    for (block, activations) in row.chunks_exact(18).zip(x.chunks_exact(32)) {
        let d = half::read_f16(block);
        // Element j is the low nibble of byte j; element j + 16 is the high
        // nibble. Walking indices in order means two passes over the same bytes.
        for (index, byte) in block[2..18].iter().enumerate() {
            lanes[index & 3] += d * ((byte & 0x0F) as f32 - 8.0) * activations[index];
        }
        for (index, byte) in block[2..18].iter().enumerate() {
            lanes[index & 3] += d * ((byte >> 4) as f32 - 8.0) * activations[16 + index];
        }
    }
    total(lanes)
}

pub fn q4_k(row: &[u8], x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 4];
    for (block, activations) in row.chunks_exact(144).zip(x.chunks_exact(256)) {
        let d = half::read_f16(block);
        let dmin = half::read_f16(&block[2..]);
        let scales = &block[4..16];
        let quants = &block[16..144];

        for group in 0..4 {
            let bytes = &quants[group * 32..(group + 1) * 32];
            let acts = &activations[group * 64..(group + 1) * 64];
            let (scale_low, min_low) = k::scale_min(group * 2, scales);
            let (scale_high, min_high) = k::scale_min(group * 2 + 1, scales);
            let (d_low, offset_low) = (d * scale_low as f32, dmin * min_low as f32);
            let (d_high, offset_high) = (d * scale_high as f32, dmin * min_high as f32);

            for (index, byte) in bytes.iter().enumerate() {
                lanes[index & 3] += (d_low * (byte & 0x0F) as f32 - offset_low) * acts[index];
            }
            // 32 is a multiple of 4, so the lane for element 32 + index is the
            // same one index would land in.
            for (index, byte) in bytes.iter().enumerate() {
                lanes[index & 3] += (d_high * (byte >> 4) as f32 - offset_high) * acts[32 + index];
            }
        }
    }
    total(lanes)
}

pub fn q5_k(row: &[u8], x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 4];
    for (block, activations) in row.chunks_exact(176).zip(x.chunks_exact(256)) {
        let d = half::read_f16(block);
        let dmin = half::read_f16(&block[2..]);
        let scales = &block[4..16];
        let high = &block[16..48];
        let quants = &block[48..176];

        for group in 0..4 {
            let bytes = &quants[group * 32..(group + 1) * 32];
            let acts = &activations[group * 64..(group + 1) * 64];
            let (scale_low, min_low) = k::scale_min(group * 2, scales);
            let (scale_high, min_high) = k::scale_min(group * 2 + 1, scales);
            let (d_low, offset_low) = (d * scale_low as f32, dmin * min_low as f32);
            let (d_high, offset_high) = (d * scale_high as f32, dmin * min_high as f32);
            let mask_low = 1u8 << (group * 2);
            let mask_high = 2u8 << (group * 2);

            for (index, byte) in bytes.iter().enumerate() {
                let fifth = if high[index] & mask_low != 0 { 16 } else { 0 };
                lanes[index & 3] +=
                    (d_low * ((byte & 0x0F) + fifth) as f32 - offset_low) * acts[index];
            }
            for (index, byte) in bytes.iter().enumerate() {
                let fifth = if high[index] & mask_high != 0 { 16 } else { 0 };
                lanes[index & 3] +=
                    (d_high * ((byte >> 4) + fifth) as f32 - offset_high) * acts[32 + index];
            }
        }
    }
    total(lanes)
}

pub fn q6_k(row: &[u8], x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 4];
    for (block, activations) in row.chunks_exact(210).zip(x.chunks_exact(256)) {
        let low = &block[0..128];
        let high = &block[128..192];
        let scales = &block[192..208];
        let d = half::read_f16(&block[208..]);

        for half_block in 0..2 {
            let low = &low[half_block * 64..(half_block + 1) * 64];
            let high = &high[half_block * 32..(half_block + 1) * 32];
            let scales = &scales[half_block * 8..(half_block + 1) * 8];
            let acts = &activations[half_block * 128..(half_block + 1) * 128];

            // Quads run in index order — 0..32, 32..64, 64..96, 96..128 — which
            // is what keeps the lane assignment matching the unfused loop.
            for quad in 0..4 {
                let (source, shift, scale_base) = match quad {
                    0 => (0usize, 0u32, 0usize),
                    1 => (32, 2, 2),
                    2 => (0, 4, 4),
                    _ => (32, 6, 6),
                };
                let upper_nibble = quad >= 2;

                for index in 0..32 {
                    let nibble = if upper_nibble {
                        low[source + index] >> 4
                    } else {
                        low[source + index] & 0x0F
                    };
                    let quant = (nibble | (((high[index] >> shift) & 3) << 4)) as i8 - 32;
                    let scale = scales[scale_base + index / 16] as i8;
                    lanes[index & 3] += d * scale as f32 * quant as f32 * acts[quad * 32 + index];
                }
            }
        }
    }
    total(lanes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant;

    /// Deterministic bytes that exercise every bit position.
    fn pattern(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| ((i * 37 + 11) % 251) as u8)
            .collect::<Vec<_>>()
    }

    fn activations(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.37).sin() * 2.0).collect()
    }

    /// The property the whole module exists for: fusing must not change a
    /// single bit relative to dequantize-then-dot.
    fn assert_identical(ty: GgmlType, blocks: usize) {
        let (block, size) = ty.layout().expect("sized type");
        let elements = blocks * block as usize;
        let row = pattern(blocks * size as usize);
        let x = activations(elements);

        let expanded = quant::dequantize_to_vec(ty, &row, elements).expect("dequantize");
        let reference = crate::ops::dot(&expanded, &x);
        let got = fused(ty, &row, &x).expect("fused kernel exists");

        assert_eq!(
            got.to_bits(),
            reference.to_bits(),
            "{}: fused {got} != reference {reference}",
            ty.name()
        );
    }

    #[test]
    fn q8_0_matches_the_reference_exactly() {
        assert_identical(GgmlType::Q8_0, 1);
        assert_identical(GgmlType::Q8_0, 7);
    }

    #[test]
    fn q4_0_matches_the_reference_exactly() {
        assert_identical(GgmlType::Q4_0, 1);
        assert_identical(GgmlType::Q4_0, 5);
    }

    #[test]
    fn q4_k_matches_the_reference_exactly() {
        assert_identical(GgmlType::Q4_K, 1);
        assert_identical(GgmlType::Q4_K, 3);
    }

    #[test]
    fn q5_k_matches_the_reference_exactly() {
        assert_identical(GgmlType::Q5_K, 1);
        assert_identical(GgmlType::Q5_K, 3);
    }

    #[test]
    fn q6_k_matches_the_reference_exactly() {
        assert_identical(GgmlType::Q6_K, 1);
        assert_identical(GgmlType::Q6_K, 3);
    }

    #[test]
    fn f32_matches_the_reference_exactly() {
        assert_identical(GgmlType::F32, 64);
    }

    #[test]
    fn unfused_types_report_themselves() {
        assert!(fused(GgmlType::F16, &[0; 2], &[1.0]).is_none());
        assert!(fused(GgmlType::Q2_K, &[0; 84], &[0.0; 256]).is_none());
    }
}
