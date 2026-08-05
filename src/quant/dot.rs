//! Fused dequantize-and-dot.
//!
//! The unfused path expands a weight row into an f32 scratch buffer and then
//! dots it. That writes the whole row to memory and reads it straight back —
//! for a 4096-wide row, 16 KB written and 16 KB read, per row, per token.
//! Fusing the expansion into the dot deletes both.
//!
//! Every kernel accumulates into the same four lanes, in the same index order,
//! as [`crate::ops::dot`]. That makes the fused result **bit-identical** to
//! dequantizing first, so the fast path can be diffed against the reference
//! rather than spot-checked.
//!
//! The loops are written as `chunks_exact(4)` with a constant lane index rather
//! than `lanes[index & 3]`. Both compute the same thing; only the first one
//! vectorizes. The masked form was measurably *slower* than the unfused path it
//! replaced, because the compiler could not prove the lane pattern and fell
//! back to scalar code — the shape of the loop matters more here than the
//! memory traffic it saves.

use super::{half, k};
use crate::gguf::GgmlType;

/// Sum the lanes exactly as `ops::dot` does — left to right, not
/// `iter().sum()`, which folds in a leading zero and disturbs signed zeros.
#[inline]
fn total(lanes: [f32; 4]) -> f32 {
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

/// Which types fusion is actually *faster* for, measured rather than assumed.
///
/// From `cargo run --release --example bench` on a 16-thread x86-64 machine,
/// fused versus expand-then-dot, at 2048x2048 and 4096x4096:
///
/// | type | speedup |
/// | --- | --- |
/// | Q6_K | 2.2 – 2.3x |
/// | Q8_0 | 1.5 – 1.6x |
/// | Q4_0 | 1.1 – 1.2x |
/// | Q5_K | 1.1 – 1.2x |
/// | Q4_K | **0.67x** |
/// | F32  | ~1.0x |
///
/// Q4_K loses, consistently and by a lot, which is inconvenient because it is
/// the format most models ship as. The reason is that its unpacking is already
/// cheap — one mask and one shift per byte — and the unfused path spends it in
/// two long, cleanly vectorized loops. Fusing chops that into eight short runs
/// per super-block, and the loop overhead costs more than the round trip
/// through the scratch buffer saves. Q6_K wins for the mirror-image reason: its
/// unpacking is expensive enough that halving the memory traffic dominates.
///
/// F32's "dequantize" is a byte-order conversion the compiler turns into a
/// copy, so there is nothing to fuse away.
///
/// The [`q4_k`] kernel below stays public and tested — it is what the benchmark
/// compares against, and the day the loop structure improves this table is the
/// one line to change.
pub fn supports(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q5_K | GgmlType::Q6_K
    )
}

/// Dot one weight row, still quantized, against an activation vector.
///
/// `None` when the unfused path is the faster one for this type; the caller
/// falls back to dequantize-then-dot. See [`supports`] for which is which and
/// why.
pub fn fused(ty: GgmlType, row: &[u8], x: &[f32]) -> Option<f32> {
    if !supports(ty) {
        return None;
    }
    Some(match ty {
        GgmlType::Q8_0 => q8_0(row, x),
        GgmlType::Q4_0 => q4_0(row, x),
        GgmlType::Q5_K => q5_k(row, x),
        GgmlType::Q6_K => q6_k(row, x),
        _ => return None,
    })
}

pub fn q8_0(row: &[u8], x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 4];
    for (block, activations) in row.chunks_exact(34).zip(x.chunks_exact(32)) {
        let d = half::read_f16(block);
        for (quants, acts) in block[2..34]
            .chunks_exact(4)
            .zip(activations.chunks_exact(4))
        {
            for lane in 0..4 {
                lanes[lane] += d * (quants[lane] as i8) as f32 * acts[lane];
            }
        }
    }
    total(lanes)
}

pub fn q4_0(row: &[u8], x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 4];
    for (block, activations) in row.chunks_exact(18).zip(x.chunks_exact(32)) {
        let d = half::read_f16(block);
        let quants = &block[2..18];
        // Element j is the low nibble of byte j; element j + 16 is the high
        // nibble. Ascending index order means two passes over the same bytes.
        for (bytes, acts) in quants
            .chunks_exact(4)
            .zip(activations[..16].chunks_exact(4))
        {
            for lane in 0..4 {
                lanes[lane] += d * ((bytes[lane] & 0x0F) as f32 - 8.0) * acts[lane];
            }
        }
        for (bytes, acts) in quants
            .chunks_exact(4)
            .zip(activations[16..].chunks_exact(4))
        {
            for lane in 0..4 {
                lanes[lane] += d * ((bytes[lane] >> 4) as f32 - 8.0) * acts[lane];
            }
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

            for (quants, a) in bytes.chunks_exact(4).zip(acts[..32].chunks_exact(4)) {
                for lane in 0..4 {
                    lanes[lane] += (d_low * (quants[lane] & 0x0F) as f32 - offset_low) * a[lane];
                }
            }
            for (quants, a) in bytes.chunks_exact(4).zip(acts[32..].chunks_exact(4)) {
                for lane in 0..4 {
                    lanes[lane] += (d_high * (quants[lane] >> 4) as f32 - offset_high) * a[lane];
                }
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

            for ((quants, highs), a) in bytes
                .chunks_exact(4)
                .zip(high.chunks_exact(4))
                .zip(acts[..32].chunks_exact(4))
            {
                for lane in 0..4 {
                    let fifth = if highs[lane] & mask_low != 0 { 16 } else { 0 };
                    lanes[lane] +=
                        (d_low * ((quants[lane] & 0x0F) + fifth) as f32 - offset_low) * a[lane];
                }
            }
            for ((quants, highs), a) in bytes
                .chunks_exact(4)
                .zip(high.chunks_exact(4))
                .zip(acts[32..].chunks_exact(4))
            {
                for lane in 0..4 {
                    let fifth = if highs[lane] & mask_high != 0 { 16 } else { 0 };
                    lanes[lane] +=
                        (d_high * ((quants[lane] >> 4) + fifth) as f32 - offset_high) * a[lane];
                }
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
            // keeps lane assignment matching the unfused loop.
            for quad in 0..4 {
                let (source, shift, scale_base) = match quad {
                    0 => (0usize, 0u32, 0usize),
                    1 => (32, 2, 2),
                    2 => (0, 4, 4),
                    _ => (32, 6, 6),
                };
                let upper = quad >= 2;

                // The scale changes every 16 elements; hoisting it out leaves
                // the inner run scale-invariant.
                for sub in 0..2 {
                    let scaled = d * scales[scale_base + sub] as i8 as f32;
                    let at = sub * 16;
                    let lows = &low[source + at..source + at + 16];
                    let highs = &high[at..at + 16];
                    let a = &acts[quad * 32 + at..quad * 32 + at + 16];

                    for ((l, h), act) in lows
                        .chunks_exact(4)
                        .zip(highs.chunks_exact(4))
                        .zip(a.chunks_exact(4))
                    {
                        for lane in 0..4 {
                            let nibble = if upper { l[lane] >> 4 } else { l[lane] & 0x0F };
                            let quant = (nibble | (((h[lane] >> shift) & 3) << 4)) as i8 - 32;
                            lanes[lane] += scaled * quant as f32 * act[lane];
                        }
                    }
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
    use crate::quant::testdata::{activations, row};

    /// The property the whole module exists for: fusing must not change a
    /// single bit relative to dequantize-then-dot.
    ///
    /// Calls the kernel directly rather than through `fused`, so a type that is
    /// switched off for being slow is still held to the same correctness bar.
    fn assert_identical(ty: GgmlType, blocks: usize) {
        let (block, _) = ty.layout().expect("sized type");
        let elements = blocks * block as usize;
        let row = row(ty, blocks);
        let x = activations(elements);

        let expanded = quant::dequantize_to_vec(ty, &row, elements).expect("dequantize");
        assert!(
            expanded.iter().all(|v| v.is_finite()),
            "fixture produced non-finite weights; the comparison would be vacuous"
        );
        let reference = crate::ops::dot(&expanded, &x);
        let got = match ty {
            GgmlType::Q8_0 => q8_0(&row, &x),
            GgmlType::Q4_0 => q4_0(&row, &x),
            GgmlType::Q4_K => q4_k(&row, &x),
            GgmlType::Q5_K => q5_k(&row, &x),
            GgmlType::Q6_K => q6_k(&row, &x),
            other => panic!("no kernel for {}", other.name()),
        };

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
    fn selection_matches_the_measurements() {
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q4_0,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
        ] {
            assert!(supports(ty), "{} should be fused", ty.name());
        }
        // Measured slower fused; see the table on `supports`.
        assert!(!supports(GgmlType::Q4_K));
        assert!(!supports(GgmlType::F32));
        // No kernel at all.
        assert!(!supports(GgmlType::F16));

        assert!(fused(GgmlType::Q4_K, &[0; 144], &[0.0; 256]).is_none());
        assert!(fused(GgmlType::F16, &[0; 2], &[1.0]).is_none());
        assert!(fused(GgmlType::Q8_0, &[0; 34], &[0.0; 32]).is_some());
    }
}
