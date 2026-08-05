//! Integer dot products against a quantized activation vector.
//!
//! Both sides stay small: weights are already 4 to 8 bits, the activation has
//! been squeezed to int8 by [`super::activation`], and the accumulation happens
//! in i32. Only the per-block scales are floating point, and there is one of
//! those per 32 elements rather than one per element.
//!
//! The arithmetic that makes this work is that every block's weight is an
//! affine function of a small integer:
//!
//! ```text
//! w[i] = d * n[i] - m          (K-quants; m is zero for Q8_0 and Q4_0)
//! x[i] ≈ s * a[i]
//!
//! Σ w[i]·x[i] ≈ s · ( d · Σ n[i]·a[i]  −  m · Σ a[i] )
//! ```
//!
//! So the inner loop is an integer multiply-accumulate, and the minimum term
//! collapses into a single multiply against the activation block sum that
//! `Quantized` already carries.
//!
//! **These kernels are approximate.** Everything in [`super::dot`] is
//! bit-identical to the reference; this is not, because quantizing the
//! activation discards information. The error is bounded by half a step per
//! element and is small next to a 4-bit weight's own error — the tests pin an
//! actual relative bound rather than trusting that.

use super::activation::{Quantized, BLOCK};
use super::{half, k};
use crate::gguf::GgmlType;

/// Which weight types have an integer kernel.
pub fn supports(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q4_K)
}

/// Dot one quantized weight row against a quantized activation vector.
pub fn integer(ty: GgmlType, row: &[u8], x: &Quantized) -> Option<f32> {
    Some(match ty {
        GgmlType::Q8_0 => q8_0(row, x),
        GgmlType::Q4_0 => q4_0(row, x),
        GgmlType::Q4_K => q4_k(row, x),
        _ => return None,
    })
}

#[inline]
fn total(lanes: [i32; 4]) -> i32 {
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

/// Widest possible block accumulator is 32 × 127 × 127, well inside i32.
pub fn q8_0(row: &[u8], x: &Quantized) -> f32 {
    let mut acc = 0.0f32;
    for (index, block) in row.chunks_exact(34).enumerate() {
        let d = half::read_f16(block);
        let (acts, scale, _) = x.block(index);

        let mut lanes = [0i32; 4];
        for (weights, a) in block[2..34].chunks_exact(4).zip(acts.chunks_exact(4)) {
            for lane in 0..4 {
                lanes[lane] += (weights[lane] as i8) as i32 * a[lane] as i32;
            }
        }
        acc += d * scale * total(lanes) as f32;
    }
    acc
}

/// `w = d · (n − 8)`, so the offset factors out into `8 × Σa`.
pub fn q4_0(row: &[u8], x: &Quantized) -> f32 {
    let mut acc = 0.0f32;
    for (index, block) in row.chunks_exact(18).enumerate() {
        let d = half::read_f16(block);
        let quants = &block[2..18];
        let (acts, scale, sum) = x.block(index);

        let mut lanes = [0i32; 4];
        for (weights, a) in quants.chunks_exact(4).zip(acts[..16].chunks_exact(4)) {
            for lane in 0..4 {
                lanes[lane] += (weights[lane] & 0x0F) as i32 * a[lane] as i32;
            }
        }
        for (weights, a) in quants.chunks_exact(4).zip(acts[16..].chunks_exact(4)) {
            for lane in 0..4 {
                lanes[lane] += (weights[lane] >> 4) as i32 * a[lane] as i32;
            }
        }
        acc += d * scale * (total(lanes) - 8 * sum) as f32;
    }
    acc
}

/// The one that matters: `Q4_K` is what most models ship as, and it is the type
/// the f32 fused path could not beat.
///
/// Each 256-element super-block is eight 32-element sub-blocks, and an
/// activation block is also 32, so they line up one to one — no partial blocks,
/// no straddling.
pub fn q4_k(row: &[u8], x: &Quantized) -> f32 {
    let mut acc = 0.0f32;
    for (super_index, block) in row.chunks_exact(144).enumerate() {
        let d = half::read_f16(block);
        let dmin = half::read_f16(&block[2..]);
        let scales = &block[4..16];
        let quants = &block[16..144];
        let first_sub = super_index * (256 / BLOCK);

        for group in 0..4 {
            let bytes = &quants[group * 32..(group + 1) * 32];
            let (scale_low, min_low) = k::scale_min(group * 2, scales);
            let (scale_high, min_high) = k::scale_min(group * 2 + 1, scales);

            let (acts_low, s_low, sum_low) = x.block(first_sub + group * 2);
            let (acts_high, s_high, sum_high) = x.block(first_sub + group * 2 + 1);

            let mut lanes_low = [0i32; 4];
            let mut lanes_high = [0i32; 4];
            for ((weights, a_low), a_high) in bytes
                .chunks_exact(4)
                .zip(acts_low.chunks_exact(4))
                .zip(acts_high.chunks_exact(4))
            {
                for lane in 0..4 {
                    lanes_low[lane] += (weights[lane] & 0x0F) as i32 * a_low[lane] as i32;
                    lanes_high[lane] += (weights[lane] >> 4) as i32 * a_high[lane] as i32;
                }
            }

            acc += s_low
                * ((d * scale_low as f32) * total(lanes_low) as f32
                    - (dmin * min_low as f32) * sum_low as f32);
            acc += s_high
                * ((d * scale_high as f32) * total(lanes_high) as f32
                    - (dmin * min_high as f32) * sum_high as f32);
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant;
    use crate::quant::testdata::{activations, row};

    /// Relative error of the integer path against the exact f32 dot.
    fn relative_error(ty: GgmlType, blocks: usize) -> f32 {
        let (block, _) = ty.layout().expect("sized type");
        let elements = blocks * block as usize;
        let row = row(ty, blocks);
        let x = activations(elements);

        let expanded = quant::dequantize_to_vec(ty, &row, elements).expect("dequantize");
        assert!(expanded.iter().all(|v| v.is_finite()), "bad fixture");
        let exact = crate::ops::dot(&expanded, &x);

        let mut quantized = Quantized::with_capacity(elements);
        quantized.fill(&x);
        let got = integer(ty, &row, &quantized).expect("integer kernel exists");

        let magnitude = expanded
            .iter()
            .zip(&x)
            .map(|(w, a)| (w * a).abs())
            .sum::<f32>()
            .max(f32::MIN_POSITIVE);
        (got - exact).abs() / magnitude
    }

    /// Quantizing the activation costs accuracy. This pins how much — if a
    /// kernel is ever wired up wrong, the error jumps orders of magnitude and
    /// this catches it, where an exact-equality test could not exist at all.
    #[test]
    fn q8_0_stays_close_to_the_exact_dot() {
        for blocks in [1, 4, 16] {
            let error = relative_error(GgmlType::Q8_0, blocks);
            assert!(error < 5e-3, "{blocks} blocks: relative error {error}");
        }
    }

    #[test]
    fn q4_0_stays_close_to_the_exact_dot() {
        for blocks in [1, 4, 16] {
            let error = relative_error(GgmlType::Q4_0, blocks);
            assert!(error < 5e-3, "{blocks} blocks: relative error {error}");
        }
    }

    #[test]
    fn q4_k_stays_close_to_the_exact_dot() {
        for blocks in [1, 2, 8] {
            let error = relative_error(GgmlType::Q4_K, blocks);
            assert!(error < 5e-3, "{blocks} blocks: relative error {error}");
        }
    }

    /// When the activation happens to quantize losslessly, the approximation
    /// should vanish entirely.
    ///
    /// That needs values that are exact multiples of their own step, and the
    /// step is `max|x| / 127` — so the block has to span the full int8 range in
    /// even increments. Merely being whole numbers is not enough, which is the
    /// mistake the first version of this test made.
    #[test]
    fn losslessly_quantizable_activations_are_reproduced_exactly() {
        let weights = row(GgmlType::Q8_0, 1);
        // -127, -119, ..., 121: max magnitude 127, so the step is exactly 1.
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 8.0 - 127.0).collect();

        let mut quantized = Quantized::with_capacity(32);
        quantized.fill(&x);
        assert_eq!(quantized.scales[0], 1.0, "step should be exactly one");
        assert_eq!(quantized.quants[0], -127);

        let expanded = quant::dequantize_to_vec(GgmlType::Q8_0, &weights, 32).unwrap();
        let exact = crate::ops::dot(&expanded, &x);
        let got = q8_0(&weights, &quantized);

        assert!(
            (got - exact).abs() / exact.abs().max(1.0) < 1e-5,
            "got {got}, exact {exact}"
        );
    }

    #[test]
    fn zero_activations_give_zero() {
        let mut quantized = Quantized::with_capacity(256);
        quantized.fill(&[0.0; 256]);
        assert_eq!(q4_k(&row(GgmlType::Q4_K, 1), &quantized), 0.0);
        assert_eq!(q8_0(&row(GgmlType::Q8_0, 8), &quantized), 0.0);
    }

    #[test]
    fn types_without_an_integer_kernel_report_themselves() {
        assert!(supports(GgmlType::Q4_K));
        assert!(!supports(GgmlType::Q6_K));
        assert!(!supports(GgmlType::F32));
        let quantized = Quantized::with_capacity(256);
        assert!(integer(GgmlType::Q6_K, &row(GgmlType::Q6_K, 1), &quantized).is_none());
    }
}
