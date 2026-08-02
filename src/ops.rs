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

/// Plain dot product, accumulated in four lanes.
///
/// The lanes exist so the compiler can vectorize: a single accumulator forces a
/// serial dependency chain, and f32 addition is not associative so it will not
/// split one on its own.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut lanes = [0.0f32; 4];
    let full = a.len() - a.len() % 4;
    for start in (0..full).step_by(4) {
        for lane in 0..4 {
            lanes[lane] += a[start + lane] * b[start + lane];
        }
    }
    let tail: f32 = a[full..].iter().zip(&b[full..]).map(|(x, y)| x * y).sum();
    lanes[0] + lanes[1] + lanes[2] + lanes[3] + tail
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
    fn matvec_multiplies_rows() {
        // [[1, 2], [3, 4], [0, -1]] * [1, 10]
        let matrix = [1.0, 2.0, 3.0, 4.0, 0.0, -1.0];
        let mut out = [0.0; 3];
        matvec(&matrix, &[1.0, 10.0], &mut out);
        assert_eq!(out, [21.0, 43.0, -10.0]);
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
}
