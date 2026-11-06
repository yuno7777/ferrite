//! Quantizing the activation vector, so the dot product can be integer.
//!
//! Weights arrive as small integers and get widened to f32 just to be
//! multiplied by an f32 activation. That throws away most of the arithmetic
//! width the hardware has: an AVX2 register holds 8 f32 lanes but 32 int8s, and
//! integer multiply-accumulate is cheaper besides.
//!
//! So the activation is quantized once per matvec — O(cols), against a matvec's
//! O(rows × cols) — and every row then dots against it in i32.
//!
//! **This is an approximation**, unlike everything else in `quant`. The fused
//! f32 kernels are bit-identical to the reference; these are not, and cannot
//! be. The error is bounded at half a step per element and is small next to the
//! error already present in a 4-bit weight, which is why llama.cpp does the
//! same thing. But it is a real change to the numbers, so it is measured rather
//! than assumed — see the tests, and `idot`'s accuracy bounds.

/// Elements per activation block. Matches the 32-element blocks of `Q8_0` and
/// `Q4_0`, and divides the 256-element K-quant super-blocks exactly, so one
/// format serves all of them.
pub const BLOCK: usize = 32;

/// An activation vector quantized to int8, block by block.
///
/// Reusable: `fill` overwrites in place so a decode step does not allocate.
#[derive(Clone, Debug, Default)]
pub struct Quantized {
    /// One int8 per element.
    pub quants: Vec<i8>,
    /// One scale per block. `value ≈ quant * scale`.
    pub scales: Vec<f32>,
    /// Sum of the quants in each block.
    ///
    /// K-quants subtract a per-sub-block minimum from every weight, and that
    /// term factors out of the dot product into `minimum × Σquants`. Keeping
    /// the sums here means the inner loop never has to compute them.
    pub sums: Vec<i32>,
}

impl Quantized {
    /// Capacity for `len` elements, rounded up to whole blocks.
    pub fn with_capacity(len: usize) -> Self {
        let blocks = len.div_ceil(BLOCK);
        Self {
            quants: vec![0; blocks * BLOCK],
            scales: vec![0.0; blocks],
            sums: vec![0; blocks],
        }
    }

    pub fn len(&self) -> usize {
        self.quants.len()
    }

    pub fn is_empty(&self) -> bool {
        self.quants.is_empty()
    }

    /// Quantize `x`, growing the buffers if needed.
    ///
    /// `x.len()` must be a whole number of blocks — every weight layout this
    /// serves has a block size that is a multiple of 32, so a partial block
    /// would mean a malformed tensor rather than a short vector.
    pub fn fill(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len() % BLOCK, 0, "activation length must fill blocks");
        let blocks = x.len() / BLOCK;
        if self.quants.len() < x.len() {
            self.quants.resize(x.len(), 0);
            self.scales.resize(blocks, 0.0);
            self.sums.resize(blocks, 0);
        }

        #[cfg(target_arch = "x86_64")]
        if super::avx2::available() && is_x86_feature_detected!("fma") {
            unsafe {
                self.fill_avx2(x);
            }
            return;
        }

        self.fill_scalar(x);
    }

    fn fill_scalar(&mut self, x: &[f32]) {
        for (index, values) in x.chunks_exact(BLOCK).enumerate() {
            let amax = values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = amax / 127.0;
            let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            let quants = &mut self.quants[index * BLOCK..(index + 1) * BLOCK];
            let mut sum = 0i32;
            for (slot, value) in quants.iter_mut().zip(values) {
                // Clamp as well as round: an amax that rounds up can otherwise
                // produce 128, which is not an i8.
                let quant = (value * inverse).round().clamp(-127.0, 127.0) as i8;
                *slot = quant;
                sum += quant as i32;
            }
            self.scales[index] = scale;
            self.sums[index] = sum;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn fill_avx2(&mut self, x: &[f32]) {
        use std::arch::x86_64::*;
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
        let min_i32 = _mm256_set1_epi32(-127);
        let max_i32 = _mm256_set1_epi32(127);
        let half_pos = _mm256_set1_ps(0.5);
        let half_neg = _mm256_set1_ps(-0.5);
        let zero_ps = _mm256_setzero_ps();

        for (index, values) in x.chunks_exact(BLOCK).enumerate() {
            let ptr = values.as_ptr();
            let v0 = _mm256_loadu_ps(ptr);
            let v1 = _mm256_loadu_ps(ptr.add(8));
            let v2 = _mm256_loadu_ps(ptr.add(16));
            let v3 = _mm256_loadu_ps(ptr.add(24));

            let a0 = _mm256_and_ps(v0, abs_mask);
            let a1 = _mm256_and_ps(v1, abs_mask);
            let a2 = _mm256_and_ps(v2, abs_mask);
            let a3 = _mm256_and_ps(v3, abs_mask);

            let m01 = _mm256_max_ps(a0, a1);
            let m23 = _mm256_max_ps(a2, a3);
            let m = _mm256_max_ps(m01, m23);

            let hi128 = _mm256_extractf128_ps(m, 1);
            let m128 = _mm_max_ps(_mm256_castps256_ps128(m), hi128);
            let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
            let amax = _mm_cvtss_f32(m32);

            let scale = amax / 127.0;
            let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            let inv_vec = _mm256_set1_ps(inverse);

            // Round half away from zero to match `f32::round()` bit-for-bit:
            let round_away = |v: __m256| -> __m256i {
                let scaled = _mm256_mul_ps(v, inv_vec);
                let neg_mask = _mm256_cmp_ps(scaled, zero_ps, _CMP_LT_OQ);
                let offset = _mm256_blendv_ps(half_pos, half_neg, neg_mask);
                let shifted = _mm256_add_ps(scaled, offset);
                let i32s = _mm256_cvttps_epi32(shifted);
                _mm256_min_epi32(_mm256_max_epi32(i32s, min_i32), max_i32)
            };

            let q0 = round_away(v0);
            let q1 = round_away(v1);
            let q2 = round_away(v2);
            let q3 = round_away(v3);

            let sum01 = _mm256_add_epi32(q0, q1);
            let sum23 = _mm256_add_epi32(q2, q3);
            let sum_all = _mm256_add_epi32(sum01, sum23);
            let sum_halves = _mm_add_epi32(
                _mm256_castsi256_si128(sum_all),
                _mm256_extracti128_si256(sum_all, 1),
            );
            let sum_pairs = _mm_add_epi32(sum_halves, _mm_shuffle_epi32(sum_halves, 0b01_00_11_10));
            let sum_single = _mm_add_epi32(sum_pairs, _mm_shuffle_epi32(sum_pairs, 0b00_01_00_01));
            let sum = _mm_cvtsi128_si32(sum_single);

            // Pack four 8xi32 vectors down to 32xi8 in linear order:
            let p01 = _mm256_packs_epi32(q0, q1);
            let p23 = _mm256_packs_epi32(q2, q3);
            let p01_ordered = _mm256_permute4x64_epi64(p01, 0b11_01_10_00);
            let p23_ordered = _mm256_permute4x64_epi64(p23, 0b11_01_10_00);
            let packed_i8 = _mm256_packs_epi16(p01_ordered, p23_ordered);
            let ordered_i8 = _mm256_permute4x64_epi64(packed_i8, 0b11_01_10_00);

            let dst = self.quants.as_mut_ptr().add(index * BLOCK);
            _mm256_storeu_si256(dst as *mut __m256i, ordered_i8);
            *self.scales.get_unchecked_mut(index) = scale;
            *self.sums.get_unchecked_mut(index) = sum;
        }
    }

    /// Quants and scale for one block.
    #[inline(always)]
    pub fn block(&self, index: usize) -> (&[i8], f32, i32) {
        (
            &self.quants[index * BLOCK..(index + 1) * BLOCK],
            self.scales[index],
            self.sums[index],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32) * 0.37).sin() * 3.0 + 0.1)
            .collect()
    }

    #[test]
    fn every_element_lands_within_half_a_step() {
        let x = sample(128);
        let mut q = Quantized::with_capacity(x.len());
        q.fill(&x);

        for (index, values) in x.chunks_exact(BLOCK).enumerate() {
            let (quants, scale, _) = q.block(index);
            for (original, quant) in values.iter().zip(quants) {
                let restored = *quant as f32 * scale;
                assert!(
                    (original - restored).abs() <= scale / 2.0 + 1e-6,
                    "block {index}: {original} -> {restored}, step {scale}"
                );
            }
        }
    }

    #[test]
    fn the_extreme_maps_to_the_extreme() {
        // The largest magnitude in a block should use the full int8 range,
        // otherwise precision is being left on the table.
        let mut x = vec![0.1; BLOCK];
        x[7] = -5.0;
        let mut q = Quantized::with_capacity(x.len());
        q.fill(&x);
        let (quants, scale, _) = q.block(0);
        assert_eq!(quants[7], -127);
        assert!((scale - 5.0 / 127.0).abs() < 1e-9);
    }

    #[test]
    fn sums_match_the_quants() {
        let x = sample(96);
        let mut q = Quantized::with_capacity(x.len());
        q.fill(&x);
        for index in 0..3 {
            let (quants, _, sum) = q.block(index);
            let expected: i32 = quants.iter().map(|v| *v as i32).sum();
            assert_eq!(sum, expected);
        }
    }

    #[test]
    fn an_all_zero_block_does_not_divide_by_zero() {
        let mut q = Quantized::with_capacity(BLOCK);
        q.fill(&[0.0; BLOCK]);
        let (quants, scale, sum) = q.block(0);
        assert_eq!(scale, 0.0);
        assert_eq!(sum, 0);
        assert!(quants.iter().all(|v| *v == 0));
    }

    #[test]
    fn refilling_reuses_the_buffers() {
        let mut q = Quantized::with_capacity(BLOCK);
        q.fill(&sample(BLOCK));
        let first = q.quants.as_ptr();
        q.fill(&sample(BLOCK));
        assert_eq!(q.quants.as_ptr(), first, "should not reallocate");
        assert_eq!(q.len(), BLOCK);
    }

    #[test]
    fn a_longer_vector_grows_the_buffers() {
        let mut q = Quantized::with_capacity(BLOCK);
        q.fill(&sample(BLOCK * 4));
        assert_eq!(q.len(), BLOCK * 4);
        assert_eq!(q.scales.len(), 4);
        assert_eq!(q.sums.len(), 4);
    }
}
