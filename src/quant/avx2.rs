//! AVX2 integer kernels.
//!
//! Only the integer dots are vectorized here, and that is deliberate. Regroup
//! an f32 sum and the answer changes in the last bits, so a SIMD f32 kernel can
//! only ever be *close* to the scalar one. Integer addition is associative and
//! exact, so an i32 kernel that computes the same products in any order is
//! **identical** — which means the vector path can be diffed against the scalar
//! path with `assert_eq!`, not a tolerance.
//!
//! Nothing here is compiled outside x86-64, and nothing here runs without a
//! runtime check: `std::arch` is in the standard library, but AVX2 is not in
//! the baseline target, so calling these on a machine without it is undefined
//! behaviour rather than a crash.
//!
//! ## What it actually bought
//!
//! | type | scalar int8 | AVX2 int8 |
//! | --- | --- | --- |
//! | Q4_K | 0.99 ms | **0.36 ms** |
//! | Q8_0 | 0.45 ms | 0.46 ms |
//!
//! Q8_0 is a wash. Its scalar inner loop is a flat multiply-accumulate over
//! `chunks_exact(4)` and LLVM was already emitting vector instructions for it;
//! there was nothing left to hand-write. Q4_K gains 2.7x because its nibble
//! masking is interleaved with the accumulation, which autovectorization
//! handles badly and `and`/`srli`/`maddubs` handles directly.
//!
//! Which is the general lesson: intrinsics pay where the compiler has already
//! failed, and nowhere else. Both kernels stay, because the Q8_0 one costs
//! nothing and is exact, but it earned its place by measurement rather than by
//! being SIMD.

use std::arch::x86_64::*;
use std::sync::OnceLock;

use super::activation::Quantized;
use super::half;
use super::k;

/// Whether this CPU has AVX2. Detected once, then free.
pub fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| is_x86_feature_detected!("avx2"))
}

/// Horizontal sum of eight i32 lanes.
///
/// Exact regardless of order, which is the whole reason this file only does
/// integers.
#[target_feature(enable = "avx2")]
unsafe fn hsum(value: __m256i) -> i32 {
    let halves = _mm_add_epi32(
        _mm256_castsi256_si128(value),
        _mm256_extracti128_si256(value, 1),
    );
    let pairs = _mm_add_epi32(halves, _mm_shuffle_epi32(halves, 0b01_00_11_10));
    let single = _mm_add_epi32(pairs, _mm_shuffle_epi32(pairs, 0b00_01_00_01));
    _mm_cvtsi128_si32(single)
}

/// `Σ w[i]·a[i]` over 32 signed bytes each.
///
/// `maddubs` needs its first operand unsigned, so the sign is moved onto the
/// activation with `sign_epi8` and the weight is made absolute — the classic
/// trick, and the products stay inside i16: 127·127·2 = 32258.
#[target_feature(enable = "avx2")]
unsafe fn dot32(weights: *const i8, acts: *const i8) -> i32 {
    let w = _mm256_loadu_si256(weights as *const __m256i);
    let a = _mm256_loadu_si256(acts as *const __m256i);
    let magnitude = _mm256_sign_epi8(w, w);
    let signed = _mm256_sign_epi8(a, w);
    let pairs = _mm256_maddubs_epi16(magnitude, signed);
    hsum(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
}

/// `Σ n[i]·a[i]` where `n` are 32 unsigned nibbles already isolated in a
/// register. Products reach 15·127·2 = 3810, comfortably inside i16.
#[target_feature(enable = "avx2")]
unsafe fn dot_nibbles(nibbles: __m256i, acts: *const i8) -> i32 {
    let a = _mm256_loadu_si256(acts as *const __m256i);
    let pairs = _mm256_maddubs_epi16(nibbles, a);
    hsum(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q8_0(row: &[u8], x: &Quantized) -> f32 {
    let mut acc = 0.0f32;
    for (index, block) in row.chunks_exact(34).enumerate() {
        let d = half::read_f16(block);
        let (acts, scale, _) = x.block(index);
        let sum = dot32(block[2..].as_ptr() as *const i8, acts.as_ptr());
        acc += d * scale * sum as f32;
    }
    acc
}

/// # Safety
/// Caller must have checked [`available`].
///
/// Only the integer dot is vectorized; the per-sub-block combination stays in
/// scalar f32 and in the same order as the scalar kernel, so the two agree
/// exactly rather than approximately.
#[target_feature(enable = "avx2")]
pub unsafe fn q4_k(row: &[u8], x: &Quantized) -> f32 {
    let low_mask = _mm256_set1_epi8(0x0F);
    let mut acc = 0.0f32;

    for (super_index, block) in row.chunks_exact(144).enumerate() {
        let d = half::read_f16(block);
        let dmin = half::read_f16(&block[2..]);
        let scales = &block[4..16];
        let quants = &block[16..144];
        let first_sub = super_index * 8;

        for group in 0..4 {
            let bytes = _mm256_loadu_si256(quants[group * 32..].as_ptr() as *const __m256i);
            let low = _mm256_and_si256(bytes, low_mask);
            let high = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low_mask);

            let (scale_low, min_low) = k::scale_min(group * 2, scales);
            let (scale_high, min_high) = k::scale_min(group * 2 + 1, scales);
            let (acts_low, s_low, sum_low) = x.block(first_sub + group * 2);
            let (acts_high, s_high, sum_high) = x.block(first_sub + group * 2 + 1);

            let dot_low = dot_nibbles(low, acts_low.as_ptr());
            let dot_high = dot_nibbles(high, acts_high.as_ptr());

            acc += s_low
                * ((d * scale_low as f32) * dot_low as f32
                    - (dmin * min_low as f32) * sum_low as f32);
            acc += s_high
                * ((d * scale_high as f32) * dot_high as f32
                    - (dmin * min_high as f32) * sum_high as f32);
        }
    }
    acc
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q4_0(row: &[u8], x: &Quantized) -> f32 {
    let low_mask = _mm_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut acc = 0.0f32;

    for (index, block) in row.chunks_exact(18).enumerate() {
        let d = half::read_f16(block);
        let quants = _mm_loadu_si128(block[2..18].as_ptr() as *const __m128i);
        let low = _mm_and_si128(quants, low_mask);
        let high = _mm_and_si128(_mm_srli_epi16(quants, 4), low_mask);
        let nibbles = _mm256_set_m128i(high, low);

        let (acts, scale, sum) = x.block(index);
        let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);
        let pairs = _mm256_maddubs_epi16(nibbles, a);
        let dot = hsum(_mm256_madd_epi16(pairs, ones));

        acc += d * scale * (dot - 8 * sum) as f32;
    }
    acc
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q5_k(row: &[u8], x: &Quantized) -> f32 {
    let low_mask = _mm256_set1_epi8(0x0F);
    let bit16 = _mm256_set1_epi8(16);
    let zeros = _mm256_setzero_si256();
    let mut acc = 0.0f32;

    for (super_index, block) in row.chunks_exact(176).enumerate() {
        let d = half::read_f16(block);
        let dmin = half::read_f16(&block[2..]);
        let scales = &block[4..16];
        let qh = _mm256_loadu_si256(block[16..48].as_ptr() as *const __m256i);
        let qs = &block[48..176];
        let first_sub = super_index * 8;

        for group in 0..4 {
            let bytes = _mm256_loadu_si256(qs[group * 32..].as_ptr() as *const __m256i);
            let low4 = _mm256_and_si256(bytes, low_mask);
            let high4 = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low_mask);

            let mask_low = _mm256_set1_epi8(1i8 << (group * 2));
            let mask_high = _mm256_set1_epi8(2i8 << (group * 2));

            let has_low = _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask_low), zeros);
            let fifth_low = _mm256_andnot_si256(has_low, bit16);
            let has_high = _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask_high), zeros);
            let fifth_high = _mm256_andnot_si256(has_high, bit16);

            let low = _mm256_or_si256(low4, fifth_low);
            let high = _mm256_or_si256(high4, fifth_high);

            let (scale_low, min_low) = k::scale_min(group * 2, scales);
            let (scale_high, min_high) = k::scale_min(group * 2 + 1, scales);
            let (acts_low, s_low, sum_low) = x.block(first_sub + group * 2);
            let (acts_high, s_high, sum_high) = x.block(first_sub + group * 2 + 1);

            let dot_low = dot_nibbles(low, acts_low.as_ptr());
            let dot_high = dot_nibbles(high, acts_high.as_ptr());

            acc += s_low
                * ((d * scale_low as f32) * dot_low as f32
                    - (dmin * min_low as f32) * sum_low as f32);
            acc += s_high
                * ((d * scale_high as f32) * dot_high as f32
                    - (dmin * min_high as f32) * sum_high as f32);
        }
    }
    acc
}

/// Horizontal sum of four i32 lanes in a 128-bit register.
#[target_feature(enable = "avx2")]
unsafe fn hsum128(v: __m128i) -> i32 {
    let pairs = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b01_00_11_10));
    let single = _mm_add_epi32(pairs, _mm_shuffle_epi32(pairs, 0b00_01_00_01));
    _mm_cvtsi128_si32(single)
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q6_k(row: &[u8], x: &Quantized) -> f32 {
    let low_mask = _mm256_set1_epi8(0x0F);
    let high_mask = _mm256_set1_epi8(0x30);
    let offset32 = _mm256_set1_epi8(32);
    let ones128 = _mm_set1_epi16(1);
    let mut acc = 0.0f32;

    for (super_index, block) in row.chunks_exact(210).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = half::read_f16(&block[208..]);
        let first_sub = super_index * 8;

        for half_block in 0..2 {
            let ql_half = &ql[half_block * 64..(half_block + 1) * 64];
            let qh_vec = _mm256_loadu_si256(qh[half_block * 32..].as_ptr() as *const __m256i);
            let ql_0 = _mm256_loadu_si256(ql_half[0..32].as_ptr() as *const __m256i);
            let ql_1 = _mm256_loadu_si256(ql_half[32..64].as_ptr() as *const __m256i);
            let sc = &scales[half_block * 8..(half_block + 1) * 8];

            let q0 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(ql_0, low_mask),
                    _mm256_and_si256(_mm256_slli_epi16(qh_vec, 4), high_mask),
                ),
                offset32,
            );
            let q1 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(ql_1, low_mask),
                    _mm256_and_si256(_mm256_slli_epi16(qh_vec, 2), high_mask),
                ),
                offset32,
            );
            let q2 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql_0, 4), low_mask),
                    _mm256_and_si256(qh_vec, high_mask),
                ),
                offset32,
            );
            let q3 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql_1, 4), low_mask),
                    _mm256_and_si256(_mm256_srli_epi16(qh_vec, 2), high_mask),
                ),
                offset32,
            );

            let quads = [q0, q1, q2, q3];
            for (quad_idx, &w) in quads.iter().enumerate() {
                let sub_idx = first_sub + half_block * 4 + quad_idx;
                let (acts, s_act, _) = x.block(sub_idx);
                let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);
                let mag = _mm256_sign_epi8(w, w);
                let signed = _mm256_sign_epi8(a, w);
                let pairs = _mm256_maddubs_epi16(mag, signed);

                let lo16 = _mm256_castsi256_si128(pairs);
                let hi16 = _mm256_extracti128_si256(pairs, 1);
                let dot_lo = hsum128(_mm_madd_epi16(lo16, ones128));
                let dot_hi = hsum128(_mm_madd_epi16(hi16, ones128));

                let sc_lo = sc[quad_idx * 2] as i8 as i32;
                let sc_hi = sc[quad_idx * 2 + 1] as i8 as i32;
                let weighted = sc_lo * dot_lo + sc_hi * dot_hi;
                acc += d * s_act * weighted as f32;
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgmlType;
    use crate::quant::idot;
    use crate::quant::testdata::{activations, row};

    /// The property this file is built around: same products, any order, same
    /// integer total — so the vector and scalar paths agree exactly.
    fn assert_matches_scalar(ty: GgmlType, blocks: usize) {
        if !available() {
            eprintln!("no AVX2 on this machine; skipping");
            return;
        }
        let (block, _) = ty.layout().expect("sized type");
        let elements = blocks * block as usize;
        let weights = row(ty, blocks);

        let mut quantized = Quantized::with_capacity(elements);
        quantized.fill(&activations(elements));

        let scalar = match ty {
            GgmlType::Q8_0 => idot::q8_0(&weights, &quantized),
            GgmlType::Q4_0 => idot::q4_0(&weights, &quantized),
            GgmlType::Q4_K => idot::q4_k(&weights, &quantized),
            GgmlType::Q5_K => idot::q5_k(&weights, &quantized),
            GgmlType::Q6_K => idot::q6_k(&weights, &quantized),
            other => panic!("no scalar kernel for {}", other.name()),
        };
        let vector = unsafe {
            match ty {
                GgmlType::Q8_0 => q8_0(&weights, &quantized),
                GgmlType::Q4_0 => q4_0(&weights, &quantized),
                GgmlType::Q4_K => q4_k(&weights, &quantized),
                GgmlType::Q5_K => q5_k(&weights, &quantized),
                GgmlType::Q6_K => q6_k(&weights, &quantized),
                other => panic!("no vector kernel for {}", other.name()),
            }
        };

        assert_eq!(
            vector.to_bits(),
            scalar.to_bits(),
            "{}: avx2 {vector} != scalar {scalar}",
            ty.name()
        );
    }

    #[test]
    fn q8_0_matches_the_scalar_kernel_exactly() {
        assert_matches_scalar(GgmlType::Q8_0, 1);
        assert_matches_scalar(GgmlType::Q8_0, 9);
    }

    #[test]
    fn q4_0_matches_the_scalar_kernel_exactly() {
        assert_matches_scalar(GgmlType::Q4_0, 1);
        assert_matches_scalar(GgmlType::Q4_0, 9);
    }

    #[test]
    fn q4_k_matches_the_scalar_kernel_exactly() {
        assert_matches_scalar(GgmlType::Q4_K, 1);
        assert_matches_scalar(GgmlType::Q4_K, 5);
    }

    #[test]
    fn q5_k_matches_the_scalar_kernel_exactly() {
        assert_matches_scalar(GgmlType::Q5_K, 1);
        assert_matches_scalar(GgmlType::Q5_K, 5);
    }

    #[test]
    fn q6_k_matches_the_scalar_kernel_exactly() {
        assert_matches_scalar(GgmlType::Q6_K, 1);
        assert_matches_scalar(GgmlType::Q6_K, 5);
    }

    #[test]
    fn detection_is_stable() {
        assert_eq!(available(), available());
    }
}
