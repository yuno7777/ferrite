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
            GgmlType::Q4_K => idot::q4_k(&weights, &quantized),
            other => panic!("no scalar kernel for {}", other.name()),
        };
        let vector = unsafe {
            match ty {
                GgmlType::Q8_0 => q8_0(&weights, &quantized),
                _ => q4_k(&weights, &quantized),
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
    fn q4_k_matches_the_scalar_kernel_exactly() {
        assert_matches_scalar(GgmlType::Q4_K, 1);
        assert_matches_scalar(GgmlType::Q4_K, 5);
    }

    #[test]
    fn detection_is_stable() {
        assert_eq!(available(), available());
    }
}
