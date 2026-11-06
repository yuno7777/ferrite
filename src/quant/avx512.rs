//! AVX-512 integer kernels for Q8_0 and Q4_K.

#![allow(clippy::incompatible_msrv)]

use std::arch::x86_64::*;
use std::sync::OnceLock;

use super::activation::Quantized;
use super::half;
use super::k;

pub fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE
        .get_or_init(|| is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw"))
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot32(weights: *const i8, acts: *const i8) -> i32 {
    let w = _mm256_loadu_si256(weights as *const __m256i);
    let a = _mm256_loadu_si256(acts as *const __m256i);
    let magnitude = _mm256_sign_epi8(w, w);
    let signed = _mm256_sign_epi8(a, w);
    let pairs = _mm256_maddubs_epi16(magnitude, signed);
    let added = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    let z = _mm512_castsi256_si512(added);
    _mm512_reduce_add_epi32(z)
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx512f,avx512bw")]
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

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_nibbles(nibbles: __m256i, acts: *const i8) -> i32 {
    let a = _mm256_loadu_si256(acts as *const __m256i);
    let pairs = _mm256_maddubs_epi16(nibbles, a);
    let added = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    let z = _mm512_castsi256_si512(added);
    _mm512_reduce_add_epi32(z)
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx512f,avx512bw")]
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

    fn assert_matches_scalar(ty: GgmlType, blocks: usize) {
        if !available() {
            eprintln!("no AVX-512 on this machine; skipping");
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
            "{}: avx512 {vector} != scalar {scalar}",
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
