//! ARM NEON integer kernels for Q8_0 and Q4_K.
//!
//! Evaluates dot products using AArch64 SIMD intrinsics.

use std::arch::aarch64::*;

use super::activation::Quantized;
use super::half;
use super::k;

pub fn available() -> bool {
    true
}

#[target_feature(enable = "neon")]
unsafe fn dot32(weights: *const i8, acts: *const i8) -> i32 {
    let w0 = vld1q_s8(weights);
    let a0 = vld1q_s8(acts);
    let w1 = vld1q_s8(weights.add(16));
    let a1 = vld1q_s8(acts.add(16));

    // Multiply low and high halves into i16x8 vectors
    let p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(a0));
    let p0 = vmlal_s8(p0, vget_high_s8(w0), vget_high_s8(a0));

    let p1 = vmull_s8(vget_low_s8(w1), vget_low_s8(a1));
    let p1 = vmlal_s8(p1, vget_high_s8(w1), vget_high_s8(a1));

    // Accumulate i16 into i32x4
    let acc0 = vpaddlq_s16(p0);
    let acc1 = vpadalq_s16(acc0, p1);

    vaddvq_s32(acc1)
}

#[target_feature(enable = "neon")]
unsafe fn dot_nibbles(nibbles: uint8x16_t, acts: *const i8) -> i32 {
    let n_s8 = vreinterpretq_s8_u8(nibbles);
    let a = vld1q_s8(acts);

    let p = vmull_s8(vget_low_s8(n_s8), vget_low_s8(a));
    let p = vmlal_s8(p, vget_high_s8(n_s8), vget_high_s8(a));

    let acc = vpaddlq_s16(p);
    vaddvq_s32(acc)
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "neon")]
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
#[target_feature(enable = "neon")]
pub unsafe fn q4_k(row: &[u8], x: &Quantized) -> f32 {
    let mask = vdupq_n_u8(0x0F);
    let mut acc = 0.0f32;

    for (super_index, block) in row.chunks_exact(144).enumerate() {
        let d = half::read_f16(block);
        let dmin = half::read_f16(&block[2..]);
        let scales = &block[4..16];
        let quants = &block[16..144];
        let first_sub = super_index * 8;

        for group in 0..4 {
            let b0 = vld1q_u8(quants[group * 32..].as_ptr());
            let b1 = vld1q_u8(quants[group * 32 + 16..].as_ptr());

            let low0 = vandq_u8(b0, mask);
            let high0 = vandq_u8(vshrq_n_u8(b0, 4), mask);

            let low1 = vandq_u8(b1, mask);
            let high1 = vandq_u8(vshrq_n_u8(b1, 4), mask);

            let (scale_low, min_low) = k::scale_min(group * 2, scales);
            let (scale_high, min_high) = k::scale_min(group * 2 + 1, scales);
            let (acts_low, s_low, sum_low) = x.block(first_sub + group * 2);
            let (acts_high, s_high, sum_high) = x.block(first_sub + group * 2 + 1);

            let dot_low =
                dot_nibbles(low0, acts_low.as_ptr()) + dot_nibbles(low1, acts_low[16..].as_ptr());
            let dot_high = dot_nibbles(high0, acts_high.as_ptr())
                + dot_nibbles(high1, acts_high[16..].as_ptr());

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
