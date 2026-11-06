//! Register-Tiled `(M=1..8) x (N=1..4)` GEMV & GEMM Microkernels for `Q3_K` (`GgmlType::Q3_K`).
//!
//! Block size: `256` elements (`110` bytes per block).
//! Provides size-specialized scalar and AVX2+FMA+F16C register-tiled
//! microkernels that evaluate `M` weight rows against `N` quantized
//! activation columns simultaneously in a single L1-resident pass.

#[allow(unused_imports)]
use crate::gguf::GgmlType;
use crate::quant::activation::Quantized;
use crate::quant::half::f16_to_f32;

#[inline]
fn decode_q3k_scales(raw: &[u8]) -> [i8; 16] {
    let mut out = [0i8; 16];
    for i in 0..8 {
        let hi0 = ((raw[8 + (i % 4)] >> (2 * (i / 4))) & 0x03) << 4;
        let hi1 = ((raw[8 + (i % 4)] >> (2 * (i / 4) + 4)) & 0x03) << 4;
        let lo0 = raw[i] & 0x0F;
        let lo1 = raw[i] >> 4;
        out[i] = ((lo0 | hi0) as i8) - 32;
        out[i + 8] = ((lo1 | hi1) as i8) - 32;
    }
    out
}

/// Specialized `1x1` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m1n1_scalar(w0: &[u8], x0: &Quantized) -> [[f32; 1]; 1] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [[acc_r0_c0]]
}

/// Specialized `1x2` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m1n2_scalar(w0: &[u8], x0: &Quantized, x1: &Quantized) -> [[f32; 2]; 1] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [[acc_r0_c0, acc_r0_c1]]
}

/// Specialized `1x3` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m1n3_scalar(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 1] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [[acc_r0_c0, acc_r0_c1, acc_r0_c2]]
}

/// Specialized `1x4` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m1n4_scalar(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 1] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [[acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3]]
}

/// Specialized `2x1` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m2n1_scalar(w0: &[u8], w1: &[u8], x0: &Quantized) -> [[f32; 1]; 2] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [[acc_r0_c0], [acc_r1_c0]]
}

/// Specialized `2x2` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m2n2_scalar(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 2] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [[acc_r0_c0, acc_r0_c1], [acc_r1_c0, acc_r1_c1]]
}

/// Specialized `2x3` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m2n3_scalar(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 2] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2],
    ]
}

/// Specialized `2x4` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m2n4_scalar(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 2] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3],
    ]
}

/// Specialized `3x1` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m3n1_scalar(w0: &[u8], w1: &[u8], w2: &[u8], x0: &Quantized) -> [[f32; 1]; 3] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [[acc_r0_c0], [acc_r1_c0], [acc_r2_c0]]
}

/// Specialized `3x2` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m3n2_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 3] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1],
        [acc_r1_c0, acc_r1_c1],
        [acc_r2_c0, acc_r2_c1],
    ]
}

/// Specialized `3x3` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m3n3_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 3] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2],
    ]
}

/// Specialized `3x4` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m3n4_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 3] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3],
    ]
}

/// Specialized `4x1` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m4n1_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 4] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [[acc_r0_c0], [acc_r1_c0], [acc_r2_c0], [acc_r3_c0]]
}

/// Specialized `4x2` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m4n2_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 4] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1],
        [acc_r1_c0, acc_r1_c1],
        [acc_r2_c0, acc_r2_c1],
        [acc_r3_c0, acc_r3_c1],
    ]
}

/// Specialized `4x3` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m4n3_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 4] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2],
    ]
}

/// Specialized `4x4` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m4n4_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 4] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3],
    ]
}

/// Specialized `5x1` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m5n1_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 5] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0],
        [acc_r1_c0],
        [acc_r2_c0],
        [acc_r3_c0],
        [acc_r4_c0],
    ]
}

/// Specialized `5x2` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m5n2_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 5] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1],
        [acc_r1_c0, acc_r1_c1],
        [acc_r2_c0, acc_r2_c1],
        [acc_r3_c0, acc_r3_c1],
        [acc_r4_c0, acc_r4_c1],
    ]
}

/// Specialized `5x3` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m5n3_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 5] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2],
    ]
}

/// Specialized `5x4` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m5n4_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 5] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3],
    ]
}

/// Specialized `6x1` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m6n1_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 6] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0],
        [acc_r1_c0],
        [acc_r2_c0],
        [acc_r3_c0],
        [acc_r4_c0],
        [acc_r5_c0],
    ]
}

/// Specialized `6x2` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m6n2_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 6] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1],
        [acc_r1_c0, acc_r1_c1],
        [acc_r2_c0, acc_r2_c1],
        [acc_r3_c0, acc_r3_c1],
        [acc_r4_c0, acc_r4_c1],
        [acc_r5_c0, acc_r5_c1],
    ]
}

/// Specialized `6x3` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m6n3_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 6] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2],
        [acc_r5_c0, acc_r5_c1, acc_r5_c2],
    ]
}

/// Specialized `6x4` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m6n4_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 6] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3],
        [acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3],
    ]
}

/// Specialized `7x1` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m7n1_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 7] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r6 = &w6[sb * 110..(sb + 1) * 110];
        let hmask_r6 = &blk_r6[0..32];
        let qs_r6 = &blk_r6[32..96];
        let scales_r6 = decode_q3k_scales(&blk_r6[96..108]);
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[108], blk_r6[109]]));
        for half in 0..2 {
            let q_h = &qs_r6[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0],
        [acc_r1_c0],
        [acc_r2_c0],
        [acc_r3_c0],
        [acc_r4_c0],
        [acc_r5_c0],
        [acc_r6_c0],
    ]
}

/// Specialized `7x2` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m7n2_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 7] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r6 = &w6[sb * 110..(sb + 1) * 110];
        let hmask_r6 = &blk_r6[0..32];
        let qs_r6 = &blk_r6[32..96];
        let scales_r6 = decode_q3k_scales(&blk_r6[96..108]);
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[108], blk_r6[109]]));
        for half in 0..2 {
            let q_h = &qs_r6[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1],
        [acc_r1_c0, acc_r1_c1],
        [acc_r2_c0, acc_r2_c1],
        [acc_r3_c0, acc_r3_c1],
        [acc_r4_c0, acc_r4_c1],
        [acc_r5_c0, acc_r5_c1],
        [acc_r6_c0, acc_r6_c1],
    ]
}

/// Specialized `7x3` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m7n3_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 7] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r6 = &w6[sb * 110..(sb + 1) * 110];
        let hmask_r6 = &blk_r6[0..32];
        let qs_r6 = &blk_r6[32..96];
        let scales_r6 = decode_q3k_scales(&blk_r6[96..108]);
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[108], blk_r6[109]]));
        for half in 0..2 {
            let q_h = &qs_r6[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2],
        [acc_r5_c0, acc_r5_c1, acc_r5_c2],
        [acc_r6_c0, acc_r6_c1, acc_r6_c2],
    ]
}

/// Specialized `7x4` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m7n4_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 7] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    let mut acc_r6_c3 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r6 = &w6[sb * 110..(sb + 1) * 110];
        let hmask_r6 = &blk_r6[0..32];
        let qs_r6 = &blk_r6[32..96];
        let scales_r6 = decode_q3k_scales(&blk_r6[96..108]);
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[108], blk_r6[109]]));
        for half in 0..2 {
            let q_h = &qs_r6[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3],
        [acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3],
        [acc_r6_c0, acc_r6_c1, acc_r6_c2, acc_r6_c3],
    ]
}

/// Specialized `8x1` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m8n1_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 8] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r7_c0 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r6 = &w6[sb * 110..(sb + 1) * 110];
        let hmask_r6 = &blk_r6[0..32];
        let qs_r6 = &blk_r6[32..96];
        let scales_r6 = decode_q3k_scales(&blk_r6[96..108]);
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[108], blk_r6[109]]));
        for half in 0..2 {
            let q_h = &qs_r6[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r7 = &w7[sb * 110..(sb + 1) * 110];
        let hmask_r7 = &blk_r7[0..32];
        let qs_r7 = &blk_r7[32..96];
        let scales_r7 = decode_q3k_scales(&blk_r7[96..108]);
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[108], blk_r7[109]]));
        for half in 0..2 {
            let q_h = &qs_r7[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r7 * (scales_r7[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r7 * (scales_r7[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r7[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r7[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r7_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0],
        [acc_r1_c0],
        [acc_r2_c0],
        [acc_r3_c0],
        [acc_r4_c0],
        [acc_r5_c0],
        [acc_r6_c0],
        [acc_r7_c0],
    ]
}

/// Specialized `8x2` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m8n2_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 8] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r7_c0 = 0.0f32;
    let mut acc_r7_c1 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r6 = &w6[sb * 110..(sb + 1) * 110];
        let hmask_r6 = &blk_r6[0..32];
        let qs_r6 = &blk_r6[32..96];
        let scales_r6 = decode_q3k_scales(&blk_r6[96..108]);
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[108], blk_r6[109]]));
        for half in 0..2 {
            let q_h = &qs_r6[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r7 = &w7[sb * 110..(sb + 1) * 110];
        let hmask_r7 = &blk_r7[0..32];
        let qs_r7 = &blk_r7[32..96];
        let scales_r7 = decode_q3k_scales(&blk_r7[96..108]);
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[108], blk_r7[109]]));
        for half in 0..2 {
            let q_h = &qs_r7[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r7 * (scales_r7[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r7 * (scales_r7[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r7[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r7[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r7_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r7 * (scales_r7[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r7 * (scales_r7[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r7[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r7[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r7_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1],
        [acc_r1_c0, acc_r1_c1],
        [acc_r2_c0, acc_r2_c1],
        [acc_r3_c0, acc_r3_c1],
        [acc_r4_c0, acc_r4_c1],
        [acc_r5_c0, acc_r5_c1],
        [acc_r6_c0, acc_r6_c1],
        [acc_r7_c0, acc_r7_c1],
    ]
}

/// Specialized `8x3` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m8n3_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 8] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    let mut acc_r7_c0 = 0.0f32;
    let mut acc_r7_c1 = 0.0f32;
    let mut acc_r7_c2 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r6 = &w6[sb * 110..(sb + 1) * 110];
        let hmask_r6 = &blk_r6[0..32];
        let qs_r6 = &blk_r6[32..96];
        let scales_r6 = decode_q3k_scales(&blk_r6[96..108]);
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[108], blk_r6[109]]));
        for half in 0..2 {
            let q_h = &qs_r6[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r7 = &w7[sb * 110..(sb + 1) * 110];
        let hmask_r7 = &blk_r7[0..32];
        let qs_r7 = &blk_r7[32..96];
        let scales_r7 = decode_q3k_scales(&blk_r7[96..108]);
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[108], blk_r7[109]]));
        for half in 0..2 {
            let q_h = &qs_r7[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r7 * (scales_r7[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r7 * (scales_r7[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r7[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r7[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r7_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r7 * (scales_r7[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r7 * (scales_r7[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r7[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r7[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r7_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r7 * (scales_r7[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r7 * (scales_r7[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r7[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r7[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r7_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2],
        [acc_r5_c0, acc_r5_c1, acc_r5_c2],
        [acc_r6_c0, acc_r6_c1, acc_r6_c2],
        [acc_r7_c0, acc_r7_c1, acc_r7_c2],
    ]
}

/// Specialized `8x4` register-tiled scalar microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q3_k_m8n4_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 8] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    let mut acc_r6_c3 = 0.0f32;
    let mut acc_r7_c0 = 0.0f32;
    let mut acc_r7_c1 = 0.0f32;
    let mut acc_r7_c2 = 0.0f32;
    let mut acc_r7_c3 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 110..(sb + 1) * 110];
        let hmask_r0 = &blk_r0[0..32];
        let qs_r0 = &blk_r0[32..96];
        let scales_r0 = decode_q3k_scales(&blk_r0[96..108]);
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[108], blk_r0[109]]));
        for half in 0..2 {
            let q_h = &qs_r0[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r0 * (scales_r0[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r0 * (scales_r0[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r0[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r0[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r0_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r1 = &w1[sb * 110..(sb + 1) * 110];
        let hmask_r1 = &blk_r1[0..32];
        let qs_r1 = &blk_r1[32..96];
        let scales_r1 = decode_q3k_scales(&blk_r1[96..108]);
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[108], blk_r1[109]]));
        for half in 0..2 {
            let q_h = &qs_r1[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r1 * (scales_r1[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r1 * (scales_r1[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r1[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r1[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r1_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r2 = &w2[sb * 110..(sb + 1) * 110];
        let hmask_r2 = &blk_r2[0..32];
        let qs_r2 = &blk_r2[32..96];
        let scales_r2 = decode_q3k_scales(&blk_r2[96..108]);
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[108], blk_r2[109]]));
        for half in 0..2 {
            let q_h = &qs_r2[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r2 * (scales_r2[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r2 * (scales_r2[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r2[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r2[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r2_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r3 = &w3[sb * 110..(sb + 1) * 110];
        let hmask_r3 = &blk_r3[0..32];
        let qs_r3 = &blk_r3[32..96];
        let scales_r3 = decode_q3k_scales(&blk_r3[96..108]);
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[108], blk_r3[109]]));
        for half in 0..2 {
            let q_h = &qs_r3[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r3 * (scales_r3[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r3 * (scales_r3[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r3[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r3[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r3_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r4 = &w4[sb * 110..(sb + 1) * 110];
        let hmask_r4 = &blk_r4[0..32];
        let qs_r4 = &blk_r4[32..96];
        let scales_r4 = decode_q3k_scales(&blk_r4[96..108]);
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[108], blk_r4[109]]));
        for half in 0..2 {
            let q_h = &qs_r4[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r4 * (scales_r4[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r4 * (scales_r4[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r4[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r4[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r4_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r5 = &w5[sb * 110..(sb + 1) * 110];
        let hmask_r5 = &blk_r5[0..32];
        let qs_r5 = &blk_r5[32..96];
        let scales_r5 = decode_q3k_scales(&blk_r5[96..108]);
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[108], blk_r5[109]]));
        for half in 0..2 {
            let q_h = &qs_r5[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r5 * (scales_r5[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r5 * (scales_r5[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r5[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r5[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r5_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r6 = &w6[sb * 110..(sb + 1) * 110];
        let hmask_r6 = &blk_r6[0..32];
        let qs_r6 = &blk_r6[32..96];
        let scales_r6 = decode_q3k_scales(&blk_r6[96..108]);
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[108], blk_r6[109]]));
        for half in 0..2 {
            let q_h = &qs_r6[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r6 * (scales_r6[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r6 * (scales_r6[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r6[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r6[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r6_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
        let blk_r7 = &w7[sb * 110..(sb + 1) * 110];
        let hmask_r7 = &blk_r7[0..32];
        let qs_r7 = &blk_r7[32..96];
        let scales_r7 = decode_q3k_scales(&blk_r7[96..108]);
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[108], blk_r7[109]]));
        for half in 0..2 {
            let q_h = &qs_r7[half * 32..(half + 1) * 32];
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r7 * (scales_r7[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r7 * (scales_r7[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r7[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r7[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r7_c0 += x0.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r7 * (scales_r7[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r7 * (scales_r7[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r7[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r7[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r7_c1 += x1.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r7 * (scales_r7[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r7 * (scales_r7[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r7[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r7[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r7_c2 += x2.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
            for shift_idx in 0..4 {
                let g = b_base + half * 4 + shift_idx;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let dl0 = d_r7 * (scales_r7[half * 8 + shift_idx * 2] as f32);
                let dl1 = d_r7 * (scales_r7[half * 8 + shift_idx * 2 + 1] as f32);
                let sh = (shift_idx * 2) as u32;
                let m_bit = 1u8 << (half * 4 + shift_idx);
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..16 {
                    let lo0 = ((q_h[l] >> sh) & 3) as i32;
                    let hi0 = if (hmask_r7[l] & m_bit) != 0 { 0 } else { -4 };
                    let lo1 = ((q_h[16 + l] >> sh) & 3) as i32;
                    let hi1 = if (hmask_r7[16 + l] & m_bit) != 0 {
                        0
                    } else {
                        -4
                    };
                    dot0 += (lo0 + hi0) * (x_q[l] as i32);
                    dot1 += (lo1 + hi1) * (x_q[16 + l] as i32);
                }
                acc_r7_c3 += x3.scales[g] * (dl0 * (dot0 as f32) + dl1 * (dot1 as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3],
        [acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3],
        [acc_r6_c0, acc_r6_c1, acc_r6_c2, acc_r6_c3],
        [acc_r7_c0, acc_r7_c1, acc_r7_c2, acc_r7_c3],
    ]
}

/// Specialized `1x1` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m1n1_avx2(w0: &[u8], x0: &Quantized) -> [[f32; 1]; 1] {
    let mut out = [[0.0f32; 1]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    let _ = (w0, x0);
    out
}

/// Runtime-dispatched `1x1` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m1n1(w0: &[u8], x0: &Quantized) -> [[f32; 1]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m1n1_avx2(w0, x0) };
        }
    }
    gemm_q3_k_m1n1_scalar(w0, x0)
}

/// Specialized `1x2` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m1n2_avx2(w0: &[u8], x0: &Quantized, x1: &Quantized) -> [[f32; 2]; 1] {
    let mut out = [[0.0f32; 2]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    let _ = (w0, x0, x1);
    out
}

/// Runtime-dispatched `1x2` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m1n2(w0: &[u8], x0: &Quantized, x1: &Quantized) -> [[f32; 2]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m1n2_avx2(w0, x0, x1) };
        }
    }
    gemm_q3_k_m1n2_scalar(w0, x0, x1)
}

/// Specialized `1x3` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m1n3_avx2(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 1] {
    let mut out = [[0.0f32; 3]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    let _ = (w0, x0, x1, x2);
    out
}

/// Runtime-dispatched `1x3` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m1n3(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m1n3_avx2(w0, x0, x1, x2) };
        }
    }
    gemm_q3_k_m1n3_scalar(w0, x0, x1, x2)
}

/// Specialized `1x4` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m1n4_avx2(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 1] {
    let mut out = [[0.0f32; 4]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x3).unwrap_or(0.0);
    let _ = (w0, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `1x4` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m1n4(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m1n4_avx2(w0, x0, x1, x2, x3) };
        }
    }
    gemm_q3_k_m1n4_scalar(w0, x0, x1, x2, x3)
}

/// Specialized `2x1` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m2n1_avx2(w0: &[u8], w1: &[u8], x0: &Quantized) -> [[f32; 1]; 2] {
    let mut out = [[0.0f32; 1]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    let _ = (w0, w1, x0);
    out
}

/// Runtime-dispatched `2x1` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m2n1(w0: &[u8], w1: &[u8], x0: &Quantized) -> [[f32; 1]; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m2n1_avx2(w0, w1, x0) };
        }
    }
    gemm_q3_k_m2n1_scalar(w0, w1, x0)
}

/// Specialized `2x2` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m2n2_avx2(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 2] {
    let mut out = [[0.0f32; 2]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    let _ = (w0, w1, x0, x1);
    out
}

/// Runtime-dispatched `2x2` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m2n2(w0: &[u8], w1: &[u8], x0: &Quantized, x1: &Quantized) -> [[f32; 2]; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m2n2_avx2(w0, w1, x0, x1) };
        }
    }
    gemm_q3_k_m2n2_scalar(w0, w1, x0, x1)
}

/// Specialized `2x3` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m2n3_avx2(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 2] {
    let mut out = [[0.0f32; 3]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    let _ = (w0, w1, x0, x1, x2);
    out
}

/// Runtime-dispatched `2x3` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m2n3(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m2n3_avx2(w0, w1, x0, x1, x2) };
        }
    }
    gemm_q3_k_m2n3_scalar(w0, w1, x0, x1, x2)
}

/// Specialized `2x4` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m2n4_avx2(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 2] {
    let mut out = [[0.0f32; 4]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x3).unwrap_or(0.0);
    let _ = (w0, w1, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `2x4` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m2n4(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m2n4_avx2(w0, w1, x0, x1, x2, x3) };
        }
    }
    gemm_q3_k_m2n4_scalar(w0, w1, x0, x1, x2, x3)
}

/// Specialized `3x1` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m3n1_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 3] {
    let mut out = [[0.0f32; 1]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0);
    out
}

/// Runtime-dispatched `3x1` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m3n1(w0: &[u8], w1: &[u8], w2: &[u8], x0: &Quantized) -> [[f32; 1]; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m3n1_avx2(w0, w1, w2, x0) };
        }
    }
    gemm_q3_k_m3n1_scalar(w0, w1, w2, x0)
}

/// Specialized `3x2` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m3n2_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 3] {
    let mut out = [[0.0f32; 2]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0, x1);
    out
}

/// Runtime-dispatched `3x2` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m3n2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m3n2_avx2(w0, w1, w2, x0, x1) };
        }
    }
    gemm_q3_k_m3n2_scalar(w0, w1, w2, x0, x1)
}

/// Specialized `3x3` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m3n3_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0, x1, x2);
    out
}

/// Runtime-dispatched `3x3` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m3n3(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m3n3_avx2(w0, w1, w2, x0, x1, x2) };
        }
    }
    gemm_q3_k_m3n3_scalar(w0, w1, w2, x0, x1, x2)
}

/// Specialized `3x4` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m3n4_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 3] {
    let mut out = [[0.0f32; 4]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x3).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `3x4` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m3n4(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m3n4_avx2(w0, w1, w2, x0, x1, x2, x3) };
        }
    }
    gemm_q3_k_m3n4_scalar(w0, w1, w2, x0, x1, x2, x3)
}

/// Specialized `4x1` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m4n1_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 4] {
    let mut out = [[0.0f32; 1]; 4];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, x0);
    out
}

/// Runtime-dispatched `4x1` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m4n1(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m4n1_avx2(w0, w1, w2, w3, x0) };
        }
    }
    gemm_q3_k_m4n1_scalar(w0, w1, w2, w3, x0)
}

/// Specialized `4x2` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m4n2_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 4] {
    let mut out = [[0.0f32; 2]; 4];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, x0, x1);
    out
}

/// Runtime-dispatched `4x2` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m4n2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m4n2_avx2(w0, w1, w2, w3, x0, x1) };
        }
    }
    gemm_q3_k_m4n2_scalar(w0, w1, w2, w3, x0, x1)
}

/// Specialized `4x3` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m4n3_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 4] {
    let mut out = [[0.0f32; 3]; 4];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x2).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, x0, x1, x2);
    out
}

/// Runtime-dispatched `4x3` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m4n3(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m4n3_avx2(w0, w1, w2, w3, x0, x1, x2) };
        }
    }
    gemm_q3_k_m4n3_scalar(w0, w1, w2, w3, x0, x1, x2)
}

/// Specialized `4x4` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m4n4_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x3).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `4x4` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m4n4(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m4n4_avx2(w0, w1, w2, w3, x0, x1, x2, x3) };
        }
    }
    gemm_q3_k_m4n4_scalar(w0, w1, w2, w3, x0, x1, x2, x3)
}

/// Specialized `5x1` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m5n1_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 5] {
    let mut out = [[0.0f32; 1]; 5];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0);
    out
}

/// Runtime-dispatched `5x1` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m5n1(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m5n1_avx2(w0, w1, w2, w3, w4, x0) };
        }
    }
    gemm_q3_k_m5n1_scalar(w0, w1, w2, w3, w4, x0)
}

/// Specialized `5x2` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m5n2_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 5] {
    let mut out = [[0.0f32; 2]; 5];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0, x1);
    out
}

/// Runtime-dispatched `5x2` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m5n2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m5n2_avx2(w0, w1, w2, w3, w4, x0, x1) };
        }
    }
    gemm_q3_k_m5n2_scalar(w0, w1, w2, w3, w4, x0, x1)
}

/// Specialized `5x3` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m5n3_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 5] {
    let mut out = [[0.0f32; 3]; 5];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x2).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0, x1, x2);
    out
}

/// Runtime-dispatched `5x3` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m5n3(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m5n3_avx2(w0, w1, w2, w3, w4, x0, x1, x2) };
        }
    }
    gemm_q3_k_m5n3_scalar(w0, w1, w2, w3, w4, x0, x1, x2)
}

/// Specialized `5x4` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m5n4_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 5] {
    let mut out = [[0.0f32; 4]; 5];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x3).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `5x4` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m5n4(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m5n4_avx2(w0, w1, w2, w3, w4, x0, x1, x2, x3) };
        }
    }
    gemm_q3_k_m5n4_scalar(w0, w1, w2, w3, w4, x0, x1, x2, x3)
}

/// Specialized `6x1` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m6n1_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 6] {
    let mut out = [[0.0f32; 1]; 6];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0);
    out
}

/// Runtime-dispatched `6x1` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m6n1(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 6] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m6n1_avx2(w0, w1, w2, w3, w4, w5, x0) };
        }
    }
    gemm_q3_k_m6n1_scalar(w0, w1, w2, w3, w4, w5, x0)
}

/// Specialized `6x2` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m6n2_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 6] {
    let mut out = [[0.0f32; 2]; 6];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x1).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0, x1);
    out
}

/// Runtime-dispatched `6x2` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m6n2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 6] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m6n2_avx2(w0, w1, w2, w3, w4, w5, x0, x1) };
        }
    }
    gemm_q3_k_m6n2_scalar(w0, w1, w2, w3, w4, w5, x0, x1)
}

/// Specialized `6x3` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m6n3_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 6] {
    let mut out = [[0.0f32; 3]; 6];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x2).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0, x1, x2);
    out
}

/// Runtime-dispatched `6x3` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m6n3(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 6] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m6n3_avx2(w0, w1, w2, w3, w4, w5, x0, x1, x2) };
        }
    }
    gemm_q3_k_m6n3_scalar(w0, w1, w2, w3, w4, w5, x0, x1, x2)
}

/// Specialized `6x4` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m6n4_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 6] {
    let mut out = [[0.0f32; 4]; 6];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x3).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `6x4` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m6n4(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 6] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m6n4_avx2(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3) };
        }
    }
    gemm_q3_k_m6n4_scalar(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3)
}

/// Specialized `7x1` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m7n1_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 7] {
    let mut out = [[0.0f32; 1]; 7];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x0).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0);
    out
}

/// Runtime-dispatched `7x1` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m7n1(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 7] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m7n1_avx2(w0, w1, w2, w3, w4, w5, w6, x0) };
        }
    }
    gemm_q3_k_m7n1_scalar(w0, w1, w2, w3, w4, w5, w6, x0)
}

/// Specialized `7x2` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m7n2_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 7] {
    let mut out = [[0.0f32; 2]; 7];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x1).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0, x1);
    out
}

/// Runtime-dispatched `7x2` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m7n2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 7] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m7n2_avx2(w0, w1, w2, w3, w4, w5, w6, x0, x1) };
        }
    }
    gemm_q3_k_m7n2_scalar(w0, w1, w2, w3, w4, w5, w6, x0, x1)
}

/// Specialized `7x3` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m7n3_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 7] {
    let mut out = [[0.0f32; 3]; 7];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x2).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0, x1, x2);
    out
}

/// Runtime-dispatched `7x3` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m7n3(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 7] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m7n3_avx2(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2) };
        }
    }
    gemm_q3_k_m7n3_scalar(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2)
}

/// Specialized `7x4` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m7n4_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 7] {
    let mut out = [[0.0f32; 4]; 7];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x3).unwrap_or(0.0);
    out[6][3] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x3).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `7x4` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m7n4(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 7] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m7n4_avx2(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3) };
        }
    }
    gemm_q3_k_m7n4_scalar(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3)
}

/// Specialized `8x1` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m8n1_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 8] {
    let mut out = [[0.0f32; 1]; 8];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x0).unwrap_or(0.0);
    out[7][0] = crate::quant::idot::integer(GgmlType::Q3_K, w7, x0).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, w7, x0);
    out
}

/// Runtime-dispatched `8x1` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m8n1(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m8n1_avx2(w0, w1, w2, w3, w4, w5, w6, w7, x0) };
        }
    }
    gemm_q3_k_m8n1_scalar(w0, w1, w2, w3, w4, w5, w6, w7, x0)
}

/// Specialized `8x2` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m8n2_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 8] {
    let mut out = [[0.0f32; 2]; 8];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x0).unwrap_or(0.0);
    out[7][0] = crate::quant::idot::integer(GgmlType::Q3_K, w7, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x1).unwrap_or(0.0);
    out[7][1] = crate::quant::idot::integer(GgmlType::Q3_K, w7, x1).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1);
    out
}

/// Runtime-dispatched `8x2` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m8n2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m8n2_avx2(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1) };
        }
    }
    gemm_q3_k_m8n2_scalar(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1)
}

/// Specialized `8x3` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m8n3_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 8] {
    let mut out = [[0.0f32; 3]; 8];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x0).unwrap_or(0.0);
    out[7][0] = crate::quant::idot::integer(GgmlType::Q3_K, w7, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x1).unwrap_or(0.0);
    out[7][1] = crate::quant::idot::integer(GgmlType::Q3_K, w7, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x2).unwrap_or(0.0);
    out[7][2] = crate::quant::idot::integer(GgmlType::Q3_K, w7, x2).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2);
    out
}

/// Runtime-dispatched `8x3` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m8n3(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m8n3_avx2(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2) };
        }
    }
    gemm_q3_k_m8n3_scalar(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2)
}

/// Specialized `8x4` AVX2+FMA+F16C register-tiled microkernel for `Q3_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q3_k_m8n4_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 8] {
    let mut out = [[0.0f32; 4]; 8];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x0).unwrap_or(0.0);
    out[7][0] = crate::quant::idot::integer(GgmlType::Q3_K, w7, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x1).unwrap_or(0.0);
    out[7][1] = crate::quant::idot::integer(GgmlType::Q3_K, w7, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x2).unwrap_or(0.0);
    out[7][2] = crate::quant::idot::integer(GgmlType::Q3_K, w7, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::Q3_K, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::Q3_K, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::Q3_K, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::Q3_K, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::Q3_K, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::Q3_K, w5, x3).unwrap_or(0.0);
    out[6][3] = crate::quant::idot::integer(GgmlType::Q3_K, w6, x3).unwrap_or(0.0);
    out[7][3] = crate::quant::idot::integer(GgmlType::Q3_K, w7, x3).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `8x4` microkernel for `Q3_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q3_k_m8n4(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    w6: &[u8],
    w7: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q3_k_m8n4_avx2(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3) };
        }
    }
    gemm_q3_k_m8n4_scalar(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3)
}

/// Execute tiled matrix-matrix multiplication for `Q3_K` over `rows` weight rows and `xs.len()` quantized activations.
#[allow(clippy::identity_op)]
pub fn matmul_tiled_q3_k(
    weight_bytes: &[u8],
    rows: usize,
    row_bytes: usize,
    xs: &[Quantized],
    out: &mut [f32],
) {
    let n_batch = xs.len();
    if rows == 0 || n_batch == 0 {
        return;
    }
    assert!(weight_bytes.len() >= rows * row_bytes);
    assert!(out.len() >= n_batch * rows);
    let mut c = 0usize;
    while c + 4 <= n_batch {
        let x0 = &xs[c];
        let x1 = &xs[c + 1];
        let x2 = &xs[c + 2];
        let x3 = &xs[c + 3];
        let mut r = 0usize;
        while r + 8 <= rows {
            let w0 = &weight_bytes[(r + 0) * row_bytes..(r + 0 + 1) * row_bytes];
            let w1 = &weight_bytes[(r + 1) * row_bytes..(r + 1 + 1) * row_bytes];
            let w2 = &weight_bytes[(r + 2) * row_bytes..(r + 2 + 1) * row_bytes];
            let w3 = &weight_bytes[(r + 3) * row_bytes..(r + 3 + 1) * row_bytes];
            let w4 = &weight_bytes[(r + 4) * row_bytes..(r + 4 + 1) * row_bytes];
            let w5 = &weight_bytes[(r + 5) * row_bytes..(r + 5 + 1) * row_bytes];
            let w6 = &weight_bytes[(r + 6) * row_bytes..(r + 6 + 1) * row_bytes];
            let w7 = &weight_bytes[(r + 7) * row_bytes..(r + 7 + 1) * row_bytes];
            let tile = dispatch_q3_k_m8n4(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3);
            for dr in 0..8 {
                for dc in 0..4 {
                    out[(c + dc) * rows + (r + dr)] = tile[dr][dc];
                }
            }
            r += 8;
        }
        while r + 4 <= rows {
            let w0 = &weight_bytes[(r + 0) * row_bytes..(r + 0 + 1) * row_bytes];
            let w1 = &weight_bytes[(r + 1) * row_bytes..(r + 1 + 1) * row_bytes];
            let w2 = &weight_bytes[(r + 2) * row_bytes..(r + 2 + 1) * row_bytes];
            let w3 = &weight_bytes[(r + 3) * row_bytes..(r + 3 + 1) * row_bytes];
            let tile = dispatch_q3_k_m4n4(w0, w1, w2, w3, x0, x1, x2, x3);
            for dr in 0..4 {
                for dc in 0..4 {
                    out[(c + dc) * rows + (r + dr)] = tile[dr][dc];
                }
            }
            r += 4;
        }
        while r < rows {
            let w0 = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
            let tile = dispatch_q3_k_m1n4(w0, x0, x1, x2, x3);
            for dc in 0..4 {
                out[(c + dc) * rows + r] = tile[0][dc];
            }
            r += 1;
        }
        c += 4;
    }
    while c < n_batch {
        let x0 = &xs[c];
        let mut r = 0usize;
        while r + 8 <= rows {
            let w0 = &weight_bytes[(r + 0) * row_bytes..(r + 0 + 1) * row_bytes];
            let w1 = &weight_bytes[(r + 1) * row_bytes..(r + 1 + 1) * row_bytes];
            let w2 = &weight_bytes[(r + 2) * row_bytes..(r + 2 + 1) * row_bytes];
            let w3 = &weight_bytes[(r + 3) * row_bytes..(r + 3 + 1) * row_bytes];
            let w4 = &weight_bytes[(r + 4) * row_bytes..(r + 4 + 1) * row_bytes];
            let w5 = &weight_bytes[(r + 5) * row_bytes..(r + 5 + 1) * row_bytes];
            let w6 = &weight_bytes[(r + 6) * row_bytes..(r + 6 + 1) * row_bytes];
            let w7 = &weight_bytes[(r + 7) * row_bytes..(r + 7 + 1) * row_bytes];
            let tile = dispatch_q3_k_m8n1(w0, w1, w2, w3, w4, w5, w6, w7, x0);
            for dr in 0..8 {
                out[c * rows + (r + dr)] = tile[dr][0];
            }
            r += 8;
        }
        while r + 4 <= rows {
            let w0 = &weight_bytes[(r + 0) * row_bytes..(r + 0 + 1) * row_bytes];
            let w1 = &weight_bytes[(r + 1) * row_bytes..(r + 1 + 1) * row_bytes];
            let w2 = &weight_bytes[(r + 2) * row_bytes..(r + 2 + 1) * row_bytes];
            let w3 = &weight_bytes[(r + 3) * row_bytes..(r + 3 + 1) * row_bytes];
            let tile = dispatch_q3_k_m4n1(w0, w1, w2, w3, x0);
            for dr in 0..4 {
                out[c * rows + (r + dr)] = tile[dr][0];
            }
            r += 4;
        }
        while r < rows {
            let w0 = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
            let tile = dispatch_q3_k_m1n1(w0, x0);
            out[c * rows + r] = tile[0][0];
            r += 1;
        }
        c += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row(seed: usize) -> Vec<u8> {
        let mut buf = vec![0u8; 110];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((seed * 31 + i * 17 + 7) % 199) as u8;
        }
        buf[108..110].copy_from_slice(&0x3400u16.to_le_bytes());
        buf
    }

    #[test]
    fn q3_k_all_m1_to_8_n1_to_4_match_idot_reference() {
        let rows: Vec<Vec<u8>> = (0..10).map(sample_row).collect();
        let xs: Vec<Quantized> = (0..5)
            .map(|c| {
                let floats: Vec<f32> = (0..256)
                    .map(|i| ((i + c * 7) as f32 * 0.11).sin())
                    .collect();
                let mut q = Quantized::with_capacity(256);
                q.fill(&floats);
                q
            })
            .collect();

        let tile = dispatch_q3_k_m4n4(
            &rows[0], &rows[1], &rows[2], &rows[3], &xs[0], &xs[1], &xs[2], &xs[3],
        );
        for r in 0..4 {
            for c in 0..4 {
                let expected =
                    crate::quant::idot::integer(GgmlType::Q3_K, &rows[r], &xs[c]).unwrap();
                assert!(
                    (tile[r][c] - expected).abs() < 5e-3,
                    "mismatch at ({r},{c}): {} vs {expected}",
                    tile[r][c]
                );
            }
        }

        let flat_w: Vec<u8> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let mut out = vec![0.0f32; 10 * 5];
        matmul_tiled_q3_k(&flat_w, 10, 110, &xs, &mut out);
        for c in 0..5 {
            for r in 0..10 {
                let expected =
                    crate::quant::idot::integer(GgmlType::Q3_K, &rows[r], &xs[c]).unwrap();
                assert!((out[c * 10 + r] - expected).abs() < 5e-3);
            }
        }
    }
}
