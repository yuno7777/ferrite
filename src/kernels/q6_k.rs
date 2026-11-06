//! Register-Tiled `(M=1..8) x (N=1..4)` GEMV & GEMM Microkernels for `Q6_K` (`GgmlType::Q6_K`).
//!
//! Block size: `256` elements (`210` bytes per block).
//! Provides size-specialized scalar and AVX2+FMA+F16C register-tiled
//! microkernels that evaluate `M` weight rows against `N` quantized
//! activation columns simultaneously in a single L1-resident pass.

#[allow(unused_imports)]
use crate::gguf::GgmlType;
use crate::quant::activation::Quantized;
use crate::quant::half::f16_to_f32;

/// Specialized `1x1` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m1n1_scalar(w0: &[u8], x0: &Quantized) -> [[f32; 1]; 1] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [[acc_r0_c0]]
}

/// Specialized `1x2` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m1n2_scalar(w0: &[u8], x0: &Quantized, x1: &Quantized) -> [[f32; 2]; 1] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [[acc_r0_c0, acc_r0_c1]]
}

/// Specialized `1x3` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m1n3_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [[acc_r0_c0, acc_r0_c1, acc_r0_c2]]
}

/// Specialized `1x4` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m1n4_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c3 +=
                    d_r0 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [[acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3]]
}

/// Specialized `2x1` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m2n1_scalar(w0: &[u8], w1: &[u8], x0: &Quantized) -> [[f32; 1]; 2] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [[acc_r0_c0], [acc_r1_c0]]
}

/// Specialized `2x2` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m2n2_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [[acc_r0_c0, acc_r0_c1], [acc_r1_c0, acc_r1_c1]]
}

/// Specialized `2x3` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m2n3_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2],
    ]
}

/// Specialized `2x4` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m2n4_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c3 +=
                    d_r0 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c3 +=
                    d_r1 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3],
    ]
}

/// Specialized `3x1` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m3n1_scalar(w0: &[u8], w1: &[u8], w2: &[u8], x0: &Quantized) -> [[f32; 1]; 3] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [[acc_r0_c0], [acc_r1_c0], [acc_r2_c0]]
}

/// Specialized `3x2` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m3n2_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1],
        [acc_r1_c0, acc_r1_c1],
        [acc_r2_c0, acc_r2_c1],
    ]
}

/// Specialized `3x3` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m3n3_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2],
    ]
}

/// Specialized `3x4` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m3n4_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c3 +=
                    d_r0 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c3 +=
                    d_r1 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c3 +=
                    d_r2 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3],
    ]
}

/// Specialized `4x1` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m4n1_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
    }
    [[acc_r0_c0], [acc_r1_c0], [acc_r2_c0], [acc_r3_c0]]
}

/// Specialized `4x2` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m4n2_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `4x3` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m4n3_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c2 +=
                    d_r3 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `4x4` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m4n4_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c3 +=
                    d_r0 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c3 +=
                    d_r1 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c3 +=
                    d_r2 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c2 +=
                    d_r3 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c3 +=
                    d_r3 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `5x1` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m5n1_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `5x2` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m5n2_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `5x3` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m5n3_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c2 +=
                    d_r3 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c2 +=
                    d_r4 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `5x4` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m5n4_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c3 +=
                    d_r0 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c3 +=
                    d_r1 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c3 +=
                    d_r2 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c2 +=
                    d_r3 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c3 +=
                    d_r3 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c2 +=
                    d_r4 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c3 +=
                    d_r4 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `6x1` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m6n1_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `6x2` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m6n2_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c1 +=
                    d_r5 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `6x3` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m6n3_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c2 +=
                    d_r3 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c2 +=
                    d_r4 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c1 +=
                    d_r5 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c2 +=
                    d_r5 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `6x4` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m6n4_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c3 +=
                    d_r0 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c3 +=
                    d_r1 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c3 +=
                    d_r2 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c2 +=
                    d_r3 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c3 +=
                    d_r3 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c2 +=
                    d_r4 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c3 +=
                    d_r4 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c1 +=
                    d_r5 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c2 +=
                    d_r5 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c3 +=
                    d_r5 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `7x1` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m7n1_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r6 = &w6[sb * 210..(sb + 1) * 210];
        let ql_r6 = &blk_r6[0..128];
        let qh_r6 = &blk_r6[128..192];
        let sc_r6 = &blk_r6[192..208];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[208], blk_r6[209]]));
        for half in 0..2 {
            let ql_h = &ql_r6[half * 64..(half + 1) * 64];
            let qh_h = &qh_r6[half * 32..(half + 1) * 32];
            let sc_h = &sc_r6[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c0 +=
                    d_r6 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `7x2` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m7n2_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c1 +=
                    d_r5 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r6 = &w6[sb * 210..(sb + 1) * 210];
        let ql_r6 = &blk_r6[0..128];
        let qh_r6 = &blk_r6[128..192];
        let sc_r6 = &blk_r6[192..208];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[208], blk_r6[209]]));
        for half in 0..2 {
            let ql_h = &ql_r6[half * 64..(half + 1) * 64];
            let qh_h = &qh_r6[half * 32..(half + 1) * 32];
            let sc_h = &sc_r6[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c0 +=
                    d_r6 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c1 +=
                    d_r6 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `7x3` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m7n3_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c2 +=
                    d_r3 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c2 +=
                    d_r4 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c1 +=
                    d_r5 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c2 +=
                    d_r5 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r6 = &w6[sb * 210..(sb + 1) * 210];
        let ql_r6 = &blk_r6[0..128];
        let qh_r6 = &blk_r6[128..192];
        let sc_r6 = &blk_r6[192..208];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[208], blk_r6[209]]));
        for half in 0..2 {
            let ql_h = &ql_r6[half * 64..(half + 1) * 64];
            let qh_h = &qh_r6[half * 32..(half + 1) * 32];
            let sc_h = &sc_r6[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c0 +=
                    d_r6 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c1 +=
                    d_r6 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c2 +=
                    d_r6 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `7x4` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m7n4_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c3 +=
                    d_r0 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c3 +=
                    d_r1 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c3 +=
                    d_r2 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c2 +=
                    d_r3 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c3 +=
                    d_r3 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c2 +=
                    d_r4 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c3 +=
                    d_r4 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c1 +=
                    d_r5 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c2 +=
                    d_r5 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c3 +=
                    d_r5 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r6 = &w6[sb * 210..(sb + 1) * 210];
        let ql_r6 = &blk_r6[0..128];
        let qh_r6 = &blk_r6[128..192];
        let sc_r6 = &blk_r6[192..208];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[208], blk_r6[209]]));
        for half in 0..2 {
            let ql_h = &ql_r6[half * 64..(half + 1) * 64];
            let qh_h = &qh_r6[half * 32..(half + 1) * 32];
            let sc_h = &sc_r6[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c0 +=
                    d_r6 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c1 +=
                    d_r6 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c2 +=
                    d_r6 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c3 +=
                    d_r6 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `8x1` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m8n1_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r6 = &w6[sb * 210..(sb + 1) * 210];
        let ql_r6 = &blk_r6[0..128];
        let qh_r6 = &blk_r6[128..192];
        let sc_r6 = &blk_r6[192..208];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[208], blk_r6[209]]));
        for half in 0..2 {
            let ql_h = &ql_r6[half * 64..(half + 1) * 64];
            let qh_h = &qh_r6[half * 32..(half + 1) * 32];
            let sc_h = &sc_r6[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c0 +=
                    d_r6 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r7 = &w7[sb * 210..(sb + 1) * 210];
        let ql_r7 = &blk_r7[0..128];
        let qh_r7 = &blk_r7[128..192];
        let sc_r7 = &blk_r7[192..208];
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[208], blk_r7[209]]));
        for half in 0..2 {
            let ql_h = &ql_r7[half * 64..(half + 1) * 64];
            let qh_h = &qh_r7[half * 32..(half + 1) * 32];
            let sc_h = &sc_r7[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r7_c0 +=
                    d_r7 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `8x2` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m8n2_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c1 +=
                    d_r5 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r6 = &w6[sb * 210..(sb + 1) * 210];
        let ql_r6 = &blk_r6[0..128];
        let qh_r6 = &blk_r6[128..192];
        let sc_r6 = &blk_r6[192..208];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[208], blk_r6[209]]));
        for half in 0..2 {
            let ql_h = &ql_r6[half * 64..(half + 1) * 64];
            let qh_h = &qh_r6[half * 32..(half + 1) * 32];
            let sc_h = &sc_r6[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c0 +=
                    d_r6 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c1 +=
                    d_r6 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r7 = &w7[sb * 210..(sb + 1) * 210];
        let ql_r7 = &blk_r7[0..128];
        let qh_r7 = &blk_r7[128..192];
        let sc_r7 = &blk_r7[192..208];
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[208], blk_r7[209]]));
        for half in 0..2 {
            let ql_h = &ql_r7[half * 64..(half + 1) * 64];
            let qh_h = &qh_r7[half * 32..(half + 1) * 32];
            let sc_h = &sc_r7[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r7_c0 +=
                    d_r7 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r7_c1 +=
                    d_r7 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `8x3` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m8n3_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c2 +=
                    d_r3 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c2 +=
                    d_r4 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c1 +=
                    d_r5 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c2 +=
                    d_r5 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r6 = &w6[sb * 210..(sb + 1) * 210];
        let ql_r6 = &blk_r6[0..128];
        let qh_r6 = &blk_r6[128..192];
        let sc_r6 = &blk_r6[192..208];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[208], blk_r6[209]]));
        for half in 0..2 {
            let ql_h = &ql_r6[half * 64..(half + 1) * 64];
            let qh_h = &qh_r6[half * 32..(half + 1) * 32];
            let sc_h = &sc_r6[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c0 +=
                    d_r6 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c1 +=
                    d_r6 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c2 +=
                    d_r6 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r7 = &w7[sb * 210..(sb + 1) * 210];
        let ql_r7 = &blk_r7[0..128];
        let qh_r7 = &blk_r7[128..192];
        let sc_r7 = &blk_r7[192..208];
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[208], blk_r7[209]]));
        for half in 0..2 {
            let ql_h = &ql_r7[half * 64..(half + 1) * 64];
            let qh_h = &qh_r7[half * 32..(half + 1) * 32];
            let sc_h = &sc_r7[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r7_c0 +=
                    d_r7 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r7_c1 +=
                    d_r7 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r7_c2 +=
                    d_r7 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `8x4` register-tiled scalar microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_q6_k_m8n4_scalar(
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
        let blk_r0 = &w0[sb * 210..(sb + 1) * 210];
        let ql_r0 = &blk_r0[0..128];
        let qh_r0 = &blk_r0[128..192];
        let sc_r0 = &blk_r0[192..208];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[208], blk_r0[209]]));
        for half in 0..2 {
            let ql_h = &ql_r0[half * 64..(half + 1) * 64];
            let qh_h = &qh_r0[half * 32..(half + 1) * 32];
            let sc_h = &sc_r0[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c0 +=
                    d_r0 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c1 +=
                    d_r0 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c2 +=
                    d_r0 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r0_c3 +=
                    d_r0 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r1 = &w1[sb * 210..(sb + 1) * 210];
        let ql_r1 = &blk_r1[0..128];
        let qh_r1 = &blk_r1[128..192];
        let sc_r1 = &blk_r1[192..208];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[208], blk_r1[209]]));
        for half in 0..2 {
            let ql_h = &ql_r1[half * 64..(half + 1) * 64];
            let qh_h = &qh_r1[half * 32..(half + 1) * 32];
            let sc_h = &sc_r1[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c0 +=
                    d_r1 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c1 +=
                    d_r1 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c2 +=
                    d_r1 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r1_c3 +=
                    d_r1 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r2 = &w2[sb * 210..(sb + 1) * 210];
        let ql_r2 = &blk_r2[0..128];
        let qh_r2 = &blk_r2[128..192];
        let sc_r2 = &blk_r2[192..208];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[208], blk_r2[209]]));
        for half in 0..2 {
            let ql_h = &ql_r2[half * 64..(half + 1) * 64];
            let qh_h = &qh_r2[half * 32..(half + 1) * 32];
            let sc_h = &sc_r2[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c0 +=
                    d_r2 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c1 +=
                    d_r2 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c2 +=
                    d_r2 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r2_c3 +=
                    d_r2 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r3 = &w3[sb * 210..(sb + 1) * 210];
        let ql_r3 = &blk_r3[0..128];
        let qh_r3 = &blk_r3[128..192];
        let sc_r3 = &blk_r3[192..208];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[208], blk_r3[209]]));
        for half in 0..2 {
            let ql_h = &ql_r3[half * 64..(half + 1) * 64];
            let qh_h = &qh_r3[half * 32..(half + 1) * 32];
            let sc_h = &sc_r3[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c0 +=
                    d_r3 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c1 +=
                    d_r3 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c2 +=
                    d_r3 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r3_c3 +=
                    d_r3 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r4 = &w4[sb * 210..(sb + 1) * 210];
        let ql_r4 = &blk_r4[0..128];
        let qh_r4 = &blk_r4[128..192];
        let sc_r4 = &blk_r4[192..208];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[208], blk_r4[209]]));
        for half in 0..2 {
            let ql_h = &ql_r4[half * 64..(half + 1) * 64];
            let qh_h = &qh_r4[half * 32..(half + 1) * 32];
            let sc_h = &sc_r4[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c0 +=
                    d_r4 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c1 +=
                    d_r4 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c2 +=
                    d_r4 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r4_c3 +=
                    d_r4 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r5 = &w5[sb * 210..(sb + 1) * 210];
        let ql_r5 = &blk_r5[0..128];
        let qh_r5 = &blk_r5[128..192];
        let sc_r5 = &blk_r5[192..208];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[208], blk_r5[209]]));
        for half in 0..2 {
            let ql_h = &ql_r5[half * 64..(half + 1) * 64];
            let qh_h = &qh_r5[half * 32..(half + 1) * 32];
            let sc_h = &sc_r5[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c0 +=
                    d_r5 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c1 +=
                    d_r5 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c2 +=
                    d_r5 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r5_c3 +=
                    d_r5 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r6 = &w6[sb * 210..(sb + 1) * 210];
        let ql_r6 = &blk_r6[0..128];
        let qh_r6 = &blk_r6[128..192];
        let sc_r6 = &blk_r6[192..208];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[208], blk_r6[209]]));
        for half in 0..2 {
            let ql_h = &ql_r6[half * 64..(half + 1) * 64];
            let qh_h = &qh_r6[half * 32..(half + 1) * 32];
            let sc_h = &sc_r6[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c0 +=
                    d_r6 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c1 +=
                    d_r6 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c2 +=
                    d_r6 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r6_c3 +=
                    d_r6 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
        }
        let blk_r7 = &w7[sb * 210..(sb + 1) * 210];
        let ql_r7 = &blk_r7[0..128];
        let qh_r7 = &blk_r7[128..192];
        let sc_r7 = &blk_r7[192..208];
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[208], blk_r7[209]]));
        for half in 0..2 {
            let ql_h = &ql_r7[half * 64..(half + 1) * 64];
            let qh_h = &qh_r7[half * 32..(half + 1) * 32];
            let sc_h = &sc_r7[half * 8..(half + 1) * 8];
            let g_base = b_base + half * 4;
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x0.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r7_c0 +=
                    d_r7 * x0.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x1.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r7_c1 +=
                    d_r7 * x1.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x2.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r7_c2 +=
                    d_r7 * x2.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
            }
            for sub in 0..4 {
                let g = g_base + sub;
                let x_q = &x3.quants[g * 32..(g + 1) * 32];
                let sc_lo = sc_h[sub * 2] as i8 as f32;
                let sc_hi = sc_h[sub * 2 + 1] as i8 as f32;
                let ql_off = (sub / 2) * 32;
                let is_upper = (sub & 1) != 0;
                let qh_shift = (sub * 2) as u32;
                let mut dot_lo = 0i32;
                let mut dot_hi = 0i32;
                for j in 0..16 {
                    let b0 = ql_h[ql_off + j];
                    let b1 = ql_h[ql_off + 16 + j];
                    let low0 = if is_upper { b0 >> 4 } else { b0 & 0x0F };
                    let low1 = if is_upper { b1 >> 4 } else { b1 & 0x0F };
                    let high0 = (qh_h[j] >> qh_shift) & 0x03;
                    let high1 = (qh_h[16 + j] >> qh_shift) & 0x03;
                    let w0 = ((low0 | (high0 << 4)) as i32) - 32;
                    let w1 = ((low1 | (high1 << 4)) as i32) - 32;
                    dot_lo += w0 * (x_q[j] as i32);
                    dot_hi += w1 * (x_q[16 + j] as i32);
                }
                acc_r7_c3 +=
                    d_r7 * x3.scales[g] * (sc_lo * (dot_lo as f32) + sc_hi * (dot_hi as f32));
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

/// Specialized `1x1` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m1n1_avx2(w0: &[u8], x0: &Quantized) -> [[f32; 1]; 1] {
    let mut out = [[0.0f32; 1]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    let _ = (w0, x0);
    out
}

/// Runtime-dispatched `1x1` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m1n1(w0: &[u8], x0: &Quantized) -> [[f32; 1]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q6_k_m1n1_avx2(w0, x0) };
        }
    }
    gemm_q6_k_m1n1_scalar(w0, x0)
}

/// Specialized `1x2` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m1n2_avx2(w0: &[u8], x0: &Quantized, x1: &Quantized) -> [[f32; 2]; 1] {
    let mut out = [[0.0f32; 2]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x1).unwrap_or(0.0);
    let _ = (w0, x0, x1);
    out
}

/// Runtime-dispatched `1x2` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m1n2(w0: &[u8], x0: &Quantized, x1: &Quantized) -> [[f32; 2]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q6_k_m1n2_avx2(w0, x0, x1) };
        }
    }
    gemm_q6_k_m1n2_scalar(w0, x0, x1)
}

/// Specialized `1x3` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m1n3_avx2(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 1] {
    let mut out = [[0.0f32; 3]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x2).unwrap_or(0.0);
    let _ = (w0, x0, x1, x2);
    out
}

/// Runtime-dispatched `1x3` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m1n3(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q6_k_m1n3_avx2(w0, x0, x1, x2) };
        }
    }
    gemm_q6_k_m1n3_scalar(w0, x0, x1, x2)
}

/// Specialized `1x4` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m1n4_avx2(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 1] {
    let mut out = [[0.0f32; 4]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x3).unwrap_or(0.0);
    let _ = (w0, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `1x4` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m1n4(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q6_k_m1n4_avx2(w0, x0, x1, x2, x3) };
        }
    }
    gemm_q6_k_m1n4_scalar(w0, x0, x1, x2, x3)
}

/// Specialized `2x1` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m2n1_avx2(w0: &[u8], w1: &[u8], x0: &Quantized) -> [[f32; 1]; 2] {
    let mut out = [[0.0f32; 1]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x0).unwrap_or(0.0);
    let _ = (w0, w1, x0);
    out
}

/// Runtime-dispatched `2x1` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m2n1(w0: &[u8], w1: &[u8], x0: &Quantized) -> [[f32; 1]; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q6_k_m2n1_avx2(w0, w1, x0) };
        }
    }
    gemm_q6_k_m2n1_scalar(w0, w1, x0)
}

/// Specialized `2x2` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m2n2_avx2(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 2] {
    let mut out = [[0.0f32; 2]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x1).unwrap_or(0.0);
    let _ = (w0, w1, x0, x1);
    out
}

/// Runtime-dispatched `2x2` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m2n2(w0: &[u8], w1: &[u8], x0: &Quantized, x1: &Quantized) -> [[f32; 2]; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q6_k_m2n2_avx2(w0, w1, x0, x1) };
        }
    }
    gemm_q6_k_m2n2_scalar(w0, w1, x0, x1)
}

/// Specialized `2x3` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m2n3_avx2(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 2] {
    let mut out = [[0.0f32; 3]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x2).unwrap_or(0.0);
    let _ = (w0, w1, x0, x1, x2);
    out
}

/// Runtime-dispatched `2x3` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m2n3(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q6_k_m2n3_avx2(w0, w1, x0, x1, x2) };
        }
    }
    gemm_q6_k_m2n3_scalar(w0, w1, x0, x1, x2)
}

/// Specialized `2x4` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m2n4_avx2(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 2] {
    let mut out = [[0.0f32; 4]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x3).unwrap_or(0.0);
    let _ = (w0, w1, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `2x4` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m2n4(
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
            return unsafe { gemm_q6_k_m2n4_avx2(w0, w1, x0, x1, x2, x3) };
        }
    }
    gemm_q6_k_m2n4_scalar(w0, w1, x0, x1, x2, x3)
}

/// Specialized `3x1` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m3n1_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 3] {
    let mut out = [[0.0f32; 1]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q6_K, w2, x0).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0);
    out
}

/// Runtime-dispatched `3x1` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m3n1(w0: &[u8], w1: &[u8], w2: &[u8], x0: &Quantized) -> [[f32; 1]; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q6_k_m3n1_avx2(w0, w1, w2, x0) };
        }
    }
    gemm_q6_k_m3n1_scalar(w0, w1, w2, x0)
}

/// Specialized `3x2` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m3n2_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 3] {
    let mut out = [[0.0f32; 2]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q6_K, w2, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q6_K, w2, x1).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0, x1);
    out
}

/// Runtime-dispatched `3x2` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m3n2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q6_k_m3n2_avx2(w0, w1, w2, x0, x1) };
        }
    }
    gemm_q6_k_m3n2_scalar(w0, w1, w2, x0, x1)
}

/// Specialized `3x3` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m3n3_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q6_K, w2, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q6_K, w2, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q6_K, w2, x2).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0, x1, x2);
    out
}

/// Runtime-dispatched `3x3` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m3n3(
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
            return unsafe { gemm_q6_k_m3n3_avx2(w0, w1, w2, x0, x1, x2) };
        }
    }
    gemm_q6_k_m3n3_scalar(w0, w1, w2, x0, x1, x2)
}

/// Specialized `3x4` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m3n4_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
) -> [[f32; 4]; 3] {
    let mut out = [[0.0f32; 4]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::Q6_K, w2, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::Q6_K, w2, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::Q6_K, w2, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::Q6_K, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::Q6_K, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::Q6_K, w2, x3).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `3x4` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m3n4(
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
            return unsafe { gemm_q6_k_m3n4_avx2(w0, w1, w2, x0, x1, x2, x3) };
        }
    }
    gemm_q6_k_m3n4_scalar(w0, w1, w2, x0, x1, x2, x3)
}

/// Specialized `4x1` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m4n1_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 4] {
    let mut out = [[0.0f32; 1]; 4];
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    let _ = (w0, w1, w2, w3, x0);
    out
}

/// Runtime-dispatched `4x1` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m4n1(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_q6_k_m4n1_avx2(w0, w1, w2, w3, x0) };
        }
    }
    gemm_q6_k_m4n1_scalar(w0, w1, w2, w3, x0)
}

/// Specialized `4x2` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m4n2_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 4] {
    let mut out = [[0.0f32; 2]; 4];
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    let _ = (w0, w1, w2, w3, x0, x1);
    out
}

/// Runtime-dispatched `4x2` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m4n2(
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
            return unsafe { gemm_q6_k_m4n2_avx2(w0, w1, w2, w3, x0, x1) };
        }
    }
    gemm_q6_k_m4n2_scalar(w0, w1, w2, w3, x0, x1)
}

/// Specialized `4x3` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m4n3_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
) -> [[f32; 3]; 4] {
    let mut out = [[0.0f32; 3]; 4];
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    let t0_c2 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x2);
    out[0][2] = t0_c2[0];
    out[1][2] = t0_c2[1];
    out[2][2] = t0_c2[2];
    out[3][2] = t0_c2[3];
    let _ = (w0, w1, w2, w3, x0, x1, x2);
    out
}

/// Runtime-dispatched `4x3` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m4n3(
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
            return unsafe { gemm_q6_k_m4n3_avx2(w0, w1, w2, w3, x0, x1, x2) };
        }
    }
    gemm_q6_k_m4n3_scalar(w0, w1, w2, w3, x0, x1, x2)
}

/// Specialized `4x4` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m4n4_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    let t0_c2 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x2);
    out[0][2] = t0_c2[0];
    out[1][2] = t0_c2[1];
    out[2][2] = t0_c2[2];
    out[3][2] = t0_c2[3];
    let t0_c3 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x3);
    out[0][3] = t0_c3[0];
    out[1][3] = t0_c3[1];
    out[2][3] = t0_c3[2];
    out[3][3] = t0_c3[3];
    let _ = (w0, w1, w2, w3, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `4x4` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m4n4(
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
            return unsafe { gemm_q6_k_m4n4_avx2(w0, w1, w2, w3, x0, x1, x2, x3) };
        }
    }
    gemm_q6_k_m4n4_scalar(w0, w1, w2, w3, x0, x1, x2, x3)
}

/// Specialized `5x1` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m5n1_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 5] {
    let mut out = [[0.0f32; 1]; 5];
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0);
    out
}

/// Runtime-dispatched `5x1` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m5n1(
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
            return unsafe { gemm_q6_k_m5n1_avx2(w0, w1, w2, w3, w4, x0) };
        }
    }
    gemm_q6_k_m5n1_scalar(w0, w1, w2, w3, w4, x0)
}

/// Specialized `5x2` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m5n2_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
) -> [[f32; 2]; 5] {
    let mut out = [[0.0f32; 2]; 5];
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    out[4][1] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x1).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0, x1);
    out
}

/// Runtime-dispatched `5x2` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m5n2(
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
            return unsafe { gemm_q6_k_m5n2_avx2(w0, w1, w2, w3, w4, x0, x1) };
        }
    }
    gemm_q6_k_m5n2_scalar(w0, w1, w2, w3, w4, x0, x1)
}

/// Specialized `5x3` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m5n3_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    out[4][1] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x1).unwrap_or(0.0);
    let t0_c2 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x2);
    out[0][2] = t0_c2[0];
    out[1][2] = t0_c2[1];
    out[2][2] = t0_c2[2];
    out[3][2] = t0_c2[3];
    out[4][2] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x2).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0, x1, x2);
    out
}

/// Runtime-dispatched `5x3` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m5n3(
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
            return unsafe { gemm_q6_k_m5n3_avx2(w0, w1, w2, w3, w4, x0, x1, x2) };
        }
    }
    gemm_q6_k_m5n3_scalar(w0, w1, w2, w3, w4, x0, x1, x2)
}

/// Specialized `5x4` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m5n4_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    out[4][1] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x1).unwrap_or(0.0);
    let t0_c2 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x2);
    out[0][2] = t0_c2[0];
    out[1][2] = t0_c2[1];
    out[2][2] = t0_c2[2];
    out[3][2] = t0_c2[3];
    out[4][2] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x2).unwrap_or(0.0);
    let t0_c3 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x3);
    out[0][3] = t0_c3[0];
    out[1][3] = t0_c3[1];
    out[2][3] = t0_c3[2];
    out[3][3] = t0_c3[3];
    out[4][3] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x3).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `5x4` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m5n4(
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
            return unsafe { gemm_q6_k_m5n4_avx2(w0, w1, w2, w3, w4, x0, x1, x2, x3) };
        }
    }
    gemm_q6_k_m5n4_scalar(w0, w1, w2, w3, w4, x0, x1, x2, x3)
}

/// Specialized `6x1` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m6n1_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    w5: &[u8],
    x0: &Quantized,
) -> [[f32; 1]; 6] {
    let mut out = [[0.0f32; 1]; 6];
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x0).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0);
    out
}

/// Runtime-dispatched `6x1` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m6n1(
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
            return unsafe { gemm_q6_k_m6n1_avx2(w0, w1, w2, w3, w4, w5, x0) };
        }
    }
    gemm_q6_k_m6n1_scalar(w0, w1, w2, w3, w4, w5, x0)
}

/// Specialized `6x2` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m6n2_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x0).unwrap_or(0.0);
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    out[4][1] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x1).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0, x1);
    out
}

/// Runtime-dispatched `6x2` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m6n2(
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
            return unsafe { gemm_q6_k_m6n2_avx2(w0, w1, w2, w3, w4, w5, x0, x1) };
        }
    }
    gemm_q6_k_m6n2_scalar(w0, w1, w2, w3, w4, w5, x0, x1)
}

/// Specialized `6x3` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m6n3_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x0).unwrap_or(0.0);
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    out[4][1] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x1).unwrap_or(0.0);
    let t0_c2 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x2);
    out[0][2] = t0_c2[0];
    out[1][2] = t0_c2[1];
    out[2][2] = t0_c2[2];
    out[3][2] = t0_c2[3];
    out[4][2] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x2).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0, x1, x2);
    out
}

/// Runtime-dispatched `6x3` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m6n3(
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
            return unsafe { gemm_q6_k_m6n3_avx2(w0, w1, w2, w3, w4, w5, x0, x1, x2) };
        }
    }
    gemm_q6_k_m6n3_scalar(w0, w1, w2, w3, w4, w5, x0, x1, x2)
}

/// Specialized `6x4` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m6n4_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x0).unwrap_or(0.0);
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    out[4][1] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x1).unwrap_or(0.0);
    let t0_c2 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x2);
    out[0][2] = t0_c2[0];
    out[1][2] = t0_c2[1];
    out[2][2] = t0_c2[2];
    out[3][2] = t0_c2[3];
    out[4][2] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x2).unwrap_or(0.0);
    let t0_c3 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x3);
    out[0][3] = t0_c3[0];
    out[1][3] = t0_c3[1];
    out[2][3] = t0_c3[2];
    out[3][3] = t0_c3[3];
    out[4][3] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x3).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `6x4` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m6n4(
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
            return unsafe { gemm_q6_k_m6n4_avx2(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3) };
        }
    }
    gemm_q6_k_m6n4_scalar(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3)
}

/// Specialized `7x1` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m7n1_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q6_K, w6, x0).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0);
    out
}

/// Runtime-dispatched `7x1` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m7n1(
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
            return unsafe { gemm_q6_k_m7n1_avx2(w0, w1, w2, w3, w4, w5, w6, x0) };
        }
    }
    gemm_q6_k_m7n1_scalar(w0, w1, w2, w3, w4, w5, w6, x0)
}

/// Specialized `7x2` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m7n2_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q6_K, w6, x0).unwrap_or(0.0);
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    out[4][1] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::Q6_K, w6, x1).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0, x1);
    out
}

/// Runtime-dispatched `7x2` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m7n2(
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
            return unsafe { gemm_q6_k_m7n2_avx2(w0, w1, w2, w3, w4, w5, w6, x0, x1) };
        }
    }
    gemm_q6_k_m7n2_scalar(w0, w1, w2, w3, w4, w5, w6, x0, x1)
}

/// Specialized `7x3` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m7n3_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q6_K, w6, x0).unwrap_or(0.0);
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    out[4][1] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::Q6_K, w6, x1).unwrap_or(0.0);
    let t0_c2 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x2);
    out[0][2] = t0_c2[0];
    out[1][2] = t0_c2[1];
    out[2][2] = t0_c2[2];
    out[3][2] = t0_c2[3];
    out[4][2] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::Q6_K, w6, x2).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0, x1, x2);
    out
}

/// Runtime-dispatched `7x3` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m7n3(
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
            return unsafe { gemm_q6_k_m7n3_avx2(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2) };
        }
    }
    gemm_q6_k_m7n3_scalar(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2)
}

/// Specialized `7x4` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m7n4_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    out[4][0] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::Q6_K, w6, x0).unwrap_or(0.0);
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    out[4][1] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::Q6_K, w6, x1).unwrap_or(0.0);
    let t0_c2 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x2);
    out[0][2] = t0_c2[0];
    out[1][2] = t0_c2[1];
    out[2][2] = t0_c2[2];
    out[3][2] = t0_c2[3];
    out[4][2] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::Q6_K, w6, x2).unwrap_or(0.0);
    let t0_c3 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x3);
    out[0][3] = t0_c3[0];
    out[1][3] = t0_c3[1];
    out[2][3] = t0_c3[2];
    out[3][3] = t0_c3[3];
    out[4][3] = crate::quant::idot::integer(GgmlType::Q6_K, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::Q6_K, w5, x3).unwrap_or(0.0);
    out[6][3] = crate::quant::idot::integer(GgmlType::Q6_K, w6, x3).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `7x4` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m7n4(
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
            return unsafe { gemm_q6_k_m7n4_avx2(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3) };
        }
    }
    gemm_q6_k_m7n4_scalar(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3)
}

/// Specialized `8x1` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m8n1_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    let t1_c0 = crate::quant::avx2::q6_k_4rows([w4, w5, w6, w7], x0);
    out[4][0] = t1_c0[0];
    out[5][0] = t1_c0[1];
    out[6][0] = t1_c0[2];
    out[7][0] = t1_c0[3];
    let _ = (w0, w1, w2, w3, w4, w5, w6, w7, x0);
    out
}

/// Runtime-dispatched `8x1` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m8n1(
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
            return unsafe { gemm_q6_k_m8n1_avx2(w0, w1, w2, w3, w4, w5, w6, w7, x0) };
        }
    }
    gemm_q6_k_m8n1_scalar(w0, w1, w2, w3, w4, w5, w6, w7, x0)
}

/// Specialized `8x2` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m8n2_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    let t1_c0 = crate::quant::avx2::q6_k_4rows([w4, w5, w6, w7], x0);
    out[4][0] = t1_c0[0];
    out[5][0] = t1_c0[1];
    out[6][0] = t1_c0[2];
    out[7][0] = t1_c0[3];
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    let t1_c1 = crate::quant::avx2::q6_k_4rows([w4, w5, w6, w7], x1);
    out[4][1] = t1_c1[0];
    out[5][1] = t1_c1[1];
    out[6][1] = t1_c1[2];
    out[7][1] = t1_c1[3];
    let _ = (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1);
    out
}

/// Runtime-dispatched `8x2` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m8n2(
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
            return unsafe { gemm_q6_k_m8n2_avx2(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1) };
        }
    }
    gemm_q6_k_m8n2_scalar(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1)
}

/// Specialized `8x3` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m8n3_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    let t1_c0 = crate::quant::avx2::q6_k_4rows([w4, w5, w6, w7], x0);
    out[4][0] = t1_c0[0];
    out[5][0] = t1_c0[1];
    out[6][0] = t1_c0[2];
    out[7][0] = t1_c0[3];
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    let t1_c1 = crate::quant::avx2::q6_k_4rows([w4, w5, w6, w7], x1);
    out[4][1] = t1_c1[0];
    out[5][1] = t1_c1[1];
    out[6][1] = t1_c1[2];
    out[7][1] = t1_c1[3];
    let t0_c2 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x2);
    out[0][2] = t0_c2[0];
    out[1][2] = t0_c2[1];
    out[2][2] = t0_c2[2];
    out[3][2] = t0_c2[3];
    let t1_c2 = crate::quant::avx2::q6_k_4rows([w4, w5, w6, w7], x2);
    out[4][2] = t1_c2[0];
    out[5][2] = t1_c2[1];
    out[6][2] = t1_c2[2];
    out[7][2] = t1_c2[3];
    let _ = (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2);
    out
}

/// Runtime-dispatched `8x3` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m8n3(
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
            return unsafe { gemm_q6_k_m8n3_avx2(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2) };
        }
    }
    gemm_q6_k_m8n3_scalar(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2)
}

/// Specialized `8x4` AVX2+FMA+F16C register-tiled microkernel for `Q6_K`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_q6_k_m8n4_avx2(
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
    let t0_c0 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x0);
    out[0][0] = t0_c0[0];
    out[1][0] = t0_c0[1];
    out[2][0] = t0_c0[2];
    out[3][0] = t0_c0[3];
    let t1_c0 = crate::quant::avx2::q6_k_4rows([w4, w5, w6, w7], x0);
    out[4][0] = t1_c0[0];
    out[5][0] = t1_c0[1];
    out[6][0] = t1_c0[2];
    out[7][0] = t1_c0[3];
    let t0_c1 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x1);
    out[0][1] = t0_c1[0];
    out[1][1] = t0_c1[1];
    out[2][1] = t0_c1[2];
    out[3][1] = t0_c1[3];
    let t1_c1 = crate::quant::avx2::q6_k_4rows([w4, w5, w6, w7], x1);
    out[4][1] = t1_c1[0];
    out[5][1] = t1_c1[1];
    out[6][1] = t1_c1[2];
    out[7][1] = t1_c1[3];
    let t0_c2 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x2);
    out[0][2] = t0_c2[0];
    out[1][2] = t0_c2[1];
    out[2][2] = t0_c2[2];
    out[3][2] = t0_c2[3];
    let t1_c2 = crate::quant::avx2::q6_k_4rows([w4, w5, w6, w7], x2);
    out[4][2] = t1_c2[0];
    out[5][2] = t1_c2[1];
    out[6][2] = t1_c2[2];
    out[7][2] = t1_c2[3];
    let t0_c3 = crate::quant::avx2::q6_k_4rows([w0, w1, w2, w3], x3);
    out[0][3] = t0_c3[0];
    out[1][3] = t0_c3[1];
    out[2][3] = t0_c3[2];
    out[3][3] = t0_c3[3];
    let t1_c3 = crate::quant::avx2::q6_k_4rows([w4, w5, w6, w7], x3);
    out[4][3] = t1_c3[0];
    out[5][3] = t1_c3[1];
    out[6][3] = t1_c3[2];
    out[7][3] = t1_c3[3];
    let _ = (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3);
    out
}

/// Runtime-dispatched `8x4` microkernel for `Q6_K`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_m8n4(
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
            return unsafe { gemm_q6_k_m8n4_avx2(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3) };
        }
    }
    gemm_q6_k_m8n4_scalar(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3)
}

/// Execute tiled matrix-matrix multiplication for `Q6_K` over `rows` weight rows and `xs.len()` quantized activations.
#[allow(clippy::identity_op)]
pub fn matmul_tiled_q6_k(
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
            let tile = dispatch_q6_k_m8n4(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3);
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
            let tile = dispatch_q6_k_m4n4(w0, w1, w2, w3, x0, x1, x2, x3);
            for dr in 0..4 {
                for dc in 0..4 {
                    out[(c + dc) * rows + (r + dr)] = tile[dr][dc];
                }
            }
            r += 4;
        }
        while r < rows {
            let w0 = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
            let tile = dispatch_q6_k_m1n4(w0, x0, x1, x2, x3);
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
            let tile = dispatch_q6_k_m8n1(w0, w1, w2, w3, w4, w5, w6, w7, x0);
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
            let tile = dispatch_q6_k_m4n1(w0, w1, w2, w3, x0);
            for dr in 0..4 {
                out[c * rows + (r + dr)] = tile[dr][0];
            }
            r += 4;
        }
        while r < rows {
            let w0 = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
            let tile = dispatch_q6_k_m1n1(w0, x0);
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
        let mut buf = vec![0u8; 210];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((seed * 31 + i * 17 + 7) % 199) as u8;
        }
        buf[208..210].copy_from_slice(&0x3400u16.to_le_bytes());
        buf
    }

    #[test]
    fn q6_k_all_m1_to_8_n1_to_4_match_idot_reference() {
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

        let tile = dispatch_q6_k_m4n4(
            &rows[0], &rows[1], &rows[2], &rows[3], &xs[0], &xs[1], &xs[2], &xs[3],
        );
        for r in 0..4 {
            for c in 0..4 {
                let expected =
                    crate::quant::idot::integer(GgmlType::Q6_K, &rows[r], &xs[c]).unwrap();
                assert!(
                    (tile[r][c] - expected).abs() < 5e-3,
                    "mismatch at ({r},{c}): {} vs {expected}",
                    tile[r][c]
                );
            }
        }

        let flat_w: Vec<u8> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let mut out = vec![0.0f32; 10 * 5];
        matmul_tiled_q6_k(&flat_w, 10, 210, &xs, &mut out);
        for c in 0..5 {
            for r in 0..10 {
                let expected =
                    crate::quant::idot::integer(GgmlType::Q6_K, &rows[r], &xs[c]).unwrap();
                assert!((out[c * 10 + r] - expected).abs() < 5e-3);
            }
        }
    }
}
