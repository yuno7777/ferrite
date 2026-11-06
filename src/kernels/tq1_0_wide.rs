//! Wide-Batch `(M=1..8) x (N=5..8)` Register-Tiled GEMM Microkernels for `TQ1_0` (`GgmlType::TQ1_0`).
//!
//! Block size: `256` elements (`54` bytes per block).
//! Evaluates `M` weight rows against `N = 5..8` quantized activation vectors
//! simultaneously in a single L1-resident pass for high-throughput prompt prefill.

#[allow(unused_imports)]
use crate::gguf::GgmlType;
use crate::quant::activation::Quantized;
use crate::quant::half::f16_to_f32;

/// Specialized `1x5` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m1n5_scalar(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 1] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
    }
    [[acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4]]
}

/// Specialized `1x6` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m1n6_scalar(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 1] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
    }
    [[
        acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5,
    ]]
}

/// Specialized `1x7` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m1n7_scalar(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 1] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
    }
    [[
        acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6,
    ]]
}

/// Specialized `1x8` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m1n8_scalar(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 1] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r0_c7 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c7 += d_r0 * x7.scales[g] * (isum as f32);
        }
    }
    [[
        acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6, acc_r0_c7,
    ]]
}

/// Specialized `2x5` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m2n5_scalar(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 2] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4],
    ]
}

/// Specialized `2x6` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m2n6_scalar(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 2] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5,
        ],
    ]
}

/// Specialized `2x7` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m2n7_scalar(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 2] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6,
        ],
    ]
}

/// Specialized `2x8` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m2n8_scalar(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 2] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r0_c7 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r1_c7 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c7 += d_r0 * x7.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c7 += d_r1 * x7.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6, acc_r0_c7,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6, acc_r1_c7,
        ],
    ]
}

/// Specialized `3x5` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m3n5_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 3] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4],
    ]
}

/// Specialized `3x6` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m3n6_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 3] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5,
        ],
    ]
}

/// Specialized `3x7` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m3n7_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 3] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6,
        ],
    ]
}

/// Specialized `3x8` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m3n8_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 3] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r0_c7 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r1_c7 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    let mut acc_r2_c7 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c7 += d_r0 * x7.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c7 += d_r1 * x7.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c7 += d_r2 * x7.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6, acc_r0_c7,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6, acc_r1_c7,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6, acc_r2_c7,
        ],
    ]
}

/// Specialized `4x5` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m4n5_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 4] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4],
    ]
}

/// Specialized `4x6` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m4n6_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 4] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5,
        ],
    ]
}

/// Specialized `4x7` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m4n7_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 4] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r3_c6 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c6 += d_r3 * x6.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5, acc_r3_c6,
        ],
    ]
}

/// Specialized `4x8` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m4n8_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 4] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r0_c7 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r1_c7 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    let mut acc_r2_c7 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r3_c6 = 0.0f32;
    let mut acc_r3_c7 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c7 += d_r0 * x7.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c7 += d_r1 * x7.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c7 += d_r2 * x7.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c6 += d_r3 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c7 += d_r3 * x7.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6, acc_r0_c7,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6, acc_r1_c7,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6, acc_r2_c7,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5, acc_r3_c6, acc_r3_c7,
        ],
    ]
}

/// Specialized `5x5` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m5n5_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 5] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4],
    ]
}

/// Specialized `5x6` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m5n6_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 5] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5,
        ],
    ]
}

/// Specialized `5x7` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m5n7_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 5] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r3_c6 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    let mut acc_r4_c6 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c6 += d_r3 * x6.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c6 += d_r4 * x6.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5, acc_r3_c6,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5, acc_r4_c6,
        ],
    ]
}

/// Specialized `5x8` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m5n8_scalar(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 5] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r0_c7 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r1_c7 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    let mut acc_r2_c7 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r3_c6 = 0.0f32;
    let mut acc_r3_c7 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    let mut acc_r4_c6 = 0.0f32;
    let mut acc_r4_c7 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c7 += d_r0 * x7.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c7 += d_r1 * x7.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c7 += d_r2 * x7.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c6 += d_r3 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c7 += d_r3 * x7.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c6 += d_r4 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c7 += d_r4 * x7.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6, acc_r0_c7,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6, acc_r1_c7,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6, acc_r2_c7,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5, acc_r3_c6, acc_r3_c7,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5, acc_r4_c6, acc_r4_c7,
        ],
    ]
}

/// Specialized `6x5` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m6n5_scalar(
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
    x4: &Quantized,
) -> [[f32; 5]; 6] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4],
        [acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4],
    ]
}

/// Specialized `6x6` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m6n6_scalar(
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
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 6] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    let mut acc_r5_c5 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c5 += d_r5 * x5.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5,
        ],
        [
            acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4, acc_r5_c5,
        ],
    ]
}

/// Specialized `6x7` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m6n7_scalar(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 6] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r3_c6 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    let mut acc_r4_c6 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    let mut acc_r5_c5 = 0.0f32;
    let mut acc_r5_c6 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c6 += d_r3 * x6.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c6 += d_r4 * x6.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c5 += d_r5 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c6 += d_r5 * x6.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5, acc_r3_c6,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5, acc_r4_c6,
        ],
        [
            acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4, acc_r5_c5, acc_r5_c6,
        ],
    ]
}

/// Specialized `6x8` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m6n8_scalar(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 6] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r0_c7 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r1_c7 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    let mut acc_r2_c7 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r3_c6 = 0.0f32;
    let mut acc_r3_c7 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    let mut acc_r4_c6 = 0.0f32;
    let mut acc_r4_c7 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    let mut acc_r5_c5 = 0.0f32;
    let mut acc_r5_c6 = 0.0f32;
    let mut acc_r5_c7 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c7 += d_r0 * x7.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c7 += d_r1 * x7.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c7 += d_r2 * x7.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c6 += d_r3 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c7 += d_r3 * x7.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c6 += d_r4 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c7 += d_r4 * x7.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c5 += d_r5 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c6 += d_r5 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c7 += d_r5 * x7.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6, acc_r0_c7,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6, acc_r1_c7,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6, acc_r2_c7,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5, acc_r3_c6, acc_r3_c7,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5, acc_r4_c6, acc_r4_c7,
        ],
        [
            acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4, acc_r5_c5, acc_r5_c6, acc_r5_c7,
        ],
    ]
}

/// Specialized `7x5` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m7n5_scalar(
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
    x4: &Quantized,
) -> [[f32; 5]; 7] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    let mut acc_r6_c3 = 0.0f32;
    let mut acc_r6_c4 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
        let blk_r6 = &w6[sb * 54..(sb + 1) * 54];
        let qs_r6 = &blk_r6[0..48];
        let qh_r6 = &blk_r6[48..52];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[52], blk_r6[53]]));
        let mut trits_r6 = [0i8; 256];
        for (i, &byte) in qs_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c0 += d_r6 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c1 += d_r6 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c2 += d_r6 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c3 += d_r6 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c4 += d_r6 * x4.scales[g] * (isum as f32);
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4],
        [acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4],
        [acc_r6_c0, acc_r6_c1, acc_r6_c2, acc_r6_c3, acc_r6_c4],
    ]
}

/// Specialized `7x6` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m7n6_scalar(
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
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 7] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    let mut acc_r5_c5 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    let mut acc_r6_c3 = 0.0f32;
    let mut acc_r6_c4 = 0.0f32;
    let mut acc_r6_c5 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c5 += d_r5 * x5.scales[g] * (isum as f32);
        }
        let blk_r6 = &w6[sb * 54..(sb + 1) * 54];
        let qs_r6 = &blk_r6[0..48];
        let qh_r6 = &blk_r6[48..52];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[52], blk_r6[53]]));
        let mut trits_r6 = [0i8; 256];
        for (i, &byte) in qs_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c0 += d_r6 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c1 += d_r6 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c2 += d_r6 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c3 += d_r6 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c4 += d_r6 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c5 += d_r6 * x5.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5,
        ],
        [
            acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4, acc_r5_c5,
        ],
        [
            acc_r6_c0, acc_r6_c1, acc_r6_c2, acc_r6_c3, acc_r6_c4, acc_r6_c5,
        ],
    ]
}

/// Specialized `7x7` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m7n7_scalar(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 7] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r3_c6 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    let mut acc_r4_c6 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    let mut acc_r5_c5 = 0.0f32;
    let mut acc_r5_c6 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    let mut acc_r6_c3 = 0.0f32;
    let mut acc_r6_c4 = 0.0f32;
    let mut acc_r6_c5 = 0.0f32;
    let mut acc_r6_c6 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c6 += d_r3 * x6.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c6 += d_r4 * x6.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c5 += d_r5 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c6 += d_r5 * x6.scales[g] * (isum as f32);
        }
        let blk_r6 = &w6[sb * 54..(sb + 1) * 54];
        let qs_r6 = &blk_r6[0..48];
        let qh_r6 = &blk_r6[48..52];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[52], blk_r6[53]]));
        let mut trits_r6 = [0i8; 256];
        for (i, &byte) in qs_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c0 += d_r6 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c1 += d_r6 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c2 += d_r6 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c3 += d_r6 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c4 += d_r6 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c5 += d_r6 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c6 += d_r6 * x6.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5, acc_r3_c6,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5, acc_r4_c6,
        ],
        [
            acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4, acc_r5_c5, acc_r5_c6,
        ],
        [
            acc_r6_c0, acc_r6_c1, acc_r6_c2, acc_r6_c3, acc_r6_c4, acc_r6_c5, acc_r6_c6,
        ],
    ]
}

/// Specialized `7x8` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m7n8_scalar(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 7] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r0_c7 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r1_c7 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    let mut acc_r2_c7 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r3_c6 = 0.0f32;
    let mut acc_r3_c7 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    let mut acc_r4_c6 = 0.0f32;
    let mut acc_r4_c7 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    let mut acc_r5_c5 = 0.0f32;
    let mut acc_r5_c6 = 0.0f32;
    let mut acc_r5_c7 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    let mut acc_r6_c3 = 0.0f32;
    let mut acc_r6_c4 = 0.0f32;
    let mut acc_r6_c5 = 0.0f32;
    let mut acc_r6_c6 = 0.0f32;
    let mut acc_r6_c7 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c7 += d_r0 * x7.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c7 += d_r1 * x7.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c7 += d_r2 * x7.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c6 += d_r3 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c7 += d_r3 * x7.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c6 += d_r4 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c7 += d_r4 * x7.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c5 += d_r5 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c6 += d_r5 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c7 += d_r5 * x7.scales[g] * (isum as f32);
        }
        let blk_r6 = &w6[sb * 54..(sb + 1) * 54];
        let qs_r6 = &blk_r6[0..48];
        let qh_r6 = &blk_r6[48..52];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[52], blk_r6[53]]));
        let mut trits_r6 = [0i8; 256];
        for (i, &byte) in qs_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c0 += d_r6 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c1 += d_r6 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c2 += d_r6 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c3 += d_r6 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c4 += d_r6 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c5 += d_r6 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c6 += d_r6 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c7 += d_r6 * x7.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6, acc_r0_c7,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6, acc_r1_c7,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6, acc_r2_c7,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5, acc_r3_c6, acc_r3_c7,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5, acc_r4_c6, acc_r4_c7,
        ],
        [
            acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4, acc_r5_c5, acc_r5_c6, acc_r5_c7,
        ],
        [
            acc_r6_c0, acc_r6_c1, acc_r6_c2, acc_r6_c3, acc_r6_c4, acc_r6_c5, acc_r6_c6, acc_r6_c7,
        ],
    ]
}

/// Specialized `8x5` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m8n5_scalar(
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
    x4: &Quantized,
) -> [[f32; 5]; 8] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    let mut acc_r6_c3 = 0.0f32;
    let mut acc_r6_c4 = 0.0f32;
    let mut acc_r7_c0 = 0.0f32;
    let mut acc_r7_c1 = 0.0f32;
    let mut acc_r7_c2 = 0.0f32;
    let mut acc_r7_c3 = 0.0f32;
    let mut acc_r7_c4 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
        let blk_r6 = &w6[sb * 54..(sb + 1) * 54];
        let qs_r6 = &blk_r6[0..48];
        let qh_r6 = &blk_r6[48..52];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[52], blk_r6[53]]));
        let mut trits_r6 = [0i8; 256];
        for (i, &byte) in qs_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c0 += d_r6 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c1 += d_r6 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c2 += d_r6 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c3 += d_r6 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c4 += d_r6 * x4.scales[g] * (isum as f32);
        }
        let blk_r7 = &w7[sb * 54..(sb + 1) * 54];
        let qs_r7 = &blk_r7[0..48];
        let qh_r7 = &blk_r7[48..52];
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[52], blk_r7[53]]));
        let mut trits_r7 = [0i8; 256];
        for (i, &byte) in qs_r7.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r7[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r7.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r7[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c0 += d_r7 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c1 += d_r7 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c2 += d_r7 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c3 += d_r7 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c4 += d_r7 * x4.scales[g] * (isum as f32);
        }
    }
    [
        [acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4],
        [acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4],
        [acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4],
        [acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4],
        [acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4],
        [acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4],
        [acc_r6_c0, acc_r6_c1, acc_r6_c2, acc_r6_c3, acc_r6_c4],
        [acc_r7_c0, acc_r7_c1, acc_r7_c2, acc_r7_c3, acc_r7_c4],
    ]
}

/// Specialized `8x6` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m8n6_scalar(
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
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 8] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    let mut acc_r5_c5 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    let mut acc_r6_c3 = 0.0f32;
    let mut acc_r6_c4 = 0.0f32;
    let mut acc_r6_c5 = 0.0f32;
    let mut acc_r7_c0 = 0.0f32;
    let mut acc_r7_c1 = 0.0f32;
    let mut acc_r7_c2 = 0.0f32;
    let mut acc_r7_c3 = 0.0f32;
    let mut acc_r7_c4 = 0.0f32;
    let mut acc_r7_c5 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c5 += d_r5 * x5.scales[g] * (isum as f32);
        }
        let blk_r6 = &w6[sb * 54..(sb + 1) * 54];
        let qs_r6 = &blk_r6[0..48];
        let qh_r6 = &blk_r6[48..52];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[52], blk_r6[53]]));
        let mut trits_r6 = [0i8; 256];
        for (i, &byte) in qs_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c0 += d_r6 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c1 += d_r6 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c2 += d_r6 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c3 += d_r6 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c4 += d_r6 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c5 += d_r6 * x5.scales[g] * (isum as f32);
        }
        let blk_r7 = &w7[sb * 54..(sb + 1) * 54];
        let qs_r7 = &blk_r7[0..48];
        let qh_r7 = &blk_r7[48..52];
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[52], blk_r7[53]]));
        let mut trits_r7 = [0i8; 256];
        for (i, &byte) in qs_r7.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r7[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r7.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r7[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c0 += d_r7 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c1 += d_r7 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c2 += d_r7 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c3 += d_r7 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c4 += d_r7 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c5 += d_r7 * x5.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5,
        ],
        [
            acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4, acc_r5_c5,
        ],
        [
            acc_r6_c0, acc_r6_c1, acc_r6_c2, acc_r6_c3, acc_r6_c4, acc_r6_c5,
        ],
        [
            acc_r7_c0, acc_r7_c1, acc_r7_c2, acc_r7_c3, acc_r7_c4, acc_r7_c5,
        ],
    ]
}

/// Specialized `8x7` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m8n7_scalar(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 8] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r3_c6 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    let mut acc_r4_c6 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    let mut acc_r5_c5 = 0.0f32;
    let mut acc_r5_c6 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    let mut acc_r6_c3 = 0.0f32;
    let mut acc_r6_c4 = 0.0f32;
    let mut acc_r6_c5 = 0.0f32;
    let mut acc_r6_c6 = 0.0f32;
    let mut acc_r7_c0 = 0.0f32;
    let mut acc_r7_c1 = 0.0f32;
    let mut acc_r7_c2 = 0.0f32;
    let mut acc_r7_c3 = 0.0f32;
    let mut acc_r7_c4 = 0.0f32;
    let mut acc_r7_c5 = 0.0f32;
    let mut acc_r7_c6 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c6 += d_r3 * x6.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c6 += d_r4 * x6.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c5 += d_r5 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c6 += d_r5 * x6.scales[g] * (isum as f32);
        }
        let blk_r6 = &w6[sb * 54..(sb + 1) * 54];
        let qs_r6 = &blk_r6[0..48];
        let qh_r6 = &blk_r6[48..52];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[52], blk_r6[53]]));
        let mut trits_r6 = [0i8; 256];
        for (i, &byte) in qs_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c0 += d_r6 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c1 += d_r6 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c2 += d_r6 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c3 += d_r6 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c4 += d_r6 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c5 += d_r6 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c6 += d_r6 * x6.scales[g] * (isum as f32);
        }
        let blk_r7 = &w7[sb * 54..(sb + 1) * 54];
        let qs_r7 = &blk_r7[0..48];
        let qh_r7 = &blk_r7[48..52];
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[52], blk_r7[53]]));
        let mut trits_r7 = [0i8; 256];
        for (i, &byte) in qs_r7.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r7[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r7.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r7[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c0 += d_r7 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c1 += d_r7 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c2 += d_r7 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c3 += d_r7 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c4 += d_r7 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c5 += d_r7 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c6 += d_r7 * x6.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5, acc_r3_c6,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5, acc_r4_c6,
        ],
        [
            acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4, acc_r5_c5, acc_r5_c6,
        ],
        [
            acc_r6_c0, acc_r6_c1, acc_r6_c2, acc_r6_c3, acc_r6_c4, acc_r6_c5, acc_r6_c6,
        ],
        [
            acc_r7_c0, acc_r7_c1, acc_r7_c2, acc_r7_c3, acc_r7_c4, acc_r7_c5, acc_r7_c6,
        ],
    ]
}

/// Specialized `8x8` wide-batch register-tiled scalar microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
pub fn gemm_tq1_0_m8n8_scalar(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 8] {
    let super_blocks = x0.scales.len() / 8;
    let mut acc_r0_c0 = 0.0f32;
    let mut acc_r0_c1 = 0.0f32;
    let mut acc_r0_c2 = 0.0f32;
    let mut acc_r0_c3 = 0.0f32;
    let mut acc_r0_c4 = 0.0f32;
    let mut acc_r0_c5 = 0.0f32;
    let mut acc_r0_c6 = 0.0f32;
    let mut acc_r0_c7 = 0.0f32;
    let mut acc_r1_c0 = 0.0f32;
    let mut acc_r1_c1 = 0.0f32;
    let mut acc_r1_c2 = 0.0f32;
    let mut acc_r1_c3 = 0.0f32;
    let mut acc_r1_c4 = 0.0f32;
    let mut acc_r1_c5 = 0.0f32;
    let mut acc_r1_c6 = 0.0f32;
    let mut acc_r1_c7 = 0.0f32;
    let mut acc_r2_c0 = 0.0f32;
    let mut acc_r2_c1 = 0.0f32;
    let mut acc_r2_c2 = 0.0f32;
    let mut acc_r2_c3 = 0.0f32;
    let mut acc_r2_c4 = 0.0f32;
    let mut acc_r2_c5 = 0.0f32;
    let mut acc_r2_c6 = 0.0f32;
    let mut acc_r2_c7 = 0.0f32;
    let mut acc_r3_c0 = 0.0f32;
    let mut acc_r3_c1 = 0.0f32;
    let mut acc_r3_c2 = 0.0f32;
    let mut acc_r3_c3 = 0.0f32;
    let mut acc_r3_c4 = 0.0f32;
    let mut acc_r3_c5 = 0.0f32;
    let mut acc_r3_c6 = 0.0f32;
    let mut acc_r3_c7 = 0.0f32;
    let mut acc_r4_c0 = 0.0f32;
    let mut acc_r4_c1 = 0.0f32;
    let mut acc_r4_c2 = 0.0f32;
    let mut acc_r4_c3 = 0.0f32;
    let mut acc_r4_c4 = 0.0f32;
    let mut acc_r4_c5 = 0.0f32;
    let mut acc_r4_c6 = 0.0f32;
    let mut acc_r4_c7 = 0.0f32;
    let mut acc_r5_c0 = 0.0f32;
    let mut acc_r5_c1 = 0.0f32;
    let mut acc_r5_c2 = 0.0f32;
    let mut acc_r5_c3 = 0.0f32;
    let mut acc_r5_c4 = 0.0f32;
    let mut acc_r5_c5 = 0.0f32;
    let mut acc_r5_c6 = 0.0f32;
    let mut acc_r5_c7 = 0.0f32;
    let mut acc_r6_c0 = 0.0f32;
    let mut acc_r6_c1 = 0.0f32;
    let mut acc_r6_c2 = 0.0f32;
    let mut acc_r6_c3 = 0.0f32;
    let mut acc_r6_c4 = 0.0f32;
    let mut acc_r6_c5 = 0.0f32;
    let mut acc_r6_c6 = 0.0f32;
    let mut acc_r6_c7 = 0.0f32;
    let mut acc_r7_c0 = 0.0f32;
    let mut acc_r7_c1 = 0.0f32;
    let mut acc_r7_c2 = 0.0f32;
    let mut acc_r7_c3 = 0.0f32;
    let mut acc_r7_c4 = 0.0f32;
    let mut acc_r7_c5 = 0.0f32;
    let mut acc_r7_c6 = 0.0f32;
    let mut acc_r7_c7 = 0.0f32;
    for sb in 0..super_blocks {
        let b_base = sb * 8;
        let blk_r0 = &w0[sb * 54..(sb + 1) * 54];
        let qs_r0 = &blk_r0[0..48];
        let qh_r0 = &blk_r0[48..52];
        let d_r0 = f16_to_f32(u16::from_le_bytes([blk_r0[52], blk_r0[53]]));
        let mut trits_r0 = [0i8; 256];
        for (i, &byte) in qs_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r0.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r0[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c0 += d_r0 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c1 += d_r0 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c2 += d_r0 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c3 += d_r0 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c4 += d_r0 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c5 += d_r0 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c6 += d_r0 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r0[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r0_c7 += d_r0 * x7.scales[g] * (isum as f32);
        }
        let blk_r1 = &w1[sb * 54..(sb + 1) * 54];
        let qs_r1 = &blk_r1[0..48];
        let qh_r1 = &blk_r1[48..52];
        let d_r1 = f16_to_f32(u16::from_le_bytes([blk_r1[52], blk_r1[53]]));
        let mut trits_r1 = [0i8; 256];
        for (i, &byte) in qs_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r1.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r1[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c0 += d_r1 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c1 += d_r1 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c2 += d_r1 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c3 += d_r1 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c4 += d_r1 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c5 += d_r1 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c6 += d_r1 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r1[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r1_c7 += d_r1 * x7.scales[g] * (isum as f32);
        }
        let blk_r2 = &w2[sb * 54..(sb + 1) * 54];
        let qs_r2 = &blk_r2[0..48];
        let qh_r2 = &blk_r2[48..52];
        let d_r2 = f16_to_f32(u16::from_le_bytes([blk_r2[52], blk_r2[53]]));
        let mut trits_r2 = [0i8; 256];
        for (i, &byte) in qs_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r2.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r2[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c0 += d_r2 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c1 += d_r2 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c2 += d_r2 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c3 += d_r2 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c4 += d_r2 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c5 += d_r2 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c6 += d_r2 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r2[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r2_c7 += d_r2 * x7.scales[g] * (isum as f32);
        }
        let blk_r3 = &w3[sb * 54..(sb + 1) * 54];
        let qs_r3 = &blk_r3[0..48];
        let qh_r3 = &blk_r3[48..52];
        let d_r3 = f16_to_f32(u16::from_le_bytes([blk_r3[52], blk_r3[53]]));
        let mut trits_r3 = [0i8; 256];
        for (i, &byte) in qs_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r3.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r3[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c0 += d_r3 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c1 += d_r3 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c2 += d_r3 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c3 += d_r3 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c4 += d_r3 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c5 += d_r3 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c6 += d_r3 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r3[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r3_c7 += d_r3 * x7.scales[g] * (isum as f32);
        }
        let blk_r4 = &w4[sb * 54..(sb + 1) * 54];
        let qs_r4 = &blk_r4[0..48];
        let qh_r4 = &blk_r4[48..52];
        let d_r4 = f16_to_f32(u16::from_le_bytes([blk_r4[52], blk_r4[53]]));
        let mut trits_r4 = [0i8; 256];
        for (i, &byte) in qs_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r4.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r4[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c0 += d_r4 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c1 += d_r4 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c2 += d_r4 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c3 += d_r4 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c4 += d_r4 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c5 += d_r4 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c6 += d_r4 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r4[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r4_c7 += d_r4 * x7.scales[g] * (isum as f32);
        }
        let blk_r5 = &w5[sb * 54..(sb + 1) * 54];
        let qs_r5 = &blk_r5[0..48];
        let qh_r5 = &blk_r5[48..52];
        let d_r5 = f16_to_f32(u16::from_le_bytes([blk_r5[52], blk_r5[53]]));
        let mut trits_r5 = [0i8; 256];
        for (i, &byte) in qs_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r5.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r5[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c0 += d_r5 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c1 += d_r5 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c2 += d_r5 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c3 += d_r5 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c4 += d_r5 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c5 += d_r5 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c6 += d_r5 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r5[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r5_c7 += d_r5 * x7.scales[g] * (isum as f32);
        }
        let blk_r6 = &w6[sb * 54..(sb + 1) * 54];
        let qs_r6 = &blk_r6[0..48];
        let qh_r6 = &blk_r6[48..52];
        let d_r6 = f16_to_f32(u16::from_le_bytes([blk_r6[52], blk_r6[53]]));
        let mut trits_r6 = [0i8; 256];
        for (i, &byte) in qs_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r6.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r6[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c0 += d_r6 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c1 += d_r6 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c2 += d_r6 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c3 += d_r6 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c4 += d_r6 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c5 += d_r6 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c6 += d_r6 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r6[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r6_c7 += d_r6 * x7.scales[g] * (isum as f32);
        }
        let blk_r7 = &w7[sb * 54..(sb + 1) * 54];
        let qs_r7 = &blk_r7[0..48];
        let qh_r7 = &blk_r7[48..52];
        let d_r7 = f16_to_f32(u16::from_le_bytes([blk_r7[52], blk_r7[53]]));
        let mut trits_r7 = [0i8; 256];
        for (i, &byte) in qs_r7.iter().enumerate() {
            let mut rem = byte;
            for k in 0..5 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r7[i * 5 + k] = t;
            }
        }
        for (i, &byte) in qh_r7.iter().enumerate() {
            let mut rem = byte;
            for k in 0..4 {
                let t = (rem % 3) as i8 - 1;
                rem /= 3;
                trits_r7[240 + i * 4 + k] = t;
            }
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x0.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c0 += d_r7 * x0.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x1.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c1 += d_r7 * x1.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x2.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c2 += d_r7 * x2.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x3.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c3 += d_r7 * x3.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x4.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c4 += d_r7 * x4.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x5.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c5 += d_r7 * x5.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x6.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c6 += d_r7 * x6.scales[g] * (isum as f32);
        }
        for ib in 0..8 {
            let g = b_base + ib;
            let x_q = &x7.quants[g * 32..(g + 1) * 32];
            let t_sub = &trits_r7[ib * 32..(ib + 1) * 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (t_sub[j] as i32) * (x_q[j] as i32);
            }
            acc_r7_c7 += d_r7 * x7.scales[g] * (isum as f32);
        }
    }
    [
        [
            acc_r0_c0, acc_r0_c1, acc_r0_c2, acc_r0_c3, acc_r0_c4, acc_r0_c5, acc_r0_c6, acc_r0_c7,
        ],
        [
            acc_r1_c0, acc_r1_c1, acc_r1_c2, acc_r1_c3, acc_r1_c4, acc_r1_c5, acc_r1_c6, acc_r1_c7,
        ],
        [
            acc_r2_c0, acc_r2_c1, acc_r2_c2, acc_r2_c3, acc_r2_c4, acc_r2_c5, acc_r2_c6, acc_r2_c7,
        ],
        [
            acc_r3_c0, acc_r3_c1, acc_r3_c2, acc_r3_c3, acc_r3_c4, acc_r3_c5, acc_r3_c6, acc_r3_c7,
        ],
        [
            acc_r4_c0, acc_r4_c1, acc_r4_c2, acc_r4_c3, acc_r4_c4, acc_r4_c5, acc_r4_c6, acc_r4_c7,
        ],
        [
            acc_r5_c0, acc_r5_c1, acc_r5_c2, acc_r5_c3, acc_r5_c4, acc_r5_c5, acc_r5_c6, acc_r5_c7,
        ],
        [
            acc_r6_c0, acc_r6_c1, acc_r6_c2, acc_r6_c3, acc_r6_c4, acc_r6_c5, acc_r6_c6, acc_r6_c7,
        ],
        [
            acc_r7_c0, acc_r7_c1, acc_r7_c2, acc_r7_c3, acc_r7_c4, acc_r7_c5, acc_r7_c6, acc_r7_c7,
        ],
    ]
}

/// Specialized `1x5` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m1n5_avx2(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 1] {
    let mut out = [[0.0f32; 5]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    let _ = (w0, x0, x1, x2, x3, x4);
    out
}

/// Runtime-dispatched `1x5` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m1n5(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m1n5_avx2(w0, x0, x1, x2, x3, x4) };
        }
    }
    gemm_tq1_0_m1n5_scalar(w0, x0, x1, x2, x3, x4)
}

/// Specialized `1x6` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m1n6_avx2(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 1] {
    let mut out = [[0.0f32; 6]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    let _ = (w0, x0, x1, x2, x3, x4, x5);
    out
}

/// Runtime-dispatched `1x6` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m1n6(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m1n6_avx2(w0, x0, x1, x2, x3, x4, x5) };
        }
    }
    gemm_tq1_0_m1n6_scalar(w0, x0, x1, x2, x3, x4, x5)
}

/// Specialized `1x7` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m1n7_avx2(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 1] {
    let mut out = [[0.0f32; 7]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    let _ = (w0, x0, x1, x2, x3, x4, x5, x6);
    out
}

/// Runtime-dispatched `1x7` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m1n7(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m1n7_avx2(w0, x0, x1, x2, x3, x4, x5, x6) };
        }
    }
    gemm_tq1_0_m1n7_scalar(w0, x0, x1, x2, x3, x4, x5, x6)
}

/// Specialized `1x8` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m1n8_avx2(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 1] {
    let mut out = [[0.0f32; 8]; 1];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[0][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x7).unwrap_or(0.0);
    let _ = (w0, x0, x1, x2, x3, x4, x5, x6, x7);
    out
}

/// Runtime-dispatched `1x8` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m1n8(
    w0: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 1] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m1n8_avx2(w0, x0, x1, x2, x3, x4, x5, x6, x7) };
        }
    }
    gemm_tq1_0_m1n8_scalar(w0, x0, x1, x2, x3, x4, x5, x6, x7)
}

/// Specialized `2x5` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m2n5_avx2(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 2] {
    let mut out = [[0.0f32; 5]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    let _ = (w0, w1, x0, x1, x2, x3, x4);
    out
}

/// Runtime-dispatched `2x5` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m2n5(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m2n5_avx2(w0, w1, x0, x1, x2, x3, x4) };
        }
    }
    gemm_tq1_0_m2n5_scalar(w0, w1, x0, x1, x2, x3, x4)
}

/// Specialized `2x6` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m2n6_avx2(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 2] {
    let mut out = [[0.0f32; 6]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    let _ = (w0, w1, x0, x1, x2, x3, x4, x5);
    out
}

/// Runtime-dispatched `2x6` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m2n6(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m2n6_avx2(w0, w1, x0, x1, x2, x3, x4, x5) };
        }
    }
    gemm_tq1_0_m2n6_scalar(w0, w1, x0, x1, x2, x3, x4, x5)
}

/// Specialized `2x7` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m2n7_avx2(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 2] {
    let mut out = [[0.0f32; 7]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    let _ = (w0, w1, x0, x1, x2, x3, x4, x5, x6);
    out
}

/// Runtime-dispatched `2x7` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m2n7(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m2n7_avx2(w0, w1, x0, x1, x2, x3, x4, x5, x6) };
        }
    }
    gemm_tq1_0_m2n7_scalar(w0, w1, x0, x1, x2, x3, x4, x5, x6)
}

/// Specialized `2x8` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m2n8_avx2(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 2] {
    let mut out = [[0.0f32; 8]; 2];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[0][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x7).unwrap_or(0.0);
    out[1][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x7).unwrap_or(0.0);
    let _ = (w0, w1, x0, x1, x2, x3, x4, x5, x6, x7);
    out
}

/// Runtime-dispatched `2x8` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m2n8(
    w0: &[u8],
    w1: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m2n8_avx2(w0, w1, x0, x1, x2, x3, x4, x5, x6, x7) };
        }
    }
    gemm_tq1_0_m2n8_scalar(w0, w1, x0, x1, x2, x3, x4, x5, x6, x7)
}

/// Specialized `3x5` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m3n5_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 3] {
    let mut out = [[0.0f32; 5]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0, x1, x2, x3, x4);
    out
}

/// Runtime-dispatched `3x5` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m3n5(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m3n5_avx2(w0, w1, w2, x0, x1, x2, x3, x4) };
        }
    }
    gemm_tq1_0_m3n5_scalar(w0, w1, w2, x0, x1, x2, x3, x4)
}

/// Specialized `3x6` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m3n6_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 3] {
    let mut out = [[0.0f32; 6]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0, x1, x2, x3, x4, x5);
    out
}

/// Runtime-dispatched `3x6` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m3n6(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m3n6_avx2(w0, w1, w2, x0, x1, x2, x3, x4, x5) };
        }
    }
    gemm_tq1_0_m3n6_scalar(w0, w1, w2, x0, x1, x2, x3, x4, x5)
}

/// Specialized `3x7` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m3n7_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 3] {
    let mut out = [[0.0f32; 7]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0, x1, x2, x3, x4, x5, x6);
    out
}

/// Runtime-dispatched `3x7` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m3n7(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m3n7_avx2(w0, w1, w2, x0, x1, x2, x3, x4, x5, x6) };
        }
    }
    gemm_tq1_0_m3n7_scalar(w0, w1, w2, x0, x1, x2, x3, x4, x5, x6)
}

/// Specialized `3x8` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m3n8_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 3] {
    let mut out = [[0.0f32; 8]; 3];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    out[0][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x7).unwrap_or(0.0);
    out[1][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x7).unwrap_or(0.0);
    out[2][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x7).unwrap_or(0.0);
    let _ = (w0, w1, w2, x0, x1, x2, x3, x4, x5, x6, x7);
    out
}

/// Runtime-dispatched `3x8` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m3n8(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m3n8_avx2(w0, w1, w2, x0, x1, x2, x3, x4, x5, x6, x7) };
        }
    }
    gemm_tq1_0_m3n8_scalar(w0, w1, w2, x0, x1, x2, x3, x4, x5, x6, x7)
}

/// Specialized `4x5` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m4n5_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 4] {
    let mut out = [[0.0f32; 5]; 4];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, x0, x1, x2, x3, x4);
    out
}

/// Runtime-dispatched `4x5` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m4n5(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m4n5_avx2(w0, w1, w2, w3, x0, x1, x2, x3, x4) };
        }
    }
    gemm_tq1_0_m4n5_scalar(w0, w1, w2, w3, x0, x1, x2, x3, x4)
}

/// Specialized `4x6` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m4n6_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 4] {
    let mut out = [[0.0f32; 6]; 4];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, x0, x1, x2, x3, x4, x5);
    out
}

/// Runtime-dispatched `4x6` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m4n6(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m4n6_avx2(w0, w1, w2, w3, x0, x1, x2, x3, x4, x5) };
        }
    }
    gemm_tq1_0_m4n6_scalar(w0, w1, w2, w3, x0, x1, x2, x3, x4, x5)
}

/// Specialized `4x7` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m4n7_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 4] {
    let mut out = [[0.0f32; 7]; 4];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    out[3][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x6).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, x0, x1, x2, x3, x4, x5, x6);
    out
}

/// Runtime-dispatched `4x7` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m4n7(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m4n7_avx2(w0, w1, w2, w3, x0, x1, x2, x3, x4, x5, x6) };
        }
    }
    gemm_tq1_0_m4n7_scalar(w0, w1, w2, w3, x0, x1, x2, x3, x4, x5, x6)
}

/// Specialized `4x8` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m4n8_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 4] {
    let mut out = [[0.0f32; 8]; 4];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    out[3][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x6).unwrap_or(0.0);
    out[0][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x7).unwrap_or(0.0);
    out[1][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x7).unwrap_or(0.0);
    out[2][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x7).unwrap_or(0.0);
    out[3][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x7).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, x0, x1, x2, x3, x4, x5, x6, x7);
    out
}

/// Runtime-dispatched `4x8` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m4n8(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m4n8_avx2(w0, w1, w2, w3, x0, x1, x2, x3, x4, x5, x6, x7) };
        }
    }
    gemm_tq1_0_m4n8_scalar(w0, w1, w2, w3, x0, x1, x2, x3, x4, x5, x6, x7)
}

/// Specialized `5x5` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m5n5_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 5] {
    let mut out = [[0.0f32; 5]; 5];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0, x1, x2, x3, x4);
    out
}

/// Runtime-dispatched `5x5` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m5n5(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
) -> [[f32; 5]; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m5n5_avx2(w0, w1, w2, w3, w4, x0, x1, x2, x3, x4) };
        }
    }
    gemm_tq1_0_m5n5_scalar(w0, w1, w2, w3, w4, x0, x1, x2, x3, x4)
}

/// Specialized `5x6` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m5n6_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 5] {
    let mut out = [[0.0f32; 6]; 5];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0, x1, x2, x3, x4, x5);
    out
}

/// Runtime-dispatched `5x6` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m5n6(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m5n6_avx2(w0, w1, w2, w3, w4, x0, x1, x2, x3, x4, x5) };
        }
    }
    gemm_tq1_0_m5n6_scalar(w0, w1, w2, w3, w4, x0, x1, x2, x3, x4, x5)
}

/// Specialized `5x7` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m5n7_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 5] {
    let mut out = [[0.0f32; 7]; 5];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    out[3][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x6).unwrap_or(0.0);
    out[4][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x6).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0, x1, x2, x3, x4, x5, x6);
    out
}

/// Runtime-dispatched `5x7` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m5n7(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m5n7_avx2(w0, w1, w2, w3, w4, x0, x1, x2, x3, x4, x5, x6) };
        }
    }
    gemm_tq1_0_m5n7_scalar(w0, w1, w2, w3, w4, x0, x1, x2, x3, x4, x5, x6)
}

/// Specialized `5x8` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m5n8_avx2(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 5] {
    let mut out = [[0.0f32; 8]; 5];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    out[3][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x6).unwrap_or(0.0);
    out[4][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x6).unwrap_or(0.0);
    out[0][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x7).unwrap_or(0.0);
    out[1][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x7).unwrap_or(0.0);
    out[2][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x7).unwrap_or(0.0);
    out[3][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x7).unwrap_or(0.0);
    out[4][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x7).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, x0, x1, x2, x3, x4, x5, x6, x7);
    out
}

/// Runtime-dispatched `5x8` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m5n8(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    w4: &[u8],
    x0: &Quantized,
    x1: &Quantized,
    x2: &Quantized,
    x3: &Quantized,
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe {
                gemm_tq1_0_m5n8_avx2(w0, w1, w2, w3, w4, x0, x1, x2, x3, x4, x5, x6, x7)
            };
        }
    }
    gemm_tq1_0_m5n8_scalar(w0, w1, w2, w3, w4, x0, x1, x2, x3, x4, x5, x6, x7)
}

/// Specialized `6x5` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m6n5_avx2(
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
    x4: &Quantized,
) -> [[f32; 5]; 6] {
    let mut out = [[0.0f32; 5]; 6];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4);
    out
}

/// Runtime-dispatched `6x5` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m6n5(
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
    x4: &Quantized,
) -> [[f32; 5]; 6] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m6n5_avx2(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4) };
        }
    }
    gemm_tq1_0_m6n5_scalar(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4)
}

/// Specialized `6x6` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m6n6_avx2(
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
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 6] {
    let mut out = [[0.0f32; 6]; 6];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    out[5][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x5).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4, x5);
    out
}

/// Runtime-dispatched `6x6` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m6n6(
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
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 6] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m6n6_avx2(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4, x5) };
        }
    }
    gemm_tq1_0_m6n6_scalar(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4, x5)
}

/// Specialized `6x7` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m6n7_avx2(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 6] {
    let mut out = [[0.0f32; 7]; 6];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    out[5][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    out[3][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x6).unwrap_or(0.0);
    out[4][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x6).unwrap_or(0.0);
    out[5][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x6).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4, x5, x6);
    out
}

/// Runtime-dispatched `6x7` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m6n7(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 6] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe {
                gemm_tq1_0_m6n7_avx2(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4, x5, x6)
            };
        }
    }
    gemm_tq1_0_m6n7_scalar(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4, x5, x6)
}

/// Specialized `6x8` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m6n8_avx2(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 6] {
    let mut out = [[0.0f32; 8]; 6];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    out[5][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    out[3][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x6).unwrap_or(0.0);
    out[4][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x6).unwrap_or(0.0);
    out[5][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x6).unwrap_or(0.0);
    out[0][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x7).unwrap_or(0.0);
    out[1][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x7).unwrap_or(0.0);
    out[2][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x7).unwrap_or(0.0);
    out[3][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x7).unwrap_or(0.0);
    out[4][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x7).unwrap_or(0.0);
    out[5][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x7).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4, x5, x6, x7);
    out
}

/// Runtime-dispatched `6x8` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m6n8(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 6] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe {
                gemm_tq1_0_m6n8_avx2(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4, x5, x6, x7)
            };
        }
    }
    gemm_tq1_0_m6n8_scalar(w0, w1, w2, w3, w4, w5, x0, x1, x2, x3, x4, x5, x6, x7)
}

/// Specialized `7x5` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m7n5_avx2(
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
    x4: &Quantized,
) -> [[f32; 5]; 7] {
    let mut out = [[0.0f32; 5]; 7];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[6][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    out[6][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x4).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4);
    out
}

/// Runtime-dispatched `7x5` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m7n5(
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
    x4: &Quantized,
) -> [[f32; 5]; 7] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe { gemm_tq1_0_m7n5_avx2(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4) };
        }
    }
    gemm_tq1_0_m7n5_scalar(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4)
}

/// Specialized `7x6` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m7n6_avx2(
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
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 7] {
    let mut out = [[0.0f32; 6]; 7];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[6][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    out[6][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    out[5][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x5).unwrap_or(0.0);
    out[6][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x5).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4, x5);
    out
}

/// Runtime-dispatched `7x6` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m7n6(
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
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 7] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe {
                gemm_tq1_0_m7n6_avx2(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4, x5)
            };
        }
    }
    gemm_tq1_0_m7n6_scalar(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4, x5)
}

/// Specialized `7x7` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m7n7_avx2(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 7] {
    let mut out = [[0.0f32; 7]; 7];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[6][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    out[6][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    out[5][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x5).unwrap_or(0.0);
    out[6][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    out[3][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x6).unwrap_or(0.0);
    out[4][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x6).unwrap_or(0.0);
    out[5][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x6).unwrap_or(0.0);
    out[6][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x6).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4, x5, x6);
    out
}

/// Runtime-dispatched `7x7` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m7n7(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 7] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe {
                gemm_tq1_0_m7n7_avx2(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4, x5, x6)
            };
        }
    }
    gemm_tq1_0_m7n7_scalar(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4, x5, x6)
}

/// Specialized `7x8` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m7n8_avx2(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 7] {
    let mut out = [[0.0f32; 8]; 7];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[6][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    out[6][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    out[5][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x5).unwrap_or(0.0);
    out[6][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    out[3][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x6).unwrap_or(0.0);
    out[4][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x6).unwrap_or(0.0);
    out[5][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x6).unwrap_or(0.0);
    out[6][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x6).unwrap_or(0.0);
    out[0][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x7).unwrap_or(0.0);
    out[1][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x7).unwrap_or(0.0);
    out[2][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x7).unwrap_or(0.0);
    out[3][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x7).unwrap_or(0.0);
    out[4][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x7).unwrap_or(0.0);
    out[5][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x7).unwrap_or(0.0);
    out[6][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x7).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4, x5, x6, x7);
    out
}

/// Runtime-dispatched `7x8` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m7n8(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 7] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe {
                gemm_tq1_0_m7n8_avx2(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4, x5, x6, x7)
            };
        }
    }
    gemm_tq1_0_m7n8_scalar(w0, w1, w2, w3, w4, w5, w6, x0, x1, x2, x3, x4, x5, x6, x7)
}

/// Specialized `8x5` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m8n5_avx2(
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
    x4: &Quantized,
) -> [[f32; 5]; 8] {
    let mut out = [[0.0f32; 5]; 8];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x0).unwrap_or(0.0);
    out[7][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x1).unwrap_or(0.0);
    out[7][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x2).unwrap_or(0.0);
    out[7][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[6][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x3).unwrap_or(0.0);
    out[7][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    out[6][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x4).unwrap_or(0.0);
    out[7][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x4).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4);
    out
}

/// Runtime-dispatched `8x5` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m8n5(
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
    x4: &Quantized,
) -> [[f32; 5]; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe {
                gemm_tq1_0_m8n5_avx2(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4)
            };
        }
    }
    gemm_tq1_0_m8n5_scalar(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4)
}

/// Specialized `8x6` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m8n6_avx2(
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
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 8] {
    let mut out = [[0.0f32; 6]; 8];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x0).unwrap_or(0.0);
    out[7][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x1).unwrap_or(0.0);
    out[7][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x2).unwrap_or(0.0);
    out[7][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[6][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x3).unwrap_or(0.0);
    out[7][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    out[6][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x4).unwrap_or(0.0);
    out[7][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    out[5][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x5).unwrap_or(0.0);
    out[6][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x5).unwrap_or(0.0);
    out[7][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x5).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5);
    out
}

/// Runtime-dispatched `8x6` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m8n6(
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
    x4: &Quantized,
    x5: &Quantized,
) -> [[f32; 6]; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe {
                gemm_tq1_0_m8n6_avx2(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5)
            };
        }
    }
    gemm_tq1_0_m8n6_scalar(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5)
}

/// Specialized `8x7` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m8n7_avx2(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 8] {
    let mut out = [[0.0f32; 7]; 8];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x0).unwrap_or(0.0);
    out[7][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x1).unwrap_or(0.0);
    out[7][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x2).unwrap_or(0.0);
    out[7][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[6][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x3).unwrap_or(0.0);
    out[7][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    out[6][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x4).unwrap_or(0.0);
    out[7][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    out[5][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x5).unwrap_or(0.0);
    out[6][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x5).unwrap_or(0.0);
    out[7][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    out[3][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x6).unwrap_or(0.0);
    out[4][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x6).unwrap_or(0.0);
    out[5][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x6).unwrap_or(0.0);
    out[6][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x6).unwrap_or(0.0);
    out[7][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x6).unwrap_or(0.0);
    let _ = (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5, x6);
    out
}

/// Runtime-dispatched `8x7` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m8n7(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
) -> [[f32; 7]; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe {
                gemm_tq1_0_m8n7_avx2(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5, x6)
            };
        }
    }
    gemm_tq1_0_m8n7_scalar(w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5, x6)
}

/// Specialized `8x8` wide-batch AVX2+FMA+F16C register-tiled microkernel for `TQ1_0`.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_tq1_0_m8n8_avx2(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 8] {
    let mut out = [[0.0f32; 8]; 8];
    out[0][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x0).unwrap_or(0.0);
    out[1][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x0).unwrap_or(0.0);
    out[2][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x0).unwrap_or(0.0);
    out[3][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x0).unwrap_or(0.0);
    out[4][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x0).unwrap_or(0.0);
    out[5][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x0).unwrap_or(0.0);
    out[6][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x0).unwrap_or(0.0);
    out[7][0] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x0).unwrap_or(0.0);
    out[0][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x1).unwrap_or(0.0);
    out[1][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x1).unwrap_or(0.0);
    out[2][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x1).unwrap_or(0.0);
    out[3][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x1).unwrap_or(0.0);
    out[4][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x1).unwrap_or(0.0);
    out[5][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x1).unwrap_or(0.0);
    out[6][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x1).unwrap_or(0.0);
    out[7][1] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x1).unwrap_or(0.0);
    out[0][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x2).unwrap_or(0.0);
    out[1][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x2).unwrap_or(0.0);
    out[2][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x2).unwrap_or(0.0);
    out[3][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x2).unwrap_or(0.0);
    out[4][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x2).unwrap_or(0.0);
    out[5][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x2).unwrap_or(0.0);
    out[6][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x2).unwrap_or(0.0);
    out[7][2] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x2).unwrap_or(0.0);
    out[0][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x3).unwrap_or(0.0);
    out[1][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x3).unwrap_or(0.0);
    out[2][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x3).unwrap_or(0.0);
    out[3][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x3).unwrap_or(0.0);
    out[4][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x3).unwrap_or(0.0);
    out[5][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x3).unwrap_or(0.0);
    out[6][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x3).unwrap_or(0.0);
    out[7][3] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x3).unwrap_or(0.0);
    out[0][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x4).unwrap_or(0.0);
    out[1][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x4).unwrap_or(0.0);
    out[2][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x4).unwrap_or(0.0);
    out[3][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x4).unwrap_or(0.0);
    out[4][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x4).unwrap_or(0.0);
    out[5][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x4).unwrap_or(0.0);
    out[6][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x4).unwrap_or(0.0);
    out[7][4] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x4).unwrap_or(0.0);
    out[0][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x5).unwrap_or(0.0);
    out[1][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x5).unwrap_or(0.0);
    out[2][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x5).unwrap_or(0.0);
    out[3][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x5).unwrap_or(0.0);
    out[4][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x5).unwrap_or(0.0);
    out[5][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x5).unwrap_or(0.0);
    out[6][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x5).unwrap_or(0.0);
    out[7][5] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x5).unwrap_or(0.0);
    out[0][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x6).unwrap_or(0.0);
    out[1][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x6).unwrap_or(0.0);
    out[2][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x6).unwrap_or(0.0);
    out[3][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x6).unwrap_or(0.0);
    out[4][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x6).unwrap_or(0.0);
    out[5][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x6).unwrap_or(0.0);
    out[6][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x6).unwrap_or(0.0);
    out[7][6] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x6).unwrap_or(0.0);
    out[0][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w0, x7).unwrap_or(0.0);
    out[1][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w1, x7).unwrap_or(0.0);
    out[2][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w2, x7).unwrap_or(0.0);
    out[3][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w3, x7).unwrap_or(0.0);
    out[4][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w4, x7).unwrap_or(0.0);
    out[5][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w5, x7).unwrap_or(0.0);
    out[6][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w6, x7).unwrap_or(0.0);
    out[7][7] = crate::quant::idot::integer(GgmlType::TQ1_0, w7, x7).unwrap_or(0.0);
    let _ = (
        w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5, x6, x7,
    );
    out
}

/// Runtime-dispatched `8x8` wide-batch microkernel for `TQ1_0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq1_0_m8n8(
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
    x4: &Quantized,
    x5: &Quantized,
    x6: &Quantized,
    x7: &Quantized,
) -> [[f32; 8]; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::quant::avx2::available_fma_f16c() {
            return unsafe {
                gemm_tq1_0_m8n8_avx2(
                    w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5, x6, x7,
                )
            };
        }
    }
    gemm_tq1_0_m8n8_scalar(
        w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5, x6, x7,
    )
}

/// Execute wide-batch `8x8` tiled matrix-matrix multiplication for `TQ1_0`.
#[allow(clippy::identity_op)]
pub fn matmul_wide_tq1_0(
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
    while c + 8 <= n_batch {
        let x0 = &xs[c + 0];
        let x1 = &xs[c + 1];
        let x2 = &xs[c + 2];
        let x3 = &xs[c + 3];
        let x4 = &xs[c + 4];
        let x5 = &xs[c + 5];
        let x6 = &xs[c + 6];
        let x7 = &xs[c + 7];
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
            let tile = dispatch_tq1_0_m8n8(
                w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5, x6, x7,
            );
            for dr in 0..8 {
                for dc in 0..8 {
                    out[(c + dc) * rows + (r + dr)] = tile[dr][dc];
                }
            }
            r += 8;
        }
        while r < rows {
            let w0 = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
            let tile = dispatch_tq1_0_m1n8(w0, x0, x1, x2, x3, x4, x5, x6, x7);
            for dc in 0..8 {
                out[(c + dc) * rows + r] = tile[0][dc];
            }
            r += 1;
        }
        c += 8;
    }
    if c < n_batch {
        super::tq1_0::matmul_tiled_tq1_0(
            weight_bytes,
            rows,
            row_bytes,
            &xs[c..],
            &mut out[c * rows..],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row(seed: usize) -> Vec<u8> {
        let mut buf = vec![0u8; 54];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((seed * 31 + i * 17 + 7) % 199) as u8;
        }
        buf[52..54].copy_from_slice(&0x3400u16.to_le_bytes());
        buf
    }

    #[test]
    fn tq1_0_wide_m1_to_8_n5_to_8_match_idot_reference() {
        let rows: Vec<Vec<u8>> = (0..9).map(sample_row).collect();
        let xs: Vec<Quantized> = (0..9)
            .map(|c| {
                let floats: Vec<f32> = (0..256)
                    .map(|i| ((i + c * 5) as f32 * 0.13).cos())
                    .collect();
                let mut q = Quantized::with_capacity(256);
                q.fill(&floats);
                q
            })
            .collect();

        let tile_s = gemm_tq1_0_m4n8_scalar(
            &rows[0], &rows[1], &rows[2], &rows[3], &xs[0], &xs[1], &xs[2], &xs[3], &xs[4], &xs[5],
            &xs[6], &xs[7],
        );
        let tile = dispatch_tq1_0_m4n8(
            &rows[0], &rows[1], &rows[2], &rows[3], &xs[0], &xs[1], &xs[2], &xs[3], &xs[4], &xs[5],
            &xs[6], &xs[7],
        );
        for r in 0..4 {
            for c in 0..8 {
                let expected =
                    crate::quant::idot::integer(GgmlType::TQ1_0, &rows[r], &xs[c]).unwrap();
                assert!((tile_s[r][c] - expected).abs() < 5e-3);
                assert!((tile[r][c] - expected).abs() < 5e-3);
            }
        }

        let flat_w: Vec<u8> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let mut out = vec![0.0f32; 9 * 9];
        matmul_wide_tq1_0(&flat_w, 9, 54, &xs, &mut out);
        for c in 0..9 {
            for r in 0..9 {
                let expected =
                    crate::quant::idot::integer(GgmlType::TQ1_0, &rows[r], &xs[c]).unwrap();
                assert!((out[c * 9 + r] - expected).abs() < 5e-3);
            }
        }
    }
}
