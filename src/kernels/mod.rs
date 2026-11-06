//! Register-Tiled `(M=1..8) x (N=1..8)` SIMD Microkernel Matrix.
//!
//! Provides 896 size-specialized scalar and AVX2+FMA+F16C GEMV and batched GEMM
//! microkernels across all 14 quantized weight formats (`Q8_0`, `Q4_0`, `Q4_1`,
//! `Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, `IQ4_XS`,
//! `TQ1_0`, and `TQ2_0`) plus unified runtime dispatch.

use crate::gguf::GgmlType;
use crate::quant::activation::Quantized;

pub mod iq4_nl;
pub mod iq4_nl_wide;
pub mod iq4_xs;
pub mod iq4_xs_wide;
pub mod q2_k;
pub mod q2_k_wide;
pub mod q3_k;
pub mod q3_k_wide;
pub mod q4_0;
pub mod q4_0_wide;
pub mod q4_1;
pub mod q4_1_wide;
pub mod q4_k;
pub mod q4_k_wide;
pub mod q5_0;
pub mod q5_0_wide;
pub mod q5_1;
pub mod q5_1_wide;
pub mod q5_k;
pub mod q5_k_wide;
pub mod q6_k;
pub mod q6_k_wide;
pub mod q8_0;
pub mod q8_0_wide;
pub mod tq1_0;
pub mod tq1_0_wide;
pub mod tq2_0;
pub mod tq2_0_wide;

/// Unified tiled GEMM dispatcher across all 14 quantized GGUF weight formats.
pub fn matmul_quantized_tiled(
    ty: GgmlType,
    weight_bytes: &[u8],
    rows: usize,
    row_bytes: usize,
    xs: &[Quantized],
    out: &mut [f32],
) -> bool {
    match ty {
        GgmlType::Q8_0 => {
            if xs.len() >= 8 {
                q8_0_wide::matmul_wide_q8_0(weight_bytes, rows, row_bytes, xs, out);
            } else {
                q8_0::matmul_tiled_q8_0(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::Q4_0 => {
            if xs.len() >= 8 {
                q4_0_wide::matmul_wide_q4_0(weight_bytes, rows, row_bytes, xs, out);
            } else {
                q4_0::matmul_tiled_q4_0(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::Q4_1 => {
            if xs.len() >= 8 {
                q4_1_wide::matmul_wide_q4_1(weight_bytes, rows, row_bytes, xs, out);
            } else {
                q4_1::matmul_tiled_q4_1(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::Q5_0 => {
            if xs.len() >= 8 {
                q5_0_wide::matmul_wide_q5_0(weight_bytes, rows, row_bytes, xs, out);
            } else {
                q5_0::matmul_tiled_q5_0(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::Q5_1 => {
            if xs.len() >= 8 {
                q5_1_wide::matmul_wide_q5_1(weight_bytes, rows, row_bytes, xs, out);
            } else {
                q5_1::matmul_tiled_q5_1(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::Q2_K => {
            if xs.len() >= 8 {
                q2_k_wide::matmul_wide_q2_k(weight_bytes, rows, row_bytes, xs, out);
            } else {
                q2_k::matmul_tiled_q2_k(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::Q3_K => {
            if xs.len() >= 8 {
                q3_k_wide::matmul_wide_q3_k(weight_bytes, rows, row_bytes, xs, out);
            } else {
                q3_k::matmul_tiled_q3_k(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::Q4_K => {
            if xs.len() >= 8 {
                q4_k_wide::matmul_wide_q4_k(weight_bytes, rows, row_bytes, xs, out);
            } else {
                q4_k::matmul_tiled_q4_k(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::Q5_K => {
            if xs.len() >= 8 {
                q5_k_wide::matmul_wide_q5_k(weight_bytes, rows, row_bytes, xs, out);
            } else {
                q5_k::matmul_tiled_q5_k(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::Q6_K => {
            if xs.len() >= 8 {
                q6_k_wide::matmul_wide_q6_k(weight_bytes, rows, row_bytes, xs, out);
            } else {
                q6_k::matmul_tiled_q6_k(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::IQ4_NL => {
            if xs.len() >= 8 {
                iq4_nl_wide::matmul_wide_iq4_nl(weight_bytes, rows, row_bytes, xs, out);
            } else {
                iq4_nl::matmul_tiled_iq4_nl(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::IQ4_XS => {
            if xs.len() >= 8 {
                iq4_xs_wide::matmul_wide_iq4_xs(weight_bytes, rows, row_bytes, xs, out);
            } else {
                iq4_xs::matmul_tiled_iq4_xs(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::TQ1_0 => {
            if xs.len() >= 8 {
                tq1_0_wide::matmul_wide_tq1_0(weight_bytes, rows, row_bytes, xs, out);
            } else {
                tq1_0::matmul_tiled_tq1_0(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        GgmlType::TQ2_0 => {
            if xs.len() >= 8 {
                tq2_0_wide::matmul_wide_tq2_0(weight_bytes, rows, row_bytes, xs, out);
            } else {
                tq2_0::matmul_tiled_tq2_0(weight_bytes, rows, row_bytes, xs, out);
            }
            true
        }
        _ => false,
    }
}
