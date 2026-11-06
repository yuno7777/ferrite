//! Register-Tiled `(M=1..8) x (N=1..4)` SIMD Microkernel Matrix.
//!
//! Provides size-specialized scalar and AVX2+FMA+F16C GEMV and batched GEMM
//! microkernels across all 14 quantized weight formats (`Q8_0`, `Q4_0`, `Q4_1`,
//! `Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, `IQ4_XS`,
//! `TQ1_0`, and `TQ2_0`) plus unified runtime dispatch.

use crate::gguf::GgmlType;
use crate::quant::activation::Quantized;

pub mod iq4_nl;
pub mod iq4_xs;
pub mod q2_k;
pub mod q3_k;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_0;
pub mod q5_1;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod tq1_0;
pub mod tq2_0;

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
            q8_0::matmul_tiled_q8_0(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::Q4_0 => {
            q4_0::matmul_tiled_q4_0(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::Q4_1 => {
            q4_1::matmul_tiled_q4_1(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::Q5_0 => {
            q5_0::matmul_tiled_q5_0(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::Q5_1 => {
            q5_1::matmul_tiled_q5_1(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::Q2_K => {
            q2_k::matmul_tiled_q2_k(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::Q3_K => {
            q3_k::matmul_tiled_q3_k(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::Q4_K => {
            q4_k::matmul_tiled_q4_k(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::Q5_K => {
            q5_k::matmul_tiled_q5_k(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::Q6_K => {
            q6_k::matmul_tiled_q6_k(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::IQ4_NL => {
            iq4_nl::matmul_tiled_iq4_nl(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::IQ4_XS => {
            iq4_xs::matmul_tiled_iq4_xs(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::TQ1_0 => {
            tq1_0::matmul_tiled_tq1_0(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        GgmlType::TQ2_0 => {
            tq2_0::matmul_tiled_tq2_0(weight_bytes, rows, row_bytes, xs, out);
            true
        }
        _ => false,
    }
}
