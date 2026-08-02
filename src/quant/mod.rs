//! Turning stored blocks back into f32.
//!
//! GGUF weights are quantized in fixed-size blocks: 32 or 256 elements sharing
//! one or two scale factors. Expanding a block is cheap; the expensive part is
//! that it happens for every weight on every token, which is why the matmul
//! eventually fuses this into its inner loop instead of materializing rows.
//!
//! This module is the unfused reference. It is what correctness is measured
//! against.

use std::io::{Error, ErrorKind, Result};

use crate::gguf::GgmlType;

pub mod half;
pub mod k;
pub mod legacy;

fn unsupported(ty: GgmlType) -> Error {
    Error::new(
        ErrorKind::Unsupported,
        format!("{} is not implemented yet", ty.name()),
    )
}

fn bad(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

/// Expand `bytes` into `out`. `out.len()` sets the element count and must be a
/// whole number of blocks.
pub fn dequantize(ty: GgmlType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let (block, size) = ty.layout().ok_or_else(|| unsupported(ty))?;
    let elements = out.len() as u64;
    if elements % block != 0 {
        return Err(bad(format!(
            "{elements} elements is not a whole number of {} blocks",
            ty.name()
        )));
    }
    let expected = elements / block * size;
    if bytes.len() as u64 != expected {
        return Err(bad(format!(
            "{} needs {expected} bytes for {elements} elements, got {}",
            ty.name(),
            bytes.len()
        )));
    }

    match ty {
        GgmlType::F32 => {
            for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *slot = f32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)"));
            }
        }
        GgmlType::F16 => {
            for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *slot = half::f16(u16::from_le_bytes(
                    chunk.try_into().expect("chunks_exact(2)"),
                ));
            }
        }
        GgmlType::BF16 => {
            for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *slot = half::bf16(u16::from_le_bytes(
                    chunk.try_into().expect("chunks_exact(2)"),
                ));
            }
        }
        GgmlType::Q4_K => per_block(bytes, out, 256, 144, k::q4_k),
        GgmlType::Q4_0 => per_block(bytes, out, 32, 18, legacy::q4_0),
        GgmlType::Q4_1 => per_block(bytes, out, 32, 20, legacy::q4_1),
        GgmlType::Q8_0 => per_block(bytes, out, 32, 34, legacy::q8_0),
        other => return Err(unsupported(other)),
    }
    Ok(())
}

/// Walk stored blocks and expanded elements in lockstep. Lengths were checked
/// by the caller, so the zip cannot silently truncate.
fn per_block(
    bytes: &[u8],
    out: &mut [f32],
    block: usize,
    size: usize,
    expand: impl Fn(&[u8], &mut [f32]),
) {
    for (stored, slots) in bytes.chunks_exact(size).zip(out.chunks_exact_mut(block)) {
        expand(stored, slots);
    }
}

/// Allocating form, for one-off inspection.
pub fn dequantize_to_vec(ty: GgmlType, bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0; elements];
    dequantize(ty, bytes, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_passes_through() {
        let bytes: Vec<u8> = [1.0f32, -2.5, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = dequantize_to_vec(GgmlType::F32, &bytes, 3).unwrap();
        assert_eq!(out, vec![1.0, -2.5, 0.0]);
    }

    #[test]
    fn f16_and_bf16_expand() {
        let out = dequantize_to_vec(GgmlType::F16, &[0x00, 0x3C, 0x00, 0xC0], 2).unwrap();
        assert_eq!(out, vec![1.0, -2.0]);
        let out = dequantize_to_vec(GgmlType::BF16, &[0x80, 0x3F], 1).unwrap();
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn wrong_byte_count_is_an_error() {
        let err = dequantize_to_vec(GgmlType::F32, &[0, 0, 0], 1).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        let err = dequantize_to_vec(GgmlType::Other(99), &[], 0).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unsupported);
    }
}
