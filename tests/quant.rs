//! Dequantization through the whole path: GGUF file, mapped tensor bytes, f32.
//!
//! The unit tests in `src/quant` pin down block layouts in isolation. These
//! check that a tensor written to a file comes back out with the same numbers,
//! which is where an offset or alignment mistake would show up.

use std::fs;
use std::path::{Path, PathBuf};

use ferrite::gguf::{GgmlType, Gguf};
use ferrite::quant;
use ferrite::synth::Builder;

fn write(name: &str, builder: &Builder) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ferrite_quant_{name}.gguf"));
    builder.write(&path).expect("write fixture");
    path
}

fn dequantize_tensor(path: &Path, name: &str) -> Vec<f32> {
    let model = Gguf::open(path).expect("open");
    let tensor = model.tensor(name).expect("tensor present");
    let bytes = model.tensor_bytes(tensor).expect("tensor bytes");
    quant::dequantize_to_vec(tensor.ty, bytes, tensor.elements() as usize).expect("dequantize")
}

/// Quantize to Q8_0 the way the reference does: one f16 scale per 32 values,
/// chosen so the largest magnitude maps to 127.
fn quantize_q8_0(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::new();
    let mut scales = Vec::new();
    for block in values.chunks(32) {
        let max = block.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let scale = max / 127.0;
        // Round-trip the scale through f16, since that is what gets stored.
        let stored = half_bits(scale);
        let scale = f16_value(stored);
        scales.push(scale);
        bytes.extend_from_slice(&stored.to_le_bytes());
        for value in block {
            let q = if scale == 0.0 {
                0
            } else {
                (value / scale).round().clamp(-127.0, 127.0) as i8
            };
            bytes.push(q as u8);
        }
    }
    (bytes, scales)
}

/// f32 to the nearest f16 bit pattern. Only handles the normal range, which is
/// all a scale ever needs.
fn half_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    assert!(
        (1..=30).contains(&exponent),
        "test helper only covers normal f16 values"
    );
    let mantissa = ((bits >> 13) & 0x3FF) as u16;
    sign | ((exponent as u16) << 10) | mantissa
}

fn f16_value(bits: u16) -> f32 {
    quant::half::f16(bits)
}

#[test]
fn f32_tensor_survives_the_file() {
    let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.25 - 8.0).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let path = write(
        "f32",
        &Builder::new().tensor("w", &[64], GgmlType::F32, bytes),
    );

    assert_eq!(dequantize_tensor(&path, "w"), values);
    fs::remove_file(path).ok();
}

#[test]
fn q8_0_round_trips_within_half_a_step() {
    let values: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let (bytes, scales) = quantize_q8_0(&values);
    let path = write(
        "q8_0",
        &Builder::new().tensor("w", &[128], GgmlType::Q8_0, bytes),
    );

    let out = dequantize_tensor(&path, "w");
    assert_eq!(out.len(), values.len());
    for (index, (original, restored)) in values.iter().zip(&out).enumerate() {
        // Nothing can be off by more than half a quantization step.
        let tolerance = scales[index / 32] / 2.0 + f32::EPSILON;
        assert!(
            (original - restored).abs() <= tolerance,
            "element {index}: {original} -> {restored}, tolerance {tolerance}"
        );
    }
    fs::remove_file(path).ok();
}

#[test]
fn k_quant_tensor_matches_a_direct_block_expansion() {
    // One Q4_K super-block with recognizable contents.
    let mut block = Vec::with_capacity(144);
    block.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    block.extend_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5
    block.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    block.extend((0..128).map(|i| (i % 251) as u8));

    let path = write(
        "q4_k",
        &Builder::new().tensor("w", &[256], GgmlType::Q4_K, block.clone()),
    );

    let from_file = dequantize_tensor(&path, "w");
    let mut direct = [0.0f32; 256];
    quant::k::q4_k(&block, &mut direct);

    assert_eq!(from_file, direct.to_vec());
    // Sanity: a real super-block is not all zeros.
    assert!(direct.iter().any(|v| *v != 0.0));
    fs::remove_file(path).ok();
}

#[test]
fn unsupported_types_are_reported_not_guessed() {
    let err = quant::dequantize_to_vec(GgmlType::Q2_K, &[0; 84], 256).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    assert!(err.to_string().contains("Q2_K"), "{err}");
}
