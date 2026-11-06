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

#[test]
fn fp8_e4m3_tensor_survives_the_file_and_blockwise_scaling() {
    let test_bytes: Vec<u8> = (0..128).map(|i| (i * 7) as u8).collect();
    let path = write(
        "f8_e4m3",
        &Builder::new().tensor("fp8_weight", &[128], GgmlType::F8_E4M3, test_bytes.clone()),
    );

    let from_file = dequantize_tensor(&path, "fp8_weight");
    let mut direct = vec![0.0f32; 128];
    quant::fp8::dequantize_e4m3(&test_bytes, &mut direct);

    assert_eq!(from_file.len(), 128);
    for (a, b) in from_file.iter().zip(&direct) {
        if a.is_nan() {
            assert!(b.is_nan());
        } else {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // Verify blockwise scaling round-trip
    let block_size = 64;
    let original: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin() * 25.0).collect();
    let mut quantized_bytes = vec![0u8; 128];
    let mut scales = vec![0.0f32; 2];
    quant::fp8::quantize_fp8_blockwise(&original, block_size, &mut quantized_bytes, &mut scales);

    let mut restored = vec![0.0f32; 128];
    quant::fp8::dequantize_fp8_blockwise(&quantized_bytes, &scales, block_size, &mut restored);

    for (&orig, &rec) in original.iter().zip(&restored) {
        assert!(
            (orig - rec).abs() < 1.0,
            "blockwise FP8 error too large: {orig} vs {rec}"
        );
    }

    fs::remove_file(path).ok();
}

#[test]
fn test_awq_and_gptq_group_dequantization() {
    let group_size = 32;
    let _num_words = 8; // 8 * 8 = 64 weights = 2 groups of 32
    let packed = vec![
        0x12345678u32,
        0x9abcdef0u32,
        0x13579bdfu32,
        0x2468ace0u32,
        0x01234567u32,
        0x89abcdefu32,
        0xfedcba98u32,
        0x76543210u32,
    ];
    let scales = vec![0.5f32, 0.25f32];
    let zeros = vec![2.0f32, 1.0f32];

    let mut out = vec![0.0f32; 64];
    ferrite::dequantize_awq_int4(&packed, &scales, Some(&zeros), group_size, &mut out)
        .expect("dequantize awq");

    assert_eq!(out.len(), 64);
    // Spot check first element:
    // packed[0] low nibble = 0x8 = 8.0
    // (8.0 - 2.0) * 0.5 = 3.0
    assert_eq!(out[0], 3.0);

    // Verify fused dot product agrees with unpacked dot product
    let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).cos()).collect();
    let expected: f32 = out.iter().zip(&x).map(|(w, xi)| w * xi).sum();
    let fused_awq = ferrite::dot_awq_int4(&packed, &scales, Some(&zeros), group_size, &x);
    assert!((fused_awq - expected).abs() < 1e-4);

    let mut gptq_out = vec![0.0f32; 64];
    ferrite::dequantize_gptq_int4(&packed, &scales, Some(&zeros), group_size, &mut gptq_out)
        .expect("dequantize gptq");
    assert_eq!(out, gptq_out);

    let fused_gptq = ferrite::dot_gptq_int4(&packed, &scales, Some(&zeros), group_size, &x);
    assert_eq!(fused_awq, fused_gptq);
}

#[test]
fn test_outlier_aware_quantization_w8a8_and_w4a16() {
    let rows = 8;
    let cols = 64;
    let mut weights = vec![0.0f32; rows * cols];

    // Bulk channels with small normal weights in [-0.05, 0.05]
    for r in 0..rows {
        for c in 0..cols {
            weights[r * cols + c] = ((r * 17 + c * 31) % 97) as f32 / 1000.0 - 0.048;
        }
    }

    // Two prominent outlier channels: 12 and 45
    for r in 0..rows {
        weights[r * cols + 12] = 8.5 + (r as f32) * 0.2;
        weights[r * cols + 45] = -12.0 - (r as f32) * 0.15;
    }

    // 1. W8A8 Mode
    let config_w8a8 = ferrite::OutlierConfig {
        mode: ferrite::OutlierQuantMode::W8A8,
        threshold_multiplier: 4.0,
        max_outlier_ratio: 0.10,
        min_magnitude: 1.0,
    };

    let outlier_w8a8 = ferrite::OutlierMatrix::from_weights(&weights, rows, cols, config_w8a8)
        .expect("quantize W8A8");

    assert_eq!(outlier_w8a8.outlier_channels, vec![12, 45]);
    assert!((outlier_w8a8.outlier_ratio() - 2.0 / 64.0).abs() < 1e-5);

    let mut dequant_w8a8 = vec![0.0f32; rows * cols];
    outlier_w8a8
        .dequantize(&mut dequant_w8a8)
        .expect("dequantize W8A8");

    // Outlier channels must be preserved in FP16 precision
    for r in 0..rows {
        let orig_12 = weights[r * cols + 12];
        let dq_12 = dequant_w8a8[r * cols + 12];
        assert!(
            (orig_12 - dq_12).abs() < 0.02,
            "outlier channel 12 preserved"
        );

        let orig_45 = weights[r * cols + 45];
        let dq_45 = dequant_w8a8[r * cols + 45];
        assert!(
            (orig_45 - dq_45).abs() < 0.02,
            "outlier channel 45 preserved"
        );
    }

    let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.1).sin()).collect();
    let mut expected_y = vec![0.0f32; rows];
    for r in 0..rows {
        for c in 0..cols {
            expected_y[r] += weights[r * cols + c] * x[c];
        }
    }

    let mut actual_y_w8a8 = vec![0.0f32; rows];
    outlier_w8a8
        .matvec(&x, &mut actual_y_w8a8)
        .expect("matvec W8A8");

    for r in 0..rows {
        let diff = (expected_y[r] - actual_y_w8a8[r]).abs();
        let rel = diff / expected_y[r].abs();
        assert!(rel < 0.05, "row {r} W8A8 matvec error {rel} too large");
    }

    // 2. W4A16 Mode
    let config_w4a16 = ferrite::OutlierConfig {
        mode: ferrite::OutlierQuantMode::W4A16,
        threshold_multiplier: 4.0,
        max_outlier_ratio: 0.10,
        min_magnitude: 1.0,
    };

    let outlier_w4a16 = ferrite::OutlierMatrix::from_weights(&weights, rows, cols, config_w4a16)
        .expect("quantize W4A16");

    assert_eq!(outlier_w4a16.outlier_channels, vec![12, 45]);

    let mut dequant_w4a16 = vec![0.0f32; rows * cols];
    outlier_w4a16
        .dequantize(&mut dequant_w4a16)
        .expect("dequantize W4A16");

    for r in 0..rows {
        let orig_12 = weights[r * cols + 12];
        let dq_12 = dequant_w4a16[r * cols + 12];
        assert!((orig_12 - dq_12).abs() < 0.02);
    }

    let mut actual_y_w4a16 = vec![0.0f32; rows];
    outlier_w4a16
        .matvec(&x, &mut actual_y_w4a16)
        .expect("matvec W4A16");

    for r in 0..rows {
        let diff = (expected_y[r] - actual_y_w4a16[r]).abs();
        let rel = diff / expected_y[r].abs();
        assert!(rel < 0.05, "row {r} W4A16 matvec error {rel} too large");
    }
}

#[test]
fn test_smoothquant_scale_migration_and_w8a8_matvec() {
    use ferrite::quant::smooth::{
        compute_smoothing_scales, compute_weight_channel_max, smooth_activations, smooth_weights,
        SmoothQuantConfig, SmoothQuantLinear,
    };

    let rows = 8;
    let cols = 16;

    // Synthetic weights
    let mut weights = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            weights[r * cols + c] = ((r as f32 * 0.7 + c as f32 * 0.3).sin() * 0.5) + 0.1;
        }
    }

    // Activations with massive channel outliers at indices 3 and 7
    let mut x = vec![1.0f32; cols];
    for (i, val) in x.iter_mut().enumerate() {
        *val = ((i as f32) * 0.4).cos();
    }
    x[3] = 64.0; // Outlier channel 3
    x[7] = -48.0; // Outlier channel 7

    // Activation channel maximum statistics (from calibration)
    let mut act_channel_max = vec![1.5f32; cols];
    act_channel_max[3] = 64.0;
    act_channel_max[7] = 48.0;

    // Reference unquantized f32 result
    let mut expected_y = vec![0.0f32; rows];
    for r in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += weights[r * cols + c] * x[c];
        }
        expected_y[r] = sum;
    }

    // 1. Verify scale computation
    let wt_channel_max = compute_weight_channel_max(&weights, rows, cols);
    let config = SmoothQuantConfig {
        alpha: 0.5,
        epsilon: 1e-5,
    };
    let scales = compute_smoothing_scales(&act_channel_max, &wt_channel_max, &config);

    // Outlier channels 3 and 7 must receive much higher scale factors
    assert!(scales[3] > 4.0);
    assert!(scales[7] > 4.0);
    assert!(scales[0] < 2.5);

    // 2. Verify mathematical equivalence of smoothed activations and smoothed weights
    let mut x_smooth = x.clone();
    smooth_activations(&mut x_smooth, &scales);
    // Outliers in activations should be significantly suppressed
    assert!(x_smooth[3].abs() < 16.0);
    assert!(x_smooth[7].abs() < 16.0);

    let mut w_smooth = weights.clone();
    smooth_weights(&mut w_smooth, rows, cols, &scales);

    // Unquantized product X_smooth * W_smooth must match original X * W exactly
    for r in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += w_smooth[r * cols + c] * x_smooth[c];
        }
        assert!(
            (sum - expected_y[r]).abs() < 1e-4,
            "mathematical equivalence broken: {sum} vs {}",
            expected_y[r]
        );
    }

    // 3. Quantize with SmoothQuantLinear (W8A8)
    let linear = SmoothQuantLinear::from_f32(&weights, rows, cols, &act_channel_max, config)
        .expect("from_f32");

    // Dequantize and check weight preservation
    let mut dequant_w = vec![0.0f32; rows * cols];
    linear.dequantize(&mut dequant_w).expect("dequantize");
    for i in 0..weights.len() {
        assert!((weights[i] - dequant_w[i]).abs() < 0.05);
    }

    // Evaluate W8A8 matvec
    let mut actual_y = vec![0.0f32; rows];
    linear.matvec(&x, &mut actual_y).expect("matvec");

    for r in 0..rows {
        let diff = (expected_y[r] - actual_y[r]).abs();
        let rel = diff / expected_y[r].abs().max(1e-3);
        assert!(rel < 0.08, "row {r} SmoothQuant error {rel} too large");
    }
}

#[test]
fn test_fp8_e5m2_conversion_and_vectorized_gemm() {
    use ferrite::quant::fp8::{f32_to_fp8_e4m3, f32_to_fp8_e5m2, fp8_e5m2_to_f32};
    use ferrite::{gemm_fp8_batched, gemm_fp8_fused, Fp8HardwareTuning};

    // 1. E5M2 format conversions
    assert_eq!(fp8_e5m2_to_f32(0x00), 0.0);
    assert_eq!(fp8_e5m2_to_f32(0x3C), 1.0);
    assert_eq!(fp8_e5m2_to_f32(0xBC), -1.0);
    assert!(fp8_e5m2_to_f32(0x7C).is_infinite());
    assert!(fp8_e5m2_to_f32(0x7F).is_nan());

    let test_vals = [0.0, 1.0, -1.0, 2.0, -4.0, 16.0, 1024.0];
    for &val in &test_vals {
        let byte = f32_to_fp8_e5m2(val);
        let dec = fp8_e5m2_to_f32(byte);
        assert_eq!(val, dec, "exact E5M2 roundtrip failed for {val}");
    }

    // 2. Vectorized GEMM vs reference float matrix-vector product
    let rows = 4;
    let cols = 64;
    let block_size = 32;

    // Create synthetic weight and activations
    let mut w_f32 = vec![0.0f32; rows * cols];
    let mut w_e4m3 = vec![0u8; rows * cols];
    let mut w_e5m2 = vec![0u8; rows * cols];

    for (i, w) in w_f32.iter_mut().enumerate() {
        *w = ((i as f32 * 0.17).sin() * 4.0).round() * 0.5;
        w_e4m3[i] = f32_to_fp8_e4m3(*w);
        w_e5m2[i] = f32_to_fp8_e5m2(*w);
    }

    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.23).cos()).collect();
    let scales = vec![1.0f32; (rows * cols) / block_size];

    // Reference float matrix-vector
    let mut expected_e4m3 = vec![0.0f32; rows];
    let mut expected_e5m2 = vec![0.0f32; rows];

    for r in 0..rows {
        let mut sum4 = 0.0f32;
        let mut sum5 = 0.0f32;
        for (c, &x_val) in x.iter().enumerate().take(cols) {
            let idx = r * cols + c;
            sum4 += ferrite::quant::fp8::fp8_e4m3_to_f32(w_e4m3[idx]) * x_val;
            sum5 += fp8_e5m2_to_f32(w_e5m2[idx]) * x_val;
        }
        expected_e4m3[r] = sum4;
        expected_e5m2[r] = sum5;
    }

    // Fused FP8 GEMM (E4M3)
    let mut actual_e4m3 = vec![0.0f32; rows];
    gemm_fp8_fused(
        &w_e4m3,
        Some(&scales),
        block_size,
        false,
        &x,
        &mut actual_e4m3,
        rows,
        cols,
    );
    for r in 0..rows {
        assert!(
            (actual_e4m3[r] - expected_e4m3[r]).abs() < 1e-4,
            "E4M3 fused GEMM mismatch at row {r}: {} vs {}",
            actual_e4m3[r],
            expected_e4m3[r]
        );
    }

    // Fused FP8 GEMM (E5M2)
    let mut actual_e5m2 = vec![0.0f32; rows];
    gemm_fp8_fused(
        &w_e5m2,
        Some(&scales),
        block_size,
        true,
        &x,
        &mut actual_e5m2,
        rows,
        cols,
    );
    for r in 0..rows {
        assert!(
            (actual_e5m2[r] - expected_e5m2[r]).abs() < 1e-4,
            "E5M2 fused GEMM mismatch at row {r}: {} vs {}",
            actual_e5m2[r],
            expected_e5m2[r]
        );
    }

    // 3. Batched FP8 GEMM with auto-tuning
    let batch = 3;
    let mut x_batch = Vec::with_capacity(batch * cols);
    for b in 0..batch {
        for c in 0..cols {
            x_batch.push(((b * cols + c) as f32 * 0.1).sin());
        }
    }

    let mut out_batch = vec![0.0f32; batch * rows];
    let tuning = Fp8HardwareTuning::auto_detect();
    gemm_fp8_batched(
        &w_e4m3,
        Some(&scales),
        block_size,
        false,
        &x_batch,
        &mut out_batch,
        batch,
        rows,
        cols,
        Some(&tuning),
    );

    // Verify each batch output matches individual matvec
    for b in 0..batch {
        let mut single_out = vec![0.0f32; rows];
        let x_tok = &x_batch[b * cols..(b + 1) * cols];
        gemm_fp8_fused(
            &w_e4m3,
            Some(&scales),
            block_size,
            false,
            x_tok,
            &mut single_out,
            rows,
            cols,
        );
        let batch_slice = &out_batch[b * rows..(b + 1) * rows];
        assert_eq!(batch_slice, single_out.as_slice());
    }
}
