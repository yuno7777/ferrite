use std::fs;
use std::path::PathBuf;

use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::quant::dequantize_to_vec;
use ferrite::quant::quantize::quantize_file;
use ferrite::synth::Builder;

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("ferrite_test_quant_{name}_{nanos}.gguf"));
    p
}

#[test]
fn quantize_model_f32_to_q8_0_and_q4_0_and_q4_k() {
    let src_path = temp_path("src_f32");
    let q8_path = temp_path("out_q8_0");
    let q4_path = temp_path("out_q4_0");
    let q4k_path = temp_path("out_q4_k");

    let num_elements = 256;
    let mut f32_vals = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        f32_vals.push(((i as f32) * 0.05).sin() * 2.5);
    }
    let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|x| x.to_le_bytes()).collect();

    let builder = Builder::new()
        .meta("general.architecture", Value::String("test_arch".into()))
        .meta("test.param", Value::U32(42))
        .tensor("weight", &[num_elements as u64], GgmlType::F32, f32_bytes);
    builder.write(&src_path).unwrap();

    // 1. Quantize to Q8_0
    quantize_file(&src_path, &q8_path, GgmlType::Q8_0).unwrap();
    let q8_gguf = Gguf::open(&q8_path).unwrap();
    assert_eq!(q8_gguf.meta_str("general.architecture"), Some("test_arch"));
    assert_eq!(q8_gguf.tensors.len(), 1);
    assert_eq!(q8_gguf.tensors[0].name, "weight");
    assert_eq!(q8_gguf.tensors[0].ty, GgmlType::Q8_0);

    let q8_bytes = q8_gguf.tensor_bytes(&q8_gguf.tensors[0]).unwrap();
    let q8_reconstructed = dequantize_to_vec(GgmlType::Q8_0, q8_bytes, num_elements).unwrap();
    for (orig, rec) in f32_vals.iter().zip(&q8_reconstructed) {
        assert!((orig - rec).abs() <= 0.05, "q8 diff: {orig} vs {rec}");
    }

    // 2. Quantize to Q4_0
    quantize_file(&src_path, &q4_path, GgmlType::Q4_0).unwrap();
    let q4_gguf = Gguf::open(&q4_path).unwrap();
    assert_eq!(q4_gguf.tensors[0].ty, GgmlType::Q4_0);

    let q4_bytes = q4_gguf.tensor_bytes(&q4_gguf.tensors[0]).unwrap();
    let q4_reconstructed = dequantize_to_vec(GgmlType::Q4_0, q4_bytes, num_elements).unwrap();
    for (orig, rec) in f32_vals.iter().zip(&q4_reconstructed) {
        assert!((orig - rec).abs() <= 0.35, "q4 diff: {orig} vs {rec}");
    }

    // 3. Quantize to Q4_K
    quantize_file(&src_path, &q4k_path, GgmlType::Q4_K).unwrap();
    let q4k_gguf = Gguf::open(&q4k_path).unwrap();
    assert_eq!(q4k_gguf.tensors[0].ty, GgmlType::Q4_K);

    let q4k_bytes = q4k_gguf.tensor_bytes(&q4k_gguf.tensors[0]).unwrap();
    let q4k_reconstructed = dequantize_to_vec(GgmlType::Q4_K, q4k_bytes, num_elements).unwrap();
    for (orig, rec) in f32_vals.iter().zip(&q4k_reconstructed) {
        assert!((orig - rec).abs() <= 0.5, "q4k diff: {orig} vs {rec}");
    }

    // Cleanup
    let _ = fs::remove_file(src_path);
    let _ = fs::remove_file(q8_path);
    let _ = fs::remove_file(q4_path);
    let _ = fs::remove_file(q4k_path);
}
