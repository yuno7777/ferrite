use std::fs;
use std::path::PathBuf;

use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::synth::Builder;
use ferrite::GgufEditor;

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("ferrite_test_edit_{name}_{nanos}.gguf"));
    p
}

#[test]
fn test_zero_copy_gguf_tensor_in_place_edit() {
    let path = temp_path("tensor_edit");

    // 1. Build a synthetic GGUF model with F32 and Q8_0 tensors
    let original_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let f32_bytes: Vec<u8> = original_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

    // 32 elements of Q8_0 = 34 bytes (2 bytes scale + 32 bytes i8)
    let mut q8_bytes = vec![0u8; 34];
    q8_bytes[0] = 0x00;
    q8_bytes[1] = 0x3c; // f16 1.0
    for (i, slot) in q8_bytes[2..34].iter_mut().enumerate() {
        *slot = i as u8;
    }

    let builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .tensor("test_f32", &[2, 3], GgmlType::F32, f32_bytes)
        .tensor("test_q8", &[32], GgmlType::Q8_0, q8_bytes);
    builder.write(&path).expect("write model");

    // 2. Open with GgufEditor
    let mut editor = GgufEditor::open(&path).expect("open editor");
    assert_eq!(editor.tensors.len(), 2);
    assert!(editor.tensor("test_f32").is_some());
    assert!(editor.tensor("test_q8").is_some());

    // 3. Verify initial reading
    let read_back_f32 = editor.read_tensor_f32("test_f32").expect("read f32");
    assert_eq!(read_back_f32, original_f32);

    // 4. In-place modify F32 tensor
    let modified_f32 = vec![10.5f32, 20.25, 30.125, 40.0, 50.5, 60.75];
    editor
        .write_tensor_f32("test_f32", &modified_f32)
        .expect("write f32");

    // 5. In-place update Q8_0 scale factor
    editor
        .update_q8_0_scale_in_place("test_q8", 0, 2.5)
        .expect("update q8 scale");

    // 6. Flush changes to disk
    editor.flush().expect("flush editor");
    drop(editor);

    // 7. Verify via read-only Gguf::open that file on disk is updated in-place
    let reopened = Gguf::open(&path).expect("reopen gguf");
    let t_f32 = reopened.tensor("test_f32").unwrap();
    let bytes_f32 = reopened.tensor_bytes(t_f32).unwrap();
    let vals: Vec<f32> = bytes_f32
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals, modified_f32);

    let t_q8 = reopened.tensor("test_q8").unwrap();
    let bytes_q8 = reopened.tensor_bytes(t_q8).unwrap();
    let scale_f16 = ferrite::quant::half::read_f16(&bytes_q8[0..2]);
    assert!((scale_f16 - 2.5).abs() < 1e-3);

    fs::remove_file(path).ok();
}

#[test]
fn test_in_place_lora_merging() {
    let path = temp_path("lora_merge");

    // Shape: rows = 2, cols = 3 -> GGUF dims = [3, 2]
    // Initial weights W:
    // [1.0, 2.0, 3.0]
    // [4.0, 5.0, 6.0]
    let initial_w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_bytes: Vec<u8> = initial_w.iter().flat_map(|v| v.to_le_bytes()).collect();

    let builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .tensor("layer.weight", &[3, 2], GgmlType::F32, w_bytes);
    builder.write(&path).expect("write model");

    // LoRA with rank = 1:
    // A: shape [1, 3] = [0.5, 1.0, -0.5]
    // B: shape [2, 1] = [2.0, 4.0]
    // scale: 0.5
    // Delta = B * A:
    // Row 0: 2.0 * [0.5, 1.0, -0.5] = [1.0, 2.0, -1.0]
    // Row 1: 4.0 * [0.5, 1.0, -0.5] = [2.0, 4.0, -2.0]
    // scaled Delta (* 0.5):
    // Row 0: [0.5, 1.0, -0.5]
    // Row 1: [1.0, 2.0, -1.0]
    // Expected W_new:
    // [1.5, 3.0, 2.5]
    // [5.0, 7.0, 5.0]
    let lora_a = vec![0.5f32, 1.0, -0.5];
    let lora_b = vec![2.0f32, 4.0];
    let rank = 1;
    let scale = 0.5f32;

    let mut editor = GgufEditor::open(&path).expect("open editor");
    editor
        .apply_lora_in_place("layer.weight", &lora_a, &lora_b, rank, scale)
        .expect("apply lora");
    editor.flush().expect("flush");
    drop(editor);

    let reopened = Gguf::open(&path).expect("reopen");
    let t = reopened.tensor("layer.weight").unwrap();
    let bytes = reopened.tensor_bytes(t).unwrap();
    let merged: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let expected = vec![1.5f32, 3.0, 2.5, 5.0, 7.0, 5.0];
    assert_eq!(merged, expected);

    fs::remove_file(path).ok();
}
