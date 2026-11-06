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

#[test]
fn test_unstructured_magnitude_pruning_inplace() {
    use ferrite::prune::{prune_unstructured_slice, ModelPruner};

    // 1. Slice unit test
    let mut data = vec![
        -10.0f32, 0.1, -0.05, 5.0, 0.01, 20.0, -0.02, 12.0, 0.03, -15.0,
    ];
    let stats = prune_unstructured_slice(&mut data, 0.4); // 40% sparsity -> 4 smallest elements pruned
    assert_eq!(stats.total_parameters, 10);
    assert_eq!(stats.pruned_parameters, 4);
    assert!((stats.sparsity - 0.4).abs() < 1e-4);

    // Elements 0.01, -0.02, 0.03, -0.05 should be 0.0
    assert_eq!(data[1], 0.1);
    assert_eq!(data[2], 0.0);
    assert_eq!(data[4], 0.0);
    assert_eq!(data[6], 0.0);
    assert_eq!(data[8], 0.0);

    // 2. In-place GGUF file test
    let path = temp_path("unstructured_pruning");
    let original = [10.0f32, 0.2, 0.1, -15.0, 0.05, 30.0];
    let bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();

    let builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .tensor("weights_f32", &[6], GgmlType::F32, bytes);
    builder.write(&path).expect("write model");

    let stats = ModelPruner::prune_unstructured_inplace(&path, 0.5, Some(&["weights_f32"]))
        .expect("prune inplace");
    assert_eq!(stats.pruned_parameters, 3);

    // Reopen and verify file content on disk
    let gguf = Gguf::open(&path).expect("open");
    let t = gguf.tensor("weights_f32").unwrap();
    let b = gguf.tensor_bytes(t).unwrap();
    let read_back: Vec<f32> = b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // 3 smallest (|0.05|, |0.1|, |0.2|) must be zeroed
    assert_eq!(read_back[0], 10.0);
    assert_eq!(read_back[1], 0.0);
    assert_eq!(read_back[2], 0.0);
    assert_eq!(read_back[3], -15.0);
    assert_eq!(read_back[4], 0.0);
    assert_eq!(read_back[5], 30.0);

    fs::remove_file(path).ok();
}

#[test]
fn test_structured_head_and_width_pruning_inplace() {
    use ferrite::prune::ModelPruner;

    let path = temp_path("structured_pruning");

    // 4 heads, each head has 2 rows, cols = 4 -> rows = 8, cols = 4 (dims = [4, 8])
    // Total = 32 elements
    let mut weights = [0.0f32; 32];

    // Head 0 (rows 0..2): norm ~ 2.0
    for r in 0..2 {
        for c in 0..4 {
            weights[r * 4 + c] = 1.0;
        }
    }
    // Head 1 (rows 2..4): very small weights (norm ~ 0.02) -> should be pruned!
    for r in 2..4 {
        for c in 0..4 {
            weights[r * 4 + c] = 0.01;
        }
    }
    // Head 2 (rows 4..6): large weights (norm ~ 8.0)
    for r in 4..6 {
        for c in 0..4 {
            weights[r * 4 + c] = 4.0;
        }
    }
    // Head 3 (rows 6..8): norm ~ 4.0
    for r in 6..8 {
        for c in 0..4 {
            weights[r * 4 + c] = 2.0;
        }
    }

    let bytes: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();

    let builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .tensor("attn_q", &[4, 8], GgmlType::F32, bytes.clone())
        .tensor("mlp_down", &[4, 8], GgmlType::F32, bytes);
    builder.write(&path).expect("write model");

    // 1. Prune 1 attention head in attn_q
    let (stats, pruned_heads) =
        ModelPruner::prune_attention_heads_inplace(&path, "attn_q", 4, 1).expect("prune heads");
    assert_eq!(pruned_heads, vec![1]); // Head 1 had smallest norm
    assert_eq!(stats.pruned_parameters, 8); // 2 rows * 4 cols

    // Reopen and verify Head 1 is zeroed while other heads remain intact
    let gguf = Gguf::open(&path).expect("reopen");
    let t_q = gguf.tensor("attn_q").unwrap();
    let b_q = gguf.tensor_bytes(t_q).unwrap();
    let q_vals: Vec<f32> = b_q
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // Rows 0..2 (Head 0) intact
    assert_eq!(q_vals[0], 1.0);
    // Rows 2..4 (Head 1) zeroed
    for r in 2..4 {
        for c in 0..4 {
            assert_eq!(q_vals[r * 4 + c], 0.0);
        }
    }
    // Rows 4..6 (Head 2) intact
    assert_eq!(q_vals[4 * 4], 4.0);

    // 2. Prune width in mlp_down (25% of columns = 1 column out of 4)
    let width_stats =
        ModelPruner::prune_width_inplace(&path, "mlp_down", 0.25).expect("prune width");
    assert_eq!(width_stats.pruned_parameters, 8); // 1 col * 8 rows

    fs::remove_file(path).ok();
}
