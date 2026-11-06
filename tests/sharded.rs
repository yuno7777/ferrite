use std::fs;
use std::path::PathBuf;

use ferrite::gguf::{discover_shards, GgmlType, Gguf, Value};
use ferrite::model::{Model, State};
use ferrite::synth::Builder;

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("ferrite_test_sharded_{name}_{nanos}.gguf"));
    p
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
        .collect()
}

#[test]
fn test_open_sharded_explicit_paths() {
    let path_0 = temp_path("exp_part0");
    let path_1 = temp_path("exp_part1");

    let t0_data = pattern(32 * 4, 11);
    let t1_data = pattern(32 * 4, 22);
    let t2_data = {
        let mut d = vec![0u8; 18];
        d[0] = 0x00;
        d[1] = 0x3c; // f16 1.0
        d
    };

    let shard_0 = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("general.name", Value::String("test_sharded".into()))
        .meta("split.no", Value::U16(0))
        .meta("split.count", Value::U16(2))
        .tensor("tensor_0", &[32], GgmlType::F32, t0_data.clone());
    shard_0.write(&path_0).unwrap();

    let shard_1 = Builder::new()
        .meta("split.no", Value::U16(1))
        .meta("split.count", Value::U16(2))
        .tensor("tensor_1", &[32], GgmlType::F32, t1_data.clone())
        .tensor("tensor_2", &[32], GgmlType::Q4_0, t2_data.clone());
    shard_1.write(&path_1).unwrap();

    let model = Gguf::open_sharded(&[&path_0, &path_1]).expect("open_sharded");
    assert_eq!(model.shard_count(), 2);
    assert_eq!(model.arch(), Some("llama"));
    assert_eq!(model.meta_str("general.name"), Some("test_sharded"));
    assert_eq!(model.tensors.len(), 3);

    let ti0 = model.tensor("tensor_0").expect("tensor_0");
    assert_eq!(ti0.shard_index, 0);
    assert_eq!(model.tensor_bytes(ti0).unwrap(), t0_data.as_slice());

    let ti1 = model.tensor("tensor_1").expect("tensor_1");
    assert_eq!(ti1.shard_index, 1);
    assert_eq!(model.tensor_bytes(ti1).unwrap(), t1_data.as_slice());

    let ti2 = model.tensor("tensor_2").expect("tensor_2");
    assert_eq!(ti2.shard_index, 1);
    assert_eq!(model.tensor_bytes(ti2).unwrap(), t2_data.as_slice());

    let len0 = fs::metadata(&path_0).unwrap().len();
    let len1 = fs::metadata(&path_1).unwrap().len();
    assert_eq!(model.file_len(), len0 + len1);

    let _ = fs::remove_file(path_0);
    let _ = fs::remove_file(path_1);
}

#[test]
fn test_open_sharded_auto_discovery() {
    let mut dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    dir.push(format!("ferrite_shards_{nanos}"));
    fs::create_dir_all(&dir).unwrap();

    let path_1 = dir.join("model-00001-of-00002.gguf");
    let path_2 = dir.join("model-00002-of-00002.gguf");

    let t1_data = pattern(32 * 4, 101);
    let t2_data = pattern(32 * 4, 102);

    let shard_1 = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("split.no", Value::U16(0))
        .meta("split.count", Value::U16(2))
        .tensor("layer_1.weight", &[32], GgmlType::F32, t1_data.clone());
    shard_1.write(&path_1).unwrap();

    let shard_2 = Builder::new()
        .meta("split.no", Value::U16(1))
        .meta("split.count", Value::U16(2))
        .tensor("layer_2.weight", &[32], GgmlType::F32, t2_data.clone());
    shard_2.write(&path_2).unwrap();

    // Verify discover_shards on path_1
    let discovered = discover_shards(&path_1).expect("discover_shards on shard 1");
    assert_eq!(discovered.len(), 2);
    assert_eq!(discovered[0], path_1);
    assert_eq!(discovered[1], path_2);

    // Verify discover_shards on path_2 also resolves both shards with shard 1 first
    let discovered_from_2 = discover_shards(&path_2).expect("discover_shards on shard 2");
    assert_eq!(discovered_from_2.len(), 2);
    assert_eq!(discovered_from_2[0], path_1);
    assert_eq!(discovered_from_2[1], path_2);

    // Opening shard 1 auto-loads both shards
    let model1 = Gguf::open(&path_1).expect("open auto");
    assert_eq!(model1.shard_count(), 2);
    assert_eq!(model1.tensors.len(), 2);
    assert!(model1.tensor("layer_1.weight").is_some());
    assert!(model1.tensor("layer_2.weight").is_some());

    // Opening shard 2 also auto-loads both shards with shard 1 as primary
    let model2 = Gguf::open(&path_2).expect("open from shard 2");
    assert_eq!(model2.shard_count(), 2);
    assert_eq!(model2.arch(), Some("llama"));
    assert_eq!(model2.tensors.len(), 2);

    let _ = fs::remove_file(path_1);
    let _ = fs::remove_file(path_2);
    let _ = fs::remove_dir(dir);
}

#[test]
fn test_open_sharded_duplicate_tensor_rejected() {
    let path_0 = temp_path("dup_part0");
    let path_1 = temp_path("dup_part1");

    let t_data = pattern(32 * 4, 1);

    let shard_0 = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .tensor("shared_weight", &[32], GgmlType::F32, t_data.clone());
    shard_0.write(&path_0).unwrap();

    let shard_1 = Builder::new().tensor("shared_weight", &[32], GgmlType::F32, t_data.clone());
    shard_1.write(&path_1).unwrap();

    let err = Gguf::open_sharded(&[&path_0, &path_1]).unwrap_err();
    assert!(
        err.to_string()
            .contains("duplicate tensor name shared_weight"),
        "unexpected error: {err}"
    );

    let _ = fs::remove_file(path_0);
    let _ = fs::remove_file(path_1);
}

#[test]
fn test_sharded_model_inference_matches_single_file() {
    let single_path = temp_path("single_model");
    let shard0_path = temp_path("sharded_model_0");
    let shard1_path = temp_path("sharded_model_1");

    let dim = 32u64;
    let vocab = 16u64;
    let embed_bytes = pattern((dim * vocab * 4) as usize, 10);
    let q_bytes = pattern((dim * dim * 4) as usize, 20);
    let k_bytes = pattern((dim * dim * 4) as usize, 30);
    let v_bytes = pattern((dim * dim * 4) as usize, 40);
    let out_bytes = pattern((dim * dim * 4) as usize, 50);
    let ffn_gate = pattern((dim * dim * 4) as usize, 60);
    let ffn_up = pattern((dim * dim * 4) as usize, 70);
    let ffn_down = pattern((dim * dim * 4) as usize, 80);
    let attn_norm = pattern((dim * 4) as usize, 90);
    let ffn_norm = pattern((dim * 4) as usize, 100);
    let out_norm = pattern((dim * 4) as usize, 110);

    // Build complete single-file model
    let single_builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(1))
        .meta("llama.embedding_length", Value::U32(dim as u32))
        .meta("llama.feed_forward_length", Value::U32(dim as u32))
        .meta("llama.attention.head_count", Value::U32(1))
        .meta("llama.attention.head_count_kv", Value::U32(1))
        .meta("llama.context_length", Value::U32(128))
        .meta("llama.rope.freq_base", Value::F32(10000.0))
        .tensor(
            "token_embd.weight",
            &[dim, vocab],
            GgmlType::F32,
            embed_bytes.clone(),
        )
        .tensor(
            "blk.0.attn_q.weight",
            &[dim, dim],
            GgmlType::F32,
            q_bytes.clone(),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[dim, dim],
            GgmlType::F32,
            k_bytes.clone(),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[dim, dim],
            GgmlType::F32,
            v_bytes.clone(),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[dim, dim],
            GgmlType::F32,
            out_bytes.clone(),
        )
        .tensor(
            "blk.0.ffn_gate.weight",
            &[dim, dim],
            GgmlType::F32,
            ffn_gate.clone(),
        )
        .tensor(
            "blk.0.ffn_up.weight",
            &[dim, dim],
            GgmlType::F32,
            ffn_up.clone(),
        )
        .tensor(
            "blk.0.ffn_down.weight",
            &[dim, dim],
            GgmlType::F32,
            ffn_down.clone(),
        )
        .tensor(
            "blk.0.attn_norm.weight",
            &[dim],
            GgmlType::F32,
            attn_norm.clone(),
        )
        .tensor(
            "blk.0.ffn_norm.weight",
            &[dim],
            GgmlType::F32,
            ffn_norm.clone(),
        )
        .tensor(
            "output_norm.weight",
            &[dim],
            GgmlType::F32,
            out_norm.clone(),
        );
    single_builder.write(&single_path).unwrap();

    // Build sharded model: shard 0 has metadata, embedding, attention weights
    let shard0_builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(1))
        .meta("llama.embedding_length", Value::U32(dim as u32))
        .meta("llama.feed_forward_length", Value::U32(dim as u32))
        .meta("llama.attention.head_count", Value::U32(1))
        .meta("llama.attention.head_count_kv", Value::U32(1))
        .meta("llama.context_length", Value::U32(128))
        .meta("llama.rope.freq_base", Value::F32(10000.0))
        .tensor(
            "token_embd.weight",
            &[dim, vocab],
            GgmlType::F32,
            embed_bytes,
        )
        .tensor("blk.0.attn_q.weight", &[dim, dim], GgmlType::F32, q_bytes)
        .tensor("blk.0.attn_k.weight", &[dim, dim], GgmlType::F32, k_bytes)
        .tensor("blk.0.attn_v.weight", &[dim, dim], GgmlType::F32, v_bytes)
        .tensor(
            "blk.0.attn_output.weight",
            &[dim, dim],
            GgmlType::F32,
            out_bytes,
        )
        .tensor("blk.0.attn_norm.weight", &[dim], GgmlType::F32, attn_norm);
    shard0_builder.write(&shard0_path).unwrap();

    // Shard 1 has FFN weights and output norm
    let shard1_builder = Builder::new()
        .tensor(
            "blk.0.ffn_gate.weight",
            &[dim, dim],
            GgmlType::F32,
            ffn_gate,
        )
        .tensor("blk.0.ffn_up.weight", &[dim, dim], GgmlType::F32, ffn_up)
        .tensor(
            "blk.0.ffn_down.weight",
            &[dim, dim],
            GgmlType::F32,
            ffn_down,
        )
        .tensor("blk.0.ffn_norm.weight", &[dim], GgmlType::F32, ffn_norm)
        .tensor("output_norm.weight", &[dim], GgmlType::F32, out_norm);
    shard1_builder.write(&shard1_path).unwrap();

    let single_gguf = Gguf::open(&single_path).unwrap();
    let sharded_gguf = Gguf::open_sharded(&[&shard0_path, &shard1_path]).unwrap();

    assert_eq!(single_gguf.tensors.len(), 11);
    assert_eq!(sharded_gguf.tensors.len(), 11);

    let single_model = Model::load(&single_gguf).unwrap();
    let sharded_model = Model::load(&sharded_gguf).unwrap();

    let mut single_state = State::with_context(&single_model.config, 64);
    let mut sharded_state = State::with_context(&sharded_model.config, 64);

    let prompt = [1u32, 5, 2, 8];
    single_model.prefill(&mut single_state, &prompt, 0).unwrap();
    sharded_model
        .prefill(&mut sharded_state, &prompt, 0)
        .unwrap();

    assert_eq!(single_state.logits.len(), sharded_state.logits.len());
    for (a, b) in single_state.logits.iter().zip(&sharded_state.logits) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "sharded logits must be bit-identical"
        );
    }

    let _ = fs::remove_file(single_path);
    let _ = fs::remove_file(shard0_path);
    let _ = fs::remove_file(shard1_path);
}

#[test]
fn test_split_gguf_roundtrip() {
    let mut dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    dir.push(format!("ferrite_split_test_{nanos}"));
    fs::create_dir_all(&dir).unwrap();

    let single_path = dir.join("monolith.gguf");
    let dim = 32u64;
    let vocab = 16u64;
    let embed_bytes = pattern((dim * vocab * 4) as usize, 10);
    let q_bytes = pattern((dim * dim * 4) as usize, 20);
    let k_bytes = pattern((dim * dim * 4) as usize, 30);
    let v_bytes = pattern((dim * dim * 4) as usize, 40);
    let out_bytes = pattern((dim * dim * 4) as usize, 50);

    let single_builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(1))
        .meta("llama.embedding_length", Value::U32(dim as u32))
        .meta("llama.feed_forward_length", Value::U32(dim as u32))
        .meta("llama.attention.head_count", Value::U32(1))
        .meta("llama.attention.head_count_kv", Value::U32(1))
        .meta("llama.context_length", Value::U32(128))
        .meta("llama.rope.freq_base", Value::F32(10000.0))
        .tensor(
            "token_embd.weight",
            &[dim, vocab],
            GgmlType::F32,
            embed_bytes.clone(),
        )
        .tensor(
            "blk.0.attn_q.weight",
            &[dim, dim],
            GgmlType::F32,
            q_bytes.clone(),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[dim, dim],
            GgmlType::F32,
            k_bytes.clone(),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[dim, dim],
            GgmlType::F32,
            v_bytes.clone(),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[dim, dim],
            GgmlType::F32,
            out_bytes.clone(),
        );
    single_builder.write(&single_path).unwrap();

    let split_prefix = dir.join("sharded_model");
    let shards = ferrite::split_gguf(
        &single_path,
        &split_prefix,
        &ferrite::SplitOptions {
            shards: 2,
            max_bytes_per_shard: None,
        },
    )
    .expect("split_gguf");

    assert_eq!(shards.len(), 2);
    for shard in &shards {
        assert!(shard.exists());
    }

    let loaded_shards = Gguf::open(&shards[0]).expect("open sharded from primary");
    assert_eq!(loaded_shards.shard_count(), 2);
    assert_eq!(loaded_shards.tensors.len(), 5);

    let orig_gguf = Gguf::open(&single_path).unwrap();
    for t_orig in &orig_gguf.tensors {
        let t_split = loaded_shards
            .tensor(&t_orig.name)
            .expect("tensor in split model");
        let b1 = orig_gguf.tensor_bytes(t_orig).unwrap();
        let b2 = loaded_shards.tensor_bytes(t_split).unwrap();
        assert_eq!(b1, b2, "tensor data mismatch for {}", t_orig.name);
    }

    let _ = fs::remove_dir_all(dir);
}
