use ferrite::convert::{convert_safetensors_to_gguf, create_synthetic_safetensors};
use ferrite::gguf::{GgmlType, Gguf};
use ferrite::model::{Model, State};
use std::fs;
use std::path::PathBuf;

const DIM: usize = 4;
const FFN: usize = 8;
const VOCAB: usize = 4;

fn embeddings() -> Vec<f32> {
    vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 2.0, 0.0, -1.0, //
        1.0, 2.0, 3.0, 4.0, //
        -2.0, 0.5, 1.0, 0.0,
    ]
}

fn identity(dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; dim * dim];
    for i in 0..dim {
        out[i * dim + i] = 1.0;
    }
    out
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn setup_test_model_dir(dir_name: &str) -> (PathBuf, PathBuf) {
    let base_dir = std::env::temp_dir().join(format!("ferrite_st_{dir_name}"));
    let _ = fs::remove_dir_all(&base_dir);
    fs::create_dir_all(&base_dir).expect("create dir");

    let config_json = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": 1,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "vocab_size": 4
    }"#;
    fs::write(base_dir.join("config.json"), config_json).expect("write config.json");

    let emb_bytes = bytes(&embeddings());
    let id_bytes = bytes(&identity(DIM));
    let norm_bytes = bytes(&[1.0; DIM]);
    let ffn_gate_bytes = bytes(&[0.1; DIM * FFN]);
    let ffn_down_bytes = bytes(&[0.1; FFN * DIM]);

    // PyTorch SafeTensors shapes: [out_features, in_features] or [vocab, dim]
    let tensors: Vec<(&str, GgmlType, &[u64], &[u8])> = vec![
        (
            "model.embed_tokens.weight",
            GgmlType::F32,
            &[VOCAB as u64, DIM as u64],
            &emb_bytes,
        ),
        (
            "model.layers.0.input_layernorm.weight",
            GgmlType::F32,
            &[DIM as u64],
            &norm_bytes,
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            &[DIM as u64, DIM as u64],
            &id_bytes,
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            GgmlType::F32,
            &[DIM as u64, DIM as u64],
            &id_bytes,
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            GgmlType::F32,
            &[DIM as u64, DIM as u64],
            &id_bytes,
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            GgmlType::F32,
            &[DIM as u64, DIM as u64],
            &id_bytes,
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            GgmlType::F32,
            &[DIM as u64],
            &norm_bytes,
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            GgmlType::F32,
            &[FFN as u64, DIM as u64],
            &ffn_gate_bytes,
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            GgmlType::F32,
            &[FFN as u64, DIM as u64],
            &ffn_gate_bytes,
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            GgmlType::F32,
            &[DIM as u64, FFN as u64],
            &ffn_down_bytes,
        ),
        (
            "model.norm.weight",
            GgmlType::F32,
            &[DIM as u64],
            &norm_bytes,
        ),
        (
            "lm_head.weight",
            GgmlType::F32,
            &[VOCAB as u64, DIM as u64],
            &emb_bytes,
        ),
    ];

    let st_path = base_dir.join("model.safetensors");
    create_synthetic_safetensors(&st_path, &tensors).expect("create safetensors");

    let out_gguf = std::env::temp_dir().join(format!("converted_{dir_name}.gguf"));
    (base_dir, out_gguf)
}

#[test]
fn test_convert_directory_to_gguf_and_load_inference() {
    let (model_dir, out_gguf) = setup_test_model_dir("full_dir");

    // Convert from directory containing config.json and model.safetensors
    convert_safetensors_to_gguf(&model_dir, &out_gguf, None).expect("convert safetensors");

    assert!(out_gguf.exists());
    let gguf = Gguf::open(&out_gguf).expect("open converted gguf");

    // Verify architecture and hyperparameters
    assert_eq!(gguf.arch(), Some("llama"));
    assert_eq!(gguf.meta_u64("llama.block_count"), Some(1));
    assert_eq!(gguf.meta_u64("llama.embedding_length"), Some(4));
    assert_eq!(gguf.meta_u64("llama.feed_forward_length"), Some(8));
    assert_eq!(gguf.meta_u64("llama.attention.head_count"), Some(2));
    assert_eq!(gguf.meta_u64("llama.attention.head_count_kv"), Some(2));

    // Verify mapped tensor names exist
    assert!(gguf.tensor("token_embd.weight").is_some());
    assert!(gguf.tensor("blk.0.attn_norm.weight").is_some());
    assert!(gguf.tensor("blk.0.attn_q.weight").is_some());
    assert!(gguf.tensor("blk.0.attn_k.weight").is_some());
    assert!(gguf.tensor("blk.0.attn_v.weight").is_some());
    assert!(gguf.tensor("blk.0.attn_output.weight").is_some());
    assert!(gguf.tensor("blk.0.ffn_norm.weight").is_some());
    assert!(gguf.tensor("blk.0.ffn_gate.weight").is_some());
    assert!(gguf.tensor("blk.0.ffn_up.weight").is_some());
    assert!(gguf.tensor("blk.0.ffn_down.weight").is_some());
    assert!(gguf.tensor("output_norm.weight").is_some());
    assert!(gguf.tensor("output.weight").is_some());

    // Verify tensor dimensions were reversed to GGUF column-major order:
    // SafeTensors embed_tokens was [4, 4]; in GGUF token_embd.weight shape is [4, 4]
    let token_embd = gguf.tensor("token_embd.weight").unwrap();
    assert_eq!(token_embd.dims, vec![4, 4]);

    // SafeTensors ffn_gate was [8, 4]; in GGUF blk.0.ffn_gate.weight shape is [4, 8]
    let ffn_gate = gguf.tensor("blk.0.ffn_gate.weight").unwrap();
    assert_eq!(ffn_gate.dims, vec![4, 8]);

    // Load converted model and run forward inference pass
    let model = Model::load(&gguf).expect("load converted model");
    let mut state = State::new(&model.config);
    model
        .forward(&mut state, 0, 0)
        .expect("forward pass on converted model");

    assert_eq!(state.logits.len(), VOCAB);
    assert!(state.logits.iter().all(|x| x.is_finite()));

    // Run batched prefill
    let prompt = vec![0u32, 1, 2];
    model
        .prefill(&mut state, &prompt, 0)
        .expect("prefill pass on converted model");

    fs::remove_dir_all(model_dir).ok();
    fs::remove_file(out_gguf).ok();
}

#[test]
fn test_convert_single_safetensors_file() {
    let (model_dir, out_gguf) = setup_test_model_dir("single_file");
    let st_file = model_dir.join("model.safetensors");

    // Convert directly from single .safetensors file path
    convert_safetensors_to_gguf(&st_file, &out_gguf, Some("llama")).expect("convert single file");

    assert!(out_gguf.exists());
    let gguf = Gguf::open(&out_gguf).expect("open converted gguf");
    let model = Model::load(&gguf).expect("load model");

    let mut state = State::new(&model.config);
    model.forward(&mut state, 1, 0).expect("forward pass");
    assert!(state.logits.iter().all(|x| x.is_finite()));

    fs::remove_dir_all(model_dir).ok();
    fs::remove_file(out_gguf).ok();
}
