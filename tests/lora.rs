//! Integration tests for zero-copy dynamic LoRA adapter application.

use std::fs;
use std::path::PathBuf;

use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::lora::LoraAdapter;
use ferrite::model::{Model, State};
use ferrite::synth::Builder;

const DIM: usize = 4;
const VOCAB: usize = 4;
const FFN: usize = 8;
const EPS: f32 = 1e-5;

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn embeddings() -> Vec<f32> {
    vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 2.0, 0.0, -1.0, //
        1.0, 2.0, 3.0, 4.0, //
        -2.0, 0.5, 1.0, 0.0,
    ]
}

fn identity(n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

fn base_model_builder(custom_down: Option<&[f32]>) -> Builder {
    let zero = vec![0.0; DIM * DIM];
    let default_down = vec![0.0; FFN * DIM];
    let down_slice = custom_down.unwrap_or(&default_down);
    Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(1))
        .meta("llama.embedding_length", Value::U32(DIM as u32))
        .meta("llama.feed_forward_length", Value::U32(FFN as u32))
        .meta("llama.attention.head_count", Value::U32(2))
        .meta("llama.attention.head_count_kv", Value::U32(2))
        .meta("llama.context_length", Value::U32(16))
        .meta("llama.attention.layer_norm_rms_epsilon", Value::F32(EPS))
        .meta("llama.rope.freq_base", Value::F32(10000.0))
        .meta("tokenizer.ggml.model", Value::String("llama".into()))
        .meta(
            "tokenizer.ggml.tokens",
            Value::Array((0..VOCAB).map(|i| Value::String(format!("t{i}"))).collect()),
        )
        .tensor(
            "token_embd.weight",
            &[DIM as u64, VOCAB as u64],
            GgmlType::F32,
            bytes(&embeddings()),
        )
        .tensor(
            "blk.0.attn_norm.weight",
            &[DIM as u64],
            GgmlType::F32,
            bytes(&[1.0; DIM]),
        )
        .tensor(
            "blk.0.attn_q.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&zero),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&zero),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&identity(DIM)),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&identity(DIM)),
        )
        .tensor(
            "blk.0.ffn_norm.weight",
            &[DIM as u64],
            GgmlType::F32,
            bytes(&[1.0; DIM]),
        )
        .tensor(
            "blk.0.ffn_gate.weight",
            &[DIM as u64, FFN as u64],
            GgmlType::F32,
            bytes(&[1.0; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_up.weight",
            &[DIM as u64, FFN as u64],
            GgmlType::F32,
            bytes(&[1.0; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_down.weight",
            &[FFN as u64, DIM as u64],
            GgmlType::F32,
            bytes(down_slice),
        )
        .tensor(
            "output_norm.weight",
            &[DIM as u64],
            GgmlType::F32,
            bytes(&[1.0; DIM]),
        )
        .tensor(
            "output.weight",
            &[DIM as u64, VOCAB as u64],
            GgmlType::F32,
            bytes(&identity(DIM)),
        )
}

fn write_file(name: &str, builder: &Builder) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ferrite_lora_test_{name}.gguf"));
    builder.write(&path).expect("write gguf");
    path
}

#[test]
fn lora_adapter_forward_and_prefill_match() {
    let base_path = write_file("base_model", &base_model_builder(None));

    // Rank 2 adapter for attn_v:
    // A has shape [DIM, rank] = [4, 2]
    let a_vals = vec![
        0.5, 0.0, 1.0, -0.5, // row 0
        -1.0, 0.5, 0.0, 2.0, // row 1
    ];
    // B has shape [rank, DIM] = [2, 4]
    let b_vals = vec![
        1.0, 0.0, // row 0
        0.0, 1.0, // row 1
        -0.5, 0.5, // row 2
        0.5, -0.5, // row 3
    ];
    let alpha = 2.0;

    let adapter_builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("adapter.lora.alpha", Value::F32(alpha))
        .tensor(
            "blk.0.attn_v.weight.lora_a",
            &[4, 2],
            GgmlType::F32,
            bytes(&a_vals),
        )
        .tensor(
            "blk.0.attn_v.weight.lora_b",
            &[2, 4],
            GgmlType::F32,
            bytes(&b_vals),
        );

    let lora_path = write_file("adapter", &adapter_builder);

    let base_gguf = Gguf::open(&base_path).expect("open base");
    let mut model = Model::load(&base_gguf).expect("load model");

    let lora_gguf = Gguf::open(&lora_path).expect("open lora");
    let adapter = LoraAdapter::load(&lora_gguf, None).expect("load lora");
    model.apply_lora(adapter);

    let prompt = [1u32, 2, 3, 0];

    // 1. Sequential forward
    let mut state_seq = State::new(&model.config);
    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }

    // 2. Batched prefill
    let mut state_batch = State::new(&model.config);
    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_batch.logits),
        bits(&state_seq.logits),
        "LoRA prefill logits must match sequential forward exactly"
    );

    // 3. Clear LoRA and verify output changes back
    let lora_logits = state_seq.logits.clone();
    model.clear_lora();

    let mut state_clean = State::new(&model.config);
    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_clean, tok, pos).expect("forward");
    }
    assert_ne!(
        bits(&state_clean.logits),
        bits(&lora_logits),
        "logits must differ after removing LoRA"
    );

    fs::remove_file(base_path).ok();
    fs::remove_file(lora_path).ok();
}

#[test]
fn lora_matches_baked_weights_mathematically() {
    // Construct base model
    let base_path = write_file("base_for_bake", &base_model_builder(None));

    // Rank 1 adapter for down projection:
    // A has shape [FFN, 1] = [8, 1]
    let a_vals = vec![0.5, -0.5, 1.0, 0.0, 0.2, -0.1, 0.8, -0.4];
    // B has shape [1, DIM] = [1, 4]
    let b_vals = vec![1.0, -1.0, 0.5, 2.0];
    let alpha = 1.0; // scale = 1.0 / 1 = 1.0

    let adapter_builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("adapter.lora.alpha", Value::F32(alpha))
        .tensor(
            "blk.0.ffn_down.weight.lora_a",
            &[8, 1],
            GgmlType::F32,
            bytes(&a_vals),
        )
        .tensor(
            "blk.0.ffn_down.weight.lora_b",
            &[1, 4],
            GgmlType::F32,
            bytes(&b_vals),
        );

    let lora_path = write_file("down_adapter", &adapter_builder);

    // Construct mathematically equivalent baked model:
    // delta_down[r, c] = b_vals[r] * a_vals[c]
    let mut baked_down = vec![0.0f32; 4 * 8];
    for r in 0..4 {
        for c in 0..8 {
            baked_down[r * 8 + c] = b_vals[r] * a_vals[c];
        }
    }

    let baked_builder = base_model_builder(Some(&baked_down));
    let baked_path = write_file("baked_model", &baked_builder);
    let merged_path = std::env::temp_dir().join("ferrite_lora_merged_model.gguf");
    let merge_base = Gguf::open(&base_path).expect("open merge base");
    let merge_adapter = Gguf::open(&lora_path).expect("open merge adapter");
    ferrite::lora::merge_adapters(&merge_base, &[&merge_adapter], &merged_path)
        .expect("merge adapter");
    let merged = Gguf::open(&merged_path).expect("open merged model");
    let merged_model = Model::load(&merged).expect("load merged model");
    let mut merged_state = State::new(&merged_model.config);
    merged_model
        .forward(&mut merged_state, 1, 0)
        .expect("forward merged");

    // Run base + LoRA
    let base_gguf = Gguf::open(&base_path).expect("open base");
    let mut model_lora = Model::load(&base_gguf).expect("load base");
    let lora_gguf = Gguf::open(&lora_path).expect("open lora");
    let adapter = LoraAdapter::load(&lora_gguf, None).expect("load lora");
    model_lora.apply_lora(adapter);

    let mut state_lora = State::new(&model_lora.config);
    model_lora
        .forward(&mut state_lora, 1, 0)
        .expect("forward lora");

    // Run baked model
    let baked_gguf = Gguf::open(&baked_path).expect("open baked");
    let model_baked = Model::load(&baked_gguf).expect("load baked");
    let mut state_baked = State::new(&model_baked.config);
    model_baked
        .forward(&mut state_baked, 1, 0)
        .expect("forward baked");

    for (a, b) in state_lora.logits.iter().zip(&state_baked.logits) {
        assert!(
            (a - b).abs() < 1e-5,
            "LoRA dynamic forward ({a}) must match baked weights ({b})"
        );
    }
    for (a, b) in state_lora.logits.iter().zip(&merged_state.logits) {
        assert!(
            (a - b).abs() < 1e-5,
            "merged LoRA output mismatch: {a} vs {b}"
        );
    }

    fs::remove_file(base_path).ok();
    fs::remove_file(lora_path).ok();
    fs::remove_file(baked_path).ok();
    fs::remove_file(merged_path).ok();
}
