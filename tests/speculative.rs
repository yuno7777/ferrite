use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::model::{Model, State};
use ferrite::sample::Sampler;
use ferrite::speculative::SpeculativeEngine;
use ferrite::synth::Builder;
use std::fs;
use std::path::PathBuf;

const DIM: usize = 4;
const FFN: usize = 8;
const VOCAB: usize = 4;
const EPS: f32 = 1e-5;

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

fn make_model_fixture(name: &str, bias: f32) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ferrite_spec_{name}.gguf"));
    let builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(1))
        .meta("llama.embedding_length", Value::U32(DIM as u32))
        .meta("llama.feed_forward_length", Value::U32(FFN as u32))
        .meta("llama.attention.head_count", Value::U32(2))
        .meta("llama.attention.head_count_kv", Value::U32(2))
        .meta("llama.context_length", Value::U32(32))
        .meta("llama.attention.layer_norm_rms_epsilon", Value::F32(EPS))
        .meta("tokenizer.ggml.model", Value::String("llama".into()))
        .meta(
            "tokenizer.ggml.tokens",
            Value::Array((0..VOCAB).map(|i| Value::String(format!("t{i}"))).collect()),
        )
        .meta(
            "tokenizer.ggml.scores",
            Value::Array((0..VOCAB).map(|i| Value::F32(i as f32)).collect()),
        )
        .meta(
            "tokenizer.ggml.token_type",
            Value::Array((0..VOCAB).map(|_| Value::I32(1)).collect()),
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
            bytes(&identity(DIM)),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&identity(DIM)),
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
            bytes(&[0.0; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_up.weight",
            &[DIM as u64, FFN as u64],
            GgmlType::F32,
            bytes(&[0.0; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_down.weight",
            &[FFN as u64, DIM as u64],
            GgmlType::F32,
            bytes(&[0.0; FFN * DIM]),
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
            bytes(&{
                let mut out = identity(DIM);
                for v in out.iter_mut() {
                    *v += bias;
                }
                out
            }),
        );

    builder.write(&path).expect("write");
    path
}

#[test]
fn speculative_identical_models_accept_all_draft_tokens() {
    let path = make_model_fixture("identical", 0.0);
    let gguf = Gguf::open(&path).expect("open");
    let target_model = Model::load(&gguf).expect("load target");
    let draft_model = Model::load(&gguf).expect("load draft");

    let prompt = [0u32, 1];
    let total_to_generate = 6;

    // 1. Baseline target autoregressive generation
    let mut baseline_state = State::new(&target_model.config);
    target_model
        .prefill(&mut baseline_state, &prompt, 0)
        .expect("prefill");

    let mut baseline_tokens = Vec::new();
    let mut baseline_sampler = Sampler::greedy();
    let mut pos = prompt.len();

    while baseline_tokens.len() < total_to_generate {
        let tok = baseline_sampler.sample(&mut baseline_state.logits);
        baseline_tokens.push(tok);
        target_model
            .forward(&mut baseline_state, tok, pos)
            .expect("forward");
        pos += 1;
    }

    // 2. Speculative generation with identical draft model (100% acceptance rate)
    let mut target_state = State::new(&target_model.config);
    let mut draft_state = State::new(&draft_model.config);

    target_model
        .prefill(&mut target_state, &prompt, 0)
        .expect("prefill target");
    draft_model
        .prefill(&mut draft_state, &prompt, 0)
        .expect("prefill draft");

    let mut engine = SpeculativeEngine::new(&target_model, &draft_model, 3);
    let mut spec_tokens = Vec::new();
    let mut spec_sampler = Sampler::greedy();
    let mut spec_pos = prompt.len();

    while spec_tokens.len() < total_to_generate {
        let emitted = engine
            .step(
                &mut target_state,
                &mut draft_state,
                spec_pos,
                &mut spec_sampler,
            )
            .expect("step");
        spec_pos += emitted.len();
        spec_tokens.extend(emitted);
    }
    spec_tokens.truncate(total_to_generate);

    assert_eq!(
        spec_tokens, baseline_tokens,
        "speculative decoding must produce bit-identical output to autoregressive decode"
    );
    assert_eq!(
        engine.stats.acceptance_rate(),
        1.0,
        "identical models must have 100% acceptance rate"
    );
    assert!(
        engine.stats.tokens_per_batch() > 1.0,
        "speculative decoding must produce multiple tokens per target batch"
    );

    fs::remove_file(path).ok();
}

#[test]
fn speculative_divergent_draft_model_corrects_and_matches_target() {
    let target_path = make_model_fixture("target", 0.0);
    let draft_path = make_model_fixture("draft_divergent", 0.5);

    let target_gguf = Gguf::open(&target_path).expect("open target");
    let draft_gguf = Gguf::open(&draft_path).expect("open draft");

    let target_model = Model::load(&target_gguf).expect("load target");
    let draft_model = Model::load(&draft_gguf).expect("load draft");

    let prompt = [1u32, 2];
    let total_to_generate = 6;

    // 1. Target standard autoregressive generation
    let mut baseline_state = State::new(&target_model.config);
    target_model
        .prefill(&mut baseline_state, &prompt, 0)
        .expect("prefill");

    let mut baseline_tokens = Vec::new();
    let mut baseline_sampler = Sampler::greedy();
    let mut pos = prompt.len();

    while baseline_tokens.len() < total_to_generate {
        let tok = baseline_sampler.sample(&mut baseline_state.logits);
        baseline_tokens.push(tok);
        target_model
            .forward(&mut baseline_state, tok, pos)
            .expect("forward");
        pos += 1;
    }

    // 2. Speculative generation with divergent draft model
    let mut target_state = State::new(&target_model.config);
    let mut draft_state = State::new(&draft_model.config);

    target_model
        .prefill(&mut target_state, &prompt, 0)
        .expect("prefill target");
    draft_model
        .prefill(&mut draft_state, &prompt, 0)
        .expect("prefill draft");

    let mut engine = SpeculativeEngine::new(&target_model, &draft_model, 2);
    let mut spec_tokens = Vec::new();
    let mut spec_sampler = Sampler::greedy();
    let mut spec_pos = prompt.len();

    while spec_tokens.len() < total_to_generate {
        let emitted = engine
            .step(
                &mut target_state,
                &mut draft_state,
                spec_pos,
                &mut spec_sampler,
            )
            .expect("step");
        spec_pos += emitted.len();
        spec_tokens.extend(emitted);
    }
    spec_tokens.truncate(total_to_generate);

    // CRITICAL: Even when the draft model differs, the target model's output MUST be strictly preserved!
    assert_eq!(
        spec_tokens, baseline_tokens,
        "speculative output must strictly match target model baseline regardless of draft quality"
    );

    fs::remove_file(target_path).ok();
    fs::remove_file(draft_path).ok();
}

#[test]
fn speculative_tree_drafting_matches_target_baseline() {
    let target_path = make_model_fixture("tree_tgt", 0.0);
    let draft_path = make_model_fixture("tree_dft", 0.3); // Divergent draft weights

    let target_gguf = Gguf::open(&target_path).expect("open target");
    let draft_gguf = Gguf::open(&draft_path).expect("open draft");

    let target_model = Model::load(&target_gguf).expect("load target");
    let draft_model = Model::load(&draft_gguf).expect("load draft");

    let prompt = vec![0u32, 1];
    let total_to_generate = 6;

    // 1. Direct autoregressive baseline from target model
    let mut baseline_state = State::new(&target_model.config);
    target_model
        .prefill(&mut baseline_state, &prompt, 0)
        .expect("prefill target");

    let mut baseline_tokens = Vec::new();
    let mut baseline_sampler = Sampler::greedy();
    let mut pos = prompt.len();

    while baseline_tokens.len() < total_to_generate {
        let tok = baseline_sampler.sample(&mut baseline_state.logits);
        baseline_tokens.push(tok);
        target_model
            .forward(&mut baseline_state, tok, pos)
            .expect("forward");
        pos += 1;
    }

    // 2. Tree speculative decoding with 2 candidate branches of depth 2
    let mut target_state = State::new(&target_model.config);
    let mut draft_state = State::new(&draft_model.config);

    target_model
        .prefill(&mut target_state, &prompt, 0)
        .expect("prefill target");
    draft_model
        .prefill(&mut draft_state, &prompt, 0)
        .expect("prefill draft");

    let mut tree_engine = ferrite::speculative::SpeculativeTreeEngine::new(
        &target_model,
        &draft_model,
        2, // depth
        2, // branch_count
    );
    let mut spec_tokens = Vec::new();
    let mut spec_sampler = Sampler::greedy();
    let mut spec_pos = prompt.len();

    while spec_tokens.len() < total_to_generate {
        let emitted = tree_engine
            .step(
                &mut target_state,
                &mut draft_state,
                spec_pos,
                &mut spec_sampler,
            )
            .expect("tree step");
        spec_pos += emitted.len();
        spec_tokens.extend(emitted);
    }
    spec_tokens.truncate(total_to_generate);

    assert_eq!(
        spec_tokens, baseline_tokens,
        "speculative tree output must strictly match target model baseline"
    );
    assert!(tree_engine.stats.proposed > 0);
    assert!(tree_engine.stats.total_emitted >= total_to_generate);

    fs::remove_file(target_path).ok();
    fs::remove_file(draft_path).ok();
}
