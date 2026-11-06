use ferrite::eval::{cross_entropy_loss, PerplexityEvaluator};
use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::model::Model;
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

fn make_eval_fixture(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ferrite_eval_{name}.gguf"));
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
            bytes(&[0.1; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_up.weight",
            &[DIM as u64, FFN as u64],
            GgmlType::F32,
            bytes(&[0.1; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_down.weight",
            &[FFN as u64, DIM as u64],
            GgmlType::F32,
            bytes(&[0.1; FFN * DIM]),
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
            bytes(&embeddings()),
        );

    fs::write(&path, builder.build()).expect("write gguf");
    path
}

#[test]
fn test_cross_entropy_loss_numerical_properties() {
    // 2-class test
    let logits = vec![0.0f32, 0.0f32];
    let loss_0 = cross_entropy_loss(&logits, 0).expect("loss");
    let loss_1 = cross_entropy_loss(&logits, 1).expect("loss");
    assert!((loss_0 - 2.0f64.ln()).abs() < 1e-6);
    assert!((loss_1 - 2.0f64.ln()).abs() < 1e-6);

    // Extreme logit difference: target logit dominant -> loss -> 0
    let logits_dominant = vec![50.0f32, -50.0f32];
    let dominant_loss = cross_entropy_loss(&logits_dominant, 0).expect("loss");
    assert!(dominant_loss < 1e-10);
}

#[test]
fn test_perplexity_evaluator_disjoint_chunks() {
    let model_path = make_eval_fixture("disjoint");
    let gguf = Gguf::open(&model_path).expect("open gguf");
    let model = Model::load(&gguf).expect("load model");

    let tokens = vec![0u32, 1, 2, 3, 0, 1, 2, 3, 1, 2]; // 10 tokens -> 9 target evaluations
    let evaluator = PerplexityEvaluator::new(&model, 4, 4); // ctx=4, stride=4
    let report = evaluator.evaluate(&tokens).expect("evaluate");

    assert_eq!(report.total_tokens, 10);
    assert_eq!(report.evaluated_tokens, 9);
    assert!(report.perplexity >= 1.0);
    assert!(!report.chunks.is_empty());
    assert_eq!(
        report
            .chunks
            .iter()
            .map(|chunk| chunk.count)
            .collect::<Vec<_>>(),
        vec![3, 3, 3]
    );
    assert_eq!(
        report
            .chunks
            .iter()
            .map(|chunk| chunk.start_token)
            .collect::<Vec<_>>(),
        vec![1, 4, 7]
    );

    fs::remove_file(model_path).ok();
}

#[test]
fn test_perplexity_evaluator_sliding_window() {
    let model_path = make_eval_fixture("sliding");
    let gguf = Gguf::open(&model_path).expect("open gguf");
    let model = Model::load(&gguf).expect("load model");

    let tokens = vec![0u32, 1, 2, 3, 0, 1, 2, 3, 1, 2]; // 10 tokens
    let evaluator = PerplexityEvaluator::new(&model, 4, 2); // ctx=4, stride=2 (sliding window)
    let report = evaluator.evaluate(&tokens).expect("evaluate");

    assert_eq!(report.total_tokens, 10);
    assert_eq!(report.evaluated_tokens, 9);
    assert!(report.perplexity >= 1.0);
    assert_eq!(
        report.chunks.iter().map(|chunk| chunk.count).sum::<usize>(),
        9
    );

    fs::remove_file(model_path).ok();
}

#[test]
fn test_perplexity_evaluator_rejects_single_token() {
    let model_path = make_eval_fixture("single");
    let gguf = Gguf::open(&model_path).expect("open gguf");
    let model = Model::load(&gguf).expect("load model");

    let tokens = vec![0u32];
    let evaluator = PerplexityEvaluator::new(&model, 4, 4);
    let err = evaluator.evaluate(&tokens).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

    fs::remove_file(model_path).ok();
}
