//! Forward-pass tests against a model small enough to work out by hand.
//!
//! A transformer with the wrong wiring does not crash, it produces plausible
//! numbers. So these fixtures zero out most of the network, leaving a path
//! whose output can be computed exactly and compared — the residual stream, the
//! KV cache, and the attention average each get their own arithmetic check.

use std::fs;
use std::path::{Path, PathBuf};

use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::model::{Config, Model, State};
use ferrite::ops;
use ferrite::synth::Builder;

const DIM: usize = 4;
const VOCAB: usize = 4;
const EPS: f32 = 1e-5;

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn identity(n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

/// Embedding table: one distinct vector per token.
///
/// Deliberately not collinear. An earlier version used `(token + 1) *
/// (channel + 1)`, which makes every row a multiple of the same vector — RMS
/// normalization then erases the difference between tokens entirely, and any
/// test asking "does history change the output" passes vacuously.
fn embeddings() -> Vec<f32> {
    vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 2.0, 0.0, -1.0, //
        1.0, 2.0, 3.0, 4.0, //
        -2.0, 0.5, 1.0, 0.0,
    ]
}

fn embedding_of(token: usize) -> Vec<f32> {
    embeddings()[token * DIM..(token + 1) * DIM].to_vec()
}

/// `attention` and `value` select which paths stay live; everything else is
/// zeroed so the arithmetic stays closed-form.
fn fixture(kv_heads: usize, value_path: bool) -> Builder {
    let zero = vec![0.0; DIM * DIM];
    let kv_dim = kv_heads * (DIM / 2);
    let heads = 2;

    let mut builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(1))
        .meta("llama.embedding_length", Value::U32(DIM as u32))
        .meta("llama.feed_forward_length", Value::U32(DIM as u32))
        .meta("llama.attention.head_count", Value::U32(heads))
        .meta("llama.attention.head_count_kv", Value::U32(kv_heads as u32))
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
        // Zero queries and keys mean every attention score is zero, so softmax
        // spreads weight evenly over the past. That is what makes the expected
        // value a plain average.
        .tensor(
            "blk.0.attn_q.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&zero),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&vec![0.0; DIM * kv_dim]),
        );

    let value_weights = if value_path {
        identity(DIM)
    } else {
        zero.clone()
    };
    builder = builder
        .tensor(
            "blk.0.attn_v.weight",
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&value_weights[..DIM * kv_dim]),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&if value_path {
                identity(DIM)
            } else {
                zero.clone()
            }),
        )
        .tensor(
            "blk.0.ffn_norm.weight",
            &[DIM as u64],
            GgmlType::F32,
            bytes(&[1.0; DIM]),
        )
        // A zeroed gate makes the whole feed-forward branch contribute nothing.
        .tensor(
            "blk.0.ffn_gate.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&zero),
        )
        .tensor(
            "blk.0.ffn_up.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&zero),
        )
        .tensor(
            "blk.0.ffn_down.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&zero),
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
        );
    builder
}

fn write(name: &str, builder: &Builder) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ferrite_model_{name}.gguf"));
    builder.write(&path).expect("write fixture");
    path
}

fn normalized(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    ops::rms_norm(x, &vec![1.0f32; x.len()], EPS, &mut out);
    out
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "element {index}: got {a}, expected {b}\n  actual {actual:?}\n  expected {expected:?}"
        );
    }
}

fn run(path: &Path, tokens: &[u32]) -> Vec<f32> {
    let gguf = Gguf::open(path).expect("open");
    let model = Model::load(&gguf).expect("load");
    let mut state = State::new(&model.config);
    for (position, token) in tokens.iter().enumerate() {
        model
            .forward(&mut state, *token, position)
            .expect("forward");
    }
    state.logits.clone()
}

#[test]
fn dead_network_passes_the_embedding_through() {
    // Attention and feed-forward both contribute zero, so the residual stream
    // is exactly the embedding, and the logits are its normalization.
    let path = write("passthrough", &fixture(2, false));
    let logits = run(&path, &[2]);
    assert_close(&logits, &normalized(&embedding_of(2)));
    fs::remove_file(path).ok();
}

#[test]
fn attention_adds_the_average_of_past_values() {
    // With identity value and output projections, the attention branch returns
    // the mean of the normalized inputs it has seen. Position 0 sees only
    // itself.
    let path = write("value_path", &fixture(2, true));

    let e0 = embedding_of(1);
    let expected_first = {
        let mut x = e0.clone();
        for (slot, delta) in x.iter_mut().zip(normalized(&e0)) {
            *slot += delta;
        }
        normalized(&x)
    };
    assert_close(&run(&path, &[1]), &expected_first);

    // Position 1 averages both cached values: this is the check that the KV
    // cache is written, read back, and weighted correctly.
    let e1 = embedding_of(3);
    let expected_second = {
        let (n0, n1) = (normalized(&e0), normalized(&e1));
        let mut x = e1.clone();
        for channel in 0..DIM {
            x[channel] += 0.5 * (n0[channel] + n1[channel]);
        }
        normalized(&x)
    };
    assert_close(&run(&path, &[1, 3]), &expected_second);

    fs::remove_file(path).ok();
}

#[test]
fn grouped_query_attention_runs_with_fewer_kv_heads() {
    let path = write("gqa", &fixture(1, false));
    let gguf = Gguf::open(&path).expect("open");
    let config = Config::from_gguf(&gguf).expect("config");
    assert_eq!(config.kv_heads, 1);
    assert_eq!(config.group(), 2, "two query heads share one kv head");
    assert_eq!(config.kv_dim(), 2);

    let logits = run(&path, &[0, 1, 2]);
    assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");
    fs::remove_file(path).ok();
}

#[test]
fn decoding_is_deterministic_and_position_dependent() {
    let path = write("determinism", &fixture(2, true));
    assert_eq!(run(&path, &[1, 2]), run(&path, &[1, 2]));
    assert_ne!(
        run(&path, &[1, 2]),
        run(&path, &[3, 2]),
        "history must change the outcome"
    );
    fs::remove_file(path).ok();
}

#[test]
fn threading_does_not_change_the_answer() {
    // Rows are independent and each is reduced in the same order regardless of
    // how they are split, so this must hold exactly, not approximately.
    let path = write("threads", &fixture(2, true));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let logits_for = |threads: usize| {
        let mut state = State::new(&model.config);
        state.threads = threads;
        for (position, token) in [1u32, 3, 2].iter().enumerate() {
            model
                .forward(&mut state, *token, position)
                .expect("forward");
        }
        state.logits.clone()
    };

    assert_eq!(
        logits_for(1),
        logits_for(4),
        "bit-identical, not merely close"
    );
    assert_eq!(logits_for(1), logits_for(8));
    fs::remove_file(path).ok();
}

#[test]
fn running_past_the_cache_is_an_error_not_a_corruption() {
    let path = write("overflow", &fixture(2, false));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");
    let mut state = State::with_context(&model.config, 2);

    assert!(model.forward(&mut state, 0, 0).is_ok());
    assert!(model.forward(&mut state, 0, 1).is_ok());
    let err = model.forward(&mut state, 0, 2).expect_err("must refuse");
    assert!(err.to_string().contains("past the 2 token cache"), "{err}");

    fs::remove_file(path).ok();
}
