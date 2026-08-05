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
/// Deliberately not equal to DIM: the matvec workspace is sized for the widest
/// matrix in the model, so a narrower one has to be sliced down to its own
/// width. When those were the same number, that slicing could be — and was —
/// omitted without any test noticing.
const FFN: usize = 8;
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
        .meta("llama.feed_forward_length", Value::U32(FFN as u32))
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
            &[DIM as u64, FFN as u64],
            GgmlType::F32,
            bytes(&vec![0.0; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_up.weight",
            &[DIM as u64, FFN as u64],
            GgmlType::F32,
            bytes(&vec![0.0; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_down.weight",
            &[FFN as u64, DIM as u64],
            GgmlType::F32,
            bytes(&vec![0.0; FFN * DIM]),
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
fn the_fused_matvec_agrees_with_the_reference() {
    // Same guarantee as the threading test, one level down: fusing the
    // dequantize into the dot must not move a single bit.
    let path = write("fused", &fixture(2, true));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let x: Vec<f32> = (0..DIM).map(|i| (i as f32 * 0.7).cos()).collect();
    let weight = &model.layers[0].wo;
    let mut scratch = vec![0.0; weight.cols];
    let (mut fused, mut reference) = (vec![0.0; weight.rows], vec![0.0; weight.rows]);

    weight.matvec(&x, &mut fused, &mut scratch).expect("fused");
    weight
        .matvec_reference(&x, &mut reference, &mut scratch)
        .expect("reference");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&fused), bits(&reference));
    fs::remove_file(path).ok();
}

/// A model whose matmuls are `Q8_0`, so the integer kernels are actually
/// reached. Dimensions are multiples of 32 because that is the block size.
fn quantized_fixture() -> Builder {
    const DIM: u64 = 32;
    const VOCAB: u64 = 32;

    /// One `Q8_0` block: an f16 scale of 0.0625, then 32 signed bytes.
    fn q8_0(rows: u64, cols: u64, seed: u64) -> Vec<u8> {
        let mut out = Vec::new();
        let mut state = seed;
        for _ in 0..rows * (cols / 32) {
            out.extend_from_slice(&0x2C00u16.to_le_bytes());
            for _ in 0..32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                out.push((state >> 33) as u8);
            }
        }
        out
    }

    fn ones(n: usize) -> Vec<u8> {
        (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect()
    }

    let mut builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(2))
        .meta("llama.embedding_length", Value::U32(DIM as u32))
        // This fixture's own width, not the module-level FFN: the integer path
        // needs every matrix to be a whole number of 32-element blocks.
        .meta("llama.feed_forward_length", Value::U32(DIM as u32))
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
            &[DIM, VOCAB],
            GgmlType::Q8_0,
            q8_0(VOCAB, DIM, 1),
        );

    for layer in 0..2u64 {
        let seed = 100 + layer * 10;
        builder = builder
            .tensor(
                &format!("blk.{layer}.attn_norm.weight"),
                &[DIM],
                GgmlType::F32,
                ones(DIM as usize),
            )
            .tensor(
                &format!("blk.{layer}.ffn_norm.weight"),
                &[DIM],
                GgmlType::F32,
                ones(DIM as usize),
            );
        for (index, name) in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ]
        .iter()
        .enumerate()
        {
            builder = builder.tensor(
                &format!("blk.{layer}.{name}.weight"),
                &[DIM, DIM],
                GgmlType::Q8_0,
                q8_0(DIM, DIM, seed + index as u64),
            );
        }
    }

    builder
        .tensor(
            "output_norm.weight",
            &[DIM],
            GgmlType::F32,
            ones(DIM as usize),
        )
        .tensor(
            "output.weight",
            &[DIM, VOCAB],
            GgmlType::Q8_0,
            q8_0(VOCAB, DIM, 999),
        )
}

#[test]
fn the_integer_path_tracks_the_exact_one_through_a_whole_forward_pass() {
    // Per-kernel accuracy is pinned in the unit tests; this checks the wiring.
    // A misrouted activation or a mismatched block index would not produce a
    // slightly different number, it would produce a completely different one.
    let path = write("int8", &quantized_fixture());
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let logits_for = |int8: bool| {
        let mut state = State::new(&model.config);
        state.set_int8(int8);
        for (position, token) in [1u32, 5, 2].iter().enumerate() {
            model
                .forward(&mut state, *token, position)
                .expect("forward");
        }
        state.logits.clone()
    };

    let approximate = logits_for(true);
    let exact = logits_for(false);

    assert!(exact.iter().all(|v| v.is_finite()), "{exact:?}");
    assert!(
        exact.iter().any(|v| v.abs() > 1e-3),
        "fixture is degenerate, the comparison would prove nothing: {exact:?}"
    );

    let scale = exact.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    for (index, (a, e)) in approximate.iter().zip(&exact).enumerate() {
        assert!(
            (a - e).abs() / scale < 0.02,
            "logit {index}: int8 {a}, exact {e} (scale {scale})"
        );
    }
    // And they should not be *identical* — that would mean the flag does
    // nothing and the test is watching one path twice.
    assert_ne!(approximate, exact, "set_int8 appears to have no effect");

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
