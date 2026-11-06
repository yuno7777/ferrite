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

#[test]
fn the_batched_matmul_agrees_with_the_matvec() {
    let path = write("batched_matmul", &fixture(2, true));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let batch = 3;
    let weight = &model.layers[0].wo;
    let mut x_batch = Vec::new();
    for b in 0..batch {
        for i in 0..weight.cols {
            x_batch.push(((b * 10 + i) as f32 * 0.7).cos());
        }
    }

    let mut scratch = vec![0.0; weight.cols];
    let mut out_matmul = vec![0.0; batch * weight.rows];
    weight
        .matmul(&x_batch, &mut out_matmul, batch, &mut scratch)
        .expect("matmul");

    let mut out_sequential = vec![0.0; batch * weight.rows];
    for b in 0..batch {
        let x_tok = &x_batch[b * weight.cols..(b + 1) * weight.cols];
        let out_tok = &mut out_sequential[b * weight.rows..(b + 1) * weight.rows];
        weight.matvec(x_tok, out_tok, &mut scratch).expect("matvec");
    }

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&out_matmul), bits(&out_sequential));
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

#[test]
fn batched_prefill_matches_sequential_forward() {
    let path = write("prefill_test", &fixture(2, true));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let prompt = [1u32, 2, 3];

    // 1. Sequential single-token forward:
    let mut state_seq = State::new(&model.config);
    for (pos, &token) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, token, pos).expect("forward");
    }

    // 2. Batched prefill in one pass:
    let mut state_batch = State::new(&model.config);
    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_batch.logits),
        bits(&state_seq.logits),
        "prefill final logits must be bit-identical to sequential forward"
    );

    // KV caches must match across the prompt tokens:
    let kv_len = model.config.layers * prompt.len() * model.config.kv_dim();
    assert_eq!(
        bits(&state_batch.key_cache()[..kv_len]),
        bits(&state_seq.key_cache()[..kv_len]),
        "key cache must be bit-identical"
    );
    assert_eq!(
        bits(&state_batch.value_cache()[..kv_len]),
        bits(&state_seq.value_cache()[..kv_len]),
        "value cache must be bit-identical"
    );

    // Subsequent single-token decoding from prefilled state must match exactly:
    model
        .forward(&mut state_seq, 0, 3)
        .expect("forward seq next");
    model
        .forward(&mut state_batch, 0, 3)
        .expect("forward batch next");
    assert_eq!(
        bits(&state_batch.logits),
        bits(&state_seq.logits),
        "post-prefill decode step must be bit-identical"
    );

    fs::remove_file(path).ok();
}

fn qwen2_fixture() -> Builder {
    let kv_heads = 2;
    let kv_dim = kv_heads * (DIM / 2);

    Builder::new()
        .meta("general.architecture", Value::String("qwen2".into()))
        .meta("qwen2.block_count", Value::U32(1))
        .meta("qwen2.embedding_length", Value::U32(DIM as u32))
        .meta("qwen2.feed_forward_length", Value::U32(FFN as u32))
        .meta("qwen2.attention.head_count", Value::U32(2))
        .meta("qwen2.attention.head_count_kv", Value::U32(kv_heads as u32))
        .meta("qwen2.context_length", Value::U32(16))
        .meta("qwen2.attention.layer_norm_rms_epsilon", Value::F32(EPS))
        .meta("qwen2.rope.freq_base", Value::F32(1_000_000.0))
        .meta("tokenizer.ggml.model", Value::String("qwen2".into()))
        .meta(
            "tokenizer.ggml.tokens",
            Value::Array((0..VOCAB).map(|i| Value::String(format!("t{i}"))).collect()),
        )
        .meta(
            "tokenizer.ggml.merges",
            Value::Array(vec![Value::String("t 0".into())]),
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
            "blk.0.attn_q.bias",
            &[DIM as u64],
            GgmlType::F32,
            bytes(&[0.5; DIM]),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&identity(DIM)),
        )
        .tensor(
            "blk.0.attn_k.bias",
            &[kv_dim as u64],
            GgmlType::F32,
            bytes(&vec![0.25; kv_dim]),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&identity(DIM)),
        )
        .tensor(
            "blk.0.attn_v.bias",
            &[kv_dim as u64],
            GgmlType::F32,
            bytes(&vec![0.1; kv_dim]),
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
            bytes(&identity(DIM)),
        )
}

#[test]
fn qwen2_runs_with_attention_biases_and_prefill() {
    let path = write("qwen2_test", &qwen2_fixture());
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    assert_eq!(model.config.arch, "qwen2");
    assert_eq!(model.config.rope_base, 1_000_000.0);
    assert!(model.layers[0].bq.is_some());
    assert!(model.layers[0].bk.is_some());
    assert!(model.layers[0].bv.is_some());

    let prompt = [0u32, 1, 2];

    // 1. Sequential forward
    let mut state_seq = State::new(&model.config);
    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }
    assert!(state_seq.logits.iter().all(|v| v.is_finite()));

    // 2. Batched prefill
    let mut state_batch = State::new(&model.config);
    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&state_batch.logits), bits(&state_seq.logits));

    fs::remove_file(path).ok();
}

fn gemma2_fixture() -> Builder {
    let mut builder = Builder::new()
        .meta("general.architecture", Value::String("gemma2".into()))
        .meta("gemma2.block_count", Value::U32(2))
        .meta("gemma2.embedding_length", Value::U32(DIM as u32))
        .meta("gemma2.feed_forward_length", Value::U32(FFN as u32))
        .meta("gemma2.attention.head_count", Value::U32(2))
        .meta("gemma2.attention.head_count_kv", Value::U32(2))
        .meta("gemma2.attention.key_length", Value::U32((DIM / 2) as u32))
        .meta("gemma2.context_length", Value::U32(16))
        .meta("gemma2.attention.layer_norm_rms_epsilon", Value::F32(EPS))
        .meta("gemma2.attention.logit_softcapping", Value::F32(50.0))
        .meta("gemma2.final_logit_softcapping", Value::F32(30.0))
        .meta("gemma2.attention.sliding_window", Value::U32(2))
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

    for idx in 0..2 {
        builder = builder
            .tensor(
                &format!("blk.{idx}.attn_norm.weight"),
                &[DIM as u64],
                GgmlType::F32,
                bytes(&[1.0; DIM]),
            )
            .tensor(
                &format!("blk.{idx}.post_attention_norm.weight"),
                &[DIM as u64],
                GgmlType::F32,
                bytes(&[1.0; DIM]),
            )
            .tensor(
                &format!("blk.{idx}.attn_q.weight"),
                &[DIM as u64, DIM as u64],
                GgmlType::F32,
                bytes(&identity(DIM)),
            )
            .tensor(
                &format!("blk.{idx}.attn_k.weight"),
                &[DIM as u64, DIM as u64],
                GgmlType::F32,
                bytes(&identity(DIM)),
            )
            .tensor(
                &format!("blk.{idx}.attn_v.weight"),
                &[DIM as u64, DIM as u64],
                GgmlType::F32,
                bytes(&identity(DIM)),
            )
            .tensor(
                &format!("blk.{idx}.attn_output.weight"),
                &[DIM as u64, DIM as u64],
                GgmlType::F32,
                bytes(&identity(DIM)),
            )
            .tensor(
                &format!("blk.{idx}.ffn_norm.weight"),
                &[DIM as u64],
                GgmlType::F32,
                bytes(&[1.0; DIM]),
            )
            .tensor(
                &format!("blk.{idx}.post_ffn_norm.weight"),
                &[DIM as u64],
                GgmlType::F32,
                bytes(&[1.0; DIM]),
            )
            .tensor(
                &format!("blk.{idx}.ffn_gate.weight"),
                &[DIM as u64, FFN as u64],
                GgmlType::F32,
                bytes(&[0.0; DIM * FFN]),
            )
            .tensor(
                &format!("blk.{idx}.ffn_up.weight"),
                &[DIM as u64, FFN as u64],
                GgmlType::F32,
                bytes(&[0.0; DIM * FFN]),
            )
            .tensor(
                &format!("blk.{idx}.ffn_down.weight"),
                &[FFN as u64, DIM as u64],
                GgmlType::F32,
                bytes(&[0.0; FFN * DIM]),
            );
    }
    builder
}

#[test]
fn gemma2_runs_with_softcapping_and_norms() {
    let path = write("gemma2_test", &gemma2_fixture());
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    assert_eq!(model.config.arch, "gemma2");
    assert_eq!(model.config.attn_logit_softcapping, Some(50.0));
    assert_eq!(model.config.final_logit_softcapping, Some(30.0));
    assert_eq!(model.config.sliding_window, Some(2));
    assert_eq!(model.config.embedding_scale, Some((DIM as f32).sqrt()));
    assert!(model.layers[0].post_attn_norm.is_some());
    assert!(model.layers[0].post_ffn_norm.is_some());

    let prompt = [0u32, 1, 2, 3];

    // 1. Sequential forward
    let mut state_seq = State::new(&model.config);
    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }
    assert!(state_seq.logits.iter().all(|v| v.is_finite()));
    // Final logits must be softcapped within [-30.0, 30.0]
    for &l in &state_seq.logits {
        assert!(
            (-30.0..=30.0).contains(&l),
            "logit {l} exceeded softcap 30.0"
        );
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
        "gemma2 prefill logits must match sequential forward"
    );

    fs::remove_file(path).ok();
}

#[test]
fn threaded_prefill_matches_single_thread() {
    let path = write("threaded_prefill", &fixture(2, true));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let prompt = [1u32, 2, 3, 0];

    let run_prefill = |threads: usize| {
        let mut state = State::new(&model.config);
        state.threads = threads;
        model.prefill(&mut state, &prompt, 0).expect("prefill");
        let mut all_logits = vec![0.0f32; prompt.len() * model.config.vocab];
        let mut state_all = State::new(&model.config);
        state_all.threads = threads;
        model
            .prefill_all_logits(&mut state_all, &prompt, 0, &mut all_logits)
            .expect("prefill_all");
        (state.logits, all_logits)
    };

    let (logits_1, all_1) = run_prefill(1);
    let (logits_4, all_4) = run_prefill(4);

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&logits_1),
        bits(&logits_4),
        "threaded prefill logits match 1 thread"
    );
    assert_eq!(
        bits(&all_1),
        bits(&all_4),
        "threaded all_logits match 1 thread"
    );

    fs::remove_file(path).ok();
}

#[test]
fn linear_rope_scaling_prefill_matches_forward() {
    let mut builder = fixture(2, true);
    builder = builder
        .meta("llama.rope.scaling.type", Value::String("linear".into()))
        .meta("llama.rope.scaling.factor", Value::F32(2.5));
    let path = write("linear_rope", &builder);
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    assert_eq!(
        model.config.rope_scaling,
        Some(ops::RopeScaling::Linear { factor: 2.5 })
    );

    let prompt = [1u32, 2, 3, 0];
    let mut state_seq = State::new(&model.config);
    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }

    let mut state_batch = State::new(&model.config);
    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_batch.logits),
        bits(&state_seq.logits),
        "linear rope prefill logits must match sequential forward"
    );

    fs::remove_file(path).ok();
}

#[test]
fn yarn_rope_scaling_prefill_matches_forward() {
    let mut builder = fixture(2, true);
    builder = builder
        .meta("llama.rope.scaling.type", Value::String("yarn".into()))
        .meta("llama.rope.scaling.factor", Value::F32(4.0))
        .meta("llama.rope.scaling.original_context_length", Value::U32(8));
    let path = write("yarn_rope", &builder);
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    assert!(matches!(
        model.config.rope_scaling,
        Some(ops::RopeScaling::Yarn {
            factor,
            orig_ctx: 8,
            ..
        }) if (factor - 4.0).abs() < 1e-6
    ));

    let prompt = [1u32, 2, 3, 0];
    let mut state_seq = State::new(&model.config);
    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }

    let mut state_batch = State::new(&model.config);
    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_batch.logits),
        bits(&state_seq.logits),
        "yarn rope prefill logits must match sequential forward"
    );

    fs::remove_file(path).ok();
}

#[test]
fn partial_rotary_dim_prefill_matches_forward() {
    let mut builder = fixture(2, true);
    builder = builder.meta("llama.rope.dimension_count", Value::U32(0));
    let path = write("partial_rope", &builder);
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");
    assert_eq!(model.config.rope_dim, 0);

    let prompt = [1u32, 2, 3, 0];
    let mut state_seq = State::new(&model.config);
    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }

    let mut state_batch = State::new(&model.config);
    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_batch.logits),
        bits(&state_seq.logits),
        "partial rope prefill logits must match sequential forward"
    );

    fs::remove_file(path).ok();
}

#[test]
fn chunked_prefill_matches_single_pass_and_sequential_forward() {
    let path = write("chunked_prefill", &fixture(2, true));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let prompt = [1u32, 2, 3, 0, 2, 1, 3, 2];

    // 1. Sequential forward
    let mut state_seq = State::new(&model.config);
    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }

    // 2. Full single-pass prefill
    let mut state_full = State::new(&model.config);
    model.prefill(&mut state_full, &prompt, 0).expect("prefill");

    // 3. Chunked prefill with chunk size 2
    let mut state_chunk2 = State::new(&model.config);
    model
        .prefill_chunked(&mut state_chunk2, &prompt, 0, 2)
        .expect("prefill_chunk2");

    // 4. Chunked prefill with uneven chunk size 3 (8 = 3 + 3 + 2)
    let mut state_chunk3 = State::new(&model.config);
    model
        .prefill_chunked(&mut state_chunk3, &prompt, 0, 3)
        .expect("prefill_chunk3");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_full.logits),
        bits(&state_seq.logits),
        "full prefill must match sequential forward"
    );
    assert_eq!(
        bits(&state_chunk2.logits),
        bits(&state_seq.logits),
        "chunked prefill (chunk 2) must match sequential forward"
    );
    assert_eq!(
        bits(&state_chunk3.logits),
        bits(&state_seq.logits),
        "chunked prefill (chunk 3) must match sequential forward"
    );

    fs::remove_file(path).ok();
}

#[test]
fn chunked_prefill_all_logits_matches_full_prefill_all_logits() {
    let path = write("chunked_all_logits", &fixture(2, true));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let prompt = [1u32, 2, 3, 0, 2, 1, 3, 2];
    let vocab = model.config.vocab;

    // Full prefill_all_logits
    let mut all_full = vec![0.0f32; prompt.len() * vocab];
    let mut state_full = State::new(&model.config);
    model
        .prefill_all_logits(&mut state_full, &prompt, 0, &mut all_full)
        .expect("prefill_all");

    // Chunked prefill_all_logits (chunk size 3)
    let mut all_chunked = vec![0.0f32; prompt.len() * vocab];
    let mut state_chunked = State::new(&model.config);
    model
        .prefill_all_logits_chunked(&mut state_chunked, &prompt, 0, &mut all_chunked, 3)
        .expect("prefill_all_chunked");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&all_chunked),
        bits(&all_full),
        "chunked all_logits must match full all_logits exactly"
    );

    fs::remove_file(path).ok();
}

#[test]
fn chunked_prefill_runner_stateful_step_and_progress() {
    let path = write("chunked_runner", &fixture(2, true));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let prompt = vec![1u32, 2, 3, 0, 2];

    // Reference sequential forward
    let mut state_ref = State::new(&model.config);
    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_ref, tok, pos).expect("forward");
    }

    // Stateful runner stepping with chunk size 2
    let mut runner = ferrite::model::ChunkedPrefill::new(&model, prompt.clone(), 0, 2);
    let mut state = State::new(&model.config);

    assert_eq!(runner.total_tokens(), 5);
    assert_eq!(runner.processed_tokens(), 0);
    assert!(!runner.is_finished());

    // Step 1: tokens 0..2
    let step1 = runner.step(&mut state).expect("step 1");
    assert_eq!(step1, 2);
    assert_eq!(runner.processed_tokens(), 2);
    assert!(!runner.is_finished());

    // Step 2: tokens 2..4
    let step2 = runner.step(&mut state).expect("step 2");
    assert_eq!(step2, 2);
    assert_eq!(runner.processed_tokens(), 4);
    assert!(!runner.is_finished());

    // Step 3: tokens 4..5
    let step3 = runner.step(&mut state).expect("step 3");
    assert_eq!(step3, 1);
    assert_eq!(runner.processed_tokens(), 5);
    assert!(runner.is_finished());

    // Step 4: already finished
    let step4 = runner.step(&mut state).expect("step 4");
    assert_eq!(step4, 0);

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state.logits),
        bits(&state_ref.logits),
        "stepwise chunked runner logits must match sequential forward"
    );

    fs::remove_file(path).ok();
}

#[test]
fn quantized_kv_cache_q8_0_forward_and_prefill_match() {
    let path = write("quantized_kv_q8_0", &fixture(2, true));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let prompt = vec![1u32, 2, 3, 0, 2];

    // Q8_0 sequential forward
    let mut state_seq =
        ferrite::model::State::with_kv_cache(&model.config, 64, ferrite::kv::KvCacheType::Q8_0);
    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }

    // Q8_0 batched prefill
    let mut state_batch =
        ferrite::model::State::with_kv_cache(&model.config, 64, ferrite::kv::KvCacheType::Q8_0);
    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    // Sequential forward and batched prefill must produce identical logits on Q8_0 cache
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_seq.logits),
        bits(&state_batch.logits),
        "Q8_0 batched prefill logits must match sequential forward"
    );

    // Compare with F32 baseline to verify bounded numerical tolerance
    let mut state_f32 =
        ferrite::model::State::with_kv_cache(&model.config, 64, ferrite::kv::KvCacheType::F32);
    model.prefill(&mut state_f32, &prompt, 0).expect("prefill");

    let max_diff = state_batch
        .logits
        .iter()
        .zip(&state_f32.logits)
        .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));

    assert!(
        max_diff < 0.05,
        "Q8_0 logit diff {max_diff} exceeds bounded tolerance against F32 baseline"
    );

    fs::remove_file(path).ok();
}

#[test]
fn quantized_kv_cache_q4_0_forward_and_prefill_match() {
    let path = write("quantized_kv_q4_0", &fixture(2, true));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let prompt = vec![1u32, 2, 3, 0, 2];

    // Q4_0 sequential forward
    let mut state_seq =
        ferrite::model::State::with_kv_cache(&model.config, 64, ferrite::kv::KvCacheType::Q4_0);
    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }

    // Q4_0 batched prefill
    let mut state_batch =
        ferrite::model::State::with_kv_cache(&model.config, 64, ferrite::kv::KvCacheType::Q4_0);
    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    // Sequential forward and batched prefill must produce identical logits on Q4_0 cache
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_seq.logits),
        bits(&state_batch.logits),
        "Q4_0 batched prefill logits must match sequential forward"
    );

    // Compare with F32 baseline to verify bounded numerical tolerance
    let mut state_f32 =
        ferrite::model::State::with_kv_cache(&model.config, 64, ferrite::kv::KvCacheType::F32);
    model.prefill(&mut state_f32, &prompt, 0).expect("prefill");

    let max_diff = state_batch
        .logits
        .iter()
        .zip(&state_f32.logits)
        .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));

    assert!(
        max_diff < 0.15,
        "Q4_0 logit diff {max_diff} exceeds bounded tolerance against F32 baseline"
    );

    fs::remove_file(path).ok();
}

#[test]
fn moe_sparse_dispatch_prefill_and_forward_match() {
    let zero = vec![0.0f32; DIM * DIM];
    let heads = 2;
    let kv_heads = 2;
    let kv_dim = kv_heads * (DIM / 2);

    let mut builder = Builder::new()
        .meta("general.architecture", Value::String("mixtral".into()))
        .meta("mixtral.block_count", Value::U32(1))
        .meta("mixtral.embedding_length", Value::U32(DIM as u32))
        .meta("mixtral.feed_forward_length", Value::U32(FFN as u32))
        .meta("mixtral.attention.head_count", Value::U32(heads))
        .meta(
            "mixtral.attention.head_count_kv",
            Value::U32(kv_heads as u32),
        )
        .meta("mixtral.context_length", Value::U32(16))
        .meta("mixtral.attention.layer_norm_rms_epsilon", Value::F32(EPS))
        .meta("mixtral.rope.freq_base", Value::F32(10000.0))
        .meta("mixtral.expert_count", Value::U32(4))
        .meta("mixtral.expert_used_count", Value::U32(2))
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
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&vec![0.0; DIM * kv_dim]),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&zero),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&zero),
        )
        .tensor(
            "blk.0.ffn_norm.weight",
            &[DIM as u64],
            GgmlType::F32,
            bytes(&[1.0; DIM]),
        )
        // Router gate: shape [DIM, expert_count = 4]
        .tensor(
            "blk.0.ffn_gate_inp.weight",
            &[DIM as u64, 4],
            GgmlType::F32,
            bytes(&[
                0.5, -0.2, 0.1, 0.8, -0.4, 0.6, -0.3, 0.2, 0.1, 0.0, 0.7, -0.5, 0.3, 0.9, -0.1, 0.4,
            ]),
        );

    // 4 experts with non-zero weights
    for exp in 0..4 {
        let val = (exp as f32 + 1.0) * 0.1;
        builder = builder
            .tensor(
                &format!("blk.0.ffn_gate.{exp}.weight"),
                &[DIM as u64, FFN as u64],
                GgmlType::F32,
                bytes(&[val; DIM * FFN]),
            )
            .tensor(
                &format!("blk.0.ffn_up.{exp}.weight"),
                &[DIM as u64, FFN as u64],
                GgmlType::F32,
                bytes(&[val * 0.5; DIM * FFN]),
            )
            .tensor(
                &format!("blk.0.ffn_down.{exp}.weight"),
                &[FFN as u64, DIM as u64],
                GgmlType::F32,
                bytes(&[val * 0.25; FFN * DIM]),
            );
    }

    builder = builder
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

    let path = write("moe_test", &builder);
    let gguf = Gguf::open(&path).expect("open fixture");
    let model = Model::load(&gguf).expect("load model");

    assert_eq!(model.config.expert_count, Some(4));
    assert_eq!(model.config.expert_used_count, Some(2));
    assert!(model.layers[0].moe_gate.is_some());

    let mut state_seq = State::new(&model.config);
    let mut state_batch = State::new(&model.config);
    let prompt = [0u32, 1u32, 2u32];

    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }

    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_seq.logits),
        bits(&state_batch.logits),
        "MoE batched prefill logits must exactly match sequential forward"
    );

    let mut all_logits = vec![0.0f32; prompt.len() * model.config.vocab];
    let mut state_all = State::new(&model.config);
    model
        .prefill_all_logits(&mut state_all, &prompt, 0, &mut all_logits)
        .expect("prefill_all_logits");

    let last_start = (prompt.len() - 1) * model.config.vocab;
    let last_logits = &all_logits[last_start..last_start + model.config.vocab];
    assert_eq!(
        bits(&state_seq.logits),
        bits(last_logits),
        "MoE prefill_all_logits final token must match sequential forward"
    );

    fs::remove_file(path).ok();
}

#[test]
fn deepseek_moe_with_shared_experts_forward_and_prefill_match() {
    let zero = vec![0.0f32; DIM * DIM];
    let heads = 2;
    let kv_heads = 2;
    let kv_dim = kv_heads * (DIM / 2);

    let mut builder = Builder::new()
        .meta("general.architecture", Value::String("deepseek2".into()))
        .meta("deepseek2.block_count", Value::U32(1))
        .meta("deepseek2.embedding_length", Value::U32(DIM as u32))
        .meta("deepseek2.feed_forward_length", Value::U32(FFN as u32))
        .meta("deepseek2.attention.head_count", Value::U32(heads))
        .meta(
            "deepseek2.attention.head_count_kv",
            Value::U32(kv_heads as u32),
        )
        .meta("deepseek2.context_length", Value::U32(16))
        .meta(
            "deepseek2.attention.layer_norm_rms_epsilon",
            Value::F32(EPS),
        )
        .meta("deepseek2.rope.freq_base", Value::F32(10000.0))
        .meta("deepseek2.expert_count", Value::U32(4))
        .meta("deepseek2.expert_used_count", Value::U32(2))
        .meta("deepseek2.expert_shared_count", Value::U32(1))
        .meta("tokenizer.ggml.model", Value::String("deepseek2".into()))
        .meta(
            "tokenizer.ggml.tokens",
            Value::Array((0..VOCAB).map(|i| Value::String(format!("t{i}"))).collect()),
        )
        .meta("tokenizer.ggml.merges", Value::Array(Vec::new()))
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
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&vec![0.0; DIM * kv_dim]),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&zero),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&zero),
        )
        .tensor(
            "blk.0.ffn_norm.weight",
            &[DIM as u64],
            GgmlType::F32,
            bytes(&[1.0; DIM]),
        )
        // Router gate: shape [DIM, expert_count = 4]
        .tensor(
            "blk.0.ffn_gate_inp.weight",
            &[DIM as u64, 4],
            GgmlType::F32,
            bytes(&[
                0.5, -0.2, 0.1, 0.8, -0.4, 0.6, -0.3, 0.2, 0.1, 0.0, 0.7, -0.5, 0.3, 0.9, -0.1, 0.4,
            ]),
        );

    // 4 routed experts
    for exp in 0..4 {
        let val = (exp as f32 + 1.0) * 0.1;
        builder = builder
            .tensor(
                &format!("blk.0.ffn_gate.{exp}.weight"),
                &[DIM as u64, FFN as u64],
                GgmlType::F32,
                bytes(&[val; DIM * FFN]),
            )
            .tensor(
                &format!("blk.0.ffn_up.{exp}.weight"),
                &[DIM as u64, FFN as u64],
                GgmlType::F32,
                bytes(&[val * 0.5; DIM * FFN]),
            )
            .tensor(
                &format!("blk.0.ffn_down.{exp}.weight"),
                &[FFN as u64, DIM as u64],
                GgmlType::F32,
                bytes(&[val * 0.25; FFN * DIM]),
            );
    }

    // Shared expert weights (always evaluated and added to routed top-k experts)
    builder = builder
        .tensor(
            "blk.0.ffn_gate_shexp.weight",
            &[DIM as u64, FFN as u64],
            GgmlType::F32,
            bytes(&[0.2f32; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_up_shexp.weight",
            &[DIM as u64, FFN as u64],
            GgmlType::F32,
            bytes(&[0.3f32; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_down_shexp.weight",
            &[FFN as u64, DIM as u64],
            GgmlType::F32,
            bytes(&[0.1f32; FFN * DIM]),
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

    let path = write("deepseek2_moe_test", &builder);
    let gguf = Gguf::open(&path).expect("open fixture");
    let model = Model::load(&gguf).expect("load model");
    let tok = ferrite::Tokenizer::from_gguf(&gguf).expect("load deepseek tokenizer");

    assert_eq!(tok.kind, ferrite::tokenizer::Kind::Bpe);
    assert_eq!(model.config.expert_count, Some(4));
    assert_eq!(model.config.expert_used_count, Some(2));
    assert_eq!(model.config.expert_shared_count, Some(1));
    assert!(model.layers[0].moe_gate_shexp.is_some());

    let mut state_seq = State::new(&model.config);
    let mut state_batch = State::new(&model.config);
    let prompt = [0u32, 1u32, 2u32];

    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }

    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_seq.logits),
        bits(&state_batch.logits),
        "DeepSeek MoE with shared experts batched prefill logits must match sequential forward"
    );

    let mut all_logits = vec![0.0f32; prompt.len() * model.config.vocab];
    let mut state_all = State::new(&model.config);
    model
        .prefill_all_logits(&mut state_all, &prompt, 0, &mut all_logits)
        .expect("prefill_all_logits");

    let last_start = (prompt.len() - 1) * model.config.vocab;
    let last_logits = &all_logits[last_start..last_start + model.config.vocab];
    assert_eq!(
        bits(&state_seq.logits),
        bits(last_logits),
        "DeepSeek MoE prefill_all_logits final token must match sequential forward"
    );

    fs::remove_file(path).ok();
}

#[test]
fn deepseek_mla_forward_and_prefill_match() {
    let q_rank = 4;
    let kv_rank = 4;
    let nope_dim = 2;
    let rope_dim = 2;
    let v_hdim = 2;
    let heads = 2;
    let head_dim = nope_dim + rope_dim; // 4
    let q_dim = heads * head_dim; // 8
    let kv_a_dim = kv_rank + rope_dim; // 6
    let kv_b_dim = heads * (nope_dim + v_hdim); // 8
    let v_dim = heads * v_hdim; // 4

    let mut q_a = vec![0.0f32; q_rank * DIM];
    for i in 0..q_rank.min(DIM) {
        q_a[i * DIM + i] = 1.0;
    }
    let mut q_b = vec![0.0f32; q_dim * q_rank];
    for i in 0..q_dim.min(q_rank) {
        q_b[i * q_rank + i] = 1.0;
    }
    let mut kv_a = vec![0.0f32; kv_a_dim * DIM];
    for i in 0..kv_a_dim.min(DIM) {
        kv_a[i * DIM + i] = 1.0;
    }
    let mut kv_b = vec![0.0f32; kv_b_dim * kv_rank];
    for i in 0..kv_b_dim.min(kv_rank) {
        kv_b[i * kv_rank + i] = 1.0;
    }
    let mut wo = vec![0.0f32; DIM * v_dim];
    for i in 0..DIM.min(v_dim) {
        wo[i * v_dim + i] = 1.0;
    }

    let builder = Builder::new()
        .meta("general.architecture", Value::String("deepseek2".into()))
        .meta("deepseek2.block_count", Value::U32(1))
        .meta("deepseek2.embedding_length", Value::U32(DIM as u32))
        .meta("deepseek2.feed_forward_length", Value::U32(FFN as u32))
        .meta("deepseek2.attention.head_count", Value::U32(heads as u32))
        .meta(
            "deepseek2.attention.head_count_kv",
            Value::U32(heads as u32),
        )
        .meta(
            "deepseek2.attention.key_length",
            Value::U32(head_dim as u32),
        )
        .meta(
            "deepseek2.attention.value_length",
            Value::U32(v_hdim as u32),
        )
        .meta("deepseek2.attention.q_lora_rank", Value::U32(q_rank as u32))
        .meta(
            "deepseek2.attention.kv_lora_rank",
            Value::U32(kv_rank as u32),
        )
        .meta(
            "deepseek2.attention.qk_nope_head_dim",
            Value::U32(nope_dim as u32),
        )
        .meta(
            "deepseek2.attention.qk_rope_head_dim",
            Value::U32(rope_dim as u32),
        )
        .meta("deepseek2.context_length", Value::U32(16))
        .meta(
            "deepseek2.attention.layer_norm_rms_epsilon",
            Value::F32(EPS),
        )
        .meta("deepseek2.rope.freq_base", Value::F32(10000.0))
        .meta("tokenizer.ggml.model", Value::String("gpt2".into()))
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
            "blk.0.attn_q_a.weight",
            &[DIM as u64, q_rank as u64],
            GgmlType::F32,
            bytes(&q_a),
        )
        .tensor(
            "blk.0.attn_q_a_norm.weight",
            &[q_rank as u64],
            GgmlType::F32,
            bytes(&vec![1.0; q_rank]),
        )
        .tensor(
            "blk.0.attn_q_b.weight",
            &[q_rank as u64, q_dim as u64],
            GgmlType::F32,
            bytes(&q_b),
        )
        .tensor(
            "blk.0.attn_kv_a.weight",
            &[DIM as u64, kv_a_dim as u64],
            GgmlType::F32,
            bytes(&kv_a),
        )
        .tensor(
            "blk.0.attn_kv_a_norm.weight",
            &[kv_rank as u64],
            GgmlType::F32,
            bytes(&vec![1.0; kv_rank]),
        )
        .tensor(
            "blk.0.attn_kv_b.weight",
            &[kv_rank as u64, kv_b_dim as u64],
            GgmlType::F32,
            bytes(&kv_b),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[v_dim as u64, DIM as u64],
            GgmlType::F32,
            bytes(&wo),
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
            bytes(&identity(DIM)),
        );

    let path = write("deepseek2_mla_test", &builder);
    let gguf = Gguf::open(&path).expect("open fixture");
    let model = Model::load(&gguf).expect("load model");

    assert_eq!(model.config.q_lora_rank, Some(4));
    assert_eq!(model.config.kv_lora_rank, Some(4));
    assert_eq!(model.config.qk_nope_head_dim, Some(2));
    assert_eq!(model.config.qk_rope_head_dim, Some(2));
    assert_eq!(model.config.v_head_dim, Some(2));
    assert!(model.layers[0].attn_kv_a.is_some());
    assert!(model.layers[0].attn_kv_b.is_some());

    let mut state_seq = State::new(&model.config);
    let mut state_batch = State::new(&model.config);
    let prompt = [0u32, 1u32, 2u32];

    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }

    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_seq.logits),
        bits(&state_batch.logits),
        "DeepSeek MLA batched prefill logits must match sequential forward"
    );

    fs::remove_file(path).ok();
}

#[test]
fn qwen2_vl_mrope_forward_and_prefill_match() {
    let heads = 2;
    let head_dim = DIM / heads; // 2

    let builder = Builder::new()
        .meta("general.architecture", Value::String("qwen2vl".into()))
        .meta("qwen2vl.block_count", Value::U32(1))
        .meta("qwen2vl.embedding_length", Value::U32(DIM as u32))
        .meta("qwen2vl.feed_forward_length", Value::U32(FFN as u32))
        .meta("qwen2vl.attention.head_count", Value::U32(heads as u32))
        .meta("qwen2vl.attention.head_count_kv", Value::U32(heads as u32))
        .meta("qwen2vl.attention.key_length", Value::U32(head_dim as u32))
        .meta("qwen2vl.context_length", Value::U32(16))
        .meta("qwen2vl.attention.layer_norm_rms_epsilon", Value::F32(EPS))
        .meta("qwen2vl.rope.freq_base", Value::F32(1_000_000.0))
        .meta(
            "qwen2vl.rope.dimension_sections",
            Value::Array(vec![
                Value::U32(1),
                Value::U32(0),
                Value::U32(0),
                Value::U32(0),
            ]),
        )
        .meta("tokenizer.ggml.model", Value::String("gpt2".into()))
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
            "blk.0.attn_q.bias",
            &[DIM as u64],
            GgmlType::F32,
            bytes(&[0.1; DIM]),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&identity(DIM)),
        )
        .tensor(
            "blk.0.attn_k.bias",
            &[DIM as u64],
            GgmlType::F32,
            bytes(&[0.05; DIM]),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&identity(DIM)),
        )
        .tensor(
            "blk.0.attn_v.bias",
            &[DIM as u64],
            GgmlType::F32,
            bytes(&[0.02; DIM]),
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
            bytes(&identity(DIM)),
        );

    let path = write("qwen2_vl_mrope_test", &builder);
    let gguf = Gguf::open(&path).expect("open fixture");
    let model = Model::load(&gguf).expect("load model");

    assert_eq!(model.config.mrope_sections, Some([1, 0, 0, 0]));

    let mut state_seq = State::new(&model.config);
    let mut state_batch = State::new(&model.config);
    let prompt = [0u32, 1u32, 2u32];

    for (pos, &tok) in prompt.iter().enumerate() {
        model.forward(&mut state_seq, tok, pos).expect("forward");
    }

    model
        .prefill(&mut state_batch, &prompt, 0)
        .expect("prefill");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_seq.logits),
        bits(&state_batch.logits),
        "Qwen2-VL M-RoPE batched prefill logits must match sequential forward"
    );

    fs::remove_file(path).ok();
}
