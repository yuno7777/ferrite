//! Write a small but complete llama-architecture GGUF, so every command has
//! something to run against without downloading a model.
//!
//! ```text
//! cargo run --example fixture -- tiny.gguf
//! cargo run -- info tiny.gguf -t
//! cargo run -- tokenize tiny.gguf "hello world"
//! cargo run -- run tiny.gguf "hello" -n 8 --temp 0.8
//! ```
//!
//! The weights are pseudo-random, so generation is gibberish by construction.
//! What it demonstrates is that the pipeline runs: parse, tokenize, forward,
//! sample, decode.

use ferrite::gguf::{GgmlType, Value};
use ferrite::synth::Builder;

const EMBEDDING: usize = 32;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const FFN: usize = 64;
const LAYERS: usize = 2;

/// Deterministic weights in roughly [-0.5, 0.5]. A fixed sequence keeps the
/// file byte-identical between runs, which makes it usable in tests.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }

    fn tensor(&mut self, len: usize) -> Vec<u8> {
        (0..len).flat_map(|_| self.next().to_le_bytes()).collect()
    }
}

fn ones(len: usize) -> Vec<u8> {
    (0..len).flat_map(|_| 1.0f32.to_le_bytes()).collect()
}

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tiny.gguf".to_string());

    // A byte-level vocab: a space is stored as U+0120, and the merge list is
    // what lets "hello" and " world" come back as single tokens.
    let vocab = [
        "<s>",
        "</s>",
        "h",
        "e",
        "l",
        "o",
        "\u{0120}",
        "w",
        "r",
        "d",
        "ll",
        "he",
        "hell",
        "hello",
        "\u{0120}w",
        "or",
        "\u{0120}wor",
        "\u{0120}worl",
        "\u{0120}world",
    ];
    let merges = [
        "l l",
        "h e",
        "he ll",
        "hell o",
        "\u{0120} w",
        "o r",
        "\u{0120}w or",
        "\u{0120}wor l",
        "\u{0120}worl d",
    ];
    let mut token_types = vec![Value::I32(1); vocab.len()];
    token_types[0] = Value::I32(3); // <s> and </s> are control tokens
    token_types[1] = Value::I32(3);

    let head_dim = EMBEDDING / HEADS;
    let kv_dim = KV_HEADS * head_dim;
    let mut noise = Noise(0x5EED);

    let mut builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("general.name", Value::String("ferrite-fixture".into()))
        .meta("llama.block_count", Value::U32(LAYERS as u32))
        .meta("llama.embedding_length", Value::U32(EMBEDDING as u32))
        .meta("llama.feed_forward_length", Value::U32(FFN as u32))
        .meta("llama.attention.head_count", Value::U32(HEADS as u32))
        .meta("llama.attention.head_count_kv", Value::U32(KV_HEADS as u32))
        .meta("llama.context_length", Value::U32(512))
        .meta("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5))
        .meta("llama.rope.freq_base", Value::F32(10000.0))
        .meta("tokenizer.ggml.model", Value::String("gpt2".into()))
        .meta(
            "tokenizer.ggml.tokens",
            Value::Array(vocab.iter().map(|t| Value::String((*t).into())).collect()),
        )
        .meta("tokenizer.ggml.token_type", Value::Array(token_types))
        .meta(
            "tokenizer.ggml.merges",
            Value::Array(merges.iter().map(|m| Value::String((*m).into())).collect()),
        )
        .meta("tokenizer.ggml.bos_token_id", Value::U32(0))
        .meta("tokenizer.ggml.eos_token_id", Value::U32(1))
        .tensor(
            "token_embd.weight",
            &[EMBEDDING as u64, vocab.len() as u64],
            GgmlType::F32,
            noise.tensor(EMBEDDING * vocab.len()),
        );

    for layer in 0..LAYERS {
        builder = builder
            .tensor(
                &format!("blk.{layer}.attn_norm.weight"),
                &[EMBEDDING as u64],
                GgmlType::F32,
                ones(EMBEDDING),
            )
            .tensor(
                &format!("blk.{layer}.attn_q.weight"),
                &[EMBEDDING as u64, EMBEDDING as u64],
                GgmlType::F32,
                noise.tensor(EMBEDDING * EMBEDDING),
            )
            // Key and value project down to the smaller grouped-query width.
            .tensor(
                &format!("blk.{layer}.attn_k.weight"),
                &[EMBEDDING as u64, kv_dim as u64],
                GgmlType::F32,
                noise.tensor(EMBEDDING * kv_dim),
            )
            .tensor(
                &format!("blk.{layer}.attn_v.weight"),
                &[EMBEDDING as u64, kv_dim as u64],
                GgmlType::F32,
                noise.tensor(EMBEDDING * kv_dim),
            )
            .tensor(
                &format!("blk.{layer}.attn_output.weight"),
                &[EMBEDDING as u64, EMBEDDING as u64],
                GgmlType::F32,
                noise.tensor(EMBEDDING * EMBEDDING),
            )
            .tensor(
                &format!("blk.{layer}.ffn_norm.weight"),
                &[EMBEDDING as u64],
                GgmlType::F32,
                ones(EMBEDDING),
            )
            .tensor(
                &format!("blk.{layer}.ffn_gate.weight"),
                &[EMBEDDING as u64, FFN as u64],
                GgmlType::F32,
                noise.tensor(EMBEDDING * FFN),
            )
            .tensor(
                &format!("blk.{layer}.ffn_up.weight"),
                &[EMBEDDING as u64, FFN as u64],
                GgmlType::F32,
                noise.tensor(EMBEDDING * FFN),
            )
            .tensor(
                &format!("blk.{layer}.ffn_down.weight"),
                &[FFN as u64, EMBEDDING as u64],
                GgmlType::F32,
                noise.tensor(FFN * EMBEDDING),
            );
    }

    builder = builder
        .tensor(
            "output_norm.weight",
            &[EMBEDDING as u64],
            GgmlType::F32,
            ones(EMBEDDING),
        )
        .tensor(
            "output.weight",
            &[EMBEDDING as u64, vocab.len() as u64],
            GgmlType::F32,
            noise.tensor(EMBEDDING * vocab.len()),
        );

    builder.write(std::path::Path::new(&path))?;
    println!("wrote {path}");
    Ok(())
}
