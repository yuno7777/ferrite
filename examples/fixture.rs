//! Write a small valid GGUF so you can exercise the CLI without downloading a
//! model.
//!
//! ```text
//! cargo run --example fixture -- tiny.gguf
//! cargo run -- info tiny.gguf -t
//! ```

use ferrite::gguf::{GgmlType, Value};
use ferrite::synth::Builder;

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tiny.gguf".to_string());

    // A byte-level vocab: a space is stored as U+0120, and the merge list is
    // what lets "hello" and " world" come back out as single tokens.
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
    token_types[0] = Value::I32(3); // <s> is a control token
    token_types[1] = Value::I32(3);

    let embedding = 256u64;
    let layers = 2u64;

    let mut b = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("general.name", Value::String("ferrite-fixture".into()))
        .meta("llama.block_count", Value::U32(layers as u32))
        .meta("llama.embedding_length", Value::U32(embedding as u32))
        .meta(
            "llama.feed_forward_length",
            Value::U32(embedding as u32 * 4),
        )
        .meta("llama.attention.head_count", Value::U32(8))
        .meta("llama.attention.head_count_kv", Value::U32(2))
        .meta("llama.context_length", Value::U32(2048))
        .meta("llama.rope.freq_base", Value::F32(10000.0))
        .meta("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5))
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
            &[embedding, vocab.len() as u64],
            GgmlType::F32,
            vec![0u8; (embedding as usize * vocab.len()) * 4],
        );

    // One Q4_K block per layer, so the quant histogram has something to show.
    for layer in 0..layers {
        b = b.tensor(
            &format!("blk.{layer}.attn_q.weight"),
            &[embedding, embedding],
            GgmlType::Q4_K,
            vec![0u8; (embedding * embedding / 256 * 144) as usize],
        );
    }
    b = b.tensor(
        "output_norm.weight",
        &[embedding],
        GgmlType::F32,
        vec![0u8; embedding as usize * 4],
    );

    b.write(std::path::Path::new(&path))?;
    println!("wrote {path}");
    Ok(())
}
