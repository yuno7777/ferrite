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

    let vocab = ["<s>", "</s>", "hello", " world"];
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
