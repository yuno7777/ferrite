# Ferrite Library Usage Guide

Ferrite can be consumed directly as a pure-Rust library with zero dependencies. This guide demonstrates how to interact with models programmatically using the public API.

## Adding Ferrite to Your Project

Add Ferrite as a path or git dependency in your `Cargo.toml`:

```toml
[dependencies]
ferrite = { path = "../ferrite" }
```

---

## 1. Opening a GGUF Container and Inspecting Metadata

The `Gguf` struct memory-maps the container and indexes all metadata keys and tensor descriptors:

```rust
use std::path::Path;
use ferrite::gguf::Gguf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("model.gguf");
    let gguf = Gguf::open(path)?;

    // Read high-level model properties
    println!("Architecture: {:?}", gguf.arch());
    println!("Alignment: {} bytes", gguf.alignment);
    println!("Tensors count: {}", gguf.tensors.len());

    // Inspect hyperparameters
    if let Some(layers) = gguf.layer_count() {
        println!("Transformer layers: {layers}");
    }
    if let Some(dim) = gguf.embedding_length() {
        println!("Embedding dimension: {dim}");
    }
    if let Some(heads) = gguf.head_count() {
        println!("Attention heads: {heads}");
    }

    Ok(())
}
```

---

## 2. Tokenizing and Detokenizing Text

Construct a `Tokenizer` directly from the model container's embedded vocabulary:

```rust
use ferrite::gguf::Gguf;
use ferrite::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = Gguf::open("model.gguf")?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    // Encode text into token IDs (with optional BOS token)
    let add_bos = true;
    let tokens = tokenizer.encode("Hello, Ferrite!", add_bos);
    println!("Tokens: {:?}", tokens);

    // Decode individual pieces
    for &id in &tokens {
        if let Some(piece) = tokenizer.decode_piece(id) {
            print!("{piece}");
        }
    }
    println!();

    Ok(())
}
```

---

## 3. Loading a Model and Initializing State

Load the transformer weights and initialize an execution `State` allocating the KV cache:

```rust
use ferrite::gguf::Gguf;
use ferrite::model::{Model, State};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = Gguf::open("model.gguf")?;
    let model = Model::load(&gguf)?;

    // Allocate runtime state:
    // - Context length: 2048 tokens
    // - Thread pool size: 8 workers
    let ctx_len = 2048;
    let threads = 8;
    let mut state = State::new(&model, ctx_len, threads);

    // Optional: toggle integer activation quantization (on by default)
    state.set_int8(true);

    println!("KV cache allocated: {:.2} MB", state.kv.size_bytes() as f64 / 1_048_576.0);

    Ok(())
}
```

---

## 4. Ingesting Prompts via Batched Prefill

Process the input prompt tokens using batched prefill to compute and store initial KV representations:

```rust
use ferrite::gguf::Gguf;
use ferrite::model::{Model, State};
use ferrite::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = Gguf::open("model.gguf")?;
    let model = Model::load(&gguf)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let mut state = State::new(&model, 1024, 4);

    let prompt = "Explain CPU cache hierarchies:";
    let prompt_tokens = tokenizer.encode(prompt, true);

    // Prefill evaluates all prompt tokens using compute-bound batched GEMM
    // Returns logits for the final token in the prompt
    let mut logits = model.prefill(&mut state, &prompt_tokens)?;

    println!("Prefill completed. Logits size: {}", logits.len());

    Ok(())
}
```

---

## 5. Sampling and Autoregressive Decoding

Generate tokens one by one using `Sampler`:

```rust
use ferrite::gguf::Gguf;
use ferrite::model::{Model, State};
use ferrite::sample::Sampler;
use ferrite::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = Gguf::open("model.gguf")?;
    let model = Model::load(&gguf)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let mut state = State::new(&model, 512, 4);

    let prompt_tokens = tokenizer.encode("Once upon a time", true);
    let mut logits = model.prefill(&mut state, &prompt_tokens)?;

    // Configure sampler:
    // - Temperature: 0.7 (0.0 for greedy)
    // - Top-K: 40
    // - Top-P: 0.95
    // - Seed: 42
    let mut sampler = Sampler::new(0.7, 40, 0.95, 42);

    let max_new_tokens = 32;
    for _ in 0..max_new_tokens {
        // Sample next token ID from output logits
        let next_token = sampler.sample(&mut logits);

        // Check for EOS
        if Some(next_token) == model.config.eos_token_id {
            break;
        }

        // Print token string
        if let Some(piece) = tokenizer.decode_piece(next_token) {
            print!("{piece}");
        }

        // Single-token autoregressive forward step
        logits = model.forward(&mut state, next_token)?;
    }
    println!();

    Ok(())
}
```

---

## 6. Speculative Decoding Acceleration

Accelerate generation with a smaller draft model using the speculative decoding engine:

```rust
use ferrite::gguf::Gguf;
use ferrite::model::{Model, State};
use ferrite::speculative::SpeculativeEngine;

fn run_speculative(
    target_gguf: &Gguf,
    draft_gguf: &Gguf,
    prompt: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let target = Model::load(target_gguf)?;
    let draft = Model::load(draft_gguf)?;

    let mut target_state = State::new(&target, 2048, 8);
    let mut draft_state = State::new(&draft, 2048, 8);

    // K = 3 candidate draft tokens per verification step
    let lookahead_k = 3;
    let mut engine = SpeculativeEngine::new(&target, &draft, lookahead_k);

    let generated_tokens = engine.generate(
        &mut target_state,
        &mut draft_state,
        prompt,
        64,   // max tokens
        0.0,  // temperature
    )?;

    println!("Accepted tokens: {}", generated_tokens.len());
    println!("Acceptance rate: {:.1}%", engine.stats.acceptance_rate() * 100.0);

    Ok(())
}
```
