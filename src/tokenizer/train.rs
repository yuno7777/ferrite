//! Pure-Rust Byte-Pair Encoding (BPE) vocabulary training.
//!
//! Trains a BPE vocabulary directly from an input text corpus and writes a
//! GGUF-compatible vocabulary file that can be loaded directly by [`crate::Tokenizer::from_gguf`].

use std::collections::HashMap;
use std::fs;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;

use crate::gguf::Value;
use crate::synth::Builder;
use crate::tokenizer::bpe;
use crate::tokenizer::bytes;

/// Trained vocabulary artifacts.
#[derive(Debug, Clone)]
pub struct TrainedVocab {
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub kinds: Vec<u32>,
    pub merges: Vec<String>,
}

/// Configuration for BPE vocabulary training.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub vocab_size: usize,
    pub special_tokens: Vec<String>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1000,
            special_tokens: vec!["<unk>".into(), "<s>".into(), "</s>".into()],
        }
    }
}

/// Train a byte-level BPE vocabulary from raw text.
pub fn train_bpe(corpus: &str, config: &TrainConfig) -> Result<TrainedVocab> {
    if config.vocab_size < config.special_tokens.len() + 256 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "vocab_size {} is too small for {} special tokens and 256 byte tokens",
                config.vocab_size,
                config.special_tokens.len()
            ),
        ));
    }

    let mut tokens = Vec::with_capacity(config.vocab_size);
    let mut scores = Vec::with_capacity(config.vocab_size);
    let mut kinds = Vec::with_capacity(config.vocab_size);
    let mut merges = Vec::new();

    // 1. Special tokens (Unknown = 2, Control = 3)
    for tok in &config.special_tokens {
        tokens.push(tok.clone());
        scores.push(0.0);
        kinds.push(if tok == "<unk>" { 2 } else { 3 });
    }

    // 2. Base byte tokens
    for b in 0u8..=255 {
        let ch = bytes::encode_byte(b);
        tokens.push(ch.to_string());
        scores.push(0.0);
        kinds.push(1); // Normal
    }

    struct Word {
        symbols: Vec<String>,
        freq: usize,
    }

    // 3. Pretokenize and count words
    let mut word_counts: HashMap<Vec<String>, usize> = HashMap::new();
    for chunk in bpe::pretokenize(corpus) {
        let encoded = bytes::encode_str(chunk);
        let symbols: Vec<String> = encoded.chars().map(|c| c.to_string()).collect();
        if !symbols.is_empty() {
            *word_counts.entry(symbols).or_insert(0) += 1;
        }
    }

    let mut words: Vec<Word> = word_counts
        .into_iter()
        .map(|(symbols, freq)| Word { symbols, freq })
        .collect();

    // 4. Iteratively extract most frequent pairs
    while tokens.len() < config.vocab_size {
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        for word in &words {
            if word.symbols.len() < 2 {
                continue;
            }
            for i in 0..word.symbols.len() - 1 {
                *pair_counts
                    .entry((word.symbols[i].clone(), word.symbols[i + 1].clone()))
                    .or_default() += word.freq;
            }
        }

        if pair_counts.is_empty() {
            break;
        }

        // Deterministic selection: max frequency, tie-break by string order
        let (best_pair, best_count) = pair_counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)))
            .unwrap();

        if best_count == 0 {
            break;
        }

        let merged = format!("{}{}", best_pair.0, best_pair.1);
        let merge_spec = format!("{} {}", best_pair.0, best_pair.1);

        tokens.push(merged.clone());
        scores.push(best_count as f32);
        kinds.push(1);
        merges.push(merge_spec);

        // Update word representations in-place
        for word in words.iter_mut() {
            if word.symbols.len() < 2 {
                continue;
            }
            let mut new_symbols = Vec::with_capacity(word.symbols.len());
            let mut i = 0;
            while i < word.symbols.len() {
                if i + 1 < word.symbols.len()
                    && word.symbols[i] == best_pair.0
                    && word.symbols[i + 1] == best_pair.1
                {
                    new_symbols.push(merged.clone());
                    i += 2;
                } else {
                    new_symbols.push(word.symbols[i].clone());
                    i += 1;
                }
            }
            word.symbols = new_symbols;
        }
    }

    Ok(TrainedVocab {
        tokens,
        scores,
        kinds,
        merges,
    })
}

/// Write a trained vocabulary directly into a GGUF file.
pub fn write_vocab_gguf(vocab: &TrainedVocab, out_path: &Path) -> Result<()> {
    let builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("tokenizer.ggml.model", Value::String("gpt2".into()))
        .meta(
            "tokenizer.ggml.tokens",
            Value::Array(
                vocab
                    .tokens
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect(),
            ),
        )
        .meta(
            "tokenizer.ggml.scores",
            Value::Array(vocab.scores.iter().map(|s| Value::F32(*s)).collect()),
        )
        .meta(
            "tokenizer.ggml.token_type",
            Value::Array(vocab.kinds.iter().map(|k| Value::U32(*k)).collect()),
        )
        .meta(
            "tokenizer.ggml.merges",
            Value::Array(
                vocab
                    .merges
                    .iter()
                    .map(|m| Value::String(m.clone()))
                    .collect(),
            ),
        )
        .meta("tokenizer.ggml.bos_token_id", Value::U32(1))
        .meta("tokenizer.ggml.eos_token_id", Value::U32(2))
        .meta("tokenizer.ggml.unknown_token_id", Value::U32(0));

    builder.write(out_path)
}

/// Train a vocabulary from a text file and write the resulting GGUF vocabulary model.
pub fn train_vocab_file(
    corpus_path: &Path,
    out_path: &Path,
    config: &TrainConfig,
) -> Result<TrainedVocab> {
    let text = fs::read_to_string(corpus_path)?;
    let vocab = train_bpe(&text, config)?;
    write_vocab_gguf(&vocab, out_path)?;
    Ok(vocab)
}
