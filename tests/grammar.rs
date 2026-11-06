use std::fs;
use std::path::PathBuf;

use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::grammar::{ChoiceGrammar, Grammar, JsonGrammar};
use ferrite::sample::Sampler;
use ferrite::synth::Builder;
use ferrite::tokenizer::Tokenizer;

fn fixture() -> Builder {
    let tokens = vec![
        "t0".to_string(),       // id 0
        "1".to_string(),        // id 1
        "2".to_string(),        // id 2
        "a".to_string(),        // id 3
        "\"name\"".to_string(), // id 4
        ":".to_string(),        // id 5
        "true".to_string(),     // id 6
        "{".to_string(),        // id 7
        "}".to_string(),        // id 8
    ];
    let vocab_len = tokens.len();
    let dim = 4;

    Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(1))
        .meta("llama.embedding_length", Value::U32(dim as u32))
        .meta("llama.feed_forward_length", Value::U32(8))
        .meta("llama.attention.head_count", Value::U32(1))
        .meta("llama.attention.head_count_kv", Value::U32(1))
        .meta("llama.context_length", Value::U32(16))
        .meta("tokenizer.ggml.model", Value::String("llama".into()))
        .meta(
            "tokenizer.ggml.tokens",
            Value::Array(tokens.into_iter().map(Value::String).collect()),
        )
        .meta("tokenizer.ggml.bos_token_id", Value::U32(0))
        .meta("tokenizer.ggml.eos_token_id", Value::U32(0))
        .tensor(
            "token_embd.weight",
            &[dim as u64, vocab_len as u64],
            GgmlType::F32,
            vec![0u8; dim * vocab_len * 4],
        )
}

fn temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "ferrite-test-grammar-{}-{name}.gguf",
        std::process::id()
    ));
    path
}

#[test]
fn test_regex_grammar_logit_filtering() {
    let path = temp_path("regex");
    fs::write(&path, fixture().build()).unwrap();
    let gguf = Gguf::open(&path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();

    let mut grammar = Grammar::regex(r"[0-9]+").unwrap();
    let mut logits = vec![1.0f32; tok.tokens.len()];

    grammar.filter_logits(&mut logits, &tok);

    // Tokens '1' (id 1) and '2' (id 2) must be preserved
    assert!(logits[1] > -100.0);
    assert!(logits[2] > -100.0);

    // Token 'a' (id 3) must be masked to -INFINITY
    assert_eq!(logits[3], -f32::INFINITY);

    // EOS (id 0) must be masked because [0-9]+ has not matched any digit yet
    assert_eq!(logits[0], -f32::INFINITY);

    // Advance with token 1 ('1')
    assert!(grammar.advance_token(1, &tok));
    assert!(grammar.is_accepting());

    // Filter logits again; now EOS (id 0) should be allowed because regex is satisfied
    let mut logits2 = vec![1.0f32; tok.tokens.len()];
    grammar.filter_logits(&mut logits2, &tok);
    assert!(logits2[0] > -100.0); // EOS allowed
    assert!(logits2[1] > -100.0); // More digits allowed
    assert_eq!(logits2[3], -f32::INFINITY); // 'a' still prohibited

    let _ = fs::remove_file(&path);
}

#[test]
fn test_json_grammar_flow() {
    let mut json = JsonGrammar::new();

    // Start JSON document: '{'
    assert!(json.advance(b"{"));
    assert!(!json.is_accepting());

    // Key: '"name"'
    assert!(json.advance(b"\"name\""));
    assert!(!json.is_accepting());

    // Colon: ':'
    assert!(json.advance(b":"));
    assert!(!json.is_accepting());

    // Value: '123'
    assert!(json.advance(b"123"));
    assert!(!json.is_accepting());

    // Closing brace: '}'
    assert!(json.advance(b"}"));
    assert!(json.is_accepting());

    // Invalid trailing characters
    assert!(!json.advance(b"invalid"));
}

#[test]
fn test_choice_grammar_guided_sampling() {
    let grammar = ChoiceGrammar::new(&["true", "false"]);
    let mut g = Grammar::Choice(grammar);

    let path = temp_path("choice");
    fs::write(&path, fixture().build()).unwrap();
    let gguf = Gguf::open(&path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();

    let mut logits = vec![0.0f32; tok.tokens.len()];
    logits[3] = 10.0; // token 'a' has highest initial logit
    logits[6] = 2.0; // token 'true' has lower logit

    g.filter_logits(&mut logits, &tok);

    // Token 'a' must be masked out despite having highest initial logit
    assert_eq!(logits[3], -f32::INFINITY);

    // Token 'true' must remain valid
    assert_eq!(logits[6], 2.0);

    let mut sampler = Sampler::greedy();
    let selected = sampler.sample(&mut logits);
    assert_eq!(selected, 6); // Selected token 'true'

    assert!(g.advance_token(selected, &tok));
    assert!(g.is_accepting());

    let _ = fs::remove_file(&path);
}
