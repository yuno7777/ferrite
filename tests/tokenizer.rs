//! Tokenizer tests against hand-built vocabularies.
//!
//! Each test writes a small GGUF carrying its own vocab, so the exact expected
//! token ids are known rather than guessed at.

use std::fs;
use std::path::{Path, PathBuf};

use ferrite::gguf::{Gguf, Value};
use ferrite::synth::Builder;
use ferrite::tokenizer::{Kind, Tokenizer};

const NORMAL: i32 = 1;
const UNKNOWN: i32 = 2;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const BYTE: i32 = 6;

fn write(name: &str, builder: &Builder) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ferrite_tok_{name}.gguf"));
    builder.write(&path).expect("write fixture");
    path
}

fn strings(items: &[&str]) -> Value {
    Value::Array(items.iter().map(|s| Value::String((*s).into())).collect())
}

fn ints(items: &[i32]) -> Value {
    Value::Array(items.iter().map(|v| Value::I32(*v)).collect())
}

fn floats(items: &[f32]) -> Value {
    Value::Array(items.iter().map(|v| Value::F32(*v)).collect())
}

/// Byte-level BPE vocab that can build exactly "hello" and " world".
fn bpe_fixture(add_bos: bool) -> Builder {
    let tokens = [
        "<s>",
        "</s>",
        "<|special|>",
        "h",
        "e",
        "l",
        "o",
        "\u{0120}", // the stored form of a space
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
    let mut types = vec![NORMAL; tokens.len()];
    types[0] = CONTROL;
    types[1] = CONTROL;
    types[2] = USER_DEFINED;

    Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("tokenizer.ggml.model", Value::String("gpt2".into()))
        .meta("tokenizer.ggml.tokens", strings(&tokens))
        .meta("tokenizer.ggml.token_type", ints(&types))
        .meta(
            "tokenizer.ggml.merges",
            strings(&[
                "l l",
                "h e",
                "he ll",
                "hell o",
                "\u{0120} w",
                "o r",
                "\u{0120}w or",
                "\u{0120}wor l",
                "\u{0120}worl d",
            ]),
        )
        .meta("tokenizer.ggml.bos_token_id", Value::U32(0))
        .meta("tokenizer.ggml.eos_token_id", Value::U32(1))
        .meta("tokenizer.ggml.add_bos_token", Value::Bool(add_bos))
}

/// SentencePiece vocab. Every intermediate prefix has to exist for the merge
/// chain to reach the whole word, which is how these vocabs really look.
fn spm_fixture() -> Builder {
    let tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "\u{2581}",
        "h",
        "e",
        "l",
        "o",
        "w",
        "r",
        "d",
        "\u{2581}h",
        "\u{2581}he",
        "\u{2581}hel",
        "\u{2581}hell",
        "\u{2581}hello",
        "\u{2581}w",
        "\u{2581}wo",
        "\u{2581}wor",
        "\u{2581}worl",
        "\u{2581}world",
        "<0x5A>",
    ];
    let mut types = vec![NORMAL; tokens.len()];
    types[0] = UNKNOWN;
    types[1] = CONTROL;
    types[2] = CONTROL;
    types[21] = BYTE;
    // Longer pieces score higher, as in a real unigram vocab.
    let scores: Vec<f32> = tokens.iter().map(|t| t.chars().count() as f32).collect();

    Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("tokenizer.ggml.model", Value::String("llama".into()))
        .meta("tokenizer.ggml.tokens", strings(&tokens))
        .meta("tokenizer.ggml.token_type", ints(&types))
        .meta("tokenizer.ggml.scores", floats(&scores))
        .meta("tokenizer.ggml.unknown_token_id", Value::U32(0))
        .meta("tokenizer.ggml.bos_token_id", Value::U32(1))
        .meta("tokenizer.ggml.eos_token_id", Value::U32(2))
}

fn load(path: &Path) -> Tokenizer {
    let model = Gguf::open(path).expect("open fixture");
    Tokenizer::from_gguf(&model).expect("load tokenizer")
}

#[test]
fn bpe_merges_to_the_longest_tokens() {
    let path = write("bpe", &bpe_fixture(false));
    let tok = load(&path);

    assert_eq!(tok.kind, Kind::Bpe);
    assert_eq!(tok.vocab_size(), 20);
    assert!(!tok.add_bos, "gpt2 vocabs do not add BOS unless asked");

    let ids = tok.encode("hello world", true);
    assert_eq!(
        ids,
        vec![tok.id("hello").unwrap(), tok.id("\u{0120}world").unwrap()],
        "got {:?}",
        ids.iter().map(|i| tok.token(*i)).collect::<Vec<_>>()
    );
    assert_eq!(tok.decode(&ids), "hello world");

    fs::remove_file(path).ok();
}

#[test]
fn bpe_honours_the_files_bos_policy() {
    let path = write("bpe_bos", &bpe_fixture(true));
    let tok = load(&path);

    assert!(tok.add_bos);
    let ids = tok.encode("hello", true);
    assert_eq!(ids.first(), Some(&0), "BOS should lead");
    // add_special: false must not prepend it.
    assert_eq!(tok.encode("hello", false), vec![tok.id("hello").unwrap()]);
    // BOS is a control token, so it must not show up in decoded text.
    assert_eq!(tok.decode(&ids), "hello");

    fs::remove_file(path).ok();
}

#[test]
fn special_tokens_are_matched_literally() {
    let path = write("bpe_special", &bpe_fixture(false));
    let tok = load(&path);

    let ids = tok.encode("hello<|special|> world", false);
    assert_eq!(
        ids,
        vec![
            tok.id("hello").unwrap(),
            tok.id("<|special|>").unwrap(),
            tok.id("\u{0120}world").unwrap()
        ]
    );

    fs::remove_file(path).ok();
}

#[test]
fn bpe_without_merges_is_rejected() {
    let builder = Builder::new()
        .meta("tokenizer.ggml.model", Value::String("gpt2".into()))
        .meta("tokenizer.ggml.tokens", strings(&["a", "b"]));
    let path = write("bpe_no_merges", &builder);

    let model = Gguf::open(&path).expect("open");
    let err = Tokenizer::from_gguf(&model).expect_err("must reject");
    assert!(err.to_string().contains("merges"), "{err}");

    fs::remove_file(path).ok();
}

#[test]
fn spm_merges_by_score() {
    let path = write("spm", &spm_fixture());
    let tok = load(&path);

    assert_eq!(tok.kind, Kind::Spm);
    assert!(tok.add_bos, "SentencePiece models prepend BOS by default");

    let ids = tok.encode("hello world", false);
    assert_eq!(
        ids,
        vec![
            tok.id("\u{2581}hello").unwrap(),
            tok.id("\u{2581}world").unwrap()
        ],
        "got {:?}",
        ids.iter().map(|i| tok.token(*i)).collect::<Vec<_>>()
    );
    // The dummy prefix space must not survive decoding.
    assert_eq!(tok.decode(&ids), "hello world");

    let with_bos = tok.encode("hello world", true);
    assert_eq!(with_bos.first(), Some(&1));

    fs::remove_file(path).ok();
}

#[test]
fn spm_falls_back_to_byte_tokens() {
    let path = write("spm_bytes", &spm_fixture());
    let tok = load(&path);

    // "Z" is not in the vocab, but <0x5A> is. The leading U+2581 is the dummy
    // prefix: "Z" normalizes to "_Z", and since "_Z" is not a token either, the
    // prefix stays a token of its own.
    let ids = tok.encode("Z", false);
    assert_eq!(
        ids,
        vec![tok.id("\u{2581}").unwrap(), tok.id("<0x5A>").unwrap()]
    );
    assert_eq!(tok.decode(&ids), "Z", "the dummy prefix must not survive");

    // No byte token for "Q", so it must land on <unk> rather than vanish.
    let ids = tok.encode("Q", false);
    assert_eq!(ids, vec![tok.id("\u{2581}").unwrap(), 0]);

    fs::remove_file(path).ok();
}

#[test]
fn test_train_bpe_vocabulary_and_round_trip() {
    let corpus = "The quick brown fox jumps over the lazy dog. The quick fox loves to run and jump in the forest.";
    let config = ferrite::TrainConfig {
        vocab_size: 280,
        special_tokens: vec!["<unk>".into(), "<s>".into(), "</s>".into()],
    };

    let vocab = ferrite::train_bpe(corpus, &config).expect("train BPE");
    assert_eq!(vocab.tokens.len(), 280);
    assert_eq!(vocab.merges.len(), 21);

    let path = std::env::temp_dir().join("ferrite_test_trained_vocab.gguf");
    ferrite::write_vocab_gguf(&vocab, &path).expect("write trained vocab GGUF");

    let gguf = Gguf::open(&path).expect("open trained vocab GGUF");
    let tok = Tokenizer::from_gguf(&gguf).expect("load Tokenizer from trained GGUF");
    assert_eq!(tok.kind, Kind::Bpe);
    assert_eq!(tok.tokens.len(), 280);

    let sample_text = "The quick brown fox jumps over the lazy dog.";
    let encoded = tok.encode(sample_text, false);
    assert!(!encoded.is_empty());

    let decoded = tok.decode(&encoded);
    assert_eq!(decoded, sample_text);

    fs::remove_file(path).ok();
}

#[test]
fn test_train_vocab_file_integration() {
    let corpus_path = std::env::temp_dir().join("ferrite_test_corpus.txt");
    let out_vocab_path = std::env::temp_dir().join("ferrite_test_file_vocab.gguf");

    fs::write(
        &corpus_path,
        "Machine learning models require tokenizers. Tokenization converts text into subword token units.",
    )
    .expect("write test corpus");

    let config = ferrite::TrainConfig {
        vocab_size: 275,
        ..Default::default()
    };

    let vocab = ferrite::train_vocab_file(&corpus_path, &out_vocab_path, &config)
        .expect("train_vocab_file");
    assert_eq!(vocab.tokens.len(), 275);

    let gguf = Gguf::open(&out_vocab_path).expect("open GGUF");
    let tok = Tokenizer::from_gguf(&gguf).expect("load Tokenizer");

    let test_str = "Machine learning token units.";
    let ids = tok.encode(test_str, false);
    assert_eq!(tok.decode(&ids), test_str);

    fs::remove_file(corpus_path).ok();
    fs::remove_file(out_vocab_path).ok();
}

#[test]
fn test_streaming_detokenizer_utf8_multibyte_split() {
    let builder = bpe_fixture(false);
    let path = write("stream_utf8", &builder);
    let gguf = Gguf::open(&path).expect("open");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");

    let mut stream = tokenizer.streaming_detokenizer();

    // 4-byte UTF-8 character: crab emoji '\u{1F980}' -> [0xF0, 0x9F, 0xA6, 0x80]
    let crab_bytes = "🦀".as_bytes();
    assert_eq!(crab_bytes.len(), 4);

    // Feed first byte: incomplete sequence, must yield None
    assert_eq!(stream.feed_bytes(&[crab_bytes[0]]), None);
    assert_eq!(stream.buffered_bytes(), 1);

    // Feed second byte: still incomplete, must yield None
    assert_eq!(stream.feed_bytes(&[crab_bytes[1]]), None);
    assert_eq!(stream.buffered_bytes(), 2);

    // Feed third and fourth bytes plus an ASCII byte '!'
    let emitted = stream.feed_bytes(&[crab_bytes[2], crab_bytes[3], b'!']);
    assert_eq!(emitted, Some("🦀!".to_string()));
    assert_eq!(stream.buffered_bytes(), 0);

    // Test multi-byte character followed by partial next character
    // 3-byte euro symbol '\u{20AC}' -> [0xE2, 0x82, 0xAC]
    let euro_bytes = "€".as_bytes();
    // Feed euro bytes plus first 2 bytes of crab
    let chunk1 = stream.feed_bytes(&[
        euro_bytes[0],
        euro_bytes[1],
        euro_bytes[2],
        crab_bytes[0],
        crab_bytes[1],
    ]);
    assert_eq!(chunk1, Some("€".to_string()));
    assert_eq!(stream.buffered_bytes(), 2);

    // Feed remaining crab bytes
    let chunk2 = stream.feed_bytes(&[crab_bytes[2], crab_bytes[3]]);
    assert_eq!(chunk2, Some("🦀".to_string()));
    assert_eq!(stream.buffered_bytes(), 0);

    // Finish on empty buffer returns None
    assert_eq!(stream.finish(), None);

    // Incomplete sequence flushed at the end produces lossy UTF-8
    assert_eq!(stream.feed_bytes(&[0xE2, 0x82]), None);
    let flushed = stream.finish();
    assert!(flushed.is_some());
    assert_eq!(stream.buffered_bytes(), 0);

    fs::remove_file(path).ok();
}

#[test]
fn test_streaming_detokenizer_with_tokens() {
    let builder = bpe_fixture(true);
    let path = write("stream_tokens", &builder);
    let gguf = Gguf::open(&path).expect("open");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");

    let text = "hello world";
    let ids = tokenizer.encode(text, true);

    let mut stream = tokenizer.streaming_detokenizer();
    let mut streamed_out = String::new();
    for &id in &ids {
        if let Some(chunk) = stream.step(id) {
            streamed_out.push_str(&chunk);
        }
    }
    if let Some(tail) = stream.finish() {
        streamed_out.push_str(&tail);
    }

    assert_eq!(streamed_out, "hello world");

    // Batch detokenize agrees
    let mut stream2 = tokenizer.streaming_detokenizer();
    let batch_out = stream2.detokenize(&ids);
    assert_eq!(batch_out, "hello world");

    fs::remove_file(path).ok();
}
