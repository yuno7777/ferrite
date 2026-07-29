//! Round-trip and rejection tests for the GGUF parser.
//!
//! Everything here builds its own file, so the suite needs no model download
//! and no network.

use std::fs;
use std::path::PathBuf;

use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::synth::Builder;

/// 4096 bytes of a recognizable pattern — enough to catch an off-by-one in the
/// data offset, which is the bug this whole test exists for.
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

fn fixture() -> Builder {
    Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("general.name", Value::String("test".into()))
        .meta("llama.block_count", Value::U32(2))
        .meta("llama.embedding_length", Value::U32(256))
        .meta("llama.rope.freq_base", Value::F32(10000.0))
        .meta("answer", Value::I64(-42))
        .meta(
            "tokenizer.ggml.tokens",
            Value::Array(vec![
                Value::String("<s>".into()),
                Value::String("hello".into()),
            ]),
        )
        .tensor(
            "token_embd.weight",
            &[256, 4],
            GgmlType::F32,
            pattern(256 * 4 * 4),
        )
        .tensor(
            "blk.0.attn_q.weight",
            &[256, 256],
            GgmlType::Q4_K,
            pattern(256 * 256 / 256 * 144),
        )
}

fn write(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ferrite_{name}.gguf"));
    fs::write(&path, bytes).expect("write fixture");
    path
}

#[test]
fn round_trips_metadata_and_tensors() {
    let path = write("round_trip", &fixture().build());
    let model = Gguf::open(&path).expect("open");

    assert_eq!(model.version, 3);
    assert_eq!(model.arch(), Some("llama"));
    assert_eq!(model.meta_str("general.name"), Some("test"));
    assert_eq!(model.arch_u64("block_count"), Some(2));
    assert_eq!(model.arch_f32("rope.freq_base"), Some(10000.0));
    // Signed values must not silently widen into a u64.
    assert_eq!(model.meta_u64("answer"), None);
    assert_eq!(model.get("answer"), Some(&Value::I64(-42)));
    assert_eq!(
        model.get("tokenizer.ggml.tokens").unwrap().as_strings(),
        Some(vec!["<s>", "hello"])
    );

    assert_eq!(model.tensors.len(), 2);
    assert_eq!(model.data_offset % model.alignment, 0);

    let embd = model.tensor("token_embd.weight").expect("tensor present");
    assert_eq!(embd.dims, vec![256, 4]);
    assert_eq!(embd.elements(), 1024);
    assert_eq!(embd.byte_len, Some(4096));
    assert_eq!(embd.shape(), "256x4");

    // The bytes, exactly, at the right offset.
    assert_eq!(model.tensor_bytes(embd).unwrap(), pattern(4096).as_slice());

    let q = model.tensor("blk.0.attn_q.weight").expect("tensor present");
    assert_eq!(q.ty, GgmlType::Q4_K);
    assert_eq!(q.byte_len, Some(256 * 256 / 256 * 144));
    assert_eq!(
        model.tensor_bytes(q).unwrap(),
        pattern(256 * 256 / 256 * 144).as_slice()
    );

    assert!(model.tensor("nope").is_none());

    let hist = model.quant_histogram();
    assert_eq!(hist.len(), 2);
    // Sorted by bytes descending: Q4_K's 36 KB beats F32's 4 KB.
    assert_eq!(hist[0].0, GgmlType::Q4_K);

    fs::remove_file(path).ok();
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = fixture().build();
    bytes[0..4].copy_from_slice(b"XXXX");
    let path = write("bad_magic", &bytes);
    let err = Gguf::open(&path).expect_err("must reject");
    assert!(err.to_string().contains("bad magic"), "{err}");
    fs::remove_file(path).ok();
}

#[test]
fn rejects_unsupported_version() {
    let mut bytes = fixture().build();
    bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
    let path = write("bad_version", &bytes);
    let err = Gguf::open(&path).expect_err("must reject");
    assert!(err.to_string().contains("version 1"), "{err}");
    fs::remove_file(path).ok();
}

#[test]
fn rejects_truncated_file() {
    // The common real-world corruption: an interrupted download. The header is
    // intact and the tensor table promises data that is not there.
    let full = fixture().build();
    let path = write("truncated", &full[..full.len() - 1024]);
    let err = Gguf::open(&path).expect_err("must reject");
    assert!(
        err.to_string().contains("past end of file"),
        "expected a truncation error, got: {err}"
    );
    fs::remove_file(path).ok();
}

#[test]
fn rejects_empty_file() {
    let path = write("empty", b"");
    let err = Gguf::open(&path).expect_err("must reject");
    assert!(err.to_string().contains("empty"), "{err}");
    fs::remove_file(path).ok();
}

#[test]
fn rejects_absurd_metadata_count() {
    // A corrupt count must not make the parser allocate until the OOM killer
    // arrives; it should hit end-of-file instead.
    let mut bytes = fixture().build();
    bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    let path = write("absurd_count", &bytes);
    let err = Gguf::open(&path).expect_err("must reject");
    // Which guard catches it first depends on what the garbage decodes to —
    // an implausible length or a plain end-of-file. Either is a clean error
    // rather than an allocation the size of the count.
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{err}");
    fs::remove_file(path).ok();
}
