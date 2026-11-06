use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::model::Model;
use ferrite::server::{Server, ServerConfig};
use ferrite::synth::Builder;
use ferrite::Tokenizer;

const DIM: usize = 4;
const FFN: usize = 8;
const VOCAB: usize = 4;
const EPS: f32 = 1e-5;

fn bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn identity(n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

fn embeddings() -> Vec<f32> {
    let mut out = vec![0.0; VOCAB * DIM];
    for tok in 0..VOCAB {
        for col in 0..DIM {
            out[tok * DIM + col] = if tok % DIM == col { 1.0 } else { 0.0 };
        }
    }
    out
}

fn fixture() -> Builder {
    Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(1))
        .meta("llama.embedding_length", Value::U32(DIM as u32))
        .meta("llama.feed_forward_length", Value::U32(FFN as u32))
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
            "blk.0.attn_k.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&identity(DIM)),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&identity(DIM)),
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

fn write_fixture(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ferrite_server_test_{name}.gguf"));
    fixture().write(&path).expect("write fixture");
    path
}

fn setup_server(
    name: &str,
) -> (
    PathBuf,
    u16,
    Arc<AtomicBool>,
    std::thread::JoinHandle<std::io::Result<()>>,
) {
    let path = write_fixture(name);
    let gguf = Box::leak(Box::new(Gguf::open(&path).expect("open")));
    let model = Box::leak(Box::new(Model::load(gguf).expect("load")));
    let tokenizer = Box::leak(Box::new(Tokenizer::from_gguf(gguf).expect("tokenizer")));

    let config = ServerConfig {
        host: "127.0.0.1".into(),
        port: 0, // OS assigns available ephemeral port
        threads: 2,
        context: 16,
        kv_cache: ferrite::kv::KvCacheType::F32,
    };

    let server = Server::new(model, tokenizer, config);
    let listener = server.listen().expect("bind listener");
    let port = listener.local_addr().expect("local addr").port();
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_signal = Arc::clone(&shutdown);

    let handle = std::thread::spawn(move || server.serve_with_shutdown(&listener, shutdown_signal));

    // Short sleep to allow listener loop to run
    std::thread::sleep(std::time::Duration::from_millis(50));
    (path, port, shutdown, handle)
}

#[test]
fn test_health_and_models_endpoints() {
    let (path, port, shutdown, handle) = setup_server("health");

    // 1. GET /health
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .expect("write request");

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).expect("read status");
    assert!(status_line.contains("200 OK"), "{status_line}");

    let mut body = String::new();
    let mut in_body = false;
    let mut line = String::new();
    while reader.read_line(&mut line).unwrap_or(0) > 0 {
        if in_body {
            body.push_str(&line);
        } else if line.trim().is_empty() {
            in_body = true;
        }
        line.clear();
    }
    assert!(body.contains("\"status\":\"ok\""), "body was: {body}");
    assert!(body.contains("\"model\":\"llama\""), "body was: {body}");

    // 2. GET /v1/models
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .expect("write request");

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).expect("read status");
    assert!(status_line.contains("200 OK"), "{status_line}");

    shutdown.store(true, Ordering::Relaxed);
    let _ = handle.join();
    std::fs::remove_file(path).ok();
}

#[test]
fn test_chat_completions_non_streaming() {
    let (path, port, shutdown, handle) = setup_server("non_streaming");

    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let req_body =
        r#"{"messages":[{"role":"user","content":"t0 t1"}],"max_tokens":4,"stream":false}"#;
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
        req_body.len(),
        req_body
    );
    stream.write_all(req.as_bytes()).expect("write request");

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).expect("read status");
    assert!(status_line.contains("200 OK"), "{status_line}");

    let mut response_text = String::new();
    reader
        .read_to_string(&mut response_text)
        .expect("read response");
    assert!(response_text.contains("chat.completion"));
    assert!(response_text.contains("\"finish_reason\":\"stop\""));

    shutdown.store(true, Ordering::Relaxed);
    let _ = handle.join();
    std::fs::remove_file(path).ok();
}

#[test]
fn test_chat_completions_streaming_sse() {
    let (path, port, shutdown, handle) = setup_server("streaming_sse");

    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let req_body = r#"{"messages":[{"role":"user","content":"t0"}],"max_tokens":3,"stream":true}"#;
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
        req_body.len(),
        req_body
    );
    stream.write_all(req.as_bytes()).expect("write request");

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).expect("read status");
    assert!(status_line.contains("200 OK"), "{status_line}");

    let mut sse_lines = Vec::new();
    let mut line = String::new();
    while reader.read_line(&mut line).unwrap_or(0) > 0 {
        if line.starts_with("data: ") {
            sse_lines.push(line.trim().to_string());
        }
        line.clear();
    }

    assert!(!sse_lines.is_empty(), "expected SSE data chunks");
    assert_eq!(sse_lines.last().map(String::as_str), Some("data: [DONE]"));
    assert!(sse_lines[0].contains("chat.completion.chunk"));

    shutdown.store(true, Ordering::Relaxed);
    let _ = handle.join();
    std::fs::remove_file(path).ok();
}

#[test]
fn test_not_found_endpoint() {
    let (path, port, shutdown, handle) = setup_server("not_found");

    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .write_all(b"GET /nonexistent HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .expect("write request");

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).expect("read status");
    assert!(status_line.contains("404 Not Found"), "{status_line}");

    shutdown.store(true, Ordering::Relaxed);
    let _ = handle.join();
    std::fs::remove_file(path).ok();
}
