use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ferrite::embed::{cosine_similarity, EmbedEngine, Pooling};
use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::model::{Model, State};
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

fn write_file(name: &str, builder: &Builder) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ferrite_embed_test_{name}.gguf"));
    builder.write(&path).expect("write fixture");
    path
}

#[test]
fn test_embed_engine_pooling_and_normalization() {
    let path = write_file("engine_test", &fixture());
    let gguf = Gguf::open(&path).expect("open gguf");
    let model = Model::load(&gguf).expect("load model");
    let mut state = State::new(&model.config);

    let engine_mean = EmbedEngine::new(&model, Pooling::Mean, true);
    let engine_cls = EmbedEngine::new(&model, Pooling::Cls, true);
    let engine_last = EmbedEngine::new(&model, Pooling::Last, true);

    let prompt = vec![0u32, 1, 2];

    let emb_mean = engine_mean.embed(&mut state, &prompt).expect("embed mean");
    let emb_cls = engine_cls.embed(&mut state, &prompt).expect("embed cls");
    let emb_last = engine_last.embed(&mut state, &prompt).expect("embed last");

    assert_eq!(emb_mean.len(), DIM);
    assert_eq!(emb_cls.len(), DIM);
    assert_eq!(emb_last.len(), DIM);

    // All should have unit L2 norm
    let norm_mean = emb_mean.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_cls = emb_cls.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_last = emb_last.iter().map(|v| v * v).sum::<f32>().sqrt();

    assert!((norm_mean - 1.0).abs() < 1e-5);
    assert!((norm_cls - 1.0).abs() < 1e-5);
    assert!((norm_last - 1.0).abs() < 1e-5);

    // Cosine similarity with self is 1.0
    let sim_self = cosine_similarity(&emb_mean, &emb_mean);
    assert!((sim_self - 1.0).abs() < 1e-5);

    // Embed another divergent sequence
    let prompt2 = vec![3u32, 2, 1];
    let emb2 = engine_mean
        .embed(&mut state, &prompt2)
        .expect("embed prompt2");
    let sim_diff = cosine_similarity(&emb_mean, &emb2);
    assert!(sim_diff < 0.99);

    fs::remove_file(path).ok();
}

#[test]
fn test_server_v1_embeddings_endpoint() {
    let path = write_file("server_test", &fixture());
    let gguf = Box::leak(Box::new(Gguf::open(&path).expect("open gguf")));
    let model = Box::leak(Box::new(Model::load(gguf).expect("load model")));
    let tokenizer = Box::leak(Box::new(
        Tokenizer::from_gguf(gguf).expect("load tokenizer"),
    ));

    let config = ServerConfig {
        host: "127.0.0.1".into(),
        port: 0,
        threads: 1,
        context: 64,
        kv_cache: ferrite::kv::KvCacheType::F32,
    };

    let server = Server::new(model, tokenizer, config);
    let listener = server.listen().expect("bind listener");
    let port = listener.local_addr().expect("local addr").port();
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_signal = Arc::clone(&shutdown);

    let handle = std::thread::spawn(move || server.serve_with_shutdown(&listener, shutdown_signal));
    std::thread::sleep(std::time::Duration::from_millis(50));

    // 1. Single string embedding request
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let body = r#"{"input": "t0 t1"}"#;
    let req = format!(
        "POST /v1/embeddings HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(req.as_bytes()).expect("write request");

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).expect("read status");
    assert!(
        status_line.contains("200 OK"),
        "expected 200 OK, got: {status_line}"
    );

    let mut content_length = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).expect("read header");
        if line == "\r\n" {
            break;
        }
        if line.to_lowercase().starts_with("content-length:") {
            content_length = line["content-length:".len()..]
                .trim()
                .parse::<usize>()
                .unwrap();
        }
    }

    let mut response_body = vec![0u8; content_length];
    reader.read_exact(&mut response_body).expect("read body");
    let body_str = String::from_utf8(response_body).expect("utf8");
    assert!(body_str.contains(r#""object":"list""#));
    assert!(body_str.contains(r#""object":"embedding""#));
    assert!(body_str.contains(r#""index":0"#));

    // 2. Batch string array embedding request
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect batch");
    let batch_body = r#"{"input": ["t0", "t1 t2"]}"#;
    let req = format!(
        "POST /v1/embeddings HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: {}\r\n\r\n{}",
        batch_body.len(),
        batch_body
    );
    stream
        .write_all(req.as_bytes())
        .expect("write batch request");

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).expect("read status");
    assert!(status_line.contains("200 OK"));

    let mut content_length = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).expect("read header");
        if line == "\r\n" {
            break;
        }
        if line.to_lowercase().starts_with("content-length:") {
            content_length = line["content-length:".len()..]
                .trim()
                .parse::<usize>()
                .unwrap();
        }
    }

    let mut response_body = vec![0u8; content_length];
    reader.read_exact(&mut response_body).expect("read body");
    let body_str = String::from_utf8(response_body).expect("utf8");
    assert!(body_str.contains(r#""index":0"#));
    assert!(body_str.contains(r#""index":1"#));

    shutdown.store(true, Ordering::Relaxed);
    TcpStream::connect(("127.0.0.1", port)).ok();
    handle.join().ok();

    fs::remove_file(path).ok();
}
