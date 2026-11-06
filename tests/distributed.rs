use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::thread;

use ferrite::distributed::{run_worker, DistributedDriver};
use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::model::{Model, State};
use ferrite::synth::Builder;

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

fn fixture_2layers() -> Builder {
    let mut b = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(2))
        .meta("llama.embedding_length", Value::U32(DIM as u32))
        .meta("llama.feed_forward_length", Value::U32(FFN as u32))
        .meta("llama.attention.head_count", Value::U32(1))
        .meta("llama.attention.head_count_kv", Value::U32(1))
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
        );

    for layer in 0..2 {
        b = b
            .tensor(
                &format!("blk.{layer}.attn_norm.weight"),
                &[DIM as u64],
                GgmlType::F32,
                bytes(&[1.0; DIM]),
            )
            .tensor(
                &format!("blk.{layer}.attn_q.weight"),
                &[DIM as u64, DIM as u64],
                GgmlType::F32,
                bytes(&identity(DIM)),
            )
            .tensor(
                &format!("blk.{layer}.attn_k.weight"),
                &[DIM as u64, DIM as u64],
                GgmlType::F32,
                bytes(&identity(DIM)),
            )
            .tensor(
                &format!("blk.{layer}.attn_v.weight"),
                &[DIM as u64, DIM as u64],
                GgmlType::F32,
                bytes(&identity(DIM)),
            )
            .tensor(
                &format!("blk.{layer}.attn_output.weight"),
                &[DIM as u64, DIM as u64],
                GgmlType::F32,
                bytes(&identity(DIM)),
            )
            .tensor(
                &format!("blk.{layer}.ffn_norm.weight"),
                &[DIM as u64],
                GgmlType::F32,
                bytes(&[1.0; DIM]),
            )
            .tensor(
                &format!("blk.{layer}.ffn_gate.weight"),
                &[DIM as u64, FFN as u64],
                GgmlType::F32,
                bytes(&[0.1 * (layer as f32 + 1.0); DIM * FFN]),
            )
            .tensor(
                &format!("blk.{layer}.ffn_up.weight"),
                &[DIM as u64, FFN as u64],
                GgmlType::F32,
                bytes(&[0.1 * (layer as f32 + 1.0); DIM * FFN]),
            )
            .tensor(
                &format!("blk.{layer}.ffn_down.weight"),
                &[FFN as u64, DIM as u64],
                GgmlType::F32,
                bytes(&[0.1 * (layer as f32 + 1.0); FFN * DIM]),
            );
    }

    b.tensor(
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

fn write(name: &str, builder: &Builder) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "ferrite-test-dist-{}-{name}.gguf",
        std::process::id()
    ));
    fs::write(&path, builder.build()).expect("write fixture");
    path
}

#[test]
fn test_distributed_pipeline_parallel_forward_parity() {
    let path = write("dist_forward", &fixture_2layers());
    let gguf = Gguf::open(&path).expect("open fixture");
    let model = Model::load(&gguf).expect("load model");

    // 1. Single-node local reference forward pass
    let mut state_ref = State::new(&model.config);
    let prompt = [0u32, 1, 2];

    for (pos, &tok) in prompt.iter().enumerate() {
        model
            .forward(&mut state_ref, tok, pos)
            .expect("local forward");
    }

    // 2. Start distributed worker for Layer 1 on ephemeral port
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr").to_string();

    let worker_path = path.clone();
    let worker_thread = thread::spawn(move || {
        let worker_gguf = Gguf::open(&worker_path).expect("worker open");
        let worker_model = Model::load(&worker_gguf).expect("worker model load");
        run_worker(&worker_model, 1..2, listener).expect("worker run");
    });

    // 3. Connect driver executing Layer 0 locally and Layer 1 remotely
    let mut state_dist = State::new(&model.config);
    let mut driver = DistributedDriver::connect(&model, 0..1, 1..2, &addr).expect("connect driver");

    for (pos, &tok) in prompt.iter().enumerate() {
        driver
            .forward(&mut state_dist, tok, pos)
            .expect("distributed forward");
    }

    // 4. Verify logits match bit-identically
    let bits = |s: &[f32]| s.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_dist.logits),
        bits(&state_ref.logits),
        "Distributed pipeline-parallel forward logits must match single-node reference"
    );

    // Shutdown worker
    driver.shutdown().expect("shutdown");
    worker_thread.join().expect("join worker");

    fs::remove_file(path).ok();
}
