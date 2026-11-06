use std::fs;
use std::path::PathBuf;

use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::model::Model;
use ferrite::sample::Sampler;
use ferrite::scheduler::{BatchScheduler, SchedulerConfig, SequenceRequest};
use ferrite::synth::Builder;
use ferrite::tokenizer::Tokenizer;

const DIM: usize = 4;
const VOCAB: usize = 4;
const FFN: usize = 8;
const EPS: f32 = 1e-5;

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn embeddings() -> Vec<f32> {
    vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 2.0, 0.0, -1.0, //
        1.0, 2.0, 3.0, 4.0, //
        -2.0, 0.5, 1.0, 0.0,
    ]
}

fn fixture() -> Builder {
    let zero = vec![0.0; DIM * DIM];
    let kv_dim = DIM / 2;
    let heads = 2;
    let ffn_zero = vec![0.0; DIM * FFN];
    let kv_zero = vec![0.0; DIM * kv_dim];

    Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(1))
        .meta("llama.embedding_length", Value::U32(DIM as u32))
        .meta("llama.feed_forward_length", Value::U32(FFN as u32))
        .meta("llama.attention.head_count", Value::U32(heads))
        .meta("llama.attention.head_count_kv", Value::U32(1))
        .meta("llama.context_length", Value::U32(32))
        .meta("llama.attention.layer_norm_rms_epsilon", Value::F32(EPS))
        .meta("llama.rope.freq_base", Value::F32(10000.0))
        .meta("tokenizer.ggml.model", Value::String("llama".into()))
        .meta(
            "tokenizer.ggml.tokens",
            Value::Array((0..VOCAB).map(|i| Value::String(format!("t{i}"))).collect()),
        )
        .meta("tokenizer.ggml.eos_token_id", Value::U32(3))
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
            bytes(&zero),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&kv_zero),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&kv_zero),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&zero),
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
            bytes(&ffn_zero),
        )
        .tensor(
            "blk.0.ffn_up.weight",
            &[DIM as u64, FFN as u64],
            GgmlType::F32,
            bytes(&ffn_zero),
        )
        .tensor(
            "blk.0.ffn_down.weight",
            &[FFN as u64, DIM as u64],
            GgmlType::F32,
            bytes(&ffn_zero),
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
            bytes(&embeddings()),
        )
}

fn temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "ferrite-test-sched-{}-{name}.gguf",
        std::process::id()
    ));
    path
}

#[test]
fn test_concurrent_multi_sequence_scheduling() {
    let path = temp_path("multi_seq");
    fs::write(&path, fixture().build()).unwrap();
    let gguf = Gguf::open(&path).unwrap();
    let model = Model::load(&gguf).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();

    let sched_config = SchedulerConfig {
        max_batch_size: 4,
        max_tokens_per_step: 16,
        prefill_chunk_size: 2,
        tokens_per_block: 2,
    };
    let mut scheduler = BatchScheduler::new(&model.config, 32, sched_config);

    // Sequence 1: 4 prompt tokens, max 3 generated tokens
    let req1 = SequenceRequest {
        id: 1,
        prompt: vec![0, 1, 2, 0],
        max_tokens: 3,
        sampler: Sampler::greedy(),
        grammar: None,
    };
    scheduler.add_sequence(req1);

    // Sequence 2: 2 prompt tokens, max 2 generated tokens
    let req2 = SequenceRequest {
        id: 2,
        prompt: vec![1, 2],
        max_tokens: 2,
        sampler: Sampler::greedy(),
        grammar: None,
    };
    scheduler.add_sequence(req2);

    assert_eq!(scheduler.waiting_count(), 2);
    assert_eq!(scheduler.active_count(), 0);

    let mut step_count = 0;
    while !scheduler.is_empty() && step_count < 20 {
        let _outputs = scheduler.step(&model, &tokenizer).expect("scheduler step");
        step_count += 1;
    }

    assert!(scheduler.is_empty(), "all sequences must complete");
    let finished = scheduler.take_finished();
    assert_eq!(finished.len(), 2);

    let seq1 = finished.iter().find(|(id, _, _)| *id == 1).unwrap();
    let seq2 = finished.iter().find(|(id, _, _)| *id == 2).unwrap();

    assert!(seq1.1.len() <= 3);
    assert!(seq2.1.len() <= 2);

    // Verify prompt blocks were cached in radix tree (3 total: 2 for seq1, 1 for seq2)
    assert_eq!(scheduler.prefix_cache.len(), 3);
    assert_eq!(scheduler.pool.allocated_blocks(), 3);

    // Evict cached blocks and verify complete pool reclamation
    let evicted = scheduler.prefix_cache.evict_lru(&mut scheduler.pool, 3);
    assert_eq!(evicted, 3);
    assert_eq!(scheduler.pool.allocated_blocks(), 0);

    let _ = fs::remove_file(&path);
}

#[test]
fn test_dynamic_arrival_during_continuous_batching() {
    let path = temp_path("dynamic_arrival");
    fs::write(&path, fixture().build()).unwrap();
    let gguf = Gguf::open(&path).unwrap();
    let model = Model::load(&gguf).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();

    let sched_config = SchedulerConfig {
        max_batch_size: 4,
        max_tokens_per_step: 8,
        prefill_chunk_size: 2,
        tokens_per_block: 2,
    };
    let mut scheduler = BatchScheduler::new(&model.config, 32, sched_config);

    // Submit Sequence A
    scheduler.add_sequence(SequenceRequest {
        id: 100,
        prompt: vec![0, 1],
        max_tokens: 4,
        sampler: Sampler::greedy(),
        grammar: None,
    });

    // Step 1: Pre-fills Sequence A
    let _ = scheduler.step(&model, &tokenizer).unwrap();
    assert_eq!(scheduler.active_count(), 1);

    // Dynamically submit Sequence B during Sequence A's execution
    scheduler.add_sequence(SequenceRequest {
        id: 200,
        prompt: vec![2, 0, 1, 2],
        max_tokens: 2,
        sampler: Sampler::greedy(),
        grammar: None,
    });

    // Step until all complete
    let mut iterations = 0;
    while !scheduler.is_empty() && iterations < 30 {
        let _ = scheduler.step(&model, &tokenizer).unwrap();
        iterations += 1;
    }

    assert!(scheduler.is_empty());
    let finished = scheduler.take_finished();
    assert_eq!(finished.len(), 2);

    let a = finished.iter().find(|(id, _, _)| *id == 100).unwrap();
    let b = finished.iter().find(|(id, _, _)| *id == 200).unwrap();
    assert!(!a.1.is_empty());
    assert!(!b.1.is_empty());

    // Verify cached prefix blocks and complete reclamation upon LRU eviction
    assert_eq!(scheduler.prefix_cache.len(), 3);
    assert_eq!(scheduler.pool.allocated_blocks(), 3);
    let evicted = scheduler.prefix_cache.evict_lru(&mut scheduler.pool, 3);
    assert_eq!(evicted, 3);
    assert_eq!(scheduler.pool.allocated_blocks(), 0);

    let _ = fs::remove_file(&path);
}
