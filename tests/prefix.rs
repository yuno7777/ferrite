use std::fs;
use std::path::PathBuf;

use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::model::{Model, State};
use ferrite::paged::{PagedPool, PagedSequence};
use ferrite::prefix::RadixCache;
use ferrite::synth::Builder;

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
        "ferrite-test-prefix-{}-{name}.gguf",
        std::process::id()
    ));
    path
}

#[test]
fn test_radix_cache_prompt_reuse_equivalence() {
    let path = temp_path("equiv");
    fs::write(&path, fixture().build()).unwrap();
    let gguf = Gguf::open(&path).unwrap();
    let model = Model::load(&gguf).unwrap();

    let tokens_per_block = 2;
    let mut pool = PagedPool::new(
        model.config.layers,
        model.config.kv_dim(),
        tokens_per_block,
        16,
    );
    let mut cache = RadixCache::with_block_size(tokens_per_block);

    // Prompt A: 4 tokens = 2 complete blocks [0, 1] and [2, 3]
    let prompt_a = vec![0, 1, 2, 3];
    let mut seq_a = PagedSequence::new();
    let mut state_a = State::new(&model.config);
    model
        .prefill_paged(&mut pool, &mut seq_a, &mut state_a, &prompt_a, 0)
        .unwrap();

    // Insert into radix cache
    cache.insert(&prompt_a, &seq_a.block_table, &mut pool);
    assert_eq!(cache.len(), 2);

    // Sequence B has same 4-token prompt + suffix token 1
    let prompt_b = vec![0, 1, 2, 3, 1];
    let (matched_blocks, matched_tokens) = cache.match_prefix(&prompt_b);
    assert_eq!(matched_tokens, 4);
    assert_eq!(matched_blocks.len(), 2);

    let mut seq_b = PagedSequence::new();
    cache.apply_prefix_to_sequence(&matched_blocks, matched_tokens, &mut seq_b, &mut pool);

    // Prefill remaining token [1] starting at position matched_tokens
    let remaining = &prompt_b[matched_tokens..];
    let mut state_b = State::new(&model.config);
    model
        .prefill_paged(
            &mut pool,
            &mut seq_b,
            &mut state_b,
            remaining,
            matched_tokens,
        )
        .unwrap();

    // Compute ground-truth baseline without caching
    let mut seq_ground_truth = PagedSequence::new();
    let mut state_ground_truth = State::new(&model.config);
    model
        .prefill_paged(
            &mut pool,
            &mut seq_ground_truth,
            &mut state_ground_truth,
            &prompt_b,
            0,
        )
        .unwrap();

    assert_eq!(state_b.logits.len(), state_ground_truth.logits.len());
    for (v_cached, v_gt) in state_b.logits.iter().zip(state_ground_truth.logits.iter()) {
        assert_eq!(v_cached.to_bits(), v_gt.to_bits());
    }

    seq_a.free(&mut pool);
    seq_b.free(&mut pool);
    seq_ground_truth.free(&mut pool);

    let _ = fs::remove_file(&path);
}

#[test]
fn test_radix_cache_multi_turn_branching() {
    let mut pool = PagedPool::new(1, 4, 2, 16);
    let mut cache = RadixCache::with_block_size(2);

    // Turn 1 (Common System Prompt): [10, 20]
    let sys = vec![10, 20];
    let b_sys = pool.allocate_block();
    cache.insert(&sys, &[b_sys], &mut pool);
    assert_eq!(cache.len(), 1);

    // Branch 1: [10, 20, 30, 40]
    let b1 = vec![10, 20, 30, 40];
    let b_user1 = pool.allocate_block();
    cache.insert(&b1, &[b_sys, b_user1], &mut pool);
    assert_eq!(cache.len(), 2);

    // Branch 2: [10, 20, 50, 60]
    let b2 = vec![10, 20, 50, 60];
    let b_user2 = pool.allocate_block();
    cache.insert(&b2, &[b_sys, b_user2], &mut pool);
    assert_eq!(cache.len(), 3);

    // Matching Branch 1
    let (matched, tokens) = cache.match_prefix(&b1);
    assert_eq!(tokens, 4);
    assert_eq!(matched, vec![b_sys, b_user1]);

    // Matching Branch 2
    let (matched, tokens) = cache.match_prefix(&b2);
    assert_eq!(tokens, 4);
    assert_eq!(matched, vec![b_sys, b_user2]);

    // Evict least-recently-used (Branch 1 was accessed before Branch 2)
    let evicted = cache.evict_lru(&mut pool, 1);
    assert_eq!(evicted, 1);
    assert_eq!(cache.len(), 2);

    // Branch 1 now only matches root system prompt
    let (matched, tokens) = cache.match_prefix(&b1);
    assert_eq!(tokens, 2);
    assert_eq!(matched, vec![b_sys]);
}
