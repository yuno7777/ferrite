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

#[test]
fn test_radix_cache_disk_persistence_roundtrip() {
    let mut pool = PagedPool::new(2, 8, 4, 16);
    let mut cache = RadixCache::with_block_size(4);

    let b1 = pool.allocate_block();
    let b2 = pool.allocate_block();

    // Populate recognizable patterns in key and value
    pool.blocks[b1].key.fill(1.25);
    pool.blocks[b1].value.fill(2.50);
    pool.blocks[b2].key.fill(3.75);
    pool.blocks[b2].value.fill(5.00);

    let prompt = vec![100, 101, 102, 103, 200, 201, 202, 203];
    cache.insert(&prompt, &[b1, b2], &mut pool);

    let mut path = std::env::temp_dir();
    path.push(format!(
        "ferrite-test-prefix-persist-{}.bin",
        std::process::id()
    ));

    cache.save_to_file(&pool, &path).expect("save to file");

    let mut pool_loaded = PagedPool::new(2, 8, 4, 16);
    let mut loaded_cache =
        RadixCache::load_from_file(&path, &mut pool_loaded).expect("load from file");

    assert_eq!(loaded_cache.len(), 2);
    let (matched, tokens) = loaded_cache.match_prefix(&prompt);
    assert_eq!(tokens, 8);
    assert_eq!(matched.len(), 2);

    let lb1 = matched[0];
    let lb2 = matched[1];
    assert_eq!(
        pool_loaded.blocks[lb1].key,
        vec![1.25; pool.blocks[b1].key.len()]
    );
    assert_eq!(
        pool_loaded.blocks[lb1].value,
        vec![2.50; pool.blocks[b1].value.len()]
    );
    assert_eq!(
        pool_loaded.blocks[lb2].key,
        vec![3.75; pool.blocks[b2].key.len()]
    );
    assert_eq!(
        pool_loaded.blocks[lb2].value,
        vec![5.00; pool.blocks[b2].value.len()]
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn test_radix_cache_page_swapping_out_and_in() {
    use ferrite::prefix::PageSwapManager;

    let layers = 1;
    let kv_dim = 4;
    let tokens_per_block = 2;
    let mut pool = PagedPool::new(layers, kv_dim, tokens_per_block, 4);
    let mut cache = RadixCache::with_block_size(tokens_per_block);

    let b1 = pool.allocate_block();
    let b2 = pool.allocate_block();
    pool.blocks[b1].key.fill(11.0);
    pool.blocks[b1].value.fill(22.0);
    pool.blocks[b2].key.fill(33.0);
    pool.blocks[b2].value.fill(44.0);

    let prompt = vec![1, 2, 3, 4];
    cache.insert(&prompt, &[b1, b2], &mut pool);

    // Release initial sequence ownership so cache is sole owner of blocks
    pool.free_block(b1);
    pool.free_block(b2);

    let mut swap_path = std::env::temp_dir();
    swap_path.push(format!("ferrite-test-page-swap-{}.bin", std::process::id()));

    let block_bytes = layers * tokens_per_block * kv_dim * 8;
    let mut swap_mgr = PageSwapManager::new(&swap_path, block_bytes).expect("swap manager");

    let free_before = pool.free_blocks();

    // Swap out 1 LRU leaf block (b2)
    let swapped_count = cache
        .swap_out_lru_nodes(&mut pool, &mut swap_mgr, 1)
        .expect("swap out");
    assert_eq!(swapped_count, 1);
    assert_eq!(swap_mgr.active_swapped_pages(), 1);
    assert_eq!(pool.free_blocks(), free_before + 1);

    // Match with swap-in
    let (matched, tokens) = cache
        .match_prefix_with_swap_in(&prompt, &mut pool, &mut swap_mgr)
        .expect("match with swap in");
    assert_eq!(tokens, 4);
    assert_eq!(matched.len(), 2);
    assert_eq!(swap_mgr.active_swapped_pages(), 0);

    let reloaded_b2 = matched[1];
    assert_eq!(
        pool.blocks[reloaded_b2].key,
        vec![33.0; pool.blocks[b2].key.len()]
    );
    assert_eq!(
        pool.blocks[reloaded_b2].value,
        vec![44.0; pool.blocks[b2].value.len()]
    );

    let _ = fs::remove_file(&swap_path);
}

#[test]
fn test_page_swap_manager_slot_recycling() {
    use ferrite::prefix::PageSwapManager;

    let mut pool = PagedPool::new(1, 4, 2, 8);
    let b = pool.allocate_block();
    pool.blocks[b].key.fill(7.0);

    let mut swap_path = std::env::temp_dir();
    swap_path.push(format!(
        "ferrite-test-swap-recycle-{}.bin",
        std::process::id()
    ));

    let block_bytes = 2 * 4 * 8;
    let mut swap_mgr = PageSwapManager::new(&swap_path, block_bytes).expect("swap mgr");

    let slot = swap_mgr.swap_out_block(&mut pool, b).expect("swap out");
    assert_eq!(slot, 0);
    assert_eq!(swap_mgr.active_swapped_pages(), 1);

    let b_new = swap_mgr.swap_in_block(&mut pool, slot).expect("swap in");
    assert_eq!(swap_mgr.active_swapped_pages(), 0);
    assert_eq!(pool.blocks[b_new].key[0], 7.0);

    // Swap out another block; should reuse slot 0
    let b2 = pool.allocate_block();
    let slot2 = swap_mgr.swap_out_block(&mut pool, b2).expect("swap out 2");
    assert_eq!(slot2, 0);

    let _ = fs::remove_file(&swap_path);
}
