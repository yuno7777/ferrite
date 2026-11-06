use std::fs;
use std::path::PathBuf;

use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::model::{Model, State};
use ferrite::paged::{PagedPool, PagedSequence};
use ferrite::synth::Builder;

const DIM: usize = 4;
const VOCAB: usize = 4;
const FFN: usize = 8;
const EPS: f32 = 1e-5;

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn identity(n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

fn embeddings() -> Vec<f32> {
    vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 2.0, 0.0, -1.0, //
        1.0, 2.0, 3.0, 4.0, //
        -2.0, 0.5, 1.0, 0.0,
    ]
}

fn fixture(kv_heads: usize, value_path: bool) -> Builder {
    let zero = vec![0.0; DIM * DIM];
    let kv_dim = kv_heads * (DIM / 2);
    let heads = 2;

    let mut builder = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(1))
        .meta("llama.embedding_length", Value::U32(DIM as u32))
        .meta("llama.feed_forward_length", Value::U32(FFN as u32))
        .meta("llama.attention.head_count", Value::U32(heads))
        .meta("llama.attention.head_count_kv", Value::U32(kv_heads as u32))
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
            bytes(&zero),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&vec![0.0; DIM * kv_dim]),
        );

    let value_weights = if value_path {
        identity(DIM)
    } else {
        zero.clone()
    };
    builder = builder
        .tensor(
            "blk.0.attn_v.weight",
            &[DIM as u64, kv_dim as u64],
            GgmlType::F32,
            bytes(&value_weights[..DIM * kv_dim]),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[DIM as u64, DIM as u64],
            GgmlType::F32,
            bytes(&if value_path {
                identity(DIM)
            } else {
                zero.clone()
            }),
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
        );
    builder
}

fn write(name: &str, builder: &Builder) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ferrite_paged_test_{name}.gguf"));
    builder.write(&path).expect("write fixture");
    path
}

#[test]
fn paged_pool_allocation_and_free_tracking() {
    let mut pool = PagedPool::new(2, 4, 16, 4);
    assert_eq!(pool.total_blocks(), 4);
    assert_eq!(pool.free_blocks(), 4);
    assert_eq!(pool.allocated_blocks(), 0);

    let b1 = pool.allocate_block();
    let b2 = pool.allocate_block();
    assert_eq!(pool.allocated_blocks(), 2);
    assert_eq!(pool.free_blocks(), 2);

    pool.free_block(b1);
    assert_eq!(pool.allocated_blocks(), 1);
    assert_eq!(pool.free_blocks(), 3);

    pool.free_block(b2);
    assert_eq!(pool.allocated_blocks(), 0);
    assert_eq!(pool.free_blocks(), 4);
}

#[test]
fn paged_sequence_fork_and_cow_isolation() {
    let mut pool = PagedPool::new(2, 4, 16, 8);
    let mut seq1 = PagedSequence::new();

    // Store 2 tokens in seq1
    let k = [1.0, 2.0, 3.0, 4.0];
    let v = [0.1, 0.2, 0.3, 0.4];
    pool.store(&mut seq1, 0, 0, &k, &v).expect("store 0");
    pool.store(&mut seq1, 0, 1, &k, &v).expect("store 1");

    assert_eq!(seq1.block_table.len(), 1);
    let shared_block_id = seq1.block_table[0];
    assert_eq!(pool.ref_counts[shared_block_id], 1);

    // Fork seq2 from seq1
    let mut seq2 = seq1.fork(&mut pool);
    assert_eq!(seq2.block_table, seq1.block_table);
    assert_eq!(pool.ref_counts[shared_block_id], 2);

    // Mutating seq2 triggers Copy-on-Write
    let k_new = [9.0, 9.0, 9.0, 9.0];
    pool.store(&mut seq2, 0, 1, &k_new, &v).expect("store cow");

    // seq1 and seq2 now point to different physical blocks
    assert_ne!(seq1.block_table[0], seq2.block_table[0]);
    assert_eq!(pool.ref_counts[seq1.block_table[0]], 1);
    assert_eq!(pool.ref_counts[seq2.block_table[0]], 1);

    // Verify seq1 data is untouched
    let b1 = seq1.block_table[0];
    assert_eq!(&pool.blocks[b1].key[4..8], &k);

    // Verify seq2 has the new data
    let b2 = seq2.block_table[0];
    assert_eq!(&pool.blocks[b2].key[4..8], &k_new);

    // Free both sequences
    seq1.free(&mut pool);
    seq2.free(&mut pool);
    assert_eq!(pool.allocated_blocks(), 0);
}

#[test]
fn paged_prefill_and_forward_matches_contiguous_model() {
    let path = write("parity", &fixture(1, true));
    let gguf = Gguf::open(&path).expect("open");
    let model = Model::load(&gguf).expect("load");

    let prompt = vec![1u32, 2, 3, 0, 2];

    // 1. Contiguous baseline
    let mut state_ref = State::new(&model.config);
    model.prefill(&mut state_ref, &prompt, 0).expect("prefill");

    // 2. Paged prefill
    let mut pool = PagedPool::new(model.config.layers, model.config.kv_dim(), 16, 8);
    let mut seq = PagedSequence::new();
    let mut state_paged = State::new(&model.config);

    model
        .prefill_paged(&mut pool, &mut seq, &mut state_paged, &prompt, 0)
        .expect("prefill_paged");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_paged.logits),
        bits(&state_ref.logits),
        "paged prefill logits must match contiguous prefill logits exactly"
    );

    // 3. Test sequential decode step on both
    let next_token = 3u32;
    model
        .forward(&mut state_ref, next_token, prompt.len())
        .expect("forward ref");

    model
        .forward_paged(
            &mut pool,
            &mut seq,
            &mut state_paged,
            next_token,
            prompt.len(),
        )
        .expect("forward paged");

    assert_eq!(
        bits(&state_paged.logits),
        bits(&state_ref.logits),
        "paged decode forward logits must match contiguous forward logits exactly"
    );

    fs::remove_file(path).ok();
}

#[test]
fn test_paged_v2_hierarchical_page_table_mapping_and_fork() {
    let mut pool = PagedPool::new(1, 4, 4, 16);
    let mut seq = ferrite::PagedSequenceV2::new(4); // 4 blocks per L2 table

    // Map virtual block 0, 1, 5 (crossing into second L2 page table)
    let b0 = pool.allocate_block();
    let b1 = pool.allocate_block();
    let b5 = pool.allocate_block();
    seq.page_table.map(0, b0);
    seq.page_table.map(1, b1);
    seq.page_table.map(5, b5);

    assert_eq!(seq.page_table.lookup(0), Some(b0));
    assert_eq!(seq.page_table.lookup(1), Some(b1));
    assert_eq!(seq.page_table.lookup(2), None);
    assert_eq!(seq.page_table.lookup(5), Some(b5));

    // Test fork with reference counting
    let mut forked = seq.fork(&mut pool);
    assert_eq!(pool.ref_counts[b0], 2);
    assert_eq!(pool.ref_counts[b1], 2);
    assert_eq!(pool.ref_counts[b5], 2);

    // Free forked sequence
    forked.free(&mut pool);
    assert_eq!(pool.ref_counts[b0], 1);
    assert_eq!(pool.ref_counts[b1], 1);
    assert_eq!(pool.ref_counts[b5], 1);

    // Free original sequence
    seq.free(&mut pool);
    assert_eq!(pool.ref_counts[b0], 0);
    assert_eq!(pool.ref_counts[b1], 0);
    assert_eq!(pool.ref_counts[b5], 0);
}

#[test]
fn test_paged_v2_matches_paged_v1_and_reference_forward() {
    let path = write("paged_v2", &fixture(2, true));
    let gguf = Gguf::open(&path).expect("open fixture");
    let model = Model::load(&gguf).expect("load model");

    let prompt = vec![0u32, 1u32, 2u32];

    // 1. Contiguous reference
    let mut state_ref = State::new(&model.config);
    model.prefill(&mut state_ref, &prompt, 0).expect("prefill");

    // 2. PagedAttention v2 with hierarchical page table
    let mut pool = PagedPool::new(model.config.layers, model.config.kv_dim(), 16, 8);
    let mut seq_v2 = ferrite::PagedSequenceV2::new(ferrite::BLOCKS_PER_PAGE_TABLE);
    let mut state_v2 = State::new(&model.config);

    model
        .prefill_paged_v2(&mut pool, &mut seq_v2, &mut state_v2, &prompt, 0)
        .expect("prefill_paged_v2");

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&state_v2.logits),
        bits(&state_ref.logits),
        "PagedAttention v2 prefill logits must match reference"
    );

    // 3. Decode step
    let next_token = 3u32;
    model
        .forward(&mut state_ref, next_token, prompt.len())
        .expect("forward ref");

    model
        .forward_paged_v2(
            &mut pool,
            &mut seq_v2,
            &mut state_v2,
            next_token,
            prompt.len(),
        )
        .expect("forward_paged_v2");

    assert_eq!(
        bits(&state_v2.logits),
        bits(&state_ref.logits),
        "PagedAttention v2 decode logits must match reference"
    );

    fs::remove_file(path).ok();
}
