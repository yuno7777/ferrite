use std::fs;
use std::path::PathBuf;

use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::kv::{select_h2o_indices, KvCache, KvCacheType};
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

fn fixture(context_len: usize) -> Builder {
    Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(1))
        .meta("llama.embedding_length", Value::U32(DIM as u32))
        .meta("llama.feed_forward_length", Value::U32(FFN as u32))
        .meta("llama.attention.head_count", Value::U32(1))
        .meta("llama.attention.head_count_kv", Value::U32(1))
        .meta("llama.context_length", Value::U32(context_len as u32))
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
            bytes(&[0.1; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_up.weight",
            &[DIM as u64, FFN as u64],
            GgmlType::F32,
            bytes(&[0.1; DIM * FFN]),
        )
        .tensor(
            "blk.0.ffn_down.weight",
            &[FFN as u64, DIM as u64],
            GgmlType::F32,
            bytes(&[0.1; FFN * DIM]),
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

fn write(name: &str, builder: &Builder) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "ferrite-test-kv-{}-{name}.gguf",
        std::process::id()
    ));
    fs::write(&path, builder.build()).expect("write fixture");
    path
}

#[test]
fn test_kv_cache_evict_streaming_f32_and_quantized() {
    for cache_type in [KvCacheType::F32, KvCacheType::Q8_0, KvCacheType::Q4_0] {
        let mut cache = KvCache::new(1, 16, DIM, cache_type);

        // Store 16 distinct tokens
        for pos in 0..16 {
            let val = pos as f32 + 1.0;
            let vec = [val; DIM];
            cache.store(0, pos, &vec, &vec);
        }

        // Evict with StreamingLLM: keep sink_tokens = 4, target max_len = 12
        let new_len = cache.evict_streaming(16, 12, 4);
        assert_eq!(new_len, 12);

        // For F32, verify exact positions
        if cache_type == KvCacheType::F32 {
            let (k_slice, _) = cache.as_f32_slices().unwrap();
            // Sinks (0..4) must retain original values 1.0, 2.0, 3.0, 4.0
            for pos in 0..4 {
                let val = (pos + 1) as f32;
                assert_eq!(k_slice[pos * DIM], val);
            }
            // Positions 4..12 should have the shifted values originally at 8..16 (values 9.0..16.0)
            for (dst_pos, orig_pos) in (4..12).zip(8..16) {
                let val = (orig_pos + 1) as f32;
                assert_eq!(k_slice[dst_pos * DIM], val);
            }
            // Tail positions 12..16 should be zeroed
            for pos in 12..16 {
                assert_eq!(k_slice[pos * DIM], 0.0);
            }
        }
    }
}

#[test]
fn test_streaming_llm_continuous_generation_beyond_context() {
    let context_len = 16;
    let path = write("streaming_beyond_context", &fixture(context_len));
    let gguf = Gguf::open(&path).expect("open fixture");
    let model = Model::load(&gguf).expect("load model");

    let mut state = State::with_context(&model.config, context_len);
    let mut current_pos = 0;

    // Generate 64 tokens continuously on a 16-token physical context window!
    for step in 0..64 {
        let tok = (step % VOCAB) as u32;
        model
            .forward_streaming(&mut state, tok, &mut current_pos, 4)
            .expect("forward_streaming should succeed indefinitely");
        // Verify current_pos never exceeds context_len
        assert!(current_pos <= context_len);
        // Verify logits are finite and valid
        assert!(state.logits.iter().all(|l| l.is_finite()));
    }

    fs::remove_file(path).ok();
}

#[test]
fn test_h2o_heavy_hitter_selection_and_cache_compaction() {
    // Scores for 10 tokens: positions 0,1 are sinks, positions 8,9 are recent window
    // Positions 2..8 have middle scores
    let scores = vec![0.9, 0.8, 0.1, 0.95, 0.05, 0.85, 0.2, 0.3, 0.4, 0.5];
    let keep = select_h2o_indices(&scores, 2, 2, 2);

    // Must keep sinks 0, 1
    assert!(keep.contains(&0));
    assert!(keep.contains(&1));
    // Must keep window 8, 9
    assert!(keep.contains(&8));
    assert!(keep.contains(&9));
    // Top 2 middle scores are at index 3 (0.95) and index 5 (0.85)
    assert!(keep.contains(&3));
    assert!(keep.contains(&5));
    assert_eq!(keep.len(), 6);

    let mut cache = KvCache::new(1, 10, DIM, KvCacheType::F32);
    for pos in 0..10 {
        let val = (pos + 1) as f32;
        let v = [val; DIM];
        cache.store(0, pos, &v, &v);
    }

    cache.compact_indices(&keep);

    let (k, _) = cache.as_f32_slices().unwrap();
    // Compacted order: values for [0, 1, 3, 5, 8, 9] -> [1.0, 2.0, 4.0, 6.0, 9.0, 10.0]
    let expected = [1.0f32, 2.0, 4.0, 6.0, 9.0, 10.0];
    for (i, &exp) in expected.iter().enumerate() {
        assert_eq!(k[i * DIM], exp);
    }
    // Tail positions 6..10 must be zeroed
    for i in 6..10 {
        assert_eq!(k[i * DIM], 0.0);
    }
}

#[test]
fn test_mla_kv_cache_evict_range_and_compact() {
    use ferrite::kv::MlaKvCache;

    let layers = 1;
    let context = 16;
    let latent_dim = 4;
    let rope_dim = 2;
    let mut cache = MlaKvCache::new(layers, context, latent_dim, rope_dim);

    for pos in 0..16 {
        let val = (pos + 1) as f32;
        let latent = [val; 4];
        let rope = [val * 10.0; 2];
        cache.store(0, pos, &latent, &rope);
    }

    // Evict streaming: sinks = 2, max_len = 10 (evicts 6 tokens from pos 2..8)
    let new_len = cache.evict_streaming(16, 10, 2);
    assert_eq!(new_len, 10);

    // Positions 0..2 are sinks (vals 1.0, 2.0)
    for pos in 0..2 {
        let (lat, rope) = cache.get(0, pos);
        assert_eq!(lat[0], (pos + 1) as f32);
        assert_eq!(rope[0], ((pos + 1) * 10) as f32);
    }

    // Positions 2..10 are shifted from 8..16 (vals 9.0..16.0)
    for (dst, orig) in (2..10).zip(8..16) {
        let (lat, rope) = cache.get(0, dst);
        assert_eq!(lat[0], (orig + 1) as f32);
        assert_eq!(rope[0], ((orig + 1) * 10) as f32);
    }

    // Compact indices test
    let keep = vec![0, 1, 3, 5];
    cache.compact_indices(&keep);
    assert_eq!(cache.get(0, 0).0[0], 1.0);
    assert_eq!(cache.get(0, 1).0[0], 2.0);
    assert_eq!(cache.get(0, 2).0[0], 10.0); // was at dst pos 3 (orig 10)
    assert_eq!(cache.get(0, 3).0[0], 12.0); // was at dst pos 5 (orig 12)
    assert_eq!(cache.get(0, 4).0[0], 0.0); // zeroed
}

#[test]
fn test_adaptive_entropy_calculation() {
    use ferrite::kv::compute_distribution_entropy;

    // 1. One-hot / delta distribution has entropy 0.0
    let concentrated = vec![1000.0, 0.0, 0.0, 0.0];
    let (h_conc, norm_conc) = compute_distribution_entropy(&concentrated, true);
    assert!(h_conc < 1e-4, "expected near-zero entropy, got {h_conc}");
    assert!(norm_conc < 1e-4);

    // 2. Uniform distribution has normalized entropy 1.0
    let uniform = vec![5.0, 5.0, 5.0, 5.0];
    let (h_unif, norm_unif) = compute_distribution_entropy(&uniform, true);
    let expected_max = (4.0f32).ln();
    assert!((h_unif - expected_max).abs() < 1e-4);
    assert!((norm_unif - 1.0).abs() < 1e-4);
}

#[test]
fn test_adaptive_entropy_indices_selection() {
    use ferrite::kv::{select_adaptive_entropy_indices, AdaptiveEntropyConfig};

    let config = AdaptiveEntropyConfig {
        sink_tokens: 2,
        window_tokens: 2,
        min_capacity: 4,
        max_capacity: 10,
        target_entropy_ratio: 0.8,
    };

    // Low entropy logits (sharply peaked on token 0)
    let low_entropy = vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let keep_low = select_adaptive_entropy_indices(&low_entropy, &config);
    // Should compact down to min_capacity (4)
    assert_eq!(keep_low.len(), 4);
    assert!(keep_low.contains(&0));
    assert!(keep_low.contains(&1));
    assert!(keep_low.contains(&10));
    assert!(keep_low.contains(&11));

    // High entropy logits (uniform)
    let high_entropy = vec![2.0; 12];
    let keep_high = select_adaptive_entropy_indices(&high_entropy, &config);
    // Should retain up to max_capacity (10)
    assert_eq!(keep_high.len(), 10);
    assert!(keep_high.contains(&0));
    assert!(keep_high.contains(&1));
    assert!(keep_high.contains(&10));
    assert!(keep_high.contains(&11));
}

#[test]
fn test_state_and_model_adaptive_streaming_beyond_context() {
    use ferrite::kv::AdaptiveEntropyConfig;

    let context_len = 16;
    let path = write("adaptive_entropy_beyond_context", &fixture(context_len));
    let gguf = Gguf::open(&path).expect("open fixture");
    let model = Model::load(&gguf).expect("load model");

    let mut state = State::with_context(&model.config, context_len);
    let mut current_pos = 0;

    let config = AdaptiveEntropyConfig {
        sink_tokens: 2,
        window_tokens: 4,
        min_capacity: 8,
        max_capacity: 12,
        target_entropy_ratio: 0.75,
    };

    // Run 48 tokens through a 16-token context with adaptive streaming
    for step in 0..48 {
        let tok = (step % VOCAB) as u32;
        model
            .forward_adaptive_streaming(&mut state, tok, &mut current_pos, &config, None)
            .expect("forward_adaptive_streaming should succeed");
        assert!(current_pos <= context_len);
        assert!(state.logits.iter().all(|l| l.is_finite()));
    }

    fs::remove_file(path).ok();
}
