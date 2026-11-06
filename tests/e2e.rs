use std::fs;
use std::path::PathBuf;

use ferrite::audio::{
    extract_log_mel_spectrogram, stitch_audio_and_text_embeddings, AudioPatchExtractor,
    AudioProjector, AudioWaveform, SpectrogramConfig,
};
use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::model::{Model, State};
use ferrite::paged::{PagedPool, PagedSequence};
use ferrite::prefix::RadixCache;
use ferrite::quant::fp8::{f32_to_fp8_e4m3, f32_to_fp8_e5m2};
use ferrite::sample::{PenaltyConfig, Sampler};
use ferrite::synth::Builder;
use ferrite::vision::Image;
use ferrite::watermark::{WatermarkConfig, WatermarkDetector};
use ferrite::{gemm_fp8_fused, Fp8HardwareTuning};

const DIM: usize = 4;
const VOCAB: usize = 4;
const FFN: usize = 8;
const EPS: f32 = 1e-5;

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn embeddings() -> Vec<f32> {
    vec![
        1.0, 0.0, 0.0, 0.0, // Token 0
        0.0, 2.0, 0.0, -1.0, // Token 1
        1.0, 2.0, 3.0, 4.0, // Token 2
        -2.0, 0.5, 1.0, 0.0, // Token 3
    ]
}

fn build_e2e_fixture() -> Builder {
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
        .meta("llama.context_length", Value::U32(64))
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

fn temp_model_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ferrite-e2e-{}-{tag}.gguf", std::process::id()));
    p
}

#[test]
fn test_end_to_end_llama_inference_pipeline() {
    let path = temp_model_path("llama");
    fs::write(&path, build_e2e_fixture().build()).expect("write model");

    let gguf = Gguf::open(&path).expect("open gguf");
    let model = Model::load(&gguf).expect("load model");

    let mut state = State::new(&model.config);
    let prompt = vec![0, 1, 2];

    // 1. Prefill prompt
    model
        .prefill(&mut state, &prompt, 0)
        .expect("prefill prompt");

    // 2. Sample next tokens with penalties and watermarking
    let wm_config = WatermarkConfig::new(0x7777_8888_9999_AAAA, 0.5, 3.0, 1);
    let mut sampler = Sampler::new(0.7, 4, 1.0, 42)
        .with_penalties(PenaltyConfig {
            repetition_penalty: 1.2,
            frequency_penalty: 0.1,
            presence_penalty: 0.1,
            last_n: 16,
        })
        .with_watermark(wm_config);

    let mut generated = prompt.clone();
    let steps = 12;

    for _ in 0..steps {
        let tok = sampler.sample_with_history(&mut state.logits, &generated);
        let pos = generated.len();
        generated.push(tok);
        model.forward(&mut state, tok, pos).expect("forward step");
    }

    assert_eq!(generated.len(), prompt.len() + steps);

    // Verify watermark detection on generated tokens
    let wm_score = WatermarkDetector::detect(&generated, &wm_config, 1.5);
    assert!(wm_score.total_tokens >= steps);

    let _ = fs::remove_file(&path);
}

#[test]
fn test_end_to_end_paged_attention_and_prefix_caching_lifecycle() {
    let path = temp_model_path("paged_prefix");
    fs::write(&path, build_e2e_fixture().build()).expect("write model");

    let gguf = Gguf::open(&path).expect("open gguf");
    let model = Model::load(&gguf).expect("load model");

    let tokens_per_block = 2;
    let mut pool = PagedPool::new(
        model.config.layers,
        model.config.kv_dim(),
        tokens_per_block,
        16,
    );
    let mut cache = RadixCache::with_block_size(tokens_per_block);

    // System prompt: 4 tokens = 2 blocks [0, 1], [2, 3]
    let sys_prompt = vec![0, 1, 2, 3];
    let mut seq1 = PagedSequence::new();
    let mut state1 = State::new(&model.config);

    model
        .prefill_paged(&mut pool, &mut seq1, &mut state1, &sys_prompt, 0)
        .expect("prefill seq1");

    // Index into radix cache
    cache.insert(&sys_prompt, &seq1.block_table, &mut pool);
    assert_eq!(cache.len(), 2);

    // Save prefix cache to disk
    let mut persist_path = std::env::temp_dir();
    persist_path.push(format!("ferrite-e2e-prefix-{}.bin", std::process::id()));
    cache
        .save_to_file(&pool, &persist_path)
        .expect("save prefix cache");

    // Load prefix cache into a fresh pool
    let mut pool2 = PagedPool::new(
        model.config.layers,
        model.config.kv_dim(),
        tokens_per_block,
        16,
    );
    let mut loaded_cache =
        RadixCache::load_from_file(&persist_path, &mut pool2).expect("load prefix cache");

    // Sequence 2 shares the same system prompt plus user token 1
    let full_prompt = vec![0, 1, 2, 3, 1];
    let (matched_blocks, matched_tokens) = loaded_cache.match_prefix(&full_prompt);
    assert_eq!(matched_tokens, 4);
    assert_eq!(matched_blocks.len(), 2);

    let mut seq2 = PagedSequence::new();
    loaded_cache.apply_prefix_to_sequence(&matched_blocks, matched_tokens, &mut seq2, &mut pool2);

    // Compute remaining prefill for suffix token
    let mut state2 = State::new(&model.config);
    model
        .prefill_paged(
            &mut pool2,
            &mut seq2,
            &mut state2,
            &full_prompt[matched_tokens..],
            matched_tokens,
        )
        .expect("prefill remaining");

    // Ground-truth without cache in pool2
    let mut seq_gt = PagedSequence::new();
    let mut state_gt = State::new(&model.config);
    model
        .prefill_paged(&mut pool2, &mut seq_gt, &mut state_gt, &full_prompt, 0)
        .expect("prefill ground truth");

    // Exact bit-identical parity
    assert_eq!(state2.logits.len(), state_gt.logits.len());
    for (v2, v_gt) in state2.logits.iter().zip(&state_gt.logits) {
        assert_eq!(v2.to_bits(), v_gt.to_bits());
    }

    seq1.free(&mut pool);
    seq2.free(&mut pool2);
    seq_gt.free(&mut pool2);

    let _ = fs::remove_file(&path);
    let _ = fs::remove_file(&persist_path);
}

#[test]
fn test_end_to_end_multimodal_audio_and_vision_pipeline() {
    let llm_dim = 8;

    // 1. Audio processing
    let sample_rate = 16000;
    let audio_samples = vec![0.1f32; 800]; // 0.05 seconds
    let waveform = AudioWaveform::new(audio_samples, sample_rate);

    let config = SpectrogramConfig {
        sample_rate: 16000,
        n_fft: 400,
        hop_length: 160,
        n_mels: 80,
        f_min: 0.0,
        f_max: 8000.0,
    };

    let spec = extract_log_mel_spectrogram(&waveform, &config);
    let extractor = AudioPatchExtractor::new(2, 2);
    let patches = extractor.extract_patches(&spec);

    let patch_dim = 2 * 80;
    let weight = vec![0.01f32; llm_dim * patch_dim];
    let projector = AudioProjector::new(patch_dim, llm_dim, weight, None).expect("projector");
    let audio_tokens = projector.project(&patches).expect("project audio");

    // 2. Vision processing
    let rgb = vec![128u8; 16 * 16 * 3];
    let image = Image::from_interleaved_rgb(16, 16, &rgb).expect("image");
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);

    // 3. Multimodal text and audio embedding stitching
    let text_embeddings = vec![1.0f32; 2 * llm_dim]; // 2 text prompt tokens
    let stitched = stitch_audio_and_text_embeddings(&text_embeddings, &audio_tokens, 1, llm_dim)
        .expect("stitch");

    assert_eq!(stitched.len(), (2 + audio_tokens.len()) * llm_dim);
}

#[test]
fn test_end_to_end_quantization_and_gemm_execution() {
    let rows = 4;
    let cols = 64;
    let block_size = 32;

    let mut w_f32 = vec![0.0f32; rows * cols];
    let mut w_e4m3 = vec![0u8; rows * cols];
    let mut w_e5m2 = vec![0u8; rows * cols];

    for (i, w) in w_f32.iter_mut().enumerate() {
        *w = ((i as f32 * 0.13).sin() * 2.0).round() * 0.5;
        w_e4m3[i] = f32_to_fp8_e4m3(*w);
        w_e5m2[i] = f32_to_fp8_e5m2(*w);
    }

    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.19).cos()).collect();
    let scales = vec![1.0f32; (rows * cols) / block_size];

    let mut out_e4m3 = vec![0.0f32; rows];
    let mut out_e5m2 = vec![0.0f32; rows];

    let tuning = Fp8HardwareTuning::auto_detect();
    assert_eq!(tuning.unroll_k, 8);

    gemm_fp8_fused(
        &w_e4m3,
        Some(&scales),
        block_size,
        false,
        &x,
        &mut out_e4m3,
        rows,
        cols,
    );
    gemm_fp8_fused(
        &w_e5m2,
        Some(&scales),
        block_size,
        true,
        &x,
        &mut out_e5m2,
        rows,
        cols,
    );

    for r in 0..rows {
        assert!(!out_e4m3[r].is_nan());
        assert!(!out_e5m2[r].is_nan());
    }
}
