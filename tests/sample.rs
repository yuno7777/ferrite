use ferrite::sample::{PenaltyConfig, Sampler};
use ferrite::watermark::{
    hash_context, is_token_green, WatermarkConfig, WatermarkDetector, WatermarkEngine,
};

#[test]
fn test_watermark_deterministic_partition() {
    let key = 0x1234_5678_9ABC_DEF0;
    let context = vec![42, 108, 999];
    let config = WatermarkConfig::new(key, 0.5, 2.5, 2);

    let seed1 = hash_context(config.key, &context, config.context_width);
    let seed2 = hash_context(config.key, &context, config.context_width);
    assert_eq!(seed1, seed2);

    // Context difference produces distinct seeds
    let other_context = vec![42, 108, 1000];
    let seed_other = hash_context(config.key, &other_context, config.context_width);
    assert_ne!(seed1, seed_other);

    // Key difference produces distinct seeds
    let seed_wrong_key = hash_context(0xDEAD_BEEF, &context, config.context_width);
    assert_ne!(seed1, seed_wrong_key);

    // Verify empirical green ratio across vocabulary
    let vocab_size = 10000;
    let mut green_count = 0;
    for tok in 0..vocab_size {
        if is_token_green(seed1, tok as u32, config.gamma) {
            green_count += 1;
        }
    }

    let fraction = green_count as f32 / vocab_size as f32;
    assert!(
        (fraction - 0.5).abs() < 0.02,
        "expected ~0.5 green fraction, got {fraction}"
    );
}

#[test]
fn test_watermark_engine_logit_bias() {
    let key = 0xFEED_FACE_CAFE_BEEF;
    let config = WatermarkConfig::new(key, 0.5, 3.0, 1);
    let context = vec![123];

    let mut logits = vec![0.0f32; 100];
    WatermarkEngine::apply_watermark(&mut logits, &context, &config);

    let seed = hash_context(config.key, &context, config.context_width);
    for (tok, &l) in logits.iter().enumerate() {
        if is_token_green(seed, tok as u32, config.gamma) {
            assert_eq!(l, 3.0);
        } else {
            assert_eq!(l, 0.0);
        }
    }
}

#[test]
fn test_watermarked_sampling_and_detection() {
    let key = 0xA1B2_C3D4_E5F6_0718;
    let config = WatermarkConfig::new(key, 0.5, 4.0, 1);
    let vocab_size = 256;
    let steps = 80;

    // 1. Generate sequence with watermark
    let mut wm_sampler = Sampler::new(0.8, 0, 1.0, 42).with_watermark(config);
    let mut wm_tokens = vec![10u32]; // initial prompt token

    for _ in 0..steps {
        let mut logits = vec![0.0f32; vocab_size];
        let tok = wm_sampler.sample_with_watermark(&mut logits, &wm_tokens);
        wm_tokens.push(tok);
    }

    // 2. Generate unwatermarked baseline sequence
    let mut plain_sampler = Sampler::new(0.8, 0, 1.0, 42);
    let mut plain_tokens = vec![10u32];

    for _ in 0..steps {
        let mut logits = vec![0.0f32; vocab_size];
        let tok = plain_sampler.sample(&mut logits);
        plain_tokens.push(tok);
    }

    // 3. Detect with correct secret key
    let wm_score = WatermarkDetector::detect(&wm_tokens, &config, 3.5);
    assert!(
        wm_score.is_watermarked,
        "watermarked text was not detected! z={}",
        wm_score.z_score
    );
    assert!(
        wm_score.green_fraction > 0.70,
        "green fraction too low: {}",
        wm_score.green_fraction
    );
    assert!(wm_score.z_score > 3.5);
    assert!(wm_score.p_value < 1e-3);

    // 4. Detect unwatermarked text with same detector -> must fail
    let plain_score = WatermarkDetector::detect(&plain_tokens, &config, 3.5);
    assert!(
        !plain_score.is_watermarked,
        "false positive on unwatermarked text! z={}",
        plain_score.z_score
    );
    assert!((plain_score.green_fraction - 0.5).abs() < 0.15);

    // 5. Detect watermarked text with wrong secret key -> must fail
    let wrong_config = WatermarkConfig::new(0x9999_8888_7777_6666, 0.5, 4.0, 1);
    let wrong_score = WatermarkDetector::detect(&wm_tokens, &wrong_config, 3.5);
    assert!(
        !wrong_score.is_watermarked,
        "false positive with wrong secret key! z={}",
        wrong_score.z_score
    );
}

#[test]
fn test_watermark_with_penalties_integration() {
    let config = WatermarkConfig::new(0x5555_AAAA_3333_CCCC, 0.5, 2.5, 1);
    let penalties = PenaltyConfig {
        repetition_penalty: 1.5,
        frequency_penalty: 0.1,
        presence_penalty: 0.2,
        last_n: 16,
    };

    let mut sampler = Sampler::new(0.7, 50, 0.95, 123)
        .with_penalties(penalties)
        .with_watermark(config);

    let history = vec![1, 2, 3, 2, 1];
    let mut logits = vec![1.0f32; 32];

    let tok = sampler.sample_with_history(&mut logits, &history);
    assert!(tok < 32);
}
