use ferrite::ops::{
    attention_online, flash_decoding_attention, tree_reduce_splits, AttentionParams,
    FlashDecodingSplit,
};

#[test]
fn test_flash_decoding_logarithmic_tree_reduction() {
    let dim = 4;
    let s0 = FlashDecodingSplit {
        max_score: 2.0,
        denominator: 3.0,
        output: vec![1.0, 2.0, 3.0, 4.0],
    };
    let s1 = FlashDecodingSplit {
        max_score: 3.5,
        denominator: 4.5,
        output: vec![2.0, 1.0, 0.5, 3.0],
    };
    let s2 = FlashDecodingSplit {
        max_score: 1.0,
        denominator: 2.0,
        output: vec![0.0, 1.5, 2.5, 1.0],
    };
    let s3 = FlashDecodingSplit {
        max_score: 4.0,
        denominator: 5.0,
        output: vec![3.0, 3.0, 1.0, 2.0],
    };

    // Pairwise manual reduction:
    // level 1: s01 = s0.reduce(s1), s23 = s2.reduce(s3)
    // level 2: expected = s01.reduce(s23)
    let s01 = s0.reduce(&s1);
    let s23 = s2.reduce(&s3);
    let expected = s01.reduce(&s23);

    // Tree reduction
    let reduced = tree_reduce_splits(vec![s0, s1, s2, s3]);

    assert_eq!(reduced.max_score.to_bits(), expected.max_score.to_bits());
    assert!((reduced.denominator - expected.denominator).abs() < 1e-6);
    for (a, b) in reduced.output.iter().zip(&expected.output) {
        assert!((a - b).abs() < 1e-6);
    }

    let mut out = vec![0.0f32; dim];
    reduced.finalize(&mut out);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_flash_decoding_matches_attention_online_across_threads_and_lengths() {
    let head_dim = 32;
    let row_width = 64; // grouped query or interleaved head row
    let offset = 0;

    for len in [128, 256, 512, 1024] {
        let mut keys = vec![0.0f32; len * row_width];
        let mut values = vec![0.0f32; len * row_width];

        for i in 0..keys.len() {
            keys[i] = ((i * 17 + 3) % 101) as f32 / 100.0 - 0.5;
            values[i] = ((i * 31 + 7) % 97) as f32 / 100.0 - 0.48;
        }

        let query: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 13 + 5) % 89) as f32 / 100.0 - 0.45)
            .collect();

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Test with both no logit cap and logit cap, with/without sliding window
        for (logit_cap, window) in [(None, None), (Some(50.0), None), (None, Some(256))] {
            let params = AttentionParams {
                row_width,
                offset,
                len,
                scale,
                logit_cap,
                window,
                position: len,
                tile_size: 64,
                v_row_width: None,
                v_offset: None,
            };

            let mut ref_out = vec![0.0f32; head_dim];
            attention_online(&query, &keys, &values, params, &mut ref_out);

            // Test across different thread configurations (1, 2, 4, 8)
            for threads in [1, 2, 4, 8] {
                let mut flash_out = vec![0.0f32; head_dim];
                flash_decoding_attention(&query, &keys, &values, params, threads, &mut flash_out);

                for (idx, (r, f)) in ref_out.iter().zip(&flash_out).enumerate() {
                    let diff = (r - f).abs();
                    assert!(
                        diff < 1e-4,
                        "mismatch at len={len}, threads={threads}, dim={idx}: ref={r}, flash={f}, diff={diff}"
                    );
                }
            }
        }
    }
}
