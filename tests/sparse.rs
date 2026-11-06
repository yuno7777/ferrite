use ferrite::ops::AttentionParams;
use ferrite::SparseAttentionMask;

#[test]
fn test_local_window_sparse_mask() {
    let mask = SparseAttentionMask::new_local(10, 3);
    assert_eq!(mask.num_tokens, 10);

    // Causal constraints
    assert!(!mask.is_allowed(2, 3)); // future token forbidden
    assert!(mask.is_allowed(2, 2)); // self allowed
    assert!(mask.is_allowed(2, 1)); // 1 back allowed (2 - 1 = 1 < 3)
    assert!(mask.is_allowed(2, 0)); // 2 back allowed (2 - 0 = 2 < 3)

    // Window exceeded
    assert!(!mask.is_allowed(5, 1)); // 5 - 1 = 4 >= 3 forbidden
    assert!(mask.is_allowed(5, 3)); // 5 - 3 = 2 < 3 allowed
    assert!(mask.is_allowed(5, 4)); // 5 - 4 = 1 < 3 allowed
    assert!(mask.is_allowed(5, 5)); // 5 - 5 = 0 < 3 allowed

    assert_eq!(mask.active_keys_for_query(5), vec![3, 4, 5]);
    assert!(mask.sparsity() > 0.0);
}

#[test]
fn test_strided_sparse_mask() {
    let mask = SparseAttentionMask::new_strided(16, 2, 4);

    // query = 10:
    // local window (< 2): 10, 9
    // strided ((10 - k) % 4 == 0): 6, 2
    let active = mask.active_keys_for_query(10);
    assert_eq!(active, vec![2, 6, 9, 10]);

    // Check individual allowed
    assert!(mask.is_allowed(10, 10));
    assert!(mask.is_allowed(10, 9));
    assert!(mask.is_allowed(10, 6));
    assert!(mask.is_allowed(10, 2));
    assert!(!mask.is_allowed(10, 8));
    assert!(!mask.is_allowed(10, 7));
    assert!(!mask.is_allowed(10, 5));
    assert!(!mask.is_allowed(10, 11)); // future
}

#[test]
fn test_bigbird_sparse_mask_and_sparse_attention() {
    let global_tokens = vec![0];
    let mask = SparseAttentionMask::new_bigbird(32, 2, 4, 1, 4, global_tokens);

    // Global token 0 should be accessible to all queries
    for q in 0..32 {
        assert!(
            mask.is_allowed(q, 0),
            "global token 0 must be attended by q={q}"
        );
    }

    // Local window tokens should always be accessible
    for q in 1..32 {
        assert!(mask.is_allowed(q, q));
        assert!(mask.is_allowed(q, q - 1));
    }

    // Test sparse attention numerical execution
    let head_dim = 16;
    let num_tokens = 32;
    let row_width = 16;

    let query = vec![0.5f32; head_dim];
    let mut keys = vec![0.0f32; num_tokens * row_width];
    let mut values = vec![0.0f32; num_tokens * row_width];

    for i in 0..num_tokens {
        for d in 0..head_dim {
            keys[i * row_width + d] = ((i * 17 + d * 3) % 97) as f32 / 100.0 - 0.45;
            values[i * row_width + d] = ((i * 31 + d * 7) % 89) as f32 / 100.0 - 0.4;
        }
    }

    let params = AttentionParams {
        row_width,
        offset: 0,
        len: num_tokens,
        scale: 1.0 / (head_dim as f32).sqrt(),
        logit_cap: None,
        window: None,
        position: num_tokens,
        tile_size: 16,
        v_row_width: None,
        v_offset: None,
    };

    let mut out = vec![0.0f32; head_dim];
    mask.sparse_attention(&query, &keys, &values, 15, params, &mut out);

    assert!(out.iter().all(|v| v.is_finite()));
    assert!(out.iter().any(|&v| v != 0.0));
}

#[test]
fn test_custom_block_sparse_mask() {
    // 16 tokens with block_size = 4 -> 4 blocks: 0, 1, 2, 3
    let active_blocks = vec![(0, 0), (1, 0), (1, 1), (2, 2), (3, 0), (3, 3)];
    let mask = SparseAttentionMask::new_block_sparse(16, 4, &active_blocks);

    // Query 5 is in block 1: attends to keys in block 0 and block 1 (keys 0..8)
    assert!(mask.is_allowed(5, 0));
    assert!(mask.is_allowed(5, 3));
    assert!(mask.is_allowed(5, 4));
    assert!(mask.is_allowed(5, 5));
    assert!(!mask.is_allowed(5, 6)); // future in block 1

    // Query 9 is in block 2: attends only to block 2 (keys 8..12)
    assert!(!mask.is_allowed(9, 0)); // block 0 forbidden for block 2
    assert!(mask.is_allowed(9, 8));
    assert!(mask.is_allowed(9, 9));
}
