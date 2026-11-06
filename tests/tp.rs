use ferrite::gguf::GgmlType;
use ferrite::model::Weight;
use ferrite::ops;
use ferrite::tp::{eval_tp_swiglu, matvec_column_parallel, matvec_row_parallel, ThreadPool};

#[test]
fn test_column_parallel_gemm_parity_across_thread_counts() {
    let rows = 32;
    let cols = 64;
    let weights_f32: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 17) % 100) as f32 * 0.02 - 1.0)
        .collect();
    let bytes: Vec<u8> = weights_f32
        .iter()
        .flat_map(|f| f.to_le_bytes().to_vec())
        .collect();

    let weight = Weight::from_raw(&bytes, GgmlType::F32, rows, cols).unwrap();

    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.05).sin()).collect();

    // Baseline reference: single-threaded Weight::matvec
    let mut baseline = vec![0.0f32; rows];
    let mut scratch = vec![0.0f32; cols];
    weight.matvec(&x, &mut baseline, &mut scratch).unwrap();

    // Test with 1, 2, 4, and 8 threads
    for num_threads in [1, 2, 4, 8] {
        let pool = ThreadPool::new(num_threads);
        let mut tp_out = vec![0.0f32; rows];
        matvec_column_parallel(&weight, &x, &mut tp_out, &pool).unwrap();

        let bits_base: Vec<u32> = baseline.iter().map(|f| f.to_bits()).collect();
        let bits_tp: Vec<u32> = tp_out.iter().map(|f| f.to_bits()).collect();
        assert_eq!(
            bits_base, bits_tp,
            "Column-parallel GEMM with {num_threads} threads must be bit-identical to baseline"
        );
    }
}

#[test]
fn test_row_parallel_gemm_parity_across_thread_counts() {
    let rows = 32;
    let cols = 64;
    let weights_f32: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 13) % 100) as f32 * 0.03 - 1.5)
        .collect();
    let bytes: Vec<u8> = weights_f32
        .iter()
        .flat_map(|f| f.to_le_bytes().to_vec())
        .collect();

    let weight = Weight::from_raw(&bytes, GgmlType::F32, rows, cols).unwrap();

    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.07).cos()).collect();

    // Baseline reference: single-threaded Weight::matvec
    let mut baseline = vec![0.0f32; rows];
    let mut scratch = vec![0.0f32; cols];
    weight.matvec(&x, &mut baseline, &mut scratch).unwrap();

    for num_threads in [1, 2, 4, 8] {
        let pool = ThreadPool::new(num_threads);
        let mut tp_out = vec![0.0f32; rows];
        matvec_row_parallel(&weight, &x, &mut tp_out, &pool).unwrap();

        for (r, (&base, &tp)) in baseline.iter().zip(&tp_out).enumerate() {
            assert!(
                (base - tp).abs() < 1e-4,
                "Row-parallel GEMM with {num_threads} threads drifted at row {r}: baseline {base} vs tp {tp}"
            );
        }
    }
}

#[test]
fn test_eval_tp_swiglu_parity() {
    let dim = 16;
    let ffn_dim = 32;

    let gate_f32: Vec<f32> = (0..ffn_dim * dim)
        .map(|i| ((i * 19) % 50) as f32 * 0.01)
        .collect();
    let up_f32: Vec<f32> = (0..ffn_dim * dim)
        .map(|i| ((i * 23) % 50) as f32 * 0.01)
        .collect();
    let down_f32: Vec<f32> = (0..dim * ffn_dim)
        .map(|i| ((i * 29) % 50) as f32 * 0.01)
        .collect();

    let gate_bytes: Vec<u8> = gate_f32
        .iter()
        .flat_map(|f| f.to_le_bytes().to_vec())
        .collect();
    let up_bytes: Vec<u8> = up_f32
        .iter()
        .flat_map(|f| f.to_le_bytes().to_vec())
        .collect();
    let down_bytes: Vec<u8> = down_f32
        .iter()
        .flat_map(|f| f.to_le_bytes().to_vec())
        .collect();

    let gate_weight = Weight::from_raw(&gate_bytes, GgmlType::F32, ffn_dim, dim).unwrap();
    let up_weight = Weight::from_raw(&up_bytes, GgmlType::F32, ffn_dim, dim).unwrap();
    let down_weight = Weight::from_raw(&down_bytes, GgmlType::F32, dim, ffn_dim).unwrap();

    let x: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // Baseline sequential SwiGLU:
    let mut gate_seq = vec![0.0f32; ffn_dim];
    let mut up_seq = vec![0.0f32; ffn_dim];
    let mut scratch_dim = vec![0.0f32; dim];
    let mut scratch_ffn = vec![0.0f32; ffn_dim];
    gate_weight
        .matvec(&x, &mut gate_seq, &mut scratch_dim)
        .unwrap();
    up_weight.matvec(&x, &mut up_seq, &mut scratch_dim).unwrap();
    for (g, u) in gate_seq.iter_mut().zip(&up_seq) {
        *g = ops::silu(*g) * u;
    }
    let mut baseline_out = vec![0.0f32; dim];
    down_weight
        .matvec(&gate_seq, &mut baseline_out, &mut scratch_ffn)
        .unwrap();

    // Tensor parallel SwiGLU:
    let pool = ThreadPool::new(4);
    let mut tp_out = vec![0.0f32; dim];
    eval_tp_swiglu(
        &gate_weight,
        &up_weight,
        &down_weight,
        &x,
        &mut tp_out,
        &pool,
    )
    .unwrap();

    for (i, (&base, &tp)) in baseline_out.iter().zip(&tp_out).enumerate() {
        assert!(
            (base - tp).abs() < 1e-4,
            "TP SwiGLU output drifted at index {i}: baseline {base} vs tp {tp}"
        );
    }
}

#[test]
fn test_tp_thread_pool_rapid_dispatch_stress() {
    let pool = ThreadPool::new(8);
    let total_tasks = 500;
    let n = 64;

    struct Ctx {
        ptr: *mut u32,
    }
    unsafe impl Sync for Ctx {}

    unsafe fn worker(ctx: *const (), start: usize, end: usize, _thread_id: usize) {
        let c = &*(ctx as *const Ctx);
        for i in start..end {
            *c.ptr.add(i) += 1;
        }
    }

    let mut counters = vec![0u32; n];
    let ctx = Ctx {
        ptr: counters.as_mut_ptr(),
    };

    for _ in 0..total_tasks {
        pool.dispatch(n, &ctx, worker);
    }

    for &count in &counters {
        assert_eq!(count, total_tasks as u32);
    }
}
