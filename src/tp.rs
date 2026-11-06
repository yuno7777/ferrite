//! Intra-node multi-threaded Column/Row Tensor Parallel (TP) GEMM engine.
//!
//! Provides zero-dependency, persistent-worker thread pool execution with barrier
//! synchronization for massive DeepSeek-V3 and Qwen GEMMs, eliminating thread
//! spawn overhead and partitioning weights across column and row dimensions.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::model::Weight;
use crate::ops;
use crate::quant;

/// A lightweight job descriptor passed to persistent worker threads without heap allocation.
#[derive(Clone, Copy)]
struct Job {
    ctx: *const (),
    run: unsafe fn(*const (), usize, usize, usize),
    total_len: usize,
}

unsafe impl Send for Job {}
unsafe impl Sync for Job {}

#[derive(Clone, Copy)]
enum JobPayload {
    Single(Job),
    Dual {
        job1: Job,
        job2: Job,
        split_thread: usize,
    },
}

#[inline]
fn execute_payload(payload: JobPayload, thread_id: usize, threads: usize) {
    match payload {
        JobPayload::Single(job) => {
            let total = job.total_len;
            let chunk = total.div_ceil(threads);
            let start = (thread_id * chunk).min(total);
            let end = ((thread_id + 1) * chunk).min(total);
            if start < end {
                unsafe {
                    (job.run)(job.ctx, start, end, thread_id);
                }
            }
        }
        JobPayload::Dual {
            job1,
            job2,
            split_thread,
        } => {
            if thread_id < split_thread {
                let t_id = thread_id;
                let t_count = split_thread;
                let total = job1.total_len;
                let chunk = total.div_ceil(t_count);
                let start = (t_id * chunk).min(total);
                let end = ((t_id + 1) * chunk).min(total);
                if start < end {
                    unsafe {
                        (job1.run)(job1.ctx, start, end, t_id);
                    }
                }
            } else {
                let t_id = thread_id - split_thread;
                let t_count = threads - split_thread;
                let total = job2.total_len;
                let chunk = total.div_ceil(t_count);
                let start = (t_id * chunk).min(total);
                let end = ((t_id + 1) * chunk).min(total);
                if start < end {
                    unsafe {
                        (job2.run)(job2.ctx, start, end, t_id);
                    }
                }
            }
        }
    }
}

const WORKER_SPIN_ITERS: usize = 32_768;

struct SharedPoolState {
    generation: AtomicUsize,
    pending_workers: AtomicUsize,
    sleeping_workers: AtomicUsize,
    shutdown: AtomicBool,
    payload: UnsafeCell<Option<JobPayload>>,
    park_lock: Mutex<()>,
    cvar_worker: Condvar,
}

unsafe impl Sync for SharedPoolState {}

/// Persistent thread pool optimized for zero-allocation tensor-parallel GEMM partitioning.
///
/// The calling thread executes partition `thread_id = 0` directly while `num_threads - 1`
/// persistent background threads execute `1..num_threads`, synchronized via a lock-free
/// atomic spin-then-park barrier.
pub struct ThreadPool {
    workers: Vec<JoinHandle<()>>,
    shared: Arc<SharedPoolState>,
    num_threads: usize,
}

impl ThreadPool {
    /// Creates a thread pool with `num_threads` total parallel workers (`num_threads - 1`
    /// background threads plus the calling thread as worker 0).
    pub fn new(num_threads: usize) -> Self {
        let threads = num_threads.max(1);
        let bg_threads = threads.saturating_sub(1);
        let shared = Arc::new(SharedPoolState {
            generation: AtomicUsize::new(0),
            pending_workers: AtomicUsize::new(0),
            sleeping_workers: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            payload: UnsafeCell::new(None),
            park_lock: Mutex::new(()),
            cvar_worker: Condvar::new(),
        });

        let mut workers = Vec::with_capacity(bg_threads);
        for thread_id in 1..threads {
            let shared_clone = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("ferrite-tp-{}", thread_id))
                .spawn(move || {
                    let mut local_gen = 0usize;
                    loop {
                        let mut spins = 0usize;
                        loop {
                            if shared_clone.shutdown.load(Ordering::Relaxed) {
                                return;
                            }
                            let g = shared_clone.generation.load(Ordering::Acquire);
                            if g != local_gen {
                                local_gen = g;
                                break;
                            }
                            if spins < WORKER_SPIN_ITERS {
                                spins += 1;
                                std::hint::spin_loop();
                            } else {
                                shared_clone.sleeping_workers.fetch_add(1, Ordering::SeqCst);
                                let mut guard = shared_clone.park_lock.lock().unwrap();
                                while !shared_clone.shutdown.load(Ordering::Relaxed)
                                    && shared_clone.generation.load(Ordering::Acquire) == local_gen
                                {
                                    guard = shared_clone.cvar_worker.wait(guard).unwrap();
                                }
                                shared_clone.sleeping_workers.fetch_sub(1, Ordering::SeqCst);
                            }
                        }

                        let payload = unsafe { (*shared_clone.payload.get()).unwrap() };
                        execute_payload(payload, thread_id, threads);
                        shared_clone.pending_workers.fetch_sub(1, Ordering::Release);
                    }
                })
                .expect("failed to spawn tp worker thread");

            workers.push(handle);
        }

        Self {
            workers,
            shared,
            num_threads: threads,
        }
    }

    /// Number of active worker threads in the pool.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    #[inline]
    fn run_payload(&self, payload: JobPayload) {
        let bg = self.num_threads.saturating_sub(1);
        unsafe {
            *self.shared.payload.get() = Some(payload);
        }
        self.shared.pending_workers.store(bg, Ordering::Relaxed);
        let next_gen = self
            .shared
            .generation
            .load(Ordering::Relaxed)
            .wrapping_add(1);
        self.shared.generation.store(next_gen, Ordering::Release);

        if self.shared.sleeping_workers.load(Ordering::SeqCst) > 0 {
            let _guard = self.shared.park_lock.lock().unwrap();
            self.shared.cvar_worker.notify_all();
        }

        // Calling thread executes partition 0 directly
        execute_payload(payload, 0, self.num_threads);

        // Wait for background workers to complete their slices
        while self.shared.pending_workers.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }

    /// Dispatches work across all persistent worker threads, blocking until completion.
    ///
    /// Executes `func(context, start, end, thread_id)` in parallel across disjoint index ranges.
    pub fn dispatch<T: Sync>(
        &self,
        total_len: usize,
        context: &T,
        func: unsafe fn(*const (), usize, usize, usize),
    ) {
        if total_len == 0 {
            return;
        }

        if self.num_threads <= 1 {
            unsafe {
                func(context as *const T as *const (), 0, total_len, 0);
            }
            return;
        }

        let job = Job {
            ctx: context as *const T as *const (),
            run: func,
            total_len,
        };
        self.run_payload(JobPayload::Single(job));
    }

    /// Dispatches two independent parallel jobs concurrently across worker partitions.
    ///
    /// For example: executing Attention Gate and Up projections simultaneously in FFN,
    /// or evaluating Q and K projections concurrently on separate thread subsets.
    pub fn dispatch_two<T1: Sync, T2: Sync>(
        &self,
        len1: usize,
        ctx1: &T1,
        func1: unsafe fn(*const (), usize, usize, usize),
        len2: usize,
        ctx2: &T2,
        func2: unsafe fn(*const (), usize, usize, usize),
    ) {
        if len1 == 0 && len2 == 0 {
            return;
        }
        if self.num_threads <= 1 {
            unsafe {
                if len1 > 0 {
                    func1(ctx1 as *const T1 as *const (), 0, len1, 0);
                }
                if len2 > 0 {
                    func2(ctx2 as *const T2 as *const (), 0, len2, 0);
                }
            }
            return;
        }

        let split_thread = (self.num_threads / 2).max(1);
        let job1 = Job {
            ctx: ctx1 as *const T1 as *const (),
            run: func1,
            total_len: len1,
        };
        let job2 = Job {
            ctx: ctx2 as *const T2 as *const (),
            run: func2,
            total_len: len2,
        };

        self.run_payload(JobPayload::Dual {
            job1,
            job2,
            split_thread,
        });
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        let _guard = self.shared.park_lock.lock().unwrap();
        self.shared.cvar_worker.notify_all();
        drop(_guard);
        for worker in self.workers.drain(..) {
            worker.join().ok();
        }
    }
}

// ---------------------------------------------------------------------------
// Column and Row Tensor Parallel GEMM Kernels
// ---------------------------------------------------------------------------

struct ColumnParallelContext<'a> {
    weight: &'a Weight<'a>,
    x: &'a [f32],
    out_ptr: *mut f32,
}

unsafe impl<'a> Sync for ColumnParallelContext<'a> {}

unsafe fn column_parallel_matvec_worker(
    ctx_ptr: *const (),
    start_row: usize,
    end_row: usize,
    _thread_id: usize,
) {
    let ctx = &*(ctx_ptr as *const ColumnParallelContext);
    let weight = ctx.weight;
    let x = ctx.x;
    let has_fused = quant::dot::supports(weight.ty);
    let mut scratch = if has_fused {
        Vec::new()
    } else {
        vec![0.0f32; weight.cols]
    };

    for r in start_row..end_row {
        let row = weight.row_slice(r);
        let val = if has_fused {
            quant::dot::fused(weight.ty, row, x).expect("supported")
        } else {
            quant::dequantize(weight.ty, row, &mut scratch).expect("valid dequant");
            ops::dot(&scratch, x)
        };
        *ctx.out_ptr.add(r) = val;
    }
}

/// Evaluates Column-Parallel GEMM `out = W * x` across persistent thread pool.
///
/// Partitions rows of the weight matrix across workers.
/// Each worker computes a disjoint slice of the output vector directly with zero-copy concatenation.
pub fn matvec_column_parallel(
    weight: &Weight,
    x: &[f32],
    out: &mut [f32],
    pool: &ThreadPool,
) -> std::io::Result<()> {
    debug_assert_eq!(x.len(), weight.cols);
    debug_assert_eq!(out.len(), weight.rows);

    let ctx = ColumnParallelContext {
        weight,
        x,
        out_ptr: out.as_mut_ptr(),
    };

    pool.dispatch(weight.rows, &ctx, column_parallel_matvec_worker);
    Ok(())
}

struct RowParallelContext<'a> {
    weight: &'a Weight<'a>,
    x: &'a [f32],
    partial_buffers: &'a [*mut f32],
    rows: usize,
}

unsafe impl<'a> Sync for RowParallelContext<'a> {}

unsafe fn row_parallel_matvec_worker(
    ctx_ptr: *const (),
    start_col: usize,
    end_col: usize,
    thread_id: usize,
) {
    let ctx = &*(ctx_ptr as *const RowParallelContext);
    let weight = ctx.weight;
    let x = ctx.x;
    let rows = ctx.rows;
    let partial = ctx.partial_buffers[thread_id];

    // For unquantized F32 matrices, row-parallel column partitioning performs partial dot products
    if weight.ty == crate::gguf::GgmlType::F32 {
        let raw_f32 = weight.bytes();
        let f32_slice: &[f32] =
            std::slice::from_raw_parts(raw_f32.as_ptr() as *const f32, rows * weight.cols);
        let x_part = &x[start_col..end_col];

        for r in 0..rows {
            let row_offset = r * weight.cols;
            let w_row_part = &f32_slice[row_offset + start_col..row_offset + end_col];
            let dot = ops::dot(w_row_part, x_part);
            *partial.add(r) = dot;
        }
    } else {
        // Fallback for block-quantized types where column slicing across block boundaries
        // requires row decompression
        let mut scratch = vec![0.0f32; weight.cols];
        let x_part = &x[start_col..end_col];

        for r in 0..rows {
            let row = weight.row_slice(r);
            quant::dequantize(weight.ty, row, &mut scratch).expect("valid dequant");
            let dot = ops::dot(&scratch[start_col..end_col], x_part);
            *partial.add(r) = dot;
        }
    }
}

/// Evaluates Row-Parallel GEMM `out = W * x` across persistent thread pool.
///
/// Partitions columns of the weight matrix and input `x` across workers,
/// executing partial GEMMs in parallel and performing an All-Reduce sum into `out`.
pub fn matvec_row_parallel(
    weight: &Weight,
    x: &[f32],
    out: &mut [f32],
    pool: &ThreadPool,
) -> std::io::Result<()> {
    debug_assert_eq!(x.len(), weight.cols);
    debug_assert_eq!(out.len(), weight.rows);

    let num_threads = pool.num_threads();
    let mut thread_buffers = vec![vec![0.0f32; weight.rows]; num_threads];
    let raw_ptrs: Vec<*mut f32> = thread_buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();

    let ctx = RowParallelContext {
        weight,
        x,
        partial_buffers: &raw_ptrs,
        rows: weight.rows,
    };

    pool.dispatch(weight.cols, &ctx, row_parallel_matvec_worker);

    // All-Reduce (Sum reduction)
    out.fill(0.0);
    for buf in &thread_buffers {
        for (slot, &val) in out.iter_mut().zip(buf) {
            *slot += val;
        }
    }

    Ok(())
}

/// Evaluates an MLP / SwiGLU block using combined Column-Parallel and Row-Parallel GEMMs.
///
/// 1. `gate` and `up` projections are computed Column-Parallel (rows sharded across workers).
/// 2. `silu(gate) * up` is evaluated thread-locally with zero cross-thread communication.
/// 3. `down` projection is computed Row-Parallel with an All-Reduce sum into `out`.
pub fn eval_tp_swiglu(
    gate_weight: &Weight,
    up_weight: &Weight,
    down_weight: &Weight,
    x: &[f32],
    out: &mut [f32],
    pool: &ThreadPool,
) -> std::io::Result<()> {
    let ffn_dim = gate_weight.rows;
    let mut gate_out = vec![0.0f32; ffn_dim];
    let mut up_out = vec![0.0f32; ffn_dim];

    // 1. Column-parallel gate and up
    matvec_column_parallel(gate_weight, x, &mut gate_out, pool)?;
    matvec_column_parallel(up_weight, x, &mut up_out, pool)?;

    // 2. Fused elementwise SwiGLU: h = silu(gate) * up
    for (g, u) in gate_out.iter_mut().zip(&up_out) {
        *g = ops::silu(*g) * u;
    }

    // 3. Row-parallel down projection with all-reduce sum
    matvec_row_parallel(down_weight, &gate_out, out, pool)?;
    Ok(())
}

/// Fused SwiGLU execution running gate and up projections concurrently across dual worker partitions.
pub fn eval_tp_swiglu_dual(
    gate_weight: &Weight,
    up_weight: &Weight,
    down_weight: &Weight,
    x: &[f32],
    out: &mut [f32],
    pool: &ThreadPool,
) -> std::io::Result<()> {
    let ffn_dim = gate_weight.rows;
    let mut gate_out = vec![0.0f32; ffn_dim];
    let mut up_out = vec![0.0f32; ffn_dim];

    let ctx1 = ColumnParallelContext {
        weight: gate_weight,
        x,
        out_ptr: gate_out.as_mut_ptr(),
    };
    let ctx2 = ColumnParallelContext {
        weight: up_weight,
        x,
        out_ptr: up_out.as_mut_ptr(),
    };

    pool.dispatch_two(
        ffn_dim,
        &ctx1,
        column_parallel_matvec_worker,
        ffn_dim,
        &ctx2,
        column_parallel_matvec_worker,
    );

    // Fused elementwise SwiGLU: h = silu(gate) * up
    for (g, u) in gate_out.iter_mut().zip(&up_out) {
        *g = ops::silu(*g) * u;
    }

    // Row-parallel down projection with all-reduce sum
    matvec_row_parallel(down_weight, &gate_out, out, pool)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgmlType;

    #[test]
    fn thread_pool_dispatch_partitions_disjoint_ranges() {
        let pool = ThreadPool::new(4);
        let n = 100;
        let mut results = vec![0usize; n];

        struct Ctx {
            ptr: *mut usize,
        }
        unsafe impl Sync for Ctx {}

        unsafe fn worker(ctx: *const (), start: usize, end: usize, thread_id: usize) {
            let c = &*(ctx as *const Ctx);
            for i in start..end {
                *c.ptr.add(i) = thread_id + 1;
            }
        }

        let ctx = Ctx {
            ptr: results.as_mut_ptr(),
        };
        pool.dispatch(n, &ctx, worker);

        for &val in &results {
            assert!((1..=4).contains(&val));
        }
    }

    #[test]
    fn column_and_row_parallel_gemm_parity() {
        let rows = 8;
        let cols = 16;
        let weights_f32: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.05).collect();
        let bytes: Vec<u8> = weights_f32
            .iter()
            .flat_map(|f| f.to_le_bytes().to_vec())
            .collect();

        let weight = Weight::from_raw(&bytes, GgmlType::F32, rows, cols).unwrap();

        let x: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Sequential reference
        let mut ref_out = vec![0.0f32; rows];
        let mut scratch = vec![0.0f32; cols];
        weight.matvec(&x, &mut ref_out, &mut scratch).unwrap();

        let pool = ThreadPool::new(4);

        // Column parallel
        let mut col_out = vec![0.0f32; rows];
        matvec_column_parallel(&weight, &x, &mut col_out, &pool).unwrap();
        for (a, b) in ref_out.iter().zip(&col_out) {
            assert!((a - b).abs() < 1e-5, "col TP {a} != {b}");
        }

        // Row parallel
        let mut row_out = vec![0.0f32; rows];
        matvec_row_parallel(&weight, &x, &mut row_out, &pool).unwrap();
        for (a, b) in ref_out.iter().zip(&row_out) {
            assert!((a - b).abs() < 1e-4, "row TP {a} != {b}");
        }
    }
}
