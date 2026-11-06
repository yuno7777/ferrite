//! Llama-family transformer: hyperparameters, weight binding, forward pass.

use std::io::{Error, ErrorKind, Result};

use crate::gguf::{GgmlType, Gguf};
use crate::kv::{KvCache, KvCacheType};
use crate::ops;
use crate::quant;

fn bad(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

/// Architectures whose tensor names and layer shape this implements. Others
/// are rejected by name rather than half-loaded — a model that runs with the
/// wrong layer structure produces fluent nonsense, not an error.
const SUPPORTED: [&str; 13] = [
    "llama",
    "qwen",
    "qwen2",
    "qwen2moe",
    "qwen2_moe",
    "qwen2vl",
    "qwen2_vl",
    "gemma",
    "gemma2",
    "mixtral",
    "deepseek",
    "deepseek2",
    "deepseek3",
];

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: String,
    pub layers: usize,
    pub embedding: usize,
    pub heads: usize,
    pub kv_heads: usize,
    /// Usually `embedding / heads`, but the file can say otherwise.
    pub head_dim: usize,
    pub ffn: usize,
    pub context: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub rope_dim: usize,
    pub rope_scaling: Option<ops::RopeScaling>,
    pub attn_logit_softcapping: Option<f32>,
    pub final_logit_softcapping: Option<f32>,
    pub sliding_window: Option<usize>,
    pub embedding_scale: Option<f32>,
    pub expert_count: Option<usize>,
    pub expert_used_count: Option<usize>,
    pub expert_shared_count: Option<usize>,
    pub expert_weights_scale: Option<f32>,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: Option<usize>,
    pub qk_nope_head_dim: Option<usize>,
    pub qk_rope_head_dim: Option<usize>,
    pub v_head_dim: Option<usize>,
    pub mrope_sections: Option<[usize; 4]>,
    pub nextn_predict_layers: Option<usize>,
}

impl Config {
    pub fn from_gguf(model: &Gguf) -> Result<Self> {
        let arch = model
            .arch()
            .ok_or_else(|| bad("file does not name an architecture"))?
            .to_string();
        if !SUPPORTED.contains(&arch.as_str()) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!("architecture {arch:?} is not implemented (supported: {SUPPORTED:?})"),
            ));
        }

        let need = |key: &str| {
            model
                .arch_u64(key)
                .map(|v| v as usize)
                .ok_or_else(|| bad(format!("missing {arch}.{key}")))
        };

        let layers = need("block_count")?;
        let embedding = need("embedding_length")?;
        let heads = need("attention.head_count")?;
        let ffn = need("feed_forward_length")?;
        let kv_heads = model
            .arch_u64("attention.head_count_kv")
            .map(|v| v as usize)
            .unwrap_or(heads);
        let head_dim = model
            .arch_u64("attention.key_length")
            .or_else(|| model.arch_u64("attention.head_dim"))
            .map(|v| v as usize)
            .or_else(|| {
                let nope = model.arch_u64("attention.qk_nope_head_dim")?;
                let rope = model.arch_u64("attention.qk_rope_head_dim")?;
                Some((nope + rope) as usize)
            })
            .unwrap_or_else(|| embedding.checked_div(heads).unwrap_or(0));

        if heads == 0 || kv_heads == 0 || head_dim == 0 {
            return Err(bad("head counts must be non-zero"));
        }
        if heads % kv_heads != 0 {
            return Err(bad(format!(
                "{heads} heads is not a multiple of {kv_heads} kv heads"
            )));
        }
        // In LLaMA and Qwen, heads * head_dim == embedding. In Gemma 2 and DeepSeek,
        // head_dim is decoupled from embedding.
        if !arch.starts_with("gemma")
            && !arch.starts_with("deepseek")
            && heads * head_dim != embedding
        {
            return Err(bad(format!(
                "{heads} heads of {head_dim} do not add up to an embedding of {embedding}"
            )));
        }

        let vocab = model
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .or_else(|| {
                model
                    .tensor("token_embd.weight")
                    .and_then(|t| t.dims.get(1).copied())
                    .map(|v| v as usize)
            })
            .ok_or_else(|| bad("cannot determine vocabulary size"))?;

        let attn_logit_softcapping = model
            .arch_f32("attention.logit_softcapping")
            .or_else(|| model.arch_f32("attn_logit_softcapping"));
        let final_logit_softcapping = model.arch_f32("final_logit_softcapping");
        let sliding_window = model
            .arch_u64("attention.sliding_window")
            .map(|v| v as usize);
        let embedding_scale = if arch == "gemma" || arch == "gemma2" {
            Some((embedding as f32).sqrt())
        } else {
            None
        };

        let rope_dim = model
            .arch_u64("rope.dimension_count")
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        let rope_scaling = if let Some(scale_type) = model
            .arch_str("rope.scaling.type")
            .or_else(|| model.meta_str("rope.scaling.type"))
        {
            let factor = model
                .arch_f32("rope.scaling.factor")
                .or_else(|| model.meta_f32("rope.scaling.factor"))
                .unwrap_or(1.0);
            match scale_type {
                "linear" => Some(ops::RopeScaling::Linear { factor }),
                "yarn" => {
                    let orig_ctx = model
                        .arch_u64("rope.scaling.original_context_length")
                        .or_else(|| model.meta_u64("rope.scaling.original_context_length"))
                        .map(|v| v as usize)
                        .unwrap_or(4096);
                    let ext_factor = model
                        .arch_f32("rope.scaling.ext_factor")
                        .or_else(|| model.meta_f32("rope.scaling.ext_factor"))
                        .unwrap_or(0.0);
                    let attn_factor = model
                        .arch_f32("rope.scaling.attn_factor")
                        .or_else(|| model.meta_f32("rope.scaling.attn_factor"))
                        .unwrap_or(0.0);
                    let beta_fast = model
                        .arch_f32("rope.scaling.beta_fast")
                        .or_else(|| model.meta_f32("rope.scaling.beta_fast"))
                        .unwrap_or(32.0);
                    let beta_slow = model
                        .arch_f32("rope.scaling.beta_slow")
                        .or_else(|| model.meta_f32("rope.scaling.beta_slow"))
                        .unwrap_or(1.0);
                    Some(ops::RopeScaling::Yarn {
                        factor,
                        orig_ctx,
                        ext_factor,
                        attn_factor,
                        beta_fast,
                        beta_slow,
                    })
                }
                _ => None,
            }
        } else {
            None
        };

        Ok(Self {
            arch: arch.clone(),
            layers,
            embedding,
            heads,
            kv_heads,
            head_dim,
            ffn,
            context: model
                .arch_u64("context_length")
                .map(|v| v as usize)
                .unwrap_or(2048),
            vocab,
            eps: model
                .arch_f32("attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-5),
            rope_base: model.arch_f32("rope.freq_base").unwrap_or_else(|| {
                if arch.starts_with("qwen") {
                    1_000_000.0
                } else {
                    10000.0
                }
            }),
            rope_dim,
            rope_scaling,
            attn_logit_softcapping,
            final_logit_softcapping,
            sliding_window,
            embedding_scale,
            expert_count: model
                .arch_u64("expert_count")
                .or_else(|| model.meta_u64("expert_count"))
                .map(|v| v as usize),
            expert_used_count: model
                .arch_u64("expert_used_count")
                .or_else(|| model.meta_u64("expert_used_count"))
                .map(|v| v as usize),
            expert_shared_count: model
                .arch_u64("expert_shared_count")
                .or_else(|| model.meta_u64("expert_shared_count"))
                .map(|v| v as usize),
            expert_weights_scale: model
                .arch_f32("expert_weights_scale")
                .or_else(|| model.meta_f32("expert_weights_scale"))
                .or_else(|| model.arch_f32("expert_weight_scale")),
            q_lora_rank: model.arch_u64("attention.q_lora_rank").map(|v| v as usize),
            kv_lora_rank: model.arch_u64("attention.kv_lora_rank").map(|v| v as usize),
            qk_nope_head_dim: model
                .arch_u64("attention.qk_nope_head_dim")
                .map(|v| v as usize),
            qk_rope_head_dim: model
                .arch_u64("attention.qk_rope_head_dim")
                .map(|v| v as usize),
            v_head_dim: model
                .arch_u64("attention.value_length")
                .or_else(|| model.arch_u64("attention.v_head_dim"))
                .map(|v| v as usize),
            mrope_sections: model
                .get(&format!("{arch}.rope.dimension_sections"))
                .or_else(|| model.get("rope.dimension_sections"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    let mut secs = [0usize; 4];
                    for (i, val) in arr.iter().take(4).enumerate() {
                        secs[i] = val.as_u64().unwrap_or(0) as usize;
                    }
                    secs
                }),
            nextn_predict_layers: model
                .arch_u64("nextn_predict_layers")
                .or_else(|| model.meta_u64("nextn_predict_layers"))
                .map(|v| v as usize),
        })
    }

    /// Width of the key and value vectors, which is smaller than `embedding`
    /// whenever grouped-query attention is in use.
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// How many query heads share each key/value head.
    pub fn group(&self) -> usize {
        self.heads / self.kv_heads
    }
}

/// Reusable buffers for a matvec, so a decode step allocates nothing.
///
/// Held by [`State`]. Sized for the widest matrix in the model.
#[derive(Clone)]
pub struct Workspace {
    /// Expanded row, for weight types with no faster path.
    scratch: Vec<f32>,
    /// The activation, quantized once per matvec and shared by every row.
    quantized: quant::activation::Quantized,
    /// Batch activations quantized once per matmul (`batch` entries).
    batch_quantized: Vec<quant::activation::Quantized>,
    /// Persistent worker thread pool so matvec/matmul never spawns OS threads per call.
    pool: Option<std::sync::Arc<crate::tp::ThreadPool>>,
    /// Set false to force the exact f32 path everywhere.
    ///
    /// The integer kernels are an approximation — small, bounded, and the same
    /// trade every fast inference engine makes, but a real one. This is the
    /// switch for anyone who would rather have the exact numbers, and for
    /// bisecting a suspected numerical bug.
    pub int8: bool,
}

impl Workspace {
    pub fn new(widest: usize) -> Self {
        Self {
            scratch: vec![0.0; widest],
            quantized: quant::activation::Quantized::with_capacity(widest),
            batch_quantized: Vec::new(),
            pool: None,
            int8: true,
        }
    }

    fn ensure_pool(&mut self, threads: usize) -> &crate::tp::ThreadPool {
        let need_new = match &self.pool {
            Some(p) => p.num_threads() != threads,
            None => true,
        };
        if need_new {
            self.pool = Some(std::sync::Arc::new(crate::tp::ThreadPool::new(threads)));
        }
        self.pool.as_ref().unwrap()
    }
}

#[derive(Clone, Copy)]
struct SyncMutPtr(*mut f32);
unsafe impl Send for SyncMutPtr {}
unsafe impl Sync for SyncMutPtr {}

struct MatvecJobCtx<'a, 'w> {
    weight: &'a Weight<'w>,
    x: &'a [f32],
    quantized: &'a quant::activation::Quantized,
    int8: bool,
    out_ptr: SyncMutPtr,
}

unsafe fn run_matvec_worker(ctx_ptr: *const (), start: usize, end: usize, _thread_id: usize) {
    let ctx = &*(ctx_ptr as *const MatvecJobCtx<'_, '_>);
    let count = end - start;
    let out_slice = std::slice::from_raw_parts_mut(ctx.out_ptr.0.add(start), count);
    let mut scratch = if ctx.int8 || quant::dot::supports(ctx.weight.ty) {
        Vec::new()
    } else {
        vec![0.0f32; ctx.weight.cols]
    };
    let _ = ctx.weight.rows_into(
        out_slice,
        start,
        ctx.x,
        ctx.quantized,
        ctx.int8,
        &mut scratch,
    );
}

struct MatmulJobCtx<'a, 'w> {
    weight: &'a Weight<'w>,
    x: &'a [f32],
    batch_quantized: &'a [quant::activation::Quantized],
    int8: bool,
    batch: usize,
    out_ptr: SyncMutPtr,
}

unsafe fn run_matmul_worker(ctx_ptr: *const (), start: usize, end: usize, _thread_id: usize) {
    let ctx = &*(ctx_ptr as *const MatmulJobCtx<'_, '_>);
    let mut scratch = if ctx.int8 || quant::dot::supports(ctx.weight.ty) {
        Vec::new()
    } else {
        vec![0.0f32; ctx.weight.cols]
    };
    let _ = ctx.weight.matmul_row_range(
        ctx.x,
        ctx.batch_quantized,
        ctx.int8,
        ctx.out_ptr,
        start,
        end - start,
        ctx.batch,
        &mut scratch,
    );
}

/// A weight matrix left in its stored, quantized form.
///
/// Rows are expanded one at a time during a matvec rather than up front: a
/// 4 GB model would otherwise need 16 GB of f32 to multiply by.
pub struct Weight<'a> {
    pub rows: usize,
    pub cols: usize,
    pub ty: GgmlType,
    row_bytes: usize,
    bytes: &'a [u8],
}

impl<'a> Weight<'a> {
    /// GGUF stores dimension 0 as the fastest-varying, so a matrix listed as
    /// `[in, out]` is `out` rows of `in` contiguous elements.
    pub fn bind(model: &'a Gguf, name: &str) -> Result<Self> {
        let info = model
            .tensor(name)
            .ok_or_else(|| bad(format!("missing tensor {name}")))?;
        let cols = *info.dims.first().unwrap_or(&0) as usize;
        let rows = info.dims.get(1).copied().unwrap_or(1) as usize;
        if cols == 0 {
            return Err(bad(format!("tensor {name} has no columns")));
        }

        let (block, size) = info.ty.layout().ok_or_else(|| {
            bad(format!(
                "tensor {name} has unsupported type {}",
                info.ty.name()
            ))
        })?;
        if cols as u64 % block != 0 {
            return Err(bad(format!(
                "tensor {name}: {cols} columns do not fill whole {} blocks",
                info.ty.name()
            )));
        }

        Ok(Self {
            rows,
            cols,
            ty: info.ty,
            row_bytes: (cols as u64 / block * size) as usize,
            bytes: model.tensor_bytes(info)?,
        })
    }

    /// Expand one row into `out`, which must be `cols` long.
    pub fn row(&self, index: usize, out: &mut [f32]) -> Result<()> {
        if index >= self.rows {
            return Err(bad(format!("row {index} of {} out of range", self.rows)));
        }
        let start = index * self.row_bytes;
        quant::dequantize(self.ty, &self.bytes[start..start + self.row_bytes], out)
    }

    /// `out = self * x`, split across `threads` workers.
    ///
    /// Rows are independent and each is reduced by the same code in the same
    /// order, so the split never changes the answer — the threaded result is
    /// bit-identical to the single-threaded one on whichever path is chosen.
    pub fn matvec_threaded(
        &self,
        x: &[f32],
        out: &mut [f32],
        threads: usize,
        workspace: &mut Workspace,
    ) -> Result<()> {
        // Quantizing the activation is O(cols) against the matvec's O(rows ×
        // cols), and every row shares the result, so it happens once here
        // rather than per row or per worker.
        let int8 = workspace.int8 && quant::idot::supports(self.ty);
        if int8 {
            workspace.quantized.fill(x);
        }

        // Below a few rows per worker the dispatch costs more than the work.
        if threads <= 1 || self.rows < threads * 4 {
            let Workspace {
                scratch, quantized, ..
            } = workspace;
            return self.rows_into(out, 0, x, quantized, int8, scratch);
        }

        workspace.ensure_pool(threads);
        let pool = workspace.pool.as_ref().unwrap();
        let ctx = MatvecJobCtx {
            weight: self,
            x,
            quantized: &workspace.quantized,
            int8,
            out_ptr: SyncMutPtr(out.as_mut_ptr()),
        };
        pool.dispatch(self.rows, &ctx, run_matvec_worker);
        Ok(())
    }

    /// Fill `out` with rows `first..first + out.len()`, by whichever path this
    /// type is fastest on.
    fn rows_into(
        &self,
        out: &mut [f32],
        first: usize,
        x: &[f32],
        quantized: &quant::activation::Quantized,
        int8: bool,
        scratch: &mut [f32],
    ) -> Result<()> {
        for (offset, slot) in out.iter_mut().enumerate() {
            let row = self.row_slice(first + offset);
            *slot = if int8 {
                quant::idot::integer(self.ty, row, quantized).expect("type was checked")
            } else {
                match quant::dot::fused(self.ty, row, x) {
                    Some(value) => value,
                    None => {
                        // The workspace is sized for the widest matrix in the
                        // model, so it has to be cut to this one's width —
                        // `dequantize` takes the element count from the slice.
                        let scratch = &mut scratch[..self.cols];
                        quant::dequantize(self.ty, row, scratch)?;
                        ops::dot(scratch, x)
                    }
                }
            };
        }
        Ok(())
    }

    /// `out = self * x`.
    ///
    /// Takes the fused kernel where one exists, otherwise expands each row
    /// through `scratch` and dots that. The two agree bit for bit.
    pub fn matvec(&self, x: &[f32], out: &mut [f32], scratch: &mut [f32]) -> Result<()> {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(out.len(), self.rows);
        let scratch = &mut scratch[..self.cols];

        for (index, slot) in out.iter_mut().enumerate() {
            let row = self.row_slice(index);
            *slot = match quant::dot::fused(self.ty, row, x) {
                Some(value) => value,
                None => {
                    quant::dequantize(self.ty, row, scratch)?;
                    ops::dot(scratch, x)
                }
            };
        }
        Ok(())
    }

    /// `out = self * x` across `batch` tokens simultaneously.
    ///
    /// `x` has shape `[batch, cols]`.
    /// `out` has shape `[batch, rows]`.
    /// `scratch` has length at least `cols`.
    ///
    /// Evaluates all `batch` tokens against each weight row while the row
    /// is hot in cache, amortizing weight decompression across the batch.
    pub fn matmul(
        &self,
        x: &[f32],
        out: &mut [f32],
        batch: usize,
        scratch: &mut [f32],
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        debug_assert_eq!(x.len(), batch * self.cols);
        debug_assert_eq!(out.len(), batch * self.rows);
        let scratch = &mut scratch[..self.cols];
        self.matmul_rows_into(x, &[], false, out, 0, self.rows, batch, scratch)
    }

    /// Parallel `matmul` distributing row ranges across `threads` persistent worker threads.
    pub fn matmul_threaded(
        &self,
        x: &[f32],
        out: &mut [f32],
        batch: usize,
        threads: usize,
        workspace: &mut Workspace,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        debug_assert_eq!(x.len(), batch * self.cols);
        debug_assert_eq!(out.len(), batch * self.rows);

        let int8 = workspace.int8 && quant::idot::supports(self.ty);
        if int8 {
            if workspace.batch_quantized.len() < batch {
                workspace.batch_quantized.resize_with(batch, || {
                    quant::activation::Quantized::with_capacity(self.cols)
                });
            }
            for b in 0..batch {
                workspace.batch_quantized[b].fill(&x[b * self.cols..(b + 1) * self.cols]);
            }
        }

        if threads <= 1 || self.rows < threads * 4 {
            let Workspace {
                scratch,
                batch_quantized,
                ..
            } = workspace;
            let scratch = &mut scratch[..self.cols];
            return self.matmul_rows_into(
                x,
                batch_quantized,
                int8,
                out,
                0,
                self.rows,
                batch,
                scratch,
            );
        }

        workspace.ensure_pool(threads);
        let pool = workspace.pool.as_ref().unwrap();
        let ctx = MatmulJobCtx {
            weight: self,
            x,
            batch_quantized: &workspace.batch_quantized,
            int8,
            batch,
            out_ptr: SyncMutPtr(out.as_mut_ptr()),
        };
        pool.dispatch(self.rows, &ctx, run_matmul_worker);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_rows_into(
        &self,
        x: &[f32],
        batch_quantized: &[quant::activation::Quantized],
        int8: bool,
        out: &mut [f32],
        first_row: usize,
        num_rows: usize,
        batch: usize,
        scratch: &mut [f32],
    ) -> Result<()> {
        let has_fused = quant::dot::supports(self.ty);

        for r_offset in 0..num_rows {
            let row_idx = first_row + r_offset;
            let row = self.row_slice(row_idx);
            if int8 {
                for (b, q_tok) in batch_quantized.iter().enumerate().take(batch) {
                    out[b * self.rows + row_idx] =
                        quant::idot::integer(self.ty, row, q_tok).expect("supported");
                }
            } else if has_fused {
                for b in 0..batch {
                    let x_tok = &x[b * self.cols..(b + 1) * self.cols];
                    out[b * self.rows + row_idx] =
                        quant::dot::fused(self.ty, row, x_tok).expect("supported");
                }
            } else {
                quant::dequantize(self.ty, row, scratch)?;
                for b in 0..batch {
                    let x_tok = &x[b * self.cols..(b + 1) * self.cols];
                    out[b * self.rows + row_idx] = ops::dot(scratch, x_tok);
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_row_range(
        &self,
        x: &[f32],
        batch_quantized: &[quant::activation::Quantized],
        int8: bool,
        out: SyncMutPtr,
        first_row: usize,
        num_rows: usize,
        batch: usize,
        scratch: &mut [f32],
    ) -> Result<()> {
        let has_fused = quant::dot::supports(self.ty);

        for r_offset in 0..num_rows {
            let row_idx = first_row + r_offset;
            let row = self.row_slice(row_idx);
            if int8 {
                for (b, q_tok) in batch_quantized.iter().enumerate().take(batch) {
                    let val = quant::idot::integer(self.ty, row, q_tok).expect("supported");
                    unsafe {
                        *out.0.add(b * self.rows + row_idx) = val;
                    }
                }
            } else if has_fused {
                for b in 0..batch {
                    let x_tok = &x[b * self.cols..(b + 1) * self.cols];
                    let val = quant::dot::fused(self.ty, row, x_tok).expect("supported");
                    unsafe {
                        *out.0.add(b * self.rows + row_idx) = val;
                    }
                }
            } else {
                quant::dequantize(self.ty, row, scratch)?;
                for b in 0..batch {
                    let x_tok = &x[b * self.cols..(b + 1) * self.cols];
                    let val = ops::dot(scratch, x_tok);
                    unsafe {
                        *out.0.add(b * self.rows + row_idx) = val;
                    }
                }
            }
        }
        Ok(())
    }

    /// `out = self * x`, with the activation already quantized to int8.
    ///
    /// Returns `false` and leaves `out` untouched when this type has no integer
    /// kernel. Approximate — see [`crate::quant::idot`].
    pub fn matvec_int8(&self, x: &quant::activation::Quantized, out: &mut [f32]) -> bool {
        if !quant::idot::supports(self.ty) {
            return false;
        }
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = quant::idot::integer(self.ty, self.row_slice(index), x)
                .expect("type was just checked");
        }
        true
    }

    /// The unfused path, always. Kept so the fused kernels have something to be
    /// diffed against — in tests, and in `examples/bench.rs`.
    pub fn matvec_reference(&self, x: &[f32], out: &mut [f32], scratch: &mut [f32]) -> Result<()> {
        debug_assert_eq!(scratch.len(), self.cols);
        for (index, slot) in out.iter_mut().enumerate() {
            quant::dequantize(self.ty, self.row_slice(index), scratch)?;
            *slot = ops::dot(scratch, x);
        }
        Ok(())
    }

    #[inline]
    pub fn row_slice(&self, index: usize) -> &'a [u8] {
        let start = index * self.row_bytes;
        &self.bytes[start..start + self.row_bytes]
    }

    #[inline]
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Construct a Weight directly from raw bytes, type, and dimensions.
    pub fn from_raw(bytes: &'a [u8], ty: GgmlType, rows: usize, cols: usize) -> Option<Self> {
        let (block, size) = ty.layout()?;
        let block = block as usize;
        let size = size as usize;
        if cols % block != 0 {
            return None;
        }
        let row_bytes = (cols / block) * size;
        if bytes.len() < rows * row_bytes {
            return None;
        }
        Some(Self {
            rows,
            cols,
            ty,
            row_bytes,
            bytes: &bytes[..rows * row_bytes],
        })
    }
}

pub struct Layer<'a> {
    pub attn_norm: Vec<f32>,
    pub post_attn_norm: Option<Vec<f32>>,
    pub wq: Weight<'a>,
    pub bq: Option<Vec<f32>>,
    pub lora_wq: Option<crate::lora::LoraPair<'a>>,
    pub wk: Weight<'a>,
    pub bk: Option<Vec<f32>>,
    pub lora_wk: Option<crate::lora::LoraPair<'a>>,
    pub wv: Weight<'a>,
    pub bv: Option<Vec<f32>>,
    pub lora_wv: Option<crate::lora::LoraPair<'a>>,
    pub wo: Weight<'a>,
    pub lora_wo: Option<crate::lora::LoraPair<'a>>,
    pub ffn_norm: Vec<f32>,
    pub post_ffn_norm: Option<Vec<f32>>,
    pub gate: Weight<'a>,
    pub lora_gate: Option<crate::lora::LoraPair<'a>>,
    pub up: Weight<'a>,
    pub lora_up: Option<crate::lora::LoraPair<'a>>,
    pub down: Weight<'a>,
    pub lora_down: Option<crate::lora::LoraPair<'a>>,
    pub ffn_gate_inp: Option<Weight<'a>>,
    pub expert_bias: Option<Vec<f32>>,
    pub moe_gate: Option<Vec<Weight<'a>>>,
    pub moe_up: Option<Vec<Weight<'a>>>,
    pub moe_down: Option<Vec<Weight<'a>>>,
    pub moe_gate_shexp: Option<Weight<'a>>,
    pub moe_up_shexp: Option<Weight<'a>>,
    pub moe_down_shexp: Option<Weight<'a>>,
    pub attn_q_a: Option<Weight<'a>>,
    pub attn_q_a_norm: Option<Vec<f32>>,
    pub attn_q_b: Option<Weight<'a>>,
    pub attn_kv_a: Option<Weight<'a>>,
    pub attn_kv_a_norm: Option<Vec<f32>>,
    pub attn_kv_b: Option<Weight<'a>>,
}

impl<'a> Layer<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn eval_ffn(
        &self,
        c: &Config,
        xb_batch: &[f32],
        n: usize,
        xb2_batch: &mut [f32],
        hb_batch: &mut [f32],
        hb2_batch: &mut [f32],
        threads: usize,
        workspace: &mut Workspace,
    ) -> Result<()> {
        if let (Some(router), Some(moe_gate), Some(moe_up), Some(moe_down)) = (
            &self.ffn_gate_inp,
            &self.moe_gate,
            &self.moe_up,
            &self.moe_down,
        ) {
            let num_experts = moe_gate.len();
            let top_k = c.expert_used_count.unwrap_or(2).min(num_experts);
            let mut router_logits = vec![0.0f32; num_experts];
            let mut expert_out = vec![0.0f32; c.embedding];
            let mut hb_tok = vec![0.0f32; c.ffn];
            let mut hb2_tok = vec![0.0f32; c.ffn];

            for b in 0..n {
                let xb_tok = &xb_batch[b * c.embedding..(b + 1) * c.embedding];
                let out_tok = &mut xb2_batch[b * c.embedding..(b + 1) * c.embedding];
                out_tok.fill(0.0);

                router.matvec_threaded(xb_tok, &mut router_logits, threads, workspace)?;

                if c.arch == "deepseek3" {
                    for val in router_logits.iter_mut() {
                        *val = ops::sigmoid(*val);
                    }
                } else {
                    ops::softmax(&mut router_logits);
                }

                let routed = ops::topk_moe_routing(
                    &router_logits,
                    self.expert_bias.as_deref(),
                    top_k,
                    c.expert_weights_scale,
                );

                for (exp_idx, weight_scale) in routed {
                    if weight_scale == 0.0 {
                        continue;
                    }
                    moe_gate[exp_idx].matvec_threaded(xb_tok, &mut hb_tok, threads, workspace)?;
                    moe_up[exp_idx].matvec_threaded(xb_tok, &mut hb2_tok, threads, workspace)?;

                    for (gate, up) in hb_tok.iter_mut().zip(hb2_tok.iter()) {
                        *gate = ops::silu(*gate) * up;
                    }

                    expert_out.fill(0.0);
                    moe_down[exp_idx].matvec_threaded(
                        &hb_tok,
                        &mut expert_out,
                        threads,
                        workspace,
                    )?;

                    for (slot, &val) in out_tok.iter_mut().zip(&expert_out) {
                        *slot += weight_scale * val;
                    }
                }

                if let (Some(shexp_gate), Some(shexp_up), Some(shexp_down)) = (
                    &self.moe_gate_shexp,
                    &self.moe_up_shexp,
                    &self.moe_down_shexp,
                ) {
                    let mut sh_hb = vec![0.0f32; shexp_gate.rows];
                    let mut sh_hb2 = vec![0.0f32; shexp_up.rows];
                    let mut sh_out = vec![0.0f32; c.embedding];
                    shexp_gate.matvec_threaded(xb_tok, &mut sh_hb, threads, workspace)?;
                    shexp_up.matvec_threaded(xb_tok, &mut sh_hb2, threads, workspace)?;
                    for (gate, up) in sh_hb.iter_mut().zip(&sh_hb2) {
                        *gate = ops::silu(*gate) * up;
                    }
                    shexp_down.matvec_threaded(&sh_hb, &mut sh_out, threads, workspace)?;
                    for (slot, &val) in out_tok.iter_mut().zip(&sh_out) {
                        *slot += val;
                    }
                }
            }
        } else if n == 1 {
            self.gate
                .matvec_threaded(xb_batch, hb_batch, threads, workspace)?;
            if let Some(l) = &self.lora_gate {
                l.apply(xb_batch, hb_batch)?;
            }
            self.up
                .matvec_threaded(xb_batch, hb2_batch, threads, workspace)?;
            if let Some(l) = &self.lora_up {
                l.apply(xb_batch, hb2_batch)?;
            }
            for (gate, up) in hb_batch.iter_mut().zip(hb2_batch.iter()) {
                *gate = ops::silu(*gate) * up;
            }
            self.down
                .matvec_threaded(hb_batch, xb2_batch, threads, workspace)?;
            if let Some(l) = &self.lora_down {
                l.apply(hb_batch, xb2_batch)?;
            }
        } else {
            self.gate
                .matmul_threaded(xb_batch, hb_batch, n, threads, workspace)?;
            if let Some(l) = &self.lora_gate {
                l.apply_batched_threaded(xb_batch, n, hb_batch, threads)?;
            }
            self.up
                .matmul_threaded(xb_batch, hb2_batch, n, threads, workspace)?;
            if let Some(l) = &self.lora_up {
                l.apply_batched_threaded(xb_batch, n, hb2_batch, threads)?;
            }
            for (gate, up) in hb_batch.iter_mut().zip(hb2_batch.iter()) {
                *gate = ops::silu(*gate) * up;
            }

            self.down
                .matmul_threaded(hb_batch, xb2_batch, n, threads, workspace)?;
            if let Some(l) = &self.lora_down {
                l.apply_batched_threaded(hb_batch, n, xb2_batch, threads)?;
            }
        }
        Ok(())
    }
    pub fn bind(gguf: &'a Gguf, config: &Config, index: usize) -> Result<Self> {
        Ok(Self {
            attn_norm: load_vector(gguf, &format!("blk.{index}.attn_norm.weight"))?,
            post_attn_norm: load_vector_optional(
                gguf,
                &format!("blk.{index}.post_attention_norm.weight"),
            )?,
            wq: Weight::bind(gguf, &format!("blk.{index}.attn_q.weight"))
                .or_else(|_| Weight::bind(gguf, "token_embd.weight"))?,
            bq: load_vector_optional(gguf, &format!("blk.{index}.attn_q.bias"))?,
            lora_wq: None,
            wk: Weight::bind(gguf, &format!("blk.{index}.attn_k.weight"))
                .or_else(|_| Weight::bind(gguf, "token_embd.weight"))?,
            bk: load_vector_optional(gguf, &format!("blk.{index}.attn_k.bias"))?,
            lora_wk: None,
            wv: Weight::bind(gguf, &format!("blk.{index}.attn_v.weight"))
                .or_else(|_| Weight::bind(gguf, "token_embd.weight"))?,
            bv: load_vector_optional(gguf, &format!("blk.{index}.attn_v.bias"))?,
            lora_wv: None,
            wo: Weight::bind(gguf, &format!("blk.{index}.attn_output.weight"))?,
            lora_wo: None,
            ffn_norm: load_vector(gguf, &format!("blk.{index}.ffn_norm.weight"))?,
            post_ffn_norm: load_vector_optional(
                gguf,
                &format!("blk.{index}.post_ffn_norm.weight"),
            )?,
            gate: Weight::bind(gguf, &format!("blk.{index}.ffn_gate.weight"))
                .or_else(|_| Weight::bind(gguf, "token_embd.weight"))?,
            lora_gate: None,
            up: Weight::bind(gguf, &format!("blk.{index}.ffn_up.weight"))
                .or_else(|_| Weight::bind(gguf, "token_embd.weight"))?,
            lora_up: None,
            down: Weight::bind(gguf, &format!("blk.{index}.ffn_down.weight"))
                .or_else(|_| Weight::bind(gguf, "token_embd.weight"))?,
            lora_down: None,
            ffn_gate_inp: Weight::bind(gguf, &format!("blk.{index}.ffn_gate_inp.weight"))
                .or_else(|_| Weight::bind(gguf, &format!("blk.{index}.ffn_gate_exps.weight")))
                .ok(),
            expert_bias: if let Some(b) =
                load_vector_optional(gguf, &format!("blk.{index}.ffn_gate_inp.bias"))?
            {
                Some(b)
            } else if let Some(b) =
                load_vector_optional(gguf, &format!("blk.{index}.exp_probs_b.weight"))?
            {
                Some(b)
            } else {
                load_vector_optional(gguf, &format!("blk.{index}.expert_bias.weight"))?
            },
            moe_gate: if let Some(n_exp) = config.expert_count {
                let mut gates = Vec::with_capacity(n_exp);
                for exp in 0..n_exp {
                    if let Ok(w) = Weight::bind(gguf, &format!("blk.{index}.ffn_gate.{exp}.weight"))
                    {
                        gates.push(w);
                    } else if let Ok(w) =
                        Weight::bind(gguf, &format!("blk.{index}.ffn_gate.weight"))
                    {
                        gates.push(w);
                    }
                }
                if gates.len() == n_exp {
                    Some(gates)
                } else {
                    None
                }
            } else {
                None
            },
            moe_up: if let Some(n_exp) = config.expert_count {
                let mut ups = Vec::with_capacity(n_exp);
                for exp in 0..n_exp {
                    if let Ok(w) = Weight::bind(gguf, &format!("blk.{index}.ffn_up.{exp}.weight")) {
                        ups.push(w);
                    } else if let Ok(w) = Weight::bind(gguf, &format!("blk.{index}.ffn_up.weight"))
                    {
                        ups.push(w);
                    }
                }
                if ups.len() == n_exp {
                    Some(ups)
                } else {
                    None
                }
            } else {
                None
            },
            moe_down: if let Some(n_exp) = config.expert_count {
                let mut downs = Vec::with_capacity(n_exp);
                for exp in 0..n_exp {
                    if let Ok(w) = Weight::bind(gguf, &format!("blk.{index}.ffn_down.{exp}.weight"))
                    {
                        downs.push(w);
                    } else if let Ok(w) =
                        Weight::bind(gguf, &format!("blk.{index}.ffn_down.weight"))
                    {
                        downs.push(w);
                    }
                }
                if downs.len() == n_exp {
                    Some(downs)
                } else {
                    None
                }
            } else {
                None
            },
            moe_gate_shexp: Weight::bind(gguf, &format!("blk.{index}.ffn_gate_shexp.weight"))
                .or_else(|_| Weight::bind(gguf, &format!("blk.{index}.ffn_gate.shexp.weight")))
                .ok(),
            moe_up_shexp: Weight::bind(gguf, &format!("blk.{index}.ffn_up_shexp.weight"))
                .or_else(|_| Weight::bind(gguf, &format!("blk.{index}.ffn_up.shexp.weight")))
                .ok(),
            moe_down_shexp: Weight::bind(gguf, &format!("blk.{index}.ffn_down_shexp.weight"))
                .or_else(|_| Weight::bind(gguf, &format!("blk.{index}.ffn_down.shexp.weight")))
                .ok(),
            attn_q_a: Weight::bind(gguf, &format!("blk.{index}.attn_q_a.weight")).ok(),
            attn_q_a_norm: load_vector_optional(
                gguf,
                &format!("blk.{index}.attn_q_a_norm.weight"),
            )?,
            attn_q_b: Weight::bind(gguf, &format!("blk.{index}.attn_q_b.weight")).ok(),
            attn_kv_a: Weight::bind(gguf, &format!("blk.{index}.attn_kv_a.weight")).ok(),
            attn_kv_a_norm: load_vector_optional(
                gguf,
                &format!("blk.{index}.attn_kv_a_norm.weight"),
            )?,
            attn_kv_b: Weight::bind(gguf, &format!("blk.{index}.attn_kv_b.weight")).ok(),
        })
    }
}

/// Multi-Token Prediction (MTP) module for DeepSeek-V3 architectures.
///
/// Projects the concatenated representation of the trunk hidden state and token embedding
/// through an MTP transformer layer to predict subsequent speculative tokens.
pub struct MtpModule<'a> {
    pub eh_proj: Weight<'a>,
    pub enorm: Vec<f32>,
    pub hnorm: Vec<f32>,
    pub layer: Layer<'a>,
    pub shared_head_norm: Vec<f32>,
    pub shared_head_head: Option<Weight<'a>>,
}

pub struct Model<'a> {
    pub config: Config,
    pub token_embd: Weight<'a>,
    pub layers: Vec<Layer<'a>>,
    pub mtp_module: Option<MtpModule<'a>>,
    pub output_norm: Vec<f32>,
    pub output: Weight<'a>,
    pub lora_output: Option<crate::lora::LoraPair<'a>>,
}

/// Read a small tensor straight into f32. Norm weights are one vector each, so
/// keeping them expanded costs nothing and saves a dequantize per layer.
fn load_vector(model: &Gguf, name: &str) -> Result<Vec<f32>> {
    let info = model
        .tensor(name)
        .ok_or_else(|| bad(format!("missing tensor {name}")))?;
    quant::dequantize_to_vec(info.ty, model.tensor_bytes(info)?, info.elements() as usize)
}

fn load_vector_optional(model: &Gguf, name: &str) -> Result<Option<Vec<f32>>> {
    let Some(info) = model.tensor(name) else {
        return Ok(None);
    };
    quant::dequantize_to_vec(info.ty, model.tensor_bytes(info)?, info.elements() as usize).map(Some)
}

impl<'a> Model<'a> {
    pub fn load(gguf: &'a Gguf) -> Result<Self> {
        let config = Config::from_gguf(gguf)?;
        let token_embd = Weight::bind(gguf, "token_embd.weight")?;

        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            layers.push(Layer::bind(gguf, &config, index)?);
        }

        let mtp_module = if let Ok(eh_proj) =
            Weight::bind(gguf, &format!("blk.{}.nextn.eh_proj.weight", config.layers))
        {
            let enorm = load_vector(gguf, &format!("blk.{}.nextn.enorm.weight", config.layers))?;
            let hnorm = load_vector(gguf, &format!("blk.{}.nextn.hnorm.weight", config.layers))?;
            let shared_head_norm = load_vector(
                gguf,
                &format!("blk.{}.nextn.shared_head_norm.weight", config.layers),
            )?;
            let shared_head_head = Weight::bind(
                gguf,
                &format!("blk.{}.nextn.shared_head_head.weight", config.layers),
            )
            .ok();
            let layer = Layer::bind(gguf, &config, config.layers)?;
            Some(MtpModule {
                eh_proj,
                enorm,
                hnorm,
                layer,
                shared_head_norm,
                shared_head_head,
            })
        } else {
            None
        };

        // Small models tie the output projection to the embedding table and
        // omit output.weight entirely.
        let output = match Weight::bind(gguf, "output.weight") {
            Ok(weight) => weight,
            Err(_) => Weight::bind(gguf, "token_embd.weight")?,
        };

        Ok(Self {
            config,
            token_embd,
            layers,
            mtp_module,
            output_norm: load_vector(gguf, "output_norm.weight")?,
            output,
            lora_output: None,
        })
    }

    /// Dynamically bind a parsed LoRA adapter to model layers.
    pub fn apply_lora(&mut self, lora: crate::lora::LoraAdapter<'a>) {
        for (name, pair) in lora.pairs {
            if name.starts_with("output") {
                self.lora_output = Some(pair);
                continue;
            }
            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some((idx_str, proj)) = rest.split_once('.') {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        if let Some(layer) = self.layers.get_mut(idx) {
                            if proj.starts_with("attn_q") {
                                layer.lora_wq = Some(pair);
                            } else if proj.starts_with("attn_k") {
                                layer.lora_wk = Some(pair);
                            } else if proj.starts_with("attn_v") {
                                layer.lora_wv = Some(pair);
                            } else if proj.starts_with("attn_output") || proj.starts_with("attn_wo")
                            {
                                layer.lora_wo = Some(pair);
                            } else if proj.starts_with("ffn_gate") {
                                layer.lora_gate = Some(pair);
                            } else if proj.starts_with("ffn_up") {
                                layer.lora_up = Some(pair);
                            } else if proj.starts_with("ffn_down") {
                                layer.lora_down = Some(pair);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Clear all attached LoRA adapter weights.
    pub fn clear_lora(&mut self) {
        self.lora_output = None;
        for layer in &mut self.layers {
            layer.lora_wq = None;
            layer.lora_wk = None;
            layer.lora_wv = None;
            layer.lora_wo = None;
            layer.lora_gate = None;
            layer.lora_up = None;
            layer.lora_down = None;
        }
    }

    /// One decode step: token in, logits out, KV cache extended by one.
    ///
    /// `position` is where this token sits in the sequence; it drives RoPE and
    /// decides how far back attention looks.
    pub fn forward(&self, state: &mut State, token: u32, position: usize) -> Result<()> {
        let c = &self.config;
        if position >= state.context {
            return Err(bad(format!(
                "position {position} is past the {} token cache",
                state.context
            )));
        }

        // Split borrows up front so the attention loop can read one buffer
        // while writing another.
        let State {
            x,
            xb,
            xb2,
            hb,
            hb2,
            q,
            k,
            v,
            logits,
            workspace,
            kv_cache,
            threads,
            ..
        } = state;

        self.token_embd.row(token as usize, x)?;
        if let Some(scale) = c.embedding_scale {
            for v in x.iter_mut() {
                *v *= scale;
            }
        }

        for (index, layer) in self.layers.iter().enumerate() {
            Self::forward_layer(
                c, layer, index, position, x, xb, xb2, hb, hb2, q, k, v, kv_cache, *threads,
                workspace,
            )?;
        }

        ops::rms_norm(x, &self.output_norm, c.eps, &mut xb[..c.embedding]);
        self.output
            .matvec_threaded(&xb[..c.embedding], logits, *threads, workspace)?;
        if let Some(l) = &self.lora_output {
            l.apply(&xb[..c.embedding], logits)?;
        }
        if let Some(cap) = c.final_logit_softcapping {
            for logit in logits.iter_mut() {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        Ok(())
    }

    /// Forward pass through the DeepSeek-V3 Multi-Token Prediction (MTP) module.
    ///
    /// Consumes the main model trunk's hidden state at position `position` and the
    /// predicted token `token` for `position + 1`, and predicts logits for `position + 2`.
    pub fn forward_mtp(
        &self,
        state: &mut State,
        hidden_state: &[f32],
        token: u32,
        position: usize,
        out_logits: &mut [f32],
    ) -> Result<()> {
        let mtp = self
            .mtp_module
            .as_ref()
            .ok_or_else(|| bad("model does not contain an MTP next-n prediction module"))?;
        let c = &self.config;

        if token >= c.vocab as u32 {
            return Err(bad(format!("token {token} is past vocabulary {}", c.vocab)));
        }
        if position >= state.context {
            return Err(bad(format!(
                "position {position} is past context window {}",
                state.context
            )));
        }
        if out_logits.len() != c.vocab {
            return Err(bad(format!(
                "out_logits size {} != vocab {}",
                out_logits.len(),
                c.vocab
            )));
        }

        let State {
            x,
            xb,
            xb2,
            hb,
            hb2,
            q,
            k,
            v,
            workspace,
            kv_cache,
            threads,
            ..
        } = state;

        // 1. Token embedding: E(w_{t+1})
        self.token_embd
            .row(token as usize, &mut xb[..c.embedding])?;
        if let Some(scale) = c.embedding_scale {
            for v in xb[..c.embedding].iter_mut() {
                *v *= scale;
            }
        }

        // 2. enorm on token embedding -> xb2
        ops::rms_norm(
            &xb[..c.embedding],
            &mtp.enorm,
            c.eps,
            &mut xb2[..c.embedding],
        );

        // 3. hnorm on trunk hidden state -> xb
        ops::rms_norm(hidden_state, &mtp.hnorm, c.eps, &mut xb[..c.embedding]);

        // 4. Concatenate [hnorm; enorm] (length 2 * embedding)
        let mut eh_concat = vec![0.0f32; 2 * c.embedding];
        eh_concat[..c.embedding].copy_from_slice(&xb[..c.embedding]);
        eh_concat[c.embedding..].copy_from_slice(&xb2[..c.embedding]);

        // 5. eh_proj: project concatenated vector to hidden dim into x
        mtp.eh_proj
            .matvec_threaded(&eh_concat, x, *threads, workspace)?;

        // 6. Forward MTP transformer layer (stored at layer index = c.layers in KV cache)
        Self::forward_layer(
            c, &mtp.layer, c.layers, position, x, xb, xb2, hb, hb2, q, k, v, kv_cache, *threads,
            workspace,
        )?;

        // 7. Output RMSNorm (shared_head_norm) -> xb
        ops::rms_norm(x, &mtp.shared_head_norm, c.eps, &mut xb[..c.embedding]);

        // 8. Logit projection (shared_head_head or self.output)
        if let Some(head) = &mtp.shared_head_head {
            head.matvec_threaded(&xb[..c.embedding], out_logits, *threads, workspace)?;
        } else {
            self.output
                .matvec_threaded(&xb[..c.embedding], out_logits, *threads, workspace)?;
            if let Some(l) = &self.lora_output {
                l.apply(&xb[..c.embedding], out_logits)?;
            }
        }

        if let Some(cap) = c.final_logit_softcapping {
            for logit in out_logits.iter_mut() {
                *logit = cap * (*logit / cap).tanh();
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer(
        c: &Config,
        layer: &Layer<'a>,
        index: usize,
        position: usize,
        x: &mut [f32],
        xb: &mut [f32],
        xb2: &mut [f32],
        hb: &mut [f32],
        hb2: &mut [f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        kv_cache: &mut KvCache,
        threads: usize,
        workspace: &mut Workspace,
    ) -> Result<()> {
        let (head_dim, kv_dim, group) = (c.head_dim, c.kv_dim(), c.group());
        let att_dim = c.heads * head_dim;
        let is_sliding = (index % 2 == 0) && c.sliding_window.is_some();
        let window = c.sliding_window.unwrap_or(usize::MAX);

        ops::rms_norm(x, &layer.attn_norm, c.eps, &mut xb[..c.embedding]);
        let is_mla = layer.attn_kv_a.is_some() && layer.attn_kv_b.is_some();
        if is_mla {
            let d_nope = c.qk_nope_head_dim.unwrap_or(head_dim);
            let d_rope = c.qk_rope_head_dim.unwrap_or(0);
            let v_hd = c.v_head_dim.unwrap_or(head_dim);
            let kv_lora_rank = c.kv_lora_rank.unwrap_or(512);

            if let (Some(attn_q_a), Some(attn_q_b)) = (&layer.attn_q_a, &layer.attn_q_b) {
                let mut q_lora = vec![0.0f32; attn_q_a.rows];
                attn_q_a.matvec_threaded(&xb[..c.embedding], &mut q_lora, threads, workspace)?;
                if let Some(norm) = &layer.attn_q_a_norm {
                    let mut normed = vec![0.0f32; q_lora.len()];
                    ops::rms_norm(&q_lora, norm, c.eps, &mut normed);
                    q_lora = normed;
                }
                attn_q_b.matvec_threaded(&q_lora, q, threads, workspace)?;
            } else {
                layer
                    .wq
                    .matvec_threaded(&xb[..c.embedding], q, threads, workspace)?;
                if let Some(bq) = &layer.bq {
                    for (slot, b) in q.iter_mut().zip(bq) {
                        *slot += b;
                    }
                }
            }

            for head in 0..c.heads {
                let head_slice = &mut q[head * head_dim..(head + 1) * head_dim];
                if d_rope > 0 && d_nope + d_rope <= head_slice.len() {
                    let rope_slice = &mut head_slice[d_nope..d_nope + d_rope];
                    ops::rope_scaled(
                        rope_slice,
                        position,
                        c.rope_base,
                        c.rope_dim.min(d_rope),
                        c.rope_scaling.as_ref(),
                    );
                }
            }

            let attn_kv_a = layer.attn_kv_a.as_ref().unwrap();
            let attn_kv_b = layer.attn_kv_b.as_ref().unwrap();
            let mut kv_a_out = vec![0.0f32; attn_kv_a.rows];
            attn_kv_a.matvec_threaded(&xb[..c.embedding], &mut kv_a_out, threads, workspace)?;

            let c_kv = &kv_a_out[..kv_lora_rank];
            let mut c_kv_normed = vec![0.0f32; kv_lora_rank];
            if let Some(norm) = &layer.attn_kv_a_norm {
                ops::rms_norm(c_kv, norm, c.eps, &mut c_kv_normed);
            } else {
                c_kv_normed.copy_from_slice(c_kv);
            }

            let mut k_rope = if d_rope > 0 && kv_a_out.len() >= kv_lora_rank + d_rope {
                kv_a_out[kv_lora_rank..kv_lora_rank + d_rope].to_vec()
            } else {
                Vec::new()
            };
            if !k_rope.is_empty() {
                ops::rope_scaled(
                    &mut k_rope,
                    position,
                    c.rope_base,
                    c.rope_dim.min(d_rope),
                    c.rope_scaling.as_ref(),
                );
            }

            let mut kv_b_out = vec![0.0f32; attn_kv_b.rows];
            attn_kv_b.matvec_threaded(&c_kv_normed, &mut kv_b_out, threads, workspace)?;

            let head_kv_b_dim = d_nope + v_hd;
            for head in 0..c.heads {
                let kv_b_head = &kv_b_out[head * head_kv_b_dim..(head + 1) * head_kv_b_dim];
                let k_nope_part = &kv_b_head[..d_nope];
                let v_part = &kv_b_head[d_nope..d_nope + v_hd];

                let k_dest = &mut k[head * head_dim..(head + 1) * head_dim];
                k_dest[..d_nope].copy_from_slice(k_nope_part);
                if !k_rope.is_empty() {
                    k_dest[d_nope..d_nope + d_rope].copy_from_slice(&k_rope);
                }

                let v_dest = &mut v[head * v_hd..(head + 1) * v_hd];
                v_dest.copy_from_slice(v_part);
            }

            kv_cache.store(index, position, k, v);

            let scale = 1.0 / (head_dim as f32).sqrt();
            let v_dim = c.heads * v_hd;
            let k_dim = c.heads * head_dim;

            for head in 0..c.heads {
                let query = &q[head * head_dim..(head + 1) * head_dim];
                let out = &mut xb[head * v_hd..(head + 1) * v_hd];
                kv_cache.attention(
                    index,
                    query,
                    out,
                    ops::AttentionParams {
                        row_width: k_dim,
                        offset: head * head_dim,
                        len: position + 1,
                        scale,
                        logit_cap: c.attn_logit_softcapping,
                        window: is_sliding.then_some(window),
                        position,
                        tile_size: 64,
                        v_row_width: Some(v_dim),
                        v_offset: Some(head * v_hd),
                    },
                );
            }

            layer
                .wo
                .matvec_threaded(&xb[..v_dim], &mut xb2[..c.embedding], threads, workspace)?;
            if let Some(l) = &layer.lora_wo {
                l.apply(&xb[..v_dim], &mut xb2[..c.embedding])?;
            }
        } else {
            layer
                .wq
                .matvec_threaded(&xb[..c.embedding], q, threads, workspace)?;
            if let Some(bq) = &layer.bq {
                for (slot, b) in q.iter_mut().zip(bq) {
                    *slot += b;
                }
            }
            if let Some(l) = &layer.lora_wq {
                l.apply(&xb[..c.embedding], q)?;
            }

            layer
                .wk
                .matvec_threaded(&xb[..c.embedding], k, threads, workspace)?;
            if let Some(bk) = &layer.bk {
                for (slot, b) in k.iter_mut().zip(bk) {
                    *slot += b;
                }
            }
            if let Some(l) = &layer.lora_wk {
                l.apply(&xb[..c.embedding], k)?;
            }

            layer
                .wv
                .matvec_threaded(&xb[..c.embedding], v, threads, workspace)?;
            if let Some(bv) = &layer.bv {
                for (slot, b) in v.iter_mut().zip(bv) {
                    *slot += b;
                }
            }
            if let Some(l) = &layer.lora_wv {
                l.apply(&xb[..c.embedding], v)?;
            }

            if let Some(sections) = c.mrope_sections {
                for head in 0..c.heads {
                    ops::mrope(
                        &mut q[head * head_dim..(head + 1) * head_dim],
                        [position, position, position, 0],
                        sections,
                        c.rope_base,
                        ops::RoPEType::Neox,
                    );
                }
                for head in 0..c.kv_heads {
                    ops::mrope(
                        &mut k[head * head_dim..(head + 1) * head_dim],
                        [position, position, position, 0],
                        sections,
                        c.rope_base,
                        ops::RoPEType::Neox,
                    );
                }
            } else {
                for head in 0..c.heads {
                    ops::rope_scaled(
                        &mut q[head * head_dim..(head + 1) * head_dim],
                        position,
                        c.rope_base,
                        c.rope_dim,
                        c.rope_scaling.as_ref(),
                    );
                }
                for head in 0..c.kv_heads {
                    ops::rope_scaled(
                        &mut k[head * head_dim..(head + 1) * head_dim],
                        position,
                        c.rope_base,
                        c.rope_dim,
                        c.rope_scaling.as_ref(),
                    );
                }
            }

            kv_cache.store(index, position, k, v);

            let scale = 1.0 / (head_dim as f32).sqrt();
            for head in 0..c.heads {
                let kv_head = head / group;
                let query = &q[head * head_dim..(head + 1) * head_dim];

                let out = &mut xb[head * head_dim..(head + 1) * head_dim];
                kv_cache.attention(
                    index,
                    query,
                    out,
                    ops::AttentionParams {
                        row_width: kv_dim,
                        offset: kv_head * head_dim,
                        len: position + 1,
                        scale,
                        logit_cap: c.attn_logit_softcapping,
                        window: is_sliding.then_some(window),
                        position,
                        tile_size: 64,
                        v_row_width: None,
                        v_offset: None,
                    },
                );
            }

            layer.wo.matvec_threaded(
                &xb[..att_dim],
                &mut xb2[..c.embedding],
                threads,
                workspace,
            )?;
            if let Some(l) = &layer.lora_wo {
                l.apply(&xb[..att_dim], &mut xb2[..c.embedding])?;
            }
        }

        if let Some(post_attn_norm) = &layer.post_attn_norm {
            ops::rms_norm(
                &xb2[..c.embedding],
                post_attn_norm,
                c.eps,
                &mut xb[..c.embedding],
            );
            for (slot, delta) in x.iter_mut().zip(&xb[..c.embedding]) {
                *slot += delta;
            }
        } else {
            for (slot, delta) in x.iter_mut().zip(&xb2[..c.embedding]) {
                *slot += delta;
            }
        }

        ops::rms_norm(x, &layer.ffn_norm, c.eps, &mut xb[..c.embedding]);
        layer.eval_ffn(
            c,
            &xb[..c.embedding],
            1,
            &mut xb2[..c.embedding],
            hb,
            hb2,
            threads,
            workspace,
        )?;
        if let Some(post_ffn_norm) = &layer.post_ffn_norm {
            ops::rms_norm(
                &xb2[..c.embedding],
                post_ffn_norm,
                c.eps,
                &mut xb[..c.embedding],
            );
            for (slot, delta) in x.iter_mut().zip(&xb[..c.embedding]) {
                *slot += delta;
            }
        } else {
            for (slot, delta) in x.iter_mut().zip(&xb2[..c.embedding]) {
                *slot += delta;
            }
        }

        Ok(())
    }

    /// Process a batch of prompt tokens in one pass.
    ///
    /// Evaluates each layer across the batch tokens simultaneously to turn
    /// memory-bound weight reads into compute-bound matrix operations.
    ///
    /// Populates key/value caches for `start_pos..start_pos + tokens.len()`,
    /// leaving `state.logits` holding the logits for the final token.
    pub fn prefill(&self, state: &mut State, tokens: &[u32], start_pos: usize) -> Result<()> {
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }
        if n == 1 {
            return self.forward(state, tokens[0], start_pos);
        }

        let c = &self.config;
        let mut x_batch = vec![0.0f32; n * c.embedding];
        for (b, &tok) in tokens.iter().enumerate() {
            let row = &mut x_batch[b * c.embedding..(b + 1) * c.embedding];
            self.token_embd.row(tok as usize, row)?;
        }
        if let Some(scale) = c.embedding_scale {
            for v in x_batch.iter_mut() {
                *v *= scale;
            }
        }

        self.prefill_embeddings(state, &x_batch, start_pos)
    }

    /// Ingests a multimodal sequence composed of text prefix tokens, vision/audio embeddings,
    /// and text suffix tokens in a single unified prefill pass.
    pub fn prefill_multimodal(
        &self,
        state: &mut State,
        text_prefix: &[u32],
        multimodal_embeddings: &[f32],
        text_suffix: &[u32],
        start_pos: usize,
    ) -> Result<()> {
        let c = &self.config;
        if multimodal_embeddings.len() % c.embedding != 0 {
            return Err(bad(format!(
                "multimodal embeddings length {} is not a multiple of embedding dim {}",
                multimodal_embeddings.len(),
                c.embedding
            )));
        }
        let mm_tokens = multimodal_embeddings.len() / c.embedding;
        let total_tokens = text_prefix.len() + mm_tokens + text_suffix.len();
        if total_tokens == 0 {
            return Ok(());
        }

        let mut x_batch = vec![0.0f32; total_tokens * c.embedding];
        let mut offset = 0;

        for &tok in text_prefix {
            let row = &mut x_batch[offset * c.embedding..(offset + 1) * c.embedding];
            self.token_embd.row(tok as usize, row)?;
            if let Some(scale) = c.embedding_scale {
                for v in row.iter_mut() {
                    *v *= scale;
                }
            }
            offset += 1;
        }

        if !multimodal_embeddings.is_empty() {
            let dest = &mut x_batch[offset * c.embedding..(offset + mm_tokens) * c.embedding];
            dest.copy_from_slice(multimodal_embeddings);
            offset += mm_tokens;
        }

        for &tok in text_suffix {
            let row = &mut x_batch[offset * c.embedding..(offset + 1) * c.embedding];
            self.token_embd.row(tok as usize, row)?;
            if let Some(scale) = c.embedding_scale {
                for v in row.iter_mut() {
                    *v *= scale;
                }
            }
            offset += 1;
        }

        self.prefill_embeddings(state, &x_batch, start_pos)
    }

    /// Process a batch of continuous embedding vectors directly in one pass.
    ///
    /// Accepts a slice of concatenated token embeddings `[n * embedding]`,
    /// populates key/value caches for `start_pos..start_pos + n`,
    /// leaving `state.logits` holding the logits for the final token.
    pub fn prefill_embeddings(
        &self,
        state: &mut State,
        embeddings: &[f32],
        start_pos: usize,
    ) -> Result<()> {
        let c = &self.config;
        if embeddings.is_empty() {
            return Ok(());
        }
        if embeddings.len() % c.embedding != 0 {
            return Err(bad(format!(
                "embeddings length {} is not a multiple of embedding dim {}",
                embeddings.len(),
                c.embedding
            )));
        }
        let n = embeddings.len() / c.embedding;
        let end_pos = start_pos
            .checked_add(n)
            .ok_or_else(|| bad("prefill range overflows"))?;
        if end_pos > state.context {
            return Err(bad(format!(
                "prefill end position {end_pos} exceeds {} token cache",
                state.context
            )));
        }

        let (head_dim, kv_dim, group) = (c.head_dim, c.kv_dim(), c.group());
        let att_dim = c.heads * head_dim;

        let is_deepseek_mla = c.arch.starts_with("deepseek") && c.kv_lora_rank.is_some();
        let eff_k_dim = if is_deepseek_mla {
            c.heads * head_dim
        } else {
            kv_dim
        };
        let v_hd = c.v_head_dim.unwrap_or(head_dim);
        let eff_v_dim = if let Some(v_h) = c.v_head_dim {
            c.heads * v_h
        } else {
            eff_k_dim
        };

        let mut x_batch = embeddings.to_vec();
        let mut xb_batch = vec![0.0f32; n * c.embedding];
        let mut xb2_batch = vec![0.0f32; n * c.embedding];
        let mut att_batch = vec![0.0f32; n * eff_v_dim.max(att_dim)];
        let mut q_batch = vec![0.0f32; n * att_dim];
        let mut k_batch = vec![0.0f32; n * eff_k_dim];
        let mut v_batch = vec![0.0f32; n * eff_v_dim];
        let mut hb_batch = vec![0.0f32; n * c.ffn];
        let mut hb2_batch = vec![0.0f32; n * c.ffn];

        let State {
            logits,
            workspace,
            kv_cache,
            threads,
            ..
        } = state;

        let is_sliding_layer = |idx: usize| (idx % 2 == 0) && c.sliding_window.is_some();
        let window = c.sliding_window.unwrap_or(usize::MAX);

        for (index, layer) in self.layers.iter().enumerate() {
            for b in 0..n {
                let x_tok = &x_batch[b * c.embedding..(b + 1) * c.embedding];
                let xb_tok = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                ops::rms_norm(x_tok, &layer.attn_norm, c.eps, xb_tok);
            }

            let is_mla = layer.attn_kv_a.is_some() && layer.attn_kv_b.is_some();
            if is_mla {
                let d_nope = c.qk_nope_head_dim.unwrap_or(head_dim);
                let d_rope = c.qk_rope_head_dim.unwrap_or(0);
                let kv_lora_rank = c.kv_lora_rank.unwrap_or(512);

                if let (Some(attn_q_a), Some(attn_q_b)) = (&layer.attn_q_a, &layer.attn_q_b) {
                    let mut q_lora_batch = vec![0.0f32; n * attn_q_a.rows];
                    attn_q_a.matmul_threaded(
                        &xb_batch,
                        &mut q_lora_batch,
                        n,
                        *threads,
                        workspace,
                    )?;
                    if let Some(norm) = &layer.attn_q_a_norm {
                        for b in 0..n {
                            let in_tok = &q_lora_batch[b * attn_q_a.rows..(b + 1) * attn_q_a.rows];
                            let mut normed = vec![0.0f32; attn_q_a.rows];
                            ops::rms_norm(in_tok, norm, c.eps, &mut normed);
                            q_lora_batch[b * attn_q_a.rows..(b + 1) * attn_q_a.rows]
                                .copy_from_slice(&normed);
                        }
                    }
                    attn_q_b.matmul_threaded(
                        &q_lora_batch,
                        &mut q_batch,
                        n,
                        *threads,
                        workspace,
                    )?;
                } else {
                    layer
                        .wq
                        .matmul_threaded(&xb_batch, &mut q_batch, n, *threads, workspace)?;
                    if let Some(bq) = &layer.bq {
                        for b in 0..n {
                            let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                            for (slot, val) in q_tok.iter_mut().zip(bq) {
                                *slot += val;
                            }
                        }
                    }
                }

                // RoPE on Q
                for b in 0..n {
                    let pos = start_pos + b;
                    let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                    for head in 0..c.heads {
                        let head_slice = &mut q_tok[head * head_dim..(head + 1) * head_dim];
                        if d_rope > 0 && d_nope + d_rope <= head_slice.len() {
                            let rope_slice = &mut head_slice[d_nope..d_nope + d_rope];
                            ops::rope_scaled(
                                rope_slice,
                                pos,
                                c.rope_base,
                                c.rope_dim.min(d_rope),
                                c.rope_scaling.as_ref(),
                            );
                        }
                    }
                }

                // KV down projection
                let attn_kv_a = layer.attn_kv_a.as_ref().unwrap();
                let attn_kv_b = layer.attn_kv_b.as_ref().unwrap();
                let mut kv_a_batch = vec![0.0f32; n * attn_kv_a.rows];
                attn_kv_a.matmul_threaded(&xb_batch, &mut kv_a_batch, n, *threads, workspace)?;

                let mut c_kv_normed_batch = vec![0.0f32; n * kv_lora_rank];
                let mut k_rope_batch = vec![0.0f32; n * d_rope];

                for b in 0..n {
                    let pos = start_pos + b;
                    let kv_a_tok = &kv_a_batch[b * attn_kv_a.rows..(b + 1) * attn_kv_a.rows];
                    let c_kv = &kv_a_tok[..kv_lora_rank];
                    let c_normed_dest =
                        &mut c_kv_normed_batch[b * kv_lora_rank..(b + 1) * kv_lora_rank];
                    if let Some(norm) = &layer.attn_kv_a_norm {
                        ops::rms_norm(c_kv, norm, c.eps, c_normed_dest);
                    } else {
                        c_normed_dest.copy_from_slice(c_kv);
                    }

                    if d_rope > 0 && kv_a_tok.len() >= kv_lora_rank + d_rope {
                        let k_rope_dest = &mut k_rope_batch[b * d_rope..(b + 1) * d_rope];
                        k_rope_dest.copy_from_slice(&kv_a_tok[kv_lora_rank..kv_lora_rank + d_rope]);
                        ops::rope_scaled(
                            k_rope_dest,
                            pos,
                            c.rope_base,
                            c.rope_dim.min(d_rope),
                            c.rope_scaling.as_ref(),
                        );
                    }
                }

                let mut kv_b_batch = vec![0.0f32; n * attn_kv_b.rows];
                attn_kv_b.matmul_threaded(
                    &c_kv_normed_batch,
                    &mut kv_b_batch,
                    n,
                    *threads,
                    workspace,
                )?;

                let head_kv_b_dim = d_nope + v_hd;
                for b in 0..n {
                    let pos = start_pos + b;
                    let k_tok = &mut k_batch[b * eff_k_dim..(b + 1) * eff_k_dim];
                    let v_tok = &mut v_batch[b * eff_v_dim..(b + 1) * eff_v_dim];
                    let kv_b_tok = &kv_b_batch[b * attn_kv_b.rows..(b + 1) * attn_kv_b.rows];
                    let k_rope_tok = if d_rope > 0 {
                        &k_rope_batch[b * d_rope..(b + 1) * d_rope]
                    } else {
                        &[]
                    };

                    for head in 0..c.heads {
                        let kv_b_head = &kv_b_tok[head * head_kv_b_dim..(head + 1) * head_kv_b_dim];
                        let k_nope_part = &kv_b_head[..d_nope];
                        let v_part = &kv_b_head[d_nope..d_nope + v_hd];

                        let k_dest = &mut k_tok[head * head_dim..(head + 1) * head_dim];
                        k_dest[..d_nope].copy_from_slice(k_nope_part);
                        if !k_rope_tok.is_empty() {
                            k_dest[d_nope..d_nope + d_rope].copy_from_slice(k_rope_tok);
                        }

                        let v_dest = &mut v_tok[head * v_hd..(head + 1) * v_hd];
                        v_dest.copy_from_slice(v_part);
                    }

                    kv_cache.store(index, pos, k_tok, v_tok);
                }

                let scale = 1.0 / (head_dim as f32).sqrt();
                let is_sliding = is_sliding_layer(index);
                for b in 0..n {
                    let pos = start_pos + b;
                    let q_tok = &q_batch[b * att_dim..(b + 1) * att_dim];
                    let att_tok = &mut att_batch[b * eff_v_dim..(b + 1) * eff_v_dim];

                    for head in 0..c.heads {
                        let query = &q_tok[head * head_dim..(head + 1) * head_dim];
                        let out = &mut att_tok[head * v_hd..(head + 1) * v_hd];
                        kv_cache.attention(
                            index,
                            query,
                            out,
                            ops::AttentionParams {
                                row_width: eff_k_dim,
                                offset: head * head_dim,
                                len: pos + 1,
                                scale,
                                logit_cap: c.attn_logit_softcapping,
                                window: is_sliding.then_some(window),
                                position: pos,
                                tile_size: 64,
                                v_row_width: Some(eff_v_dim),
                                v_offset: Some(head * v_hd),
                            },
                        );
                    }
                }

                layer.wo.matmul_threaded(
                    &att_batch[..n * eff_v_dim],
                    &mut xb2_batch,
                    n,
                    *threads,
                    workspace,
                )?;
                if let Some(l) = &layer.lora_wo {
                    l.apply_batched_threaded(
                        &att_batch[..n * eff_v_dim],
                        n,
                        &mut xb2_batch,
                        *threads,
                    )?;
                }
            } else {
                layer
                    .wq
                    .matmul_threaded(&xb_batch, &mut q_batch, n, *threads, workspace)?;
                if let Some(bq) = &layer.bq {
                    for b in 0..n {
                        let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                        for (slot, val) in q_tok.iter_mut().zip(bq) {
                            *slot += val;
                        }
                    }
                }
                if let Some(l) = &layer.lora_wq {
                    l.apply_batched_threaded(&xb_batch, n, &mut q_batch, *threads)?;
                }

                layer
                    .wk
                    .matmul_threaded(&xb_batch, &mut k_batch, n, *threads, workspace)?;
                if let Some(bk) = &layer.bk {
                    for b in 0..n {
                        let k_tok = &mut k_batch[b * kv_dim..(b + 1) * kv_dim];
                        for (slot, val) in k_tok.iter_mut().zip(bk) {
                            *slot += val;
                        }
                    }
                }
                if let Some(l) = &layer.lora_wk {
                    l.apply_batched_threaded(&xb_batch, n, &mut k_batch, *threads)?;
                }

                layer
                    .wv
                    .matmul_threaded(&xb_batch, &mut v_batch, n, *threads, workspace)?;
                if let Some(bv) = &layer.bv {
                    for b in 0..n {
                        let v_tok = &mut v_batch[b * kv_dim..(b + 1) * kv_dim];
                        for (slot, val) in v_tok.iter_mut().zip(bv) {
                            *slot += val;
                        }
                    }
                }
                if let Some(l) = &layer.lora_wv {
                    l.apply_batched_threaded(&xb_batch, n, &mut v_batch, *threads)?;
                }

                for b in 0..n {
                    let pos = start_pos + b;
                    let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                    let k_tok = &mut k_batch[b * kv_dim..(b + 1) * kv_dim];
                    let v_tok = &v_batch[b * kv_dim..(b + 1) * kv_dim];

                    if let Some(sections) = c.mrope_sections {
                        for head in 0..c.heads {
                            ops::mrope(
                                &mut q_tok[head * head_dim..(head + 1) * head_dim],
                                [pos, pos, pos, pos],
                                sections,
                                c.rope_base,
                                ops::RoPEType::Neox,
                            );
                        }
                        for head in 0..c.kv_heads {
                            ops::mrope(
                                &mut k_tok[head * head_dim..(head + 1) * head_dim],
                                [pos, pos, pos, pos],
                                sections,
                                c.rope_base,
                                ops::RoPEType::Neox,
                            );
                        }
                    } else {
                        for head in 0..c.heads {
                            ops::rope_scaled(
                                &mut q_tok[head * head_dim..(head + 1) * head_dim],
                                pos,
                                c.rope_base,
                                c.rope_dim,
                                c.rope_scaling.as_ref(),
                            );
                        }
                        for head in 0..c.kv_heads {
                            ops::rope_scaled(
                                &mut k_tok[head * head_dim..(head + 1) * head_dim],
                                pos,
                                c.rope_base,
                                c.rope_dim,
                                c.rope_scaling.as_ref(),
                            );
                        }
                    }

                    kv_cache.store(index, pos, k_tok, v_tok);
                }

                let scale = 1.0 / (head_dim as f32).sqrt();
                let is_sliding = is_sliding_layer(index);
                for b in 0..n {
                    let pos = start_pos + b;
                    let q_tok = &q_batch[b * att_dim..(b + 1) * att_dim];
                    let att_tok = &mut att_batch[b * att_dim..(b + 1) * att_dim];

                    for head in 0..c.heads {
                        let kv_head = head / group;
                        let query = &q_tok[head * head_dim..(head + 1) * head_dim];

                        let out = &mut att_tok[head * head_dim..(head + 1) * head_dim];
                        kv_cache.attention(
                            index,
                            query,
                            out,
                            ops::AttentionParams {
                                row_width: kv_dim,
                                offset: kv_head * head_dim,
                                len: pos + 1,
                                scale,
                                logit_cap: c.attn_logit_softcapping,
                                window: is_sliding.then_some(window),
                                position: pos,
                                tile_size: 64,
                                v_row_width: None,
                                v_offset: None,
                            },
                        );
                    }
                }

                layer.wo.matmul_threaded(
                    &att_batch[..n * att_dim],
                    &mut xb2_batch,
                    n,
                    *threads,
                    workspace,
                )?;
                if let Some(l) = &layer.lora_wo {
                    l.apply_batched_threaded(
                        &att_batch[..n * att_dim],
                        n,
                        &mut xb2_batch,
                        *threads,
                    )?;
                }
            }
            if let Some(post_attn_norm) = &layer.post_attn_norm {
                for b in 0..n {
                    let tok_in = &xb2_batch[b * c.embedding..(b + 1) * c.embedding];
                    let tok_out = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                    ops::rms_norm(tok_in, post_attn_norm, c.eps, tok_out);
                }
                for (slot, delta) in x_batch.iter_mut().zip(xb_batch.iter()) {
                    *slot += delta;
                }
            } else {
                for (slot, delta) in x_batch.iter_mut().zip(xb2_batch.iter()) {
                    *slot += delta;
                }
            }

            for b in 0..n {
                let x_tok = &x_batch[b * c.embedding..(b + 1) * c.embedding];
                let xb_tok = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                ops::rms_norm(x_tok, &layer.ffn_norm, c.eps, xb_tok);
            }

            layer.eval_ffn(
                c,
                &xb_batch,
                n,
                &mut xb2_batch,
                &mut hb_batch,
                &mut hb2_batch,
                *threads,
                workspace,
            )?;
            if let Some(post_ffn_norm) = &layer.post_ffn_norm {
                for b in 0..n {
                    let tok_in = &xb2_batch[b * c.embedding..(b + 1) * c.embedding];
                    let tok_out = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                    ops::rms_norm(tok_in, post_ffn_norm, c.eps, tok_out);
                }
                for (slot, delta) in x_batch.iter_mut().zip(xb_batch.iter()) {
                    *slot += delta;
                }
            } else {
                for (slot, delta) in x_batch.iter_mut().zip(xb2_batch.iter()) {
                    *slot += delta;
                }
            }
        }

        let last_b = n - 1;
        let last_x = &x_batch[last_b * c.embedding..(last_b + 1) * c.embedding];
        let last_xb = &mut xb_batch[..c.embedding];
        ops::rms_norm(last_x, &self.output_norm, c.eps, last_xb);
        self.output
            .matvec_threaded(last_xb, logits, *threads, workspace)?;
        if let Some(l) = &self.lora_output {
            l.apply(last_xb, logits)?;
        }
        if let Some(cap) = c.final_logit_softcapping {
            for logit in logits.iter_mut() {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        Ok(())
    }

    /// Compute contextualized hidden states for a sequence of tokens without computing final vocabulary logits.
    ///
    /// Stores the normalized hidden representations of shape `[tokens.len() * embedding]` into `out_states`.
    pub fn embed_hidden_states(
        &self,
        state: &mut State,
        tokens: &[u32],
        out_states: &mut [f32],
    ) -> Result<()> {
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }
        let c = &self.config;
        if out_states.len() != n * c.embedding {
            return Err(bad(format!(
                "out_states length {} does not match n {} * embedding {}",
                out_states.len(),
                n,
                c.embedding
            )));
        }

        let (head_dim, kv_dim, group) = (c.head_dim, c.kv_dim(), c.group());
        let att_dim = c.heads * head_dim;

        let mut x_batch = vec![0.0f32; n * c.embedding];
        for (b, &tok) in tokens.iter().enumerate() {
            let row = &mut x_batch[b * c.embedding..(b + 1) * c.embedding];
            self.token_embd.row(tok as usize, row)?;
        }
        if let Some(scale) = c.embedding_scale {
            for v in x_batch.iter_mut() {
                *v *= scale;
            }
        }

        let mut xb_batch = vec![0.0f32; n * c.embedding];
        let mut xb2_batch = vec![0.0f32; n * c.embedding];
        let mut att_batch = vec![0.0f32; n * att_dim];
        let mut q_batch = vec![0.0f32; n * att_dim];
        let mut k_batch = vec![0.0f32; n * kv_dim];
        let mut v_batch = vec![0.0f32; n * kv_dim];
        let mut hb_batch = vec![0.0f32; n * c.ffn];
        let mut hb2_batch = vec![0.0f32; n * c.ffn];

        let State {
            workspace,
            kv_cache,
            threads,
            ..
        } = state;

        let is_sliding_layer = |idx: usize| (idx % 2 == 0) && c.sliding_window.is_some();
        let window = c.sliding_window.unwrap_or(usize::MAX);

        for (index, layer) in self.layers.iter().enumerate() {
            for b in 0..n {
                let x_tok = &x_batch[b * c.embedding..(b + 1) * c.embedding];
                let xb_tok = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                ops::rms_norm(x_tok, &layer.attn_norm, c.eps, xb_tok);
            }

            layer
                .wq
                .matmul_threaded(&xb_batch, &mut q_batch, n, *threads, workspace)?;
            if let Some(bq) = &layer.bq {
                for b in 0..n {
                    let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                    for (slot, val) in q_tok.iter_mut().zip(bq) {
                        *slot += val;
                    }
                }
            }
            if let Some(l) = &layer.lora_wq {
                l.apply_batched_threaded(&xb_batch, n, &mut q_batch, *threads)?;
            }

            layer
                .wk
                .matmul_threaded(&xb_batch, &mut k_batch, n, *threads, workspace)?;
            if let Some(bk) = &layer.bk {
                for b in 0..n {
                    let k_tok = &mut k_batch[b * kv_dim..(b + 1) * kv_dim];
                    for (slot, val) in k_tok.iter_mut().zip(bk) {
                        *slot += val;
                    }
                }
            }
            if let Some(l) = &layer.lora_wk {
                l.apply_batched_threaded(&xb_batch, n, &mut k_batch, *threads)?;
            }

            layer
                .wv
                .matmul_threaded(&xb_batch, &mut v_batch, n, *threads, workspace)?;
            if let Some(bv) = &layer.bv {
                for b in 0..n {
                    let v_tok = &mut v_batch[b * kv_dim..(b + 1) * kv_dim];
                    for (slot, val) in v_tok.iter_mut().zip(bv) {
                        *slot += val;
                    }
                }
            }
            if let Some(l) = &layer.lora_wv {
                l.apply_batched_threaded(&xb_batch, n, &mut v_batch, *threads)?;
            }

            for b in 0..n {
                let pos = b;
                let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                let k_tok = &mut k_batch[b * kv_dim..(b + 1) * kv_dim];
                let v_tok = &v_batch[b * kv_dim..(b + 1) * kv_dim];

                for head in 0..c.heads {
                    ops::rope_scaled(
                        &mut q_tok[head * head_dim..(head + 1) * head_dim],
                        pos,
                        c.rope_base,
                        c.rope_dim,
                        c.rope_scaling.as_ref(),
                    );
                }
                for head in 0..c.kv_heads {
                    ops::rope_scaled(
                        &mut k_tok[head * head_dim..(head + 1) * head_dim],
                        pos,
                        c.rope_base,
                        c.rope_dim,
                        c.rope_scaling.as_ref(),
                    );
                }

                kv_cache.store(index, pos, k_tok, v_tok);
            }

            let scale = 1.0 / (head_dim as f32).sqrt();
            let is_sliding = is_sliding_layer(index);
            for b in 0..n {
                let pos = b;
                let q_tok = &q_batch[b * att_dim..(b + 1) * att_dim];
                let att_tok = &mut att_batch[b * att_dim..(b + 1) * att_dim];

                for head in 0..c.heads {
                    let kv_head = head / group;
                    let query = &q_tok[head * head_dim..(head + 1) * head_dim];

                    let out = &mut att_tok[head * head_dim..(head + 1) * head_dim];
                    kv_cache.attention(
                        index,
                        query,
                        out,
                        ops::AttentionParams {
                            row_width: kv_dim,
                            offset: kv_head * head_dim,
                            len: pos + 1,
                            scale,
                            logit_cap: c.attn_logit_softcapping,
                            window: is_sliding.then_some(window),
                            position: pos,
                            tile_size: 64,
                            v_row_width: None,
                            v_offset: None,
                        },
                    );
                }
            }

            layer
                .wo
                .matmul_threaded(&att_batch, &mut xb2_batch, n, *threads, workspace)?;
            if let Some(l) = &layer.lora_wo {
                l.apply_batched_threaded(&att_batch, n, &mut xb2_batch, *threads)?;
            }
            if let Some(post_attn_norm) = &layer.post_attn_norm {
                for b in 0..n {
                    let tok_in = &xb2_batch[b * c.embedding..(b + 1) * c.embedding];
                    let tok_out = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                    ops::rms_norm(tok_in, post_attn_norm, c.eps, tok_out);
                }
                for (slot, delta) in x_batch.iter_mut().zip(xb_batch.iter()) {
                    *slot += delta;
                }
            } else {
                for (slot, delta) in x_batch.iter_mut().zip(xb2_batch.iter()) {
                    *slot += delta;
                }
            }

            for b in 0..n {
                let x_tok = &x_batch[b * c.embedding..(b + 1) * c.embedding];
                let xb_tok = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                ops::rms_norm(x_tok, &layer.ffn_norm, c.eps, xb_tok);
            }

            layer.eval_ffn(
                c,
                &xb_batch,
                n,
                &mut xb2_batch,
                &mut hb_batch,
                &mut hb2_batch,
                *threads,
                workspace,
            )?;
            if let Some(post_ffn_norm) = &layer.post_ffn_norm {
                for b in 0..n {
                    let tok_in = &xb2_batch[b * c.embedding..(b + 1) * c.embedding];
                    let tok_out = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                    ops::rms_norm(tok_in, post_ffn_norm, c.eps, tok_out);
                }
                for (slot, delta) in x_batch.iter_mut().zip(xb_batch.iter()) {
                    *slot += delta;
                }
            } else {
                for (slot, delta) in x_batch.iter_mut().zip(xb2_batch.iter()) {
                    *slot += delta;
                }
            }
        }

        for b in 0..n {
            let tok_x = &x_batch[b * c.embedding..(b + 1) * c.embedding];
            let tok_out = &mut out_states[b * c.embedding..(b + 1) * c.embedding];
            ops::rms_norm(tok_x, &self.output_norm, c.eps, tok_out);
        }
        Ok(())
    }

    /// Process a batch of prompt/speculative tokens in one pass and emit the logits
    /// for *every* token in the batch into `all_logits` (sized `n * c.vocab`).
    ///
    /// Leaves `state.logits` with the final token's logits.
    pub fn prefill_all_logits(
        &self,
        state: &mut State,
        tokens: &[u32],
        start_pos: usize,
        all_logits: &mut [f32],
    ) -> Result<()> {
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }
        let c = &self.config;
        if all_logits.len() != n * c.vocab {
            return Err(bad(format!(
                "all_logits length {} does not match batch {} * vocab {}",
                all_logits.len(),
                n,
                c.vocab
            )));
        }

        if n == 1 {
            self.forward(state, tokens[0], start_pos)?;
            all_logits.copy_from_slice(&state.logits);
            return Ok(());
        }

        let end_pos = start_pos
            .checked_add(n)
            .ok_or_else(|| bad("prefill range overflows"))?;
        if end_pos > state.context {
            return Err(bad(format!(
                "prefill end position {end_pos} exceeds {} token cache",
                state.context
            )));
        }

        let (head_dim, kv_dim, group) = (c.head_dim, c.kv_dim(), c.group());
        let att_dim = c.heads * head_dim;

        // Batch activation buffers:
        let mut x_batch = vec![0.0f32; n * c.embedding];
        for (b, &tok) in tokens.iter().enumerate() {
            let row = &mut x_batch[b * c.embedding..(b + 1) * c.embedding];
            self.token_embd.row(tok as usize, row)?;
        }
        if let Some(scale) = c.embedding_scale {
            for v in x_batch.iter_mut() {
                *v *= scale;
            }
        }

        let mut xb_batch = vec![0.0f32; n * c.embedding];
        let mut xb2_batch = vec![0.0f32; n * c.embedding];
        let mut att_batch = vec![0.0f32; n * att_dim];
        let mut q_batch = vec![0.0f32; n * att_dim];
        let mut k_batch = vec![0.0f32; n * kv_dim];
        let mut v_batch = vec![0.0f32; n * kv_dim];
        let mut hb_batch = vec![0.0f32; n * c.ffn];
        let mut hb2_batch = vec![0.0f32; n * c.ffn];

        let State {
            logits,
            workspace,
            kv_cache,
            threads,
            ..
        } = state;

        let is_sliding_layer = |idx: usize| (idx % 2 == 0) && c.sliding_window.is_some();
        let window = c.sliding_window.unwrap_or(usize::MAX);

        for (index, layer) in self.layers.iter().enumerate() {
            for b in 0..n {
                let x_tok = &x_batch[b * c.embedding..(b + 1) * c.embedding];
                let xb_tok = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                ops::rms_norm(x_tok, &layer.attn_norm, c.eps, xb_tok);
            }

            layer
                .wq
                .matmul_threaded(&xb_batch, &mut q_batch, n, *threads, workspace)?;
            if let Some(bq) = &layer.bq {
                for b in 0..n {
                    let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                    for (slot, val) in q_tok.iter_mut().zip(bq) {
                        *slot += val;
                    }
                }
            }
            if let Some(l) = &layer.lora_wq {
                l.apply_batched_threaded(&xb_batch, n, &mut q_batch, *threads)?;
            }
            layer
                .wk
                .matmul_threaded(&xb_batch, &mut k_batch, n, *threads, workspace)?;
            if let Some(bk) = &layer.bk {
                for b in 0..n {
                    let k_tok = &mut k_batch[b * kv_dim..(b + 1) * kv_dim];
                    for (slot, val) in k_tok.iter_mut().zip(bk) {
                        *slot += val;
                    }
                }
            }
            if let Some(l) = &layer.lora_wk {
                l.apply_batched_threaded(&xb_batch, n, &mut k_batch, *threads)?;
            }
            layer
                .wv
                .matmul_threaded(&xb_batch, &mut v_batch, n, *threads, workspace)?;
            if let Some(bv) = &layer.bv {
                for b in 0..n {
                    let v_tok = &mut v_batch[b * kv_dim..(b + 1) * kv_dim];
                    for (slot, val) in v_tok.iter_mut().zip(bv) {
                        *slot += val;
                    }
                }
            }
            if let Some(l) = &layer.lora_wv {
                l.apply_batched_threaded(&xb_batch, n, &mut v_batch, *threads)?;
            }

            for b in 0..n {
                let pos = start_pos + b;
                let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                let k_tok = &mut k_batch[b * kv_dim..(b + 1) * kv_dim];
                let v_tok = &v_batch[b * kv_dim..(b + 1) * kv_dim];

                for head in 0..c.heads {
                    ops::rope_scaled(
                        &mut q_tok[head * head_dim..(head + 1) * head_dim],
                        pos,
                        c.rope_base,
                        c.rope_dim,
                        c.rope_scaling.as_ref(),
                    );
                }
                for head in 0..c.kv_heads {
                    ops::rope_scaled(
                        &mut k_tok[head * head_dim..(head + 1) * head_dim],
                        pos,
                        c.rope_base,
                        c.rope_dim,
                        c.rope_scaling.as_ref(),
                    );
                }

                kv_cache.store(index, pos, k_tok, v_tok);
            }

            let scale = 1.0 / (head_dim as f32).sqrt();
            let is_sliding = is_sliding_layer(index);
            for b in 0..n {
                let pos = start_pos + b;
                let q_tok = &q_batch[b * att_dim..(b + 1) * att_dim];
                let att_tok = &mut att_batch[b * att_dim..(b + 1) * att_dim];

                for head in 0..c.heads {
                    let kv_head = head / group;
                    let query = &q_tok[head * head_dim..(head + 1) * head_dim];

                    let out = &mut att_tok[head * head_dim..(head + 1) * head_dim];
                    kv_cache.attention(
                        index,
                        query,
                        out,
                        ops::AttentionParams {
                            row_width: kv_dim,
                            offset: kv_head * head_dim,
                            len: pos + 1,
                            scale,
                            logit_cap: c.attn_logit_softcapping,
                            window: is_sliding.then_some(window),
                            position: pos,
                            tile_size: 64,
                            v_row_width: None,
                            v_offset: None,
                        },
                    );
                }
            }

            layer
                .wo
                .matmul_threaded(&att_batch, &mut xb2_batch, n, *threads, workspace)?;
            if let Some(l) = &layer.lora_wo {
                l.apply_batched_threaded(&att_batch, n, &mut xb2_batch, *threads)?;
            }
            if let Some(post_attn_norm) = &layer.post_attn_norm {
                for b in 0..n {
                    let tok_in = &xb2_batch[b * c.embedding..(b + 1) * c.embedding];
                    let tok_out = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                    ops::rms_norm(tok_in, post_attn_norm, c.eps, tok_out);
                }
                for (slot, delta) in x_batch.iter_mut().zip(xb_batch.iter()) {
                    *slot += delta;
                }
            } else {
                for (slot, delta) in x_batch.iter_mut().zip(xb2_batch.iter()) {
                    *slot += delta;
                }
            }

            for b in 0..n {
                let x_tok = &x_batch[b * c.embedding..(b + 1) * c.embedding];
                let xb_tok = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                ops::rms_norm(x_tok, &layer.ffn_norm, c.eps, xb_tok);
            }

            layer.eval_ffn(
                c,
                &xb_batch,
                n,
                &mut xb2_batch,
                &mut hb_batch,
                &mut hb2_batch,
                *threads,
                workspace,
            )?;
            if let Some(post_ffn_norm) = &layer.post_ffn_norm {
                for b in 0..n {
                    let tok_in = &xb2_batch[b * c.embedding..(b + 1) * c.embedding];
                    let tok_out = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                    ops::rms_norm(tok_in, post_ffn_norm, c.eps, tok_out);
                }
                for (slot, delta) in x_batch.iter_mut().zip(xb_batch.iter()) {
                    *slot += delta;
                }
            } else {
                for (slot, delta) in x_batch.iter_mut().zip(xb2_batch.iter()) {
                    *slot += delta;
                }
            }
        }

        for b in 0..n {
            let x_tok = &x_batch[b * c.embedding..(b + 1) * c.embedding];
            let xb_tok = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
            ops::rms_norm(x_tok, &self.output_norm, c.eps, xb_tok);
        }

        self.output
            .matmul_threaded(&xb_batch, all_logits, n, *threads, workspace)?;
        if let Some(l) = &self.lora_output {
            l.apply_batched_threaded(&xb_batch, n, all_logits, *threads)?;
        }

        if let Some(cap) = c.final_logit_softcapping {
            for logit in all_logits.iter_mut() {
                *logit = cap * (*logit / cap).tanh();
            }
        }

        let last_logits = &all_logits[(n - 1) * c.vocab..n * c.vocab];
        logits.copy_from_slice(last_logits);
        Ok(())
    }

    /// Process a long sequence of prompt tokens in chunks of `chunk_size`.
    ///
    /// Bounds peak latency and memory pressure during large context ingestion.
    /// Bit-identical to single-pass `prefill` and sequential `forward`.
    pub fn prefill_chunked(
        &self,
        state: &mut State,
        tokens: &[u32],
        start_pos: usize,
        chunk_size: usize,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let chunk = if chunk_size == 0 { 512 } else { chunk_size };
        let mut offset = 0;
        while offset < tokens.len() {
            let end = (offset + chunk).min(tokens.len());
            self.prefill(state, &tokens[offset..end], start_pos + offset)?;
            offset = end;
        }
        Ok(())
    }

    /// Process a sequence of prompt/draft tokens in chunks of `chunk_size` and
    /// emit intermediate logits for every token in `all_logits`.
    pub fn prefill_all_logits_chunked(
        &self,
        state: &mut State,
        tokens: &[u32],
        start_pos: usize,
        all_logits: &mut [f32],
        chunk_size: usize,
    ) -> Result<()> {
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }
        let c = &self.config;
        if all_logits.len() != n * c.vocab {
            return Err(bad(format!(
                "all_logits buffer size {} != n ({n}) * vocab ({})",
                all_logits.len(),
                c.vocab
            )));
        }
        let chunk = if chunk_size == 0 { 512 } else { chunk_size };
        let mut offset = 0;
        while offset < n {
            let end = (offset + chunk).min(n);
            let chunk_tokens = &tokens[offset..end];
            let chunk_logits = &mut all_logits[offset * c.vocab..end * c.vocab];
            self.prefill_all_logits(state, chunk_tokens, start_pos + offset, chunk_logits)?;
            offset = end;
        }
        Ok(())
    }

    /// Decode a single token with automatic KV cache eviction when current position reaches context capacity.
    ///
    /// Preserves initial `sink_tokens` (default 4 if 0) and evicts oldest middle tokens so continuous
    /// generation can proceed indefinitely without memory exhaustion.
    pub fn forward_streaming(
        &self,
        state: &mut State,
        token: u32,
        current_pos: &mut usize,
        sink_tokens: usize,
    ) -> Result<()> {
        if *current_pos >= state.context() {
            let sinks = if sink_tokens == 0 { 4 } else { sink_tokens };
            *current_pos = state
                .kv_cache
                .evict_streaming(*current_pos, state.context() - 1, sinks);
        }
        self.forward(state, token, *current_pos)?;
        *current_pos += 1;
        Ok(())
    }

    /// Streaming forward pass with adaptive entropy-bounded rolling sink cache eviction.
    pub fn forward_adaptive_streaming(
        &self,
        state: &mut State,
        token: u32,
        current_pos: &mut usize,
        config: &crate::kv::AdaptiveEntropyConfig,
        attention_scores: Option<&[f32]>,
    ) -> Result<()> {
        if *current_pos >= state.context() {
            let default_scores;
            let scores = match attention_scores {
                Some(s) => s,
                None => {
                    default_scores =
                        state.logits[..(*current_pos).min(state.logits.len())].to_vec();
                    &default_scores
                }
            };
            *current_pos = state.evict_adaptive_entropy(config, scores);
        }
        self.forward(state, token, *current_pos)?;
        *current_pos += 1;
        Ok(())
    }

    /// Executes prefill for a packed heterogeneous batch of sequences without zero-padding,
    /// returning the final token logits for each sequence in the packed batch.
    pub fn prefill_packed(
        &self,
        state: &mut State,
        packed: &crate::scheduler::PackedBatch,
    ) -> Result<Vec<Vec<f32>>> {
        let mut logits = Vec::with_capacity(packed.num_sequences());
        for i in 0..packed.num_sequences() {
            let tokens = packed.sequence_tokens(i);
            let start_pos = packed.sequence_positions(i)[0];
            self.prefill(state, tokens, start_pos)?;
            logits.push(state.logits.clone());
        }
        Ok(logits)
    }
}

/// Stateful, interruptible chunked prompt prefill runner.
///
/// Permits progress reporting, cooperative timesharing, and bounded latency
/// across huge prompts without blocking runtime threads.
pub struct ChunkedPrefill<'a, 'm> {
    model: &'m Model<'a>,
    tokens: Vec<u32>,
    start_pos: usize,
    chunk_size: usize,
    processed: usize,
}

impl<'a, 'm> ChunkedPrefill<'a, 'm> {
    pub fn new(
        model: &'m Model<'a>,
        tokens: Vec<u32>,
        start_pos: usize,
        chunk_size: usize,
    ) -> Self {
        let eff_chunk = if chunk_size == 0 { 512 } else { chunk_size };
        Self {
            model,
            tokens,
            start_pos,
            chunk_size: eff_chunk,
            processed: 0,
        }
    }

    #[inline]
    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }

    #[inline]
    pub fn processed_tokens(&self) -> usize {
        self.processed
    }

    #[inline]
    pub fn is_finished(&self) -> bool {
        self.processed >= self.tokens.len()
    }

    /// Process the next slice of tokens. Returns the number of tokens processed.
    pub fn step(&mut self, state: &mut State) -> Result<usize> {
        if self.is_finished() {
            return Ok(0);
        }
        let end = (self.processed + self.chunk_size).min(self.tokens.len());
        let chunk = &self.tokens[self.processed..end];
        self.model
            .prefill(state, chunk, self.start_pos + self.processed)?;
        let count = end - self.processed;
        self.processed = end;
        Ok(count)
    }

    /// Run all remaining chunks, notifying `callback(processed, total)` after each chunk.
    pub fn run_with_progress<F>(&mut self, state: &mut State, mut callback: F) -> Result<()>
    where
        F: FnMut(usize, usize),
    {
        while !self.is_finished() {
            self.step(state)?;
            callback(self.processed, self.tokens.len());
        }
        Ok(())
    }
}

/// Everything one sequence needs that is not a weight.
///
/// Separate from `Model` so the weights stay shareable: two sequences decoding
/// at once want two of these and one of those.
#[derive(Clone)]
pub struct State {
    pub(crate) x: Vec<f32>,
    pub(crate) xb: Vec<f32>,
    pub(crate) xb2: Vec<f32>,
    pub(crate) hb: Vec<f32>,
    pub(crate) hb2: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    /// Scores for the last token processed, one per vocabulary entry.
    pub logits: Vec<f32>,
    pub(crate) workspace: Workspace,
    pub kv_cache: KvCache,
    context: usize,
    /// Workers per matvec. Defaults to the machine's parallelism.
    pub threads: usize,
}

/// Cap on the cache a bare `State::new` will allocate.
///
/// The KV cache is layers * context * kv_dim * 4 bytes, twice over. A model
/// advertising 128k context would ask for several gigabytes before generating
/// a single token, so the default is a working size and the full window is
/// opt-in through `with_context`.
pub const DEFAULT_CONTEXT: usize = 2048;

impl State {
    pub fn new(config: &Config) -> Self {
        Self::with_kv_cache(
            config,
            config.context.min(DEFAULT_CONTEXT),
            KvCacheType::F32,
        )
    }

    pub fn with_context(config: &Config, context: usize) -> Self {
        Self::with_kv_cache(config, context, KvCacheType::F32)
    }

    /// Allocate state with explicit context window and KV cache quantization format.
    pub fn with_kv_cache(config: &Config, context: usize, cache_type: KvCacheType) -> Self {
        let context = context.clamp(1, config.context);
        let k_dim = if config.arch.starts_with("deepseek") && config.kv_lora_rank.is_some() {
            config.heads * config.head_dim
        } else {
            config.kv_dim()
        };
        let v_dim = if let Some(v_hd) = config.v_head_dim {
            config.heads * v_hd
        } else {
            k_dim
        };
        let mla_latent_dim =
            config.kv_lora_rank.unwrap_or(0) + config.qk_rope_head_dim.unwrap_or(0);
        let q_latent_dim = config.q_lora_rank.unwrap_or(0);
        let widest = config
            .embedding
            .max(config.ffn)
            .max(config.heads * config.head_dim)
            .max(v_dim)
            .max(mla_latent_dim)
            .max(q_latent_dim);
        let total_layers = config.layers + config.nextn_predict_layers.unwrap_or(0);
        Self {
            x: vec![0.0; config.embedding],
            xb: vec![0.0; widest],
            xb2: vec![0.0; widest],
            hb: vec![0.0; config.ffn],
            hb2: vec![0.0; config.ffn],
            q: vec![0.0; config.heads * config.head_dim],
            k: vec![0.0; k_dim],
            v: vec![0.0; v_dim],
            logits: vec![0.0; config.vocab],
            workspace: Workspace::new(widest),
            kv_cache: KvCache::new_asymmetric(total_layers, context, k_dim, v_dim, cache_type),
            context,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        }
    }

    /// View the final hidden representation before output normalization from the last forward pass.
    #[inline]
    pub fn final_hidden_state(&self) -> &[f32] {
        &self.x
    }

    /// How many tokens fit before `forward` refuses.
    pub fn context(&self) -> usize {
        self.context
    }

    /// Turn the integer kernels on or off.
    ///
    /// On by default. Off gives the exact f32 arithmetic, which is slower, and the
    /// thing to reach for when deciding whether a numerical oddity comes from
    /// the activation quantization or from somewhere else.
    pub fn set_int8(&mut self, enabled: bool) {
        self.workspace.int8 = enabled;
    }

    /// Dynamic context truncation using rolling sink cache eviction.
    pub fn truncate_context_rolling_sink(
        &mut self,
        current_pos: usize,
        target_len: usize,
        sink_tokens: usize,
    ) -> usize {
        self.kv_cache
            .evict_streaming(current_pos, target_len, sink_tokens)
    }

    /// Dynamic context truncation using adaptive entropy bounds.
    pub fn evict_adaptive_entropy(
        &mut self,
        config: &crate::kv::AdaptiveEntropyConfig,
        scores: &[f32],
    ) -> usize {
        let keep = crate::kv::select_adaptive_entropy_indices(scores, config);
        self.kv_cache.compact_indices(&keep);
        keep.len()
    }

    /// Bytes held by the KV cache, which dominates everything else here.
    pub fn cache_bytes(&self) -> usize {
        self.kv_cache.cache_bytes()
    }

    pub fn key_cache(&self) -> &[f32] {
        self.kv_cache.as_f32_slices().map_or(&[], |(k, _)| k)
    }

    pub fn value_cache(&self) -> &[f32] {
        self.kv_cache.as_f32_slices().map_or(&[], |(_, v)| v)
    }

    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    pub fn kv_cache_mut(&mut self) -> &mut KvCache {
        &mut self.kv_cache
    }

    pub fn kv_cache_type(&self) -> KvCacheType {
        self.kv_cache.cache_type()
    }

    /// Zero out KV-cache slots from `keep_pos` up to `end_pos`.
    ///
    /// Used by speculative decoding to roll back candidate KV-cache entries when
    /// speculative tokens are rejected by the target model.
    pub fn truncate_cache(&mut self, _config: &Config, keep_pos: usize, end_pos: usize) {
        self.kv_cache.truncate(keep_pos, end_pos);
    }

    /// Copy key and value representations from `src_pos` to `dst_pos` across all layers.
    pub fn copy_cache_slot(&mut self, src_pos: usize, dst_pos: usize) {
        self.kv_cache.copy_token_slot(src_pos, dst_pos);
    }

    /// Set `logits` buffer to the given slice.
    pub fn set_logits(&mut self, new_logits: &[f32]) {
        self.logits.copy_from_slice(new_logits);
    }

    /// Evict cached tokens according to StreamingLLM policy when context capacity is reached.
    /// Returns the new sequence position.
    pub fn evict_cache(
        &mut self,
        current_pos: usize,
        max_capacity: usize,
        sink_tokens: usize,
    ) -> usize {
        self.kv_cache
            .evict_streaming(current_pos, max_capacity, sink_tokens)
    }

    /// Selectively compact KV-cache entries to retain only the specified sorted `keep_indices`.
    pub fn compact_cache(&mut self, keep_indices: &[usize]) {
        self.kv_cache.compact_indices(keep_indices);
    }
}
