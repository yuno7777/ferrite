//! Llama-family transformer: hyperparameters, weight binding, forward pass.

use std::io::{Error, ErrorKind, Result};

use crate::gguf::{GgmlType, Gguf};
use crate::ops;
use crate::quant;

fn bad(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

/// Architectures whose tensor names and layer shape this implements. Others
/// are rejected by name rather than half-loaded — a model that runs with the
/// wrong layer structure produces fluent nonsense, not an error.
const SUPPORTED: [&str; 4] = ["llama", "qwen2", "gemma", "gemma2"];

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
    pub attn_logit_softcapping: Option<f32>,
    pub final_logit_softcapping: Option<f32>,
    pub sliding_window: Option<usize>,
    pub embedding_scale: Option<f32>,
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
            .or_else(|| model.arch_u64("rope.dimension_count"))
            .map(|v| v as usize)
            .unwrap_or_else(|| embedding.checked_div(heads).unwrap_or(0));

        if heads == 0 || kv_heads == 0 || head_dim == 0 {
            return Err(bad("head counts must be non-zero"));
        }
        if heads % kv_heads != 0 {
            return Err(bad(format!(
                "{heads} heads is not a multiple of {kv_heads} kv heads"
            )));
        }
        // In LLaMA and Qwen, heads * head_dim == embedding. In Gemma 2, head_dim is decoupled
        // from embedding (e.g., 8 heads * 256 = 2048 != 2304 embedding) and wo projects from
        // heads * head_dim to embedding.
        if arch != "gemma" && arch != "gemma2" && heads * head_dim != embedding {
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
                if arch == "qwen2" {
                    1_000_000.0
                } else {
                    10000.0
                }
            }),
            attn_logit_softcapping,
            final_logit_softcapping,
            sliding_window,
            embedding_scale,
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
pub struct Workspace {
    /// Expanded row, for weight types with no faster path.
    scratch: Vec<f32>,
    /// The activation, quantized once per matvec and shared by every row.
    quantized: quant::activation::Quantized,
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
            int8: true,
        }
    }
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
        let Workspace {
            scratch, quantized, ..
        } = workspace;

        // Below a few rows per worker the spawn costs more than the work.
        if threads <= 1 || self.rows < threads * 4 {
            return self.rows_into(out, 0, x, quantized, int8, scratch);
        }

        let chunk = self.rows.div_ceil(threads);
        let outcomes: Vec<Result<()>> = std::thread::scope(|scope| {
            let quantized = &*quantized;
            let workers: Vec<_> = out
                .chunks_mut(chunk)
                .enumerate()
                .map(|(index, rows)| {
                    scope.spawn(move || {
                        // Only the slow path needs a scratch row, so a type
                        // with a kernel allocates nothing per worker.
                        let mut scratch = if int8 || quant::dot::supports(self.ty) {
                            Vec::new()
                        } else {
                            vec![0.0; self.cols]
                        };
                        self.rows_into(rows, index * chunk, x, quantized, int8, &mut scratch)
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .unwrap_or_else(|_| Err(bad("worker panicked")))
                })
                .collect()
        });

        outcomes.into_iter().collect::<Result<Vec<()>>>()?;
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

        let has_fused = quant::dot::supports(self.ty);

        for (r, row_idx) in (0..self.rows).enumerate() {
            let row = self.row_slice(row_idx);
            if has_fused {
                for b in 0..batch {
                    let x_tok = &x[b * self.cols..(b + 1) * self.cols];
                    out[b * self.rows + r] = quant::dot::fused(self.ty, row, x_tok).unwrap();
                }
            } else {
                // Dequantize row ONCE for all batch tokens!
                quant::dequantize(self.ty, row, scratch)?;
                for b in 0..batch {
                    let x_tok = &x[b * self.cols..(b + 1) * self.cols];
                    out[b * self.rows + r] = ops::dot(scratch, x_tok);
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
    fn row_slice(&self, index: usize) -> &'a [u8] {
        let start = index * self.row_bytes;
        &self.bytes[start..start + self.row_bytes]
    }
}

pub struct Layer<'a> {
    pub attn_norm: Vec<f32>,
    pub post_attn_norm: Option<Vec<f32>>,
    pub wq: Weight<'a>,
    pub bq: Option<Vec<f32>>,
    pub wk: Weight<'a>,
    pub bk: Option<Vec<f32>>,
    pub wv: Weight<'a>,
    pub bv: Option<Vec<f32>>,
    pub wo: Weight<'a>,
    pub ffn_norm: Vec<f32>,
    pub post_ffn_norm: Option<Vec<f32>>,
    pub gate: Weight<'a>,
    pub up: Weight<'a>,
    pub down: Weight<'a>,
}

pub struct Model<'a> {
    pub config: Config,
    pub token_embd: Weight<'a>,
    pub layers: Vec<Layer<'a>>,
    pub output_norm: Vec<f32>,
    pub output: Weight<'a>,
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
            layers.push(Layer {
                attn_norm: load_vector(gguf, &format!("blk.{index}.attn_norm.weight"))?,
                post_attn_norm: load_vector_optional(
                    gguf,
                    &format!("blk.{index}.post_attention_norm.weight"),
                )?,
                wq: Weight::bind(gguf, &format!("blk.{index}.attn_q.weight"))?,
                bq: load_vector_optional(gguf, &format!("blk.{index}.attn_q.bias"))?,
                wk: Weight::bind(gguf, &format!("blk.{index}.attn_k.weight"))?,
                bk: load_vector_optional(gguf, &format!("blk.{index}.attn_k.bias"))?,
                wv: Weight::bind(gguf, &format!("blk.{index}.attn_v.weight"))?,
                bv: load_vector_optional(gguf, &format!("blk.{index}.attn_v.bias"))?,
                wo: Weight::bind(gguf, &format!("blk.{index}.attn_output.weight"))?,
                ffn_norm: load_vector(gguf, &format!("blk.{index}.ffn_norm.weight"))?,
                post_ffn_norm: load_vector_optional(
                    gguf,
                    &format!("blk.{index}.post_ffn_norm.weight"),
                )?,
                gate: Weight::bind(gguf, &format!("blk.{index}.ffn_gate.weight"))?,
                up: Weight::bind(gguf, &format!("blk.{index}.ffn_up.weight"))?,
                down: Weight::bind(gguf, &format!("blk.{index}.ffn_down.weight"))?,
            });
        }

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
            output_norm: load_vector(gguf, "output_norm.weight")?,
            output,
        })
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
            att,
            logits,
            workspace,
            key_cache,
            value_cache,
            context,
            threads,
        } = state;

        let (head_dim, kv_dim, group) = (c.head_dim, c.kv_dim(), c.group());
        let att_dim = c.heads * head_dim;
        self.token_embd.row(token as usize, x)?;
        if let Some(scale) = c.embedding_scale {
            for v in x.iter_mut() {
                *v *= scale;
            }
        }

        let is_sliding_layer = |idx: usize| (idx % 2 == 0) && c.sliding_window.is_some();
        let window = c.sliding_window.unwrap_or(usize::MAX);

        for (index, layer) in self.layers.iter().enumerate() {
            ops::rms_norm(x, &layer.attn_norm, c.eps, &mut xb[..c.embedding]);
            layer
                .wq
                .matvec_threaded(&xb[..c.embedding], q, *threads, workspace)?;
            if let Some(bq) = &layer.bq {
                for (slot, b) in q.iter_mut().zip(bq) {
                    *slot += b;
                }
            }
            layer
                .wk
                .matvec_threaded(&xb[..c.embedding], k, *threads, workspace)?;
            if let Some(bk) = &layer.bk {
                for (slot, b) in k.iter_mut().zip(bk) {
                    *slot += b;
                }
            }
            layer
                .wv
                .matvec_threaded(&xb[..c.embedding], v, *threads, workspace)?;
            if let Some(bv) = &layer.bv {
                for (slot, b) in v.iter_mut().zip(bv) {
                    *slot += b;
                }
            }

            // Position is baked into the query and key, never into the value.
            for head in 0..c.heads {
                ops::rope(
                    &mut q[head * head_dim..(head + 1) * head_dim],
                    position,
                    c.rope_base,
                );
            }
            for head in 0..c.kv_heads {
                ops::rope(
                    &mut k[head * head_dim..(head + 1) * head_dim],
                    position,
                    c.rope_base,
                );
            }

            let slot = (index * *context + position) * kv_dim;
            key_cache[slot..slot + kv_dim].copy_from_slice(k);
            value_cache[slot..slot + kv_dim].copy_from_slice(v);

            let scale = 1.0 / (head_dim as f32).sqrt();
            let is_sliding = is_sliding_layer(index);
            for head in 0..c.heads {
                // Grouped-query attention: several query heads share one KV head.
                let kv_head = head / group;
                let query = &q[head * head_dim..(head + 1) * head_dim];

                for (past, score) in att[..=position].iter_mut().enumerate() {
                    if is_sliding && position.saturating_sub(past) >= window {
                        *score = f32::NEG_INFINITY;
                        continue;
                    }
                    let at = (index * *context + past) * kv_dim + kv_head * head_dim;
                    let mut raw = ops::dot(query, &key_cache[at..at + head_dim]) * scale;
                    if let Some(cap) = c.attn_logit_softcapping {
                        raw = cap * (raw / cap).tanh();
                    }
                    *score = raw;
                }
                ops::softmax(&mut att[..=position]);

                let out = &mut xb[head * head_dim..(head + 1) * head_dim];
                out.fill(0.0);
                for (past, weight) in att[..=position].iter().enumerate() {
                    let at = (index * *context + past) * kv_dim + kv_head * head_dim;
                    for (slot, value) in out.iter_mut().zip(&value_cache[at..at + head_dim]) {
                        *slot += weight * value;
                    }
                }
            }

            layer.wo.matvec_threaded(
                &xb[..att_dim],
                &mut xb2[..c.embedding],
                *threads,
                workspace,
            )?;
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
            layer
                .gate
                .matvec_threaded(&xb[..c.embedding], hb, *threads, workspace)?;
            layer
                .up
                .matvec_threaded(&xb[..c.embedding], hb2, *threads, workspace)?;
            // SwiGLU: the gate is squashed, the up projection is not.
            for (gate, up) in hb.iter_mut().zip(hb2.iter()) {
                *gate = ops::silu(*gate) * up;
            }
            layer
                .down
                .matvec_threaded(hb, &mut xb2[..c.embedding], *threads, workspace)?;
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
        }

        ops::rms_norm(x, &self.output_norm, c.eps, &mut xb[..c.embedding]);
        self.output
            .matvec_threaded(&xb[..c.embedding], logits, *threads, workspace)?;
        if let Some(cap) = c.final_logit_softcapping {
            for logit in logits.iter_mut() {
                *logit = cap * (*logit / cap).tanh();
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
            att,
            logits,
            workspace,
            key_cache,
            value_cache,
            context,
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
                .matmul(&xb_batch, &mut q_batch, n, &mut workspace.scratch)?;
            if let Some(bq) = &layer.bq {
                for b in 0..n {
                    let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                    for (slot, val) in q_tok.iter_mut().zip(bq) {
                        *slot += val;
                    }
                }
            }
            layer
                .wk
                .matmul(&xb_batch, &mut k_batch, n, &mut workspace.scratch)?;
            if let Some(bk) = &layer.bk {
                for b in 0..n {
                    let k_tok = &mut k_batch[b * kv_dim..(b + 1) * kv_dim];
                    for (slot, val) in k_tok.iter_mut().zip(bk) {
                        *slot += val;
                    }
                }
            }
            layer
                .wv
                .matmul(&xb_batch, &mut v_batch, n, &mut workspace.scratch)?;
            if let Some(bv) = &layer.bv {
                for b in 0..n {
                    let v_tok = &mut v_batch[b * kv_dim..(b + 1) * kv_dim];
                    for (slot, val) in v_tok.iter_mut().zip(bv) {
                        *slot += val;
                    }
                }
            }

            for b in 0..n {
                let pos = start_pos + b;
                let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                let k_tok = &mut k_batch[b * kv_dim..(b + 1) * kv_dim];
                let v_tok = &v_batch[b * kv_dim..(b + 1) * kv_dim];

                for head in 0..c.heads {
                    ops::rope(
                        &mut q_tok[head * head_dim..(head + 1) * head_dim],
                        pos,
                        c.rope_base,
                    );
                }
                for head in 0..c.kv_heads {
                    ops::rope(
                        &mut k_tok[head * head_dim..(head + 1) * head_dim],
                        pos,
                        c.rope_base,
                    );
                }

                let slot = (index * *context + pos) * kv_dim;
                key_cache[slot..slot + kv_dim].copy_from_slice(k_tok);
                value_cache[slot..slot + kv_dim].copy_from_slice(v_tok);
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

                    for (past, score) in att[..=pos].iter_mut().enumerate() {
                        if is_sliding && pos.saturating_sub(past) >= window {
                            *score = f32::NEG_INFINITY;
                            continue;
                        }
                        let at = (index * *context + past) * kv_dim + kv_head * head_dim;
                        let mut raw = ops::dot(query, &key_cache[at..at + head_dim]) * scale;
                        if let Some(cap) = c.attn_logit_softcapping {
                            raw = cap * (raw / cap).tanh();
                        }
                        *score = raw;
                    }
                    ops::softmax(&mut att[..=pos]);

                    let out = &mut att_tok[head * head_dim..(head + 1) * head_dim];
                    out.fill(0.0);
                    for (past, weight) in att[..=pos].iter().enumerate() {
                        let at = (index * *context + past) * kv_dim + kv_head * head_dim;
                        for (slot, value) in out.iter_mut().zip(&value_cache[at..at + head_dim]) {
                            *slot += weight * value;
                        }
                    }
                }
            }

            layer
                .wo
                .matmul(&att_batch, &mut xb2_batch, n, &mut workspace.scratch)?;
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

            layer
                .gate
                .matmul(&xb_batch, &mut hb_batch, n, &mut workspace.scratch)?;
            layer
                .up
                .matmul(&xb_batch, &mut hb2_batch, n, &mut workspace.scratch)?;
            for (gate, up) in hb_batch.iter_mut().zip(hb2_batch.iter()) {
                *gate = ops::silu(*gate) * up;
            }

            layer
                .down
                .matmul(&hb_batch, &mut xb2_batch, n, &mut workspace.scratch)?;
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
            .matvec(last_xb, logits, &mut workspace.scratch)?;
        if let Some(cap) = c.final_logit_softcapping {
            for logit in logits.iter_mut() {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        Ok(())
    }
}

/// Everything one sequence needs that is not a weight.
///
/// Separate from `Model` so the weights stay shareable: two sequences decoding
/// at once want two of these and one of those.
pub struct State {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    /// Scores for the last token processed, one per vocabulary entry.
    pub logits: Vec<f32>,
    workspace: Workspace,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
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
        Self::with_context(config, config.context.min(DEFAULT_CONTEXT))
    }

    pub fn with_context(config: &Config, context: usize) -> Self {
        let context = context.clamp(1, config.context);
        let kv = config.layers * context * config.kv_dim();
        let widest = config
            .embedding
            .max(config.ffn)
            .max(config.heads * config.head_dim);
        Self {
            x: vec![0.0; config.embedding],
            xb: vec![0.0; widest],
            xb2: vec![0.0; widest],
            hb: vec![0.0; config.ffn],
            hb2: vec![0.0; config.ffn],
            q: vec![0.0; config.heads * config.head_dim],
            k: vec![0.0; config.kv_dim()],
            v: vec![0.0; config.kv_dim()],
            att: vec![0.0; context],
            logits: vec![0.0; config.vocab],
            workspace: Workspace::new(widest),
            key_cache: vec![0.0; kv],
            value_cache: vec![0.0; kv],
            context,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        }
    }

    /// How many tokens fit before `forward` refuses.
    pub fn context(&self) -> usize {
        self.context
    }

    /// Turn the integer kernels on or off.
    ///
    /// On by default. Off gives the exact f32 arithmetic — slower, and the
    /// thing to reach for when deciding whether a numerical oddity comes from
    /// the activation quantization or from somewhere else.
    pub fn set_int8(&mut self, enabled: bool) {
        self.workspace.int8 = enabled;
    }

    /// Bytes held by the KV cache, which dominates everything else here.
    pub fn cache_bytes(&self) -> usize {
        (self.key_cache.len() + self.value_cache.len()) * std::mem::size_of::<f32>()
    }

    pub fn key_cache(&self) -> &[f32] {
        &self.key_cache
    }

    pub fn value_cache(&self) -> &[f32] {
        &self.value_cache
    }
}
