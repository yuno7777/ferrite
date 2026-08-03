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
const SUPPORTED: [&str; 1] = ["llama"];

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
            .arch_u64("rope.dimension_count")
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
        // The attention output projection consumes heads * head_dim and emits
        // embedding, so the two have to agree.
        if heads * head_dim != embedding {
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

        Ok(Self {
            arch,
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
            rope_base: model.arch_f32("rope.freq_base").unwrap_or(10000.0),
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
    /// order, so the result is bit-identical to the single-threaded path no
    /// matter how the split falls. That is worth preserving: a threaded kernel
    /// that merely *approximates* the reference cannot be diffed against it.
    pub fn matvec_threaded(
        &self,
        x: &[f32],
        out: &mut [f32],
        threads: usize,
        scratch: &mut [f32],
    ) -> Result<()> {
        // Below a few rows per worker the spawn costs more than the work.
        if threads <= 1 || self.rows < threads * 4 {
            return self.matvec(x, out, scratch);
        }

        let chunk = self.rows.div_ceil(threads);
        let outcomes: Vec<Result<()>> = std::thread::scope(|scope| {
            let workers: Vec<_> = out
                .chunks_mut(chunk)
                .enumerate()
                .map(|(index, rows)| {
                    scope.spawn(move || {
                        // One scratch row per worker. The allocation is dwarfed
                        // by the dequantize it feeds.
                        let mut scratch = vec![0.0; self.cols];
                        let first = index * chunk;
                        for (offset, slot) in rows.iter_mut().enumerate() {
                            let at = (first + offset) * self.row_bytes;
                            quant::dequantize(
                                self.ty,
                                &self.bytes[at..at + self.row_bytes],
                                &mut scratch,
                            )?;
                            *slot = ops::dot(&scratch, x);
                        }
                        Ok(())
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

    /// `out = self * x`, expanding one row at a time through `scratch`.
    pub fn matvec(&self, x: &[f32], out: &mut [f32], scratch: &mut [f32]) -> Result<()> {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(out.len(), self.rows);
        debug_assert_eq!(scratch.len(), self.cols);

        for (index, slot) in out.iter_mut().enumerate() {
            let start = index * self.row_bytes;
            quant::dequantize(self.ty, &self.bytes[start..start + self.row_bytes], scratch)?;
            *slot = ops::dot(scratch, x);
        }
        Ok(())
    }
}

pub struct Layer<'a> {
    pub attn_norm: Vec<f32>,
    pub wq: Weight<'a>,
    pub wk: Weight<'a>,
    pub wv: Weight<'a>,
    pub wo: Weight<'a>,
    pub ffn_norm: Vec<f32>,
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

impl<'a> Model<'a> {
    pub fn load(gguf: &'a Gguf) -> Result<Self> {
        let config = Config::from_gguf(gguf)?;
        let token_embd = Weight::bind(gguf, "token_embd.weight")?;

        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            layers.push(Layer {
                attn_norm: load_vector(gguf, &format!("blk.{index}.attn_norm.weight"))?,
                wq: Weight::bind(gguf, &format!("blk.{index}.attn_q.weight"))?,
                wk: Weight::bind(gguf, &format!("blk.{index}.attn_k.weight"))?,
                wv: Weight::bind(gguf, &format!("blk.{index}.attn_v.weight"))?,
                wo: Weight::bind(gguf, &format!("blk.{index}.attn_output.weight"))?,
                ffn_norm: load_vector(gguf, &format!("blk.{index}.ffn_norm.weight"))?,
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
            scratch,
            key_cache,
            value_cache,
            context,
            threads,
        } = state;

        let (head_dim, kv_dim, group) = (c.head_dim, c.kv_dim(), c.group());
        self.token_embd.row(token as usize, x)?;

        for (index, layer) in self.layers.iter().enumerate() {
            ops::rms_norm(x, &layer.attn_norm, c.eps, xb);
            layer
                .wq
                .matvec_threaded(xb, q, *threads, &mut scratch[..c.embedding])?;
            layer
                .wk
                .matvec_threaded(xb, k, *threads, &mut scratch[..c.embedding])?;
            layer
                .wv
                .matvec_threaded(xb, v, *threads, &mut scratch[..c.embedding])?;

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
            for head in 0..c.heads {
                // Grouped-query attention: several query heads share one KV head.
                let kv_head = head / group;
                let query = &q[head * head_dim..(head + 1) * head_dim];

                for (past, score) in att[..=position].iter_mut().enumerate() {
                    let at = (index * *context + past) * kv_dim + kv_head * head_dim;
                    *score = ops::dot(query, &key_cache[at..at + head_dim]) * scale;
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

            layer
                .wo
                .matvec_threaded(xb, xb2, *threads, &mut scratch[..c.embedding])?;
            for (slot, delta) in x.iter_mut().zip(xb2.iter()) {
                *slot += delta;
            }

            ops::rms_norm(x, &layer.ffn_norm, c.eps, xb);
            layer
                .gate
                .matvec_threaded(xb, hb, *threads, &mut scratch[..c.embedding])?;
            layer
                .up
                .matvec_threaded(xb, hb2, *threads, &mut scratch[..c.embedding])?;
            // SwiGLU: the gate is squashed, the up projection is not.
            for (gate, up) in hb.iter_mut().zip(hb2.iter()) {
                *gate = ops::silu(*gate) * up;
            }
            layer
                .down
                .matvec_threaded(hb, xb2, *threads, &mut scratch[..c.ffn])?;
            for (slot, delta) in x.iter_mut().zip(xb2.iter()) {
                *slot += delta;
            }
        }

        ops::rms_norm(x, &self.output_norm, c.eps, xb);
        self.output
            .matvec(xb, logits, &mut scratch[..c.embedding])?;
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
    scratch: Vec<f32>,
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
        Self {
            x: vec![0.0; config.embedding],
            xb: vec![0.0; config.embedding],
            xb2: vec![0.0; config.embedding],
            hb: vec![0.0; config.ffn],
            hb2: vec![0.0; config.ffn],
            q: vec![0.0; config.heads * config.head_dim],
            k: vec![0.0; config.kv_dim()],
            v: vec![0.0; config.kv_dim()],
            att: vec![0.0; context],
            logits: vec![0.0; config.vocab],
            scratch: vec![0.0; config.embedding.max(config.ffn)],
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

    /// Bytes held by the KV cache, which dominates everything else here.
    pub fn cache_bytes(&self) -> usize {
        (self.key_cache.len() + self.value_cache.len()) * std::mem::size_of::<f32>()
    }
}
