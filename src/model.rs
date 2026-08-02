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
}
