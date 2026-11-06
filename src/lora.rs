//! Low-Rank Adaptation (LoRA) for parameter-efficient adapter application.
//!
//! Applies low-rank additive weight updates $\Delta W = \frac{\alpha}{r} (B \times A)$
//! at runtime during single-token decode and batched prefill without duplicating
//! or mutating base model weights.

use std::io::{Error, ErrorKind, Result};

use crate::gguf::Gguf;
use crate::model::Weight;

fn bad(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, message.into())
}

/// A low-rank adapter pair `(A, B)` for a single projection.
///
/// In GGUF LoRA tensors:
/// - `A` has shape `[in_dim, rank]`, where `cols = in_dim`, `rows = rank`.
/// - `B` has shape `[rank, out_dim]`, where `cols = rank`, `rows = out_dim`.
///
/// Activation update is:
/// `out += (alpha / rank) * (B * (A * x))`
pub struct LoraPair<'a> {
    pub a: Weight<'a>,
    pub b: Weight<'a>,
    pub rank: usize,
    pub scale: f32,
}

impl<'a> LoraPair<'a> {
    /// Bind a LoRA pair and compute the scaling factor `alpha / rank`.
    pub fn new(a: Weight<'a>, b: Weight<'a>, alpha: Option<f32>) -> Result<Self> {
        if a.rows != b.cols {
            return Err(bad(format!(
                "LoRA dimension mismatch: A rows ({}) != B cols ({})",
                a.rows, b.cols
            )));
        }
        let rank = a.rows;
        if rank == 0 {
            return Err(bad("LoRA rank cannot be zero"));
        }
        let alpha_val = alpha.unwrap_or(rank as f32);
        let scale = alpha_val / rank as f32;
        Ok(Self { a, b, rank, scale })
    }

    /// Single-token decode delta: `out += scale * (B * (A * x))`.
    pub fn apply(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != self.a.cols {
            return Err(bad(format!(
                "LoRA input length mismatch: expected {}, got {}",
                self.a.cols,
                x.len()
            )));
        }
        if out.len() != self.b.rows {
            return Err(bad(format!(
                "LoRA output length mismatch: expected {}, got {}",
                self.b.rows,
                out.len()
            )));
        }

        let mut r_buf = vec![0.0f32; self.rank];
        let mut scratch_a = vec![0.0f32; self.a.cols];
        self.a.matvec(x, &mut r_buf, &mut scratch_a)?;

        let mut delta = vec![0.0f32; self.b.rows];
        let mut scratch_b = vec![0.0f32; self.b.cols];
        self.b.matvec(&r_buf, &mut delta, &mut scratch_b)?;

        for (slot, &d) in out.iter_mut().zip(&delta) {
            *slot += d * self.scale;
        }
        Ok(())
    }

    /// Batched prefill delta: `out += scale * (B * (A * x))`.
    pub fn apply_batched(&self, x: &[f32], batch: usize, out: &mut [f32]) -> Result<()> {
        if x.len() != batch * self.a.cols {
            return Err(bad(format!(
                "LoRA batched input length mismatch: expected {}, got {}",
                batch * self.a.cols,
                x.len()
            )));
        }
        if out.len() != batch * self.b.rows {
            return Err(bad(format!(
                "LoRA batched output length mismatch: expected {}, got {}",
                batch * self.b.rows,
                out.len()
            )));
        }

        let mut r_batch = vec![0.0f32; batch * self.rank];
        let mut scratch_a = vec![0.0f32; self.a.cols];
        self.a.matmul(x, &mut r_batch, batch, &mut scratch_a)?;

        let mut delta_batch = vec![0.0f32; batch * self.b.rows];
        let mut scratch_b = vec![0.0f32; self.b.cols];
        self.b
            .matmul(&r_batch, &mut delta_batch, batch, &mut scratch_b)?;

        for (slot, &d) in out.iter_mut().zip(&delta_batch) {
            *slot += d * self.scale;
        }
        Ok(())
    }

    /// Batched prefill delta with thread parallelism: `out += scale * (B * (A * x))`.
    pub fn apply_batched_threaded(
        &self,
        x: &[f32],
        batch: usize,
        out: &mut [f32],
        threads: usize,
    ) -> Result<()> {
        if x.len() != batch * self.a.cols {
            return Err(bad(format!(
                "LoRA batched input length mismatch: expected {}, got {}",
                batch * self.a.cols,
                x.len()
            )));
        }
        if out.len() != batch * self.b.rows {
            return Err(bad(format!(
                "LoRA batched output length mismatch: expected {}, got {}",
                batch * self.b.rows,
                out.len()
            )));
        }

        let mut r_batch = vec![0.0f32; batch * self.rank];
        let mut ws_a = crate::model::Workspace::new(self.a.cols);
        self.a
            .matmul_threaded(x, &mut r_batch, batch, threads, &mut ws_a)?;

        let mut delta_batch = vec![0.0f32; batch * self.b.rows];
        let mut ws_b = crate::model::Workspace::new(self.b.cols);
        self.b
            .matmul_threaded(&r_batch, &mut delta_batch, batch, threads, &mut ws_b)?;

        for (slot, &d) in out.iter_mut().zip(&delta_batch) {
            *slot += d * self.scale;
        }
        Ok(())
    }

    /// Dequantize and accumulate low-rank delta directly into an f32 weight matrix:
    /// `matrix[r * cols + c] += scale * sum_k (B[r, k] * A[k, c])`.
    pub fn accumulate_into(&self, matrix: &mut [f32]) -> Result<()> {
        let (rows, cols) = (self.b.rows, self.a.cols);
        if matrix.len() != rows * cols {
            return Err(bad(format!(
                "accumulate_into: target matrix length {} does not match {}x{}",
                matrix.len(),
                rows,
                cols
            )));
        }

        let mut a_dense = vec![0.0f32; self.rank * cols];
        for k in 0..self.rank {
            self.a.row(k, &mut a_dense[k * cols..(k + 1) * cols])?;
        }

        let mut b_row = vec![0.0f32; self.rank];
        for r in 0..rows {
            self.b.row(r, &mut b_row)?;
            let row_offset = r * cols;
            for k in 0..self.rank {
                let bk = b_row[k] * self.scale;
                if bk == 0.0 {
                    continue;
                }
                let a_row = &a_dense[k * cols..(k + 1) * cols];
                for (c, &a_val) in a_row.iter().enumerate() {
                    matrix[row_offset + c] += bk * a_val;
                }
            }
        }
        Ok(())
    }
}

/// A parsed LoRA adapter container loaded from a GGUF file.
pub struct LoraAdapter<'a> {
    pub alpha: Option<f32>,
    pub pairs: Vec<(String, LoraPair<'a>)>,
}

impl<'a> LoraAdapter<'a> {
    /// Load all LoRA pairs present in the GGUF file.
    pub fn load(gguf: &'a Gguf, override_alpha: Option<f32>) -> Result<Self> {
        let alpha = override_alpha
            .or_else(|| gguf.meta_f32("adapter.lora.alpha"))
            .or_else(|| gguf.meta_f32("lora.alpha"))
            .or_else(|| {
                gguf.arch()
                    .and_then(|a| gguf.meta_f32(&format!("{a}.lora.alpha")))
            });

        let mut pairs = Vec::new();

        // Scan all tensors for lora_a / loraA
        for tensor in &gguf.tensors {
            let name = &tensor.name;
            let (base, a_name, b_name) = if let Some(stripped) = name.strip_suffix(".lora_a") {
                (stripped, name.as_str(), format!("{stripped}.lora_b"))
            } else if let Some(stripped) = name.strip_suffix(".loraA") {
                (stripped, name.as_str(), format!("{stripped}.loraB"))
            } else {
                continue;
            };

            // Verify companion b tensor exists in gguf
            if gguf.tensor(&b_name).is_none() {
                continue;
            }

            let a = Weight::bind(gguf, a_name)?;
            let b = Weight::bind(gguf, &b_name)?;
            let pair = LoraPair::new(a, b, alpha)?;
            pairs.push((base.to_string(), pair));
        }

        if pairs.is_empty() {
            return Err(bad("no valid LoRA tensor pairs found in GGUF file"));
        }

        Ok(Self { alpha, pairs })
    }

    /// Find a pair matching `base_name` or variations like `base_name.weight`.
    pub fn get(&self, base_name: &str) -> Option<&LoraPair<'a>> {
        self.pairs
            .iter()
            .find(|(name, _)| {
                name == base_name
                    || name == &format!("{base_name}.weight")
                    || base_name == format!("{name}.weight")
            })
            .map(|(_, p)| p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgmlType, Value};
    use crate::synth::Builder;

    fn bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn lora_pair_apply_matches_dense_matrix_multiplication() {
        // Input dim = 4, rank = 2, output dim = 3
        // A: 2 rows of 4 cols
        let a_vals = vec![
            1.0, 2.0, 0.0, -1.0, // row 0
            0.5, 0.0, 1.5, 2.0, // row 1
        ];
        // B: 3 rows of 2 cols
        let b_vals = vec![
            2.0, -1.0, // row 0
            1.0, 3.0, // row 1
            0.0, 2.0, // row 2
        ];
        let alpha = 4.0; // scale = 4.0 / 2 = 2.0

        let builder = Builder::new()
            .meta("general.architecture", Value::String("llama".into()))
            .meta("adapter.lora.alpha", Value::F32(alpha))
            .tensor(
                "blk.0.attn_q.weight.lora_a",
                &[4, 2],
                GgmlType::F32,
                bytes(&a_vals),
            )
            .tensor(
                "blk.0.attn_q.weight.lora_b",
                &[2, 3],
                GgmlType::F32,
                bytes(&b_vals),
            );

        let path = std::env::temp_dir().join("ferrite_lora_test.gguf");
        builder.write(&path).expect("write");

        let gguf = Gguf::open(&path).expect("open");
        let adapter = LoraAdapter::load(&gguf, None).expect("load");
        assert_eq!(adapter.pairs.len(), 1);

        let pair = adapter.get("blk.0.attn_q.weight").expect("found pair");
        assert_eq!(pair.rank, 2);
        assert!((pair.scale - 2.0).abs() < 1e-6);

        let x = [1.0f32, -1.0, 2.0, 0.5];
        // Intermediate r:
        // r[0] = 1*1 + (-1)*2 + 2*0 + 0.5*(-1) = 1 - 2 - 0.5 = -1.5
        // r[1] = 1*0.5 + (-1)*0 + 2*1.5 + 0.5*2 = 0.5 + 3.0 + 1.0 = 4.5
        // Delta = B * r:
        // d[0] = -1.5*2 + 4.5*(-1) = -3 - 4.5 = -7.5
        // d[1] = -1.5*1 + 4.5*3 = -1.5 + 13.5 = 12.0
        // d[2] = -1.5*0 + 4.5*2 = 9.0
        // scaled by 2.0:
        // expected: [-15.0, 24.0, 18.0]
        let mut out = [10.0f32, 10.0, 10.0];
        pair.apply(&x, &mut out).expect("apply");

        assert!((out[0] - (10.0 - 15.0)).abs() < 1e-5);
        assert!((out[1] - (10.0 + 24.0)).abs() < 1e-5);
        assert!((out[2] - (10.0 + 18.0)).abs() < 1e-5);

        let inputs = [x.as_slice(), &[0.0, 1.0, -1.0, 2.0]].concat();
        let mut batched = [0.0f32; 6];
        pair.apply_batched(&inputs, 2, &mut batched)
            .expect("batched apply");
        let mut single = [0.0f32; 3];
        pair.apply(&inputs[4..], &mut single).expect("single apply");
        for (actual, expected) in batched[3..].iter().zip(single) {
            assert!((actual - expected).abs() < 1e-5);
        }

        // Test accumulate_into
        let mut dense_delta = vec![0.0f32; 3 * 4];
        pair.accumulate_into(&mut dense_delta)
            .expect("accumulate_into");
        // Verify dense_delta * x == scaled delta
        let mut dense_out = [0.0f32; 3];
        for r in 0..3 {
            let row = &dense_delta[r * 4..(r + 1) * 4];
            dense_out[r] = crate::ops::dot(row, &x);
        }
        assert!((dense_out[0] - (-15.0)).abs() < 1e-5);
        assert!((dense_out[1] - 24.0).abs() < 1e-5);
        assert!((dense_out[2] - 18.0).abs() < 1e-5);

        std::fs::remove_file(path).ok();
    }
}
