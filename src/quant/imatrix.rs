//! Importance Matrix (`imatrix`) calibration and importance-weighted quantization.
//!
//! When quantizing weights to ultra-low bitrates (`IQ4_NL`, `IQ4_XS`, `Q4_K`,
//! `Q2_K`, `TQ2_0`), uniform mean-squared-error minimization treats all input
//! channels equally:
//!
//! $$\mathcal{L}_{\text{uniform}} = \sum_{j} (w_j - \hat{w}_j)^2$$
//!
//! In practice, activation channels have wildly different magnitudes (outlier
//! channels can be $100\times$ larger than typical channels). An **Importance
//! Matrix** records the expected squared activation per input channel:
//!
//! $$I_j = \frac{1}{T} \sum_{t=1}^{T} x_{t,j}^2 + \epsilon$$
//!
//! and minimizes the expected output perturbation of the linear layer:
//!
//! $$\mathcal{L}_{\text{imatrix}} = \sum_{j} I_j \cdot (w_j - \hat{w}_j)^2$$

use std::collections::BTreeMap;
use std::io::{Error, ErrorKind, Result};

/// Per-tensor channel importance profiles collected during calibration.
#[derive(Clone, Debug, Default)]
pub struct ImportanceMatrix {
    /// Map from tensor name (e.g. `"blk.0.attn_q.weight"`) to per-column $\sum x_j^2$.
    pub sums_sq: BTreeMap<String, Vec<f32>>,
    /// Number of token vectors accumulated per tensor.
    pub counts: BTreeMap<String, usize>,
}

const IMATRIX_MAGIC: &[u8; 4] = b"FIMX";

impl ImportanceMatrix {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe an activation batch `x` of shape `[tokens, cols]` entering linear weight `tensor_name`.
    pub fn observe(&mut self, tensor_name: &str, x: &[f32], cols: usize) {
        if cols == 0 || x.len() % cols != 0 {
            return;
        }
        let tokens = x.len() / cols;
        let entry = self
            .sums_sq
            .entry(tensor_name.to_string())
            .or_insert_with(|| vec![0.0; cols]);
        if entry.len() != cols {
            entry.resize(cols, 0.0);
        }
        for row in x.chunks_exact(cols) {
            for (slot, &val) in entry.iter_mut().zip(row) {
                *slot += val * val;
            }
        }
        *self.counts.entry(tensor_name.to_string()).or_insert(0) += tokens;
    }

    /// Normalized importance weights $\bar{I}_j = \frac{I_j}{\text{mean}(I)} + 10^{-5}$ for `tensor_name`.
    pub fn normalized_weights(&self, tensor_name: &str) -> Option<Vec<f32>> {
        let sq = self.sums_sq.get(tensor_name)?;
        let count = (*self.counts.get(tensor_name).unwrap_or(&1)).max(1) as f32;
        if sq.is_empty() {
            return None;
        }
        let mean_sq = (sq.iter().sum::<f32>() / (sq.len() as f32 * count)).max(1e-12);
        Some(
            sq.iter()
                .map(|&s| ((s / count) / mean_sq).clamp(1e-4, 1e4))
                .collect(),
        )
    }

    /// Serialize the `ImportanceMatrix` to a compact binary buffer (`FIMX` magic).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(IMATRIX_MAGIC);
        out.extend_from_slice(&(self.sums_sq.len() as u32).to_le_bytes());
        for (name, sq) in &self.sums_sq {
            let count = *self.counts.get(name).unwrap_or(&1) as u32;
            let name_bytes = name.as_bytes();
            out.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(&count.to_le_bytes());
            out.extend_from_slice(&(sq.len() as u32).to_le_bytes());
            for &v in sq {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Deserialize an `ImportanceMatrix` from a binary buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 || &bytes[0..4] != IMATRIX_MAGIC {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "invalid imatrix magic header",
            ));
        }
        let num_entries = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let mut cursor = 8usize;
        let mut imatrix = Self::new();

        for _ in 0..num_entries {
            if cursor + 4 > bytes.len() {
                return Err(Error::new(ErrorKind::UnexpectedEof, "truncated imatrix"));
            }
            let name_len =
                u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            if cursor + name_len + 8 > bytes.len() {
                return Err(Error::new(ErrorKind::UnexpectedEof, "truncated imatrix"));
            }
            let name = String::from_utf8(bytes[cursor..cursor + name_len].to_vec())
                .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            cursor += name_len;
            let count = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            let cols =
                u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
            cursor += 8;
            if cursor + cols * 4 > bytes.len() {
                return Err(Error::new(ErrorKind::UnexpectedEof, "truncated imatrix"));
            }
            let mut sq = Vec::with_capacity(cols);
            for c in 0..cols {
                let off = cursor + c * 4;
                sq.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
            }
            cursor += cols * 4;
            imatrix.sums_sq.insert(name.clone(), sq);
            imatrix.counts.insert(name, count);
        }
        Ok(imatrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imatrix_accumulation_and_serialization_roundtrip() {
        let mut im = ImportanceMatrix::new();
        // Channel 0 has high magnitude (10.0), channel 1..4 have small magnitude (1.0)
        let batch = vec![10.0, 1.0, 1.0, 1.0, -10.0, 1.0, -1.0, 1.0];
        im.observe("blk.0.attn_q.weight", &batch, 4);

        let norm = im.normalized_weights("blk.0.attn_q.weight").unwrap();
        assert_eq!(norm.len(), 4);
        assert!(
            norm[0] > norm[1] * 50.0,
            "outlier channel must have higher importance"
        );

        let bytes = im.to_bytes();
        let loaded = ImportanceMatrix::from_bytes(&bytes).unwrap();
        let loaded_norm = loaded.normalized_weights("blk.0.attn_q.weight").unwrap();
        assert_eq!(norm, loaded_norm);
    }
}
