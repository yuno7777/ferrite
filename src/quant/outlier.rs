//! Outlier-Aware Mixed-Precision Quantization (W8A8 / W4A16 FP16 outlier channel preservation).
//!
//! Preserves high-magnitude outlier feature channels in FP16 while quantizing the remaining
//! bulk channels to INT8 (W8A8) or INT4 (W4A16). In transformer models, outlier channels
//! represent a tiny fraction (typically 0.1% to 1%) of total dimensions, but account for
//! the vast majority of quantization error if forced into low-bit representation.

use std::collections::BTreeSet;
use std::io::{Error, ErrorKind, Result};

use crate::quant::half;

fn bad(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, msg.into())
}

/// Quantization precision mode for non-outlier base weights and activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutlierQuantMode {
    /// W8A8: INT8 quantized base weights with INT8 quantized activations, FP16 outliers.
    W8A8,
    /// W4A16: INT4 quantized base weights with FP16/F32 activations, FP16 outliers.
    W4A16,
}

/// Configuration for outlier channel detection and mixed-precision quantization.
#[derive(Clone, Copy, Debug)]
pub struct OutlierConfig {
    pub mode: OutlierQuantMode,
    /// Multiplier over average column magnitude to classify a channel as an outlier.
    pub threshold_multiplier: f32,
    /// Maximum percentage of columns that can be classified as outliers (e.g. 0.05 for 5%).
    pub max_outlier_ratio: f32,
    /// Minimum absolute magnitude for a column to qualify as outlier.
    pub min_magnitude: f32,
}

impl Default for OutlierConfig {
    fn default() -> Self {
        Self {
            mode: OutlierQuantMode::W8A8,
            threshold_multiplier: 3.0,
            max_outlier_ratio: 0.05,
            min_magnitude: 1e-4,
        }
    }
}

/// An outlier-aware mixed-precision quantized weight matrix.
///
/// Stores bulk channels in low-bit quantized form while preserving outlier channels
/// in IEEE 754 binary16 (FP16).
#[derive(Clone, Debug)]
pub struct OutlierMatrix {
    pub rows: usize,
    pub cols: usize,
    pub mode: OutlierQuantMode,
    /// Indices of outlier columns (strictly sorted).
    pub outlier_channels: Vec<usize>,
    /// FP16 representation of outlier weights: shape [rows, outlier_channels.len()].
    pub outlier_weights_f16: Vec<u16>,
    /// Quantized base weights.
    /// - For W8A8: signed 8-bit integers (`i8` reinterpreted as `u8`) of shape `[rows, cols]`.
    /// - For W4A16: packed 4-bit nibbles of shape `[rows, (cols + 1) / 2]`.
    pub base_quantized: Vec<u8>,
    /// Per-row scale factors for base quantized weights: shape `[rows]`.
    pub base_scales: Vec<f32>,
}

impl OutlierMatrix {
    /// Quantize a row-major `[rows, cols]` F32 weight matrix with outlier preservation.
    pub fn from_weights(
        weights: &[f32],
        rows: usize,
        cols: usize,
        config: OutlierConfig,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(bad("matrix dimensions must be non-zero"));
        }
        if weights.len() != rows * cols {
            return Err(bad(format!(
                "weights length {} != rows * cols ({} * {})",
                weights.len(),
                rows,
                cols
            )));
        }

        // 1. Identify outlier columns by calculating maximum column magnitude
        let mut col_maxes = vec![0.0f32; cols];
        for r in 0..rows {
            let row_offset = r * cols;
            for c in 0..cols {
                let mag = weights[row_offset + c].abs();
                if mag > col_maxes[c] {
                    col_maxes[c] = mag;
                }
            }
        }

        let sum_max: f32 = col_maxes.iter().copied().sum();
        let avg_max = sum_max / (cols as f32);
        let threshold = (avg_max * config.threshold_multiplier).max(config.min_magnitude);

        // Find candidate outlier columns exceeding threshold
        let mut candidates: Vec<(usize, f32)> = col_maxes
            .iter()
            .enumerate()
            .filter(|&(_, &mag)| mag >= threshold)
            .map(|(c, &mag)| (c, mag))
            .collect();

        // Sort descending by magnitude
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let max_allowed = ((cols as f32) * config.max_outlier_ratio).ceil() as usize;
        candidates.truncate(max_allowed);

        // Sort selected outlier column indices in ascending order
        let mut outlier_channels: Vec<usize> = candidates.into_iter().map(|(c, _)| c).collect();
        outlier_channels.sort_unstable();

        let outlier_set: BTreeSet<usize> = outlier_channels.iter().copied().collect();
        let num_outliers = outlier_channels.len();

        // 2. Extract and store outlier weights in FP16
        let mut outlier_weights_f16 = Vec::with_capacity(rows * num_outliers);
        for r in 0..rows {
            let row_offset = r * cols;
            for &c in &outlier_channels {
                let w = weights[row_offset + c];
                outlier_weights_f16.push(half::f32_to_f16(w));
            }
        }

        // 3. Quantize base non-outlier channels per row
        let mut base_scales = Vec::with_capacity(rows);
        let base_row_bytes = match config.mode {
            OutlierQuantMode::W8A8 => cols,
            OutlierQuantMode::W4A16 => cols.div_ceil(2),
        };
        let mut base_quantized = vec![0u8; rows * base_row_bytes];

        for r in 0..rows {
            let row_offset = r * cols;
            let mut row_max = 0.0f32;
            for c in 0..cols {
                if !outlier_set.contains(&c) {
                    let mag = weights[row_offset + c].abs();
                    if mag > row_max {
                        row_max = mag;
                    }
                }
            }

            match config.mode {
                OutlierQuantMode::W8A8 => {
                    let scale = if row_max > 0.0 { row_max / 127.0 } else { 1.0 };
                    base_scales.push(scale);
                    let inv_scale = 1.0 / scale;
                    let out_row = &mut base_quantized[r * cols..(r + 1) * cols];

                    for c in 0..cols {
                        if outlier_set.contains(&c) {
                            out_row[c] = 0;
                        } else {
                            let quantized = (weights[row_offset + c] * inv_scale)
                                .round()
                                .clamp(-127.0, 127.0)
                                as i8;
                            out_row[c] = quantized as u8;
                        }
                    }
                }
                OutlierQuantMode::W4A16 => {
                    let scale = if row_max > 0.0 { row_max / 7.0 } else { 1.0 };
                    base_scales.push(scale);
                    let inv_scale = 1.0 / scale;
                    let row_dest =
                        &mut base_quantized[r * base_row_bytes..(r + 1) * base_row_bytes];

                    for (byte_idx, slot) in row_dest.iter_mut().enumerate() {
                        let c0 = byte_idx * 2;
                        let c1 = c0 + 1;

                        let q0 = if c0 < cols && !outlier_set.contains(&c0) {
                            (weights[row_offset + c0] * inv_scale)
                                .round()
                                .clamp(-7.0, 7.0) as i8
                        } else {
                            0
                        };

                        let q1 = if c1 < cols && !outlier_set.contains(&c1) {
                            (weights[row_offset + c1] * inv_scale)
                                .round()
                                .clamp(-7.0, 7.0) as i8
                        } else {
                            0
                        };

                        let nibble0 = (q0 & 0x0F) as u8;
                        let nibble1 = (q1 & 0x0F) as u8;
                        *slot = nibble0 | (nibble1 << 4);
                    }
                }
            }
        }

        Ok(Self {
            rows,
            cols,
            mode: config.mode,
            outlier_channels,
            outlier_weights_f16,
            base_quantized,
            base_scales,
        })
    }

    /// Multiply matrix by input vector `x`, accumulating output into `out`.
    ///
    /// `x` has length `cols`, `out` has length `rows`.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != self.cols {
            return Err(bad(format!(
                "input vector length {} != cols {}",
                x.len(),
                self.cols
            )));
        }
        if out.len() != self.rows {
            return Err(bad(format!(
                "output vector length {} != rows {}",
                out.len(),
                self.rows
            )));
        }

        let num_outliers = self.outlier_channels.len();

        match self.mode {
            OutlierQuantMode::W8A8 => {
                // Determine activation scale for INT8 activation quantization
                let mut x_max = 0.0f32;
                for (c, &val) in x.iter().enumerate() {
                    if !self.outlier_channels.binary_search(&c).is_ok() {
                        let mag = val.abs();
                        if mag > x_max {
                            x_max = mag;
                        }
                    }
                }
                let act_scale = if x_max > 0.0 { x_max / 127.0 } else { 1.0 };
                let inv_act_scale = 1.0 / act_scale;

                // Quantize non-outlier activations to i8
                let mut x_quant = vec![0i8; self.cols];
                for (c, slot) in x_quant.iter_mut().enumerate() {
                    if self.outlier_channels.binary_search(&c).is_err() {
                        *slot = (x[c] * inv_act_scale).round().clamp(-127.0, 127.0) as i8;
                    }
                }

                for (r, out_slot) in out.iter_mut().enumerate() {
                    // Integer dot product for base channels
                    let base_row = &self.base_quantized[r * self.cols..(r + 1) * self.cols];
                    let mut sum_int: i32 = 0;
                    for c in 0..self.cols {
                        let w_i8 = base_row[c] as i8;
                        let a_i8 = x_quant[c];
                        sum_int += (w_i8 as i32) * (a_i8 as i32);
                    }
                    let base_result = (sum_int as f32) * self.base_scales[r] * act_scale;

                    // Exact FP16 accumulation for outlier channels
                    let mut sum_outlier = 0.0f32;
                    let outlier_row =
                        &self.outlier_weights_f16[r * num_outliers..(r + 1) * num_outliers];
                    for (idx, &c) in self.outlier_channels.iter().enumerate() {
                        let w = half::f16(outlier_row[idx]);
                        sum_outlier += w * x[c];
                    }

                    *out_slot = base_result + sum_outlier;
                }
            }
            OutlierQuantMode::W4A16 => {
                let base_row_bytes = self.cols.div_ceil(2);

                for (r, out_slot) in out.iter_mut().enumerate() {
                    let base_row =
                        &self.base_quantized[r * base_row_bytes..(r + 1) * base_row_bytes];
                    let mut sum_base = 0.0f32;

                    for (byte_idx, &byte) in base_row.iter().enumerate() {
                        let c0 = byte_idx * 2;
                        let c1 = c0 + 1;

                        // Sign-extend 4-bit nibbles
                        let mut n0 = (byte & 0x0F) as i8;
                        if n0 >= 8 {
                            n0 -= 16;
                        }
                        if c0 < self.cols {
                            sum_base += (n0 as f32) * x[c0];
                        }

                        let mut n1 = ((byte >> 4) & 0x0F) as i8;
                        if n1 >= 8 {
                            n1 -= 16;
                        }
                        if c1 < self.cols {
                            sum_base += (n1 as f32) * x[c1];
                        }
                    }
                    sum_base *= self.base_scales[r];

                    // Exact FP16 accumulation for outlier channels
                    let mut sum_outlier = 0.0f32;
                    let outlier_row =
                        &self.outlier_weights_f16[r * num_outliers..(r + 1) * num_outliers];
                    for (idx, &c) in self.outlier_channels.iter().enumerate() {
                        let w = half::f16(outlier_row[idx]);
                        sum_outlier += w * x[c];
                    }

                    *out_slot = sum_base + sum_outlier;
                }
            }
        }

        Ok(())
    }

    /// Dequantize the full `[rows, cols]` matrix into 32-bit floats.
    pub fn dequantize(&self, out: &mut [f32]) -> Result<()> {
        if out.len() != self.rows * self.cols {
            return Err(bad("output slice size does not match matrix dimensions"));
        }

        let num_outliers = self.outlier_channels.len();

        match self.mode {
            OutlierQuantMode::W8A8 => {
                for r in 0..self.rows {
                    let base_row = &self.base_quantized[r * self.cols..(r + 1) * self.cols];
                    let scale = self.base_scales[r];
                    let out_row = &mut out[r * self.cols..(r + 1) * self.cols];

                    for c in 0..self.cols {
                        let w_i8 = base_row[c] as i8;
                        out_row[c] = (w_i8 as f32) * scale;
                    }

                    let outlier_row =
                        &self.outlier_weights_f16[r * num_outliers..(r + 1) * num_outliers];
                    for (idx, &c) in self.outlier_channels.iter().enumerate() {
                        out_row[c] = half::f16(outlier_row[idx]);
                    }
                }
            }
            OutlierQuantMode::W4A16 => {
                let base_row_bytes = self.cols.div_ceil(2);

                for r in 0..self.rows {
                    let base_row =
                        &self.base_quantized[r * base_row_bytes..(r + 1) * base_row_bytes];
                    let scale = self.base_scales[r];
                    let out_row = &mut out[r * self.cols..(r + 1) * self.cols];

                    for (byte_idx, &byte) in base_row.iter().enumerate() {
                        let c0 = byte_idx * 2;
                        let c1 = c0 + 1;

                        let mut n0 = (byte & 0x0F) as i8;
                        if n0 >= 8 {
                            n0 -= 16;
                        }
                        if c0 < self.cols {
                            out_row[c0] = (n0 as f32) * scale;
                        }

                        let mut n1 = ((byte >> 4) & 0x0F) as i8;
                        if n1 >= 8 {
                            n1 -= 16;
                        }
                        if c1 < self.cols {
                            out_row[c1] = (n1 as f32) * scale;
                        }
                    }

                    let outlier_row =
                        &self.outlier_weights_f16[r * num_outliers..(r + 1) * num_outliers];
                    for (idx, &c) in self.outlier_channels.iter().enumerate() {
                        out_row[c] = half::f16(outlier_row[idx]);
                    }
                }
            }
        }

        Ok(())
    }

    /// Ratio of channels classified and preserved as outliers in FP16.
    pub fn outlier_ratio(&self) -> f32 {
        self.outlier_channels.len() as f32 / self.cols as f32
    }
}
