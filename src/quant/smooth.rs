//! SmoothQuant offline activation-to-weight scale migration for W8A8 quantization.
//!
//! SmoothQuant addresses activation outliers in large language models by migrating
//! dynamic range difficulty from activations into weights using mathematically equivalent
//! per-channel scaling:
//!
//!   Y = (X * diag(s)^-1) * (diag(s) * W) = X_smooth * W_smooth
//!
//! where s_j = max(|X_j|)^alpha / max(|W_j|)^(1 - alpha).

use std::io::{Error, ErrorKind, Result};

/// Hyperparameters for SmoothQuant scale migration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothQuantConfig {
    /// Migration strength alpha in [0.0, 1.0].
    /// alpha = 0.5 balances activation and weight quantization difficulty equally.
    pub alpha: f32,
    /// Minimum scale floor to prevent division by zero.
    pub epsilon: f32,
}

impl Default for SmoothQuantConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            epsilon: 1e-5,
        }
    }
}

/// Computes per-column maximum absolute value across all rows of a matrix.
pub fn compute_weight_channel_max(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(weights.len(), rows * cols);
    let mut channel_max = vec![0.0f32; cols];
    for r in 0..rows {
        let row_offset = r * cols;
        for c in 0..cols {
            let val = weights[row_offset + c].abs();
            if val > channel_max[c] {
                channel_max[c] = val;
            }
        }
    }
    channel_max
}

/// Computes smoothing scale factors s_j for each input feature channel.
pub fn compute_smoothing_scales(
    act_channel_max: &[f32],
    weight_channel_max: &[f32],
    config: &SmoothQuantConfig,
) -> Vec<f32> {
    assert_eq!(
        act_channel_max.len(),
        weight_channel_max.len(),
        "channel dimensions must match"
    );
    let n = act_channel_max.len();
    let mut scales = Vec::with_capacity(n);
    let alpha = config.alpha.clamp(0.0, 1.0);
    let beta = 1.0 - alpha;
    let eps = config.epsilon.max(1e-9);

    for i in 0..n {
        let act_max = act_channel_max[i].max(eps);
        let wt_max = weight_channel_max[i].max(eps);
        let s = (act_max.powf(alpha) / wt_max.powf(beta)).clamp(1e-4, 1e4);
        scales.push(s);
    }
    scales
}

/// In-place scales weights by smoothing factors: W_ij <- W_ij * s_j.
pub fn smooth_weights(weights: &mut [f32], rows: usize, cols: usize, scales: &[f32]) {
    assert_eq!(weights.len(), rows * cols);
    assert_eq!(scales.len(), cols);
    for r in 0..rows {
        let row_offset = r * cols;
        for c in 0..cols {
            weights[row_offset + c] *= scales[c];
        }
    }
}

/// In-place scales activations by inverse smoothing factors: X_j <- X_j / s_j.
pub fn smooth_activations(activations: &mut [f32], scales: &[f32]) {
    assert_eq!(activations.len(), scales.len());
    for (act, &s) in activations.iter_mut().zip(scales) {
        if s.abs() > 1e-9 {
            *act /= s;
        }
    }
}

/// Linear projection layer quantized to W8A8 with SmoothQuant scale migration.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothQuantLinear {
    pub rows: usize,
    pub cols: usize,
    /// Symmetric INT8 quantized weights (rows * cols).
    pub qweight: Vec<i8>,
    /// Per-row weight quantization scales.
    pub row_scales: Vec<f32>,
    /// Per-channel activation smoothing scales s_j.
    pub smoothing_scales: Vec<f32>,
}

impl SmoothQuantLinear {
    /// Creates and quantizes a linear layer with offline SmoothQuant migration.
    pub fn from_f32(
        weights: &[f32],
        rows: usize,
        cols: usize,
        act_channel_max: &[f32],
        config: SmoothQuantConfig,
    ) -> Result<Self> {
        if weights.len() != rows * cols {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "weight length does not match rows * cols",
            ));
        }
        if act_channel_max.len() != cols {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "activation channel count does not match cols",
            ));
        }

        let wt_channel_max = compute_weight_channel_max(weights, rows, cols);
        let smoothing_scales = compute_smoothing_scales(act_channel_max, &wt_channel_max, &config);

        // Apply scale migration to weights: W_smooth = W * diag(s)
        let mut smoothed_weights = weights.to_vec();
        smooth_weights(&mut smoothed_weights, rows, cols, &smoothing_scales);

        // Quantize smoothed weights to symmetric INT8 with per-row scales
        let mut qweight = vec![0i8; rows * cols];
        let mut row_scales = vec![0.0f32; rows];

        for (r, row_scale) in row_scales.iter_mut().enumerate() {
            let row_offset = r * cols;
            let row_slice = &smoothed_weights[row_offset..row_offset + cols];
            let max_val = row_slice
                .iter()
                .copied()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = if max_val > 0.0 { max_val / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;
            *row_scale = scale;

            for c in 0..cols {
                let scaled = (row_slice[c] * inv_scale).round();
                qweight[row_offset + c] = scaled.clamp(-128.0, 127.0) as i8;
            }
        }

        Ok(Self {
            rows,
            cols,
            qweight,
            row_scales,
            smoothing_scales,
        })
    }

    /// Dequantizes the smoothed weights back to f32 (undoing row quantization and scale migration).
    pub fn dequantize(&self, out: &mut [f32]) -> Result<()> {
        if out.len() != self.rows * self.cols {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "output slice length does not match rows * cols",
            ));
        }

        for r in 0..self.rows {
            let row_offset = r * self.cols;
            let scale = self.row_scales[r];
            for c in 0..self.cols {
                let smoothed_val = (self.qweight[row_offset + c] as f32) * scale;
                // Original weight W_ij = W_smooth_ij / s_j
                out[row_offset + c] = smoothed_val / self.smoothing_scales[c];
            }
        }
        Ok(())
    }

    /// Evaluates matrix-vector multiplication Y = X * W with SmoothQuant W8A8 quantization.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != self.cols || out.len() != self.rows {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "dimension mismatch for matvec",
            ));
        }

        // 1. Smooth activations: x_smooth_j = x_j / s_j
        let mut x_smooth = x.to_vec();
        smooth_activations(&mut x_smooth, &self.smoothing_scales);

        // 2. Quantize smoothed activations dynamically to symmetric INT8
        let max_act = x_smooth.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()));
        let act_scale = if max_act > 0.0 { max_act / 127.0 } else { 1.0 };
        let inv_act_scale = 1.0 / act_scale;

        let mut x_int8 = vec![0i8; self.cols];
        for i in 0..self.cols {
            let val = (x_smooth[i] * inv_act_scale).round();
            x_int8[i] = val.clamp(-128.0, 127.0) as i8;
        }

        // 3. Compute INT8 fused dot products
        for (r, out_val) in out.iter_mut().enumerate() {
            let row_offset = r * self.cols;
            let w_row = &self.qweight[row_offset..row_offset + self.cols];

            let mut dot: i32 = 0;
            for c in 0..self.cols {
                dot += (x_int8[c] as i32) * (w_row[c] as i32);
            }

            *out_val = (dot as f32) * act_scale * self.row_scales[r];
        }

        Ok(())
    }
}
