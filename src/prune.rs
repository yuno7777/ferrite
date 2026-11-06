//! Direct GGUF model pruning engine supporting structured width/head and unstructured magnitude pruning.
//!
//! Operates directly on GGUF model files in-place or on memory-mapped tensor buffers,
//! enabling parameter sparsity reduction without external Python tooling or full model re-export.

use std::io::{Error, ErrorKind, Result};
use std::path::Path;

use crate::edit::GgufEditor;
use crate::gguf::GgmlType;

/// Statistics summarizing model or tensor parameter pruning.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PruningStats {
    pub total_parameters: usize,
    pub pruned_parameters: usize,
    pub sparsity: f32,
    pub tensors_pruned: usize,
}

impl PruningStats {
    pub fn new(total_parameters: usize, pruned_parameters: usize, tensors_pruned: usize) -> Self {
        let sparsity = if total_parameters > 0 {
            pruned_parameters as f32 / total_parameters as f32
        } else {
            0.0
        };
        Self {
            total_parameters,
            pruned_parameters,
            sparsity,
            tensors_pruned,
        }
    }

    pub fn combine(&self, other: &Self) -> Self {
        let total = self.total_parameters + other.total_parameters;
        let pruned = self.pruned_parameters + other.pruned_parameters;
        let tensors = self.tensors_pruned + other.tensors_pruned;
        Self::new(total, pruned, tensors)
    }
}

/// Zeroes out the lowest magnitude weights in `weights` up to `target_sparsity`.
pub fn prune_unstructured_slice(weights: &mut [f32], target_sparsity: f32) -> PruningStats {
    let total = weights.len();
    if total == 0 {
        return PruningStats::default();
    }
    let target_sparsity = target_sparsity.clamp(0.0, 1.0);
    if target_sparsity == 0.0 {
        return PruningStats::new(total, 0, 0);
    }

    let k = ((total as f32) * target_sparsity).round() as usize;
    if k == 0 {
        return PruningStats::new(total, 0, 0);
    }
    if k >= total {
        weights.fill(0.0);
        return PruningStats::new(total, total, 1);
    }

    // Collect magnitudes
    let mut magnitudes: Vec<f32> = weights.iter().map(|w| w.abs()).collect();
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = magnitudes[k - 1];

    let mut pruned = 0;
    for w in weights.iter_mut() {
        if (w.abs() < threshold || w.abs() == threshold) && pruned < k {
            *w = 0.0;
            pruned += 1;
        }
    }

    PruningStats::new(total, pruned, 1)
}

/// Computes the L2 norm of each attention head in a projection weight matrix.
pub fn compute_head_l2_norms(
    weights: &[f32],
    rows: usize,
    cols: usize,
    num_heads: usize,
) -> Vec<f32> {
    assert_eq!(weights.len(), rows * cols);
    assert!(num_heads > 0 && rows % num_heads == 0);

    let head_dim = rows / num_heads;
    let mut norms = vec![0.0f32; num_heads];

    for (h, norm) in norms.iter_mut().enumerate() {
        let head_start_row = h * head_dim;
        let head_end_row = head_start_row + head_dim;
        let mut sum_sq = 0.0f32;

        for r in head_start_row..head_end_row {
            let row_offset = r * cols;
            for c in 0..cols {
                let v = weights[row_offset + c];
                sum_sq += v * v;
            }
        }
        *norm = sum_sq.sqrt();
    }

    norms
}

/// Identifies the `prune_count` least important attention heads based on L2 norms.
pub fn select_heads_to_prune(head_norms: &[f32], prune_count: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = head_norms.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut selected: Vec<usize> = indexed.iter().take(prune_count).map(|(i, _)| *i).collect();
    selected.sort_unstable();
    selected
}

/// Zeroes out the specified attention head slices in-place.
pub fn prune_heads_slice(
    weights: &mut [f32],
    rows: usize,
    cols: usize,
    num_heads: usize,
    heads_to_prune: &[usize],
) -> PruningStats {
    assert_eq!(weights.len(), rows * cols);
    assert!(num_heads > 0 && rows % num_heads == 0);

    let head_dim = rows / num_heads;
    let mut pruned = 0;

    for &h in heads_to_prune {
        if h < num_heads {
            let head_start_row = h * head_dim;
            let head_end_row = head_start_row + head_dim;
            for r in head_start_row..head_end_row {
                let row_offset = r * cols;
                weights[row_offset..row_offset + cols].fill(0.0);
                pruned += cols;
            }
        }
    }

    PruningStats::new(rows * cols, pruned, if pruned > 0 { 1 } else { 0 })
}

/// Computes the L2 norm for each column/channel in a matrix.
pub fn compute_channel_l2_norms(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(weights.len(), rows * cols);
    let mut norms = vec![0.0f32; cols];
    for c in 0..cols {
        let mut sum_sq = 0.0f32;
        for r in 0..rows {
            let v = weights[r * cols + c];
            sum_sq += v * v;
        }
        norms[c] = sum_sq.sqrt();
    }
    norms
}

/// Zeroes out the specified column/channel indices across all rows.
pub fn prune_channels_slice(
    weights: &mut [f32],
    rows: usize,
    cols: usize,
    channels_to_prune: &[usize],
) -> PruningStats {
    assert_eq!(weights.len(), rows * cols);
    let mut pruned = 0;

    for &c in channels_to_prune {
        if c < cols {
            for r in 0..rows {
                weights[r * cols + c] = 0.0;
                pruned += 1;
            }
        }
    }

    PruningStats::new(rows * cols, pruned, if pruned > 0 { 1 } else { 0 })
}

/// Direct GGUF model pruning engine coordinating in-place file modifications.
pub struct ModelPruner;

impl ModelPruner {
    /// In-place unstructured magnitude pruning across F32 tensors in a GGUF file.
    pub fn prune_unstructured_inplace(
        path: &Path,
        target_sparsity: f32,
        tensor_names: Option<&[&str]>,
    ) -> Result<PruningStats> {
        let mut editor = GgufEditor::open(path)?;
        let mut total_stats = PruningStats::default();

        let tensor_list: Vec<String> = match tensor_names {
            Some(names) => names.iter().map(|&s| s.to_string()).collect(),
            None => editor
                .tensors
                .iter()
                .filter(|t| t.ty == GgmlType::F32)
                .map(|t| t.name.clone())
                .collect(),
        };

        for name in tensor_list {
            if let Ok(mut values) = editor.read_tensor_f32(&name) {
                let stats = prune_unstructured_slice(&mut values, target_sparsity);
                editor.write_tensor_f32(&name, &values)?;
                total_stats = total_stats.combine(&stats);
            }
        }

        editor.flush()?;
        Ok(total_stats)
    }

    /// In-place structured attention head pruning for a specified attention tensor.
    pub fn prune_attention_heads_inplace(
        path: &Path,
        tensor_name: &str,
        num_heads: usize,
        prune_count: usize,
    ) -> Result<(PruningStats, Vec<usize>)> {
        let mut editor = GgufEditor::open(path)?;
        let tensor_info = editor
            .tensor(tensor_name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "tensor not found"))?;

        if tensor_info.ty != GgmlType::F32 || tensor_info.dims.len() != 2 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "only 2D F32 attention tensors are supported for head pruning",
            ));
        }

        // GGUF dims: [cols, rows]
        let cols = tensor_info.dims[0] as usize;
        let rows = tensor_info.dims[1] as usize;

        let mut values = editor.read_tensor_f32(tensor_name)?;
        let norms = compute_head_l2_norms(&values, rows, cols, num_heads);
        let heads_to_prune = select_heads_to_prune(&norms, prune_count);

        let stats = prune_heads_slice(&mut values, rows, cols, num_heads, &heads_to_prune);
        editor.write_tensor_f32(tensor_name, &values)?;
        editor.flush()?;

        Ok((stats, heads_to_prune))
    }

    /// In-place structured channel/width pruning for a 2D F32 tensor.
    pub fn prune_width_inplace(
        path: &Path,
        tensor_name: &str,
        prune_ratio: f32,
    ) -> Result<PruningStats> {
        let mut editor = GgufEditor::open(path)?;
        let tensor_info = editor
            .tensor(tensor_name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "tensor not found"))?;

        if tensor_info.ty != GgmlType::F32 || tensor_info.dims.len() != 2 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "only 2D F32 tensors are supported for width pruning",
            ));
        }

        let cols = tensor_info.dims[0] as usize;
        let rows = tensor_info.dims[1] as usize;

        let mut values = editor.read_tensor_f32(tensor_name)?;
        let norms = compute_channel_l2_norms(&values, rows, cols);

        let prune_count = ((cols as f32) * prune_ratio.clamp(0.0, 1.0)).round() as usize;
        let channels_to_prune = select_heads_to_prune(&norms, prune_count);

        let stats = prune_channels_slice(&mut values, rows, cols, &channels_to_prune);
        editor.write_tensor_f32(tensor_name, &values)?;
        editor.flush()?;

        Ok(stats)
    }
}
