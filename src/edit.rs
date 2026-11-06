//! Zero-copy GGUF tensor in-place modification and quantized model weight editing API.
//!
//! Allows direct memory-mapped mutation of tensor bytes, in-place LoRA weight merging,
//! and quantization modifications in GGUF files without creating copies or re-serializing.

use std::collections::HashMap;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

use crate::gguf::{GgmlType, Gguf, TensorInfo};
use crate::map::MmapMut;
use crate::quant::half;

fn bad(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, msg.into())
}

/// Zero-copy editor for modifying GGUF model files directly in-place.
pub struct GgufEditor {
    pub path: PathBuf,
    map: MmapMut,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub alignment: u64,
    index: HashMap<String, usize>,
}

impl GgufEditor {
    /// Opens a GGUF file for zero-copy in-place editing.
    pub fn open(path: &Path) -> Result<Self> {
        // Inspect container metadata and tensor layout
        let (tensors, data_offset, alignment) = {
            let gguf = Gguf::open(path)?;
            if !gguf.shard_maps.is_empty() {
                return Err(bad(
                    "in-place editing of multi-file sharded GGUF models directly is unsupported; edit individual shard files",
                ));
            }
            (gguf.tensors, gguf.data_offset, gguf.alignment)
        };

        let mut index = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            index.insert(t.name.clone(), i);
        }

        // Open read-write shared memory mapping
        let map = MmapMut::open(path)?;

        Ok(Self {
            path: path.to_path_buf(),
            map,
            tensors,
            data_offset,
            alignment,
            index,
        })
    }

    /// Retrieve tensor information by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// Read-only slice view into a tensor's byte storage.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensor(name)
            .ok_or_else(|| bad(format!("tensor {name} not found")))?;
        let len = info
            .byte_len
            .ok_or_else(|| bad(format!("tensor {name} has unknown byte layout")))?
            as usize;
        let start = (self.data_offset + info.offset) as usize;
        let end = start
            .checked_add(len)
            .ok_or_else(|| bad("tensor byte range overflows"))?;

        if end > self.map.len() {
            return Err(bad(format!("tensor {name} byte range exceeds file bounds")));
        }

        Ok(&self.map.as_slice()[start..end])
    }

    /// Mutable zero-copy slice view into a tensor's byte storage.
    pub fn tensor_bytes_mut(&mut self, name: &str) -> Result<&mut [u8]> {
        let info = self
            .tensor(name)
            .ok_or_else(|| bad(format!("tensor {name} not found")))?;
        let len = info
            .byte_len
            .ok_or_else(|| bad(format!("tensor {name} has unknown byte layout")))?
            as usize;
        let start = (self.data_offset + info.offset) as usize;
        let end = start
            .checked_add(len)
            .ok_or_else(|| bad("tensor byte range overflows"))?;

        if end > self.map.len() {
            return Err(bad(format!("tensor {name} byte range exceeds file bounds")));
        }

        Ok(&mut self.map.as_mut_slice()[start..end])
    }

    /// Overwrite tensor bytes in-place.
    pub fn write_tensor_bytes(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let dest = self.tensor_bytes_mut(name)?;
        if dest.len() != bytes.len() {
            return Err(bad(format!(
                "byte length mismatch for tensor {name}: expected {}, got {}",
                dest.len(),
                bytes.len()
            )));
        }
        dest.copy_from_slice(bytes);
        Ok(())
    }

    /// Read an F32 tensor's values into a vector.
    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensor(name)
            .ok_or_else(|| bad(format!("tensor {name} not found")))?;
        if info.ty != GgmlType::F32 {
            return Err(bad(format!(
                "tensor {name} is of type {:?}, expected F32",
                info.ty
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let count = info.elements() as usize;
        let mut out = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(4) {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            out.push(val);
        }
        Ok(out)
    }

    /// Overwrite an F32 tensor with new values in-place.
    pub fn write_tensor_f32(&mut self, name: &str, values: &[f32]) -> Result<()> {
        let info = self
            .tensor(name)
            .ok_or_else(|| bad(format!("tensor {name} not found")))?;
        if info.ty != GgmlType::F32 {
            return Err(bad(format!(
                "tensor {name} is of type {:?}, expected F32",
                info.ty
            )));
        }
        let expected = info.elements() as usize;
        if values.len() != expected {
            return Err(bad(format!(
                "value count mismatch for tensor {name}: expected {expected}, got {}",
                values.len()
            )));
        }

        let dest = self.tensor_bytes_mut(name)?;
        for (idx, &v) in values.iter().enumerate() {
            dest[idx * 4..(idx + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    }

    /// In-place LoRA weight merging into a 2D F32 tensor.
    ///
    /// Computes $W \mathrel{+}= \text{scale} \cdot (B \times A)$ in-place.
    ///
    /// In GGUF, a 2D tensor has dims `[cols, rows]`.
    /// - `lora_a` has shape `[rank, cols]` (length = `rank * cols`).
    /// - `lora_b` has shape `[rows, rank]` (length = `rows * rank`).
    pub fn apply_lora_in_place(
        &mut self,
        tensor_name: &str,
        lora_a: &[f32],
        lora_b: &[f32],
        rank: usize,
        scale: f32,
    ) -> Result<()> {
        let info = self
            .tensor(tensor_name)
            .ok_or_else(|| bad(format!("tensor {tensor_name} not found")))?;
        if info.ty != GgmlType::F32 {
            return Err(bad(format!(
                "in-place LoRA merging requires F32 weights, but {tensor_name} has type {:?}",
                info.ty
            )));
        }
        if info.dims.len() != 2 {
            return Err(bad(format!(
                "in-place LoRA merging requires a 2D matrix, but {tensor_name} has {} dims",
                info.dims.len()
            )));
        }
        let cols = info.dims[0] as usize;
        let rows = info.dims[1] as usize;

        if rank == 0 {
            return Err(bad("LoRA rank must be non-zero"));
        }
        if lora_a.len() != rank * cols {
            return Err(bad(format!(
                "lora_a length {} != rank * cols ({} * {})",
                lora_a.len(),
                rank,
                cols
            )));
        }
        if lora_b.len() != rows * rank {
            return Err(bad(format!(
                "lora_b length {} != rows * rank ({} * {})",
                lora_b.len(),
                rows,
                rank
            )));
        }

        let dest_bytes = self.tensor_bytes_mut(tensor_name)?;

        // Update W[r * cols + c] in-place
        for r in 0..rows {
            let b_row = &lora_b[r * rank..(r + 1) * rank];
            for c in 0..cols {
                let mut delta = 0.0f32;
                for k in 0..rank {
                    delta += b_row[k] * lora_a[k * cols + c];
                }
                let offset = (r * cols + c) * 4;
                let current =
                    f32::from_le_bytes(dest_bytes[offset..offset + 4].try_into().unwrap());
                let updated = current + scale * delta;
                dest_bytes[offset..offset + 4].copy_from_slice(&updated.to_le_bytes());
            }
        }

        Ok(())
    }

    /// In-place update of a Q8_0 quantization block's f16 scale factor.
    pub fn update_q8_0_scale_in_place(
        &mut self,
        name: &str,
        block_index: usize,
        new_scale: f32,
    ) -> Result<()> {
        let info = self
            .tensor(name)
            .ok_or_else(|| bad(format!("tensor {name} not found")))?;
        if info.ty != GgmlType::Q8_0 {
            return Err(bad(format!("tensor {name} is not Q8_0")));
        }
        let total_blocks = (info.elements() as usize) / 32;
        if block_index >= total_blocks {
            return Err(bad(format!(
                "block index {block_index} >= total blocks {total_blocks}"
            )));
        }

        let dest = self.tensor_bytes_mut(name)?;
        let block_offset = block_index * 34;
        half::write_f16(&mut dest[block_offset..block_offset + 2], new_scale);
        Ok(())
    }

    /// Flush all in-place modifications to disk.
    pub fn flush(&self) -> Result<()> {
        self.map.flush()
    }
}
