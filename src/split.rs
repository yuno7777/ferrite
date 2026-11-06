//! Native GGUF model splitting and multi-file distributed shard partitioning.

use std::fs;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

use crate::gguf::{Gguf, Value};
use crate::synth::Builder;

fn bad(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, msg.into())
}

/// Options configuring GGUF model splitting.
#[derive(Clone, Debug)]
pub struct SplitOptions {
    /// Number of shards to produce (default: 2).
    pub shards: usize,
    /// Maximum target byte size per shard. When set, overrides or adjusts shards.
    pub max_bytes_per_shard: Option<usize>,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            shards: 2,
            max_bytes_per_shard: None,
        }
    }
}

/// Splits a monolithic GGUF file into multiple sharded GGUF files.
///
/// Follows standard GGUF sharding naming convention:
/// `<prefix>-00001-of-0000N.gguf`, `<prefix>-00002-of-0000N.gguf`, etc.
/// Shard 0 contains full model metadata (tokenizer, config, hyperparameters)
/// and a partition of tensors. Sibling shards contain remaining tensors and split metadata.
pub fn split_gguf(
    input_path: &Path,
    output_prefix: &Path,
    options: &SplitOptions,
) -> Result<Vec<PathBuf>> {
    let gguf = Gguf::open(input_path)?;
    if gguf.tensors.is_empty() {
        return Err(bad("cannot split GGUF model with 0 tensors"));
    }

    // Determine tensor byte sizes
    let mut tensor_sizes = Vec::with_capacity(gguf.tensors.len());
    let mut total_data_bytes: usize = 0;
    for t in &gguf.tensors {
        let (block, size) =
            t.ty.layout()
                .ok_or_else(|| bad(format!("unknown layout for tensor {}", t.name)))?;
        let bytes = (t.elements() / block * size) as usize;
        tensor_sizes.push(bytes);
        total_data_bytes += bytes;
    }

    let num_shards = if let Some(max_bytes) = options.max_bytes_per_shard {
        if max_bytes == 0 {
            return Err(bad("max_bytes_per_shard must be greater than zero"));
        }
        total_data_bytes
            .div_ceil(max_bytes)
            .max(2)
            .min(gguf.tensors.len())
    } else {
        options.shards.max(2).min(gguf.tensors.len())
    };

    if num_shards > 9999 {
        return Err(bad("cannot split into more than 9999 shards"));
    }

    // Partition tensors across shards balanced by byte size
    let target_bytes_per_shard = total_data_bytes.div_ceil(num_shards);
    let mut shard_tensor_indices: Vec<Vec<usize>> = vec![Vec::new(); num_shards];
    let mut current_shard = 0;
    let mut current_shard_bytes = 0;

    for (t_idx, &size) in tensor_sizes.iter().enumerate() {
        if current_shard + 1 < num_shards
            && current_shard_bytes + size > target_bytes_per_shard
            && !shard_tensor_indices[current_shard].is_empty()
        {
            current_shard += 1;
            current_shard_bytes = 0;
        }
        shard_tensor_indices[current_shard].push(t_idx);
        current_shard_bytes += size;
    }

    let prefix_str = output_prefix.to_string_lossy();
    let parent_dir = output_prefix.parent().unwrap_or_else(|| Path::new("."));
    if !parent_dir.as_os_str().is_empty() {
        fs::create_dir_all(parent_dir)?;
    }

    let mut written_paths = Vec::with_capacity(num_shards);

    for (s_idx, tensor_indices) in shard_tensor_indices.iter().enumerate() {
        let shard_no = s_idx + 1;
        let shard_name = format!("{prefix_str}-{shard_no:05}-of-{num_shards:05}.gguf");
        let shard_path = PathBuf::from(shard_name);

        let mut builder = Builder::new();

        // Shard metadata
        if s_idx == 0 {
            // Shard 0 carries all model metadata from input
            for (k, v) in &gguf.meta {
                builder = builder.meta(k, v.clone());
            }
        } else if let Some(arch) = gguf.arch() {
            builder = builder.meta("general.architecture", Value::String(arch.to_string()));
        }

        builder = builder
            .meta("split.no", Value::U16(s_idx as u16))
            .meta("split.count", Value::U16(num_shards as u16))
            .meta("split.tensors.count", Value::I32(gguf.tensors.len() as i32));

        // Add assigned tensors
        for &t_idx in tensor_indices {
            let t = &gguf.tensors[t_idx];
            let data = gguf.tensor_bytes(t)?.to_vec();
            builder = builder.tensor(&t.name, &t.dims, t.ty, data);
        }

        builder.write(&shard_path)?;
        written_paths.push(shard_path);
    }

    Ok(written_paths)
}
