//! Quantized and full-precision KV cache storage and attention kernels.
//!
//! Storing past keys and values in 32-bit floating point dominates memory usage
//! on long contexts (e.g., 2 GB for 2048 tokens on 8B models). Quantizing the KV
//! cache to 8-bit (`Q8_0`) or 4-bit (`Q4_0`) cuts memory by 3.5x to 6.4x with
//! negligible impact on generation perplexity.

use crate::ops::{self, AttentionParams};

/// Size in elements of each quantized KV block.
pub const KV_BLOCK_SIZE: usize = 32;

/// Supported KV cache quantization formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvCacheType {
    /// Full 32-bit floating point precision (default).
    #[default]
    F32,
    /// 8-bit symmetric block quantization (32 elements + f32 scale; 36 bytes/block).
    Q8_0,
    /// 4-bit packed block quantization (32 elements + f32 scale; 20 bytes/block).
    Q4_0,
}

/// F32 storage for DeepSeek MLA's compact KV latent and decoupled RoPE key.
/// This stores one latent row and one shared RoPE key per token, independent of
/// the number of attention heads. Attention projection is performed by the
/// model layer, which owns the per-head up-projection weights.
#[derive(Clone, Debug)]
pub struct MlaKvCache {
    latent: Vec<f32>,
    rope_key: Vec<f32>,
    layers: usize,
    context: usize,
    latent_dim: usize,
    rope_dim: usize,
}

impl MlaKvCache {
    pub fn new(layers: usize, context: usize, latent_dim: usize, rope_dim: usize) -> Self {
        Self {
            latent: vec![0.0; layers * context * latent_dim],
            rope_key: vec![0.0; layers * context * rope_dim],
            layers,
            context,
            latent_dim,
            rope_dim,
        }
    }

    pub fn cache_bytes(&self) -> usize {
        (self.latent.len() + self.rope_key.len()) * std::mem::size_of::<f32>()
    }

    pub fn store(&mut self, layer: usize, pos: usize, latent: &[f32], rope_key: &[f32]) {
        assert!(layer < self.layers && pos < self.context);
        assert_eq!(latent.len(), self.latent_dim);
        assert_eq!(rope_key.len(), self.rope_dim);
        let slot = layer * self.context + pos;
        let latent_start = slot * self.latent_dim;
        self.latent[latent_start..latent_start + self.latent_dim].copy_from_slice(latent);
        let rope_start = slot * self.rope_dim;
        self.rope_key[rope_start..rope_start + self.rope_dim].copy_from_slice(rope_key);
    }

    pub fn get(&self, layer: usize, pos: usize) -> (&[f32], &[f32]) {
        assert!(layer < self.layers && pos < self.context);
        let slot = layer * self.context + pos;
        let latent_start = slot * self.latent_dim;
        let rope_start = slot * self.rope_dim;
        (
            &self.latent[latent_start..latent_start + self.latent_dim],
            &self.rope_key[rope_start..rope_start + self.rope_dim],
        )
    }

    pub fn truncate(&mut self, keep_pos: usize, end_pos: usize) {
        for layer in 0..self.layers {
            for pos in keep_pos..end_pos.min(self.context) {
                let slot = layer * self.context + pos;
                let latent_start = slot * self.latent_dim;
                self.latent[latent_start..latent_start + self.latent_dim].fill(0.0);
                let rope_start = slot * self.rope_dim;
                self.rope_key[rope_start..rope_start + self.rope_dim].fill(0.0);
            }
        }
    }

    /// Evaluates direct latent attention over cached latent representations.
    pub fn attention(
        &self,
        layer: usize,
        q_nope_projected: &[f32],
        q_rope: &[f32],
        params: crate::ops::MlaAttentionParams,
        out_latent_sum: &mut [f32],
    ) {
        assert!(layer < self.layers);
        assert!(params.len <= self.context);
        let lat_start = layer * self.context * self.latent_dim;
        let rope_start = layer * self.context * self.rope_dim;
        crate::ops::attention_mla_latent(
            q_nope_projected,
            q_rope,
            &self.latent[lat_start..],
            &self.rope_key[rope_start..],
            params,
            out_latent_sum,
        );
    }
}

impl std::str::FromStr for KvCacheType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Ok(Self::F32),
            "q8_0" | "q8" => Ok(Self::Q8_0),
            "q4_0" | "q4" => Ok(Self::Q4_0),
            other => Err(format!(
                "unknown KV cache type '{other}', expected f32, q8_0, or q4_0"
            )),
        }
    }
}

/// A 32-element Q8_0 KV cache block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ8_0 {
    pub scale: f32,
    pub qs: [i8; KV_BLOCK_SIZE],
}

impl Default for BlockQ8_0 {
    fn default() -> Self {
        Self {
            scale: 0.0,
            qs: [0; KV_BLOCK_SIZE],
        }
    }
}

impl BlockQ8_0 {
    /// Quantize up to 32 elements of `src` into signed 8-bit quants.
    pub fn quantize(src: &[f32]) -> Self {
        let amax = src.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = amax / 127.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut qs = [0i8; KV_BLOCK_SIZE];
        for (slot, &val) in qs.iter_mut().zip(src) {
            *slot = (val * inv).round().clamp(-127.0, 127.0) as i8;
        }
        Self { scale, qs }
    }

    /// Dequantize element at index `0..32`.
    #[inline]
    pub fn dequantize(&self, index: usize) -> f32 {
        self.scale * (self.qs[index] as f32)
    }
}

/// A 32-element Q4_0 KV cache block (16 bytes packed nibbles + 4 byte f32 scale).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ4_0 {
    pub scale: f32,
    pub qs: [u8; KV_BLOCK_SIZE / 2],
}

impl Default for BlockQ4_0 {
    fn default() -> Self {
        Self {
            scale: 0.0,
            qs: [0x88; KV_BLOCK_SIZE / 2], // 8 - 8 = 0.0
        }
    }
}

impl BlockQ4_0 {
    /// Quantize up to 32 elements of `src` into 4-bit nibbles centred on 8.
    pub fn quantize(src: &[f32]) -> Self {
        let amax = src.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = amax / 8.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut qs = [0u8; KV_BLOCK_SIZE / 2];
        for j in 0..16 {
            let v0 = if j < src.len() {
                (src[j] * inv).round().clamp(-8.0, 7.0) as i32 + 8
            } else {
                8
            };
            let v1 = if j + 16 < src.len() {
                (src[j + 16] * inv).round().clamp(-8.0, 7.0) as i32 + 8
            } else {
                8
            };
            qs[j] = (v0 as u8 & 0x0F) | (((v1 as u8) & 0x0F) << 4);
        }
        Self { scale, qs }
    }

    /// Dequantize element at index `0..32`.
    #[inline]
    pub fn dequantize(&self, index: usize) -> f32 {
        let nibble = if index < 16 {
            (self.qs[index] & 0x0F) as f32 - 8.0
        } else {
            (self.qs[index - 16] >> 4) as f32 - 8.0
        };
        self.scale * nibble
    }
}

#[inline]
fn dot_block_q8_0(q: &[f32], block: &BlockQ8_0) -> f32 {
    let mut sum = 0.0f32;
    for (x, &w) in q.iter().zip(&block.qs) {
        sum += x * (w as f32);
    }
    block.scale * sum
}

#[inline]
fn add_weighted_block_q8_0(out: &mut [f32], weight: f32, block: &BlockQ8_0) {
    let scaled = weight * block.scale;
    for (o, &w) in out.iter_mut().zip(&block.qs) {
        *o += scaled * (w as f32);
    }
}

#[inline]
fn dot_block_q4_0(q: &[f32], block: &BlockQ4_0) -> f32 {
    let mut sum = 0.0f32;
    let len = q.len().min(KV_BLOCK_SIZE);
    let n_low = 16.min(len);
    for (j, item) in q.iter().enumerate().take(n_low) {
        let n0 = (block.qs[j] & 0x0F) as f32 - 8.0;
        sum += *item * n0;
    }
    if len > 16 {
        for j in 0..(len - 16) {
            let n1 = (block.qs[j] >> 4) as f32 - 8.0;
            sum += q[j + 16] * n1;
        }
    }
    block.scale * sum
}

#[inline]
fn add_weighted_block_q4_0(out: &mut [f32], weight: f32, block: &BlockQ4_0) {
    let scaled = weight * block.scale;
    let len = out.len().min(KV_BLOCK_SIZE);
    let n_low = 16.min(len);
    for (j, item) in out.iter_mut().enumerate().take(n_low) {
        let n0 = (block.qs[j] & 0x0F) as f32 - 8.0;
        *item += scaled * n0;
    }
    if len > 16 {
        for j in 0..(len - 16) {
            let n1 = (block.qs[j] >> 4) as f32 - 8.0;
            out[j + 16] += scaled * n1;
        }
    }
}

/// KV cache memory storage supporting F32, Q8_0, and Q4_0 formats.
#[derive(Clone, Debug)]
pub enum KvCache {
    F32 {
        key: Vec<f32>,
        value: Vec<f32>,
        layers: usize,
        context: usize,
        kv_dim: usize,
    },
    Q8_0 {
        key: Vec<BlockQ8_0>,
        value: Vec<BlockQ8_0>,
        layers: usize,
        context: usize,
        kv_dim: usize,
        blocks_per_token: usize,
    },
    Q4_0 {
        key: Vec<BlockQ4_0>,
        value: Vec<BlockQ4_0>,
        layers: usize,
        context: usize,
        kv_dim: usize,
        blocks_per_token: usize,
    },
}

impl KvCache {
    /// Allocate a KV cache buffer for `layers` layers, `context` positions, and `kv_dim` dimensions.
    pub fn new(layers: usize, context: usize, kv_dim: usize, cache_type: KvCacheType) -> Self {
        Self::new_asymmetric(layers, context, kv_dim, kv_dim, cache_type)
    }

    /// Allocate a KV cache buffer for `layers` layers, `context` positions, with asymmetric `k_dim` and `v_dim`.
    pub fn new_asymmetric(
        layers: usize,
        context: usize,
        k_dim: usize,
        v_dim: usize,
        cache_type: KvCacheType,
    ) -> Self {
        match cache_type {
            KvCacheType::F32 => Self::F32 {
                key: vec![0.0; layers * context * k_dim],
                value: vec![0.0; layers * context * v_dim],
                layers,
                context,
                kv_dim: k_dim,
            },
            KvCacheType::Q8_0 => {
                assert_eq!(
                    k_dim, v_dim,
                    "quantized KV cache requires equal key/value widths"
                );
                let blocks_per_token = k_dim.div_ceil(KV_BLOCK_SIZE);
                let total_blocks = layers * context * blocks_per_token;
                Self::Q8_0 {
                    key: vec![BlockQ8_0::default(); total_blocks],
                    value: vec![BlockQ8_0::default(); total_blocks],
                    layers,
                    context,
                    kv_dim: k_dim,
                    blocks_per_token,
                }
            }
            KvCacheType::Q4_0 => {
                assert_eq!(
                    k_dim, v_dim,
                    "quantized KV cache requires equal key/value widths"
                );
                let blocks_per_token = k_dim.div_ceil(KV_BLOCK_SIZE);
                let total_blocks = layers * context * blocks_per_token;
                Self::Q4_0 {
                    key: vec![BlockQ4_0::default(); total_blocks],
                    value: vec![BlockQ4_0::default(); total_blocks],
                    layers,
                    context,
                    kv_dim: k_dim,
                    blocks_per_token,
                }
            }
        }
    }

    /// The quantization format of this cache.
    pub fn cache_type(&self) -> KvCacheType {
        match self {
            Self::F32 { .. } => KvCacheType::F32,
            Self::Q8_0 { .. } => KvCacheType::Q8_0,
            Self::Q4_0 { .. } => KvCacheType::Q4_0,
        }
    }

    /// Approximate total bytes occupied by keys and values in memory.
    pub fn cache_bytes(&self) -> usize {
        match self {
            Self::F32 { key, value, .. } => (key.len() + value.len()) * std::mem::size_of::<f32>(),
            Self::Q8_0 { key, value, .. } => {
                (key.len() + value.len()) * std::mem::size_of::<BlockQ8_0>()
            }
            Self::Q4_0 { key, value, .. } => {
                (key.len() + value.len()) * std::mem::size_of::<BlockQ4_0>()
            }
        }
    }

    /// Store a key vector and value vector at `(layer, pos)`.
    pub fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        match self {
            Self::F32 {
                key,
                value,
                layers,
                context,
                kv_dim,
            } => {
                let v_dim = value.len() / (*layers * *context);
                let slot_k = (layer * *context + pos) * *kv_dim;
                let k_len = k.len().min(*kv_dim);
                key[slot_k..slot_k + k_len].copy_from_slice(&k[..k_len]);
                let slot_v = (layer * *context + pos) * v_dim;
                let v_len = v.len().min(v_dim);
                value[slot_v..slot_v + v_len].copy_from_slice(&v[..v_len]);
            }
            Self::Q8_0 {
                key,
                value,
                context,
                kv_dim,
                blocks_per_token,
                ..
            } => {
                let base_block = (layer * *context + pos) * *blocks_per_token;
                for b in 0..*blocks_per_token {
                    let start = b * KV_BLOCK_SIZE;
                    let end = (start + KV_BLOCK_SIZE).min(*kv_dim);
                    key[base_block + b] = BlockQ8_0::quantize(&k[start..end]);
                    value[base_block + b] = BlockQ8_0::quantize(&v[start..end]);
                }
            }
            Self::Q4_0 {
                key,
                value,
                context,
                kv_dim,
                blocks_per_token,
                ..
            } => {
                let base_block = (layer * *context + pos) * *blocks_per_token;
                for b in 0..*blocks_per_token {
                    let start = b * KV_BLOCK_SIZE;
                    let end = (start + KV_BLOCK_SIZE).min(*kv_dim);
                    key[base_block + b] = BlockQ4_0::quantize(&k[start..end]);
                    value[base_block + b] = BlockQ4_0::quantize(&v[start..end]);
                }
            }
        }
    }

    /// Compute tiled online attention for one query head over the stored KV history.
    pub fn attention(&self, layer: usize, query: &[f32], out: &mut [f32], params: AttentionParams) {
        match self {
            Self::F32 {
                key,
                value,
                layers,
                context,
                kv_dim,
            } => {
                let v_dim = value.len() / (*layers * *context);
                let layer_offset_k = layer * *context * *kv_dim;
                let layer_offset_v = layer * *context * v_dim;
                ops::attention_online(
                    query,
                    &key[layer_offset_k..],
                    &value[layer_offset_v..],
                    params,
                    out,
                );
            }
            Self::Q8_0 {
                key,
                value,
                context,
                blocks_per_token,
                ..
            } => {
                let layer_offset = layer * *context * *blocks_per_token;
                attention_online_q8_0(
                    query,
                    &key[layer_offset..],
                    &value[layer_offset..],
                    *blocks_per_token,
                    params,
                    out,
                );
            }
            Self::Q4_0 {
                key,
                value,
                context,
                blocks_per_token,
                ..
            } => {
                let layer_offset = layer * *context * *blocks_per_token;
                attention_online_q4_0(
                    query,
                    &key[layer_offset..],
                    &value[layer_offset..],
                    *blocks_per_token,
                    params,
                    out,
                );
            }
        }
    }

    /// Zero out KV-cache slots from `keep_pos` up to `end_pos`.
    pub fn truncate(&mut self, keep_pos: usize, end_pos: usize) {
        match self {
            Self::F32 {
                key,
                value,
                layers,
                context,
                kv_dim,
            } => {
                let v_dim = value.len() / (*layers * *context);
                let limit = end_pos.min(*context);
                for layer in 0..*layers {
                    for pos in keep_pos..limit {
                        let slot_k = (layer * *context + pos) * *kv_dim;
                        key[slot_k..slot_k + *kv_dim].fill(0.0);
                        let slot_v = (layer * *context + pos) * v_dim;
                        value[slot_v..slot_v + v_dim].fill(0.0);
                    }
                }
            }
            Self::Q8_0 {
                key,
                value,
                layers,
                context,
                blocks_per_token,
                ..
            } => {
                let limit = end_pos.min(*context);
                for layer in 0..*layers {
                    for pos in keep_pos..limit {
                        let slot = (layer * *context + pos) * *blocks_per_token;
                        key[slot..slot + *blocks_per_token].fill(BlockQ8_0::default());
                        value[slot..slot + *blocks_per_token].fill(BlockQ8_0::default());
                    }
                }
            }
            Self::Q4_0 {
                key,
                value,
                layers,
                context,
                blocks_per_token,
                ..
            } => {
                let limit = end_pos.min(*context);
                for layer in 0..*layers {
                    for pos in keep_pos..limit {
                        let slot = (layer * *context + pos) * *blocks_per_token;
                        key[slot..slot + *blocks_per_token].fill(BlockQ4_0::default());
                        value[slot..slot + *blocks_per_token].fill(BlockQ4_0::default());
                    }
                }
            }
        }
    }

    /// Shift tokens left across all layers by evicting `evict_count` entries starting at `evict_start`.
    ///
    /// Tokens in `[evict_start + evict_count .. current_len]` are shifted left to `evict_start`,
    /// and the vacated slots at the tail are cleared.
    pub fn evict_range(&mut self, evict_start: usize, evict_count: usize, current_len: usize) {
        if evict_count == 0 || evict_start >= current_len {
            return;
        }
        let actual_count = evict_count.min(current_len - evict_start);
        let shift_src_start = evict_start + actual_count;
        let shift_len = current_len.saturating_sub(shift_src_start);
        let new_len = current_len - actual_count;

        match self {
            Self::F32 {
                key,
                value,
                layers,
                context,
                kv_dim,
            } => {
                for layer in 0..*layers {
                    let base = layer * *context * *kv_dim;
                    let dst_offset = base + evict_start * *kv_dim;
                    let src_offset = base + shift_src_start * *kv_dim;
                    let num_floats = shift_len * *kv_dim;
                    if num_floats > 0 {
                        key.copy_within(src_offset..src_offset + num_floats, dst_offset);
                        value.copy_within(src_offset..src_offset + num_floats, dst_offset);
                    }
                    let clear_offset = base + new_len * *kv_dim;
                    let clear_floats = actual_count * *kv_dim;
                    key[clear_offset..clear_offset + clear_floats].fill(0.0);
                    value[clear_offset..clear_offset + clear_floats].fill(0.0);
                }
            }
            Self::Q8_0 {
                key,
                value,
                layers,
                context,
                blocks_per_token,
                ..
            } => {
                for layer in 0..*layers {
                    let base = layer * *context * *blocks_per_token;
                    let dst_offset = base + evict_start * *blocks_per_token;
                    let src_offset = base + shift_src_start * *blocks_per_token;
                    let num_blocks = shift_len * *blocks_per_token;
                    if num_blocks > 0 {
                        key.copy_within(src_offset..src_offset + num_blocks, dst_offset);
                        value.copy_within(src_offset..src_offset + num_blocks, dst_offset);
                    }
                    let clear_offset = base + new_len * *blocks_per_token;
                    let clear_blocks = actual_count * *blocks_per_token;
                    key[clear_offset..clear_offset + clear_blocks].fill(BlockQ8_0::default());
                    value[clear_offset..clear_offset + clear_blocks].fill(BlockQ8_0::default());
                }
            }
            Self::Q4_0 {
                key,
                value,
                layers,
                context,
                blocks_per_token,
                ..
            } => {
                for layer in 0..*layers {
                    let base = layer * *context * *blocks_per_token;
                    let dst_offset = base + evict_start * *blocks_per_token;
                    let src_offset = base + shift_src_start * *blocks_per_token;
                    let num_blocks = shift_len * *blocks_per_token;
                    if num_blocks > 0 {
                        key.copy_within(src_offset..src_offset + num_blocks, dst_offset);
                        value.copy_within(src_offset..src_offset + num_blocks, dst_offset);
                    }
                    let clear_offset = base + new_len * *blocks_per_token;
                    let clear_blocks = actual_count * *blocks_per_token;
                    key[clear_offset..clear_offset + clear_blocks].fill(BlockQ4_0::default());
                    value[clear_offset..clear_offset + clear_blocks].fill(BlockQ4_0::default());
                }
            }
        }
    }

    /// Evict middle tokens using StreamingLLM policy:
    /// Preserves initial `sink_tokens` (0..sink_tokens) and shifts recent tokens left to maintain
    /// the total length at `max_len`. Returns the new effective sequence length.
    pub fn evict_streaming(
        &mut self,
        current_len: usize,
        max_len: usize,
        sink_tokens: usize,
    ) -> usize {
        if current_len <= max_len || current_len <= sink_tokens {
            return current_len;
        }
        let evict_count = current_len - max_len;
        let evict_start = sink_tokens;
        self.evict_range(evict_start, evict_count, current_len);
        current_len - evict_count
    }

    /// Selectively compact KV cache to retain only entries in `keep_indices` (must be sorted ascending).
    ///
    /// Copies retained entries to sequential positions `0..keep_indices.len()` and clears the remainder.
    pub fn compact_indices(&mut self, keep_indices: &[usize]) {
        if keep_indices.is_empty() {
            return;
        }

        match self {
            Self::F32 {
                key,
                value,
                layers,
                context,
                kv_dim,
            } => {
                for layer in 0..*layers {
                    let base = layer * *context * *kv_dim;
                    for (dst_pos, &src_pos) in keep_indices.iter().enumerate() {
                        if dst_pos != src_pos && src_pos < *context {
                            let dst = base + dst_pos * *kv_dim;
                            let src = base + src_pos * *kv_dim;
                            key.copy_within(src..src + *kv_dim, dst);
                            value.copy_within(src..src + *kv_dim, dst);
                        }
                    }
                    let clear_start = base + keep_indices.len() * *kv_dim;
                    let clear_end = base + *context * *kv_dim;
                    if clear_start < clear_end {
                        key[clear_start..clear_end].fill(0.0);
                        value[clear_start..clear_end].fill(0.0);
                    }
                }
            }
            Self::Q8_0 {
                key,
                value,
                layers,
                context,
                blocks_per_token,
                ..
            } => {
                for layer in 0..*layers {
                    let base = layer * *context * *blocks_per_token;
                    for (dst_pos, &src_pos) in keep_indices.iter().enumerate() {
                        if dst_pos != src_pos && src_pos < *context {
                            let dst = base + dst_pos * *blocks_per_token;
                            let src = base + src_pos * *blocks_per_token;
                            key.copy_within(src..src + *blocks_per_token, dst);
                            value.copy_within(src..src + *blocks_per_token, dst);
                        }
                    }
                    let clear_start = base + keep_indices.len() * *blocks_per_token;
                    let clear_end = base + *context * *blocks_per_token;
                    if clear_start < clear_end {
                        key[clear_start..clear_end].fill(BlockQ8_0::default());
                        value[clear_start..clear_end].fill(BlockQ8_0::default());
                    }
                }
            }
            Self::Q4_0 {
                key,
                value,
                layers,
                context,
                blocks_per_token,
                ..
            } => {
                for layer in 0..*layers {
                    let base = layer * *context * *blocks_per_token;
                    for (dst_pos, &src_pos) in keep_indices.iter().enumerate() {
                        if dst_pos != src_pos && src_pos < *context {
                            let dst = base + dst_pos * *blocks_per_token;
                            let src = base + src_pos * *blocks_per_token;
                            key.copy_within(src..src + *blocks_per_token, dst);
                            value.copy_within(src..src + *blocks_per_token, dst);
                        }
                    }
                    let clear_start = base + keep_indices.len() * *blocks_per_token;
                    let clear_end = base + *context * *blocks_per_token;
                    if clear_start < clear_end {
                        key[clear_start..clear_end].fill(BlockQ4_0::default());
                        value[clear_start..clear_end].fill(BlockQ4_0::default());
                    }
                }
            }
        }
    }

    /// Access raw f32 key/value slices if this cache is in F32 format.
    pub fn as_f32_slices(&self) -> Option<(&[f32], &[f32])> {
        match self {
            Self::F32 { key, value, .. } => Some((key, value)),
            _ => None,
        }
    }
}

/// Eviction policy configurations for context compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// StreamingLLM: Retain `sink_tokens` initial tokens + recent `window_tokens`.
    StreamingLLM {
        sink_tokens: usize,
        window_tokens: usize,
    },
    /// Heavy-Hitter Oracle (H2O): Retain initial `sink_tokens`, top `heavy_hitters`, and recent `window_tokens`.
    H2O {
        sink_tokens: usize,
        heavy_hitters: usize,
        window_tokens: usize,
    },
}

/// Compute retained indices using H2O (Heavy-Hitter Oracle) policy over cumulative attention scores.
pub fn select_h2o_indices(
    scores: &[f32],
    sink_tokens: usize,
    heavy_hitters: usize,
    window_tokens: usize,
) -> Vec<usize> {
    let n = scores.len();
    if n <= sink_tokens + heavy_hitters + window_tokens {
        return (0..n).collect();
    }
    let sinks_end = sink_tokens.min(n);
    let window_start = n.saturating_sub(window_tokens);
    let mut selected: std::collections::HashSet<usize> =
        (0..sinks_end).chain(window_start..n).collect();

    let mut middle_indices: Vec<usize> = (sinks_end..window_start).collect();
    middle_indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for &idx in middle_indices.iter().take(heavy_hitters) {
        selected.insert(idx);
    }

    let mut result: Vec<usize> = selected.into_iter().collect();
    result.sort_unstable();
    result
}

fn attention_online_q8_0(
    query: &[f32],
    keys: &[BlockQ8_0],
    values: &[BlockQ8_0],
    blocks_per_token: usize,
    params: AttentionParams,
    out: &mut [f32],
) {
    let AttentionParams {
        offset,
        len,
        scale,
        logit_cap,
        window,
        position,
        tile_size,
        ..
    } = params;
    out.fill(0.0);
    let mut max_score = f32::NEG_INFINITY;
    let mut denominator = 0.0f32;
    let mut scores = vec![0.0f32; tile_size];

    let aligned = offset % KV_BLOCK_SIZE == 0 && query.len() % KV_BLOCK_SIZE == 0;
    let block_start = offset / KV_BLOCK_SIZE;
    let num_blocks = query.len() / KV_BLOCK_SIZE;

    for start in (0..len).step_by(tile_size) {
        let end = (start + tile_size).min(len);
        for (i, past) in (start..end).enumerate() {
            if window.is_some_and(|w| position.saturating_sub(past) >= w) {
                scores[i] = f32::NEG_INFINITY;
                continue;
            }
            let tok_base = past * blocks_per_token;
            let tok_keys = &keys[tok_base..tok_base + blocks_per_token];

            let dot_val = if aligned {
                let mut sum = 0.0f32;
                for b in 0..num_blocks {
                    let q_chunk = &query[b * KV_BLOCK_SIZE..(b + 1) * KV_BLOCK_SIZE];
                    sum += dot_block_q8_0(q_chunk, &tok_keys[block_start + b]);
                }
                sum
            } else {
                let mut sum = 0.0f32;
                for (k, &q_val) in query.iter().enumerate() {
                    let elem = offset + k;
                    let b = elem / KV_BLOCK_SIZE;
                    let j = elem % KV_BLOCK_SIZE;
                    sum += q_val * tok_keys[b].dequantize(j);
                }
                sum
            };

            let mut score = dot_val * scale;
            if let Some(cap) = logit_cap {
                score = cap * (score / cap).tanh();
            }
            scores[i] = score;
        }

        let tile_max = scores[..end - start]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if !tile_max.is_finite() {
            continue;
        }
        let next_max = max_score.max(tile_max);
        let old_scale = if max_score.is_finite() {
            (max_score - next_max).exp()
        } else {
            0.0
        };
        denominator *= old_scale;
        for v in out.iter_mut() {
            *v *= old_scale;
        }

        for (i, past) in (start..end).enumerate() {
            if !scores[i].is_finite() {
                continue;
            }
            let weight = (scores[i] - next_max).exp();
            denominator += weight;
            let tok_base = past * blocks_per_token;
            let tok_vals = &values[tok_base..tok_base + blocks_per_token];

            if aligned {
                for b in 0..num_blocks {
                    let out_chunk = &mut out[b * KV_BLOCK_SIZE..(b + 1) * KV_BLOCK_SIZE];
                    add_weighted_block_q8_0(out_chunk, weight, &tok_vals[block_start + b]);
                }
            } else {
                for (k, out_slot) in out.iter_mut().enumerate() {
                    let elem = offset + k;
                    let b = elem / KV_BLOCK_SIZE;
                    let j = elem % KV_BLOCK_SIZE;
                    *out_slot += weight * tok_vals[b].dequantize(j);
                }
            }
        }
        max_score = next_max;
    }

    if denominator > 0.0 {
        for value in out {
            *value /= denominator;
        }
    }
}

fn attention_online_q4_0(
    query: &[f32],
    keys: &[BlockQ4_0],
    values: &[BlockQ4_0],
    blocks_per_token: usize,
    params: AttentionParams,
    out: &mut [f32],
) {
    let AttentionParams {
        offset,
        len,
        scale,
        logit_cap,
        window,
        position,
        tile_size,
        ..
    } = params;
    out.fill(0.0);
    let mut max_score = f32::NEG_INFINITY;
    let mut denominator = 0.0f32;
    let mut scores = vec![0.0f32; tile_size];

    let aligned = offset % KV_BLOCK_SIZE == 0 && query.len() % KV_BLOCK_SIZE == 0;
    let block_start = offset / KV_BLOCK_SIZE;
    let num_blocks = query.len() / KV_BLOCK_SIZE;

    for start in (0..len).step_by(tile_size) {
        let end = (start + tile_size).min(len);
        for (i, past) in (start..end).enumerate() {
            if window.is_some_and(|w| position.saturating_sub(past) >= w) {
                scores[i] = f32::NEG_INFINITY;
                continue;
            }
            let tok_base = past * blocks_per_token;
            let tok_keys = &keys[tok_base..tok_base + blocks_per_token];

            let dot_val = if aligned {
                let mut sum = 0.0f32;
                for b in 0..num_blocks {
                    let q_chunk = &query[b * KV_BLOCK_SIZE..(b + 1) * KV_BLOCK_SIZE];
                    sum += dot_block_q4_0(q_chunk, &tok_keys[block_start + b]);
                }
                sum
            } else {
                let mut sum = 0.0f32;
                for (k, &q_val) in query.iter().enumerate() {
                    let elem = offset + k;
                    let b = elem / KV_BLOCK_SIZE;
                    let j = elem % KV_BLOCK_SIZE;
                    sum += q_val * tok_keys[b].dequantize(j);
                }
                sum
            };

            let mut score = dot_val * scale;
            if let Some(cap) = logit_cap {
                score = cap * (score / cap).tanh();
            }
            scores[i] = score;
        }

        let tile_max = scores[..end - start]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if !tile_max.is_finite() {
            continue;
        }
        let next_max = max_score.max(tile_max);
        let old_scale = if max_score.is_finite() {
            (max_score - next_max).exp()
        } else {
            0.0
        };
        denominator *= old_scale;
        for v in out.iter_mut() {
            *v *= old_scale;
        }

        for (i, past) in (start..end).enumerate() {
            if !scores[i].is_finite() {
                continue;
            }
            let weight = (scores[i] - next_max).exp();
            denominator += weight;
            let tok_base = past * blocks_per_token;
            let tok_vals = &values[tok_base..tok_base + blocks_per_token];

            if aligned {
                for b in 0..num_blocks {
                    let out_chunk = &mut out[b * KV_BLOCK_SIZE..(b + 1) * KV_BLOCK_SIZE];
                    add_weighted_block_q4_0(out_chunk, weight, &tok_vals[block_start + b]);
                }
            } else {
                for (k, out_slot) in out.iter_mut().enumerate() {
                    let elem = offset + k;
                    let b = elem / KV_BLOCK_SIZE;
                    let j = elem % KV_BLOCK_SIZE;
                    *out_slot += weight * tok_vals[b].dequantize(j);
                }
            }
        }
        max_score = next_max;
    }

    if denominator > 0.0 {
        for value in out {
            *value /= denominator;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mla_cache_stores_compact_latent_and_rope_rows() {
        let mut cache = MlaKvCache::new(2, 4, 3, 2);
        cache.store(1, 2, &[1.0, 2.0, 3.0], &[4.0, 5.0]);
        assert_eq!(cache.get(1, 2), (&[1.0, 2.0, 3.0][..], &[4.0, 5.0][..]));
        assert_eq!(cache.cache_bytes(), 2 * 4 * 5 * std::mem::size_of::<f32>());
        cache.truncate(2, 3);
        assert_eq!(cache.get(1, 2), (&[0.0, 0.0, 0.0][..], &[0.0, 0.0][..]));
    }

    #[test]
    fn mla_cache_direct_attention_evaluates_correctly() {
        let mut cache = MlaKvCache::new(1, 2, 2, 2);
        // Token 0: latent [1.0, 0.0], rope_key [0.0, 1.0]
        cache.store(0, 0, &[1.0, 0.0], &[0.0, 1.0]);
        // Token 1: latent [0.0, 2.0], rope_key [1.0, 0.0]
        cache.store(0, 1, &[0.0, 2.0], &[1.0, 0.0]);

        let q_nope = [1.0, 0.0];
        let q_rope = [0.0, 1.0];
        let mut out = [0.0; 2];

        // Evaluating attention over 2 tokens with scale 1.0
        cache.attention(
            0,
            &q_nope,
            &q_rope,
            crate::ops::MlaAttentionParams {
                latent_dim: 2,
                rope_dim: 2,
                len: 2,
                scale: 1.0,
                logit_cap: None,
            },
            &mut out,
        );

        // Token 0 score: dot(q_nope, [1, 0]) + dot(q_rope, [0, 1]) = 1 + 1 = 2.0
        // Token 1 score: dot(q_nope, [0, 2]) + dot(q_rope, [1, 0]) = 0 + 0 = 0.0
        // Softmax weights:
        // exp(2 - 2) = 1.0
        // exp(0 - 2) = exp(-2) ~= 0.135335
        // den = 1.0 + exp(-2) ~= 1.135335
        // w0 = 1.0 / 1.135335 ~= 0.8808
        // w1 = exp(-2) / 1.135335 ~= 0.1192
        // out[0] = w0 * 1.0 + w1 * 0.0 = 0.8808
        // out[1] = w0 * 0.0 + w1 * 2.0 = 0.2384
        let w0 = 1.0f32 / (1.0f32 + (-2.0f32).exp());
        let w1 = (-2.0f32).exp() / (1.0f32 + (-2.0f32).exp());
        assert!((out[0] - (w0 * 1.0)).abs() < 1e-5);
        assert!((out[1] - (w1 * 2.0)).abs() < 1e-5);

        // A partially filled cache must not attend to capacity beyond its live prefix.
        cache.attention(
            0,
            &q_nope,
            &q_rope,
            crate::ops::MlaAttentionParams {
                latent_dim: 2,
                rope_dim: 2,
                len: 1,
                scale: 1.0,
                logit_cap: None,
            },
            &mut out,
        );
        assert_eq!(out, [1.0, 0.0]);
    }

    #[test]
    fn f32_cache_supports_distinct_key_and_value_widths() {
        let mut cache = KvCache::new_asymmetric(1, 2, 4, 2, KvCacheType::F32);
        cache.store(0, 0, &[1.0, 0.0, 0.0, 0.0], &[3.0, 5.0]);
        let mut out = [0.0; 2];
        cache.attention(
            0,
            &[1.0, 0.0],
            &mut out,
            ops::AttentionParams {
                row_width: 4,
                offset: 0,
                len: 1,
                scale: 1.0,
                logit_cap: None,
                window: None,
                position: 0,
                tile_size: 1,
                v_row_width: Some(2),
                v_offset: Some(0),
            },
        );
        assert_eq!(out, [3.0, 5.0]);
        assert_eq!(
            cache.cache_bytes(),
            (4 + 2) * std::mem::size_of::<f32>() * 2
        );
    }

    #[test]
    fn q8_0_quantize_dequantize_roundtrip() {
        let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.125).collect();
        let block = BlockQ8_0::quantize(&original);
        assert!(block.scale > 0.0);
        for (i, &orig) in original.iter().enumerate() {
            let recovered = block.dequantize(i);
            assert!(
                (orig - recovered).abs() <= block.scale * 0.5 + 1e-6,
                "diff too large at index {i}: {orig} vs {recovered}"
            );
        }
    }

    #[test]
    fn q4_0_quantize_dequantize_roundtrip() {
        let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.125).collect();
        let block = BlockQ4_0::quantize(&original);
        assert!(block.scale > 0.0);
        for (i, &orig) in original.iter().enumerate() {
            let recovered = block.dequantize(i);
            assert!(
                (orig - recovered).abs() <= block.scale * 0.5 + 1e-6,
                "diff too large at index {i}: {orig} vs {recovered}"
            );
        }
    }

    #[test]
    fn q8_0_cache_memory_reduction() {
        let f32_cache = KvCache::new(4, 512, 128, KvCacheType::F32);
        let q8_cache = KvCache::new(4, 512, 128, KvCacheType::Q8_0);
        let q4_cache = KvCache::new(4, 512, 128, KvCacheType::Q4_0);

        let f32_bytes = f32_cache.cache_bytes();
        let q8_bytes = q8_cache.cache_bytes();
        let q4_bytes = q4_cache.cache_bytes();

        assert!(
            q8_bytes * 3 < f32_bytes,
            "q8_bytes {q8_bytes} should be ~3.5x smaller than f32_bytes {f32_bytes}"
        );
        assert!(
            q4_bytes * 5 < f32_bytes,
            "q4_bytes {q4_bytes} should be ~6.4x smaller than f32_bytes {f32_bytes}"
        );
    }
}
