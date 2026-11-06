//! Paged KV-cache and physical block table allocator (vLLM-style virtual memory).
//!
//! Organizes KV cache into fixed-size physical blocks (e.g., 16 tokens) mapped
//! via logical-to-physical block tables. Eliminates memory fragmentation, allows
//! dynamic demand paging, and supports zero-copy Copy-on-Write (CoW) forking
//! for parallel sampling, speculative branches, and multi-turn sharing.

use std::io::{Error, ErrorKind, Result};

use crate::ops::{self, AttentionParams};

/// Number of token slots per physical KV block.
pub const DEFAULT_BLOCK_TOKENS: usize = 16;

/// Identifier of a physical block within a [`PagedPool`].
pub type BlockId = usize;

/// Storage for one physical KV block across all layers for `tokens_per_block` tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalBlock {
    /// Keys: layout is `layer * tokens_per_block * kv_dim + slot * kv_dim + element`.
    pub key: Vec<f32>,
    /// Values: layout is `layer * tokens_per_block * kv_dim + slot * kv_dim + element`.
    pub value: Vec<f32>,
}

impl PhysicalBlock {
    pub fn new(layers: usize, tokens_per_block: usize, kv_dim: usize) -> Self {
        let size = layers * tokens_per_block * kv_dim;
        Self {
            key: vec![0.0; size],
            value: vec![0.0; size],
        }
    }

    pub fn clear(&mut self) {
        self.key.fill(0.0);
        self.value.fill(0.0);
    }
}

/// Global or per-worker pool of physical KV blocks with reference counting.
#[derive(Clone, Debug)]
pub struct PagedPool {
    pub layers: usize,
    pub kv_dim: usize,
    pub tokens_per_block: usize,
    pub blocks: Vec<PhysicalBlock>,
    pub ref_counts: Vec<usize>,
    pub free_list: Vec<BlockId>,
}

impl PagedPool {
    /// Create a new pool with pre-allocated capacity of `initial_blocks`.
    pub fn new(
        layers: usize,
        kv_dim: usize,
        tokens_per_block: usize,
        initial_blocks: usize,
    ) -> Self {
        assert!(layers > 0, "paged KV pool needs at least one layer");
        assert!(kv_dim > 0, "paged KV pool needs a non-empty KV dimension");
        assert!(
            tokens_per_block > 0,
            "paged KV blocks must hold at least one token"
        );
        let mut blocks = Vec::with_capacity(initial_blocks);
        let mut ref_counts = Vec::with_capacity(initial_blocks);
        let mut free_list = Vec::with_capacity(initial_blocks);

        for i in 0..initial_blocks {
            blocks.push(PhysicalBlock::new(layers, tokens_per_block, kv_dim));
            ref_counts.push(0);
            free_list.push(i);
        }

        Self {
            layers,
            kv_dim,
            tokens_per_block,
            blocks,
            ref_counts,
            free_list,
        }
    }

    /// Total blocks currently managed by the pool.
    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Number of free blocks available for immediate allocation.
    pub fn free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Number of blocks currently actively referenced.
    pub fn allocated_blocks(&self) -> usize {
        self.total_blocks() - self.free_blocks()
    }

    /// Total heap memory in bytes occupied by the physical pool.
    pub fn memory_bytes(&self) -> usize {
        let per_block =
            self.layers * self.tokens_per_block * self.kv_dim * std::mem::size_of::<f32>() * 2;
        self.blocks.len() * per_block
    }

    /// Allocate a physical block from the free list or grow pool if empty.
    pub fn allocate_block(&mut self) -> BlockId {
        if let Some(id) = self.free_list.pop() {
            self.ref_counts[id] = 1;
            self.blocks[id].clear();
            id
        } else {
            let id = self.blocks.len();
            self.blocks.push(PhysicalBlock::new(
                self.layers,
                self.tokens_per_block,
                self.kv_dim,
            ));
            self.ref_counts.push(1);
            id
        }
    }

    /// Increment reference count of a physical block (used during sequence forking).
    pub fn inc_ref(&mut self, block_id: BlockId) {
        if block_id < self.ref_counts.len() {
            self.ref_counts[block_id] += 1;
        }
    }

    /// Decrement reference count and return block to free list if count reaches zero.
    pub fn free_block(&mut self, block_id: BlockId) {
        if block_id < self.ref_counts.len() && self.ref_counts[block_id] > 0 {
            self.ref_counts[block_id] -= 1;
            if self.ref_counts[block_id] == 0 {
                self.free_list.push(block_id);
            }
        }
    }

    /// Ensure that `block_table[b_log]` is uniquely owned before modification (Copy-on-Write).
    pub fn prepare_for_write(&mut self, block_table: &mut [BlockId], b_log: usize) {
        let old_id = block_table[b_log];
        if self.ref_counts[old_id] > 1 {
            let old_key = self.blocks[old_id].key.clone();
            let old_val = self.blocks[old_id].value.clone();
            let new_id = self.allocate_block();
            self.blocks[new_id].key = old_key;
            self.blocks[new_id].value = old_val;
            self.free_block(old_id);
            block_table[b_log] = new_id;
        }
    }

    /// Write key and value vector for `layer` and logical `position` in `sequence`.
    pub fn store(
        &mut self,
        sequence: &mut PagedSequence,
        layer: usize,
        position: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        if layer >= self.layers {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "KV layer is out of range",
            ));
        }
        if k.len() != self.kv_dim || v.len() != self.kv_dim {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("KV vectors must both have length {}", self.kv_dim),
            ));
        }
        let b_log = position / self.tokens_per_block;
        let slot = position % self.tokens_per_block;

        while sequence.block_table.len() <= b_log {
            let new_block = self.allocate_block();
            sequence.block_table.push(new_block);
        }

        self.prepare_for_write(&mut sequence.block_table, b_log);

        let phys_id = sequence.block_table[b_log];
        let block = &mut self.blocks[phys_id];
        let offset = (layer * self.tokens_per_block + slot) * self.kv_dim;

        block.key[offset..offset + self.kv_dim].copy_from_slice(k);
        block.value[offset..offset + self.kv_dim].copy_from_slice(v);

        if position + 1 > sequence.seq_len {
            sequence.seq_len = position + 1;
        }
        Ok(())
    }

    /// Compute tiled online attention over physical blocks indexed by `sequence.block_table`.
    pub fn attention(
        &self,
        sequence: &PagedSequence,
        layer: usize,
        query: &[f32],
        out: &mut [f32],
        params: AttentionParams,
    ) -> Result<()> {
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

        let head_dim = query.len();
        let b_tokens = self.tokens_per_block;
        if layer >= self.layers
            || offset
                .checked_add(head_dim)
                .is_none_or(|end| end > self.kv_dim)
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "attention KV range is out of bounds",
            ));
        }
        if out.len() != head_dim || tile_size == 0 || len > sequence.seq_len {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "invalid paged attention dimensions",
            ));
        }
        if sequence.block_table.len() < len.div_ceil(b_tokens)
            || sequence.block_table[..len.div_ceil(b_tokens)]
                .iter()
                .any(|&id| id >= self.blocks.len() || self.ref_counts[id] == 0)
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "invalid paged block table",
            ));
        }

        for start in (0..len).step_by(tile_size) {
            let end = (start + tile_size).min(len);
            for (i, past) in (start..end).enumerate() {
                if window.is_some_and(|w| position.saturating_sub(past) >= w) {
                    scores[i] = f32::NEG_INFINITY;
                    continue;
                }
                let b_log = past / b_tokens;
                let slot = past % b_tokens;
                let phys_id = sequence.block_table[b_log];
                let block = &self.blocks[phys_id];
                let k_offset = (layer * b_tokens + slot) * self.kv_dim + offset;
                let k_slice = &block.key[k_offset..k_offset + head_dim];

                let mut score = ops::dot(query, k_slice) * scale;
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

                let b_log = past / b_tokens;
                let slot = past % b_tokens;
                let phys_id = sequence.block_table[b_log];
                let block = &self.blocks[phys_id];
                let v_offset = (layer * b_tokens + slot) * self.kv_dim + offset;
                let v_slice = &block.value[v_offset..v_offset + head_dim];

                for (slot_out, &val) in out.iter_mut().zip(v_slice) {
                    *slot_out += weight * val;
                }
            }
            max_score = next_max;
        }

        if denominator > 0.0 {
            for val in out {
                *val /= denominator;
            }
        }
        Ok(())
    }
}

/// A logical sequence mapping logical token positions to physical blocks in a [`PagedPool`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PagedSequence {
    /// Logical-to-physical block table.
    pub block_table: Vec<BlockId>,
    /// Number of valid tokens stored in this sequence.
    pub seq_len: usize,
}

impl PagedSequence {
    pub fn new() -> Self {
        Self {
            block_table: Vec::new(),
            seq_len: 0,
        }
    }

    /// Fork this sequence into a new child sequence with shared Copy-on-Write blocks.
    pub fn fork(&self, pool: &mut PagedPool) -> Self {
        for &block_id in &self.block_table {
            pool.inc_ref(block_id);
        }
        Self {
            block_table: self.block_table.clone(),
            seq_len: self.seq_len,
        }
    }

    /// Free all physical blocks referenced by this sequence back to the pool.
    pub fn free(&mut self, pool: &mut PagedPool) {
        for block_id in self.block_table.drain(..) {
            pool.free_block(block_id);
        }
        self.seq_len = 0;
    }

    /// Truncate sequence back to `keep_len` tokens, freeing unneeded trailing blocks.
    pub fn truncate(&mut self, pool: &mut PagedPool, keep_len: usize) {
        if keep_len >= self.seq_len {
            return;
        }
        let needed_blocks = keep_len.div_ceil(pool.tokens_per_block);
        while self.block_table.len() > needed_blocks {
            if let Some(block_id) = self.block_table.pop() {
                pool.free_block(block_id);
            }
        }
        self.seq_len = keep_len;
    }
}

impl crate::model::Model<'_> {
    /// Evaluate a single decode token using paged KV-cache storage.
    pub fn forward_paged(
        &self,
        pool: &mut PagedPool,
        seq: &mut PagedSequence,
        state: &mut crate::model::State,
        token: u32,
        position: usize,
    ) -> Result<()> {
        let c = &self.config;
        if position >= state.context() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "position {position} exceeds context window {}",
                    state.context()
                ),
            ));
        }
        validate_model_pool(c.layers, c.kv_dim(), pool)?;
        if position != seq.seq_len {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "paged decode position {position} does not match sequence length {}",
                    seq.seq_len
                ),
            ));
        }

        let crate::model::State {
            x,
            xb,
            xb2,
            hb,
            hb2,
            q,
            k,
            v,
            logits,
            workspace,
            threads,
            ..
        } = state;

        let (head_dim, kv_dim, group) = (c.head_dim, c.kv_dim(), c.group());
        let att_dim = c.heads * head_dim;

        self.token_embd.row(token as usize, x)?;
        if let Some(scale) = c.embedding_scale {
            for val in x.iter_mut() {
                *val *= scale;
            }
        }

        let is_sliding_layer = |idx: usize| (idx % 2 == 0) && c.sliding_window.is_some();
        let window = c.sliding_window.unwrap_or(usize::MAX);

        for (index, layer) in self.layers.iter().enumerate() {
            ops::rms_norm(x, &layer.attn_norm, c.eps, &mut xb[..c.embedding]);
            layer
                .wq
                .matvec_threaded(&xb[..c.embedding], q, *threads, workspace)?;
            if let Some(bq) = &layer.bq {
                for (slot, b) in q.iter_mut().zip(bq) {
                    *slot += b;
                }
            }
            if let Some(l) = &layer.lora_wq {
                l.apply(&xb[..c.embedding], q)?;
            }

            layer
                .wk
                .matvec_threaded(&xb[..c.embedding], k, *threads, workspace)?;
            if let Some(bk) = &layer.bk {
                for (slot, b) in k.iter_mut().zip(bk) {
                    *slot += b;
                }
            }
            if let Some(l) = &layer.lora_wk {
                l.apply(&xb[..c.embedding], k)?;
            }

            layer
                .wv
                .matvec_threaded(&xb[..c.embedding], v, *threads, workspace)?;
            if let Some(bv) = &layer.bv {
                for (slot, b) in v.iter_mut().zip(bv) {
                    *slot += b;
                }
            }
            if let Some(l) = &layer.lora_wv {
                l.apply(&xb[..c.embedding], v)?;
            }

            for head in 0..c.heads {
                ops::rope_scaled(
                    &mut q[head * head_dim..(head + 1) * head_dim],
                    position,
                    c.rope_base,
                    c.rope_dim,
                    c.rope_scaling.as_ref(),
                );
            }
            for head in 0..c.kv_heads {
                ops::rope_scaled(
                    &mut k[head * head_dim..(head + 1) * head_dim],
                    position,
                    c.rope_base,
                    c.rope_dim,
                    c.rope_scaling.as_ref(),
                );
            }

            pool.store(seq, index, position, k, v)?;

            let scale = 1.0 / (head_dim as f32).sqrt();
            let is_sliding = is_sliding_layer(index);
            for head in 0..c.heads {
                let kv_head = head / group;
                let query = &q[head * head_dim..(head + 1) * head_dim];
                let out = &mut xb[head * head_dim..(head + 1) * head_dim];

                pool.attention(
                    seq,
                    index,
                    query,
                    out,
                    AttentionParams {
                        row_width: kv_dim,
                        offset: kv_head * head_dim,
                        len: position + 1,
                        scale,
                        logit_cap: c.attn_logit_softcapping,
                        window: is_sliding.then_some(window),
                        position,
                        tile_size: 64,
                    },
                )?;
            }

            layer.wo.matvec_threaded(
                &xb[..att_dim],
                &mut xb2[..c.embedding],
                *threads,
                workspace,
            )?;
            if let Some(l) = &layer.lora_wo {
                l.apply(&xb[..att_dim], &mut xb2[..c.embedding])?;
            }
            if let Some(post_attn_norm) = &layer.post_attn_norm {
                ops::rms_norm(
                    &xb2[..c.embedding],
                    post_attn_norm,
                    c.eps,
                    &mut xb[..c.embedding],
                );
                for (x_i, xb_i) in x.iter_mut().zip(&xb[..c.embedding]) {
                    *x_i += xb_i;
                }
            } else {
                for (x_i, xb2_i) in x.iter_mut().zip(&xb2[..c.embedding]) {
                    *x_i += xb2_i;
                }
            }

            ops::rms_norm(x, &layer.ffn_norm, c.eps, &mut xb[..c.embedding]);
            layer
                .gate
                .matvec_threaded(&xb[..c.embedding], hb, *threads, workspace)?;
            if let Some(l) = &layer.lora_gate {
                l.apply(&xb[..c.embedding], hb)?;
            }
            layer
                .up
                .matvec_threaded(&xb[..c.embedding], hb2, *threads, workspace)?;
            if let Some(l) = &layer.lora_up {
                l.apply(&xb[..c.embedding], hb2)?;
            }
            for (gate, up) in hb.iter_mut().zip(hb2.iter()) {
                *gate = ops::silu(*gate) * up;
            }
            layer
                .down
                .matvec_threaded(hb, &mut xb2[..c.embedding], *threads, workspace)?;
            if let Some(l) = &layer.lora_down {
                l.apply(hb, &mut xb2[..c.embedding])?;
            }
            if let Some(post_ffn_norm) = &layer.post_ffn_norm {
                ops::rms_norm(
                    &xb2[..c.embedding],
                    post_ffn_norm,
                    c.eps,
                    &mut xb[..c.embedding],
                );
                for (x_i, xb_i) in x.iter_mut().zip(&xb[..c.embedding]) {
                    *x_i += xb_i;
                }
            } else {
                for (x_i, xb2_i) in x.iter_mut().zip(&xb2[..c.embedding]) {
                    *x_i += xb2_i;
                }
            }
        }

        ops::rms_norm(x, &self.output_norm, c.eps, &mut xb[..c.embedding]);
        self.output
            .matvec_threaded(&xb[..c.embedding], logits, *threads, workspace)?;
        if let Some(cap) = c.final_logit_softcapping {
            for logit in logits.iter_mut() {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        Ok(())
    }

    /// Prefill prompt tokens into paged KV-cache storage.
    pub fn prefill_paged(
        &self,
        pool: &mut PagedPool,
        seq: &mut PagedSequence,
        state: &mut crate::model::State,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let c = &self.config;
        validate_model_pool(c.layers, c.kv_dim(), pool)?;
        if start_pos != seq.seq_len {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "paged prefill start {start_pos} does not match sequence length {}",
                    seq.seq_len
                ),
            ));
        }
        let n = tokens.len();
        if start_pos + n > state.context() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "prefill past context window: start {start_pos} + {n} tokens > {}",
                    state.context()
                ),
            ));
        }

        let (head_dim, kv_dim, group) = (c.head_dim, c.kv_dim(), c.group());
        let att_dim = c.heads * head_dim;

        let mut x_batch = vec![0.0f32; n * c.embedding];
        let mut xb_batch = vec![0.0f32; n * c.embedding];
        let mut xb2_batch = vec![0.0f32; n * c.embedding];
        let mut att_batch = vec![0.0f32; n * att_dim];
        let mut q_batch = vec![0.0f32; n * att_dim];
        let mut k_batch = vec![0.0f32; n * kv_dim];
        let mut v_batch = vec![0.0f32; n * kv_dim];
        let mut hb_batch = vec![0.0f32; n * c.ffn];
        let mut hb2_batch = vec![0.0f32; n * c.ffn];

        let crate::model::State {
            logits,
            workspace,
            threads,
            ..
        } = state;

        for (b, &tok) in tokens.iter().enumerate() {
            let row = &mut x_batch[b * c.embedding..(b + 1) * c.embedding];
            self.token_embd.row(tok as usize, row)?;
            if let Some(scale) = c.embedding_scale {
                for v in row.iter_mut() {
                    *v *= scale;
                }
            }
        }

        let is_sliding_layer = |idx: usize| (idx % 2 == 0) && c.sliding_window.is_some();
        let window = c.sliding_window.unwrap_or(usize::MAX);

        for (index, layer) in self.layers.iter().enumerate() {
            for b in 0..n {
                let x_tok = &x_batch[b * c.embedding..(b + 1) * c.embedding];
                let xb_tok = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                ops::rms_norm(x_tok, &layer.attn_norm, c.eps, xb_tok);
            }

            layer
                .wq
                .matmul_threaded(&xb_batch, &mut q_batch, n, *threads, workspace)?;
            if let Some(bq) = &layer.bq {
                for b in 0..n {
                    let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                    for (slot, val) in q_tok.iter_mut().zip(bq) {
                        *slot += val;
                    }
                }
            }
            if let Some(l) = &layer.lora_wq {
                l.apply_batched_threaded(&xb_batch, n, &mut q_batch, *threads)?;
            }

            layer
                .wk
                .matmul_threaded(&xb_batch, &mut k_batch, n, *threads, workspace)?;
            if let Some(bk) = &layer.bk {
                for b in 0..n {
                    let k_tok = &mut k_batch[b * kv_dim..(b + 1) * kv_dim];
                    for (slot, val) in k_tok.iter_mut().zip(bk) {
                        *slot += val;
                    }
                }
            }
            if let Some(l) = &layer.lora_wk {
                l.apply_batched_threaded(&xb_batch, n, &mut k_batch, *threads)?;
            }

            layer
                .wv
                .matmul_threaded(&xb_batch, &mut v_batch, n, *threads, workspace)?;
            if let Some(bv) = &layer.bv {
                for b in 0..n {
                    let v_tok = &mut v_batch[b * kv_dim..(b + 1) * kv_dim];
                    for (slot, val) in v_tok.iter_mut().zip(bv) {
                        *slot += val;
                    }
                }
            }
            if let Some(l) = &layer.lora_wv {
                l.apply_batched_threaded(&xb_batch, n, &mut v_batch, *threads)?;
            }

            for b in 0..n {
                let pos = start_pos + b;
                let q_tok = &mut q_batch[b * att_dim..(b + 1) * att_dim];
                let k_tok = &mut k_batch[b * kv_dim..(b + 1) * kv_dim];
                let v_tok = &v_batch[b * kv_dim..(b + 1) * kv_dim];

                for head in 0..c.heads {
                    ops::rope_scaled(
                        &mut q_tok[head * head_dim..(head + 1) * head_dim],
                        pos,
                        c.rope_base,
                        c.rope_dim,
                        c.rope_scaling.as_ref(),
                    );
                }
                for head in 0..c.kv_heads {
                    ops::rope_scaled(
                        &mut k_tok[head * head_dim..(head + 1) * head_dim],
                        pos,
                        c.rope_base,
                        c.rope_dim,
                        c.rope_scaling.as_ref(),
                    );
                }

                pool.store(seq, index, pos, k_tok, v_tok)?;
            }

            let scale = 1.0 / (head_dim as f32).sqrt();
            let is_sliding = is_sliding_layer(index);
            for b in 0..n {
                let pos = start_pos + b;
                let q_tok = &q_batch[b * att_dim..(b + 1) * att_dim];
                let att_tok = &mut att_batch[b * att_dim..(b + 1) * att_dim];

                for head in 0..c.heads {
                    let kv_head = head / group;
                    let query = &q_tok[head * head_dim..(head + 1) * head_dim];
                    let out = &mut att_tok[head * head_dim..(head + 1) * head_dim];

                    pool.attention(
                        seq,
                        index,
                        query,
                        out,
                        AttentionParams {
                            row_width: kv_dim,
                            offset: kv_head * head_dim,
                            len: pos + 1,
                            scale,
                            logit_cap: c.attn_logit_softcapping,
                            window: is_sliding.then_some(window),
                            position: pos,
                            tile_size: 64,
                        },
                    )?;
                }
            }

            layer
                .wo
                .matmul_threaded(&att_batch, &mut xb2_batch, n, *threads, workspace)?;
            if let Some(l) = &layer.lora_wo {
                l.apply_batched_threaded(&att_batch, n, &mut xb2_batch, *threads)?;
            }
            if let Some(post_attn_norm) = &layer.post_attn_norm {
                for b in 0..n {
                    let xb2_tok = &xb2_batch[b * c.embedding..(b + 1) * c.embedding];
                    let xb_tok = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                    ops::rms_norm(xb2_tok, post_attn_norm, c.eps, xb_tok);
                }
                for (x_i, xb_i) in x_batch.iter_mut().zip(&xb_batch) {
                    *x_i += xb_i;
                }
            } else {
                for (x_i, xb2_i) in x_batch.iter_mut().zip(&xb2_batch) {
                    *x_i += xb2_i;
                }
            }

            for b in 0..n {
                let x_tok = &x_batch[b * c.embedding..(b + 1) * c.embedding];
                let xb_tok = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                ops::rms_norm(x_tok, &layer.ffn_norm, c.eps, xb_tok);
            }

            layer
                .gate
                .matmul_threaded(&xb_batch, &mut hb_batch, n, *threads, workspace)?;
            if let Some(l) = &layer.lora_gate {
                l.apply_batched_threaded(&xb_batch, n, &mut hb_batch, *threads)?;
            }
            layer
                .up
                .matmul_threaded(&xb_batch, &mut hb2_batch, n, *threads, workspace)?;
            if let Some(l) = &layer.lora_up {
                l.apply_batched_threaded(&xb_batch, n, &mut hb2_batch, *threads)?;
            }
            for (gate, up) in hb_batch.iter_mut().zip(hb2_batch.iter()) {
                *gate = ops::silu(*gate) * up;
            }
            layer
                .down
                .matmul_threaded(&hb_batch, &mut xb2_batch, n, *threads, workspace)?;
            if let Some(l) = &layer.lora_down {
                l.apply_batched_threaded(&hb_batch, n, &mut xb2_batch, *threads)?;
            }
            if let Some(post_ffn_norm) = &layer.post_ffn_norm {
                for b in 0..n {
                    let tok_in = &xb2_batch[b * c.embedding..(b + 1) * c.embedding];
                    let tok_out = &mut xb_batch[b * c.embedding..(b + 1) * c.embedding];
                    ops::rms_norm(tok_in, post_ffn_norm, c.eps, tok_out);
                }
                for (slot, delta) in x_batch.iter_mut().zip(xb_batch.iter()) {
                    *slot += delta;
                }
            } else {
                for (slot, delta) in x_batch.iter_mut().zip(xb2_batch.iter()) {
                    *slot += delta;
                }
            }
        }

        let last_x = &x_batch[(n - 1) * c.embedding..n * c.embedding];
        let mut last_normed = vec![0.0f32; c.embedding];
        ops::rms_norm(last_x, &self.output_norm, c.eps, &mut last_normed);
        self.output
            .matvec_threaded(&last_normed, logits, *threads, workspace)?;
        if let Some(cap) = c.final_logit_softcapping {
            for logit in logits.iter_mut() {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        Ok(())
    }
}

fn validate_model_pool(layers: usize, kv_dim: usize, pool: &PagedPool) -> Result<()> {
    if pool.layers != layers || pool.kv_dim != kv_dim || pool.tokens_per_block == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "paged KV pool dimensions do not match the model",
        ));
    }
    Ok(())
}
