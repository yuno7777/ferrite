//! Radix tree prefix caching and prompt deduplication.
//!
//! Repeated system prompts, few-shot exemplars, agentic scratchpads, and
//! multi-turn dialogues share identical token prefixes. The radix tree cache
//! indexes previously computed KV blocks by exact token sequences, enabling
//! instantaneous (0 ms) prefill reuse across requests and dynamic LRU eviction
//! under memory pressure.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::paged::{BlockId, PagedPool, PagedSequence, DEFAULT_BLOCK_TOKENS};

static ACCESS_TICK: AtomicU64 = AtomicU64::new(1);

fn next_tick() -> u64 {
    ACCESS_TICK.fetch_add(1, Ordering::Relaxed)
}

/// A node in the Radix prefix tree storing one token block and its physical block ID.
#[derive(Clone, Debug)]
pub struct RadixNode {
    /// Token sequence represented by this block.
    pub tokens: Vec<u32>,
    /// Physical block in [`PagedPool`] storing keys and values for these tokens.
    pub block_id: BlockId,
    /// Child branches descending from this prefix.
    pub children: Vec<RadixNode>,
    /// Logical timestamp of the last cache hit/read, used for LRU eviction.
    pub last_accessed: u64,
}

impl RadixNode {
    pub fn new(tokens: Vec<u32>, block_id: BlockId) -> Self {
        Self {
            tokens,
            block_id,
            children: Vec::new(),
            last_accessed: next_tick(),
        }
    }
}

/// Radix tree cache for deduplicating and indexing KV-cache prefixes.
#[derive(Clone, Debug, Default)]
pub struct RadixCache {
    /// Root children representing first-block prefix candidates.
    pub roots: Vec<RadixNode>,
    /// Block size in tokens (must match the pool's `tokens_per_block`).
    pub tokens_per_block: usize,
}

impl RadixCache {
    /// Create a new prefix cache with default block size (16 tokens).
    pub fn new() -> Self {
        Self::with_block_size(DEFAULT_BLOCK_TOKENS)
    }

    /// Create a new prefix cache with a custom token block size.
    pub fn with_block_size(tokens_per_block: usize) -> Self {
        Self {
            roots: Vec::new(),
            tokens_per_block,
        }
    }

    /// Match a prompt against cached prefixes.
    ///
    /// Returns:
    /// - `matched_blocks`: List of physical `BlockId`s ready to be bound to a [`PagedSequence`].
    /// - `matched_tokens`: Total number of tokens matched from the start of `prompt`.
    pub fn match_prefix(&mut self, prompt: &[u32]) -> (Vec<BlockId>, usize) {
        let mut matched_blocks = Vec::new();
        let mut matched_tokens = 0;
        let mut pos = 0;
        let b = self.tokens_per_block;

        let mut current_children = &mut self.roots;

        while pos + b <= prompt.len() {
            let chunk = &prompt[pos..pos + b];
            let mut found_index = None;

            for (idx, child) in current_children.iter().enumerate() {
                if child.tokens == chunk {
                    found_index = Some(idx);
                    break;
                }
            }

            if let Some(idx) = found_index {
                let child = &mut current_children[idx];
                child.last_accessed = next_tick();
                matched_blocks.push(child.block_id);
                matched_tokens += b;
                pos += b;
                current_children = &mut child.children;
            } else {
                break;
            }
        }

        (matched_blocks, matched_tokens)
    }

    /// Insert newly evaluated prompt blocks into the radix tree.
    ///
    /// Increments reference counts in `pool` for newly indexed blocks so they are
    /// protected from premature deallocation while stored in cache.
    pub fn insert(&mut self, prompt: &[u32], block_table: &[BlockId], pool: &mut PagedPool) {
        let b = self.tokens_per_block;
        let num_blocks = prompt.len() / b;
        let blocks_to_insert = num_blocks.min(block_table.len());

        let mut current_children = &mut self.roots;
        let mut pos = 0;

        for &phys_id in &block_table[..blocks_to_insert] {
            let chunk = &prompt[pos..pos + b];

            let mut found_index = None;
            for (idx, child) in current_children.iter().enumerate() {
                if child.tokens == chunk {
                    found_index = Some(idx);
                    break;
                }
            }

            if let Some(idx) = found_index {
                current_children[idx].last_accessed = next_tick();
                current_children = &mut current_children[idx].children;
            } else {
                pool.inc_ref(phys_id);
                let new_node = RadixNode::new(chunk.to_vec(), phys_id);
                current_children.push(new_node);
                let last = current_children.len() - 1;
                current_children = &mut current_children[last].children;
            }
            pos += b;
        }
    }

    /// Total number of prefix blocks currently indexed in the cache.
    pub fn len(&self) -> usize {
        fn count_nodes(nodes: &[RadixNode]) -> usize {
            let mut sum = nodes.len();
            for node in nodes {
                sum += count_nodes(&node.children);
            }
            sum
        }
        count_nodes(&self.roots)
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Evict `target_count` least-recently-used leaf blocks from the cache to relieve memory pressure.
    pub fn evict_lru(&mut self, pool: &mut PagedPool, target_count: usize) -> usize {
        let mut evicted = 0;
        for _ in 0..target_count {
            if !self.evict_one_leaf(pool) {
                break;
            }
            evicted += 1;
        }
        evicted
    }

    fn evict_one_leaf(&mut self, pool: &mut PagedPool) -> bool {
        // Collect candidate leaf paths with their access timestamps
        let mut oldest_time = u64::MAX;
        let mut oldest_path = Vec::new();

        fn find_oldest_leaf(
            nodes: &[RadixNode],
            path: &mut Vec<usize>,
            best_time: &mut u64,
            best_path: &mut Vec<usize>,
        ) {
            for (i, node) in nodes.iter().enumerate() {
                path.push(i);
                if node.children.is_empty() {
                    if node.last_accessed < *best_time {
                        *best_time = node.last_accessed;
                        *best_path = path.clone();
                    }
                } else {
                    find_oldest_leaf(&node.children, path, best_time, best_path);
                }
                path.pop();
            }
        }

        let mut path_scratch = Vec::new();
        find_oldest_leaf(
            &self.roots,
            &mut path_scratch,
            &mut oldest_time,
            &mut oldest_path,
        );

        if oldest_path.is_empty() {
            return false;
        }

        // Traverse down the path and remove the leaf
        let mut current = &mut self.roots;
        for &idx in &oldest_path[..oldest_path.len() - 1] {
            current = &mut current[idx].children;
        }
        let leaf_idx = *oldest_path.last().unwrap();
        let removed = current.swap_remove(leaf_idx);
        pool.free_block(removed.block_id);
        true
    }

    /// Bind matched prefix blocks to a sequence and skip recomputing them.
    pub fn apply_prefix_to_sequence(
        &self,
        matched_blocks: &[BlockId],
        matched_tokens: usize,
        seq: &mut PagedSequence,
        pool: &mut PagedPool,
    ) {
        seq.free(pool);
        for &b in matched_blocks {
            pool.inc_ref(b);
            seq.block_table.push(b);
        }
        seq.seq_len = matched_tokens;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_cache_empty_matches_zero() {
        let mut cache = RadixCache::with_block_size(4);
        let prompt = vec![1, 2, 3, 4, 5];
        let (blocks, tokens) = cache.match_prefix(&prompt);
        assert!(blocks.is_empty());
        assert_eq!(tokens, 0);
    }

    #[test]
    fn prefix_cache_insert_and_exact_match() {
        let mut pool = PagedPool::new(1, 4, 4, 16);
        let mut cache = RadixCache::with_block_size(4);

        let b1 = pool.allocate_block();
        let b2 = pool.allocate_block();

        let prompt = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let block_table = vec![b1, b2];

        cache.insert(&prompt, &block_table, &mut pool);
        assert_eq!(cache.len(), 2);
        // allocate_block set ref_count to 1, insert inc_ref to 2
        assert_eq!(pool.ref_counts[b1], 2);
        assert_eq!(pool.ref_counts[b2], 2);

        // Matching full 8 tokens
        let (blocks, matched) = cache.match_prefix(&prompt);
        assert_eq!(blocks, vec![b1, b2]);
        assert_eq!(matched, 8);

        // Matching prefix of 5 tokens (first block matched, 5th token partial)
        let prompt_partial = vec![10, 20, 30, 40, 99];
        let (blocks_part, matched_part) = cache.match_prefix(&prompt_partial);
        assert_eq!(blocks_part, vec![b1]);
        assert_eq!(matched_part, 4);
    }

    #[test]
    fn prefix_cache_branching() {
        let mut pool = PagedPool::new(1, 4, 4, 16);
        let mut cache = RadixCache::with_block_size(4);

        let b_root = pool.allocate_block();
        let b_a = pool.allocate_block();
        let b_b = pool.allocate_block();

        // System prompt: tokens [1, 2, 3, 4]
        // Branch A: [1, 2, 3, 4] + [10, 11, 12, 13]
        // Branch B: [1, 2, 3, 4] + [20, 21, 22, 23]
        let branch_a = vec![1, 2, 3, 4, 10, 11, 12, 13];
        let branch_b = vec![1, 2, 3, 4, 20, 21, 22, 23];

        cache.insert(&branch_a, &[b_root, b_a], &mut pool);
        cache.insert(&branch_b, &[b_root, b_b], &mut pool);

        assert_eq!(cache.len(), 3); // 1 shared root + 2 children
        assert_eq!(pool.ref_counts[b_root], 2); // allocated (1) + cached once (1)
        assert_eq!(pool.ref_counts[b_a], 2);
        assert_eq!(pool.ref_counts[b_b], 2);

        let (out_a, t_a) = cache.match_prefix(&branch_a);
        assert_eq!(out_a, vec![b_root, b_a]);
        assert_eq!(t_a, 8);

        let (out_b, t_b) = cache.match_prefix(&branch_b);
        assert_eq!(out_b, vec![b_root, b_b]);
        assert_eq!(t_b, 8);
    }

    #[test]
    fn prefix_cache_lru_eviction() {
        let mut pool = PagedPool::new(1, 4, 4, 16);
        let mut cache = RadixCache::with_block_size(4);

        let b1 = pool.allocate_block();
        let b2 = pool.allocate_block();

        cache.insert(&[1, 2, 3, 4], &[b1], &mut pool);
        cache.insert(&[5, 6, 7, 8], &[b2], &mut pool);
        assert_eq!(cache.len(), 2);

        // Release sequence-level ownership so cache is sole owner
        pool.free_block(b1);
        pool.free_block(b2);
        assert_eq!(pool.ref_counts[b1], 1);
        assert_eq!(pool.ref_counts[b2], 1);

        // Evict 1 LRU block
        let evicted = cache.evict_lru(&mut pool, 1);
        assert_eq!(evicted, 1);
        assert_eq!(cache.len(), 1);
        assert_eq!(pool.free_blocks(), 15); // b1 freed back to pool, b2 still cached
    }
}
