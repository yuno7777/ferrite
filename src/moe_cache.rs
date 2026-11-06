//! Mixture-of-Experts (`MoE`) LRU Expert Cache & Router-Aware Prefetcher.
//!
//! Modern sparse MoE models (`Mixtral-8x7B`, `Qwen2-57B-A14B`, `DeepSeek-V2/V3`)
//! activate only top-$k$ experts per token out of $N$ total experts per layer.
//! However, natural language tokens exhibit strong temporal locality in router
//! selection ( typically >70% expert reuse across consecutive tokens in a domain).
//!
//! This module implements:
//! - [`ExpertKey`]: `(layer_idx, expert_idx)` identifier.
//! - [`ExpertCache`]: Bounded-capacity LRU cache tracking hot experts in resident
//!   working memory with hit/miss/eviction telemetry.
//! - [`ExpertRouterPrefetcher`]: Tracks per-layer expert activation frequencies and
//!   transition probabilities (`expert_t -> expert_{t+1}`) to predict and prefetch
//!   the most likely experts for the next layer or next token before router dispatch.

/// Unique key identifying a single expert inside a specific Transformer layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: usize,
    pub expert: usize,
}

/// Telemetry counters for [`ExpertCache`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub prefetches: u64,
}

impl ExpertCacheStats {
    /// Compute the cache hit ratio in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[derive(Clone, Debug)]
struct CacheEntry<T> {
    key: ExpertKey,
    value: T,
    last_used_tick: u64,
    pin_count: u32,
}

/// Bounded LRU cache for Mixture-of-Experts weight slices or dequantized tiles.
#[derive(Clone, Debug)]
pub struct ExpertCache<T> {
    capacity: usize,
    tick: u64,
    entries: Vec<CacheEntry<T>>,
    stats: ExpertCacheStats,
}

impl<T> ExpertCache<T> {
    /// Create a new expert cache holding at most `capacity` experts simultaneously.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ExpertCache capacity must be > 0");
        Self {
            capacity,
            tick: 0,
            entries: Vec::with_capacity(capacity),
            stats: ExpertCacheStats::default(),
        }
    }

    /// Number of currently cached experts.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum number of experts the cache can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Snapshot of cache hit/miss/eviction telemetry.
    pub fn stats(&self) -> ExpertCacheStats {
        self.stats
    }

    /// Check whether `key` is currently resident without updating LRU recency.
    pub fn contains(&self, key: ExpertKey) -> bool {
        self.entries.iter().any(|e| e.key == key)
    }

    /// Look up `key` in the cache, updating its LRU timestamp on hit.
    pub fn get(&mut self, key: ExpertKey) -> Option<&T> {
        self.tick += 1;
        if let Some(idx) = self.entries.iter().position(|e| e.key == key) {
            self.entries[idx].last_used_tick = self.tick;
            self.stats.hits += 1;
            Some(&self.entries[idx].value)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Retrieve a cached expert or load it on demand via `loader`, evicting the
    /// least-recently-used unpinned entry if full.
    pub fn get_or_insert_with<F>(&mut self, key: ExpertKey, loader: F) -> &T
    where
        F: FnOnce(ExpertKey) -> T,
    {
        self.tick += 1;
        if let Some(idx) = self.entries.iter().position(|e| e.key == key) {
            self.entries[idx].last_used_tick = self.tick;
            self.stats.hits += 1;
            return &self.entries[idx].value;
        }

        self.stats.misses += 1;
        let val = loader(key);
        let idx = self.insert_internal(key, val);
        &self.entries[idx].value
    }

    /// Speculatively prefetch `key` into the cache if not already present.
    pub fn prefetch_with<F>(&mut self, key: ExpertKey, loader: F)
    where
        F: FnOnce(ExpertKey) -> T,
    {
        if self.contains(key) {
            return;
        }
        self.tick += 1;
        self.stats.prefetches += 1;
        let val = loader(key);
        let _ = self.insert_internal(key, val);
    }

    /// Pin an expert so it cannot be evicted while active.
    pub fn pin(&mut self, key: ExpertKey) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.key == key) {
            entry.pin_count += 1;
            true
        } else {
            false
        }
    }

    /// Unpin a previously pinned expert.
    pub fn unpin(&mut self, key: ExpertKey) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.key == key) {
            entry.pin_count = entry.pin_count.saturating_sub(1);
            true
        } else {
            false
        }
    }

    fn insert_internal(&mut self, key: ExpertKey, value: T) -> usize {
        if self.entries.len() >= self.capacity {
            // Find unpinned entry with smallest last_used_tick (or fallback to oldest overall)
            let evict_idx = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.pin_count == 0)
                .min_by_key(|(_, e)| e.last_used_tick)
                .map(|(i, _)| i)
                .unwrap_or_else(|| {
                    self.entries
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, e)| e.last_used_tick)
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                });

            self.entries[evict_idx] = CacheEntry {
                key,
                value,
                last_used_tick: self.tick,
                pin_count: 0,
            };
            self.stats.evictions += 1;
            evict_idx
        } else {
            let idx = self.entries.len();
            self.entries.push(CacheEntry {
                key,
                value,
                last_used_tick: self.tick,
                pin_count: 0,
            });
            idx
        }
    }
}

/// Router-aware cross-layer and temporal expert predictor.
///
/// Tracks:
/// 1. Per-layer activation counts `freq[layer][expert]`.
/// 2. Cross-layer conditional transitions `trans[layer][e_cur * n_experts + e_next]`
///    recording how often choosing `e_cur` in `layer` is followed by choosing
///    `e_next` in `layer + 1`.
#[derive(Clone, Debug)]
pub struct ExpertRouterPrefetcher {
    pub n_layers: usize,
    pub n_experts: usize,
    freq: Vec<u64>,
    cross_layer_trans: Vec<u32>,
    last_layer_active: Vec<Option<usize>>,
}

impl ExpertRouterPrefetcher {
    /// Construct a new router prefetcher for `n_layers` and `n_experts` per layer.
    pub fn new(n_layers: usize, n_experts: usize) -> Self {
        assert!(n_layers > 0 && n_experts > 0);
        Self {
            n_layers,
            n_experts,
            freq: vec![0; n_layers * n_experts],
            cross_layer_trans: vec![0; n_layers.saturating_sub(1) * n_experts * n_experts],
            last_layer_active: vec![None; n_layers],
        }
    }

    /// Record the top-$k$ selected experts for `layer` on the current token.
    pub fn record_dispatch(&mut self, layer: usize, selected_experts: &[usize]) {
        if layer >= self.n_layers || selected_experts.is_empty() {
            return;
        }
        for &e in selected_experts {
            if e < self.n_experts {
                self.freq[layer * self.n_experts + e] += 1;
            }
        }
        let primary = selected_experts[0];
        if primary < self.n_experts {
            if layer > 0 {
                if let Some(prev_e) = self.last_layer_active[layer - 1] {
                    let base = (layer - 1) * self.n_experts * self.n_experts;
                    let idx = base + prev_e * self.n_experts + primary;
                    self.cross_layer_trans[idx] = self.cross_layer_trans[idx].saturating_add(1);
                }
            }
            self.last_layer_active[layer] = Some(primary);
        }
    }

    /// Predict the top `top_n` experts most likely to be needed in `layer + 1`
    /// given that `current_experts` were selected in `layer`.
    pub fn predict_next_layer(
        &self,
        layer: usize,
        current_experts: &[usize],
        top_n: usize,
    ) -> Vec<ExpertKey> {
        let next_layer = layer + 1;
        if next_layer >= self.n_layers || top_n == 0 {
            return Vec::new();
        }
        let n_exp = self.n_experts;
        let base = layer * n_exp * n_exp;
        let mut scores = vec![0u64; n_exp];

        for &e_cur in current_experts {
            if e_cur < n_exp {
                for (e_next, s) in scores.iter_mut().enumerate().take(n_exp) {
                    *s += self.cross_layer_trans[base + e_cur * n_exp + e_next] as u64 * 100;
                }
            }
        }
        // Tie-break with global frequency in `next_layer`
        for (e_next, s) in scores.iter_mut().enumerate().take(n_exp) {
            *s += self.freq[next_layer * n_exp + e_next];
        }

        let mut ranked: Vec<(usize, u64)> = scores.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .take(top_n.min(n_exp))
            .map(|(expert, _)| ExpertKey {
                layer: next_layer,
                expert,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_expert_cache_evicts_least_recently_used_and_respects_pins() {
        let mut cache = ExpertCache::new(3);
        let k0 = ExpertKey {
            layer: 0,
            expert: 0,
        };
        let k1 = ExpertKey {
            layer: 0,
            expert: 1,
        };
        let k2 = ExpertKey {
            layer: 0,
            expert: 2,
        };
        let k3 = ExpertKey {
            layer: 0,
            expert: 3,
        };

        assert_eq!(*cache.get_or_insert_with(k0, |_| 100), 100);
        assert_eq!(*cache.get_or_insert_with(k1, |_| 101), 101);
        assert_eq!(*cache.get_or_insert_with(k2, |_| 102), 102);

        // Touch k0 so k1 becomes oldest; pin k1 so k2 must be evicted instead
        assert_eq!(cache.get(k0), Some(&100));
        assert!(cache.pin(k1));

        assert_eq!(*cache.get_or_insert_with(k3, |_| 103), 103);
        assert!(cache.contains(k0));
        assert!(cache.contains(k1));
        assert!(!cache.contains(k2));
        assert!(cache.contains(k3));
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 4);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn router_prefetcher_learns_cross_layer_transitions() {
        let mut prefetcher = ExpertRouterPrefetcher::new(4, 8);
        for _ in 0..10 {
            prefetcher.record_dispatch(0, &[2, 5]);
            prefetcher.record_dispatch(1, &[6, 1]);
        }
        let preds = prefetcher.predict_next_layer(0, &[2], 2);
        assert_eq!(preds.len(), 2);
        assert_eq!(
            preds[0],
            ExpertKey {
                layer: 1,
                expert: 6
            }
        );
    }
}
