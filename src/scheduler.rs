//! Continuous batching and multi-sequence scheduling engine.
//!
//! Concurrently schedules multiple heterogeneous inference requests (varying prompt lengths,
//! arrival times, and decode horizons) in shared forward execution passes using paged KV-cache
//! memory management and radix tree prefix caching.

use std::collections::VecDeque;
use std::io::Result;

use crate::grammar::Grammar;
use crate::model::{Config, Model, State};
use crate::paged::{PagedPool, PagedSequence, DEFAULT_BLOCK_TOKENS};
use crate::prefix::RadixCache;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;

/// Unique identifier for an inference request sequence.
pub type SequenceId = u64;

/// Status and lifecycle phase of an inference sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Waiting in scheduling queue for memory / batch capacity.
    Waiting,
    /// Ingesting prompt tokens across one or more chunks.
    Prefilling { progress: usize },
    /// Generating autoregressive completion tokens step-by-step.
    Running,
    /// Sequence generation terminated.
    Finished(FinishReason),
}

/// Reason why sequence generation completed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// Model emitted End-Of-Sequence token.
    Eos,
    /// Generation reached user-specified max_tokens limit.
    LengthLimit,
    /// Explicit cancellation.
    Cancelled,
}

/// Request parameters to submit a new sequence to the scheduler.
#[derive(Clone, Debug)]
pub struct SequenceRequest {
    pub id: SequenceId,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampler: Sampler,
    pub grammar: Option<Grammar>,
}

/// A managed sequence within the continuous batch scheduler.
#[derive(Clone, Debug)]
pub struct Sequence {
    pub id: SequenceId,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub generated: Vec<u32>,
    pub paged_seq: PagedSequence,
    pub sampler: Sampler,
    pub grammar: Option<Grammar>,
    pub status: SequenceStatus,
    pub last_token: Option<u32>,
}

impl Sequence {
    pub fn new(req: SequenceRequest) -> Self {
        Self {
            id: req.id,
            prompt: req.prompt,
            max_tokens: req.max_tokens,
            generated: Vec::new(),
            paged_seq: PagedSequence::new(),
            sampler: req.sampler,
            grammar: req.grammar,
            status: SequenceStatus::Waiting,
            last_token: None,
        }
    }
}

/// Scheduler tuning and resource limits.
#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    /// Maximum concurrent sequences evaluated in a single step.
    pub max_batch_size: usize,
    /// Maximum token budget per step combining prefill and decode tokens.
    pub max_tokens_per_step: usize,
    /// Maximum chunk size for prompt prefill in tokens.
    pub prefill_chunk_size: usize,
    /// Token block size in paged memory.
    pub tokens_per_block: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_tokens_per_step: 512,
            prefill_chunk_size: 128,
            tokens_per_block: DEFAULT_BLOCK_TOKENS,
        }
    }
}

/// Output event emitted by the scheduler on each step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepOutput {
    pub sequence_id: SequenceId,
    pub token: u32,
    pub is_finished: bool,
    pub finish_reason: Option<FinishReason>,
}

/// Dynamic continuous batch scheduler coordinating paged memory and model forward passes.
pub struct BatchScheduler {
    pub pool: PagedPool,
    pub prefix_cache: RadixCache,
    pub waiting: VecDeque<Sequence>,
    pub active: Vec<Sequence>,
    pub finished: Vec<(SequenceId, Vec<u32>, FinishReason)>,
    pub config: SchedulerConfig,
    state: State,
}

impl BatchScheduler {
    /// Initialize a continuous batch scheduler.
    pub fn new(
        model_config: &Config,
        initial_blocks: usize,
        scheduler_config: SchedulerConfig,
    ) -> Self {
        let pool = PagedPool::new(
            model_config.layers,
            model_config.kv_dim(),
            scheduler_config.tokens_per_block,
            initial_blocks,
        );
        let prefix_cache = RadixCache::with_block_size(scheduler_config.tokens_per_block);
        let state = State::new(model_config);

        Self {
            pool,
            prefix_cache,
            waiting: VecDeque::new(),
            active: Vec::new(),
            finished: Vec::new(),
            config: scheduler_config,
            state,
        }
    }

    /// Submit a new sequence request to the queue.
    pub fn add_sequence(&mut self, request: SequenceRequest) -> SequenceId {
        let id = request.id;
        self.waiting.push_back(Sequence::new(request));
        id
    }

    /// Check if all waiting and active work has completed.
    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty() && self.active.is_empty()
    }

    /// Number of sequences currently running in the active batch.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Number of sequences waiting in queue.
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Take all completed sequences and their generated token histories.
    pub fn take_finished(&mut self) -> Vec<(SequenceId, Vec<u32>, FinishReason)> {
        std::mem::take(&mut self.finished)
    }

    /// Execute one continuous batching scheduling cycle.
    pub fn step(&mut self, model: &Model, tokenizer: &Tokenizer) -> Result<Vec<StepOutput>> {
        let mut outputs = Vec::new();

        // 1. Admit waiting sequences into active batch if capacity exists
        while self.active.len() < self.config.max_batch_size && !self.waiting.is_empty() {
            let mut seq = self.waiting.pop_front().unwrap();

            // Check prefix cache for reusable KV prefix
            let (matched_blocks, matched_tokens) = self.prefix_cache.match_prefix(&seq.prompt);
            if matched_tokens > 0 {
                self.prefix_cache.apply_prefix_to_sequence(
                    &matched_blocks,
                    matched_tokens,
                    &mut seq.paged_seq,
                    &mut self.pool,
                );
            }

            seq.status = SequenceStatus::Prefilling {
                progress: matched_tokens,
            };
            self.active.push(seq);
        }

        let mut token_budget = self.config.max_tokens_per_step;
        let mut newly_finished = Vec::new();

        // 2. Run single-token decode for sequences in Running state
        for (idx, seq) in self.active.iter_mut().enumerate() {
            if token_budget == 0 {
                break;
            }
            if let SequenceStatus::Running = seq.status {
                let last = seq.last_token.unwrap_or(0);
                let pos = seq.paged_seq.seq_len;

                model.forward_paged(
                    &mut self.pool,
                    &mut seq.paged_seq,
                    &mut self.state,
                    last,
                    pos,
                )?;
                token_budget -= 1;

                if let Some(grammar) = &seq.grammar {
                    grammar.filter_logits(&mut self.state.logits, tokenizer);
                }

                let next_token = seq
                    .sampler
                    .sample_with_history(&mut self.state.logits, &seq.generated);

                if let Some(grammar) = &mut seq.grammar {
                    grammar.advance_token(next_token, tokenizer);
                }

                seq.generated.push(next_token);
                seq.last_token = Some(next_token);

                let is_eos = Some(next_token) == tokenizer.eos;
                let is_limit = seq.generated.len() >= seq.max_tokens;

                if is_eos || is_limit {
                    let reason = if is_eos {
                        FinishReason::Eos
                    } else {
                        FinishReason::LengthLimit
                    };
                    seq.status = SequenceStatus::Finished(reason);
                    newly_finished.push(idx);

                    outputs.push(StepOutput {
                        sequence_id: seq.id,
                        token: next_token,
                        is_finished: true,
                        finish_reason: Some(reason),
                    });
                } else {
                    outputs.push(StepOutput {
                        sequence_id: seq.id,
                        token: next_token,
                        is_finished: false,
                        finish_reason: None,
                    });
                }
            }
        }

        // 3. Run chunked prefill for sequences in Prefilling state
        for (idx, seq) in self.active.iter_mut().enumerate() {
            if token_budget == 0 {
                break;
            }
            if let SequenceStatus::Prefilling { progress } = &mut seq.status {
                let start = *progress;
                let chunk_len = (seq.prompt.len() - start)
                    .min(self.config.prefill_chunk_size)
                    .min(token_budget);

                if chunk_len > 0 {
                    let chunk = &seq.prompt[start..start + chunk_len];
                    model.prefill_paged(
                        &mut self.pool,
                        &mut seq.paged_seq,
                        &mut self.state,
                        chunk,
                        start,
                    )?;
                    *progress += chunk_len;
                    token_budget -= chunk_len;
                }

                // If prefill completed, generate first completion token
                if *progress == seq.prompt.len() {
                    // Cache completed prompt blocks into radix tree
                    self.prefix_cache.insert(
                        &seq.prompt,
                        &seq.paged_seq.block_table,
                        &mut self.pool,
                    );

                    if let Some(grammar) = &seq.grammar {
                        grammar.filter_logits(&mut self.state.logits, tokenizer);
                    }

                    let first_token = seq
                        .sampler
                        .sample_with_history(&mut self.state.logits, &seq.generated);

                    if let Some(grammar) = &mut seq.grammar {
                        grammar.advance_token(first_token, tokenizer);
                    }

                    seq.generated.push(first_token);
                    seq.last_token = Some(first_token);

                    let is_eos = Some(first_token) == tokenizer.eos;
                    let is_limit = seq.generated.len() >= seq.max_tokens;

                    if is_eos || is_limit {
                        let reason = if is_eos {
                            FinishReason::Eos
                        } else {
                            FinishReason::LengthLimit
                        };
                        seq.status = SequenceStatus::Finished(reason);
                        newly_finished.push(idx);

                        outputs.push(StepOutput {
                            sequence_id: seq.id,
                            token: first_token,
                            is_finished: true,
                            finish_reason: Some(reason),
                        });
                    } else {
                        seq.status = SequenceStatus::Running;
                        outputs.push(StepOutput {
                            sequence_id: seq.id,
                            token: first_token,
                            is_finished: false,
                            finish_reason: None,
                        });
                    }
                }
            }
        }

        // 4. Clean up finished sequences and return their blocks to pool
        if !newly_finished.is_empty() {
            // Sort descending to safely swap_remove
            newly_finished.sort_unstable();
            newly_finished.dedup();
            for &idx in newly_finished.iter().rev() {
                let mut seq = self.active.swap_remove(idx);
                let reason = match seq.status {
                    SequenceStatus::Finished(r) => r,
                    _ => FinishReason::Cancelled,
                };
                seq.paged_seq.free(&mut self.pool);
                self.finished.push((seq.id, seq.generated, reason));
            }
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_initialization() {
        let config = Config {
            arch: "llama".into(),
            layers: 2,
            embedding: 64,
            heads: 4,
            kv_heads: 4,
            head_dim: 16,
            ffn: 128,
            context: 512,
            vocab: 100,
            eps: 1e-5,
            rope_base: 10000.0,
            rope_dim: 16,
            rope_scaling: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            sliding_window: None,
            embedding_scale: None,
            expert_count: None,
            expert_used_count: None,
            expert_shared_count: None,
            q_lora_rank: None,
            kv_lora_rank: None,
            qk_nope_head_dim: None,
            qk_rope_head_dim: None,
            v_head_dim: None,
        };

        let scheduler = BatchScheduler::new(&config, 32, SchedulerConfig::default());
        assert!(scheduler.is_empty());
        assert_eq!(scheduler.active_count(), 0);
        assert_eq!(scheduler.waiting_count(), 0);
    }
}
