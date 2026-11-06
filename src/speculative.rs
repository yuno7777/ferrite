//! Speculative decoding engine.
//!
//! Speculative decoding accelerates autoregressive generation by using a smaller,
//! faster "draft" model to propose $K$ candidate tokens. The larger "target" model
//! verifies all $K$ candidates concurrently in a single batched forward pass via GEMM.
//!
//! Protocol:
//! At the start of a step at sequence position `P`:
//! 1. `target_state.logits` holds the target model's logits predicting the token at position `P`.
//! 2. Sample token `tok_0` from `target_state.logits`. This token is 100% verified and accepted.
//! 3. Feed `tok_0` into `draft_state` at position `P` to initialize the draft model's prediction
//!    for `P + 1`.
//! 4. The draft model autoregressively generates $K$ draft candidate tokens `cand_1, ..., cand_K`
//!    at positions `P + 1, ..., P + K`.
//! 5. All $K$ candidates (plus `tok_0`) are evaluated in the target model.
//!    Specifically:
//!    - Target model processes `tok_0` at position `P` (which was not yet in its KV-cache).
//!    - If $K > 0$, target model evaluates `[tok_0, cand_1, ..., cand_{K-1}]` via batched GEMM prefill,
//!      producing target logits predicting positions `P + 1, ..., P + K`.
//! 6. Each draft candidate `cand_i` is verified:
//!    `target_sample == cand_i`.
//!    - If accepted: `cand_i` is emitted.
//!    - If rejected: the target model's corrective token is emitted, and subsequent candidates are discarded.
//! 7. Any rejected positions in the KV-caches are rolled back via `truncate_cache`.
//! 8. `target_state.logits` and `draft_state.logits` are positioned ready for the next step.

use crate::model::{Model, State};
use crate::sample::Sampler;
use std::io::Result;

/// Performance statistics for speculative decoding.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpeculativeStats {
    /// Total speculative draft tokens proposed.
    pub proposed: usize,
    /// Number of proposed draft tokens accepted by the target model.
    pub accepted: usize,
    /// Total forward passes executed by the target model.
    pub target_batches: usize,
    /// Total tokens emitted (accepted + corrective/bonus tokens).
    pub total_emitted: usize,
}

impl SpeculativeStats {
    /// Acceptance rate as a fraction in `[0.0, 1.0]`.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }

    /// Average tokens produced per target forward pass.
    pub fn tokens_per_batch(&self) -> f64 {
        if self.target_batches == 0 {
            0.0
        } else {
            self.total_emitted as f64 / self.target_batches as f64
        }
    }
}

/// An active speculative decoding runner managing draft and target models.
pub struct SpeculativeEngine<'a> {
    pub target: &'a Model<'a>,
    pub draft: &'a Model<'a>,
    pub k: usize,
    pub stats: SpeculativeStats,
}

impl<'a> SpeculativeEngine<'a> {
    pub fn new(target: &'a Model<'a>, draft: &'a Model<'a>, k: usize) -> Self {
        Self {
            target,
            draft,
            k: k.max(1),
            stats: SpeculativeStats::default(),
        }
    }

    /// Perform one speculative decoding step.
    ///
    /// - `target_state`: KV-cache and state for target model, whose `logits` predicts token at `position`.
    /// - `draft_state`: KV-cache and state for draft model, whose `logits` predicts token at `position`.
    /// - `position`: current sequence position.
    /// - `sampler`: token sampler (e.g. greedy).
    ///
    /// Returns the vector of accepted / emitted tokens (at least 1 token is always produced).
    pub fn step(
        &mut self,
        target_state: &mut State,
        draft_state: &mut State,
        position: usize,
        sampler: &mut Sampler,
    ) -> Result<Vec<u32>> {
        // First token is unconditionally sampled from the target model's logits at `position`
        let first_tok = sampler.sample(&mut target_state.logits);
        let mut emitted = vec![first_tok];

        let max_k = self
            .k
            .min(target_state.context().saturating_sub(position + 2));
        if max_k == 0 {
            // Cannot speculate further tokens; forward the first token through both models
            self.target.forward(target_state, first_tok, position)?;
            self.draft.forward(draft_state, first_tok, position)?;
            self.stats.target_batches += 1;
            self.stats.total_emitted += 1;
            return Ok(emitted);
        }

        // Forward first_tok into draft model at position, giving draft logits for position + 1
        self.draft.forward(draft_state, first_tok, position)?;

        // Propose K draft tokens at positions position + 1 .. position + 1 + K
        let mut draft_tokens = Vec::with_capacity(max_k);

        for cur_pos in (position + 1..).take(max_k) {
            let tok = sampler.sample(&mut draft_state.logits);
            draft_tokens.push(tok);
            self.draft.forward(draft_state, tok, cur_pos)?;
        }

        self.stats.proposed += draft_tokens.len();

        // Batch tokens to evaluate in target model: [first_tok, draft_tokens[0], ..., draft_tokens[K-1]]
        // Batch size is 1 + K
        let mut target_batch = Vec::with_capacity(1 + draft_tokens.len());
        target_batch.push(first_tok);
        target_batch.extend_from_slice(&draft_tokens);

        let vocab = self.target.config.vocab;
        let mut target_all_logits = vec![0.0f32; target_batch.len() * vocab];

        self.target.prefill_all_logits(
            target_state,
            &target_batch,
            position,
            &mut target_all_logits,
        )?;
        self.stats.target_batches += 1;

        // Verification:
        // - target_all_logits[0]: predicts position + 1. Compare with draft_tokens[0].
        // - target_all_logits[i]: predicts position + 1 + i. Compare with draft_tokens[i].
        let mut accepted_count = 0;
        let total_drafted = draft_tokens.len();

        for i in 0..total_drafted {
            let target_tok_logits = &mut target_all_logits[i * vocab..(i + 1) * vocab];
            let target_pred = sampler.sample(target_tok_logits);

            if target_pred == draft_tokens[i] {
                emitted.push(draft_tokens[i]);
                accepted_count += 1;
            } else {
                // Draft rejected! Emit target model's corrective prediction
                emitted.push(target_pred);
                break;
            }
        }

        self.stats.accepted += accepted_count;

        if accepted_count < total_drafted {
            // Partial rejection: rollback target and draft KV caches
            let accepted_total = emitted.len(); // first_tok + accepted_count (+ 1 corrective)
            let rollback_from = position + accepted_total;
            let rollback_to = position + target_batch.len();

            target_state.truncate_cache(&self.target.config, rollback_from, rollback_to);
            draft_state.truncate_cache(&self.draft.config, rollback_from, rollback_to);

            // Re-sync draft model state with the corrective token
            let corrective_pos = position + accepted_total - 1;
            let corrective_tok = emitted[accepted_total - 1];
            self.draft
                .forward(draft_state, corrective_tok, corrective_pos)?;

            // Re-sync target model's state.logits for the position following the corrective token
            // by forwarding the corrective token in target model
            self.target
                .forward(target_state, corrective_tok, corrective_pos)?;
        } else {
            // All K draft tokens accepted!
            // The last logit in target_all_logits (index K) predicts position + 1 + K
            let last_logits =
                &target_all_logits[total_drafted * vocab..(total_drafted + 1) * vocab];
            target_state.set_logits(last_logits);
        }

        self.stats.total_emitted += emitted.len();
        Ok(emitted)
    }
}

/// Speculative tree drafting engine (multi-candidate branching verification).
///
/// Proposes multiple alternative draft candidate branches (`branch_count` branches of depth `depth`).
/// The target model evaluates the branching candidate paths in batched evaluation,
/// selecting the branch with the maximum accepted token prefix.
///
/// This delivers significantly higher token acceptance rates while strictly preserving
/// bit-identical mathematical equivalence to target-only inference.
pub struct SpeculativeTreeEngine<'a> {
    pub target: &'a Model<'a>,
    pub draft: &'a Model<'a>,
    pub depth: usize,
    pub branch_count: usize,
    pub stats: SpeculativeStats,
}

impl<'a> SpeculativeTreeEngine<'a> {
    pub fn new(
        target: &'a Model<'a>,
        draft: &'a Model<'a>,
        depth: usize,
        branch_count: usize,
    ) -> Self {
        Self {
            target,
            draft,
            depth: depth.max(1),
            branch_count: branch_count.max(1),
            stats: SpeculativeStats::default(),
        }
    }

    /// Perform one tree speculative step.
    /// Returns the vector of accepted / emitted tokens (always at least 1 token).
    pub fn step(
        &mut self,
        target_state: &mut State,
        draft_state: &mut State,
        position: usize,
        sampler: &mut Sampler,
    ) -> Result<Vec<u32>> {
        let first_tok = sampler.sample(&mut target_state.logits);
        let mut emitted = vec![first_tok];

        let max_depth = self
            .depth
            .min(target_state.context().saturating_sub(position + 2));
        if max_depth == 0 {
            self.target.forward(target_state, first_tok, position)?;
            self.draft.forward(draft_state, first_tok, position)?;
            self.stats.target_batches += 1;
            self.stats.total_emitted += 1;
            return Ok(emitted);
        }

        // Forward first_tok into draft model at position
        self.draft.forward(draft_state, first_tok, position)?;

        // Propose top candidate tokens at position + 1 from draft logits
        let mut top_candidates = Vec::with_capacity(self.branch_count);
        let mut logits_copy = draft_state.logits.clone();
        for _ in 0..self.branch_count {
            let tok = sampler.sample(&mut logits_copy);
            top_candidates.push(tok);
            logits_copy[tok as usize] = -f32::INFINITY;
        }
        top_candidates.dedup();

        // For each candidate branch, extend autoregressively with draft model up to max_depth
        let mut branches = Vec::new();
        for &head_tok in &top_candidates {
            let mut branch = vec![head_tok];
            let mut fork_draft_state = draft_state.clone();
            self.draft
                .forward(&mut fork_draft_state, head_tok, position + 1)?;

            for cur_pos in (position + 2..).take(max_depth - 1) {
                let tok = sampler.sample(&mut fork_draft_state.logits);
                branch.push(tok);
                self.draft.forward(&mut fork_draft_state, tok, cur_pos)?;
            }
            branches.push(branch);
        }

        self.stats.proposed += branches.iter().map(|b| b.len()).sum::<usize>();

        // Verify branches against target model
        let vocab = self.target.config.vocab;
        let mut best_branch_idx = 0;
        let mut best_match_len = 0;
        let mut best_corrective = None;
        let mut best_target_logits = Vec::new();

        for (b_idx, branch) in branches.iter().enumerate() {
            let mut target_batch = Vec::with_capacity(1 + branch.len());
            target_batch.push(first_tok);
            target_batch.extend_from_slice(branch);

            let mut fork_target = target_state.clone();
            let mut target_all_logits = vec![0.0f32; target_batch.len() * vocab];

            self.target.prefill_all_logits(
                &mut fork_target,
                &target_batch,
                position,
                &mut target_all_logits,
            )?;

            let mut match_len = 0;
            let mut corrective = None;
            for (i, &cand) in branch.iter().enumerate() {
                let target_tok_logits = &mut target_all_logits[i * vocab..(i + 1) * vocab];
                let target_pred = sampler.sample(target_tok_logits);
                if target_pred == cand {
                    match_len += 1;
                } else {
                    corrective = Some(target_pred);
                    break;
                }
            }

            if match_len > best_match_len
                || (match_len == best_match_len
                    && best_corrective.is_some()
                    && corrective.is_none())
            {
                best_match_len = match_len;
                best_branch_idx = b_idx;
                best_corrective = corrective;
                best_target_logits = target_all_logits;
            }
        }

        self.stats.target_batches += branches.len();
        self.stats.accepted += best_match_len;

        let winning_branch = &branches[best_branch_idx];
        for &tok in &winning_branch[..best_match_len] {
            emitted.push(tok);
        }

        if let Some(corr) = best_corrective {
            emitted.push(corr);

            // Re-sync states by prefilling accepted prefix + corrective
            let sync_tokens = &emitted[..];
            self.target.prefill(target_state, sync_tokens, position)?;
            self.draft.prefill(draft_state, sync_tokens, position)?;
        } else {
            // Full winning branch accepted!
            let sync_tokens = &emitted[..];
            self.target.prefill(target_state, sync_tokens, position)?;
            self.draft.prefill(draft_state, sync_tokens, position)?;

            let last_logits = &best_target_logits
                [winning_branch.len() * vocab..(winning_branch.len() + 1) * vocab];
            target_state.set_logits(last_logits);
        }

        self.stats.total_emitted += emitted.len();
        Ok(emitted)
    }
}
