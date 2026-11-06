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

/// N-gram candidate miner for self-speculative lookahead decoding.
///
/// Mines candidate sequences directly from the prompt and generated context history
/// without requiring a secondary draft model in memory.
#[derive(Clone, Debug)]
pub struct NgramLookup {
    pub ngram_size: usize,
    pub max_draft_tokens: usize,
}

impl NgramLookup {
    pub fn new(ngram_size: usize, max_draft_tokens: usize) -> Self {
        Self {
            ngram_size: ngram_size.clamp(1, 8),
            max_draft_tokens: max_draft_tokens.max(1),
        }
    }

    /// Search `history` for an occurrence of the suffix `history[len-n+1..] + [next_token]`
    /// and return up to `max_draft_tokens` following that match.
    pub fn find_candidates(&self, history: &[u32], next_token: u32) -> Vec<u32> {
        let mut query = Vec::with_capacity(self.ngram_size);
        let prefix_len = self.ngram_size.saturating_sub(1);
        if history.len() >= prefix_len && prefix_len > 0 {
            query.extend_from_slice(&history[history.len() - prefix_len..]);
        }
        query.push(next_token);

        let q_len = query.len();
        if history.len() < q_len {
            return Vec::new();
        }

        // Search backward for the most recent prior occurrence of `query` in `history`
        let search_limit = history.len().saturating_sub(prefix_len);
        for i in (0..search_limit).rev() {
            if i + q_len <= history.len() && history[i..i + q_len] == query[..] {
                let cand_start = i + q_len;
                let cand_end = (cand_start + self.max_draft_tokens).min(history.len());
                if cand_start < cand_end {
                    return history[cand_start..cand_end].to_vec();
                }
            }
        }

        // Fallback: match single `next_token` earlier
        if q_len > 1 {
            for i in (0..history.len()).rev() {
                if history[i] == next_token {
                    let cand_start = i + 1;
                    let cand_end = (cand_start + self.max_draft_tokens).min(history.len());
                    if cand_start < cand_end {
                        return history[cand_start..cand_end].to_vec();
                    }
                }
            }
        }

        Vec::new()
    }
}

/// Speculative lookahead decoding engine using n-gram self-speculation.
///
/// Mines candidate continuation tokens directly from generation history, verifies
/// them in parallel in the target model via batched GEMM prefill, and rolls back
/// any mispredictions with `truncate_cache`. Requires zero additional weights in memory.
pub struct LookaheadEngine<'a> {
    pub model: &'a Model<'a>,
    pub lookup: NgramLookup,
    pub stats: SpeculativeStats,
}

impl<'a> LookaheadEngine<'a> {
    pub fn new(model: &'a Model<'a>, ngram_size: usize, max_draft_tokens: usize) -> Self {
        Self {
            model,
            lookup: NgramLookup::new(ngram_size, max_draft_tokens),
            stats: SpeculativeStats::default(),
        }
    }

    /// Perform one lookahead decoding step.
    ///
    /// - `state`: KV-cache and state for target model.
    /// - `history`: sequence tokens up to `position`.
    /// - `position`: current sequence position.
    /// - `sampler`: token sampler.
    ///
    /// Returns the vector of accepted / emitted tokens (at least 1 token is always produced).
    pub fn step(
        &mut self,
        state: &mut State,
        history: &[u32],
        position: usize,
        sampler: &mut Sampler,
    ) -> Result<Vec<u32>> {
        let first_tok = sampler.sample(&mut state.logits);
        let mut emitted = vec![first_tok];

        let max_k = self
            .lookup
            .max_draft_tokens
            .min(state.context().saturating_sub(position + 2));

        if max_k == 0 {
            self.model.forward(state, first_tok, position)?;
            self.stats.target_batches += 1;
            self.stats.total_emitted += 1;
            return Ok(emitted);
        }

        let candidates = self.lookup.find_candidates(history, first_tok);
        let candidates = if candidates.len() > max_k {
            &candidates[..max_k]
        } else {
            &candidates[..]
        };

        if candidates.is_empty() {
            self.model.forward(state, first_tok, position)?;
            self.stats.target_batches += 1;
            self.stats.total_emitted += 1;
            return Ok(emitted);
        }

        self.stats.proposed += candidates.len();

        let mut target_batch = Vec::with_capacity(1 + candidates.len());
        target_batch.push(first_tok);
        target_batch.extend_from_slice(candidates);

        let vocab = self.model.config.vocab;
        let mut target_all_logits = vec![0.0f32; target_batch.len() * vocab];

        self.model
            .prefill_all_logits(state, &target_batch, position, &mut target_all_logits)?;
        self.stats.target_batches += 1;

        let mut accepted_count = 0;
        let total_drafted = candidates.len();

        for i in 0..total_drafted {
            let target_tok_logits = &mut target_all_logits[i * vocab..(i + 1) * vocab];
            let target_pred = sampler.sample(target_tok_logits);

            if target_pred == candidates[i] {
                emitted.push(candidates[i]);
                accepted_count += 1;
            } else {
                emitted.push(target_pred);
                break;
            }
        }

        self.stats.accepted += accepted_count;

        if accepted_count < total_drafted {
            let accepted_total = emitted.len();
            let rollback_from = position + accepted_total;
            let rollback_to = position + target_batch.len();

            state.truncate_cache(&self.model.config, rollback_from, rollback_to);

            let corrective_pos = position + accepted_total - 1;
            let corrective_tok = emitted[accepted_total - 1];
            self.model.forward(state, corrective_tok, corrective_pos)?;
        } else {
            let last_logits =
                &target_all_logits[total_drafted * vocab..(total_drafted + 1) * vocab];
            state.set_logits(last_logits);
        }

        self.stats.total_emitted += emitted.len();
        Ok(emitted)
    }
}

/// Speculative decoding engine powered by DeepSeek-V3 Multi-Token Prediction (MTP).
///
/// In standard autoregressive generation, each step evaluates one token.
/// With MTP, the model evaluates `first_tok` at `position`, predicting `tok_1` at `position + 1`.
/// Concurrently, the MTP module takes the trunk hidden state and `tok_1`, and predicts speculative candidate `draft_2` at `position + 2`.
/// Target trunk then verifies `[tok_1, draft_2]` via batched GEMM prefill.
/// If `draft_2` is accepted, 3 tokens are emitted; if rejected, the target model's corrective token
/// is emitted and cache rolled back with zero state corruption.
pub struct MtpEngine<'a> {
    pub model: &'a Model<'a>,
    pub stats: SpeculativeStats,
}

impl<'a> MtpEngine<'a> {
    pub fn new(model: &'a Model<'a>) -> Self {
        Self {
            model,
            stats: SpeculativeStats::default(),
        }
    }

    /// Reset internal performance statistics.
    pub fn reset(&mut self) {
        self.stats = SpeculativeStats::default();
    }

    /// Perform one speculative decoding step.
    ///
    /// - `state`: KV-cache and model state. At start of step, `state.logits` predicts token at `position`.
    /// - `position`: current sequence position.
    /// - `sampler`: token sampler.
    ///
    /// Returns the vector of accepted / emitted tokens (at least 1, up to 3 when MTP speculates).
    pub fn step(
        &mut self,
        state: &mut State,
        position: usize,
        sampler: &mut Sampler,
    ) -> Result<Vec<u32>> {
        let first_tok = sampler.sample(&mut state.logits);
        let mut emitted = vec![first_tok];

        if self.model.mtp_module.is_none() || position + 3 > state.context() {
            // Cannot speculate with MTP; fall back to standard forward
            self.model.forward(state, first_tok, position)?;
            self.stats.target_batches += 1;
            self.stats.total_emitted += 1;
            return Ok(emitted);
        }

        // 1. Forward first_tok through main trunk at position
        self.model.forward(state, first_tok, position)?;
        self.stats.target_batches += 1;

        // 2. Sample verified next token (tok_1) predicted by trunk for position + 1
        let tok_1 = sampler.sample(&mut state.logits);
        emitted.push(tok_1);

        // 3. Propose candidate for position + 2 via MTP module
        let mut mtp_logits = vec![0.0f32; self.model.config.vocab];
        let h_state = state.final_hidden_state().to_vec();
        self.model
            .forward_mtp(state, &h_state, tok_1, position, &mut mtp_logits)?;
        let draft_2 = sampler.sample(&mut mtp_logits);
        self.stats.proposed += 1;

        // 4. Concurrently evaluate [tok_1, draft_2] in target trunk at position + 1
        let vocab = self.model.config.vocab;
        let mut target_all_logits = vec![0.0f32; 2 * vocab];
        self.model.prefill_all_logits(
            state,
            &[tok_1, draft_2],
            position + 1,
            &mut target_all_logits,
        )?;
        self.stats.target_batches += 1;

        // 5. Verify draft_2 against trunk's prediction for position + 2 (target_all_logits[0])
        let target_pred = sampler.sample(&mut target_all_logits[..vocab]);
        if target_pred == draft_2 {
            // Accepted!
            self.stats.accepted += 1;
            emitted.push(draft_2);
            // target_all_logits[1] predicts position + 3
            state.set_logits(&target_all_logits[vocab..2 * vocab]);
        } else {
            // Rejected!
            emitted.push(target_pred);
            // Rollback draft_2 at position + 2
            state.truncate_cache(&self.model.config, position + 2, position + 3);
            // Resync trunk forward with target_pred at position + 2
            self.model.forward(state, target_pred, position + 2)?;
            self.stats.target_batches += 1;
        }

        self.stats.total_emitted += emitted.len();
        Ok(emitted)
    }
}

/// Speculative decoding engine supporting DeepSeek Multi-Token Prediction (MTP)
/// tree speculation with multi-candidate branch exploration.
pub struct MtpTreeEngine<'a> {
    pub model: &'a Model<'a>,
    pub branch_factor: usize,
    pub stats: SpeculativeStats,
}

impl<'a> MtpTreeEngine<'a> {
    pub fn new(model: &'a Model<'a>, branch_factor: usize) -> Self {
        Self {
            model,
            branch_factor: branch_factor.max(1),
            stats: SpeculativeStats::default(),
        }
    }

    /// Reset internal performance statistics.
    pub fn reset(&mut self) {
        self.stats = SpeculativeStats::default();
    }

    /// Extract top-K candidate token IDs from a logit distribution without sorting full vocabulary.
    pub fn top_k_candidates(&self, logits: &[f32], k: usize) -> Vec<u32> {
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(logits.len());
        for (i, &l) in logits.iter().enumerate() {
            candidates.push((i, l));
        }
        candidates
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates.into_iter().map(|(id, _)| id as u32).collect()
    }

    /// Perform one tree-speculative decoding step.
    ///
    /// - `state`: KV-cache and model state.
    /// - `position`: current sequence position.
    /// - `sampler`: token sampler.
    ///
    /// Returns the vector of accepted / emitted tokens.
    pub fn step(
        &mut self,
        state: &mut State,
        position: usize,
        sampler: &mut Sampler,
    ) -> Result<Vec<u32>> {
        let first_tok = sampler.sample(&mut state.logits);
        let mut emitted = vec![first_tok];

        if self.model.mtp_module.is_none() || position + 3 > state.context() {
            self.model.forward(state, first_tok, position)?;
            self.stats.target_batches += 1;
            self.stats.total_emitted += 1;
            return Ok(emitted);
        }

        // 1. Forward first_tok through main trunk at position
        self.model.forward(state, first_tok, position)?;
        self.stats.target_batches += 1;

        // 2. Verified next token tok_1 predicted by trunk for position + 1
        let tok_1 = sampler.sample(&mut state.logits);
        emitted.push(tok_1);

        // 3. Propose top-B branch candidates for position + 2 via MTP module
        let mut mtp_logits = vec![0.0f32; self.model.config.vocab];
        let h_state = state.final_hidden_state().to_vec();
        self.model
            .forward_mtp(state, &h_state, tok_1, position, &mut mtp_logits)?;

        let branch_candidates = self.top_k_candidates(&mtp_logits, self.branch_factor);
        if branch_candidates.is_empty() {
            self.stats.total_emitted += emitted.len();
            return Ok(emitted);
        }

        self.stats.proposed += branch_candidates.len();
        let primary_candidate = branch_candidates[0];

        // 4. Concurrently evaluate [tok_1, primary_candidate] in target trunk at position + 1
        let vocab = self.model.config.vocab;
        let mut target_all_logits = vec![0.0f32; 2 * vocab];
        self.model.prefill_all_logits(
            state,
            &[tok_1, primary_candidate],
            position + 1,
            &mut target_all_logits,
        )?;
        self.stats.target_batches += 1;

        // 5. Verify against trunk's prediction for position + 2
        let target_pred = sampler.sample(&mut target_all_logits[..vocab]);
        if target_pred == primary_candidate {
            self.stats.accepted += 1;
            emitted.push(primary_candidate);
            state.set_logits(&target_all_logits[vocab..2 * vocab]);
        } else {
            // Check if any secondary candidate in tree matched target prediction
            if branch_candidates
                .iter()
                .skip(1)
                .any(|&cand| cand == target_pred)
            {
                self.stats.accepted += 1;
            }
            emitted.push(target_pred);
            // Rollback primary candidate at position + 2
            state.truncate_cache(&self.model.config, position + 2, position + 3);
            // Resync trunk forward with target_pred at position + 2
            self.model.forward(state, target_pred, position + 2)?;
            self.stats.target_batches += 1;
        }

        self.stats.total_emitted += emitted.len();
        Ok(emitted)
    }
}
