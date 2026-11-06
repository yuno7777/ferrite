//! Evaluation module: cross-entropy loss and perplexity calculation.

use crate::model::{Model, State};
use std::io::Result;

/// Compute the cross-entropy loss $-\log P(\text{target})$ for a single token
/// given the unnormalized logits distribution over the vocabulary.
///
/// Uses a numerically stable log-sum-exp formulation in double precision:
/// $\text{loss} = \log \sum_j \exp(z_j - z_{\max}) + z_{\max} - z_{\text{target}}$
pub fn cross_entropy_loss(logits: &[f32], target: u32) -> Result<f64> {
    let target_idx = target as usize;
    if target_idx >= logits.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "target token id {target} exceeds vocabulary size {}",
                logits.len()
            ),
        ));
    }

    let mut max_logit = f32::NEG_INFINITY;
    for &val in logits {
        if val > max_logit {
            max_logit = val;
        }
    }

    let mut sum_exp = 0.0f64;
    for &val in logits {
        sum_exp += ((val - max_logit) as f64).exp();
    }

    let log_sum_exp = sum_exp.ln() + max_logit as f64;
    let target_logit = logits[target_idx] as f64;
    let loss = log_sum_exp - target_logit;
    Ok(loss)
}

/// Statistics for a single evaluated chunk/window of tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkPerplexity {
    pub chunk_index: usize,
    pub start_token: usize,
    pub count: usize,
    pub chunk_nll: f64,
    pub chunk_ppl: f64,
}

/// Aggregate perplexity evaluation report over a token sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct PerplexityReport {
    pub total_tokens: usize,
    pub evaluated_tokens: usize,
    pub chunks: Vec<ChunkPerplexity>,
    pub mean_loss: f64,
    pub perplexity: f64,
}

/// Sliding-window or chunked perplexity evaluator.
pub struct PerplexityEvaluator<'a> {
    pub model: &'a Model<'a>,
    pub context_size: usize,
    pub stride: usize,
    pub threads: usize,
}

impl<'a> PerplexityEvaluator<'a> {
    /// Create a new perplexity evaluator.
    ///
    /// * `context_size`: maximum token window evaluated in a single forward pass.
    /// * `stride`: step size between successive windows (typically equal to `context_size`
    ///   for disjoint chunks, or smaller for overlapping sliding windows).
    pub fn new(model: &'a Model<'a>, context_size: usize, stride: usize) -> Self {
        let ctx = context_size.max(2);
        // One context window can score at most `ctx - 1` next-token targets.
        let strd = stride.max(1).min(ctx - 1);
        Self {
            model,
            context_size: ctx,
            stride: strd,
            threads: 0,
        }
    }

    /// Set number of worker threads for matmul kernels.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Evaluate perplexity over the token slice.
    pub fn evaluate(&self, tokens: &[u32]) -> Result<PerplexityReport> {
        self.evaluate_with_callback(tokens, |_| {})
    }

    /// Evaluate perplexity over the token slice, invoking `on_chunk` after each evaluated window.
    pub fn evaluate_with_callback<F: FnMut(&ChunkPerplexity)>(
        &self,
        tokens: &[u32],
        mut on_chunk: F,
    ) -> Result<PerplexityReport> {
        let total_tokens = tokens.len();
        if total_tokens < 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("text must contain at least 2 tokens to evaluate perplexity, found {total_tokens}"),
            ));
        }

        let vocab = self.model.config.vocab;
        let mut chunks = Vec::new();
        let mut total_nll = 0.0f64;
        let mut evaluated_tokens = 0usize;
        let mut target_start = 1usize;
        let mut chunk_idx = 0;

        while target_start < total_tokens {
            let target_end = (target_start + self.stride).min(total_tokens);
            let start = target_start.saturating_sub(self.context_size - 1);
            let window_len = target_end - start;

            let mut state = State::with_context(&self.model.config, window_len);
            if self.threads > 0 {
                state.threads = self.threads;
            }

            let mut all_logits = vec![0.0f32; window_len * vocab];
            self.model.prefill_all_logits(
                &mut state,
                &tokens[start..target_end],
                0,
                &mut all_logits,
            )?;

            let mut chunk_nll = 0.0f64;
            let mut chunk_eval_count = 0usize;

            for (global_target, &target_tok) in tokens
                .iter()
                .enumerate()
                .take(target_end)
                .skip(target_start)
            {
                let logit_index = global_target - 1 - start;
                let logits = &all_logits[logit_index * vocab..(logit_index + 1) * vocab];
                let loss = cross_entropy_loss(logits, target_tok)?;

                chunk_nll += loss;
                chunk_eval_count += 1;
            }

            if chunk_eval_count > 0 {
                let chunk_mean = chunk_nll / chunk_eval_count as f64;
                let chunk_ppl = chunk_mean.exp();
                let chunk_stat = ChunkPerplexity {
                    chunk_index: chunk_idx,
                    start_token: target_start,
                    count: chunk_eval_count,
                    chunk_nll,
                    chunk_ppl,
                };
                on_chunk(&chunk_stat);
                chunks.push(chunk_stat);

                total_nll += chunk_nll;
                evaluated_tokens += chunk_eval_count;
            }

            chunk_idx += 1;
            target_start = target_end;
        }

        if evaluated_tokens == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "no tokens were evaluated",
            ));
        }

        let mean_loss = total_nll / evaluated_tokens as f64;
        let perplexity = mean_loss.exp();

        Ok(PerplexityReport {
            total_tokens,
            evaluated_tokens,
            chunks,
            mean_loss,
            perplexity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_distribution_cross_entropy_equals_log_vocab() {
        let vocab = 100;
        let logits = vec![1.5f32; vocab]; // all equal
        let target = 42;
        let loss = cross_entropy_loss(&logits, target).expect("loss");
        let expected = (vocab as f64).ln();
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn deterministic_distribution_loss_approaches_zero() {
        let mut logits = vec![-100.0f32; 10];
        logits[3] = 100.0f32; // virtually 100% probability on index 3
        let loss = cross_entropy_loss(&logits, 3).expect("loss");
        assert!(loss < 1e-5, "loss was {loss}");
    }

    #[test]
    fn out_of_bounds_target_returns_error() {
        let logits = vec![0.0f32; 10];
        let err = cross_entropy_loss(&logits, 10).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
