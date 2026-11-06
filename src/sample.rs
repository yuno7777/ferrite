//! Turning logits into a token.
//!
//! Temperature, top-k and top-p compose in that order, which is the order
//! everyone else uses: swapping top-k and top-p changes the output, so it is
//! worth being deliberate about.

use crate::ops;

/// xorshift64*, seeded. Deterministic runs matter more here than statistical
/// perfection: "same seed, same output" is what makes a sampling bug findable.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Zero is a fixed point of xorshift, so it can never be the state.
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`, using the 24 bits an f32 can actually hold.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[derive(Clone, Debug)]
pub struct Sampler {
    /// 0 means greedy. Higher flattens the distribution.
    pub temperature: f32,
    /// Keep only the k most likely tokens. 0 disables.
    pub top_k: usize,
    /// Keep the smallest set whose probabilities reach p. 1.0 disables.
    pub top_p: f32,
    /// Minimum probability scale relative to top token (p >= p_max * min_p). 0.0 disables.
    pub min_p: f32,
    /// Tail free sampling parameter z in (0, 1]. 1.0 disables.
    pub tfs_z: f32,
    pub penalties: PenaltyConfig,
    pub watermark: Option<crate::watermark::WatermarkConfig>,
    pub rng: Rng,
}

/// Configuration for repetition, frequency, and presence penalties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PenaltyConfig {
    /// Multiplicative penalty on previous tokens: > 1.0 penalizes, 1.0 disables.
    pub repetition_penalty: f32,
    /// Additive penalty proportional to count of occurrences: > 0.0 penalizes, 0.0 disables.
    pub frequency_penalty: f32,
    /// Additive penalty on any token that appeared at least once: > 0.0 penalizes, 0.0 disables.
    pub presence_penalty: f32,
    /// Window of trailing tokens to penalize (0 means entire sequence).
    pub last_n: usize,
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self {
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            last_n: 64,
        }
    }
}

/// Apply repetition, frequency, and presence penalties to raw unnormalized logits.
pub fn apply_penalties(logits: &mut [f32], tokens: &[u32], config: &PenaltyConfig) {
    if tokens.is_empty() {
        return;
    }
    if (config.repetition_penalty - 1.0).abs() < 1e-6
        && config.frequency_penalty.abs() < 1e-6
        && config.presence_penalty.abs() < 1e-6
    {
        return;
    }

    let window = if config.last_n == 0 || config.last_n >= tokens.len() {
        tokens
    } else {
        &tokens[tokens.len() - config.last_n..]
    };

    let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &tok in window {
        *counts.entry(tok).or_insert(0) += 1;
    }

    for (tok, count) in counts {
        if let Some(logit) = logits.get_mut(tok as usize) {
            // Multiplicative repetition penalty
            if (config.repetition_penalty - 1.0).abs() > 1e-6 && config.repetition_penalty > 0.0 {
                if *logit > 0.0 {
                    *logit /= config.repetition_penalty;
                } else {
                    *logit *= config.repetition_penalty;
                }
            }
            // Additive frequency penalty
            if config.frequency_penalty.abs() > 1e-6 {
                *logit -= (count as f32) * config.frequency_penalty;
            }
            // Additive presence penalty
            if config.presence_penalty.abs() > 1e-6 {
                *logit -= config.presence_penalty;
            }
        }
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::greedy()
    }
}

impl Sampler {
    /// Always take the most likely token. Reproducible, and the right default
    /// when you are still checking whether the model works at all.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            tfs_z: 1.0,
            penalties: PenaltyConfig::default(),
            watermark: None,
            rng: Rng::new(1),
        }
    }

    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            tfs_z: 1.0,
            penalties: PenaltyConfig::default(),
            watermark: None,
            rng: Rng::new(seed),
        }
    }

    pub fn new_advanced(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        tfs_z: f32,
        seed: u64,
    ) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p,
            tfs_z,
            penalties: PenaltyConfig::default(),
            watermark: None,
            rng: Rng::new(seed),
        }
    }

    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    pub fn with_tfs(mut self, tfs_z: f32) -> Self {
        self.tfs_z = tfs_z;
        self
    }

    pub fn with_penalties(mut self, penalties: PenaltyConfig) -> Self {
        self.penalties = penalties;
        self
    }

    pub fn with_repetition_penalty(mut self, rep: f32) -> Self {
        self.penalties.repetition_penalty = rep;
        self
    }

    pub fn with_frequency_penalty(mut self, freq: f32) -> Self {
        self.penalties.frequency_penalty = freq;
        self
    }

    pub fn with_presence_penalty(mut self, pres: f32) -> Self {
        self.penalties.presence_penalty = pres;
        self
    }

    pub fn with_watermark(mut self, config: crate::watermark::WatermarkConfig) -> Self {
        self.watermark = Some(config);
        self
    }

    /// Sample token after applying repetition/frequency/presence penalties and watermark from history.
    pub fn sample_with_history(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        apply_penalties(logits, history, &self.penalties);
        if let Some(ref wm) = self.watermark {
            crate::watermark::WatermarkEngine::apply_watermark(logits, history, wm);
        }
        self.sample(logits)
    }

    /// Sample token applying watermark bias based on preceding context.
    pub fn sample_with_watermark(&mut self, logits: &mut [f32], context: &[u32]) -> u32 {
        if let Some(ref wm) = self.watermark {
            crate::watermark::WatermarkEngine::apply_watermark(logits, context, wm);
        }
        self.sample(logits)
    }

    /// Pick a token. `logits` is modified in place; it is scratch after this.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        if self.temperature <= 0.0 {
            return argmax(logits);
        }

        let inverse = 1.0 / self.temperature;
        for logit in logits.iter_mut() {
            *logit *= inverse;
        }
        ops::softmax(logits);

        // Index by probability, descending.
        let mut ranked: Vec<(f32, u32)> = logits
            .iter()
            .enumerate()
            .map(|(index, p)| (*p, index as u32))
            .collect();

        let keep = match self.top_k {
            0 => ranked.len(),
            k => k.min(ranked.len()),
        };
        if keep < ranked.len() {
            ranked.select_nth_unstable_by(keep - 1, |a, b| b.0.total_cmp(&a.0));
            ranked.truncate(keep);
        }
        ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        // 1. Min-P truncation: remove tokens where p < p_max * min_p
        if self.min_p > 0.0 && !ranked.is_empty() {
            let threshold = ranked[0].0 * self.min_p;
            let mut keep_count = 0;
            for (p, _) in &ranked {
                if *p >= threshold {
                    keep_count += 1;
                } else {
                    break;
                }
            }
            ranked.truncate(keep_count.max(1));
        }

        // 2. Tail Free Sampling (TFS): truncate based on normalized second derivative
        if self.tfs_z < 1.0 && ranked.len() > 2 {
            let mut d = Vec::with_capacity(ranked.len() - 1);
            for i in 0..ranked.len() - 1 {
                d.push(ranked[i].0 - ranked[i + 1].0);
            }
            let mut d2 = Vec::with_capacity(d.len() - 1);
            for i in 0..d.len() - 1 {
                d2.push((d[i] - d[i + 1]).abs());
            }
            let d2_sum: f32 = d2.iter().sum();
            if d2_sum > 0.0 {
                let mut cum = 0.0;
                let mut cutoff = ranked.len();
                for (i, v) in d2.iter().enumerate() {
                    cum += v / d2_sum;
                    if cum >= self.tfs_z {
                        cutoff = i + 1;
                        break;
                    }
                }
                ranked.truncate(cutoff.max(1));
            }
        }

        // 3. Nucleus (Top-P): the shortest prefix whose mass reaches top_p.
        if self.top_p < 1.0 {
            let mut total = 0.0;
            let mut cutoff = ranked.len();
            for (position, (probability, _)) in ranked.iter().enumerate() {
                total += probability;
                if total >= self.top_p {
                    cutoff = position + 1;
                    break;
                }
            }
            ranked.truncate(cutoff.max(1));
        }

        let mass: f32 = ranked.iter().map(|(p, _)| p).sum();
        let mut target = self.rng.next_f32() * mass;
        for (probability, token) in &ranked {
            target -= probability;
            if target <= 0.0 {
                return *token;
            }
        }
        // Only reachable through floating-point drift at the very end.
        ranked.last().map(|(_, token)| *token).unwrap_or(0)
    }
}

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_takes_the_maximum() {
        let mut sampler = Sampler::greedy();
        assert_eq!(sampler.sample(&mut [0.1, 0.9, 0.3]), 1);
        assert_eq!(sampler.sample(&mut [5.0, -1.0, 2.0]), 0);
        // Ties resolve to the first, deterministically.
        assert_eq!(sampler.sample(&mut [1.0, 1.0]), 0);
    }

    #[test]
    fn top_k_of_one_is_greedy_at_any_temperature() {
        let mut sampler = Sampler::new(5.0, 1, 1.0, 42);
        for _ in 0..20 {
            assert_eq!(sampler.sample(&mut [0.0, 3.0, 1.0]), 1);
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_sequence() {
        let draw = |seed| {
            let mut sampler = Sampler::new(1.0, 0, 1.0, seed);
            (0..10)
                .map(|_| sampler.sample(&mut [1.0, 2.0, 3.0, 0.5]))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8), "different seeds should diverge");
    }

    #[test]
    fn sampling_follows_the_distribution() {
        // Logits chosen so token 1 is roughly e^2 times as likely as token 0.
        let mut sampler = Sampler::new(1.0, 0, 1.0, 12345);
        let mut counts = [0usize; 2];
        for _ in 0..4000 {
            counts[sampler.sample(&mut [0.0, 2.0]) as usize] += 1;
        }
        let observed = counts[1] as f32 / 4000.0;
        let expected = 2.0f32.exp() / (1.0 + 2.0f32.exp()); // ~0.881
        assert!(
            (observed - expected).abs() < 0.03,
            "observed {observed}, expected around {expected}"
        );
    }

    #[test]
    fn top_p_drops_the_tail() {
        // Token 0 alone carries more than 90% of the mass.
        let mut sampler = Sampler::new(1.0, 0, 0.9, 99);
        for _ in 0..50 {
            assert_eq!(sampler.sample(&mut [10.0, 0.0, 0.0, 0.0]), 0);
        }
    }

    #[test]
    fn rng_is_uniform_enough_and_never_leaves_the_unit_interval() {
        let mut rng = Rng::new(0); // seed 0 must still work
        let mut sum = 0.0;
        for _ in 0..10_000 {
            let value = rng.next_f32();
            assert!((0.0..1.0).contains(&value), "{value} out of range");
            sum += value;
        }
        let mean = sum / 10_000.0;
        assert!((mean - 0.5f32).abs() < 0.02, "mean {mean}");
    }

    #[test]
    fn min_p_drops_sub_threshold_tail() {
        // Logits: token 0 is dominant (~0.8), token 1 is ~0.15, token 2 is ~0.05
        // With min_p = 0.2, threshold is 0.8 * 0.2 = 0.16. Token 2 must be dropped.
        let mut sampler = Sampler::new(1.0, 0, 1.0, 42).with_min_p(0.25);
        for _ in 0..50 {
            let res = sampler.sample(&mut [5.0, 3.5, 0.0]);
            assert!(
                res == 0 || res == 1,
                "token 2 should be filtered out by min_p"
            );
        }
    }

    #[test]
    fn tfs_truncates_long_tail() {
        let mut sampler = Sampler::new(1.0, 0, 1.0, 42).with_tfs(0.9);
        // Logits with a sharp drop after token 1
        for _ in 0..50 {
            let res = sampler.sample(&mut [10.0, 8.0, 1.0, 0.5, 0.1, 0.05]);
            assert!(res == 0 || res == 1, "tail tokens should be removed by TFS");
        }
    }

    #[test]
    fn repetition_penalty_reduces_repeated_token_probability() {
        let mut logits = [2.0, 1.0];
        let history = [0]; // token 0 previously generated
        let config = PenaltyConfig {
            repetition_penalty: 2.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            last_n: 10,
        };
        apply_penalties(&mut logits, &history, &config);
        // Logit 0 was 2.0, with rep penalty 2.0 it becomes 1.0
        assert!((logits[0] - 1.0).abs() < 1e-5);
        assert!((logits[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn frequency_and_presence_penalties() {
        let mut logits = [10.0, 10.0];
        let history = [0, 0, 0]; // token 0 generated 3 times
        let config = PenaltyConfig {
            repetition_penalty: 1.0,
            frequency_penalty: 1.5,
            presence_penalty: 2.0,
            last_n: 10,
        };
        apply_penalties(&mut logits, &history, &config);
        // Token 0: 10.0 - (3 * 1.5) - 2.0 = 10.0 - 4.5 - 2.0 = 3.5
        assert!((logits[0] - 3.5).abs() < 1e-5);
        // Token 1 untouched: 10.0
        assert!((logits[1] - 10.0).abs() < 1e-5);
    }
}
