//! Turning logits into a token.
//!
//! Temperature, top-k and top-p compose in that order, which is the order
//! everyone else uses — swapping top-k and top-p changes the output, so it is
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
    pub rng: Rng,
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
            rng: Rng::new(1),
        }
    }

    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            rng: Rng::new(seed),
        }
    }

    /// Pick a token. `logits` is modified in place — it is scratch after this.
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

        // Index by probability, descending. Only the head matters, so when
        // top-k is small the rest is left unsorted.
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

        // Nucleus: the shortest prefix whose mass reaches top_p.
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
}
