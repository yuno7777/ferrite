//! Model watermarking and cryptographic token signature injection.
//!
//! Implements green-red list watermarking (Kirchenbauer et al.) and cryptographic
//! PRF token signature injection. Modifies logits during generation to bias token
//! selection toward a pseudo-random green partition determined by secret key and
//! preceding context, enabling statistical provenance verification and detection.

/// Configuration for model watermarking and cryptographic token signature injection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WatermarkConfig {
    /// Secret cryptographic key used to seed the pseudo-random partition.
    pub key: u64,
    /// Fraction of vocabulary tokens assigned to the green list (typically 0.25 to 0.5).
    pub gamma: f32,
    /// Additive logit bias injected onto green list tokens (typically 1.0 to 3.0).
    pub delta: f32,
    /// Number of preceding context tokens hashed to determine current token partition.
    pub context_width: usize,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        Self {
            key: 0xCAFE_BABE_DEAD_BEEF,
            gamma: 0.5,
            delta: 2.0,
            context_width: 1,
        }
    }
}

impl WatermarkConfig {
    pub fn new(key: u64, gamma: f32, delta: f32, context_width: usize) -> Self {
        Self {
            key,
            gamma: gamma.clamp(0.01, 0.99),
            delta: delta.max(0.0),
            context_width: context_width.max(1),
        }
    }
}

/// Keyed pseudo-random hash function for context tokens.
pub fn hash_context(key: u64, context: &[u32], context_width: usize) -> u64 {
    let window = if context.is_empty() || context_width == 0 {
        &[][..]
    } else if context.len() <= context_width {
        context
    } else {
        &context[context.len() - context_width..]
    };

    let mut h = key ^ 0x517c_c1b7_2722_0a95;
    for &tok in window {
        h = h.rotate_left(17) ^ (tok as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 31;
    }
    h
}

/// Evaluates whether a given token belongs to the green partition for a context seed.
pub fn is_token_green(seed: u64, tok: u32, gamma: f32) -> bool {
    let mut h = seed.rotate_left(13) ^ (tok as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;

    let val = (h as f64) / (u64::MAX as f64);
    val < (gamma as f64)
}

/// Watermark injection engine that alters logits during autoregressive sampling.
pub struct WatermarkEngine;

impl WatermarkEngine {
    /// Injects green-list logit bias into `logits` given the preceding `context`.
    pub fn apply_watermark(logits: &mut [f32], context: &[u32], config: &WatermarkConfig) {
        if config.delta <= 0.0 || logits.is_empty() {
            return;
        }

        let seed = hash_context(config.key, context, config.context_width);
        for (tok, logit) in logits.iter_mut().enumerate() {
            if is_token_green(seed, tok as u32, config.gamma) {
                *logit += config.delta;
            }
        }
    }
}

/// Statistical verification report produced by the watermark detector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WatermarkScore {
    /// Number of tokens evaluated for watermarking.
    pub total_tokens: usize,
    /// Number of tokens that fell in their respective green partition.
    pub green_tokens: usize,
    /// Empirical green token fraction.
    pub green_fraction: f32,
    /// Standard normal z-score under null hypothesis of unwatermarked text.
    pub z_score: f32,
    /// Approximate one-sided p-value against null hypothesis.
    pub p_value: f32,
    /// Boolean decision whether text exhibits statistically significant watermarking.
    pub is_watermarked: bool,
}

/// Complementary error function approximation for p-value calculation.
#[allow(clippy::excessive_precision)]
fn erfc_approx(x: f32) -> f32 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let ans = t * poly.exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Detects and evaluates watermark signatures in generated token sequences.
pub struct WatermarkDetector;

impl WatermarkDetector {
    /// Detect whether `tokens` contain watermark signatures generated with `config`.
    pub fn detect(tokens: &[u32], config: &WatermarkConfig, z_threshold: f32) -> WatermarkScore {
        if tokens.len() <= config.context_width {
            return WatermarkScore {
                total_tokens: 0,
                green_tokens: 0,
                green_fraction: 0.0,
                z_score: 0.0,
                p_value: 1.0,
                is_watermarked: false,
            };
        }

        let mut total_evaluated = 0;
        let mut green_count = 0;

        for i in config.context_width..tokens.len() {
            let context = &tokens[..i];
            let seed = hash_context(config.key, context, config.context_width);
            let target_tok = tokens[i];

            if is_token_green(seed, target_tok, config.gamma) {
                green_count += 1;
            }
            total_evaluated += 1;
        }

        let n = total_evaluated as f32;
        let gamma = config.gamma;
        let expected_green = n * gamma;
        let std_dev = (n * gamma * (1.0 - gamma)).sqrt();

        let z_score = if std_dev > 0.0 {
            (green_count as f32 - expected_green) / std_dev
        } else {
            0.0
        };

        // One-sided p-value: P(Z >= z) = 0.5 * erfc(z / sqrt(2))
        let p_value = 0.5 * erfc_approx(z_score / std::f32::consts::SQRT_2);
        let green_fraction = if total_evaluated > 0 {
            green_count as f32 / total_evaluated as f32
        } else {
            0.0
        };

        let is_watermarked = z_score >= z_threshold;

        WatermarkScore {
            total_tokens: total_evaluated,
            green_tokens: green_count,
            green_fraction,
            z_score,
            p_value,
            is_watermarked,
        }
    }
}
