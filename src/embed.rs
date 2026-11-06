//! Embedding model engine and vector operations.
//!
//! Provides pooling (Mean, CLS, Last), unit L2 normalization, cosine similarity,
//! and batch embedding extraction.

use std::io::{Error, ErrorKind, Result};
use std::str::FromStr;

use crate::model::{Model, State};

/// Pooling strategy to collapse token representations into a single sentence vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pooling {
    #[default]
    Mean,
    Cls,
    Last,
}

impl FromStr for Pooling {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "cls" => Ok(Self::Cls),
            "last" => Ok(Self::Last),
            other => Err(Error::new(
                ErrorKind::InvalidInput,
                format!("unknown pooling type {other:?}, expected 'mean', 'cls', or 'last'"),
            )),
        }
    }
}

/// Normalizes a vector in-place to unit Euclidean (L2) norm.
pub fn l2_normalize(vec: &mut [f32]) {
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for v in vec.iter_mut() {
            *v *= inv;
        }
    }
}

/// Computes cosine similarity between two vectors: `(a . b) / (||a|| * ||b||)`.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 1e-12 {
        dot / denom
    } else {
        0.0
    }
}

/// Collapse a sequence of hidden token representations `[num_tokens, embedding_dim]`
/// into a single embedding vector according to `strategy`.
pub fn pool(
    hidden_states: &[f32],
    num_tokens: usize,
    embedding_dim: usize,
    strategy: Pooling,
    normalize: bool,
) -> Vec<f32> {
    assert_eq!(
        hidden_states.len(),
        num_tokens * embedding_dim,
        "hidden_states size mismatch"
    );
    let mut out = vec![0.0f32; embedding_dim];
    if num_tokens == 0 {
        return out;
    }

    match strategy {
        Pooling::Cls => {
            out.copy_from_slice(&hidden_states[..embedding_dim]);
        }
        Pooling::Last => {
            let start = (num_tokens - 1) * embedding_dim;
            out.copy_from_slice(&hidden_states[start..start + embedding_dim]);
        }
        Pooling::Mean => {
            for b in 0..num_tokens {
                let row = &hidden_states[b * embedding_dim..(b + 1) * embedding_dim];
                for (dst, &src) in out.iter_mut().zip(row) {
                    *dst += src;
                }
            }
            let scale = 1.0 / (num_tokens as f32);
            for v in out.iter_mut() {
                *v *= scale;
            }
        }
    }

    if normalize {
        l2_normalize(&mut out);
    }
    out
}

/// High-level embedding computation engine.
pub struct EmbedEngine<'a> {
    pub model: &'a Model<'a>,
    pub pooling: Pooling,
    pub normalize: bool,
}

impl<'a> EmbedEngine<'a> {
    pub fn new(model: &'a Model<'a>, pooling: Pooling, normalize: bool) -> Self {
        Self {
            model,
            pooling,
            normalize,
        }
    }

    /// Embed a single token sequence into a normalized representation vector.
    pub fn embed(&self, state: &mut State, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Ok(vec![0.0f32; self.model.config.embedding]);
        }
        let n = tokens.len();
        let mut hidden_states = vec![0.0f32; n * self.model.config.embedding];
        self.model
            .embed_hidden_states(state, tokens, &mut hidden_states)?;
        Ok(pool(
            &hidden_states,
            n,
            self.model.config.embedding,
            self.pooling,
            self.normalize,
        ))
    }

    /// Embed multiple token sequences sequentially.
    pub fn embed_batch(
        &self,
        state: &mut State,
        token_sequences: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(token_sequences.len());
        for seq in token_sequences {
            embeddings.push(self.embed(state, seq)?);
        }
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let mut v = [3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        let c = [2.0f32, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pooling_modes() {
        let hidden = [
            1.0f32, 2.0, // token 0
            3.0, 4.0, // token 1
            5.0, 6.0, // token 2
        ];
        let cls = pool(&hidden, 3, 2, Pooling::Cls, false);
        assert_eq!(cls, vec![1.0, 2.0]);

        let last = pool(&hidden, 3, 2, Pooling::Last, false);
        assert_eq!(last, vec![5.0, 6.0]);

        let mean = pool(&hidden, 3, 2, Pooling::Mean, false);
        assert_eq!(mean, vec![3.0, 4.0]);
    }
}
