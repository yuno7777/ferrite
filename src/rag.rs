//! Zero-Dependency Local Hybrid RAG Engine (`Q8_0` Quantized Dense Vectors + BM25 + RRF).
//!
//! Provides an embedded, single-file retrieval-augmented generation (RAG) index
//! directly inside `ferrite`:
//! - **Quantized Dense Vector Search**: Stores L2-normalized document chunk
//!   embeddings in `Q8_0` format (`4x` smaller than FP32) and scores queries via
//!   hardware-accelerated `BlockQ8_0::dot_f32`.
//! - **BM25 Lexical Scoring**: Exact keyword matching with inverse document
//!   frequency (`k1 = 1.2`, `b = 0.75`) so rare identifiers, code symbols, and
//!   product numbers are never lost by dense embeddings.
//! - **Reciprocal Rank Fusion (RRF)**: Combines dense semantic rank and BM25
//!   lexical rank (`1 / (k + rank_dense) + 1 / (k + rank_bm25)`) for state-of-the-art
//!   hybrid retrieval.
//! - **Binary Persistence (`FRAG`)**: Saves and mmap-loads the entire hybrid
//!   store to/from disk with zero external dependencies.

use crate::kv::BlockQ8_0;
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// Magic header for serialized Ferrite RAG indices (`"FRAG"`).
pub const FRAG_MAGIC: [u8; 4] = *b"FRAG";
/// Format version for `FRAG` binary files.
pub const FRAG_VERSION: u32 = 1;

/// A single indexed document chunk with quantized dense embedding and tokenized terms.
#[derive(Clone, Debug)]
pub struct RagChunk {
    pub id: usize,
    pub doc_id: String,
    pub text: String,
    pub q8_embedding: Vec<BlockQ8_0>,
    pub term_freqs: HashMap<String, u32>,
    pub token_count: usize,
}

/// Hybrid retrieval result returned by [`HybridRagStore::search_hybrid`].
#[derive(Clone, Debug)]
pub struct RagSearchResult {
    pub chunk_id: usize,
    pub doc_id: String,
    pub text: String,
    pub dense_score: f32,
    pub bm25_score: f32,
    pub rrf_score: f32,
}

/// Embedded hybrid vector + BM25 retrieval store.
#[derive(Clone, Debug)]
pub struct HybridRagStore {
    pub embed_dim: usize,
    pub chunks: Vec<RagChunk>,
    doc_freq: HashMap<String, usize>,
    total_tokens: usize,
}

impl HybridRagStore {
    /// Create a new hybrid RAG store for embeddings of dimension `embed_dim`
    /// (padded internally to a multiple of 32 for `Q8_0` blocks).
    pub fn new(embed_dim: usize) -> Self {
        assert!(embed_dim > 0, "embed_dim must be > 0");
        Self {
            embed_dim,
            chunks: Vec::new(),
            doc_freq: HashMap::new(),
            total_tokens: 0,
        }
    }

    #[inline]
    fn padded_dim(&self) -> usize {
        self.embed_dim.div_ceil(32) * 32
    }

    /// Split a long document into overlapping character-aware word chunks.
    pub fn chunk_text(text: &str, max_words: usize, overlap_words: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }
        let max_w = max_words.max(1);
        let step = max_w.saturating_sub(overlap_words).max(1);
        let mut out = Vec::new();
        let mut start = 0;
        while start < words.len() {
            let end = (start + max_w).min(words.len());
            out.push(words[start..end].join(" "));
            if end == words.len() {
                break;
            }
            start += step;
        }
        out
    }

    /// Normalize and tokenize text into lowercase alphanumeric terms for BM25.
    pub fn tokenize_terms(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_ascii_lowercase())
            .collect()
    }

    /// Insert a document chunk and its dense embedding into the store.
    pub fn add_chunk(
        &mut self,
        doc_id: impl Into<String>,
        text: impl Into<String>,
        embedding: &[f32],
    ) -> usize {
        assert_eq!(
            embedding.len(),
            self.embed_dim,
            "embedding length must match store embed_dim"
        );
        let text_str = text.into();
        let terms = Self::tokenize_terms(&text_str);
        let token_count = terms.len();
        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for t in terms {
            *term_freqs.entry(t).or_insert(0) += 1;
        }
        for term in term_freqs.keys() {
            *self.doc_freq.entry(term.clone()).or_insert(0) += 1;
        }
        self.total_tokens += token_count;

        // L2-normalize and pad to multiple of 32 for Q8_0 quantization
        let p_dim = self.padded_dim();
        let mut padded = vec![0.0f32; p_dim];
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for (i, &v) in embedding.iter().enumerate() {
                padded[i] = v / norm;
            }
        } else {
            padded[..self.embed_dim].copy_from_slice(embedding);
        }

        let q8_embedding: Vec<BlockQ8_0> =
            padded.chunks_exact(32).map(BlockQ8_0::quantize).collect();

        let id = self.chunks.len();
        self.chunks.push(RagChunk {
            id,
            doc_id: doc_id.into(),
            text: text_str,
            q8_embedding,
            term_freqs,
            token_count,
        });
        id
    }

    /// Compute dense cosine similarity between `query_embedding` and chunk `chunk`.
    pub fn score_dense(&self, query_padded: &[f32], chunk: &RagChunk) -> f32 {
        let mut sum = 0.0f32;
        for (b, blk) in chunk.q8_embedding.iter().enumerate() {
            sum += blk.dot_f32(&query_padded[b * 32..(b + 1) * 32]);
        }
        sum
    }

    /// Compute Okapi BM25 lexical score for `query_terms` against `chunk`.
    pub fn score_bm25(&self, query_terms: &[String], chunk: &RagChunk) -> f32 {
        if self.chunks.is_empty() || query_terms.is_empty() {
            return 0.0;
        }
        let n_docs = self.chunks.len() as f32;
        let avg_dl = (self.total_tokens as f32 / n_docs).max(1.0);
        let k1 = 1.2f32;
        let b = 0.75f32;
        let dl = chunk.token_count as f32;

        let mut score = 0.0f32;
        for term in query_terms {
            let tf = *chunk.term_freqs.get(term).unwrap_or(&0) as f32;
            if tf <= 0.0 {
                continue;
            }
            let df = *self.doc_freq.get(term).unwrap_or(&0) as f32;
            let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
            let denom = tf + k1 * (1.0 - b + b * (dl / avg_dl));
            score += idf * (tf * (k1 + 1.0)) / denom;
        }
        score
    }

    /// Perform hybrid search combining `Q8_0` dense cosine similarity and BM25
    /// via Reciprocal Rank Fusion (`rrf_k = 60.0`).
    pub fn search_hybrid(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Vec<RagSearchResult> {
        if self.chunks.is_empty() || top_k == 0 {
            return Vec::new();
        }
        assert_eq!(query_embedding.len(), self.embed_dim);

        let p_dim = self.padded_dim();
        let mut q_padded = vec![0.0f32; p_dim];
        let norm = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for (i, &v) in query_embedding.iter().enumerate() {
                q_padded[i] = v / norm;
            }
        } else {
            q_padded[..self.embed_dim].copy_from_slice(query_embedding);
        }

        let query_terms = Self::tokenize_terms(query_text);
        let n = self.chunks.len();

        let mut dense_scores: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.score_dense(&q_padded, c)))
            .collect();
        dense_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut bm25_scores: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.score_bm25(&query_terms, c)))
            .collect();
        bm25_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut dense_rank = vec![0usize; n];
        let mut dense_val = vec![0.0f32; n];
        for (rank, &(idx, val)) in dense_scores.iter().enumerate() {
            dense_rank[idx] = rank + 1;
            dense_val[idx] = val;
        }

        let mut bm25_rank = vec![0usize; n];
        let mut bm25_val = vec![0.0f32; n];
        for (rank, &(idx, val)) in bm25_scores.iter().enumerate() {
            bm25_rank[idx] = rank + 1;
            bm25_val[idx] = val;
        }

        let rrf_k = 60.0f32;
        let mut results: Vec<RagSearchResult> = (0..n)
            .map(|idx| {
                let c = &self.chunks[idx];
                let d_rrf = 1.0 / (rrf_k + dense_rank[idx] as f32);
                let b_rrf = if bm25_val[idx] > 0.0 {
                    1.0 / (rrf_k + bm25_rank[idx] as f32)
                } else {
                    0.0
                };
                RagSearchResult {
                    chunk_id: c.id,
                    doc_id: c.doc_id.clone(),
                    text: c.text.clone(),
                    dense_score: dense_val[idx],
                    bm25_score: bm25_val[idx],
                    rrf_score: d_rrf + b_rrf,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.rrf_score
                .total_cmp(&a.rrf_score)
                .then_with(|| b.dense_score.total_cmp(&a.dense_score))
        });
        results.truncate(top_k.min(n));
        results
    }

    /// Save the hybrid RAG store to a binary `FRAG` file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FRAG_MAGIC);
        bytes.extend_from_slice(&FRAG_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(self.embed_dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.chunks.len() as u32).to_le_bytes());

        for chunk in &self.chunks {
            let doc_bytes = chunk.doc_id.as_bytes();
            bytes.extend_from_slice(&(doc_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(doc_bytes);

            let text_bytes = chunk.text.as_bytes();
            bytes.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(text_bytes);

            bytes.extend_from_slice(&(chunk.q8_embedding.len() as u32).to_le_bytes());
            for blk in &chunk.q8_embedding {
                bytes.extend_from_slice(&blk.scale.to_le_bytes());
                for &q in &blk.qs {
                    bytes.push(q as u8);
                }
            }
        }
        std::fs::write(path, bytes)
    }

    /// Load a [`HybridRagStore`] from a binary `FRAG` file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        if data.len() < 16 || data[0..4] != FRAG_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid FRAG magic header",
            ));
        }
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != FRAG_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported FRAG version {version}"),
            ));
        }
        let embed_dim = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let n_chunks = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

        let mut store = Self::new(embed_dim);
        let mut cursor = 16usize;

        for id in 0..n_chunks {
            if cursor + 4 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated doc_id",
                ));
            }
            let doc_len = u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            let doc_id = String::from_utf8_lossy(&data[cursor..cursor + doc_len]).into_owned();
            cursor += doc_len;

            let text_len =
                u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            let text = String::from_utf8_lossy(&data[cursor..cursor + text_len]).into_owned();
            cursor += text_len;

            let n_blks = u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            let mut q8_embedding = Vec::with_capacity(n_blks);
            for _ in 0..n_blks {
                let scale = f32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap());
                cursor += 4;
                let mut qs = [0i8; 32];
                for j in 0..32 {
                    qs[j] = data[cursor + j] as i8;
                }
                cursor += 32;
                q8_embedding.push(BlockQ8_0 { scale, qs });
            }

            let terms = Self::tokenize_terms(&text);
            let token_count = terms.len();
            let mut term_freqs: HashMap<String, u32> = HashMap::new();
            for t in terms {
                *term_freqs.entry(t).or_insert(0) += 1;
            }
            for term in term_freqs.keys() {
                *store.doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
            store.total_tokens += token_count;
            store.chunks.push(RagChunk {
                id,
                doc_id,
                text,
                q8_embedding,
                term_freqs,
                token_count,
            });
        }
        Ok(store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_rag_store_combines_dense_and_bm25_and_roundtrips() {
        let mut store = HybridRagStore::new(32);
        let mut e0 = vec![0.0f32; 32];
        e0[0] = 1.0;
        let mut e1 = vec![0.0f32; 32];
        e1[1] = 1.0;

        store.add_chunk(
            "doc_rust.md",
            "Ferrite uses 4-row AVX2 register tiling for fast GGUF inference.",
            &e0,
        );
        store.add_chunk(
            "doc_python.md",
            "Python scripts use dynamic typing and garbage collection.",
            &e1,
        );

        let results = store.search_hybrid("AVX2 GGUF inference", &e0, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "doc_rust.md");
        assert!(results[0].dense_score > 0.99);
        assert!(results[0].bm25_score > 0.0);

        let tmp = std::env::temp_dir().join("ferrite_test_rag.frag");
        store.save(&tmp).unwrap();
        let loaded = HybridRagStore::load(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);

        let results2 = loaded.search_hybrid("AVX2 GGUF inference", &e0, 1);
        assert_eq!(results2[0].doc_id, "doc_rust.md");
    }
}
