//! ferrite: pure-Rust LLM inference.
//!
//! No dependencies. Not "minimal dependencies": `cargo tree` prints one line.
//! Reads GGUF straight out of a memory mapping, including the models LM Studio
//! already downloaded.
//!
//! ```no_run
//! let path = ferrite::lmstudio::resolve("Llama-3.2-1B")?;
//! let model = ferrite::Gguf::open(&path)?;
//! println!("{:?} with {} tensors", model.arch(), model.tensors.len());
//! # Ok::<(), std::io::Error>(())
//! ```

pub mod convert;
pub mod distributed;
pub mod edit;
pub mod embed;
pub mod eval;
pub mod gguf;
pub mod grammar;
pub mod kv;
pub mod lmstudio;
pub mod lora;
pub mod map;
pub mod merge;
pub mod model;
pub mod ops;
pub mod paged;
pub mod prefill;
pub mod prefix;
pub mod quant;
pub mod sample;
pub mod scheduler;
pub mod server;
pub mod speculative;
pub mod split;
pub mod synth;
pub mod template;
pub mod tokenizer;
pub mod tools;
pub mod tp;
pub mod vision;

pub use convert::{convert_safetensors_to_gguf, JsonValue, SafeTensorsHeader};
pub use edit::GgufEditor;
pub use embed::{cosine_similarity, l2_normalize, pool, EmbedEngine, Pooling};
pub use eval::{cross_entropy_loss, ChunkPerplexity, PerplexityEvaluator, PerplexityReport};
pub use gguf::{discover_shards, GgmlType, Gguf, TensorInfo, Value};
pub use grammar::{ChoiceGrammar, Grammar, JsonGrammar, RegexMatcher, SchemaGrammar, SchemaNode};
pub use kv::{BlockQ4_0, BlockQ8_0, KvCache, KvCacheType, MlaKvCache};
pub use lora::{LoraAdapter, LoraPair};
pub use map::{Mmap, MmapMut};
pub use model::ChunkedPrefill;
pub use ops::{dot_i8, gemm_int8_fused, quantize_int8_dynamic, RopeCache};
pub use paged::{
    BlockId, HierarchicalPageTable, PageTable, PagedPool, PagedSequence, PagedSequenceV2,
    PhysicalBlock, BLOCKS_PER_PAGE_TABLE, DEFAULT_BLOCK_TOKENS,
};
pub use prefill::{AsyncChunkedPrefill, PrefillChunkEvent};
pub use prefix::{RadixCache, RadixNode};
pub use quant::awq::{dequantize_awq_int4, dequantize_gptq_int4, dot_awq_int4, dot_gptq_int4};
pub use quant::outlier::{OutlierConfig, OutlierMatrix, OutlierQuantMode};
pub use quant::quantize::{quantize_file, quantize_model};
pub use scheduler::{
    BatchScheduler, FinishReason, SchedulerConfig, SequenceId, SequenceRequest, StepOutput,
};
pub use server::{Server, ServerConfig};
pub use speculative::{
    DraftKvRecycler, MedusaEngine, MedusaHead, MtpEngine, MtpTreeEngine, SpeculativeEngine,
    SpeculativeRecycleEngine, SpeculativeStats, SpeculativeTreeEngine,
};
pub use split::{split_gguf, SplitOptions};
pub use template::{ChatMessage, ChatTemplate};
pub use tokenizer::train::{
    train_bpe, train_vocab_file, write_vocab_gguf, TrainConfig, TrainedVocab,
};
pub use tokenizer::{DetokenizerOptions, StreamingDetokenizer, Tokenizer};
pub use tp::{
    eval_tp_swiglu, eval_tp_swiglu_dual, matvec_column_parallel, matvec_row_parallel, ThreadPool,
};
pub use vision::{Image, MultiModalProjector, VisionConfig, VisionEncoder};
