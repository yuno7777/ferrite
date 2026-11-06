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
pub mod dualpipe;
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
pub mod sparse;
pub mod speculative;
pub mod split;
pub mod synth;
pub mod template;
pub mod tenant;
pub mod tokenizer;
pub mod tools;
pub mod tp;
pub mod vision;

pub use convert::{convert_safetensors_to_gguf, JsonValue, SafeTensorsHeader};
pub use dualpipe::{
    DualPipeDirection, DualPipeEngine, DualPipeOp, DualPipeSchedule, DualPipeStep, LayerPartition,
};
pub use edit::GgufEditor;
pub use embed::{cosine_similarity, l2_normalize, pool, EmbedEngine, Pooling};
pub use eval::{cross_entropy_loss, ChunkPerplexity, PerplexityEvaluator, PerplexityReport};
pub use gguf::{discover_shards, GgmlType, Gguf, TensorInfo, Value};
pub use kv::{
    compute_distribution_entropy, select_adaptive_entropy_indices, AdaptiveEntropyConfig,
    BlockQ4_0, BlockQ8_0, KvCache, KvCacheType, MlaKvCache,
};
pub use map::{Mmap, MmapMut};
pub use model::ChunkedPrefill;
pub use ops::{
    dot_i8, flash_decoding_attention, gemm_int8_fused, quantize_int8_dynamic, tree_reduce_splits,
    FlashDecodingSplit, RopeCache,
};
pub use paged::{
    BlockId, HierarchicalPageTable, PageTable, PagedPool, PagedSequence, PagedSequenceV2,
    PhysicalBlock, BLOCKS_PER_PAGE_TABLE, DEFAULT_BLOCK_TOKENS,
};
pub use prefill::{AsyncChunkedPrefill, PrefillChunkEvent};
pub use prefix::{RadixCache, RadixNode};
pub use quant::awq::{dequantize_awq_int4, dequantize_gptq_int4, dot_awq_int4, dot_gptq_int4};
pub use quant::outlier::{OutlierConfig, OutlierMatrix, OutlierQuantMode};
pub use quant::quantize::{quantize_file, quantize_model};
pub use quant::smooth::{
    compute_smoothing_scales, compute_weight_channel_max, smooth_activations, smooth_weights,
    SmoothQuantConfig, SmoothQuantLinear,
};
pub use scheduler::{
    BatchScheduler, FinishReason, MicroBatchConfig, MicroBatchPrefillEngine, PackedBatch,
    SchedulerConfig, SequenceId, SequenceRequest, StepOutput,
};
pub use server::{Server, ServerConfig};
pub use sparse::{SparseAttentionMask, SparsePattern};
pub use speculative::{
    DraftKvRecycler, MedusaEngine, MedusaHead, MtpEngine, MtpTreeEngine, SpeculativeEngine,
    SpeculativeRecycleEngine, SpeculativeStats, SpeculativeTreeEngine,
};
pub use split::{split_gguf, SplitOptions};
pub use template::{ChatMessage, ChatTemplate};
pub use tenant::{
    AdmissionError, PriorityRequestQueue, RequestPriority, TenantConfig, TenantGuard,
    TenantMetrics, TenantRegistry, TokenBucket,
};
pub use tokenizer::train::{
    train_bpe, train_vocab_file, write_vocab_gguf, TrainConfig, TrainedVocab,
};
pub use tokenizer::{DetokenizerOptions, StreamingDetokenizer, Tokenizer};
pub use tp::{
    eval_tp_swiglu, eval_tp_swiglu_dual, matvec_column_parallel, matvec_row_parallel, ThreadPool,
};
pub use vision::{Image, MultiModalProjector, VisionConfig, VisionEncoder};
