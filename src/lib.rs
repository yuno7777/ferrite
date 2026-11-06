//! ferrite -- pure-Rust LLM inference.
//!
//! No dependencies. Not "minimal dependencies" -- `cargo tree` prints one line.
//! Reads GGUF straight out of a memory mapping, including the models LM Studio
//! already downloaded.
//!
//! ```no_run
//! let path = ferrite::lmstudio::resolve("Llama-3.2-1B")?;
//! let model = ferrite::Gguf::open(&path)?;
//! println!("{:?} with {} tensors", model.arch(), model.tensors.len());
//! # Ok::<(), std::io::Error>(())
//! ```

pub mod gguf;
pub mod kv;
pub mod lmstudio;
pub mod lora;
pub mod map;
pub mod model;
pub mod ops;
pub mod paged;
pub mod prefix;
pub mod quant;
pub mod sample;
pub mod server;
pub mod speculative;
pub mod synth;
pub mod tokenizer;

pub use gguf::{GgmlType, Gguf, TensorInfo, Value};
pub use kv::{BlockQ4_0, BlockQ8_0, KvCache, KvCacheType};
pub use lora::{LoraAdapter, LoraPair};
pub use map::Mmap;
pub use model::ChunkedPrefill;
pub use paged::{BlockId, PagedPool, PagedSequence, PhysicalBlock, DEFAULT_BLOCK_TOKENS};
pub use prefix::{RadixCache, RadixNode};
pub use server::{Server, ServerConfig};
pub use speculative::{SpeculativeEngine, SpeculativeStats};
pub use tokenizer::Tokenizer;
