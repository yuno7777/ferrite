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
pub mod lmstudio;
pub mod map;
pub mod synth;

pub use gguf::{GgmlType, Gguf, TensorInfo, Value};
pub use map::Mmap;