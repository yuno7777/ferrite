//! ferrite -- pure-Rust LLM inference.
//!
//! No dependencies. Not "minimal dependencies" -- `cargo tree` prints one line.

pub mod gguf;
pub mod map;
pub mod synth;

pub use gguf::{GgmlType, Gguf, TensorInfo, Value};
pub use map::Mmap;
