# Ferrite Development Guide

This guide covers developer prerequisites, coding standards, build workflows, testing practices, and architecture rules for contributors to Ferrite.

## 1. Prerequisites and Toolchain

- **Rust Toolchain**: Stable Rust 1.80.0 or later (verified on `rustc 1.99.1`).
- **Target Platforms**: Linux (x86_64, aarch64), macOS (Apple Silicon, Intel x86_64), Windows (MSVC x86_64).
- **External Dependencies**: None. Ferrite has a strict zero-dependency policy.

## 2. Dependency Policy

Ferrite must never introduce external third-party crates into `Cargo.toml`.

- The dependency graph must always resolve to a single line:
  ```bash
  $ cargo tree --prefix none --no-dedupe
  ferrite v0.1.0 (/home/abhishek/Projects/ferrite-audit)
  ```
- Memory mapping must use inline platform syscall declarations in `src/map.rs`.
- Concurrency must use `std::thread`, `std::sync`, and standard library atomics.
- Mathematical operations and SIMD vectorization must use portable standard library types and core architecture intrinsics in `std::arch`.

Any pull request introducing dependencies into `[dependencies]` will be rejected.

## 3. Repository Structure

```text
ferrite/
├── src/
│   ├── lib.rs            # Public API exports and crate documentation
│   ├── main.rs           # CLI entry point and argument parsing
│   ├── gguf.rs           # Zero-copy GGUF container parser
│   ├── map.rs            # Inline zero-dependency OS memory mapping
│   ├── model.rs          # Model weights, transformer forward, chunked/batched prefill
│   ├── ops.rs            # Baseline vector dot products, RMSNorm, RoPE, SwiGLU
│   ├── kv.rs             # Key-Value cache ring buffer and quantization
│   ├── paged.rs          # PagedAttention block manager and page tables
│   ├── prefix.rs         # Radix tree prefix caching
│   ├── sample.rs         # Greedy, temperature, top-k, top-p, TFS, min-p sampling
│   ├── speculative.rs    # Speculative decoding and DeepSeek MTP engine
│   ├── scheduler.rs      # Continuous batching sequence scheduler
│   ├── server.rs         # HTTP/SSE OpenAI-compatible daemon
│   ├── synth.rs          # Synthetic GGUF fixture writer for testing
│   ├── template.rs       # Chat templating engine
│   ├── tools.rs          # JSON function-calling and tool invocation
│   ├── tp.rs             # Intra-node tensor parallelism
│   ├── vision.rs         # Vision-language image loader and ViT projector
│   ├── quant/
│   │   ├── mod.rs        # Quantization types and block definitions
│   │   ├── dot.rs        # Fused dequantize-and-dot kernels
│   │   ├── idot.rs       # Integer dot product with INT8 activations
│   │   ├── avx2.rs       # x86_64 AVX2 SIMD intrinsics
│   │   ├── avx512.rs     # x86_64 AVX-512 SIMD intrinsics
│   │   └── neon.rs       # AArch64 ARM NEON SIMD intrinsics
│   └── tokenizer/
│       ├── mod.rs        # Tokenizer trait and dispatch
│       ├── bpe.rs        # Byte-Pair Encoding implementation
│       ├── spm.rs        # SentencePiece subword implementation
│       └── train.rs      # In-crate BPE vocabulary trainer
├── examples/
│   ├── bench.rs          # Isolated GEMV microbenchmark harness
│   └── fixture.rs        # Synthetic GGUF model generator
├── tests/
│   ├── gguf.rs           # Container parsing and fuzz error guards
│   ├── model.rs          # Forward pass, prefill, and multi-architecture tests
│   ├── speculative.rs    # Speculative verification and rollback tests
│   └── ...               # Modular integration test suites
└── docs/                 # Technical documentation suite
```

## 4. Build, Lint, and Validation Commands

Contributors must verify the following quality gates before committing changes:

### Build
```bash
cargo build --release --all-targets
```

### Test Suite
Ferrite tests run entirely offline without downloading external models. Synthetic GGUF models are generated in-memory or in temporary fixtures via `src/synth.rs`:

```bash
cargo test --all-targets
```

### Formatting
Enforce standard Rust code formatting:

```bash
cargo fmt --check
```

### Clippy
All code must pass clippy with zero warnings:

```bash
cargo clippy --all-targets -- -D warnings
```

### Whitespace and Style Check
Ensure no trailing whitespace or forbidden characters exist:

```bash
git diff --check
```

## 5. Running Fixtures and Benchmarks

### Generate Local Synthetic Fixture
```bash
cargo run --release --example fixture -- tiny.gguf
```

### Inspect Generated Model
```bash
cargo run -- info tiny.gguf -t
```

### Run GEMV Microbenchmark
```bash
cargo run --release --example bench
```

## 6. Continuous Integration (CI)

GitHub Actions workflows run on every pull request across Linux, macOS, and Windows runners:
- `cargo test --all-targets`: Asserts 100% test pass rate.
- `cargo fmt --check`: Rejects formatting discrepancies.
- `cargo clippy --all-targets -- -D warnings`: Blocks lint violations.
- Dependency gate: Checks that `cargo tree` has zero transitive dependencies.
