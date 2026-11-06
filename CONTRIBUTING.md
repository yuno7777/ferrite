# Contributing to Ferrite

Thank you for contributing to Ferrite. Ferrite is built as an uncompromising, high-performance, zero-dependency LLM inference engine in pure Rust.

## Principles

1. **Zero External Dependencies**: `Cargo.toml` must contain zero third-party crate dependencies. All primitives (memory mapping, threading, synchronization, quantization layouts, tokenizers) are implemented using the Rust standard library and platform syscalls.
2. **Correctness Gates Speed**: Fast paths, fused kernels, and SIMD implementations must either be bit-identical to the F32 mathematical reference or, in the case of integer activation quantization, strictly bounded in error.
3. **No Network Access in Tests**: Tests must execute entirely offline without downloading external model weights. Use `src/synth.rs` to construct synthetic GGUF models.
4. **Writing Style**: Keep English straightforward and precise. Strictly avoid em dashes in documentation and comments.

## Quality Gates

Before opening a pull request, verify that every check passes cleanly:

```bash
# 1. Dependency gate (must output exactly 1 line)
cargo tree --prefix none --no-dedupe

# 2. Compile all targets
cargo build --all-targets

# 3. Run complete test suite
cargo test --all-targets

# 4. Enforce code formatting
cargo fmt --check

# 5. Zero clippy warnings
cargo clippy --all-targets -- -D warnings

# 6. Check for whitespace issues
git diff --check
```

## Pull Request Guidelines

- Group related changes into focused, atomic commits.
- Write informative commit messages describing why a change was made.
- Add regression tests for any bug fixes or new architectural features.
- If optimizing a kernel, include reproducible before-and-after microbenchmark numbers using `cargo run --release --example bench` along with CPU specifications.
