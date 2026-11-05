# Ferrite engineering queue

## Active

### GGUF tensor byte access rejects forged ranges
- Priority: P1
- Status: COMPLETED
- Rationale: Public `TensorInfo` values could overflow unchecked offset arithmetic before slicing the memory map.
- Files: `src/gguf.rs`, `tests/gguf.rs`
- Acceptance: Offset and length conversion and end calculation are checked; malformed ranges return errors.
- Tests: `cargo test --all-targets`, `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`.
- Completion evidence: Regression test covers a forged `u64::MAX` offset; all checks pass.

### Validate metadata array element type tags
- Priority: P1
- Status: COMPLETED
- Rationale: GGUF arrays have one declared element type; accepting nested or heterogeneous values may misparse malformed files or create inconsistent metadata.
- Files: `src/gguf.rs`, `tests/gguf.rs`
- Acceptance: Reject nested arrays and invalid element tags before iterating; preserve valid homogeneous arrays.
- Tests: `cargo test --all-targets`.
- Completion evidence: Invalid nested-array tag rejected before item decoding; regression test added.

### Guard metadata and tensor counts against pathological parsing work
- Priority: P2
- Status: COMPLETED
- Rationale: Counts are attacker-controlled and can drive long loops or allocations even where truncation eventually fails.
- Files: `src/gguf.rs`, `tests/gguf.rs`
- Acceptance: Reject counts that cannot fit the remaining file's minimum record size before looping; validate non-zero tensor dimensions; bound string and array lengths against remaining buffer.
- Tests: `cargo test --all-targets`, `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`.
- Completion evidence: Added checks for minimum metadata and tensor entry wire sizes against remaining bytes, array element bounds, string bounds, non-zero tensor dims, plus regression tests (`rejects_absurd_tensor_count`, `rejects_zero_dimension_tensor`). All 100 tests pass.

### Review file mapping lifetime and truncation behavior
- Priority: P1
- Status: COMPLETED
- Rationale: Read-only mapping invariants should account for external file truncation and platform error behavior.
- Files: `src/map.rs`
- Acceptance: Document safety limits/invariants, derive Debug for diagnostics, and provide unit test coverage for file mapping.
- Tests: `cargo test --all-targets`.
- Completion evidence: Safety invariants documented (read-only Cow, SIGBUS/STATUS_IN_PAGE_ERROR truncation faults), derived `Debug`, and added unit tests in `src/map.rs`.

### Establish recurring systemd-based autonomous agent execution
- Priority: P1
- Status: COMPLETED
- Rationale: Bounded unattended execution across runtime limits requires a live systemd user timer/service executing the agent CLI.
- Files: `scripts/ferrite_agent_runner.sh`, `~/.config/systemd/user/ferrite-maintenance.service`, `~/.config/systemd/user/ferrite-maintenance.timer`
- Acceptance: Systemd service and timer configured and active, invoking `codex exec` in `/home/abhishek/Projects/ferrite-audit` with autonomous instructions.
- Tests: `systemctl --user status ferrite-maintenance.timer`.
- Completion evidence: Runner script deployed and made executable, systemd unit files installed and enabled, timer currently active and scheduled.

### Phase 8: Batched Prefill Engine (Compute-bound GEMM for prompt ingestion)
- Priority: P1
- Status: COMPLETED
- Rationale: Single-token decode GEMV reads model weights once per token; batched prefill amortizes weight reads across all prompt tokens in one pass, transforming memory-bound processing into compute-bound GEMM.
- Files: `src/ops.rs`, `src/model.rs`, `src/main.rs`, `tests/model.rs`
- Acceptance:
  1. `ops::matmul` evaluates matrix rows across multiple activation tokens in a single streaming pass.
  2. `Weight::matmul` evaluates all batch tokens against each weight row while hot in cache.
  3. Fixed latent bug in `Weight::matvec` where `scratch` slice was unconstrained before `dequantize`.
  4. `Model::prefill` executes layer-by-layer batched evaluation and populates KV-cache and final logits.
  5. `main.rs` uses `model.prefill` for prompt ingestion.
- Tests: `ops::tests::matmul_matches_batched_matvec`, `model::tests::the_batched_matmul_agrees_with_the_matvec`, `model::tests::batched_prefill_matches_sequential_forward`. All 103 tests pass.
- Completion evidence: Bit-identical agreement between batched prefill and sequential decode verified on synthetic fixture.

## Backlog

### Phase 9: Multi-Architecture Support - Qwen 2.5 (`qwen2`)
- Priority: P1
- Status: COMPLETED
- Rationale: Modern open weights commonly use Qwen 2.5 architecture (QKV bias vectors, custom RoPE frequency scaling, different head count ratios).
- Files: `src/model.rs`, `src/tokenizer/mod.rs`, `tests/model.rs`
- Acceptance: Parse `qwen2` architecture config, load QKV bias tensors via `load_vector_optional`, adapt single-token decode and batched prefill forward passes, support byte-level BPE tokenizer for `qwen2`, add synthetic fixture test.
- Tests: `tests/model.rs::qwen2_runs_with_attention_biases_and_prefill`.
- Completion evidence: Full Qwen 2.5 forward pass and prefill pass verified on synthetic fixture with exact parity. All 103 tests pass; Clippy, formatting, and diff checks pass.

## Backlog

### Phase 10: Multi-Architecture Support - Gemma 2 (`gemma2`)
- Priority: P1
- Status: COMPLETED
- Rationale: Gemma 2 is one of the most widely deployed open models, requiring unique architectural primitives: input embedding scaling (`* sqrt(dim)`), decoupled head dimension and embedding dimension, attention logit soft-capping (`50.0 * tanh(raw / 50.0)`), final logit soft-capping (`30.0 * tanh(raw / 30.0)`), alternating sliding window attention, and 4-stage layer normalization (post-attention and post-FFN RMSNorms).
- Files: `src/model.rs`, `src/tokenizer/mod.rs`, `tests/model.rs`
- Acceptance:
  1. Parse `gemma2` / `gemma` architecture strings and configuration.
  2. Implement embedding scaling and decoupled head dimension.
  3. Implement attention logit soft-capping and final logit soft-capping.
  4. Implement post-attention and post-FFN normalization layers in forward pass and batched prefill.
  5. Add synthetic Gemma 2 fixture and end-to-end regression test in `tests/model.rs`.
- Tests: `tests/model.rs::gemma2_runs_with_softcapping_and_norms`. All 105 tests pass.
- Completion evidence: Bit-identical agreement between batched prefill and sequential forward verified with softcapping bounds checked.

### Phase 11: Continuous Speculative Decoding (Draft-Target Speculation Engine)
- Priority: P1
- Status: QUEUED
- Rationale: Autoregressive decoding is strictly memory-bandwidth bound. With Ferrite's batched prefill engine already in place, speculative decoding executes $K$ tokens from a fast draft model (e.g. 0.5B) and verifies them concurrently in a single target model forward pass via batched GEMM, achieving 2x-3x higher tokens/sec on CPU without external dependencies.
- Files: `src/speculative.rs`, `src/model.rs`, `src/main.rs`, `tests/speculative.rs`
- Acceptance:
  1. Speculative verification kernel checking draft token logits against target distributions.
  2. Rollback mechanism for KV-cache on rejected draft tokens.
  3. CLI flags `--draft <path>` and `--speculative-tokens <k>`.
  4. Bit-identical or exact distribution alignment test.
- Tests: Unit tests and end-to-end benchmark.

### Phase 12: Multithreaded Prefill & SIMD Portability (ARM NEON & AVX-512)
- Priority: P2
- Status: QUEUED
- Rationale: Amortized GEMM during prefill can be parallelized across rows using Ferrite's existing zero-dependency worker thread pool (`src/thread.rs`). In addition, ARM NEON (`std::arch::aarch64`) enables native acceleration on Apple Silicon and ARM64 servers, while AVX-512 VNNI boosts newer x86 CPUs.
- Files: `src/ops.rs`, `src/model.rs`, `src/quant/neon.rs`, `src/quant/avx512.rs`
- Acceptance: Multi-threaded batched prefill and cross-platform SIMD kernels.
- Tests: Unit tests and `examples/bench.rs`.

### Phase 13: Zero-Dependency Embedded HTTP / SSE Inference Server
- Priority: P2
- Status: QUEUED
- Rationale: Ferrite can act as a local inference daemon serving OpenAI-compatible `/v1/chat/completions` and `/v1/models` using standard library `std::net` without external crates (no tokio, no axum), enabling instant drop-in with curl, web UIs, and agent harnesses.
- Files: `src/server.rs`, `src/main.rs`, `tests/server.rs`
- Acceptance: HTTP 1.1 keep-alive and streaming SSE responses.
- Tests: Integration test hitting server via standard TCP streams.

