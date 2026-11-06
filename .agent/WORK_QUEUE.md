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
- Status: COMPLETED
- Rationale: Autoregressive decoding is strictly memory-bandwidth bound. With Ferrite's batched prefill engine already in place, speculative decoding executes $K$ tokens from a fast draft model (e.g. 0.5B) and verifies them concurrently in a single target model forward pass via batched GEMM, achieving 2x-3x higher tokens/sec on CPU without external dependencies.
- Files: `src/speculative.rs`, `src/model.rs`, `src/main.rs`, `tests/speculative.rs`
- Acceptance:
  1. Speculative verification kernel evaluating draft tokens concurrently via `Model::prefill_all_logits`.
  2. Rollback mechanism (`State::truncate_cache`) for target and draft KV-caches on rejected candidates.
  3. CLI flags `--draft <model>` and `--spec-k <k>` in `src/main.rs` with runtime streaming and acceptance statistics.
  4. Integration tests in `tests/speculative.rs` verifying 100% acceptance with identical models, and guaranteed exact baseline output equivalence even when the draft model diverges.
- Tests: `tests/speculative.rs::speculative_identical_models_accept_all_draft_tokens`, `tests/speculative.rs::speculative_divergent_draft_model_corrects_and_matches_target`. All 107 tests pass.
- Completion evidence: Bit-identical token output verified with divergent draft model; acceptance rate tracking verified.

### Phase 12: Multithreaded Prefill & SIMD Portability (ARM NEON & AVX-512)
- Priority: P1
- Status: COMPLETED
- Rationale: Amortized GEMM during prefill parallelized across rows using worker threads via `Weight::matmul_threaded`. Cross-platform SIMD integer kernels via ARM NEON (`src/quant/neon.rs`) for Apple Silicon and ARM64 servers, and AVX-512 (`src/quant/avx512.rs`) for modern x86 architectures without external dependencies.
- Files: `src/model.rs`, `src/quant/mod.rs`, `src/quant/idot.rs`, `src/quant/neon.rs`, `src/quant/avx512.rs`, `tests/model.rs`
- Acceptance: Multi-threaded batched prefill and cross-platform SIMD kernels verified bit-identical.
- Tests: `tests/model.rs::threaded_prefill_matches_single_thread`, `quant::avx512::tests`. All 113 tests pass.
- Completion evidence: Bit-identical logit and KV-cache verification under multi-threaded prefill; committed in `e3aa884`.

### Phase 13: Zero-Dependency Embedded HTTP / SSE Inference Daemon (OpenAI-Compatible)
- Priority: P1
- Status: COMPLETED
- Rationale: Ferrite operates as a standalone local inference server exposing OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`) with streaming Server-Sent Events (SSE) directly over `std::net::TcpListener` without external crates (zero dependencies; no tokio, no axum, no serde).
- Files: `src/server.rs`, `src/lib.rs`, `src/main.rs`, `tests/server.rs`
- Acceptance: HTTP/1.1 request handling, streaming chunked SSE responses with `data: [DONE]`, CLI subcommand `ferrite serve <model> [--host <host>] [--port <port>]`.
- Tests: Integration test suite in `tests/server.rs` testing live TCP connections, health endpoint, models endpoint, non-streaming chat completions, and real-time SSE streaming. All 117 tests pass.
- Completion evidence: Full OpenAI-compatible chat completion and SSE streaming verified with integration tests; committed in `f2af227`.

## Backlog

### Phase 14: Dynamic RoPE Scaling & Extended Context (Linear, NTK-Aware, and YaRN)
- Priority: P1
- Status: READY TO IMPLEMENT
- Rationale: Long context models (32k to 128k tokens) require frequency scaling (NTK-aware / YaRN / linear RoPE scaling) based on GGUF metadata `rope.scaling.type` and `rope.scaling.factor`.
- Files: `src/ops.rs`, `src/model.rs`, `tests/model.rs`
- Acceptance: Compute scaled RoPE inv_freq with wavelength adjustments for long contexts; bit-identical output for base lengths.

### Phase 15: Flash-Attention Tile-Based KV Kernel (Online Softmax)
- Priority: P2
- Status: QUEUED
- Rationale: Standard attention materializes the full attention score vector `O(context)`. Online softmax tiling computes attention incrementally in block tiles `O(tile_size)` with running max and sum, reducing peak memory traffic and L1/L2 cache misses during long-context generation.
- Files: `src/ops.rs`, `src/model.rs`, `examples/bench.rs`
- Acceptance: Tile-based online softmax attention kernel agreeing bit-for-bit with baseline attention.

### Phase 16: Zero-Copy LoRA Adapter Application
- Priority: P2
- Status: QUEUED
- Rationale: Load and blend GGUF LoRA adapter tensors (`lora_a`, `lora_b`) at runtime with scaling alpha, either dynamically fusing into activations or into weights without duplicating model storage.
- Files: `src/lora.rs`, `src/model.rs`, `tests/lora.rs`
- Acceptance: Load LoRA GGUF, compute low-rank additive delta `(A * B) * (alpha / rank)`, verify output against reference adapter forward pass.


