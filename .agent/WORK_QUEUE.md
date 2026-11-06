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

### Phase 14: Dynamic RoPE Scaling & Extended Context (Linear, NTK-Aware, and YaRN)
- Priority: P1
- Status: COMPLETED
- Rationale: Long context models (32k to 128k tokens) require frequency scaling (NTK-aware / YaRN / linear RoPE scaling) based on GGUF metadata `rope.scaling.type` and `rope.scaling.factor`.
- Files: `src/ops.rs`, `src/model.rs`, `tests/model.rs`
- Acceptance: Compute scaled RoPE inv_freq with wavelength adjustments for long contexts; support partial rotary dimensions (`rotary_dim <= head_dim`); bit-identical output for base lengths and matching sequential forward vs prefill.
- Tests: `ops::tests::rope_scaled_linear_at_double_pos_matches_unscaled`, `ops::tests::rope_scaled_partial_dim_preserves_tail`, `tests/model.rs::linear_rope_scaling_prefill_matches_forward`, `tests/model.rs::yarn_rope_scaling_prefill_matches_forward`, `tests/model.rs::partial_rotary_dim_prefill_matches_forward`. All 125 tests pass.
- Completion evidence: Full Linear and YaRN RoPE frequency calculations implemented and verified bit-identical between sequential decode and batched prefill.

### Phase 15: Flash-Attention Tile-Based KV Kernel (Online Softmax)
- Priority: P2
- Status: COMPLETED
- Rationale: Standard attention materializes the full attention score vector `O(context)`. Online softmax tiling computes attention incrementally in block tiles `O(tile_size)` with running max and sum, reducing peak memory traffic and L1/L2 cache misses during long-context generation.
- Files: `src/ops.rs`, `src/model.rs`, `examples/bench.rs`
- Acceptance: Tile-based online softmax attention kernel agrees with materialized attention within floating-point tolerance; applies causal, sliding-window, and logit-softcap rules.
- Tests: `ops::tests::tiled_online_attention_matches_materialized_reference`; full decode, prefill, and model suites pass.
- Completion evidence: Online softmax is active in decode, batched prefill, and all-logits prefill. It holds only one 64-position score tile at a time.

## Backlog

### Phase 16: Zero-Copy LoRA Adapter Application
- Priority: P1
- Status: COMPLETED
- Rationale: Load and blend GGUF LoRA adapter tensors (`lora_a`, `lora_b`) at runtime with scaling alpha, either dynamically fusing into activations or into weights without duplicating model storage.
- Files: `src/lora.rs`, `src/model.rs`, `src/lib.rs`, `src/main.rs`, `tests/lora.rs`
- Acceptance: Load LoRA GGUF, compute low-rank additive delta `(A * B) * (alpha / rank)`, verify output against reference adapter forward pass.
- Completion evidence: Added zero-copy GGUF-bound adapter pairs, alpha/rank scaling, decode and batched activation deltas, optional dense accumulation, and dynamic binding/clearing on model projections. Wired LoRA through decode, prefill, and `prefill_all_logits` (including speculative decoding). Reference dense-delta and batched-vs-single-token tests pass.
- Tests: `tests/lora.rs::lora_adapter_forward_and_prefill_match`, `tests/lora.rs::lora_matches_baked_weights_mathematically`. All 127 tests pass.

## Backlog

### Phase 17: Chunked Prefill & Bounded-Latency Prompt Scheduling
- Priority: P1
- Status: COMPLETED
- Rationale: Very long prompt ingestion (e.g. 16k-64k tokens) monopolizes execution threads and causes high time-to-first-token (TTFT) tail latencies. Chunked prefill partitions large prompts into bounded token chunks (e.g. 512 tokens), interleaving prompt chunk ingestion with active sequence decode steps to maintain bounded response latency.
- Files: `src/model.rs`, `src/lib.rs`, `src/main.rs`, `tests/model.rs`
- Acceptance: Break prompt ingestion into configurable chunk sizes; verify identical KV-cache states and final logits against full batched prefill; support resume/continuation across chunks.
- Completion evidence: Added `Model::prefill_chunked` and `Model::prefill_all_logits_chunked` with configurable chunk boundaries. Implemented stateful `ChunkedPrefill` runner with `.step()` and `.run_with_progress()` callbacks. Wired `--chunk-size` flag in CLI `main.rs`. Added tests verifying exact parity with unchunked batched prefill across both uniform and uneven chunk sizes. All 130 tests pass.

### Phase 18: Quantized KV Cache (Q8_0 and Q4_0 Paged/Continuous Cache)
- Priority: P1
- Status: COMPLETED
- Rationale: Storing KV activations in full FP32 consumes excessive RAM on large context lengths. Quantizing KV-cache vectors to Q8_0 or Q4_0 cuts KV memory footprint by 3.5x to 6.4x with near-zero perplexity impact.
- Files: `src/kv.rs`, `src/model.rs`, `src/lib.rs`, `src/main.rs`, `src/server.rs`, `tests/model.rs`
- Acceptance: Provide configurable KV cache quantization (f32, q8_0, q4_0); online attention dynamically reads quantized scales/quants; bit-accurate verification with bounded quantization tolerance.
- Completion evidence: Created `src/kv.rs` with `BlockQ8_0` (36 bytes/block), `BlockQ4_0` (20 bytes/block), and enum `KvCache` supporting `F32`, `Q8_0`, and `Q4_0`. Added tiled online attention kernels reading quantized blocks dynamically with alignment fast-paths and element-level fallback. Integrated into `Model::forward`, `Model::prefill`, `Model::prefill_all_logits`, `State::with_kv_cache`, `ServerConfig`, and CLI `--kv-cache`. Added unit tests for roundtrip quantization, memory footprint reduction (~3.5x and ~6.4x), and integration tests in `tests/model.rs` showing forward/prefill parity and bounded tolerance against F32. All 135 tests pass.

### Phase 19: Paged KV-Cache & Block Table Allocator
- Priority: P1
- Status: COMPLETED
- Rationale: Eliminates contiguous KV buffer pre-allocation and memory fragmentation by organizing KV cache into fixed-size physical blocks (e.g. 16 or 32 tokens) with logical-to-physical block tables (vLLM style).
- Files: `src/paged.rs`, `src/model.rs`, `src/lib.rs`, `tests/paged.rs`
- Acceptance: Dynamic non-contiguous block allocation on demand; copy-on-write fork support; attention kernel updated to follow block tables.
- Completion evidence: Implemented `src/paged.rs` with `PhysicalBlock`, `PagedPool`, `PagedSequence`, and block-table mapped online attention. Added `Model::forward_paged` and `Model::prefill_paged`. Implemented zero-copy sequence forking with refcounting and Copy-on-Write (CoW) page splitting. Added comprehensive test suite `tests/paged.rs` covering pool allocation/freeing, CoW isolation, and bit-identical parity against contiguous prefill and single-token decode. All 138 tests pass.

### Phase 20: Radix Tree Prefix Caching & Prompt Deduplication
- Priority: P2
- Status: COMPLETED
- Rationale: Repeated system prompts, multi-turn dialogues, and agentic workflows share identical token prefixes. Radix-tree prefix cache indexes previously evaluated blocks by token hashes for instantaneous 0ms ingestion of cached prefixes.
- Files: `src/prefix.rs`, `src/lib.rs`, `tests/prefix.rs`
- Acceptance: Hash-based prefix matching; skip prefill for matched prefix tokens; automatic cache eviction under memory pressure.
- Completion evidence: Created `src/prefix.rs` implementing `RadixCache` and `RadixNode` with exact token sequence matching against `PagedPool` physical blocks. Implemented `insert`, `match_prefix`, `apply_prefix_to_sequence`, and least-recently-used (`evict_lru`) leaf eviction to handle pool memory pressure. Exposed in `src/lib.rs`. Created `tests/prefix.rs` validating exact logit match between cached prefix reuse and full prompt evaluation from scratch, multi-turn branching conversation trees, and LRU eviction under memory bounds. All 141 tests pass.

### Phase 21: Grammar-Constrained Decoding (Zero-Dependency JSON Schema & Regex Masking)
- Priority: P1
- Status: COMPLETED
- Rationale: Critical for reliable structured output (JSON objects, lists, numbers, booleans, specific regex patterns) without external parser crates.
- Files: `src/grammar.rs`, `src/lib.rs`, `tests/grammar.rs`
- Acceptance: Deterministic finite automaton (DFA) / state machine checking partial UTF-8 strings against schema; dynamically masking invalid token logits before sampling; zero dependencies.
- Completion evidence: Created `src/grammar.rs` with zero external dependencies supporting: (1) Thompson NFA `RegexMatcher` with character classes, alternations, concatenations, and quantifiers (`*`, `+`, `?`); (2) streaming RFC 8259 pushdown `JsonGrammar` handling nested objects, arrays, numbers, strings, and booleans; (3) discrete `ChoiceGrammar` for exact choice selection. Implemented `Grammar::filter_logits` to mask out non-matching vocabulary tokens to `-f32::INFINITY` before sampling and gate EOS tokens until completion. Exposed in `src/lib.rs`. Created `tests/grammar.rs` verifying logit filtering, EOS suppression, and guided greedy sampling. All 144 tests pass.

### Phase 22: Min-P and Tail Free Sampling (Adaptive Logit Filtering)
- Priority: P2
- Status: COMPLETED
- Rationale: Modern sampling strategies outperforming traditional Top-P: Min-P scales truncation dynamically with top logit probability ($p \ge p_{\max} \cdot \text{scale}$); TFS calculates second derivative of sorted probabilities.
- Files: `src/sample.rs`, `src/main.rs`, `src/server.rs`
- Acceptance: Add `--min-p` and `--tfs` CLI and server flags; unit tests validating distribution truncation.
- Completion evidence: Added `min_p` and `tfs_z` to `Sampler` in `src/sample.rs`. Implemented dynamic Min-P thresholding ($p \ge p_{\max} \cdot \text{min\_p}$) and Tail Free Sampling (normalized second derivative of descending probability curve cut off at $z$). Added builder methods `with_min_p` and `with_tfs`. Wired CLI flags `--min-p` and `--tfs` into `src/main.rs` and JSON request extraction in `src/server.rs`. Added unit tests in `src/sample.rs` verifying sub-threshold tail removal. All 146 tests pass.

### Phase 23: Repetition & Frequency/Presence Penalty Engine
- Priority: P2
- Status: COMPLETED
- Rationale: Prevents degenerative repetitive loops by applying multiplicative and additive penalties on previously generated tokens within configurable context windows.
- Files: `src/sample.rs`, `src/main.rs`, `src/server.rs`
- Acceptance: Apply repetition_penalty ($l / \text{penalty}$ or $l \cdot \text{penalty}$), frequency_penalty, and presence_penalty; support in `/v1/chat/completions`.
- Completion evidence: Created `PenaltyConfig` and `apply_penalties` in `src/sample.rs` supporting multiplicative repetition penalties, frequency-proportional additive penalties, and one-time presence penalties over configurable trailing token windows (`last_n`). Added `Sampler::sample_with_history` and builder methods. Wired CLI flags `--repeat-penalty`, `--frequency-penalty`, `--presence-penalty`, `--repeat-last-n` into `src/main.rs`. Added penalty parameter extraction and token history tracking in HTTP daemon completions in `src/server.rs`. Added unit tests in `src/sample.rs` validating exact penalty arithmetic. All 148 tests pass.

### Phase 24: Continuous Batching & Multi-Sequence Scheduling
- Priority: P1
- Status: COMPLETED
- Rationale: Serving multiple concurrent user requests efficiently requires dynamic continuous batching where prefill chunks and decode steps from different requests are scheduled together in single forward passes.
- Files: `src/scheduler.rs`, `src/lib.rs`, `tests/scheduler.rs`
- Acceptance: Multi-sequence concurrency in HTTP daemon without blocking incoming requests during long generations.
- Completion evidence: Created `src/scheduler.rs` implementing `BatchScheduler`, `Sequence`, `SequenceRequest`, and `StepOutput`. Built dynamic continuous batch execution interleaving chunked prefill slices and single-token decode passes over paged memory (`PagedPool` and `PagedSequence`), with transparent prefix reuse via `RadixCache`. Exposed in `src/lib.rs`. Created `tests/scheduler.rs` validating concurrent multi-sequence generation, dynamic arrival of new sequences during ongoing decode, and complete memory reclamation upon LRU eviction. All 150 tests pass.

### Phase 25: Pure-Rust In-Engine Quantization Tool (`ferrite quantize`)
- Priority: P2
- Status: COMPLETED
- Rationale: Convert F32/F16 GGUF weights to high-speed quantized formats (Q8_0, Q4_0, Q4_K) directly inside Ferrite with zero external tools (no llama.cpp build required).
- Files: `src/quant/quantize.rs`, `src/main.rs`, `tests/quantize.rs`
- Acceptance: CLI `ferrite quantize <src.gguf> <dst.gguf> <type>`; output file valid and readable by Ferrite.
- Completion evidence: Added a zero-dependency GGUF rewriting command supporting Q8_0, Q4_0, and Q4_K output from F32/F16 tensors while preserving metadata and copying non-floating tensor encodings. Added round-trip quantization and full GGUF readback coverage for all three target formats. All targets, strict Clippy, and formatting checks pass.

### Phase 26: GGUF Multi-File Sharded Model Loader
- Priority: P2
- Status: COMPLETED
- Rationale: Large models (e.g. 70B parameters) are distributed as multiple GGUF shard files (`model-00001-of-00004.gguf`).
- Files: `src/gguf.rs`, `src/model.rs`, `src/lib.rs`, `tests/sharded.rs`
- Acceptance: Parse shard index metadata and resolve cross-file tensor mappings into a single unified virtual model.
- Completion evidence: Added `Gguf::open_sharded` and auto-shard discovery in `Gguf::open` resolving `-<idx>-of-<total>.gguf` filenames with zero-copy per-shard memory mappings. Added `shard_index` to `TensorInfo` and mapped tensor bytes dynamically to owning shard files. Added comprehensive test suite `tests/sharded.rs` validating explicit sharded paths, automatic shard discovery from any shard index, duplicate tensor rejection, and bit-identical inference output against single-file models. All 167 tests pass.

### Phase 27: Tokenizer Jinja/Mini-Template Chat Engine
- Priority: P2
- Status: COMPLETED
- Rationale: Read GGUF `tokenizer.chat_template` and format OpenAI-style message arrays (`system`, `user`, `assistant`) into exact model prompts (LLaMA-3, ChatML, Mistral, Qwen) without external template crates.
- Files: `src/template.rs`, `src/server.rs`, `src/lib.rs`, `tests/template.rs`
- Acceptance: Support standard template macros (`{% for message in messages %}`, `{{ message.content }}`); render valid conversation prompts.
- Completion evidence: Created `src/template.rs` implementing `ChatTemplate`, `ChatMessage`, zero-dependency Jinja tokenization, AST evaluation with loops (`{% for ... %}` with `loop.first`, `loop.last`, `loop.index`), conditionals (`{% if ... %}`, `{% elif ... %}`, `{% else %}`, `is defined`, `is not defined`), filters (`| trim`), whitespace trim tags (`{{-`, `-}}`, `{%-`, `-%}`), and architecture defaults (ChatML, LLaMA-3, LLaMA-2, Gemma). Integrated into `src/server.rs` OpenAI-compatible `/v1/chat/completions`. Added comprehensive integration test suite `tests/template.rs`. All 177 tests pass.

### Phase 28: Speculative Tree Drafting (Medusa / Multi-Candidate Branching)
- Priority: P2
- Status: COMPLETED
- Rationale: Extend speculative decoding from linear draft sequences to tree-structured draft verification, evaluating multiple branching candidate paths in a single target model forward pass.
- Files: `src/speculative.rs`, `src/model.rs`, `src/lib.rs`, `src/main.rs`, `tests/speculative.rs`
- Acceptance: Multi-candidate branching tree verification; best-prefix candidate selection; state synchronization; guaranteed mathematical parity with target baseline.
- Completion evidence: Added `#[derive(Clone)]` on `Workspace` and `State` in `src/model.rs`. Implemented `SpeculativeTreeEngine` in `src/speculative.rs` generating multi-path candidate branches from draft model logits and verifying them in parallel against the target model via `prefill_all_logits`. Connected `--spec-tree <b>` flag in `src/main.rs`. Re-exported in `src/lib.rs`. Added integration test `speculative_tree_drafting_matches_target_baseline` in `tests/speculative.rs` verifying bit-identical output to target model baseline. All tests pass with zero warnings.

### Phase 29: Perplexity & Cross-Entropy Evaluation Tool (`ferrite perplexity`)
- Priority: P3
- Status: COMPLETED
- Rationale: Standard tool for benchmarking model quantization accuracy and RoPE scaling perplexity across standard datasets (e.g. wikitext).
- Files: `src/eval.rs`, `src/main.rs`, `tests/eval.rs`
- Acceptance: CLI `ferrite perplexity <model> <text_file>`; compute exact cross-entropy loss, negative log likelihood, and perplexity per chunk.
- Completion evidence: Added numerically stable log-sum-exp cross-entropy, context-bounded chunk evaluation with each next-token target scored exactly once, aggregate/per-chunk NLL and perplexity reports, and CLI options `--ctx`, `--stride`, and `--threads`. Regression tests cover disjoint and overlapping strides, exact target coverage, and invalid inputs. `cargo test --all-targets`, strict Clippy, and formatting checks pass.

### Phase 30: SafeTensors to GGUF Zero-Dependency Model Transpiler (`ferrite convert`)
- Priority: P3
- Status: QUEUED
- Rationale: Convert Hugging Face `.safetensors` checkpoints and `config.json` into standardized GGUF format natively without requiring Python, PyTorch, or external libraries.
- Files: `src/convert.rs`, `src/main.rs`, `tests/convert.rs`
- Acceptance: Read Hugging Face JSON config and safetensors headers; write compliant GGUF header, metadata, and tensor data.
