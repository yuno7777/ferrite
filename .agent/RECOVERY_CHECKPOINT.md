# Recovery Checkpoint

- Timestamp: 2026-11-06T06:02:55+05:30
- Repository: `/home/abhishek/Projects/ferrite-audit`
- Branch: `codex/audit-baseline`
- Baseline commit: `e174548e30b36271c4302646cdb9fe861926321b`
- Verified HEAD: `2787c52` (`feat: batched prefill, Qwen 2.5 support, GGUF hardening and autonomous runner`)

## Completed in this Execution
1. GGUF Parser Hardening:
   - Validated array element type tags against nested arrays and unknown variants.
   - Guarded metadata count and tensor count against pathological records exceeding file buffer bounds.
   - Guarded string lengths and array counts against buffer exhaustion.
   - Enforced non-zero tensor dimensions.
   - Added unit tests: `rejects_absurd_tensor_count`, `rejects_zero_dimension_tensor`.
2. Memory Mapping Invariants:
   - Documented read-only Cow invariants, truncation handling (`SIGBUS`/`STATUS_IN_PAGE_ERROR`).
   - Derived `Debug` for `Mmap`.
   - Added unit tests in `src/map.rs`.
3. Systemd Autonomous Maintenance Runner:
   - Created `scripts/ferrite_agent_runner.sh` with `codex exec`.
   - Installed and enabled `~/.config/systemd/user/ferrite-maintenance.service` and `ferrite-maintenance.timer` (active, firing every 30m).
4. Phase 8 Batched Prefill Engine:
   - Added `ops::matmul` for batched matrix-matrix multiplication.
   - Added `Weight::matmul` amortizing weight decompression across prompt token batches.
   - Fixed latent bug in `Weight::matvec` where `scratch` slice was unconstrained before `dequantize`.
   - Added `Model::prefill` with causal attention and batched FFN/attention projection.
   - Wired `model.prefill` into CLI prompt ingestion in `src/main.rs`.
   - Added unit tests: `matmul_matches_batched_matvec`, `the_batched_matmul_agrees_with_the_matvec`, `batched_prefill_matches_sequential_forward`.
   - Bit-identical KV cache and logit outputs verified.
5. Phase 9 Multi-Architecture Support for Qwen 2.5 (`qwen2`):
   - Added `"qwen2"` to supported GGUF model architectures in `src/model.rs`.
   - Supported architecture-specific default RoPE frequencies (1,000,000.0 for Qwen 2.5).
   - Added optional attention projection biases (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`) dynamically loaded via `load_vector_optional`.
   - Updated byte-level BPE tokenizer to accept `"qwen2"`.
   - Added `qwen2_fixture` and regression test `qwen2_runs_with_attention_biases_and_prefill` in `tests/model.rs`.
6. Phase 10 Multi-Architecture Support for Gemma 2 (`gemma2` / `gemma`):
   - Added `"gemma"` and `"gemma2"` to supported GGUF model architectures and tokenizer models.
   - Implemented input embedding scaling by `sqrt(embedding_dim)`.
   - Decoupled `head_dim` from `embedding` for architectures where `heads * head_dim != embedding`.
   - Added `attn_logit_softcapping` and `final_logit_softcapping` (`cap * tanh(raw / cap)`).
   - Implemented alternating sliding window causal attention masking.
   - Implemented 4-stage layer normalization with optional `post_attention_norm` and `post_ffn_norm` across both decode `forward` and `prefill`.
   - Added `gemma2_fixture` and regression test `gemma2_runs_with_softcapping_and_norms` in `tests/model.rs`.
7. Phase 11 Continuous Speculative Decoding:
   - Added `Model::prefill_all_logits` for batched evaluation emitting all intermediate candidate logits.
   - Added `State::truncate_cache` and `State::set_logits` for precise rollback and state synchronization.
   - Built zero-dependency `SpeculativeEngine` and `SpeculativeStats` in `src/speculative.rs`.
   - Wired CLI flags `--draft <model>` and `--spec-k <k>` into `src/main.rs` with token streaming and acceptance tracking.
   - Added integration test suite `tests/speculative.rs` verifying 100% acceptance on identical models and guaranteed exact baseline parity on divergent models.
8. Phase 12 Multithreaded Prefill & SIMD Portability:
   - Implemented `Weight::matmul_threaded` partitioning row sets across worker threads with disjoint unsafe pointer writes via `SyncMutPtr`.
   - Connected `matmul_threaded` across attention and FFN projections in both `Model::prefill` and `Model::prefill_all_logits`.
   - Implemented ARM NEON SIMD kernels in `src/quant/neon.rs` (`vld1q_s8`, `vmull_s8`, `vmlal_s8`, `vpadalq_s16`, `vaddvq_s32`).
   - Implemented AVX-512 SIMD kernels in `src/quant/avx512.rs` with runtime CPU feature detection.
   - Added integration test `threaded_prefill_matches_single_thread` verifying bit-identical outputs.
9. Phase 13 Zero-Dependency Embedded HTTP / SSE Inference Daemon:
   - Implemented `src/server.rs` running directly on `std::net::TcpListener` and `std::net::TcpStream`.
   - Implemented OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health`.
   - Built real-time Server-Sent Events (SSE) streaming for `stream: true` ending in `data: [DONE]`.
   - Added CLI subcommand `ferrite serve <model> [--host <host>] [--port <port>]`.
   - Added live socket integration test suite in `tests/server.rs`.
10. Code Quality & Invariants:
   - All 117 tests pass across all targets (`cargo test --all-targets`).
   - Strictly 0 external dependencies (`cargo tree` is 1 line).
   - `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`, and `git diff --check` pass with zero errors.

11. Phase 14 Dynamic RoPE Scaling & Extended Context:
   - Implemented `RopeScaling` (Linear, YaRN) and `ops::rope_scaled` in `src/ops.rs`.
   - Supported partial rotary dimensions (`rotary_dim <= head_dim`) passing unrotated dimensions through cleanly.
   - Decoupled `head_dim` derivation from `rope.dimension_count` so partial rotary dimensions do not corrupt model `head_dim`.
   - Added unit tests in `src/ops.rs` and integration tests in `tests/model.rs` (`linear_rope_scaling_prefill_matches_forward`, `yarn_rope_scaling_prefill_matches_forward`, `partial_rotary_dim_prefill_matches_forward`).
12. Phase 15 Flash-Attention Tile-Based KV Kernel:
   - Added strided-KV `ops::attention_online` with running max/sum softmax and fixed-size score tiles.
   - Wired the kernel into single-token decode, batched prefill, and all-logits prefill.
   - Preserved causal bounds, Gemma sliding-window masking, and attention logit soft-capping.
   - Removed materialized `att` vector from `State`, eliminating $O(\text{context})$ memory allocations.
   - Added `ops::tests::tiled_online_attention_matches_materialized_reference`.
13. Verification:
   - `cargo test --all-targets` passes (127 tests).
   - Strictly 0 external dependencies (`cargo tree` is 1 line).
   - `cargo clippy --all-targets -- -D warnings` passes without warnings.
   - `cargo fmt --check` passes without differences.
   - `git diff --check` passes without errors.

## Completed in This Execution
14. Phase 16 Zero-Copy LoRA Adapter Application:
   - Added GGUF adapter loading for `lora_a`/`lora_b` pairs, with metadata or override alpha and `alpha / rank` scaling.
   - Added activation-side low-rank delta application for decode, batched prefill, and threaded batched prefill; base model tensor storage remains bound to the original GGUF.
   - Added optional dense delta accumulation and model APIs to attach or clear adapter projections.
   - Wired LoRA through `forward`, `prefill`, and `prefill_all_logits`, including the speculative decoding verification path.
   - Added CLI support via `--lora <adapter>` in both `ferrite run` and `ferrite serve`.
   - Added integration test suite `tests/lora.rs` verifying bit-identical logits between batched prefill and sequential forward, and exact mathematical match against dense baked weights.
15. Verification after Phase 16:
   - `cargo test --all-targets` passes (127 tests).
   - Strictly 0 external dependencies (`cargo tree` is 1 line: `ferrite v0.1.0`).
   - `cargo clippy --all-targets -- -D warnings` passes without warnings.
   - `cargo fmt --check` passes without differences.
   - `git diff --check` passes without errors.
16. Phase 17 Chunked Prefill & Bounded-Latency Prompt Scheduling:
   - Added `Model::prefill_chunked` and `Model::prefill_all_logits_chunked` partitioning long prompts into bounded token chunks.
   - Built stateful `ChunkedPrefill` runner with `.step()` returning chunk progress and `.run_with_progress()` callback.
   - Wired `--chunk-size <n>` CLI flag into `run` command in `src/main.rs`.
   - Verified exact numerical identity against full batched prefill and sequential forward.
   - Added integration tests in `tests/model.rs` (`chunked_prefill_matches_single_pass_and_sequential_forward`, `chunked_prefill_all_logits_matches_full_prefill_all_logits`, `chunked_prefill_runner_stateful_step_and_progress`).
17. Verification after Phase 17:
   - `cargo test --all-targets` passes (130 tests).
   - Strictly 0 external dependencies (`cargo tree` is 1 line: `ferrite v0.1.0`).
   - `cargo clippy --all-targets -- -D warnings` passes without warnings.
   - `cargo fmt --check` passes without differences.
   - `git diff --check` passes without errors.

18. Phase 18 Quantized KV Cache (Q8_0 and Q4_0 Paged/Continuous Cache):
   - Created `src/kv.rs` with `BlockQ8_0` (32 elements, 36 bytes/block) and `BlockQ4_0` (32 elements, 20 bytes/block).
   - Implemented `KvCache` enum (`F32`, `Q8_0`, `Q4_0`) cutting KV cache memory footprint by 3.5x to 6.4x.
   - Built tiled online attention kernels for Q8_0 and Q4_0 with block-aligned fast-paths and element-level fallback.
   - Integrated into `Model::forward`, `Model::prefill`, `Model::prefill_all_logits`, `State::with_kv_cache`, `ServerConfig`, and CLI `--kv-cache`.
   - Added unit tests in `src/kv.rs` and integration tests in `tests/model.rs` (`quantized_kv_cache_q8_0_forward_and_prefill_match`, `quantized_kv_cache_q4_0_forward_and_prefill_match`).
19. Verification after Phase 18:
   - `cargo test --all-targets` passes (135 tests).
   - Strictly 0 external dependencies (`cargo tree` is 1 line: `ferrite v0.1.0`).
   - `cargo clippy --all-targets -- -D warnings` passes without warnings.
   - `cargo fmt --check` passes without differences.
   - `git diff --check` passes without errors.

20. Phase 19 Paged KV-Cache & Block Table Allocator:
   - Created `src/paged.rs` with `PhysicalBlock`, `PagedPool`, `PagedSequence`, and logical-to-physical block tables.
   - Built online attention over physical blocks and implemented `Model::forward_paged` and `Model::prefill_paged`.
   - Supported instantaneous zero-copy sequence forking (`PagedSequence::fork`) with reference-counted Copy-on-Write (`prepare_for_write`) page duplication.
   - Created comprehensive integration test suite `tests/paged.rs` verifying allocation, deallocation, CoW data isolation, and bit-identical parity against contiguous execution.
22. Phase 20 Radix Tree Prefix Caching & Prompt Deduplication:
   - Created `src/prefix.rs` with `RadixCache` and `RadixNode` indexing token sequences to `PagedPool` physical blocks.
   - Implemented `match_prefix`, `insert`, `apply_prefix_to_sequence`, and `evict_lru` leaf eviction.
   - Exposed `RadixCache` and `RadixNode` in `src/lib.rs`.
   - Added integration test suite in `tests/prefix.rs` validating exact bit-level logit equivalence for cached prefix prompt reuse, multi-turn dialogue branching, and LRU eviction under memory bounds.
24. Phase 21 Grammar-Constrained Decoding:
   - Created `src/grammar.rs` with zero-dependency structural decoders:
     - `RegexMatcher` with Thompson NFA byte-level simulation, character classes, and quantifiers.
     - `JsonGrammar` RFC 8259 streaming pushdown parser supporting nested objects, arrays, booleans, and numbers.
     - `ChoiceGrammar` prefix-guided discrete string selection.
   - Built `Grammar::filter_logits` to mask non-conforming tokens to `-f32::INFINITY` and suppress EOS until completion.
   - Exposed `Grammar`, `RegexMatcher`, `JsonGrammar`, `ChoiceGrammar` in `src/lib.rs`.
   - Created integration test suite `tests/grammar.rs` verifying logit filtering, EOS suppression, and guided greedy token sampling.
26. Phase 22 Min-P and Tail Free Sampling (TFS):
   - Added `min_p` and `tfs_z` adaptive logit filtering to `Sampler` in `src/sample.rs`.
   - Built dynamic Min-P truncation ($p \ge p_{\max} \cdot \text{scale}$) and TFS second-derivative cumulative mass filtering.
   - Added `--min-p` and `--tfs` CLI options in `src/main.rs`.
   - Wired `min_p` and `tfs` into HTTP daemon OpenAI completions parsing in `src/server.rs`.
   - Added unit tests verifying tail truncation without affecting high-probability head tokens.
28. Phase 23 Repetition & Frequency/Presence Penalty Engine:
   - Created `PenaltyConfig` and `apply_penalties` in `src/sample.rs`.
   - Supported multiplicative repetition penalty ($l / \text{rep}$ or $l \cdot \text{rep}$), frequency penalty ($l - \text{count} \cdot \text{freq}$), and presence penalty ($l - \text{pres}$) over trailing context window (`last_n`).
   - Added `Sampler::sample_with_history` and builder methods in `src/sample.rs`.
   - Wired `--repeat-penalty`, `--frequency-penalty`, `--presence-penalty`, `--repeat-last-n` flags into CLI `src/main.rs`.
   - Wired penalty parameters and token sequence tracking into HTTP completions in `src/server.rs`.
   - Added unit tests in `src/sample.rs` verifying exact penalty behavior.
30. Phase 24 Continuous Batching & Multi-Sequence Scheduling:
   - Created `src/scheduler.rs` with `BatchScheduler`, `Sequence`, `SequenceRequest`, `SchedulerConfig`, and `StepOutput`.
   - Built continuous batching pipeline interleaving chunked prefill slices and decode steps in shared model passes.
   - Integrated paged virtual memory (`PagedPool`, `PagedSequence`) and Radix prefix caching (`RadixCache`).
   - Exposed `BatchScheduler`, `SchedulerConfig`, `SequenceRequest`, `StepOutput` in `src/lib.rs`.
   - Created integration test suite `tests/scheduler.rs` validating concurrent multi-sequence generation, dynamic arrival of new sequences during ongoing execution, and complete memory reclamation upon LRU eviction.
31. Verification after Phase 24:
   - `cargo test --all-targets` passes (150 tests).
   - Strictly 0 external dependencies (`cargo tree` is 1 line: `ferrite v0.1.0`).
   - `cargo clippy --all-targets -- -D warnings` passes without warnings.
   - `cargo fmt --check` passes without differences.
   - `git diff --check` passes without errors.

## Next Up in Work Queue
- Phase 25: Pure-Rust In-Engine Quantization Tool (`ferrite quantize`; status QUEUED).
