# Recovery Checkpoint

- Timestamp: 2026-11-06T04:20:20+05:30
- Repository: `/home/abhishek/Projects/ferrite-audit`
- Branch: `codex/audit-baseline`
- Baseline commit: `e174548e30b36271c4302646cdb9fe861926321b`

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
6. Code Quality:
   - All 104 tests pass across all targets.
   - `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`, and `git diff --check` pass with zero errors.

## Next Up in Work Queue
- Phase 10: Multi-Architecture Support for Gemma 2 (`gemma2` / `gemma`: embedding scaling `* sqrt(dim)`, Gemma RMSNorm `(1.0 + w) * x * rsqrt`, attention logit soft-capping, post-attention & post-FFN norm).

