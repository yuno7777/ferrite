# Changelog

Notable changes, newest first. This project has not cut a release yet, so
everything below is unreleased.

## Unreleased

### Added

- **GGUF v2/v3 reader** — all 13 metadata value types, tensor table,
  alignment-aware data offsets, zero-copy tensor access off a memory mapping.
  Corrupt and truncated files are rejected rather than half-read
- **Memory mapping with no dependencies** — `CreateFileMappingW`/`MapViewOfFile`
  and `mmap` declared inline instead of pulling in `memmap2` or `libc`
- **Quantization** — `Q4_0`, `Q4_1`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, plus
  `F32`/`F16`/`BF16`
- **Fused dequantize-and-dot kernels**, bit-identical to expanding first, and
  enabled per type by measurement
- **Tokenizers** — byte-level BPE and SentencePiece, both reconstructed from the
  model file's own vocabulary, with byte fallback and literal special-token
  matching
- **Inference** — llama-architecture forward pass with RMSNorm, RoPE,
  grouped-query attention, a KV cache and SwiGLU; matvec split across worker
  threads
- **Sampling** — greedy, temperature, top-k, top-p, over a seeded RNG
- **LM Studio interop** — reads models already downloaded, no copy or convert
- **CLI** — `list`, `where`, `info`, `tokenize`, `run`
- **Fixtures without downloads** — a small GGUF *writer*, so the test suite and
  the examples build their own model files
- **Benchmark** — `examples/bench.rs`, fused against reference at real shapes

### Notes

- Fusion is **not** used for `Q4_K` despite it being the most common format.
  Measured at 0.67x, because its unpacking is cheap enough that the unfused
  path's two long vectorized loops beat eight short fused ones. The kernel is
  kept and tested; only the selection is off
- The first version of the fused kernels was *slower* than the code it
  replaced. `lanes[index & 3]` is correct and unvectorizable; `chunks_exact(4)`
  with a constant lane index computes the same thing and is not
- Only the `llama` architecture loads. Others are refused by name at load,
  because a transformer wired with the wrong layer structure produces fluent
  nonsense rather than an error
