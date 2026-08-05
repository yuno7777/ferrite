# Changelog

Notable changes, newest first. This project has not cut a release yet, so
everything below is unreleased.

## Unreleased

### Added

- **AVX2 kernels for the integer dots**, behind a cfg gate and a cached runtime
  check. `Q4_K` drops from 0.99 ms to 0.36 ms — 3.9x against the plain
  reference. Only the integer paths are vectorized, because i32 addition is
  associative so a regrouped vector sum is *identical* to the scalar one, and
  the tests assert exactly that
- **int8 activations and integer dot products.** The activation is quantized
  once per matvec and every row dots against it in i32. Selected for `Q8_0`
  (2.2x over the reference) and `Q4_K` (1.3x) — the latter being the first path
  that beats plain expand-then-dot for the format most models ship as.
  `State::set_int8(false)` restores the exact arithmetic
- `Workspace`, holding the reusable scratch row and the quantized activation,
  so a decode step allocates nothing

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

- Every matvec strategy wins for some weight type and loses for another, so the
  choice is made per type by measurement. f32 fusion is **not** used for `Q4_K`
  (0.67x); int8 is **not** used for `Q4_0` (0.67x). Both kernels are kept and
  tested — only the selection differs
- AVX2 changed `Q8_0` by nothing at all: LLVM had already vectorized its flat
  inner loop. Intrinsics paid 2.7x on `Q4_K`, whose nibble masking
  autovectorization handles badly. They pay where the compiler has already
  failed, and nowhere else
- The int8 path is the one place the numbers change rather than just the speed.
  Bounded at half a step per element, verified against the exact path through a
  whole forward pass, and switchable
- The first version of the fused kernels was *slower* than the code it
  replaced. `lanes[index & 3]` is correct and unvectorizable; `chunks_exact(4)`
  with a constant lane index computes the same thing and is not
- Only the `llama` architecture loads. Others are refused by name at load,
  because a transformer wired with the wrong layer structure produces fluent
  nonsense rather than an error
