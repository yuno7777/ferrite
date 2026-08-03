# ferrite

Pure-Rust LLM inference. Reads GGUF. **Zero dependencies.**

Not "few dependencies" — none:

```console
$ cargo tree
ferrite v0.1.0
```

The memory mapping is two syscalls declared inline, the quantization block
layouts are written out by hand, the tokenizers are reconstructed from the
model file's own vocabulary, and the threading is `std::thread`. If it
compiles, it compiles offline.

## Status

It generates text. What works today:

- **GGUF v2/v3** — all 13 metadata types, tensor table, alignment-aware data
  offsets, zero-copy tensor access straight off an mmap
- **Quantization** — `Q4_K`, `Q5_K`, `Q6_K`, `Q4_0`, `Q4_1`, `Q8_0`, plus
  `F32`/`F16`/`BF16`. That covers what LM Studio actually serves: a `Q4_K_M`
  file is `Q4_K` plus `Q6_K`
- **Tokenizers** — byte-level BPE (Llama 3, Qwen, Phi) and SentencePiece
  (Llama 2, Mistral, Gemma), both built from the GGUF vocab, with byte fallback
  and literal special-token matching
- **Inference** — llama-architecture forward pass: RMSNorm, RoPE,
  grouped-query attention with a KV cache, SwiGLU. Threaded matvec
- **Sampling** — greedy, temperature, top-k, top-p, seeded and reproducible
- **LM Studio** — reads the models you already downloaded

Not yet: SIMD kernels, quantized activations, batched prefill, and
architectures other than `llama`. See the [roadmap](#roadmap).

## Usage

```bash
cargo build --release

ferrite list                              # models LM Studio has downloaded
ferrite where                             # directories being searched
ferrite info <model> -t                   # header, hyperparameters, quant mix
ferrite tokenize <model> "some text"      # ids and pieces
ferrite run <model> "a prompt" -n 128     # generate
```

`<model>` is a path to a `.gguf`, or any substring of an id from
`ferrite list` — `ferrite run llama-3.2 "hi"` is enough if it's unambiguous.
Ambiguous matches are an error rather than a coin flip.

`run` takes `--temp`, `--top-k`, `--top-p`, `--seed`, `--ctx` and `--threads`.
Temperature 0 (the default) is greedy and reproducible.

No model on hand? The fixture example writes a complete, tiny llama:

```bash
cargo run --example fixture -- tiny.gguf
cargo run -- info tiny.gguf -t
cargo run -- run tiny.gguf "hello" -n 8 --temp 0.8
```

Its weights are pseudo-random, so the text is gibberish by construction. What
it demonstrates is that the pipeline runs: parse, tokenize, forward, sample,
decode.

## LM Studio

ferrite reads LM Studio's model store directly — nothing is copied, converted,
or re-downloaded:

| Location | Version |
| --- | --- |
| `~/.lmstudio/models` | 0.3 and later |
| `~/.cache/lm-studio/models` | 0.2 and earlier |
| `$FERRITE_MODELS_DIR` | override, works with no LM Studio at all |

Models are identified as `publisher/repo/file`, mirroring the directory layout.
Multi-part shards (`*-00001-of-0000N.gguf`) are listed but not yet stitched
together.

## Design notes

**Every file is untrusted.** A truncated download and a hostile file are
indistinguishable, so every read is bounds-checked, no allocation is ever sized
from a count the file supplied, and tensor extents are validated against the
mapping before any kernel runs. The corrupt-input cases are in `cargo test`.

**mmap, not read.** A 4 GB model must not become a 4 GB `Vec`. `src/map.rs`
declares `CreateFileMappingW`/`MapViewOfFile` and `mmap` directly rather than
pulling in `memmap2` or `libc`. Weights stay quantized in the mapping and are
expanded one row at a time during a matvec — dequantizing a 4 GB model up front
would need 16 GB of f32.

**Correctness gates speed.** A transposed weight or a wrong RoPE produces
*fluent garbage*, not a crash. So the f32 reference path came first, and the
threaded path is required to be **bit-identical** to it rather than merely
close — a kernel that only approximates the reference can't be diffed against
it, which is exactly what you need when hunting a numerical bug.

**Tests compute the answer.** The forward-pass fixtures zero out most of the
network, leaving a path with a closed-form result: attention with identity
value and output projections returns the mean of what it has seen, so the KV
cache can be checked arithmetically rather than by eyeballing whether the
output looks like English.

## Roadmap

| Phase | Scope | State |
| --- | --- | --- |
| 1 | GGUF container, LM Studio discovery, inspector CLI | done |
| 2 | BPE and SentencePiece tokenizers from the GGUF vocab | done |
| 3 | f32 forward pass, KV cache, sampling, `run` | done |
| 4 | K-quant dequantization, threaded matvec | done |
| 5 | Fused quantized GEMV, AVX2 via `std::arch`, benchmarks vs llama.cpp | |
| 6 | Batched prefill, more architectures, speculative decoding | |

The honest performance note: the matvec currently dequantizes to f32 and then
does an f32 dot product. Fusing the dequantize into the inner loop and adding
SIMD is phase 5, and is where the tok/s number comes from.

## Development

```bash
cargo test              # no network, no model download required
cargo clippy --all-targets
cargo fmt --check
```

70 tests, none of which need a model file — the crate ships a small GGUF
*writer* (`src/synth.rs`) so fixtures are built rather than downloaded.

CI runs the suite on Linux, Windows, and macOS, and fails the build if anyone
adds a dependency.

## License

MIT. See [LICENSE](LICENSE).
