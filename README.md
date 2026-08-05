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
- **Three matvec strategies** — expand-then-dot, fused dequantize-and-dot, and
  int8 activations with integer accumulation. Chosen per weight type by
  measurement; see [Performance](#performance)
- **Sampling** — greedy, temperature, top-k, top-p, seeded and reproducible
- **LM Studio** — reads the models you already downloaded

Not yet: explicit SIMD intrinsics, batched prefill, and architectures other
than `llama`. See the [roadmap](#roadmap).

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
fused and threaded paths are required to be **bit-identical** to it rather than
merely close — a kernel that only approximates the reference can't be diffed
against it, which is exactly what you need when hunting a numerical bug.

The int8 path is the deliberate exception, since quantizing an activation
cannot be lossless. It gets bounded-error tests instead of equality ones, a
whole-forward-pass comparison against the exact path, and a switch to turn it
off.

**Tests compute the answer.** The forward-pass fixtures zero out most of the
network, leaving a path with a closed-form result: attention with identity
value and output projections returns the mean of what it has seen, so the KV
cache can be checked arithmetically rather than by eyeballing whether the
output looks like English.

## Performance

```bash
cargo run --release --example bench
```

One matvec, single-threaded, at a 2048×2048 shape from Llama-3.2-1B, on a
16-thread x86-64 machine. Three paths: expand-then-dot (the reference), the
fused f32 kernel, and the integer kernel against an int8-quantized activation.

| type | reference | fused f32 | int8 | chosen | speedup |
| --- | --- | --- | --- | --- | --- |
| Q8_0 | 0.99 ms | 0.70 ms | **0.45 ms** | int8 | 2.2x |
| Q6_K | 3.94 ms | **1.76 ms** | – | fused | 2.2x |
| Q4_K | 1.29 ms | 2.14 ms | **0.99 ms** | int8 | 1.3x |
| Q5_K | 1.58 ms | **1.51 ms** | – | fused | 1.05x |
| Q4_0 | 1.17 ms | **1.09 ms** | 1.62 ms | fused | 1.08x |

Every one of those columns has a case where it wins, which is why the choice is
made per type by measurement rather than by picking one strategy and applying
it everywhere. Two results are worth spelling out:

**Q4_K fused is *slower* than not fusing.** Its unpacking is one mask and one
shift per byte, cheap enough that the unfused path's two long vectorized loops
beat eight short fused ones per super-block. Q6_K wins for the mirror reason —
its unpacking is expensive enough that halving memory traffic dominates. Q4_K
only got faster once the activation went to int8.

**Q4_0 int8 is slower than Q4_0 fused.** Its f32 unpacking is already trivial,
so paying to quantize the activation buys nothing back.

The integer path is the only approximation in the crate — everything else is
bit-identical to the reference. Quantizing the activation costs half a step per
element, bounded and small next to a 4-bit weight's own error, and it is the
same trade every fast inference engine makes. `State::set_int8(false)` turns it
off.

Absolute numbers are still modest: 2–9 GB/s of weights per core. llama.cpp
remains faster, now mostly on hand-written SIMD rather than on algorithm.

## Roadmap

| Phase | Scope | State |
| --- | --- | --- |
| 1 | GGUF container, LM Studio discovery, inspector CLI | done |
| 2 | BPE and SentencePiece tokenizers from the GGUF vocab | done |
| 3 | f32 forward pass, KV cache, sampling, `run` | done |
| 4 | K-quant dequantization, threaded matvec | done |
| 5 | Fused quantized GEMV, benchmark harness | done |
| 6 | int8 activations and integer dot products | done |
| 7 | AVX2 via `std::arch`, batched prefill, more architectures | next |

With the algorithm-level wins taken, what is left really is instruction-level:
explicit AVX2 through `std::arch`, and processing more than one token at a
time during prefill so the weights are read once for many activations.

## Development

```bash
cargo test              # no network, no model download required
cargo clippy --all-targets
cargo fmt --check
```

90 tests, none of which need a model file — the crate ships a small GGUF
*writer* (`src/synth.rs`) so fixtures are built rather than downloaded.

CI runs the suite on Linux, Windows, and macOS, and fails the build if anyone
adds a dependency.

## License

MIT. See [LICENSE](LICENSE).
