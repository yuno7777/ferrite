# ferrite

Pure-Rust LLM inference. Reads GGUF. **Zero dependencies.**

Not "few dependencies" — none:

```console
$ cargo tree
ferrite v0.1.0
```

The memory mapping is two syscalls declared inline, the quantization block
layouts are written out by hand, and the threading is `std::thread`. If it
compiles, it compiles offline.

## Status

Phase 1 of 5. **It does not generate tokens yet.** What works today:

- GGUF v2/v3 container parsing — all 13 metadata types, tensor table,
  alignment-aware data offsets, zero-copy tensor access off an mmap
- Quantization block layouts for the Q4_0…Q8_0 and K-quant families
- LM Studio model discovery, so it reads the models you already downloaded
- `ferrite info`, which is a genuinely useful GGUF inspector on its own

Roadmap in [ROADMAP](#roadmap).

## Usage

```bash
cargo build --release

ferrite list                    # models LM Studio has downloaded
ferrite where                   # directories being searched
ferrite info <model> -t         # header, hyperparameters, quant mix, tensors
```

`<model>` is a path to a `.gguf`, or any substring of an id from
`ferrite list` — `ferrite info llama-3.2` is enough if it's unambiguous.
Ambiguous matches are an error rather than a coin flip.

No model on hand? Generate a valid one:

```bash
cargo run --example fixture -- tiny.gguf
cargo run -- info tiny.gguf -t
```

```text
size                  77.9 KB
gguf                  v3
arch                  llama
parameters            132K
layers                2
embedding             256
heads                 8
kv heads              2
tokenizer             gpt2 (4 tokens)
bos token             0 "<s>"

tensors               4
  Q4_K          2 tensors     72.0 KB   92.5%
  F32           2 tensors      5.0 KB    6.4%
```

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
mapping before any kernel runs. `cargo test` includes the corrupt-input cases.

**mmap, not read.** A 4 GB model must not become a 4 GB `Vec`. `src/map.rs`
declares `CreateFileMappingW`/`MapViewOfFile` and `mmap` directly rather than
pulling in `memmap2` or `libc`.

**Symlinks are skipped, never followed** — during model discovery, a symlinked
models directory would otherwise loop.

**Correctness gates speed.** A transposed weight or a wrong RoPE produces
*fluent garbage*, not a crash, so the f32 reference path (phase 2) lands and is
verified before any quantized or SIMD kernel is allowed on top of it.

## Roadmap

| Phase | Scope | State |
| --- | --- | --- |
| 1 | GGUF container, LM Studio discovery, inspector CLI | done |
| 2 | BPE + SPM tokenizer from the GGUF vocab | next |
| 3 | f32 forward pass, single-threaded, greedy decode — correctness reference | |
| 4 | Q4_K/Q6_K fused dequant GEMV, threading, AVX2 | |
| 5 | Benchmarks vs llama.cpp, batched prefill, speculative decoding | |

## Development

```bash
cargo test              # no network, no model download required
cargo clippy --all-targets
cargo fmt --check
```

CI runs the suite on Linux, Windows, and macOS, and fails the build if anyone
adds a dependency.

## License

MIT. See [LICENSE](LICENSE).
