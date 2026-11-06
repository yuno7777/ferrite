<div align="center">

# FERRITE

**Pure-Rust LLM Inference Engine with Zero External Dependencies**

Reads GGUF. Runs natively. Fully offline.

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-2021_edition-orange.svg" alt="Rust Edition" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" /></a>
  <a href="#zero-dependencies"><img src="https://img.shields.io/badge/dependencies-0%20external-success.svg" alt="Zero Dependencies" /></a>
  <a href="#supported-architectures"><img src="https://img.shields.io/badge/architectures-13%20supported-purple.svg" alt="13 Architectures" /></a>
  <a href="#quantization-matrix"><img src="https://img.shields.io/badge/quantization-17%20formats-informational.svg" alt="17 Quant Formats" /></a>
  <a href="#platforms"><img src="https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Platforms" /></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#technical-highlights">Highlights</a> •
  <a href="#supported-architectures">Architectures</a> •
  <a href="#quantization-matrix">Quantization</a> •
  <a href="#microbenchmark-performance">Benchmarks</a> •
  <a href="#lm-studio-integration">LM Studio</a> •
  <a href="#documentation-index">Documentation</a>
</p>

</div>

---

## Zero Dependencies

```console
$ cargo tree --prefix none --no-dedupe
ferrite v0.1.0 (/home/abhishek/Projects/ferrite-audit)
```

Ferrite contains zero third-party crate dependencies. Operating system memory mapping uses platform-native syscalls declared inline, quantization layouts are explicitly defined, tokenizers are reconstructed from the GGUF container vocabulary, and thread management uses standard library threads. If you have the Rust compiler, it builds offline.

---

## Technical Highlights

| Capability | Implementation Details |
| :--- | :--- |
| **Zero-Copy GGUF Engine** | Direct memory mapping (`mmap` on Unix, `MapViewOfFile` on Windows) for GGUF v2 and v3. Weights remain quantized in memory pages until evaluated. |
| **13 Architectures** | Full forward pass support for Llama (2/3/3.1/3.2), Qwen 2/2.5, Qwen2-MoE, Qwen2-VL, Gemma / Gemma 2, Mixtral, and DeepSeek (V2/V3/MLA/MTP). |
| **17 Quantization Formats** | Native decoding for `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`, `F8_E4M3`, and `F8_E5M2`. |
| **Fused Compute Kernels** | Fused dequantize-and-dot kernels for `Q8_0`, `Q4_0`, `Q5_K`, `Q6_K`, `F8_E4M3`, and `F8_E5M2`. Integer dot products (iDot) with INT8 activations and 32-bit accumulation. |
| **SIMD Vectorization** | Runtime feature detection for AVX2, AVX-512, and ARM NEON intrinsics with automatic fallback to scalar reference paths. |
| **Prefill Engine** | Batched and chunked prefill, asynchronous background compute with streaming events, and dynamic micro-batching with padding elimination. |
| **Dual Tokenizer Engines** | Native Byte-level BPE (Llama 3, Qwen) and SentencePiece (Llama 2, Mistral, Gemma) with byte fallback, exact special-token handling, and streaming UTF-8 preservation. |
| **KV Cache Systems** | Contiguous caches (`F32`, `Q8_0`, `Q4_0`), PagedAttention v1/v2 with hierarchical page tables and CoW isolation, and Radix tree prefix caching with binary disk persistence. |
| **Speculative Decoding** | Continuous draft-target speculative decoding, Medusa/Eagle multi-head prediction, Speculative Tree search, and DeepSeek native Multi-Token Prediction (MTP). |
| **Tooling and Serving** | OpenAI-compatible HTTP/SSE streaming server, SafeTensors converter, GGUF re-quantizer, model merger (SLERP, TIES, DARE), LoRA merge/inference, and token watermarking. |

---

## Quickstart

### Build

```bash
cargo build --release
```

### 1. Generate a Test Model Fixture

If you do not have a GGUF model on hand, generate a synthetic Llama model:

```console
$ cargo run --release --example fixture -- tiny.gguf
wrote tiny.gguf
```

> [!NOTE]
> The fixture uses pseudo-random weights to verify that the parsing, tokenization, forward pass, sampling, and decoding pipeline executes end-to-end. It is not designed to produce coherent English text.

### 2. Inspect Model Metadata and Tensors

```console
$ target/release/ferrite info tiny.gguf -t
tiny.gguf
size                  79.7 KB
gguf                  v3
name                  ferrite-fixture
arch                  llama
parameters            20K
layers                2
embedding             32
feed forward          64
heads                 4
kv heads              2
context length        512
rope freq base        10000
rms eps               0.00001
tokenizer             gpt2 (19 tokens)
bos token             0 "<s>"
eos token             1 "</s>"

tensors               21
  F32          21 tensors     77.4 KB   97.1%
```

<details>
<summary>Click to view full tensor allocation layout</summary>

```text
  token_embd.weight                        32x19              F32            2.4 KB
  blk.0.attn_norm.weight                   32                 F32             128 B
  blk.0.attn_q.weight                      32x32              F32            4.0 KB
  blk.0.attn_k.weight                      32x16              F32            2.0 KB
  blk.0.attn_v.weight                      32x16              F32            2.0 KB
  blk.0.attn_output.weight                 32x32              F32            4.0 KB
  blk.0.ffn_norm.weight                    32                 F32             128 B
  blk.0.ffn_gate.weight                    32x64              F32            8.0 KB
  blk.0.ffn_up.weight                      32x64              F32            8.0 KB
  blk.0.ffn_down.weight                    64x32              F32            8.0 KB
  blk.1.attn_norm.weight                   32                 F32             128 B
  blk.1.attn_q.weight                      32x32              F32            4.0 KB
  blk.1.attn_k.weight                      32x16              F32            2.0 KB
  blk.1.attn_v.weight                      32x16              F32            2.0 KB
  blk.1.attn_output.weight                 32x32              F32            4.0 KB
  blk.1.ffn_norm.weight                    32                 F32             128 B
  blk.1.ffn_gate.weight                    32x64              F32            8.0 KB
  blk.1.ffn_up.weight                      32x64              F32            8.0 KB
  blk.1.ffn_down.weight                    64x32              F32            8.0 KB
  output_norm.weight                       32                 F32             128 B
  output.weight                            32x19              F32            2.4 KB
```

</details>

### 3. Tokenize Text

```console
$ target/release/ferrite tokenize tiny.gguf "hello world"
Bpe vocab, 9 merges, 2 tokens
       13  "hello"
       18  " world"
```

### 4. Run Autoregressive Generation

```console
$ target/release/ferrite run tiny.gguf "hello" -n 8 --temp 0.8
llama · 1 prompt tokens · 128.0 KB kv cache · 8 threads
hellowehellhelloeeee

prefill 0.00s (339.8 tok/s) · decode 8 tokens in 0.02s (417.3 tok/s)
```

---

## Supported Architectures

Ferrite provides full forward-pass implementations for 13 model families:

| Architecture | Identifier | Attention Mechanism | Normalization | Key Innovations |
| :--- | :--- | :--- | :--- | :--- |
| **Llama** | `llama` | Multi-Head / Grouped-Query | RMSNorm | Rotary Position Embeddings (RoPE), SwiGLU |
| **Qwen 2 / 2.5** | `qwen`, `qwen2` | Grouped-Query Attention | RMSNorm | QKV attention projection biases |
| **Qwen2-MoE** | `qwen2moe`, `qwen2_moe` | GQA + Sparse MoE | RMSNorm | Shared experts with top-k routed experts |
| **Qwen2-VL** | `qwen2vl`, `qwen2_vl` | Vision-Language Attention | RMSNorm | Multimodal RoPE (M-RoPE) with 3D grid layout |
| **Gemma / Gemma 2**| `gemma`, `gemma2` | Decoupled GQA | 4-Stage RMSNorm | Logit soft-capping, post-attention scaling |
| **Mixtral** | `mixtral` | Sparse MoE (Top-2) | RMSNorm | Top-2 softmax expert gating with load balancing |
| **DeepSeek V2/V3** | `deepseek2`, `deepseek3` | Multi-Head Latent (MLA) | RMSNorm | Low-rank KV compression, auxiliary-loss-free MoE, MTP |

---

## Quantization Matrix

Ferrite provides pure-Rust dequantization and matrix-vector kernels across 17 formats:

| Format Family | Quantization Types | Block Size | Kernel Acceleration | Scale Schemes |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Floats** | `F32`, `F16`, `BF16` | 1 | Direct SIMD GEMV | IEEE 754 / Bfloat16 |
| **Legacy Quantization** | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` | 32 | Fused dequantize-and-dot, iDot | Per-block min/max delta scaling |
| **K-Quants** | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` | 256 (sub-blocks) | Super-block fused GEMV, iDot | Hierarchical 6-bit/8-bit scales |
| **Float8 Formats** | `F8_E4M3`, `F8_E5M2` | 1 | Vectorized FP8 GEMM | Saturated / clamped floating scales |
| **Advanced Schemes** | AWQ, GPTQ, SmoothQuant, Outlier W4A16 | Variable | Specialized scale migration | Channel-wise and block-wise outliers |

---

## Microbenchmark Performance

Run the GEMV benchmark harness to evaluate kernel throughput on synthetic tensors:

```bash
cargo run --release --example bench
```

> [!NOTE]
> **Measurement Scope**: This benchmark isolates single-threaded matrix-vector multiplication (GEMV) across standard transformer weight shapes. It measures compute and memory bandwidth throughput for specific kernels. It is not an end-to-end full-model tokens-per-second benchmark.

### Genuine Host Hardware Measurements

- **Processor**: AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx (4 physical cores, 8 threads)
- **SIMD Capabilities**: AVX2, FMA, BMI1, BMI2 (AVX-512 absent)
- **Operating System**: Linux 7.3.1-cachyos x86_64
- **Rust Toolchain**: rustc 1.99.1

```text
shape                         type  reference      fused       int8     best     GB/s
1B  attn_q   2048x2048        Q4_K     2.01ms     1.93ms   0.50ms    4.03x      4.4
1B  attn_q   2048x2048        Q5_K     4.65ms     1.87ms        -    2.48x      1.4
1B  attn_q   2048x2048        Q6_K     5.83ms     2.23ms        -    2.62x      1.4
1B  attn_q   2048x2048        Q8_0     1.96ms     1.08ms   0.52ms    3.77x      8.0
1B  attn_q   2048x2048        Q4_0     1.90ms     1.14ms        -    1.67x      1.9

1B  ffn_up   2048x8192        Q4_K     7.55ms     7.59ms   2.03ms    3.72x      4.3
1B  ffn_up   2048x8192        Q5_K    18.69ms     7.48ms        -    2.50x      1.4
1B  ffn_up   2048x8192        Q6_K    23.38ms     8.96ms        -    2.61x      1.4
1B  ffn_up   2048x8192        Q8_0     6.75ms     4.33ms   2.16ms    3.13x      7.7
1B  ffn_up   2048x8192        Q4_0     7.57ms     4.55ms        -    1.66x      1.9

8B  attn_q   4096x4096        Q4_K     7.57ms     7.59ms   2.02ms    3.75x      4.4
8B  attn_q   4096x4096        Q5_K    19.73ms     9.79ms        -    2.02x      1.1
8B  attn_q   4096x4096        Q6_K    33.39ms     9.87ms        -    3.38x      1.3
8B  attn_q   4096x4096        Q8_0    14.69ms     9.47ms   2.98ms    4.93x      5.6
8B  attn_q   4096x4096        Q4_0     7.54ms     4.51ms        -    1.67x      1.9

8B  ffn_down 14336x4096       Q4_K    28.18ms    26.46ms   6.74ms    4.18x      4.6
8B  ffn_down 14336x4096       Q5_K    66.16ms    26.03ms        -    2.54x      1.4
8B  ffn_down 14336x4096       Q6_K    81.97ms    33.41ms        -    2.45x      1.3
8B  ffn_down 14336x4096       Q8_0    23.60ms    15.23ms   7.98ms    2.96x      7.3
8B  ffn_down 14336x4096       Q4_0    28.47ms    17.01ms        -    1.67x      1.8
```

For complete analysis and methodology, see [docs/benchmarking.md](docs/benchmarking.md).

---

## LM Studio Integration

Ferrite automatically detects and accesses models downloaded via LM Studio without moving or duplicating files:

| Storage Location | Operating System / Application Default |
| :--- | :--- |
| `~/.lmstudio/models` | Standard model directory for LM Studio 0.3+ |
| `~/.cache/lm-studio/models` | Legacy directory for LM Studio 0.2 and earlier |
| `$FERRITE_MODELS_DIR` | Custom environment variable override |

You can specify a model by its file path or any unique substring from `ferrite list`:

```bash
ferrite run llama-3.2 "Hello world"
```

---

## Documentation Index

Explore the comprehensive guides in the [docs](docs/) directory:

| Guide | Description |
| :--- | :--- |
| [Architecture Overview](docs/architecture.md) | Deep dive into memory mapping, GGUF parsing, transformer layers, KV caching, and execution flow. |
| [CLI Reference](docs/cli-reference.md) | Detailed syntax, flags, defaults, and usage examples for all commands. |
| [Supported Formats Matrix](docs/supported-formats.md) | Container versions, metadata types, quantization layouts, architectures, and kernel support. |
| [Quantization Guide](docs/quantization.md) | Numerical error bounds, fused dequantization, integer dot products, and SIMD dispatch. |
| [Benchmarking Guide](docs/benchmarking.md) | Methodology, hardware requirements, and reproduction steps for GEMV benchmarks. |
| [Library Usage](docs/library-usage.md) | Rust API examples for loading containers, tokenizing, and running inference. |
| [Development Guide](docs/development.md) | Toolchain setup, testing policies, zero-dependency requirements, and CI gates. |
| [Contributing Guidelines](CONTRIBUTING.md) | Code standards, pull request expectations, and validation checks. |

---

## License

MIT. See [LICENSE](LICENSE).
