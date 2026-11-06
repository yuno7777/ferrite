<div align="center">

# FERRITE

**Pure-Rust LLM Inference Engine with Zero External Dependencies**

Reads GGUF. Runs natively. Fully offline.

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-2021_edition-orange.svg" alt="Rust Edition" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" /></a>
  <a href="#zero-dependencies"><img src="https://img.shields.io/badge/dependencies-0%20external-success.svg" alt="Zero Dependencies" /></a>
  <a href="#supported-architectures"><img src="https://img.shields.io/badge/architectures-17%20supported-purple.svg" alt="17 Architectures" /></a>
  <a href="#quantization-matrix"><img src="https://img.shields.io/badge/quantization-24%20formats-informational.svg" alt="24 Quant Formats" /></a>
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
ferrite v0.3.0 (/home/abhishek/Projects/ferrite)
```

Ferrite contains zero third-party crate dependencies. Operating system memory mapping uses platform-native syscalls declared inline, quantization layouts are explicitly defined, tokenizers are reconstructed from the GGUF container vocabulary, and thread management uses standard library threads. If you have the Rust compiler, it builds offline.

---

## Technical Highlights

| Capability | Implementation Details |
| :--- | :--- |
| **Zero-Copy GGUF Engine** | Direct memory mapping (`mmap` on Unix, `MapViewOfFile` on Windows) for GGUF v2 and v3. Weights remain quantized in memory pages until evaluated. |
| **17 Architectures** | Full forward pass support for Llama (2/3/3.1/3.2), Qwen 2/2.5, Qwen2-MoE, Qwen2-VL, Gemma / Gemma 2, Mixtral, DeepSeek (V2/V3/MLA/MTP), Mamba (1/2), and RWKV (6/7). |
| **24 Quantization Formats** | Native decoding and compute for `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`, `IQ2_XXS`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `TQ1_0`, `TQ2_0`, `F8_E4M3`, and `F8_E5M2`. |
| **4-Row Register-Tiled SIMD** | 4-row register-tiled AVX2+FMA+F16C integer dot products (`iDot`), `vpshufb` 1-cycle LUT kernels for `IQ4_NL`/`IQ4_XS`, and `vpsignb` multiplication-free ternary kernels for BitNet `TQ2_0`. |
| **FlashAttention-2 & MoE Cache** | Tiled online-softmax FlashAttention-2 (`F32` and direct `Q8_0` KV cache) plus LRU Mixture-of-Experts (`MoE`) expert cache with cross-layer router prefetching. |
| **Dynamic GPU & Hybrid RAG** | Runtime `dlopen`/`dlsym` CUDA and Vulkan compute probing (`DynamicGpuRuntime`), embedded `Q8_0` + BM25 + RRF hybrid RAG store (`HybridRagStore`), and single-binary WebUI playground (`/ui`). |
| **Dual Tokenizer Engines** | Native Byte-level BPE (Llama 3, Qwen) and SentencePiece (Llama 2, Mistral, Gemma) with byte fallback, exact special-token handling, and streaming UTF-8 preservation. |
| **KV Cache Systems** | Contiguous caches (`F32`, `Q8_0`, `Q4_0`), PagedAttention v1/v2 with hierarchical page tables and CoW isolation, and Radix tree prefix caching with binary disk persistence. |
| **Speculative Decoding** | Continuous draft-target speculative decoding, Medusa/Eagle multi-head prediction, Speculative Tree search, and DeepSeek native Multi-Token Prediction (MTP). |
| **Tooling and Serving** | OpenAI-compatible HTTP/SSE streaming server with built-in WebUI playground, SafeTensors converter, GGUF re-quantizer (`Q8_0`, `Q4_0`, `Q4_K`, `IQ4_NL`, `IQ4_XS`, `TQ1_0`, `TQ2_0`), model merger, LoRA merge/inference, and token watermarking. |

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

Ferrite provides full forward-pass implementations across Transformer, MoE, Vision-Language, State-Space (`Mamba`), and Linear-Attention (`RWKV`) families:

| Architecture | Identifier | Attention / Sequence Mechanism | Normalization | Key Innovations |
| :--- | :--- | :--- | :--- | :--- |
| **Llama** | `llama` | Multi-Head / Grouped-Query | RMSNorm | Rotary Position Embeddings (RoPE), SwiGLU |
| **Qwen 2 / 2.5** | `qwen`, `qwen2` | Grouped-Query Attention | RMSNorm | QKV attention projection biases |
| **Qwen2-MoE** | `qwen2moe`, `qwen2_moe` | GQA + Sparse MoE | RMSNorm | Shared experts with top-k routed experts |
| **Qwen2-VL** | `qwen2vl`, `qwen2_vl` | Vision-Language Attention | RMSNorm | Multimodal RoPE (M-RoPE) with 3D grid layout |
| **Gemma / Gemma 2**| `gemma`, `gemma2` | Decoupled GQA | 4-Stage RMSNorm | Logit soft-capping, post-attention scaling |
| **Mixtral** | `mixtral` | Sparse MoE (Top-2) | RMSNorm | Top-2 softmax expert gating with load balancing |
| **DeepSeek V2/V3** | `deepseek2`, `deepseek3` | Multi-Head Latent (MLA) | RMSNorm | Low-rank KV compression, auxiliary-loss-free MoE, MTP |
| **Mamba 1 / 2** | `mamba`, `mamba2` | Selective State-Space (SSM) | RMSNorm | $O(1)$ recurrent state, causal 1D conv, selective scan |
| **RWKV 6 / 7** | `rwkv6`, `rwkv7` | Data-Dependent Linear WKV | LayerNorm | $O(1)$ recurrent matrix state, token-shift mixing |

---

## Quantization Matrix

Ferrite provides pure-Rust dequantization and matrix-vector kernels across 24 formats:

| Format Family | Quantization Types | Block Size | Kernel Acceleration | Scale Schemes |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Floats** | `F32`, `F16`, `BF16` | 1 | Direct SIMD GEMV | IEEE 754 / Bfloat16 |
| **Legacy Quantization** | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` | 32 | 4-row tiled AVX2 iDot, fused dot | Per-block min/max delta scaling |
| **K-Quants** | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` | 256 (sub-blocks) | 4-row tiled AVX2+FMA+F16C iDot | Hierarchical 6-bit/8-bit scales |
| **Importance (`IQ*`)** | `IQ2_XXS`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS` | 32 / 256 | `vpshufb` 1-cycle LUT AVX2 iDot | Non-linear `KVALUES_IQ4NL` & lattice grids |
| **BitNet Ternary** | `TQ1_0`, `TQ2_0` | 256 | `vpsignb` multiplication-free AVX2 | 1.58-bit ternary `{-1, 0, +1}` |
| **Float8 Formats** | `F8_E4M3`, `F8_E5M2` | 1 | Vectorized FP8 GEMM | Saturated / clamped floating scales |
| **Advanced Schemes** | `imatrix`, AWQ, GPTQ, SmoothQuant, Outlier W4A16 | Variable | Activation-weighted calibration | Second-moment channel importance |

---

## Microbenchmark Performance & `llama.cpp` Verification

Run the built-in benchmark suite (`ferrite bench`) or the GEMV microbenchmark (`examples/bench.rs`):

```bash
target/release/ferrite bench --threads 4
cargo run --release --example bench
```

> [!NOTE]
> **Measurement Scope**: `examples/bench.rs` isolates single-threaded matrix-vector multiplication (GEMV) across standard transformer weight shapes, while `ferrite bench` measures both single-thread and multi-thread 4-row register-tiled AVX2 integer dot products (`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q4_0`) plus batched prefill (`pp64`) and token decode (`tg32`).

### Genuine Host Hardware Measurements (`examples/bench.rs`)

- **Processor**: AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx (4 physical cores, 8 threads)
- **SIMD Capabilities**: AVX2, FMA, F16C, BMI1, BMI2 (AVX-512 absent)
- **Operating System**: Linux 7.3.1-cachyos x86_64
- **Rust Toolchain**: rustc 1.99.1

```text
shape                         type  reference      fused       int8     best     GB/s
1B  attn_q   2048x2048        Q4_K     2.41ms     2.96ms   0.54ms    4.45x      4.1
1B  attn_q   2048x2048        Q5_K     2.96ms     2.34ms   0.73ms    4.05x      3.7
1B  attn_q   2048x2048        Q6_K     7.97ms     2.53ms   0.77ms   10.35x      4.2
1B  attn_q   2048x2048        Q8_0     1.91ms     1.21ms   0.60ms    3.20x      7.0
1B  attn_q   2048x2048        Q4_0     2.10ms     1.27ms   0.62ms    3.41x      3.6

1B  ffn_up   2048x8192        Q4_K     8.35ms     8.29ms   2.28ms    3.67x      3.9
1B  ffn_up   2048x8192        Q5_K    10.59ms     8.61ms   3.05ms    3.48x      3.5
1B  ffn_up   2048x8192        Q6_K    25.59ms    10.19ms   3.03ms    8.46x      4.2
1B  ffn_up   2048x8192        Q8_0     7.73ms     4.88ms   2.58ms    2.99x      6.4
1B  ffn_up   2048x8192        Q4_0     8.52ms     5.13ms   2.48ms    3.44x      3.5

8B  attn_q   4096x4096        Q4_K     8.51ms    12.58ms   3.26ms    2.61x      2.7
8B  attn_q   4096x4096        Q5_K    11.57ms     8.69ms   3.04ms    3.80x      3.5
8B  attn_q   4096x4096        Q6_K    25.91ms    10.29ms   3.06ms    8.47x      4.2
8B  attn_q   4096x4096        Q8_0     7.82ms     4.93ms   2.53ms    3.09x      6.6
8B  attn_q   4096x4096        Q4_0     8.57ms     5.18ms   2.46ms    3.48x      3.6

8B  ffn_down 14336x4096       Q4_K    31.04ms    30.07ms   7.87ms    3.94x      3.9
8B  ffn_down 14336x4096       Q5_K    38.08ms    30.52ms  10.40ms    3.66x      3.6
8B  ffn_down 14336x4096       Q6_K    91.45ms    36.24ms  10.66ms    8.58x      4.2
8B  ffn_down 14336x4096       Q8_0    27.85ms    17.34ms   8.88ms    3.14x      6.5
8B  ffn_down 14336x4096       Q4_0    31.44ms    18.14ms   8.56ms    3.67x      3.6
```

### Head-to-Head Verification Against `llama.cpp` (GGML C++ AVX2 + OpenMP)

Evaluated on identical `4096x4096` quantized weight tensors and activations (`threads = 4`, `g++ -O3 -mavx2 -mfma -mf16c -fopenmp`) using Ferrite v0.3.0's 4-row register-tiled AVX2+FMA+F16C kernels:

| Quant | Cosine Similarity | Ferrite Rel L2 Err | GGML C++ Rel L2 Err | Ferrite 1T | GGML C++ 1T | Ferrite 4T | GGML C++ 4T |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q4_K** | `0.99997366` | `0.006924` | `0.004188` | **1.701 ms** | 2.164 ms | **1.695 ms** | 1.685 ms |
| **Q5_K** | `0.99996585` | `0.008181` | `0.006054` | **2.604 ms** | 4.435 ms | **1.214 ms** | 1.971 ms |
| **Q6_K** | `0.99997904` | `0.007200` | `0.006830` | 4.047 ms | 2.536 ms | 1.998 ms | 1.610 ms |
| **Q8_0** | `1.00000000` | `0.012148` | `0.012148` | 1.796 ms | 1.442 ms | 1.331 ms | 1.154 ms |
| **Q4_0** | `1.00000000` | `0.007835` | `0.007835` | 2.287 ms | 1.701 ms | **1.315 ms** | 1.473 ms |

For complete analysis and methodology, see [docs/benchmarking.md](docs/benchmarking.md).

---

## Universal Local Model Discovery

Ferrite automatically detects and accesses `.gguf` models downloaded across local AI tools without moving or duplicating files:

| Storage Location | Ecosystem / Application |
| :--- | :--- |
| `~/.lmstudio/models`, `~/.cache/lm-studio/models` | LM Studio (0.2 and 0.3+) |
| `~/.cache/huggingface/hub`, `$HF_HOME/hub` | Hugging Face Hub cache (`models--<org>--<repo>/snapshots/...`) |
| `~/jan/models`, `~/.config/Jan/data/models` | Jan.ai local model store |
| `~/.local/share/nomic.ai/GPT4All` | GPT4All local model store |
| `~/.local/share/ferrite/models`, `~/models` | Native Ferrite & user model directories |
| `$FERRITE_MODELS_DIR` | Custom environment variable override |

You can specify a model by its file path or any unique substring from `ferrite list`, or run `ferrite doctor` to check hardware readiness:

```bash
ferrite doctor
ferrite chat llama-3.2
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
