# Ferrite Benchmarking and Performance Analysis

This document describes the benchmark harness in Ferrite, explains what the measurements represent, details the methodology, and provides reproducible benchmark results captured directly on hardware.

## 1. Microbenchmark Scope vs End-to-End Inference

Ferrite provides an isolated matrix-vector multiplication (GEMV) microbenchmark in `examples/bench.rs`.

### What Is Measured
- The time required to multiply a quantized weight matrix of realistic transformer dimension by a single activation vector.
- Single-threaded execution evaluating kernel memory traffic and arithmetic throughput.
- Comparison across three execution strategies:
  1. `reference`: Unfused expand-to-F32 then dot product.
  2. `fused`: Fused dequantize-and-dot product.
  3. `int8`: Integer dot product (iDot) against INT8 quantized activations with AVX2 acceleration.

### What Is Not Measured
- End-to-end token generation (tokens per second).
- Multi-core thread scaling across multiple attention heads.
- Memory bus contention during multi-layered transformer forward passes.
- KV cache access overhead, normalization, RoPE rotary embeddings, softmax, or sampling.

Matrix-vector multiplication represents the vast majority of CPU execution time during single-token autoregressive decoding, making it the most critical kernel to optimize.

## 2. Tested Shapes and Weight Types

The benchmark tests dimensions taken directly from deployed models:
- **1B attention query projection (`1B attn_q`)**: 2048x2048.
- **1B feed-forward up projection (`1B ffn_up`)**: 2048x8192.
- **8B attention query projection (`8B attn_q`)**: 4096x4096.
- **8B feed-forward down projection (`8B ffn_down`)**: 14336x4096.

Supported weight formats in the benchmark:
- `Q4_K`: Super-block 4-bit quantization with 6-bit scales.
- `Q5_K`: Super-block 5-bit quantization.
- `Q6_K`: Super-block 6-bit quantization.
- `Q8_0`: 8-bit quantization with 16-bit float scale.
- `Q4_0`: 4-bit quantization with 16-bit float scale.

## 3. Real Benchmark Results

The following measurements were collected by running `cargo run --release --example bench` on the host machine.

### System Specifications
- **CPU**: AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx (4 physical cores, 8 logical threads)
- **Base / Boost Frequency**: 2.0 GHz / 3.6 GHz
- **SIMD Capabilities**: AVX2, FMA, BMI1, BMI2 (AVX-512 absent)
- **Operating System**: Linux 7.3.1-cachyos x86_64
- **Rust Toolchain**: rustc 1.99.1, cargo 1.99.1
- **Build Profile**: Release (`opt-level = 3`, LTO enabled)

### Measured Kernel Timings (`examples/bench.rs`)

| Shape | Layer | Dimension | Type | Reference | Fused | Integer (int8) | Best Speedup | Bandwidth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1B** | `attn_q` | 2048x2048 | Q4_K | 2.41 ms | 2.96 ms | **0.54 ms** | 4.45x | 4.1 GB/s |
| **1B** | `attn_q` | 2048x2048 | Q5_K | 2.96 ms | 2.34 ms | **0.73 ms** | 4.05x | 3.7 GB/s |
| **1B** | `attn_q` | 2048x2048 | Q6_K | 7.97 ms | 2.53 ms | **0.77 ms** | 10.35x | 4.2 GB/s |
| **1B** | `attn_q` | 2048x2048 | Q8_0 | 1.91 ms | 1.21 ms | **0.60 ms** | 3.20x | 7.0 GB/s |
| **1B** | `attn_q` | 2048x2048 | Q4_0 | 2.10 ms | 1.27 ms | **0.62 ms** | 3.41x | 3.6 GB/s |
| **1B** | `ffn_up` | 2048x8192 | Q4_K | 8.35 ms | 8.29 ms | **2.28 ms** | 3.67x | 3.9 GB/s |
| **1B** | `ffn_up` | 2048x8192 | Q5_K | 10.59 ms | 8.61 ms | **3.05 ms** | 3.48x | 3.5 GB/s |
| **1B** | `ffn_up` | 2048x8192 | Q6_K | 25.59 ms | 10.19 ms | **3.03 ms** | 8.46x | 4.2 GB/s |
| **1B** | `ffn_up` | 2048x8192 | Q8_0 | 7.73 ms | 4.88 ms | **2.58 ms** | 2.99x | 6.4 GB/s |
| **1B** | `ffn_up` | 2048x8192 | Q4_0 | 8.52 ms | 5.13 ms | **2.48 ms** | 3.44x | 3.5 GB/s |
| **8B** | `attn_q` | 4096x4096 | Q4_K | 8.51 ms | 12.58 ms | **3.26 ms** | 2.61x | 2.7 GB/s |
| **8B** | `attn_q` | 4096x4096 | Q5_K | 11.57 ms | 8.69 ms | **3.04 ms** | 3.80x | 3.5 GB/s |
| **8B** | `attn_q` | 4096x4096 | Q6_K | 25.91 ms | 10.29 ms | **3.06 ms** | 8.47x | 4.2 GB/s |
| **8B** | `attn_q` | 4096x4096 | Q8_0 | 7.82 ms | 4.93 ms | **2.53 ms** | 3.09x | 6.6 GB/s |
| **8B** | `attn_q` | 4096x4096 | Q4_0 | 8.57 ms | 5.18 ms | **2.46 ms** | 3.48x | 3.6 GB/s |
| **8B** | `ffn_down` | 14336x4096 | Q4_K | 31.04 ms | 30.07 ms | **7.87 ms** | 3.94x | 3.9 GB/s |
| **8B** | `ffn_down` | 14336x4096 | Q5_K | 38.08 ms | 30.52 ms | **10.40 ms** | 3.66x | 3.6 GB/s |
| **8B** | `ffn_down` | 14336x4096 | Q6_K | 91.45 ms | 36.24 ms | **10.66 ms** | 8.58x | 4.2 GB/s |
| **8B** | `ffn_down` | 14336x4096 | Q8_0 | 27.85 ms | 17.34 ms | **8.88 ms** | 3.14x | 6.5 GB/s |
| **8B** | `ffn_down` | 14336x4096 | Q4_0 | 31.44 ms | 18.14 ms | **8.56 ms** | 3.67x | 3.6 GB/s |

## 4. Head-to-Head Verification Against `llama.cpp` (GGML C++)

In v0.2.0, `ferrite bench` includes `--export-verify <file>`, which exports the exact `4096x4096` quantized weight tensors, activation vector, exact dequantized FP32 reference outputs, and Ferrite's AVX2 `iDot` outputs and timings for direct verification against `llama.cpp`'s C++ `ggml_vec_dot_*` AVX2 + OpenMP kernels (`g++ -O3 -mavx2 -mfma -mf16c -fopenmp`):

| Quant | Cosine Similarity | Ferrite Rel L2 Err | GGML C++ Rel L2 Err | Ferrite 1T | GGML C++ 1T | Ferrite 4T | GGML C++ 4T |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q4_K** | `0.99997366` | `0.006924` | `0.004188` | **1.980 ms** | 1.839 ms | **1.259 ms** | 1.059 ms |
| **Q5_K** | `0.99996585` | `0.008181` | `0.006054` | 3.241 ms | 2.515 ms | 1.704 ms | 1.113 ms |
| **Q6_K** | `0.99997904` | `0.007200` | `0.006830` | 3.226 ms | 3.050 ms | **1.720 ms** | 1.631 ms |
| **Q8_0** | `1.00000000` | `0.012148` | `0.012148` | 1.945 ms | 1.753 ms | 1.519 ms | 0.800 ms |
| **Q4_0** | `1.00000000` | `0.007835` | `0.007835` | **2.085 ms** | 2.183 ms | 1.630 ms | 1.169 ms |

## 5. Key Performance Observations

1. **Full AVX2 Integer Dot Product Coverage (`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q4_0`)**:
   - `Q6_K` drops from 7.97 ms (reference) to 0.77 ms (AVX2 integer path), achieving a **10.35x** speedup.
   - `Q4_K` drops from 2.41 ms to 0.54 ms (4.45x speedup), and `Q5_K` drops from 2.96 ms to 0.73 ms (4.05x speedup).
   - `Q8_0` and `Q4_0` match `llama.cpp` C++ with `1.00000000` cosine similarity, and Ferrite's `Q4_0` single-thread kernel (`2.085 ms`) outperforms `llama.cpp` C++ (`2.183 ms`).
2. **Lock-Free Persistent ThreadPool**:
   - Calling thread participates as worker 0 while background workers synchronize via a lock-free atomic spin-then-park barrier, reaching 10.93 GiB/s multi-threaded bandwidth on `Q8_0` and 7.45 GiB/s on `Q6_K`.

## 6. How to Reproduce

Run the CLI benchmark suite or example microbenchmark:

```bash
target/release/ferrite bench --threads 4
cargo run --release --example bench
```

