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

In v0.3.0, `ferrite bench` includes `--export-verify <file>`, which exports the exact `4096x4096` quantized weight tensors, activation vector, exact dequantized FP32 reference outputs, and Ferrite's 4-row register-tiled AVX2+FMA+F16C `iDot` outputs and timings for direct verification against `llama.cpp`'s C++ `ggml_vec_dot_*` AVX2 + OpenMP kernels (`g++ -O3 -mavx2 -mfma -mf16c -fopenmp`):

| Quant | Cosine Similarity | Ferrite Rel L2 Err | GGML C++ Rel L2 Err | Ferrite 1T | GGML C++ 1T | Ferrite 4T | GGML C++ 4T |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q4_K** | `0.99997366` | `0.006924` | `0.004188` | **1.701 ms** | 2.164 ms | **1.695 ms** | 1.685 ms |
| **Q5_K** | `0.99996585` | `0.008181` | `0.006054` | **2.604 ms** | 4.435 ms | **1.214 ms** | 1.971 ms |
| **Q6_K** | `0.99997904` | `0.007200` | `0.006830` | 4.047 ms | 2.536 ms | 1.998 ms | 1.610 ms |
| **Q8_0** | `1.00000000` | `0.012148` | `0.012148` | 1.796 ms | 1.442 ms | 1.331 ms | 1.154 ms |
| **Q4_0** | `1.00000000` | `0.007835` | `0.007835` | 2.287 ms | 1.701 ms | **1.315 ms** | 1.473 ms |

## 5. Key Performance Observations

1. **4-Row Register-Tiled AVX2 + FMA + F16C Integer Kernels**:
   - By loading each `Q8_K` / `Q8_0` activation vector into `YMM` registers once and evaluating 4 matrix rows simultaneously while accumulating scaled sub-block integer dot products in `YMM` float vectors (`_mm256_fmadd_ps`) before a single horizontal reduction at the end of the row, Ferrite v0.3.0 outperforms `llama.cpp` C++ on `Q4_K` 1T (`1.701 ms` vs `2.164 ms`, **27.2% faster**), `Q5_K` 1T (`2.604 ms` vs `4.435 ms`, **70.3% faster**), `Q5_K` 4T (`1.214 ms` vs `1.971 ms`, **62.4% faster**), and `Q4_0` 4T (`1.315 ms` vs `1.473 ms`, **12.0% faster**).
   - `Q8_0` reaches **9.24 GiB/s** single-threaded (`1.796 ms`) and **12.47 GiB/s** 4-threaded (`1.331 ms`) with `1.00000000` cosine similarity against `llama.cpp` C++.
2. **Vectorized AVX2 Activation Quantizer (`Quantized::fill_avx2`)**:
   - Quantizes 256-element activation super-blocks using `_mm256_max_ps`, SIMD round-half-away-from-zero, and `_mm256_packs_epi32`/`_mm256_packs_epi16` with bit-identical output to scalar quantization.

## 6. How to Reproduce

Run the CLI benchmark suite or example microbenchmark:

```bash
target/release/ferrite bench --threads 4
cargo run --release --example bench
```

