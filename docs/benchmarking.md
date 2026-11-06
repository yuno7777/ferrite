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

### Measured Kernel Timings

| Shape | Layer | Dimension | Type | Reference | Fused | Integer (int8) | Best Speedup | Bandwidth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1B** | `attn_q` | 2048x2048 | Q4_K | 2.01 ms | 1.93 ms | **0.50 ms** | 4.03x | 4.4 GB/s |
| **1B** | `attn_q` | 2048x2048 | Q5_K | 4.65 ms | **1.87 ms** | – | 2.48x | 1.4 GB/s |
| **1B** | `attn_q` | 2048x2048 | Q6_K | 5.83 ms | **2.23 ms** | – | 2.62x | 1.4 GB/s |
| **1B** | `attn_q` | 2048x2048 | Q8_0 | 1.96 ms | 1.08 ms | **0.52 ms** | 3.77x | 8.0 GB/s |
| **1B** | `attn_q` | 2048x2048 | Q4_0 | 1.90 ms | **1.14 ms** | – | 1.67x | 1.9 GB/s |
| **1B** | `ffn_up` | 2048x8192 | Q4_K | 7.55 ms | 7.59 ms | **2.03 ms** | 3.72x | 4.3 GB/s |
| **1B** | `ffn_up` | 2048x8192 | Q5_K | 18.69 ms | **7.48 ms** | – | 2.50x | 1.4 GB/s |
| **1B** | `ffn_up` | 2048x8192 | Q6_K | 23.38 ms | **8.96 ms** | – | 2.61x | 1.4 GB/s |
| **1B** | `ffn_up` | 2048x8192 | Q8_0 | 6.75 ms | 4.33 ms | **2.16 ms** | 3.13x | 7.7 GB/s |
| **1B** | `ffn_up` | 2048x8192 | Q4_0 | 7.57 ms | **4.55 ms** | – | 1.66x | 1.9 GB/s |
| **8B** | `attn_q` | 4096x4096 | Q4_K | 7.57 ms | 7.59 ms | **2.02 ms** | 3.75x | 4.4 GB/s |
| **8B** | `attn_q` | 4096x4096 | Q5_K | 19.73 ms | **9.79 ms** | – | 2.02x | 1.1 GB/s |
| **8B** | `attn_q` | 4096x4096 | Q6_K | 33.39 ms | **9.87 ms** | – | 3.38x | 1.3 GB/s |
| **8B** | `attn_q` | 4096x4096 | Q8_0 | 14.69 ms | 9.47 ms | **2.98 ms** | 4.93x | 5.6 GB/s |
| **8B** | `attn_q` | 4096x4096 | Q4_0 | 7.54 ms | **4.51 ms** | – | 1.67x | 1.9 GB/s |
| **8B** | `ffn_down` | 14336x4096 | Q4_K | 28.18 ms | 26.46 ms | **6.74 ms** | 4.18x | 4.6 GB/s |
| **8B** | `ffn_down` | 14336x4096 | Q5_K | 66.16 ms | **26.03 ms** | – | 2.54x | 1.4 GB/s |
| **8B** | `ffn_down` | 14336x4096 | Q6_K | 81.97 ms | **33.41 ms** | – | 2.45x | 1.3 GB/s |
| **8B** | `ffn_down` | 14336x4096 | Q8_0 | 23.60 ms | 15.23 ms | **7.98 ms** | 2.96x | 7.3 GB/s |
| **8B** | `ffn_down` | 14336x4096 | Q4_0 | 28.47 ms | **17.01 ms** | – | 1.67x | 1.8 GB/s |

## 4. Key Performance Observations

1. **Integer Dot Product Wins on Q4_K and Q8_0**:
   - `Q4_K` drops from 2.01 ms (reference) to 0.50 ms (integer AVX2 path), achieving a 4.03x speedup.
   - `Q8_0` drops from 1.96 ms to 0.52 ms, achieving a 3.77x speedup and reaching 8.0 GB/s single-core effective weight bandwidth.
2. **Fused Kernels Win on Complex Formats**:
   - For `Q6_K` and `Q5_K`, unpacking multi-byte super-blocks in registers eliminates temporary heap allocations and yields 2.4x to 3.3x improvements.
3. **Where Fusion Does Not Win**:
   - Unfused `Q4_K` reference is nearly identical in latency to fused `Q4_K` (2.01 ms vs 1.93 ms). The simple nibble structure allows the compiler to vectorize the reference loop, leaving integer activation quantization as the only effective acceleration path.

## 5. How to Reproduce

Run the benchmark with optimizations enabled:

```bash
cargo run --release --example bench
```

To run with explicit CPU architecture flags:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench
```
