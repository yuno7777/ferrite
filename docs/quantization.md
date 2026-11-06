# Ferrite Quantization and Kernel Dispatch

In large language model inference on CPUs, memory bandwidth is typically the primary throughput bottleneck during single-token autoregressive generation. Quantization reduces the number of bytes transferred from RAM per token.

This document details the quantization formats, dequantization mechanics, compute kernels, activation quantization, and hardware dispatch mechanisms in Ferrite.

## 1. Quantization Formats and Block Structures

Ferrite implements zero-copy parsing and arithmetic for multiple quantization schemes.

### Legacy Block Formats (Block Size 32)

- **Q4_0**: 32 values packed into 18 bytes. Contains one 16-bit float scale (`f16`) and sixteen bytes holding pairs of 4-bit unsigned integers biased by 8:
  $$w = d \cdot (q - 8)$$
- **Q4_1**: 32 values packed into 20 bytes. Contains one `f16` scale, one `f16` minimum value, and sixteen bytes of 4-bit values:
  $$w = d \cdot q + m$$
- **Q8_0**: 32 values packed into 34 bytes. Contains one `f16` scale and thirty-two 8-bit signed integers (`i8`):
  $$w = d \cdot q$$

### K-Quants / Super-Block Formats (Super-Block Size 256)

K-quants organize 256 weights into sub-blocks to capture local dynamic ranges more effectively than uniform block quantization:

- **Q4_K**: 256 weights packed into 144 bytes. Consists of a super-block scale and minimum, eight 6-bit sub-block scales and minimums, and 128 bytes containing 4-bit nibbles.
- **Q5_K**: 256 weights packed into 176 bytes. Uses 4-bit low nibbles supplemented with an additional high bit unpacked from a separate 32-byte bitmask.
- **Q6_K**: 256 weights packed into 210 bytes. Encodes 6-bit weights using 4-bit lower nibbles and 2-bit upper slices paired with 8-bit sub-block scales.

### Microscaling and FP8 (Block Size 32)

- **F8_E4M3**: 32 weights stored in 33 bytes. Consists of one 8-bit scale factor and thirty-two 8-bit floating-point numbers conforming to the E4M3 standard (1 sign bit, 4 exponent bits, 3 mantissa bits).

## 2. Compute Kernels and Execution Strategies

Ferrite provides four distinct evaluation strategies for matrix-vector multiplication (GEMV):

```
Matrix Row (Quantized Weights)
        │
        ├── Strategy 1: Expand to F32 ────> Standard Dot Product (F32)
        ├── Strategy 2: Fused Dequantize ──> Direct Multiply-Accumulate in Cache (F32)
        ├── Strategy 3: Quantize Input ────> Integer Multiply-Accumulate (i32)
        └── Strategy 4: Vectorized SIMD ───> AVX2 / AVX-512 / NEON Intrinsics
```

### Strategy 1: Unfused Reference

The weight row is unpacked into a temporary buffer of 32-bit floats, followed by an F32 dot product. This path serves as the arithmetic ground truth in test suites.

### Strategy 2: Fused Dequantize-and-Dot

Unpacking and accumulation occur within the same inner loop, keeping operations inside CPU registers without writing intermediate F32 values back to memory.

However, fusion is not universally faster:
- **Wins**: On `Q6_K` and `Q5_K`, where unpacking requires multi-step bit shifting and masking, fusing cuts memory traffic and improves throughput significantly.
- **Loses**: On `Q4_K`, unpacking is a simple shift and mask. The unfused reference allows LLVM to generate two unrolled vector loops, whereas scalar fused loops can be slower than compiler-vectorized loops.

### Strategy 3: Integer Dot Product (iDot) with INT8 Activations

Instead of promoting weights to F32, the incoming F32 activation vector $x$ is quantized to INT8 once per matrix-vector product:

1. For each block of 32 activation floats, find the maximum absolute value:
   $$s_x = \frac{\max_{i} |x_i|}{127.0}$$
2. Round each float into a signed 8-bit integer:
   $$q_{x, i} = \text{round}\left(\frac{x_i}{s_x}\right)$$
3. The dot product between weight block $q_w$ and activation block $q_x$ accumulates as 32-bit integers:
   $$S = \sum_{i=0}^{31} q_{w, i} \cdot q_{x, i}$$
4. The integer sum is scaled once per block:
   $$\text{result} = S \cdot (s_w \cdot s_x)$$

#### Numerical Error Bounds
Activation quantization introduces a maximum rounding error bounded by:
$$|x_i - \tilde{x}_i| \le \frac{s_x}{2}$$

This bounded error is comparable to the precision limits of 4-bit weights. If exact floating-point accumulation is required, integer quantization can be disabled at runtime using `State::set_int8(false)`.

### Strategy 4: SIMD Hardware Acceleration

Ferrite uses explicit architecture-specific intrinsics without external assembly dependencies:

- **x86_64 AVX2**: Employs `_mm256_maddubs_epi16` and `_mm256_madd_epi16` for horizontal vector accumulation. Because integer addition is associative, the vector summation produces identical results to the scalar integer path.
- **x86_64 AVX-512**: Implements 512-bit vector registers for supported server platforms.
- **ARM NEON**: Employs `vdotq_s32` and `vmlal_s16` instructions for AArch64 systems.

## 3. Kernel Selection Policy

Ferrite selects the kernel for each format based on empirical hardware measurements rather than uniform rules:

| Weight Format | Selected Kernel | Reason |
| :--- | :--- | :--- |
| **Q4_K** | Integer Dot (iDot) + AVX2 | Vectorized nibble math delivers 3.9x to 4.0x speedups over reference. |
| **Q8_0** | Integer Dot (iDot) | Native byte products deliver 3.0x to 4.9x speedups over reference. |
| **Q6_K** | Fused Dequantize-and-Dot | Register-level unpack avoids memory round-trips (2.4x to 3.3x speedup). |
| **Q5_K** | Fused Dequantize-and-Dot | Eliminates temporary buffer overhead (2.0x to 2.5x speedup). |
| **Q4_0** | Fused Dequantize-and-Dot | Fast f16 scale multiplication beats the overhead of activation quantization. |

## 4. Key-Value (KV) Cache Quantization

In long-context inference, the Key-Value cache can consume gigabytes of RAM. Ferrite supports dynamic KV cache quantization:

| KV Format | Storage per Element | Memory Savings vs F32 | Trade-off |
| :--- | :--- | :--- | :--- |
| **F32** | 4.0 bytes | Baseline (1.0x) | Full mathematical precision. |
| **Q8_0** | 1.0625 bytes | ~3.76x reduction | Negligible perplexity degradation. |
| **Q4_0** | 0.5625 bytes | ~7.11x reduction | Minor loss in precision; suited for large context windows. |

Configure KV cache quantization using the `--kv-cache <f32|q8_0|q4_0>` CLI argument or via the `KvCacheType` enum in library code.
