# Ferrite Supported Formats and Architectures

This document provides a technical support matrix distinguishing formats recognized by container parsers from those actively evaluated in inference pipelines.

## 1. Container Formats

| Container Version | Parser Status | Inference Status | Notes |
| :--- | :--- | :--- | :--- |
| **GGUF v3** | Implemented, Tested | Implemented, Tested | Modern GGUF standard with 64-bit alignment and metadata tables. |
| **GGUF v2** | Implemented, Tested | Implemented, Tested | Earlier GGUF revision; fully supported. |
| **GGUF v1** | Unsupported | Unsupported | Deprecated format; rejected at header parsing. |
| **SafeTensors** | Implemented, Tested | Via Conversion | Converted to GGUF format via `ferrite convert`. |

## 2. Metadata Types

The GGUF container specification defines 13 metadata value types. Ferrite parses all 13 types in `src/gguf.rs`:

| Type Tag | Type Identifier | Status | Notes |
| :--- | :--- | :--- | :--- |
| 0 | `UINT8` | Implemented, Tested | Parsed and checked against remaining buffer bounds. |
| 1 | `INT8` | Implemented, Tested | Parsed and validated. |
| 2 | `UINT16` | Implemented, Tested | Little-endian decoded. |
| 3 | `INT16` | Implemented, Tested | Little-endian decoded. |
| 4 | `UINT32` | Implemented, Tested | Used for dimensions, token counts, layer counts. |
| 5 | `INT32` | Implemented, Tested | Signed 32-bit values. |
| 6 | `FLOAT32` | Implemented, Tested | Standard IEEE 754 single precision. |
| 7 | `BOOL` | Implemented, Tested | Single-byte boolean flags. |
| 8 | `STRING` | Implemented, Tested | Length-prefixed UTF-8 with bounded length checks. |
| 9 | `ARRAY` | Implemented, Tested | Homogeneous arrays. Nested arrays are rejected. |
| 10 | `UINT64` | Implemented, Tested | Used for tensor offsets and large file pointers. |
| 11 | `INT64` | Implemented, Tested | Signed 64-bit values. |
| 12 | `FLOAT64` | Implemented, Tested | IEEE 754 double precision. |

## 3. Tensor Quantization Formats

Ferrite distinguishes between tensor formats that are recognized in the file table versus those executable during forward passes:

| Format | Block Size | Layout Recognized | Dequantization | Fused Dot | Integer Dot (iDot) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F32** | 1 | Yes | Exact copy | Exact dot | Supported | Implemented, Tested |
| **F16** | 1 | Yes | IEEE 754 half | Supported | Supported | Implemented, Tested |
| **BF16** | 1 | Yes | Bit-shift unpack | Supported | Supported | Implemented, Tested |
| **Q4_0** | 32 | Yes | Fused scale/min | Implemented | Supported | Implemented, Tested |
| **Q4_1** | 32 | Yes | Fused scale/min | Implemented | Unsupported | Implemented, Tested |
| **Q5_0** | 32 | Yes | Fused scale/min | Implemented | Unsupported | Implemented, Tested |
| **Q5_1** | 32 | Yes | Fused scale/min | Implemented | Unsupported | Implemented, Tested |
| **Q8_0** | 32 | Yes | Fused scale/min | Implemented | Implemented (Selected) | Implemented, Tested |
| **Q8_1** | 32 | Yes | Fused scale/min | Implemented | Unsupported | Implemented, Tested |
| **Q2_K** | 256 | Yes | Super-block scale | Implemented | Unsupported | Implemented, Tested |
| **Q3_K** | 256 | Yes | Super-block scale | Implemented | Unsupported | Implemented, Tested |
| **Q4_K** | 256 | Yes | Super-block scale | Implemented | Implemented (Selected) | Implemented, Tested |
| **Q5_K** | 256 | Yes | Super-block scale | Implemented (Selected) | Unsupported | Implemented, Tested |
| **Q6_K** | 256 | Yes | Super-block scale | Implemented (Selected) | Unsupported | Implemented, Tested |
| **Q8_K** | 256 | Yes | Super-block scale | Implemented | Unsupported | Implemented, Tested |
| **F8_E4M3** | 32 | Yes | Block-wise FP8 | Implemented (Selected) | Supported | Implemented, Tested |
| **IQ1_S..IQ4_NL** | 32/256 | Yes (Named in info) | Unsupported | Unsupported | Unsupported | Unsupported |

## 4. Tokenizer Implementations

| Tokenizer Format | Model Architectures | Fallback Support | Status |
| :--- | :--- | :--- | :--- |
| **Byte-Level BPE** | Llama 3, Qwen 2/2.5, DeepSeek, Phi | 256-byte unicode mapping | Implemented, Tested |
| **SentencePiece (SPM)** | Llama 2, Mistral, Gemma | `<0xXX>` byte tokens | Implemented, Tested |
| **WordPiece** | BERT derivatives | None | Unsupported |

## 5. Model Architecture Support Matrix

Ferrite accepts 13 architecture identifiers in GGUF metadata (`general.architecture`):

| Architecture ID | Representative Models | Key Architectural Primitives | Status |
| :--- | :--- | :--- | :--- |
| `llama` | Llama 2, Llama 3, 3.1, 3.2 | RMSNorm, RoPE, Grouped-Query Attention, SwiGLU | Implemented, Tested |
| `qwen`, `qwen2` | Qwen 2, Qwen 2.5 | QKV attention biases, custom RoPE frequencies | Implemented, Tested |
| `qwen2moe`, `qwen2_moe` | Qwen2-57B-A14B | Top-K routing, shared expert isolation | Implemented, Tested |
| `qwen2vl`, `qwen2_vl` | Qwen2-VL | Multimodal RoPE (M-RoPE), 3D spatio-temporal grids | Implemented, Tested |
| `gemma`, `gemma2` | Gemma, Gemma 2 (2B, 9B, 27B) | Embedding scaling, attention and logit soft-capping, 4-stage RMSNorm | Implemented, Tested |
| `mixtral` | Mixtral 8x7B, 8x22B | Sparse MoE with Top-2 softmax gating | Implemented, Tested |
| `deepseek`, `deepseek2`, `deepseek3` | DeepSeek-V2, DeepSeek-V3 | Multi-Head Latent Attention (MLA), Multi-Token Prediction (MTP), FP8 | Implemented, Tested |
| Other IDs | Falcon, BLOOM, StableLM | Non-standard architectures | Unsupported |

## 6. CPU Hardware Acceleration Kernels

| Architecture | Vector Extension | Instruction Set | Kernel Status |
| :--- | :--- | :--- | :--- |
| **x86_64** | Scalar Portable | Standard x86_64 registers | Implemented, Tested |
| **x86_64** | AVX2 + FMA | `_mm256_maddubs_epi16`, `_mm256_madd_epi16` | Implemented, Tested |
| **x86_64** | AVX-512 | `_mm512_dpbusd_epi32` | Implemented, Untested on non-AVX-512 host |
| **aarch64** | ARM NEON | `vdotq_s32`, `vmlal_s16` | Implemented, Tested in cross-compilation CI |

## 7. Operating System Compatibility

| Operating System | Memory Mapping Method | Threading Model | CI Verification |
| :--- | :--- | :--- | :--- |
| **Linux (glibc / musl)** | Direct `mmap` syscall wrapper | `std::thread` worker pool | Verified |
| **macOS (Darwin)** | Direct `mmap` syscall wrapper | `std::thread` worker pool | Verified |
| **Windows** | `CreateFileMappingW` / `MapViewOfFile` | `std::thread` worker pool | Verified |
