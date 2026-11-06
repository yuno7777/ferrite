# Changelog

All notable changes to Ferrite are documented in this file. This project has not cut an official release version yet, so all entries below are unreleased.

## Unreleased

### Added

- **FP8 Block-Wise Quantization (Phase 47)**: Implemented `F8_E4M3` block-wise quantization parsing, dequantization, and fused dot products with dedicated testing.
- **Intra-Node Tensor Parallelism (Phase 46)**: Thread pool based column-parallel and row-parallel GEMM partitioning for intra-node CPU scaling.
- **Auxiliary-Loss-Free MoE Routing (Phase 45)**: Implemented DeepSeek-V3 and Qwen auxiliary-loss-free MoE routing with sigmoid expert scoring and bias adjustments.
- **Multi-Token Prediction Speculative Engine (Phase 44)**: DeepSeek-V3 Multi-Token Prediction (MTP) module integration with speculative candidate rollback and verification.
- **Multimodal RoPE for Qwen2-VL (Phase 43)**: Multi-axis 3D spatio-temporal rotary position embeddings (M-RoPE) supporting vision-language token layouts.
- **Multi-Head Latent Attention (Phase 42)**: DeepSeek-V2 and DeepSeek-V3 Multi-Head Latent Attention (MLA) low-rank KV projections and decoupled RoPE.
- **GGUF Model Merging (Phase 36)**: Implemented SLERP, TIES, and DARE parameter merging algorithms via `ferrite merge`.
- **Mixtral Sparse MoE Architecture (Phase 34)**: Top-2 softmax expert gating and feed-forward routing for Mixtral 8x7B models.
- **AVX-512 and ARM NEON Vector Kernels (Phases 32, 33)**: 512-bit vector intrinsics for modern server x86_64 CPUs and 128-bit NEON intrinsics for AArch64.
- **Multi-Adapter LoRA Merging (Phase 31)**: Multi-adapter weighted rank fusion into base GGUF model files.
- **Advanced Sampling Strategies (Phase 30)**: Added Min-P, Tail Free Sampling (TFS), and frequency, presence, and repetition penalties.
- **Quantized KV Caches (Phase 29)**: Dynamic `Q8_0` and `Q4_0` block compression for Key-Value caches to reduce memory footprint.
- **Multimodal Vision Engine (Phase 28)**: Zero-dependency PPM and uncompressed BMP image loading, bilinear resampling, and ViT projection for vision-language models.
- **Chat Templating and Tool Use (Phases 26, 27)**: Chat template rendering engine and JSON schema-guided function calling.
- **Distributed Inference (Phase 25)**: Network-based distributed tensor-parallel inference via TCP worker nodes.
- **In-Crate Vocabulary Training (Phase 24)**: BPE vocabulary training from raw text corpora with direct GGUF vocabulary container export.
- **Embeddings Engine (Phase 23)**: Mean, CLS, and last-token pooling with cosine similarity calculation via `ferrite embed`.
- **Speculative Tree Decoding (Phase 22)**: Multi-candidate tree-based draft token verification.
- **Perplexity Evaluation (Phase 21)**: Text perplexity and cross-entropy loss computation with chunked processing via `ferrite perplexity`.
- **SafeTensors to GGUF Converter (Phase 20)**: Native header parser and GGUF serialization tool via `ferrite convert`.
- **Model Re-Quantization Engine (Phase 19)**: Offline conversion between F32 and quantized representations via `ferrite quantize`.
- **Radix Tree Prefix Caching (Phase 18)**: LRU radix cache for sharing prompt KV representations across multi-turn sessions.
- **Chunked Prefill Engine (Phase 17)**: Stateful chunked prompt evaluation for bounded memory consumption.
- **Dynamic LoRA Injection (Phase 16)**: Runtime low-rank adaptation without rewriting base model weights.
- **OpenAI-Compatible HTTP Server (Phase 15)**: Server daemon providing `/v1/chat/completions` and `/v1/completions` with SSE streaming.
- **Grammar-Constrained Decoding (Phase 14)**: Formal context-free grammar and JSON Schema logit masking.
- **Continuous Batching Scheduler (Phase 13)**: Multi-sequence batch scheduler with prefill and decode multiplexing.
- **PagedAttention Engine (Phase 12)**: Non-contiguous physical block memory management for KV caches.
- **Continuous Speculative Decoding (Phase 11)**: Draft-target speculative verification engine.
- **Gemma and Gemma 2 Architectures (Phase 10)**: Support for embedding scaling, decoupled head dimensions, logit soft-capping, and 4-stage RMSNorms.
- **Qwen 2 and Qwen 2.5 Architectures (Phase 9)**: Support for QKV attention biases and custom RoPE frequency configurations.
- **Batched Prefill Engine (Phase 8)**: Compute-bound GEMM for prompt ingestion, reusing weights across all prompt tokens in a layer.
- **AVX2 Vectorized Integer Kernels**: Vectorized `_mm256_maddubs_epi16` and `_mm256_madd_epi16` acceleration for `Q4_K` and `Q8_0`.
- **Integer Dot Products (iDot)**: Dynamic INT8 activation quantization with 32-bit integer accumulation.
- **Zero-Dependency Memory Mapping**: Inline platform syscall declarations for Linux, macOS, and Windows.
- **Quantization Support**: Formats `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`, `F32`, `F16`, `BF16`.
- **Tokenizers**: Byte-level BPE and SentencePiece reconstructed from GGUF vocabularies with byte fallback.
- **Transformer Forward Pass**: RMSNorm, RoPE, Grouped-Query Attention, and SwiGLU.
- **Sampling**: Seeded greedy, temperature, top-k, and top-p sampling.
- **LM Studio Discovery**: Automatic resolution of downloaded models without copying or converting files.
