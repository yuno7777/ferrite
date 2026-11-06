# Changelog

All notable changes to Ferrite are documented in this file. This project has not cut an official release version yet, so all entries below are unreleased.

## Unreleased

### Added

- **End-to-End Verification and Release Reconciliation (Phase 77)**: Comprehensive multi-stage inference verification suite (`tests/e2e.rs`) testing end-to-end model execution, PagedAttention lifecycle, prefix cache persistence, multimodal audio/vision pipelines, and quantized FP8 GEMM.
- **Model Watermarking and Cryptographic Signatures (Phase 76)**: Green-red list token watermarking (Kirchenbauer et al.) and cryptographic PRF token signature injection with statistical z-score and p-value verification detector (`src/watermark.rs`, `src/sample.rs`).
- **Vectorized FP8 (E4M3 / E5M2) GEMM with Auto-Tuning (Phase 75)**: High-performance 8-lane fused GEMM kernels for OCP FP8 E4M3 and IEEE 754 FP8 E5M2 with hardware feature auto-tuning for modern CPUs (`src/quant/fp8.rs`, `src/ops.rs`).
- **Multi-Modal Audio Spectrogram Projection (Phase 74)**: Pure-Rust audio waveform decoding (WAV PCM), Short-Time Fourier Transform (STFT), triangular Log-Mel filterbank computation, audio patch grouping, and linear/MLP projection into language model activation space (`src/audio.rs`).
- **Radix Tree Prefix Cache Persistence and Page Swapping (Phase 73)**: Binary disk persistence format for radix tree prefix caches and memory-mapped page swapping manager for paging cold KV blocks to disk (`src/prefix.rs`).
- **Adaptive Speculative Thresholding and Temperature Calibration (Phase 72)**: Dynamic acceptance threshold tracking and draft temperature calibration for optimal speculative throughput across varying entropy regimes (`src/speculative.rs`).
- **Direct GGUF Model Pruning Engine (Phase 71)**: Structured channel and head L2-norm pruning alongside unstructured magnitude pruning with direct in-place tensor modification (`src/prune.rs`).
- **SmoothQuant W8A8 Activation Migration (Phase 70)**: Offline activation-to-weight scale migration for outlier smoothing, enabling efficient INT8 matrix-vector multiplication (`src/quant/smooth.rs`).
- **DeepSeek-V3 DualPipe Pipeline Parallel Simulation (Phase 69)**: Overlapped bidirectional pipeline parallel execution schedule with forward and backward chunk interleaving (`src/dualpipe.rs`).
- **Micro-Batching Prefill Engine (Phase 68)**: Dynamic sequence token packing with zero-padding removal for unequal-length prompt batching (`src/scheduler.rs`).
- **Dynamic Context Truncation and Rolling Sink Cache (Phase 67)**: Attention sink preservation with adaptive distribution entropy bounds for infinite context rolling eviction (`src/kv.rs`).
- **Multi-Tenant Inference Engine (Phase 66)**: Multi-tenant token bucket rate limiting, quota tracking, and priority admission queue (`src/tenant.rs`, `src/server.rs`).
- **Sparse Attention Masking Engine (Phase 65)**: Local sliding-window, strided, and BigBird random block-sparse attention mask generation with fused attention compute (`src/sparse.rs`).
- **FlashDecoding Attention Partitioning (Phase 64)**: Logarithmic tree-reduction attention kernel partitioning long context KV caches across parallel thread splits (`src/ops.rs`).
- **Asynchronous Chunked Prefill Compute (Phase 63)**: Background worker thread pool for incremental chunked prefill evaluation with event streaming (`src/prefill.rs`).
- **Zero-Copy In-Place GGUF Model Editing (Phase 62)**: Mutable memory mapping tensor editor for in-place weight adjustments, metadata mutations, and bias updates (`src/edit.rs`, `src/map.rs`).
- **Outlier-Aware Mixed-Precision Quantization (Phase 61)**: Mixed-precision W8A8 and W4A16 quantization with FP16 high-magnitude outlier channel preservation (`src/quant/outlier.rs`).
- **Speculative Draft KV-Cache Recycling (Phase 60)**: Rolling draft KV-cache preservation across successive verification rounds, eliminating redundant prompt evaluation (`src/speculative.rs`, `src/kv.rs`).
- **Native GGUF Sharding and Multi-File Partitioning (Phase 59)**: Command-line tool and library API for partitioning large GGUF containers into distributed shards (`src/split.rs`).
- **JSON Schema to CFG Automaton Compiler (Phase 58)**: Direct JSON schema parsing and compiler generating byte-level context-free grammar pushdown automata (`src/grammar.rs`).
- **PagedAttention v2 with Hierarchical Page Tables (Phase 57)**: Multi-level page tables with virtual block indirection and zero-copy fork isolation (`src/paged.rs`).
- **Fused INT8 GEMM for Batched Prefill (Phase 56)**: High-performance fused INT8 matrix-matrix multiplication for prompt batching (`src/ops.rs`).
- **Speculative Medusa / Eagle Multi-Head Draft Engine (Phase 55)**: Parallel multi-head prediction engine proposing speculative token candidates from a single model forward pass (`src/speculative.rs`).
- **Dynamic RoPE Frequency Cache and Su-YaRN Context Extension (Phase 54)**: Precomputed rotary inverse frequency cache with Su-YaRN context extension (`src/ops.rs`).
- **Continuous Streaming Detokenizer (Phase 53)**: Incremental UTF-8 byte boundary preservation across multi-token splits with special token filtering (`src/tokenizer/stream.rs`).
- **Tiled Transposed Attention Kernel (Phase 52)**: Dynamic cache-friendly tiled online softmax attention (`src/ops.rs`).
- **Continuous Dual-Partition Worker Dispatch (Phase 51)**: Dual-worker thread pool dispatch for parallel matrix operations (`src/tp.rs`).
- **AWQ and GPTQ Quantization Support (Phase 50)**: INT4 group dequantization and fused dot products for Activation-Aware Weight Quantization (`src/quant/awq.rs`).
- **DeepSeek Latent Cache Direct Attention (Phase 49)**: Compressed latent cache attention without intermediate KV expansion (`src/ops.rs`, `src/kv.rs`).
- **DeepSeek-V3 Multi-Depth Tree Speculation (Phase 48)**: Multi-depth speculative draft verification for MTP models (`src/speculative.rs`).
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
