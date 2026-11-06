# Ferrite CLI Reference

The `ferrite` command-line utility provides tools for model inspection, text generation, HTTP serving, quantization, model merging, evaluation, tokenization, and multi-modal execution.

## Syntax Overview

```bash
ferrite <command> [arguments] [options]
```

## Model Resolution and Discovery

When `<model>` is specified, Ferrite attempts to resolve it using the following rules:

1. **Direct Path**: If `<model>` points to an existing file on disk (or starts with `./`, `/`, or `..\`), it is opened directly.
2. **Universal Local Model Stores**: Ferrite searches for matching `.gguf` files across local model directories:
   - `~/.lmstudio/models` and `~/.cache/lm-studio/models` (LM Studio)
   - `~/.cache/huggingface/hub` and `$HF_HOME/hub` (Hugging Face Hub cache)
   - `~/jan/models` and `~/.config/Jan/data/models` (Jan.ai)
   - `~/.local/share/nomic.ai/GPT4All` (GPT4All)
   - `~/.local/share/ferrite/models` and `~/models` (Local model stores)
3. **Environment Override**: Setting `FERRITE_MODELS_DIR` overrides the search root for downloaded models.
4. **Fuzzy Substring Match**: Any unique substring from `ferrite list` can be used. If more than one model matches, Ferrite rejects the query to prevent running the wrong model.

---

## Commands

### `ferrite doctor [model]`

Profiles host hardware (architecture, logical CPUs, total/available RAM, active SIMD extensions, single-core Q4_K streaming bandwidth), prints model tier readiness recommendations, and lists discovered local GGUF models.

```bash
ferrite doctor
```

---

### `ferrite bench [--threads <n>] [--model <model>] [--json]`

Runs 1-thread and multi-thread GEMV benchmarks across `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `Q4_0` alongside end-to-end batched prefill (`pp64`) and token decode (`tg32`).

```bash
ferrite bench --threads 4
```

---

### `ferrite chat <model> [--system <prompt>]`

Starts an interactive multi-turn terminal chat REPL with persistent KV cache across turns and slash commands (`/clear`, `/system <prompt>`, `/stats`, `/exit`).

```bash
ferrite chat llama-3.2 --system "You are a concise Rust expert."
```

---

### `ferrite list`

Lists all `.gguf` models found across local model discovery directories or the directory specified by `FERRITE_MODELS_DIR`.

```bash
ferrite list
```

Example output:
```text
LM Studio models (~/.lmstudio/models):
  publisher/repo/model-q4_k_m.gguf (4.2 GB)
```

---

### `ferrite where`

Prints the directory paths being searched for models.

```bash
ferrite where
```

Example output:
```text
Searching:
  /home/user/.lmstudio/models
```

---

### `ferrite info <model|path> [-t]`

Displays container metadata, architecture hyperparameters, quantization breakdown, and vocabulary information.

- `-t`: Dumps the full tensor table including names, shapes, quantization types, and wire sizes.

```bash
ferrite info tiny.gguf -t
```

Example output:
```text
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

  token_embd.weight                        32x19              F32            2.4 KB
  blk.0.attn_norm.weight                   32                 F32             128 B
  ...
```

---

### `ferrite tokenize <model> <text>`

Tokenizes an input string using the model's vocabulary and displays token IDs, reconstructed byte values, and merge count.

```bash
ferrite tokenize tiny.gguf "hello world"
```

Example output:
```text
Bpe vocab, 9 merges, 2 tokens
       13  "hello"
       18  " world"
```

---

### `ferrite run <model> <prompt> [options]`

Executes autoregressive text generation from an input prompt and streams tokens to standard output.

```bash
ferrite run llama-3.2 "Why is the sky blue?" -n 128 --temp 0.7
```

Options:
- `-n <count>`: Number of tokens to generate (default: `128`).
- `--temp <t>`: Sampling temperature. `0` selects greedy sampling (default: `0`).
- `--top-k <k>`: Keep only the $k$ most probable tokens (default: `40`).
- `--top-p <p>`: Nucleus sampling probability threshold (default: `0.95`).
- `--min-p <p>`: Minimum token probability scaled relative to the top token (default: `0.0`).
- `--tfs <z>`: Tail Free Sampling parameter in `(0, 1]` (default: `1.0`, disabled).
- `--seed <s>`: Pseudorandom number generator seed (default: `1`).
- `--ctx <n>`: Maximum context window in tokens (default: `2048`).
- `--threads <n>`: Number of worker threads for matrix multiplication (default: number of logical CPU cores).
- `--chunk-size <n>`: Token chunk size during batched prompt prefill (default: `512`).
- `--kv-cache <t>`: Precision of the KV cache: `f32`, `q8_0`, or `q4_0` (default: `f32`).
- `--repeat-penalty <p>`: Multiplicative repetition penalty (default: `1.0`).
- `--frequency-penalty <p>`: Additive frequency penalty (default: `0.0`).
- `--presence-penalty <p>`: Additive presence penalty (default: `0.0`).
- `--repeat-last-n <n>`: Window of recent tokens considered for penalties (default: `64`).
- `--draft <model>`: Draft model path for continuous speculative decoding.
- `--spec-k <k>`: Speculative token lookahead depth (default: `3`).
- `--spec-tree <b>`: Branching factor for speculative tree decoding (default: `0`, disabled).
- `--mtp`: Enables DeepSeek Multi-Token Prediction speculative acceleration.
- `--lora <adapter>`: Dynamically loads and fuses a GGUF LoRA adapter at runtime.

---

### `ferrite serve <model> [options]`

Starts an HTTP server providing OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings` endpoints with Server-Sent Events (SSE) streaming, plus an embedded single-binary WebUI playground at `/ui` (and `/` when opened in a web browser).

```bash
ferrite serve llama-3.2 --host 127.0.0.1 --port 8080 --threads 8
```

Server Options:
- `--host <host>`: Network interface to bind (default: `127.0.0.1`).
- `--port <port>`: Port to listen on (default: `8080`).
- All generation options from `ferrite run` (e.g. `--ctx`, `--kv-cache`, `--threads`) are accepted.

---

### `ferrite quantize <src.gguf> <dst.gguf> <type>`

Re-quantizes a GGUF model file into a target quantization format without external tooling.

Supported target formats: `Q8_0`, `Q4_0`, `Q4_K`, `IQ4_NL`, `IQ4_XS`, `TQ1_0`, `TQ2_0`.

```bash
ferrite quantize base-f32.gguf base-iq4_nl.gguf IQ4_NL
```

---

### `ferrite merge <model_a.gguf> <model_b.gguf> <out.gguf> --method <method> [--weight <w>]`

Merges two GGUF models with identical architectures and dimensions into a combined model.

Methods:
- `slerp`: Spherical Linear Interpolation along weight geodesics.
- `ties`: Task-Informed Task Arithmetic with top-k pruning and sign consensus.
- `dare`: Drop And REscale random mask merging.

```bash
ferrite merge model-a.gguf model-b.gguf merged.gguf --method slerp --weight 0.5
```

---

### `ferrite perplexity <model> <file>`

Evaluates cross-entropy loss and text perplexity over a test text file using chunked prefill.

```bash
ferrite perplexity llama-3.2 test_corpus.txt
```

---

### `ferrite convert <src.safetensors> <dst.gguf>`

Converts raw SafeTensors model files into valid GGUF container format.

```bash
ferrite convert model.safetensors model.gguf
```

---

### `ferrite lora merge <base.gguf> <out.gguf> <adapter.gguf...>`

Fuses one or more GGUF LoRA adapters directly into a base GGUF model file, writing a new consolidated GGUF file.

```bash
ferrite lora merge base.gguf fine-tuned.gguf adapter1.gguf adapter2.gguf
```

---

### `ferrite train-vocab <corpus.txt> <out_vocab.gguf> [--vocab-size <n>]`

Trains a Byte-Pair Encoding (BPE) vocabulary from an input text corpus and serializes it to a GGUF vocabulary container.

```bash
ferrite train-vocab corpus.txt custom_vocab.gguf --vocab-size 32000
```

---

### `ferrite embed <model> <text...> [--pooling <pool>] [--compare]`

Computes high-dimensional dense embedding vectors for input strings and optionally computes pairwise cosine similarity.

Options:
- `--pooling <pool>`: Pooling strategy: `mean`, `cls`, or `last` (default: `mean`).
- `--compare`: Prints pairwise cosine similarity matrix across inputs.

```bash
ferrite embed llama-3.2 "Text sentence A" "Text sentence B" --compare
```

---

### `ferrite vision <model> --image <image> --mmproj <mmproj.gguf> [prompt]`

Runs multimodal vision-language inference with pure-Rust image decoding (PPM and uncompressed BMP) and ViT projection.

```bash
ferrite vision qwen2-vl --image test.bmp --mmproj mmproj.gguf "Describe this image"
```

---

### `ferrite distributed <worker|master> [options]`

Coordinates distributed tensor-parallel inference across multiple network endpoints.

```bash
# On worker node
ferrite distributed worker --port 9000

# On master node
ferrite distributed master --workers 192.168.1.10:9000,192.168.1.11:9000 --model llama-3.2
```
