use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use ferrite::gguf::Gguf;
use ferrite::lmstudio;
use ferrite::model::{self, Model, State};
use ferrite::sample::Sampler;
use ferrite::Tokenizer;

const USAGE: &str = "\
ferrite: pure-Rust GGUF inference

usage:
  ferrite list                      models discovered across local stores
  ferrite where                     directories being searched
  ferrite doctor [model]            hardware profiler, SIMD check, & tuning advice
  ferrite bench [--threads <n>]     GEMV, batched GEMM, & synthetic model benchmarks
  ferrite info <model|path> [-t]    header, hyperparameters, quant mix
                                    -t also dumps the tensor table
  ferrite tokenize <model> <text>   encode text, show ids and pieces
  ferrite run <model> <prompt>      generate, streaming to stdout
  ferrite chat <model>              interactive multi-turn terminal chat TUI
  ferrite serve <model>             serve HTTP/SSE API daemon
  ferrite quantize <src.gguf> <dst.gguf> <Q8_0|Q4_0|Q4_K>
  ferrite merge <model_a.gguf> <model_b.gguf> <out.gguf> --method <slerp|ties|dare> [--weight <0..1>]
  ferrite split <src.gguf> <out_prefix> [--shards <n>] [--max-bytes <bytes>]
  ferrite prune <model.gguf> --sparsity <0..1> [--tensor <name>]
  ferrite edit <model.gguf> <info|scale> <tensor> [<factor>]
  ferrite watermark <detect|generate> <model> <text_or_prompt> [--key <u64>]
  ferrite perplexity <model> <file> evaluate text perplexity and loss
  ferrite convert <src> <dst.gguf>  convert SafeTensors to GGUF format
  ferrite lora merge <base.gguf> <out.gguf> <adapter.gguf...>
  ferrite train-vocab <corpus.txt> <out_vocab.gguf> [--vocab-size <n>]
  ferrite embed <model> <text...>   generate vector embeddings & similarity
  ferrite vision <model> --image <image> --mmproj <mmproj.gguf> [prompt]

run/chat/serve options:
  -n <count>      tokens to generate (default 128)
  --temp <t>      sampling temperature, 0 for greedy (default 0)
  --top-k <k>     keep only the k most likely tokens (default 40)
  --top-p <p>     nucleus threshold (default 0.95)
  --seed <s>      sampling seed (default 1)
  --ctx <n>       KV cache size in tokens (default 2048)
  --threads <n>   workers per matvec (default: machine parallelism)
  --host <host>   host interface to bind (default: 127.0.0.1)
  --port <port>   port to bind (default: 8080)
  --system <s>    system prompt for `ferrite chat`
  --json          enforce valid RFC 8259 JSON output via grammar automaton
  --json-schema <file_or_json> enforce JSON Schema conformance
  --regex <pat>   enforce regular expression grammar on output
  --draft <model> draft model for speculative decoding acceleration
  --spec-k <k>    speculative candidate count / depth (default 3)
  --spec-tree <b> speculative candidate branch count (default 0, disabled)
  --lookahead <k> n-gram self-speculative lookahead depth (no draft model needed)
  --mtp           enable DeepSeek Multi-Token Prediction (MTP) speculation
  --lora <adapter> apply GGUF LoRA adapter dynamically
  --chunk-size <n> chunk size for prompt prefill (default 512)
  --kv-cache <t>  KV cache quantization: f32, q8_0, q4_0 (default f32)
  --min-p <p>     minimum probability scale relative to top token (default 0.0)
  --tfs <z>       tail free sampling z parameter in (0, 1] (default 1.0)
  --repeat-penalty <p> multiplicative repetition penalty (default 1.0)
  --frequency-penalty <p> additive frequency penalty (default 0.0)
  --presence-penalty <p> additive presence penalty (default 0.0)
  --repeat-last-n <n> window of recent tokens for penalties (default 64)

<model> is a path to a .gguf, or any substring of an id from `ferrite list`.
Override the search root with FERRITE_MODELS_DIR.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("list") => cmd_list(),
        Some("where") => cmd_where(),
        Some("doctor") => cmd_doctor(&args[1..]),
        Some("bench") => cmd_bench(&args[1..]),
        Some("info") => cmd_info(&args[1..]),
        Some("tokenize") => cmd_tokenize(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("chat") => cmd_chat(&args[1..]),
        Some("serve") => cmd_serve(&args[1..]),
        Some("quantize") => cmd_quantize(&args[1..]),
        Some("merge") => cmd_merge(&args[1..]),
        Some("split") => cmd_split(&args[1..]),
        Some("prune") => cmd_prune(&args[1..]),
        Some("edit") => cmd_edit(&args[1..]),
        Some("watermark") => cmd_watermark(&args[1..]),
        Some("perplexity") => cmd_perplexity(&args[1..]),
        Some("convert") => cmd_convert(&args[1..]),
        Some("lora") => cmd_lora(&args[1..]),
        Some("train-vocab") => cmd_train_vocab(&args[1..]),
        Some("embed") => cmd_embed(&args[1..]),
        Some("distributed") => cmd_distributed(&args[1..]),
        Some("vision") => cmd_vision(&args[1..]),
        Some("-h") | Some("--help") | Some("help") | None => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Some(other) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unknown command {other:?}. Try `ferrite help`."),
        )),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ferrite: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_train_vocab(args: &[String]) -> std::io::Result<()> {
    if args.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: ferrite train-vocab <corpus.txt> <out_vocab.gguf> [--vocab-size <n>]",
        ));
    }
    let corpus_path = std::path::Path::new(&args[0]);
    let out_path = std::path::Path::new(&args[1]);
    let mut vocab_size = 1000;
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--vocab-size" && i + 1 < args.len() {
            vocab_size = args[i + 1].parse().map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid vocab-size: {e}"),
                )
            })?;
            i += 2;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown option: {}", args[i]),
            ));
        }
    }
    let config = ferrite::TrainConfig {
        vocab_size,
        ..Default::default()
    };
    println!(
        "Training BPE vocabulary from {} (target size: {})...",
        corpus_path.display(),
        vocab_size
    );
    let start = Instant::now();
    let vocab = ferrite::train_vocab_file(corpus_path, out_path, &config)?;
    println!(
        "Trained {} tokens ({} merges) in {:.2?}. Saved to {}",
        vocab.tokens.len(),
        vocab.merges.len(),
        start.elapsed(),
        out_path.display()
    );
    Ok(())
}

fn cmd_embed(args: &[String]) -> std::io::Result<()> {
    if args.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: ferrite embed <model> <text...> [--pooling <mean|cls|last>] [--compare]",
        ));
    }
    let path = lmstudio::resolve(&args[0])?;
    let mut texts = Vec::new();
    let mut pooling = ferrite::Pooling::Mean;
    let mut compare = false;
    let mut threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--pooling" && i + 1 < args.len() {
            pooling = args[i + 1].parse().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("{e}"))
            })?;
            i += 2;
        } else if args[i] == "--compare" {
            compare = true;
            i += 1;
        } else if args[i] == "--threads" && i + 1 < args.len() {
            threads = args[i + 1].parse().map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid threads: {e}"),
                )
            })?;
            i += 2;
        } else if !args[i].starts_with("--") {
            texts.push(args[i].clone());
            i += 1;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown option {}", args[i]),
            ));
        }
    }
    if texts.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "no input text provided to embed",
        ));
    }

    let gguf = Gguf::open(&path)?;
    let model = Model::load(&gguf)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let engine = ferrite::EmbedEngine::new(&model, pooling, true);

    let mut state = State::new(&model.config);
    state.threads = threads;

    let mut vectors = Vec::new();
    for (idx, text) in texts.iter().enumerate() {
        let tokens = tokenizer.encode(text, true);
        let vector = engine.embed(&mut state, &tokens)?;
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!(
            "[{idx}] \"{text}\" -> dim {}, L2 norm: {norm:.4}",
            vector.len()
        );
        vectors.push(vector);
    }

    if compare && vectors.len() >= 2 {
        println!("\nCosine Similarity Matrix:");
        for i in 0..vectors.len() {
            for j in 0..vectors.len() {
                let sim = ferrite::cosine_similarity(&vectors[i], &vectors[j]);
                print!("{sim:8.4} ");
            }
            println!("  [{i}] \"{}\"", texts[i]);
        }
    }

    Ok(())
}

fn cmd_distributed(args: &[String]) -> std::io::Result<()> {
    if args.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: ferrite distributed <worker|driver> ...",
        ));
    }
    match args[0].as_str() {
        "worker" => {
            if args.len() < 2 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "usage: ferrite distributed worker <model> [--listen <host:port>] [--layers <start..end>]",
                ));
            }
            let path = lmstudio::resolve(&args[1])?;
            let gguf = Gguf::open(&path)?;
            let model = Model::load(&gguf)?;
            let listen_addr: String = flag(&args[2..], "--listen", "127.0.0.1:9090".to_string())?;
            let layers_str: String = flag(
                &args[2..],
                "--layers",
                format!("0..{}", model.config.layers),
            )?;
            let parts: Vec<&str> = layers_str.split("..").collect();
            let start: usize = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
            let end: usize = parts
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(model.config.layers);
            let listener = std::net::TcpListener::bind(&listen_addr)?;
            eprintln!("Distributed worker listening on {listen_addr} for layers {start}..{end}");
            ferrite::distributed::run_worker(&model, start..end, listener)
        }
        "driver" => {
            if args.len() < 3 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "usage: ferrite distributed driver <model> <prompt> [--remote <host:port>] [--layers <start..end>]",
                ));
            }
            let path = lmstudio::resolve(&args[1])?;
            let gguf = Gguf::open(&path)?;
            let model = Model::load(&gguf)?;
            let tokenizer = Tokenizer::from_gguf(&gguf)?;
            let remote_addr: String = flag(&args[3..], "--remote", "127.0.0.1:9090".to_string())?;
            let layers_str: String = flag(
                &args[3..],
                "--layers",
                format!("1..{}", model.config.layers),
            )?;
            let parts: Vec<&str> = layers_str.split("..").collect();
            let remote_start: usize = parts.first().and_then(|s| s.parse().ok()).unwrap_or(1);
            let remote_end: usize = parts
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(model.config.layers);
            let prompt = &args[2];
            let tokens = tokenizer.encode(prompt, true);
            let mut state = State::new(&model.config);
            let mut driver = ferrite::distributed::DistributedDriver::connect(
                &model,
                0..remote_start,
                remote_start..remote_end,
                &remote_addr,
            )?;
            for (pos, &tok) in tokens.iter().enumerate() {
                driver.forward(&mut state, tok, pos)?;
            }
            let mut sampler = ferrite::sample::Sampler::greedy();
            let next_tok = sampler.sample(&mut state.logits);
            let piece = tokenizer.piece(next_tok);
            println!("{}", String::from_utf8_lossy(&piece));
            driver.shutdown()?;
            Ok(())
        }
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unknown distributed subcommand '{other}', expected worker or driver"),
        )),
    }
}

fn cmd_lora(args: &[String]) -> std::io::Result<()> {
    if args.len() < 4 || args[0] != "merge" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: ferrite lora merge <base.gguf> <out.gguf> <adapter.gguf...>",
        ));
    }
    let base_path = lmstudio::resolve(&args[1])?;
    let base = Gguf::open(&base_path)?;
    let adapter_files: Vec<Gguf> = args[3..]
        .iter()
        .map(|path| lmstudio::resolve(path).and_then(|p| Gguf::open(&p)))
        .collect::<std::io::Result<_>>()?;
    let refs: Vec<&Gguf> = adapter_files.iter().collect();
    ferrite::lora::merge_adapters(&base, &refs, std::path::Path::new(&args[2]))
}

fn cmd_quantize(args: &[String]) -> std::io::Result<()> {
    if args.len() != 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: ferrite quantize <src.gguf> <dst.gguf> <Q8_0|Q4_0|Q4_K>",
        ));
    }
    let target = match args[2].to_ascii_uppercase().as_str() {
        "Q8_0" => ferrite::GgmlType::Q8_0,
        "Q4_0" => ferrite::GgmlType::Q4_0,
        "Q4_K" => ferrite::GgmlType::Q4_K,
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "quantization type must be Q8_0, Q4_0, or Q4_K",
            ))
        }
    };
    ferrite::quant::quantize::quantize_file(
        std::path::Path::new(&args[0]),
        std::path::Path::new(&args[1]),
        target,
    )
}

fn cmd_merge(args: &[String]) -> std::io::Result<()> {
    if args.len() < 5 || args[3] != "--method" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,
            "usage: ferrite merge <model_a.gguf> <model_b.gguf> <out.gguf> --method <slerp|ties|dare> [--weight <0..1>]"));
    }
    let method = ferrite::merge::MergeMethod::parse(&args[4])?;
    let weight = if args.len() == 7 && args[5] == "--weight" {
        args[6].parse().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid merge weight")
        })?
    } else if args.len() == 5 {
        0.5
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "expected optional --weight <0..1>",
        ));
    };
    let a = Gguf::open(&lmstudio::resolve(&args[0])?)?;
    let b = Gguf::open(&lmstudio::resolve(&args[1])?)?;
    ferrite::merge::merge_models(&a, &b, std::path::Path::new(&args[2]), method, weight)
}

fn cmd_perplexity(args: &[String]) -> std::io::Result<()> {
    let context: usize = flag(args, "--ctx", 512)?;
    let stride: usize = flag(args, "--stride", context)?;
    let threads: usize = flag(args, "--threads", 0)?;

    let flags = ["--ctx", "--stride", "--threads"];
    let mut positional = Vec::new();
    let mut index = 0;
    while index < args.len() {
        if flags.contains(&args[index].as_str()) {
            index += 2;
            continue;
        }
        positional.push(args[index].clone());
        index += 1;
    }

    let (query, file_path) = match positional.as_slice() {
        [q, f] => (q, f),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "usage: ferrite perplexity <model> <text_file> [--ctx <n>] [--stride <n>] [--threads <n>]",
            ));
        }
    };

    let text = std::fs::read_to_string(file_path)?;
    let path = lmstudio::resolve(query)?;
    let gguf = Gguf::open(&path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let model = Model::load(&gguf)?;

    let tokens = tokenizer.encode(&text, false);
    if tokens.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "file encoded to {} tokens, but at least 2 tokens are required",
                tokens.len()
            ),
        ));
    }

    let worker_count = if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    };

    eprintln!(
        "model: {} · {} tokens · ctx: {context} · stride: {stride} · threads: {worker_count}",
        model.config.arch,
        tokens.len(),
    );

    let start = Instant::now();
    let evaluator =
        ferrite::eval::PerplexityEvaluator::new(&model, context, stride).with_threads(threads);

    let report = evaluator.evaluate_with_callback(&tokens, |chunk| {
        let chunk_tokens = chunk.count;
        let nll = chunk.chunk_nll / chunk_tokens as f64;
        eprintln!(
            "  chunk {:>3} [{:>6}..{:>6}] · {:>4} tokens · nll: {:>7.4} · ppl: {:>8.4}",
            chunk.chunk_index + 1,
            chunk.start_token,
            chunk.start_token + chunk_tokens,
            chunk_tokens,
            nll,
            chunk.chunk_ppl,
        );
    })?;

    let elapsed = start.elapsed();
    println!("\nPerplexity evaluation results:");
    println!("  Total tokens:      {}", report.total_tokens);
    println!("  Evaluated tokens:  {}", report.evaluated_tokens);
    println!("  Mean loss (NLL):   {:.4}", report.mean_loss);
    println!("  Perplexity (PPL):  {:.4}", report.perplexity);
    println!(
        "  Speed:             {:.1} tok/s ({:.2}s)",
        report.evaluated_tokens as f64 / elapsed.as_secs_f64().max(1e-9),
        elapsed.as_secs_f64()
    );

    Ok(())
}

fn cmd_convert(args: &[String]) -> std::io::Result<()> {
    let arch: Option<String> = {
        let at = args.iter().position(|a| a == "--arch");
        at.and_then(|i| args.get(i + 1).cloned())
    };

    let mut positional = Vec::new();
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--arch" {
            index += 2;
            continue;
        }
        positional.push(args[index].clone());
        index += 1;
    }

    let (src, dst) = match positional.as_slice() {
        [s, d] => (s, d),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "usage: ferrite convert <safetensors_file_or_dir> <output.gguf> [--arch <arch>]",
            ));
        }
    };

    eprintln!("converting {src} -> {dst}...");
    let start = Instant::now();
    ferrite::convert::convert_safetensors_to_gguf(
        std::path::Path::new(src),
        std::path::Path::new(dst),
        arch.as_deref(),
    )?;
    let elapsed = start.elapsed();
    eprintln!(
        "successfully converted to {dst} in {:.2}s",
        elapsed.as_secs_f64()
    );
    Ok(())
}

fn cmd_where() -> std::io::Result<()> {
    let dirs = lmstudio::model_dirs();
    if dirs.is_empty() {
        println!("no model directories exist yet");
        println!("looked for ~/.lmstudio/models and ~/.cache/lm-studio/models");
        println!("set FERRITE_MODELS_DIR to point somewhere else");
        return Ok(());
    }
    for dir in dirs {
        println!("{}", dir.display());
    }
    Ok(())
}

fn cmd_list() -> std::io::Result<()> {
    let models = lmstudio::list();
    if models.is_empty() {
        println!("no .gguf files found. `ferrite where` shows the search paths.");
        return Ok(());
    }
    for m in &models {
        println!("{:>10}  {}", human(m.size), m.id);
    }
    println!("\n{} models", models.len());
    Ok(())
}

fn cmd_info(args: &[String]) -> std::io::Result<()> {
    let show_tensors = args.iter().any(|a| a == "-t" || a == "--tensors");
    let query = args.iter().find(|a| !a.starts_with('-')).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "info needs a model id or path",
        )
    })?;

    let path = lmstudio::resolve(query)?;
    let model = Gguf::open(&path)?;

    println!("{}", path.display());
    println!("{:<22}{}", "size", human(model.file_len()));
    println!("{:<22}v{}", "gguf", model.version);
    if let Some(name) = model.meta_str("general.name") {
        println!("{:<22}{}", "name", name);
    }
    println!("{:<22}{}", "arch", model.arch().unwrap_or("(unspecified)"));

    // Counting tensor elements beats trusting general.parameter_count, which is
    // frequently absent and occasionally wrong.
    let params: u64 = model.tensors.iter().map(|t| t.elements()).sum();
    println!("{:<22}{}", "parameters", si(params));

    for (label, key) in [
        ("layers", "block_count"),
        ("embedding", "embedding_length"),
        ("feed forward", "feed_forward_length"),
        ("heads", "attention.head_count"),
        ("kv heads", "attention.head_count_kv"),
        ("context length", "context_length"),
    ] {
        if let Some(v) = model.arch_u64(key) {
            println!("{label:<22}{v}");
        }
    }
    if let Some(v) = model.arch_f32("rope.freq_base") {
        println!("{:<22}{}", "rope freq base", v);
    }
    if let Some(v) = model.arch_f32("attention.layer_norm_rms_epsilon") {
        println!("{:<22}{}", "rms eps", v);
    }

    if let Some(tok) = model.meta_str("tokenizer.ggml.model") {
        let vocab = model
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        println!("{:<22}{tok} ({vocab} tokens)", "tokenizer");
        for (label, key) in [
            ("bos", "tokenizer.ggml.bos_token_id"),
            ("eos", "tokenizer.ggml.eos_token_id"),
        ] {
            if let Some(id) = model.meta_u64(key) {
                let text = model
                    .get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.get(id as usize))
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                println!("{:<22}{id} {text:?}", format!("{label} token"));
            }
        }
    }

    println!("\n{:<22}{}", "tensors", model.tensors.len());
    for (ty, count, bytes) in model.quant_histogram() {
        let share = if model.file_len() > 0 {
            bytes as f64 / model.file_len() as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  {:<10} {:>4} tensors  {:>10}  {:>5.1}%",
            ty.name(),
            count,
            human(bytes),
            share
        );
    }

    if show_tensors {
        println!();
        for t in &model.tensors {
            println!(
                "  {:<40} {:<18} {:<8} {:>12}",
                t.name,
                t.shape(),
                t.ty.name(),
                t.byte_len.map(human).unwrap_or_else(|| "?".into())
            );
        }
    }
    Ok(())
}

fn cmd_tokenize(args: &[String]) -> std::io::Result<()> {
    let (query, text) = match args {
        [query, rest @ ..] if !rest.is_empty() => (query, rest.join(" ")),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "tokenize needs a model and some text",
            ))
        }
    };

    let model = Gguf::open(&lmstudio::resolve(query)?)?;
    let tokenizer = Tokenizer::from_gguf(&model)?;
    let ids = tokenizer.encode(&text, true);

    println!(
        "{:?} vocab, {} merges, {} tokens",
        tokenizer.kind,
        tokenizer.merge_count(),
        ids.len()
    );
    for id in &ids {
        // Byte-level vocabs store a space as U+0120; showing the decoded piece
        // is more useful than showing the stored form.
        let piece = String::from_utf8_lossy(&tokenizer.piece(*id)).into_owned();
        println!("  {id:>7}  {piece:?}");
    }

    let round_trip = tokenizer.decode(&ids);
    if round_trip != text {
        // Not necessarily a bug — BOS/EOS and control tokens are dropped when
        // decoding — but worth seeing when it happens.
        println!("\ndecoded: {round_trip:?}");
    }
    Ok(())
}

/// Pull `--name value` out of the argument list, leaving the positionals.
fn flag<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> std::io::Result<T> {
    let Some(at) = args.iter().position(|a| a == name) else {
        return Ok(default);
    };
    args.get(at + 1)
        .and_then(|raw| raw.parse().ok())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{name} needs a value"),
            )
        })
}

fn cmd_run(args: &[String]) -> std::io::Result<()> {
    let limit: usize = flag(args, "-n", 128)?;
    let temperature: f32 = flag(args, "--temp", 0.0)?;
    let top_k: usize = flag(args, "--top-k", 40)?;
    let top_p: f32 = flag(args, "--top-p", 0.95)?;
    let min_p: f32 = flag(args, "--min-p", 0.0)?;
    let tfs: f32 = flag(args, "--tfs", 1.0)?;
    let repeat_penalty: f32 = flag(args, "--repeat-penalty", 1.0)?;
    let frequency_penalty: f32 = flag(args, "--frequency-penalty", 0.0)?;
    let presence_penalty: f32 = flag(args, "--presence-penalty", 0.0)?;
    let repeat_last_n: usize = flag(args, "--repeat-last-n", 64)?;
    let seed: u64 = flag(args, "--seed", 1)?;
    let context: usize = flag(args, "--ctx", model::DEFAULT_CONTEXT)?;
    let threads: usize = flag(args, "--threads", 0)?;

    let draft_query: Option<String> = {
        let at = args.iter().position(|a| a == "--draft");
        at.and_then(|i| args.get(i + 1).cloned())
    };
    let spec_k: usize = flag(args, "--spec-k", 3)?;
    let spec_tree: usize = flag(args, "--spec-tree", 0)?;
    let lookahead: usize = flag(args, "--lookahead", 0)?;
    let lookahead_ngram: usize = flag(args, "--lookahead-ngram", 3)?;
    let chunk_size: usize = flag(args, "--chunk-size", 512)?;
    let kv_cache: ferrite::kv::KvCacheType =
        flag(args, "--kv-cache", ferrite::kv::KvCacheType::F32)?;

    let lora_query: Option<String> = {
        let at = args.iter().position(|a| a == "--lora");
        at.and_then(|i| args.get(i + 1).cloned())
    };

    let mut grammar = parse_grammar_args(args)?;

    // Positionals are whatever is left once flags and their values are removed.
    let flags = [
        "-n",
        "--temp",
        "--top-k",
        "--top-p",
        "--min-p",
        "--tfs",
        "--repeat-penalty",
        "--frequency-penalty",
        "--presence-penalty",
        "--repeat-last-n",
        "--seed",
        "--ctx",
        "--threads",
        "--draft",
        "--spec-k",
        "--spec-tree",
        "--lookahead",
        "--lookahead-ngram",
        "--lora",
        "--chunk-size",
        "--kv-cache",
        "--json-schema",
        "--regex",
    ];
    let bool_flags = ["--mtp", "--json"];
    let mut positional = Vec::new();
    let mut index = 0;
    while index < args.len() {
        if flags.contains(&args[index].as_str()) {
            index += 2;
            continue;
        }
        if bool_flags.contains(&args[index].as_str()) {
            index += 1;
            continue;
        }
        positional.push(args[index].clone());
        index += 1;
    }
    let (query, prompt) = match positional.as_slice() {
        [query, rest @ ..] if !rest.is_empty() => (query.clone(), rest.join(" ")),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "run needs a model and a prompt",
            ))
        }
    };

    let path = lmstudio::resolve(&query)?;
    let gguf = Gguf::open(&path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let mut model = Model::load(&gguf)?;

    let lora_gguf = if let Some(lq) = &lora_query {
        let lora_path = lmstudio::resolve(lq)?;
        Some(Gguf::open(&lora_path)?)
    } else {
        None
    };
    if let Some(lgguf) = &lora_gguf {
        let adapter = ferrite::lora::LoraAdapter::load(lgguf, None)?;
        model.apply_lora(adapter);
    }
    let mut state = State::with_kv_cache(&model.config, context, kv_cache);
    if threads > 0 {
        state.threads = threads;
    }
    let penalties = ferrite::sample::PenaltyConfig {
        repetition_penalty: repeat_penalty,
        frequency_penalty,
        presence_penalty,
        last_n: repeat_last_n,
    };
    let mut sampler = Sampler::new(temperature, top_k, top_p, seed)
        .with_min_p(min_p)
        .with_tfs(tfs)
        .with_penalties(penalties);

    // Optional draft model for speculative decoding
    let draft_gguf = if let Some(dq) = &draft_query {
        let draft_path = lmstudio::resolve(dq)?;
        Some(Gguf::open(&draft_path)?)
    } else {
        None
    };

    let mut draft_bundle = if let Some(gguf) = &draft_gguf {
        let draft_model = Model::load(gguf)?;
        let mut draft_state = State::with_context(&draft_model.config, context);
        if threads > 0 {
            draft_state.threads = threads;
        }
        Some((draft_model, draft_state))
    } else {
        None
    };

    let prompt_tokens = tokenizer.encode(&prompt, true);
    if prompt_tokens.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "prompt encoded to no tokens",
        ));
    }
    if let Some((draft_model, _)) = &draft_bundle {
        let tree_info = if spec_tree > 0 {
            format!(" (tree branches={spec_tree})")
        } else {
            String::new()
        };
        eprintln!(
            "target: {} · draft: {} (k={spec_k}{tree_info}) · {} prompt tokens · {} kv cache · {} threads",
            model.config.arch,
            draft_model.config.arch,
            prompt_tokens.len(),
            human(state.cache_bytes() as u64),
            state.threads
        );
    } else {
        eprintln!(
            "{} · {} prompt tokens · {} kv cache · {} threads",
            model.config.arch,
            prompt_tokens.len(),
            human(state.cache_bytes() as u64),
            state.threads
        );
    }

    let mut stdout = std::io::stdout();
    print!("{prompt}");
    stdout.flush().ok();

    let prefill_start = Instant::now();
    if chunk_size > 0 && chunk_size < prompt_tokens.len() {
        model.prefill_chunked(&mut state, &prompt_tokens, 0, chunk_size)?;
        if let Some((draft_model, draft_state)) = &mut draft_bundle {
            draft_model.prefill_chunked(draft_state, &prompt_tokens, 0, chunk_size)?;
        }
    } else {
        model.prefill(&mut state, &prompt_tokens, 0)?;
        if let Some((draft_model, draft_state)) = &mut draft_bundle {
            draft_model.prefill(draft_state, &prompt_tokens, 0)?;
        }
    }
    let prefill = prefill_start.elapsed();

    // A multi-byte character can straddle two tokens, so bytes accumulate
    // until they form valid UTF-8 rather than printing one piece at a time.
    let mut pending: Vec<u8> = Vec::new();
    let mut position = prompt_tokens.len();
    let mut generated = 0;
    let decode_start = Instant::now();

    if let Some((draft_model, draft_state)) = &mut draft_bundle {
        let (acc_rate, tok_per_batch) = if spec_tree > 0 {
            let mut engine = ferrite::speculative::SpeculativeTreeEngine::new(
                &model,
                draft_model,
                spec_k,
                spec_tree,
            );
            while generated < limit && position < state.context() {
                let emitted = engine.step(&mut state, draft_state, position, &mut sampler)?;
                let mut stopped = false;
                for token in emitted {
                    if Some(token) == tokenizer.eos {
                        stopped = true;
                        break;
                    }
                    pending.extend(tokenizer.piece(token));
                    if let Ok(text) = std::str::from_utf8(&pending) {
                        print!("{text}");
                        stdout.flush().ok();
                        pending.clear();
                    }
                    position += 1;
                    generated += 1;
                    if generated >= limit || position >= state.context() {
                        stopped = true;
                        break;
                    }
                }
                if stopped {
                    break;
                }
            }
            (
                engine.stats.acceptance_rate(),
                engine.stats.tokens_per_batch(),
            )
        } else {
            let mut engine =
                ferrite::speculative::SpeculativeEngine::new(&model, draft_model, spec_k);
            while generated < limit && position < state.context() {
                let emitted = engine.step(&mut state, draft_state, position, &mut sampler)?;
                let mut stopped = false;
                for token in emitted {
                    if Some(token) == tokenizer.eos {
                        stopped = true;
                        break;
                    }
                    pending.extend(tokenizer.piece(token));
                    if let Ok(text) = std::str::from_utf8(&pending) {
                        print!("{text}");
                        stdout.flush().ok();
                        pending.clear();
                    }
                    position += 1;
                    generated += 1;
                    if generated >= limit || position >= state.context() {
                        stopped = true;
                        break;
                    }
                }
                if stopped {
                    break;
                }
            }
            (
                engine.stats.acceptance_rate(),
                engine.stats.tokens_per_batch(),
            )
        };
        println!();
        let decode = decode_start.elapsed();
        eprintln!(
            "\nprefill {:.2}s ({:.1} tok/s) · decode {generated} tokens in {:.2}s ({:.1} tok/s) · acceptance rate {:.1}% ({:.2} tok/target-step)",
            prefill.as_secs_f64(),
            prompt_tokens.len() as f64 / prefill.as_secs_f64().max(1e-9),
            decode.as_secs_f64(),
            generated as f64 / decode.as_secs_f64().max(1e-9),
            acc_rate * 100.0,
            tok_per_batch,
        );
    } else if lookahead > 0 {
        let mut engine =
            ferrite::speculative::LookaheadEngine::new(&model, lookahead_ngram, lookahead);
        let mut token_history = prompt_tokens.clone();
        while generated < limit && position < state.context() {
            let emitted = engine.step(&mut state, &token_history, position, &mut sampler)?;
            let mut stopped = false;
            for token in emitted {
                if Some(token) == tokenizer.eos {
                    stopped = true;
                    break;
                }
                token_history.push(token);
                pending.extend(tokenizer.piece(token));
                if let Ok(text) = std::str::from_utf8(&pending) {
                    print!("{text}");
                    stdout.flush().ok();
                    pending.clear();
                }
                position += 1;
                generated += 1;
                if generated >= limit || position >= state.context() {
                    stopped = true;
                    break;
                }
            }
            if stopped {
                break;
            }
        }
        println!();
        let decode = decode_start.elapsed();
        eprintln!(
            "\nprefill {:.2}s ({:.1} tok/s) · decode {generated} tokens in {:.2}s ({:.1} tok/s) · lookahead acceptance rate {:.1}% ({:.2} tok/target-step)",
            prefill.as_secs_f64(),
            prompt_tokens.len() as f64 / prefill.as_secs_f64().max(1e-9),
            decode.as_secs_f64(),
            generated as f64 / decode.as_secs_f64().max(1e-9),
            engine.stats.acceptance_rate() * 100.0,
            engine.stats.tokens_per_batch(),
        );
    } else if (args.iter().any(|a| a == "--mtp") || model.mtp_module.is_some())
        && model.mtp_module.is_some()
        && grammar.is_none()
    {
        let mut engine = ferrite::speculative::MtpEngine::new(&model);
        while generated < limit && position < state.context() {
            let emitted = engine.step(&mut state, position, &mut sampler)?;
            let mut stopped = false;
            for token in emitted {
                if Some(token) == tokenizer.eos {
                    stopped = true;
                    break;
                }
                pending.extend(tokenizer.piece(token));
                if let Ok(text) = std::str::from_utf8(&pending) {
                    print!("{text}");
                    stdout.flush().ok();
                    pending.clear();
                }
                position += 1;
                generated += 1;
                if generated >= limit || position >= state.context() {
                    stopped = true;
                    break;
                }
            }
            if stopped {
                break;
            }
        }
        println!();
        let decode = decode_start.elapsed();
        eprintln!(
            "\nprefill {:.2}s ({:.1} tok/s) · decode {generated} tokens in {:.2}s ({:.1} tok/s) · MTP acceptance rate {:.1}% ({:.2} tok/target-step)",
            prefill.as_secs_f64(),
            prompt_tokens.len() as f64 / prefill.as_secs_f64().max(1e-9),
            decode.as_secs_f64(),
            generated as f64 / decode.as_secs_f64().max(1e-9),
            engine.stats.acceptance_rate() * 100.0,
            engine.stats.tokens_per_batch(),
        );
    } else {
        let mut token_history = prompt_tokens.clone();
        while generated < limit && position < state.context() {
            if let Some(g) = &grammar {
                g.filter_logits(&mut state.logits, &tokenizer);
            }
            let token = sampler.sample_with_history(&mut state.logits, &token_history);
            if Some(token) == tokenizer.eos {
                break;
            }
            if let Some(g) = &mut grammar {
                g.advance_token(token, &tokenizer);
            }
            token_history.push(token);
            pending.extend(tokenizer.piece(token));
            if let Ok(text) = std::str::from_utf8(&pending) {
                print!("{text}");
                stdout.flush().ok();
                pending.clear();
            }

            model.forward(&mut state, token, position)?;
            position += 1;
            generated += 1;
        }
        println!();
        let decode = decode_start.elapsed();
        eprintln!(
            "\nprefill {:.2}s ({:.1} tok/s) · decode {generated} tokens in {:.2}s ({:.1} tok/s)",
            prefill.as_secs_f64(),
            prompt_tokens.len() as f64 / prefill.as_secs_f64().max(1e-9),
            decode.as_secs_f64(),
            generated as f64 / decode.as_secs_f64().max(1e-9),
        );
    }
    Ok(())
}

fn parse_grammar_args(args: &[String]) -> std::io::Result<Option<ferrite::grammar::Grammar>> {
    if args.iter().any(|a| a == "--json") {
        return Ok(Some(ferrite::grammar::Grammar::json()));
    }
    if let Some(pos) = args.iter().position(|a| a == "--json-schema") {
        let val = args.get(pos + 1).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--json-schema needs a schema file or JSON string",
            )
        })?;
        let schema_str = if std::path::Path::new(val).is_file() {
            std::fs::read_to_string(val)?
        } else {
            val.clone()
        };
        let g = ferrite::grammar::Grammar::schema(&schema_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        return Ok(Some(g));
    }
    if let Some(pos) = args.iter().position(|a| a == "--regex") {
        let pat = args.get(pos + 1).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--regex needs a pattern string",
            )
        })?;
        let g = ferrite::grammar::Grammar::regex(pat)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
        return Ok(Some(g));
    }
    Ok(None)
}

fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn si(n: u64) -> String {
    match n {
        n if n >= 1_000_000_000 => format!("{:.2}B", n as f64 / 1e9),
        n if n >= 1_000_000 => format!("{:.0}M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.0}K", n as f64 / 1e3),
        n => n.to_string(),
    }
}

fn cmd_serve(args: &[String]) -> std::io::Result<()> {
    let mut host = "127.0.0.1".to_string();
    let mut port = 8080u16;
    let mut context = 2048usize;
    let mut threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let mut query = None;
    let mut lora: Option<String> = None;
    let mut kv_cache = ferrite::kv::KvCacheType::F32;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => {
                i += 1;
                if let Some(h) = args.get(i) {
                    host = h.clone();
                }
            }
            "--port" => {
                i += 1;
                if let Some(p) = args.get(i).and_then(|s| s.parse().ok()) {
                    port = p;
                }
            }
            "--ctx" => {
                i += 1;
                if let Some(c) = args.get(i).and_then(|s| s.parse().ok()) {
                    context = c;
                }
            }
            "--threads" => {
                i += 1;
                if let Some(t) = args.get(i).and_then(|s| s.parse().ok()) {
                    threads = t;
                }
            }
            "--lora" => {
                i += 1;
                if let Some(l) = args.get(i) {
                    lora = Some(l.clone());
                }
            }
            "--kv-cache" => {
                i += 1;
                if let Some(k) = args.get(i).and_then(|s| s.parse().ok()) {
                    kv_cache = k;
                }
            }
            arg if !arg.starts_with('-') && query.is_none() => {
                query = Some(arg.to_string());
            }
            _ => {}
        }
        i += 1;
    }

    let query = query.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "serve needs a model id or path",
        )
    })?;

    let path = lmstudio::resolve(&query)?;
    let gguf = Box::leak(Box::new(Gguf::open(&path)?));
    let model = Box::leak(Box::new(Model::load(gguf)?));
    if let Some(lq) = &lora {
        let lora_path = lmstudio::resolve(lq)?;
        let lora_gguf = Box::leak(Box::new(Gguf::open(&lora_path)?));
        let adapter = ferrite::lora::LoraAdapter::load(lora_gguf, None)?;
        model.apply_lora(adapter);
    }
    let tokenizer = Box::leak(Box::new(Tokenizer::from_gguf(gguf)?));

    let config = ferrite::server::ServerConfig {
        host: host.clone(),
        port,
        threads,
        context,
        kv_cache,
        tenant_registry: None,
    };

    let server = ferrite::server::Server::new(model, tokenizer, config);
    let listener = server.listen()?;
    eprintln!(
        "ferrite HTTP inference daemon listening on http://{host}:{port} ({}, {} threads, ctx {context})",
        model.config.arch, threads
    );

    let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    server.serve_with_shutdown(&listener, shutdown)
}

fn cmd_vision(args: &[String]) -> std::io::Result<()> {
    let limit: usize = flag(args, "-n", 128)?;
    let temperature: f32 = flag(args, "--temp", 0.0)?;
    let top_k: usize = flag(args, "--top-k", 40)?;
    let top_p: f32 = flag(args, "--top-p", 0.95)?;
    let seed: u64 = flag(args, "--seed", 1)?;
    let context: usize = flag(args, "--ctx", model::DEFAULT_CONTEXT)?;
    let threads: usize = flag(args, "--threads", 0)?;

    let image_path: String = {
        let at = args.iter().position(|a| a == "--image");
        match at.and_then(|i| args.get(i + 1).cloned()) {
            Some(p) => p,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "vision requires --image <path.ppm|path.bmp>",
                ));
            }
        }
    };

    let mmproj_path: String = {
        let at = args.iter().position(|a| a == "--mmproj");
        match at.and_then(|i| args.get(i + 1).cloned()) {
            Some(p) => p,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "vision requires --mmproj <path.gguf>",
                ));
            }
        }
    };

    let flags = [
        "-n",
        "--temp",
        "--top-k",
        "--top-p",
        "--seed",
        "--ctx",
        "--threads",
        "--image",
        "--mmproj",
    ];
    let mut positional = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        if flags.contains(&args[idx].as_str()) {
            idx += 2;
            continue;
        }
        positional.push(args[idx].clone());
        idx += 1;
    }

    let (model_query, prompt) = match positional.as_slice() {
        [m] => (m.as_str(), "Describe this image in detail:".to_string()),
        [m, rest @ ..] => (m.as_str(), rest.join(" ")),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "usage: ferrite vision <model> --image <image> --mmproj <mmproj.gguf> [prompt]",
            ));
        }
    };

    let model_path = lmstudio::resolve(model_query)?;
    let model_gguf = Gguf::open(&model_path)?;
    let tokenizer = Tokenizer::from_gguf(&model_gguf)?;
    let model = Model::load(&model_gguf)?;

    let mm_path = lmstudio::resolve(&mmproj_path)?;
    let mm_gguf = Gguf::open(&mm_path)?;
    let vision_encoder = ferrite::vision::VisionEncoder::from_gguf(&mm_gguf)?;

    let image = ferrite::vision::Image::open(&image_path)?;
    let image_tokens = vision_encoder.encode_image(&image)?;

    let mut state = State::with_context(&model.config, context);
    if threads > 0 {
        state.threads = threads;
    }

    let prefix_tokens = tokenizer.encode("[USER] Image: ", true);
    let suffix_tokens = tokenizer.encode(&format!("\nPrompt: {prompt}\n[ASSISTANT] "), false);

    let start = Instant::now();
    model.prefill_multimodal(&mut state, &prefix_tokens, &image_tokens, &suffix_tokens, 0)?;

    let mut sampler = Sampler::new(temperature, top_k, top_p, seed);
    let mut next_token = sampler.sample(&mut state.logits);

    let mut stdout = std::io::stdout();
    let num_patches = image_tokens.len() / model.config.embedding;
    eprintln!(
        "Vision inference: image ({}x{} -> {} tokens) + prompt ({} tokens)",
        image.width,
        image.height,
        num_patches,
        prefix_tokens.len() + suffix_tokens.len()
    );

    let mut pos = prefix_tokens.len() + num_patches + suffix_tokens.len();
    let mut generated = 0;
    while generated < limit {
        if Some(next_token) == tokenizer.eos {
            break;
        }
        let piece = tokenizer.piece(next_token);
        stdout.write_all(&piece).ok();
        stdout.flush().ok();

        model.forward(&mut state, next_token, pos)?;
        pos += 1;
        generated += 1;
        next_token = sampler.sample(&mut state.logits);
    }
    println!();
    let elapsed = start.elapsed();
    eprintln!(
        "\nGenerated {} tokens in {:.2?} ({:.1} tok/s)",
        generated,
        elapsed,
        generated as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}

fn invalid_input(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, msg.into())
}

fn read_linux_meminfo_bytes() -> (u64, u64) {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return (0, 0);
    };
    let mut total_kb = 0u64;
    let mut avail_kb = 0u64;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kb = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        }
    }
    (total_kb * 1024, avail_kb * 1024)
}

fn valid_quant_bytes(ty: ferrite::GgmlType, elements: usize) -> Vec<u8> {
    let (block, size) = ty.layout().expect("sized quant type");
    let block = block as usize;
    let size = size as usize;
    let num_blocks = elements / block;
    let mut out = vec![0u8; num_blocks * size];
    // Safe non-NaN f16 scale bytes: 0x2C00 = 0.0625, 0x2000 = 0.0078125
    let scale = [0x00u8, 0x2C];
    let min_scale = [0x00u8, 0x20];
    for (idx, blk) in out.chunks_exact_mut(size).enumerate() {
        for (off, b) in blk.iter_mut().enumerate() {
            *b = ((idx * 13 + off * 37 + 11) % 251) as u8;
        }
        match ty {
            ferrite::GgmlType::Q4_0 | ferrite::GgmlType::Q8_0 => {
                blk[0..2].copy_from_slice(&scale);
            }
            ferrite::GgmlType::Q4_K | ferrite::GgmlType::Q5_K => {
                blk[0..2].copy_from_slice(&scale);
                blk[2..4].copy_from_slice(&min_scale);
            }
            ferrite::GgmlType::Q6_K => {
                blk[208..210].copy_from_slice(&scale);
            }
            _ => {}
        }
    }
    out
}

fn cmd_doctor(args: &[String]) -> std::io::Result<()> {
    let mut model_query: Option<String> = None;
    for arg in args {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            a if !a.starts_with('-') && model_query.is_none() => {
                model_query = Some(a.to_string());
            }
            other => return Err(invalid_input(format!("unknown flag: {other}"))),
        }
    }

    let arch = std::env::consts::ARCH;
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let (total_bytes, avail_bytes) = read_linux_meminfo_bytes();
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let avail_gb = avail_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    #[cfg(target_arch = "x86_64")]
    let (has_avx2, has_fma, has_f16c, has_avx512f) = (
        std::arch::is_x86_feature_detected!("avx2"),
        std::arch::is_x86_feature_detected!("fma"),
        std::arch::is_x86_feature_detected!("f16c"),
        std::arch::is_x86_feature_detected!("avx512f"),
    );
    #[cfg(not(target_arch = "x86_64"))]
    let (has_avx2, has_fma, has_f16c, has_avx512f) = (false, false, false, false);

    #[cfg(target_arch = "aarch64")]
    let has_neon = true;
    #[cfg(not(target_arch = "aarch64"))]
    let has_neon = false;

    println!("Ferrite Hardware & Local Readiness Doctor");
    println!("=========================================");
    println!("Architecture:      {arch}");
    println!("Logical CPUs:      {cpus}");
    if total_bytes > 0 {
        println!("Total RAM:         {total_gb:.2} GiB");
        println!("Available RAM:     {avail_gb:.2} GiB");
    }
    println!("SIMD Extensions:");
    println!(
        "  AVX2:            {}",
        if has_avx2 { "yes (active)" } else { "no" }
    );
    println!(
        "  FMA:             {}",
        if has_fma { "yes (active)" } else { "no" }
    );
    println!(
        "  F16C:            {}",
        if has_f16c { "yes (active)" } else { "no" }
    );
    println!(
        "  AVX-512F:        {}",
        if has_avx512f { "yes (active)" } else { "no" }
    );
    println!(
        "  ARM NEON:        {}",
        if has_neon { "yes (active)" } else { "no" }
    );

    // Quick Q4_K integer dot bandwidth probe (4096 x 1024)
    let cols = 4096usize;
    let rows = 1024usize;
    let raw = valid_quant_bytes(ferrite::GgmlType::Q4_K, rows * cols);
    let row_bytes = raw.len() / rows;
    let act: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.13).sin()).collect();
    let mut quantized = ferrite::quant::activation::Quantized::with_capacity(cols);
    quantized.fill(&act);
    let iters = 16usize;
    let t0 = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iters {
        for r in 0..rows {
            sink += ferrite::quant::idot::integer(
                ferrite::GgmlType::Q4_K,
                &raw[r * row_bytes..(r + 1) * row_bytes],
                &quantized,
            )
            .unwrap_or(0.0);
        }
    }
    let dt = t0.elapsed().as_secs_f64().max(1e-9);
    std::hint::black_box(sink);
    let bw_1t = (raw.len() as f64 * iters as f64) / dt / (1024.0 * 1024.0 * 1024.0);
    println!("Q4_K iDot Speed:   {bw_1t:.1} GiB/s (single-core streaming)");

    if total_bytes > 0 {
        println!("\nModel Tier Recommendations (Q4_K_M quantization):");
        let tiers = [
            ("1B - 3B (Llama-3.2-3B / Qwen2.5-3B)", 2.2),
            ("7B - 8B (Llama-3.1-8B / Qwen2.5-7B)", 5.2),
            ("13B - 14B (Qwen2.5-14B / Phi-4)", 9.0),
            ("32B (Qwen2.5-32B / DeepSeek-R1-32B)", 20.0),
            ("70B (Llama-3.3-70B)", 42.0),
        ];
        for (name, req_gb) in tiers {
            let status = if avail_gb >= req_gb * 1.15 {
                "READY (fits comfortably in available RAM)"
            } else if total_gb >= req_gb * 1.05 {
                "TIGHT (fits in total RAM; close background apps)"
            } else {
                "TOO LARGE (requires mmap paging or smaller quant)"
            };
            println!("  {name:<38} ~{req_gb:>4.1} GiB -> {status}");
        }
    }

    if let Some(q) = model_query {
        let path = lmstudio::resolve(&q)?;
        let file_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let gguf = Gguf::open(&path)?;
        let size_gb = file_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("\nTarget Model Diagnosis: {}", path.display());
        println!("  File Size:       {size_gb:.2} GiB");
        println!("  Architecture:    {}", gguf.arch().unwrap_or("unknown"));
        println!("  Tensors:         {}", gguf.tensors.len());
    }

    let discovered = lmstudio::list();
    println!(
        "\nDiscovered Local GGUF Models: {} found across LM Studio / HF / Jan / GPT4All / local",
        discovered.len()
    );
    for m in discovered.iter().take(10) {
        let mb = m.size as f64 / (1024.0 * 1024.0);
        println!("  - {} ({mb:.1} MiB)", m.id);
    }
    if discovered.len() > 10 {
        println!(
            "  ... and {} more (run `ferrite list` to see all)",
            discovered.len() - 10
        );
    }

    Ok(())
}

fn build_synthetic_bench_gguf(path: &std::path::Path) -> std::io::Result<()> {
    let dim = 256usize;
    let ffn = 512usize;
    let vocab = 256usize;
    let heads = 4u32;
    let kv_heads = 4u32;

    let bytes_f32 =
        |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let emb: Vec<f32> = (0..vocab * dim)
        .map(|i| ((i * 17 + 3) % 97) as f32 * 0.01 - 0.5)
        .collect();
    let norm = vec![1.0f32; dim];

    let q_q4k = valid_quant_bytes(ferrite::GgmlType::Q4_K, dim * dim);
    let k_q8 = valid_quant_bytes(ferrite::GgmlType::Q8_0, dim * dim);
    let v_q8 = valid_quant_bytes(ferrite::GgmlType::Q8_0, dim * dim);
    let o_q4k = valid_quant_bytes(ferrite::GgmlType::Q4_K, dim * dim);
    let ffn_up = valid_quant_bytes(ferrite::GgmlType::Q4_K, dim * ffn);
    let ffn_gate = valid_quant_bytes(ferrite::GgmlType::Q5_K, dim * ffn);
    let ffn_down = valid_quant_bytes(ferrite::GgmlType::Q6_K, ffn * dim);
    let out_w = valid_quant_bytes(ferrite::GgmlType::Q8_0, dim * vocab);

    use ferrite::gguf::Value;
    let tokens: Vec<Value> = (0..vocab)
        .map(|i| Value::String(format!("tok{i}")))
        .collect();

    ferrite::synth::Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(2))
        .meta("llama.embedding_length", Value::U32(dim as u32))
        .meta("llama.feed_forward_length", Value::U32(ffn as u32))
        .meta("llama.attention.head_count", Value::U32(heads))
        .meta("llama.attention.head_count_kv", Value::U32(kv_heads))
        .meta("llama.context_length", Value::U32(512))
        .meta("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5))
        .meta("llama.rope.freq_base", Value::F32(10000.0))
        .meta("tokenizer.ggml.model", Value::String("llama".into()))
        .meta("tokenizer.ggml.tokens", Value::Array(tokens))
        .tensor(
            "token_embd.weight",
            &[dim as u64, vocab as u64],
            ferrite::GgmlType::F32,
            bytes_f32(&emb),
        )
        .tensor(
            "blk.0.attn_norm.weight",
            &[dim as u64],
            ferrite::GgmlType::F32,
            bytes_f32(&norm),
        )
        .tensor(
            "blk.0.attn_q.weight",
            &[dim as u64, dim as u64],
            ferrite::GgmlType::Q4_K,
            q_q4k.clone(),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[dim as u64, dim as u64],
            ferrite::GgmlType::Q8_0,
            k_q8.clone(),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[dim as u64, dim as u64],
            ferrite::GgmlType::Q8_0,
            v_q8.clone(),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[dim as u64, dim as u64],
            ferrite::GgmlType::Q4_K,
            o_q4k.clone(),
        )
        .tensor(
            "blk.0.ffn_norm.weight",
            &[dim as u64],
            ferrite::GgmlType::F32,
            bytes_f32(&norm),
        )
        .tensor(
            "blk.0.ffn_up.weight",
            &[dim as u64, ffn as u64],
            ferrite::GgmlType::Q4_K,
            ffn_up.clone(),
        )
        .tensor(
            "blk.0.ffn_gate.weight",
            &[dim as u64, ffn as u64],
            ferrite::GgmlType::Q5_K,
            ffn_gate.clone(),
        )
        .tensor(
            "blk.0.ffn_down.weight",
            &[ffn as u64, dim as u64],
            ferrite::GgmlType::Q6_K,
            ffn_down.clone(),
        )
        .tensor(
            "blk.1.attn_norm.weight",
            &[dim as u64],
            ferrite::GgmlType::F32,
            bytes_f32(&norm),
        )
        .tensor(
            "blk.1.attn_q.weight",
            &[dim as u64, dim as u64],
            ferrite::GgmlType::Q4_K,
            q_q4k,
        )
        .tensor(
            "blk.1.attn_k.weight",
            &[dim as u64, dim as u64],
            ferrite::GgmlType::Q8_0,
            k_q8,
        )
        .tensor(
            "blk.1.attn_v.weight",
            &[dim as u64, dim as u64],
            ferrite::GgmlType::Q8_0,
            v_q8,
        )
        .tensor(
            "blk.1.attn_output.weight",
            &[dim as u64, dim as u64],
            ferrite::GgmlType::Q4_K,
            o_q4k,
        )
        .tensor(
            "blk.1.ffn_norm.weight",
            &[dim as u64],
            ferrite::GgmlType::F32,
            bytes_f32(&norm),
        )
        .tensor(
            "blk.1.ffn_up.weight",
            &[dim as u64, ffn as u64],
            ferrite::GgmlType::Q4_K,
            ffn_up,
        )
        .tensor(
            "blk.1.ffn_gate.weight",
            &[dim as u64, ffn as u64],
            ferrite::GgmlType::Q5_K,
            ffn_gate,
        )
        .tensor(
            "blk.1.ffn_down.weight",
            &[ffn as u64, dim as u64],
            ferrite::GgmlType::Q6_K,
            ffn_down,
        )
        .tensor(
            "output_norm.weight",
            &[dim as u64],
            ferrite::GgmlType::F32,
            bytes_f32(&norm),
        )
        .tensor(
            "output.weight",
            &[dim as u64, vocab as u64],
            ferrite::GgmlType::Q8_0,
            out_w,
        )
        .write(path)
}

struct BenchPoolCtx {
    qtype: ferrite::GgmlType,
    raw_ptr: usize,
    row_bytes: usize,
    q_ptr: usize,
    out_ptr: usize,
}

unsafe fn bench_pool_worker(ctx_ptr: *const (), start: usize, end: usize, _tid: usize) {
    let ctx = &*(ctx_ptr as *const BenchPoolCtx);
    let q = &*(ctx.q_ptr as *const ferrite::quant::activation::Quantized);
    for r in start..end {
        let row_slice = std::slice::from_raw_parts(
            (ctx.raw_ptr as *const u8).add(r * ctx.row_bytes),
            ctx.row_bytes,
        );
        let val = ferrite::quant::idot::integer(ctx.qtype, row_slice, q).unwrap_or(0.0);
        *(ctx.out_ptr as *mut f32).add(r) = val;
    }
}

fn cmd_bench(args: &[String]) -> std::io::Result<()> {
    let mut model_arg: Option<String> = None;
    let mut threads: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut pp_tokens: usize = 64;
    let mut tg_tokens: usize = 32;
    let mut json_out = false;
    let mut export_verify: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                model_arg = Some(
                    args.get(i)
                        .ok_or_else(|| invalid_input("--model requires a path"))?
                        .clone(),
                );
            }
            "--threads" => {
                i += 1;
                threads = args
                    .get(i)
                    .ok_or_else(|| invalid_input("--threads requires a number"))?
                    .parse()
                    .map_err(|_| invalid_input("invalid --threads"))?;
            }
            "--prompt-tokens" => {
                i += 1;
                pp_tokens = args
                    .get(i)
                    .ok_or_else(|| invalid_input("--prompt-tokens requires a number"))?
                    .parse()
                    .map_err(|_| invalid_input("invalid --prompt-tokens"))?;
            }
            "--gen-tokens" => {
                i += 1;
                tg_tokens = args
                    .get(i)
                    .ok_or_else(|| invalid_input("--gen-tokens requires a number"))?
                    .parse()
                    .map_err(|_| invalid_input("invalid --gen-tokens"))?;
            }
            "--export-verify" => {
                i += 1;
                export_verify = Some(
                    args.get(i)
                        .ok_or_else(|| invalid_input("--export-verify requires a path"))?
                        .clone(),
                );
            }
            "--json" => {
                json_out = true;
            }
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            arg if !arg.starts_with('-') && model_arg.is_none() => {
                model_arg = Some(arg.to_string());
            }
            arg => return Err(invalid_input(format!("unknown flag: {arg}"))),
        }
        i += 1;
    }

    let rows = 4096usize;
    let cols = 4096usize;
    let act: Vec<f32> = (0..cols).map(|idx| (idx as f32 * 0.37).sin()).collect();

    struct KernelResult {
        name: String,
        ms_f32_1t: f64,
        ms_i8_1t: f64,
        gbs_1t: f64,
        ms_i8_mt: f64,
        gbs_mt: f64,
        rel_err: f64,
        raw: Vec<u8>,
        f32_out: Vec<f32>,
        out: Vec<f32>,
    }

    let types = [
        ferrite::GgmlType::Q4_K,
        ferrite::GgmlType::Q5_K,
        ferrite::GgmlType::Q6_K,
        ferrite::GgmlType::Q8_0,
        ferrite::GgmlType::Q4_0,
    ];

    let pool = std::sync::Arc::new(ferrite::tp::ThreadPool::new(threads.max(1)));
    let mut results = Vec::new();
    let mut scratch = vec![0.0f32; cols];

    for qtype in types {
        let raw = valid_quant_bytes(qtype, rows * cols);
        let row_bytes = raw.len() / rows;
        let mut quantized = ferrite::quant::activation::Quantized::with_capacity(cols);
        quantized.fill(&act);

        // Compute exact dequantized FP32 reference across all rows
        let mut f32_out = vec![0.0f32; rows];
        for (r, slot) in f32_out.iter_mut().enumerate() {
            let row_slice = &raw[r * row_bytes..(r + 1) * row_bytes];
            ferrite::quant::dequantize(qtype, row_slice, &mut scratch)?;
            *slot = ferrite::ops::dot(&scratch, &act);
        }

        // Compare integer dot against exact dequantized f32 dot across all rows
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (r, &f_val) in f32_out.iter().enumerate() {
            let row_slice = &raw[r * row_bytes..(r + 1) * row_bytes];
            let i_val = ferrite::quant::idot::integer(qtype, row_slice, &quantized).unwrap_or(0.0);
            let d = (i_val - f_val) as f64;
            diff_sq += d * d;
            ref_sq += (f_val as f64) * (f_val as f64);
        }
        let rel_err = diff_sq.sqrt() / ref_sq.sqrt().max(1e-9);

        let iters = 16usize;
        let mut out = vec![0.0f32; rows];

        // 1T F32 reference / fused dot
        let t_f32 = Instant::now();
        for _ in 0..4 {
            for (r, slot) in out.iter_mut().enumerate() {
                let row_slice = &raw[r * row_bytes..(r + 1) * row_bytes];
                *slot = match ferrite::quant::dot::fused(qtype, row_slice, &act) {
                    Some(v) => v,
                    None => {
                        ferrite::quant::dequantize(qtype, row_slice, &mut scratch)?;
                        ferrite::ops::dot(&scratch, &act)
                    }
                };
            }
            std::hint::black_box(&out);
        }
        let ms_f32_1t = (t_f32.elapsed().as_secs_f64() * 1000.0) / 4.0;

        // Warmup 1T INT8 iDot
        quantized.fill(&act);
        for (r, slot) in out.iter_mut().enumerate() {
            *slot = ferrite::quant::idot::integer(
                qtype,
                &raw[r * row_bytes..(r + 1) * row_bytes],
                &quantized,
            )
            .unwrap_or(0.0);
        }

        // 1T INT8 iDot
        let t_i8_1t = Instant::now();
        for _ in 0..iters {
            quantized.fill(&act);
            for (r, slot) in out.iter_mut().enumerate() {
                *slot = ferrite::quant::idot::integer(
                    qtype,
                    &raw[r * row_bytes..(r + 1) * row_bytes],
                    &quantized,
                )
                .unwrap_or(0.0);
            }
            std::hint::black_box(&out);
        }
        let dt_1t = t_i8_1t.elapsed().as_secs_f64().max(1e-9);
        let ms_i8_1t = (dt_1t * 1000.0) / iters as f64;
        let gbs_1t = (raw.len() as f64 * iters as f64) / dt_1t / (1024.0 * 1024.0 * 1024.0);

        // Warmup MT INT8 iDot
        {
            let ctx = BenchPoolCtx {
                qtype,
                raw_ptr: raw.as_ptr() as usize,
                row_bytes,
                q_ptr: (&quantized as *const ferrite::quant::activation::Quantized) as usize,
                out_ptr: out.as_mut_ptr() as usize,
            };
            pool.dispatch(rows, &ctx, bench_pool_worker);
        }

        // MT INT8 iDot via persistent ThreadPool
        let t_i8_mt = Instant::now();
        for _ in 0..iters {
            quantized.fill(&act);
            let ctx = BenchPoolCtx {
                qtype,
                raw_ptr: raw.as_ptr() as usize,
                row_bytes,
                q_ptr: (&quantized as *const ferrite::quant::activation::Quantized) as usize,
                out_ptr: out.as_mut_ptr() as usize,
            };
            pool.dispatch(rows, &ctx, bench_pool_worker);
            std::hint::black_box(&out);
        }
        let dt_mt = t_i8_mt.elapsed().as_secs_f64().max(1e-9);
        let ms_i8_mt = (dt_mt * 1000.0) / iters as f64;
        let gbs_mt = (raw.len() as f64 * iters as f64) / dt_mt / (1024.0 * 1024.0 * 1024.0);

        results.push(KernelResult {
            name: qtype.name().to_string(),
            ms_f32_1t,
            ms_i8_1t,
            gbs_1t,
            ms_i8_mt,
            gbs_mt,
            rel_err,
            raw,
            f32_out,
            out,
        });
    }

    if let Some(ref verify_path) = export_verify {
        let mut f = std::fs::File::create(verify_path)?;
        f.write_all(&(rows as u32).to_le_bytes())?;
        f.write_all(&(cols as u32).to_le_bytes())?;
        f.write_all(&(results.len() as u32).to_le_bytes())?;
        for &v in &act {
            f.write_all(&v.to_le_bytes())?;
        }
        for k in &results {
            let mut name_buf = [0u8; 8];
            let b = k.name.as_bytes();
            name_buf[..b.len().min(7)].copy_from_slice(&b[..b.len().min(7)]);
            f.write_all(&name_buf)?;
            f.write_all(&(k.raw.len() as u64).to_le_bytes())?;
            f.write_all(&k.ms_i8_1t.to_le_bytes())?;
            f.write_all(&k.ms_i8_mt.to_le_bytes())?;
            f.write_all(&k.raw)?;
            for &v in &k.f32_out {
                f.write_all(&v.to_le_bytes())?;
            }
            for &v in &k.out {
                f.write_all(&v.to_le_bytes())?;
            }
        }
    }

    // End-to-end model prefill (pp) and token generation (tg) benchmark
    let temp_path =
        std::env::temp_dir().join(format!("ferrite_bench_model_{}.gguf", std::process::id()));
    let model_path = if let Some(ref m) = model_arg {
        lmstudio::resolve(m)?
    } else {
        build_synthetic_bench_gguf(&temp_path)?;
        temp_path.clone()
    };

    let gguf = Gguf::open(&model_path)?;
    let model = Model::load(&gguf)?;
    let ctx_len = (pp_tokens + tg_tokens + 32).max(256);
    let mut state = State::with_context(&model.config, ctx_len);
    state.threads = threads.max(1);

    let prompt_tokens: Vec<u32> = (0..pp_tokens)
        .map(|idx| (idx as u32) % (model.config.vocab as u32).max(1))
        .collect();

    // Warmup
    model.prefill(&mut state, &prompt_tokens[..pp_tokens.min(8)], 0)?;

    // Measure prefill (pp)
    let t_pp = Instant::now();
    model.prefill(&mut state, &prompt_tokens, 0)?;
    let dt_pp = t_pp.elapsed().as_secs_f64().max(1e-9);
    let pp_tok_s = pp_tokens as f64 / dt_pp;

    // Measure decode (tg)
    let mut tok = 1u32 % (model.config.vocab as u32).max(1);
    let t_tg = Instant::now();
    for pos in (pp_tokens..).take(tg_tokens) {
        model.forward(&mut state, tok, pos)?;
        tok = (tok + 1) % (model.config.vocab as u32).max(1);
    }
    let dt_tg = t_tg.elapsed().as_secs_f64().max(1e-9);
    let tg_tok_s = tg_tokens as f64 / dt_tg;

    drop(model);
    drop(gguf);
    if model_arg.is_none() {
        let _ = std::fs::remove_file(&temp_path);
    }

    if json_out {
        print!("{{\"threads\":{threads},\"kernels\":[");
        for (idx, k) in results.iter().enumerate() {
            if idx > 0 {
                print!(",");
            }
            print!(
                "{{\"type\":\"{}\",\"ms_f32_1t\":{:.4},\"ms_i8_1t\":{:.4},\"gbs_1t\":{:.2},\"ms_i8_mt\":{:.4},\"gbs_mt\":{:.2},\"rel_l2_err\":{:.6}}}",
                k.name, k.ms_f32_1t, k.ms_i8_1t, k.gbs_1t, k.ms_i8_mt, k.gbs_mt, k.rel_err
            );
        }
        println!(
            "],\"pp_tokens\":{pp_tokens},\"pp_tok_s\":{pp_tok_s:.2},\"tg_tokens\":{tg_tokens},\"tg_tok_s\":{tg_tok_s:.2}}}"
        );
    } else {
        println!("Ferrite Benchmark Suite (threads = {threads}, matrix = {rows}x{cols})");
        println!(
            "| {:<6} | {:>11} | {:>11} | {:>10} | {:>11} | {:>10} | {:>10} |",
            "Quant",
            "1T F32 (ms)",
            "1T i8 (ms)",
            "1T (GiB/s)",
            "MT i8 (ms)",
            "MT (GiB/s)",
            "Rel L2 Err"
        );
        println!(
            "|--------|-------------|-------------|------------|-------------|------------|------------|"
        );
        for k in &results {
            println!(
                "| {:<6} | {:>11.3} | {:>11.3} | {:>10.2} | {:>11.3} | {:>10.2} | {:>10.5} |",
                k.name, k.ms_f32_1t, k.ms_i8_1t, k.gbs_1t, k.ms_i8_mt, k.gbs_mt, k.rel_err
            );
        }
        println!();
        println!(
            "End-to-End Inference ({}):",
            model_arg.as_deref().unwrap_or("synthetic 2-layer GGUF")
        );
        println!(
            "  pp{pp_tokens} (batched prefill): {:>10.1} tok/s ({:.2} ms)",
            pp_tok_s,
            dt_pp * 1000.0
        );
        println!(
            "  tg{tg_tokens} (token decode):    {:>10.1} tok/s ({:.2} ms)",
            tg_tok_s,
            dt_tg * 1000.0
        );
    }

    Ok(())
}

fn cmd_chat(args: &[String]) -> std::io::Result<()> {
    let limit: usize = flag(args, "-n", 256)?;
    let temperature: f32 = flag(args, "--temp", 0.7)?;
    let top_k: usize = flag(args, "--top-k", 40)?;
    let top_p: f32 = flag(args, "--top-p", 0.9)?;
    let seed: u64 = flag(args, "--seed", 42)?;
    let context: usize = flag(args, "--ctx", model::DEFAULT_CONTEXT)?;
    let threads: usize = flag(args, "--threads", 0)?;
    let mut system_prompt: String = flag(
        args,
        "--system",
        "You are a helpful AI assistant.".to_string(),
    )?;

    let mut model_arg: Option<String> = None;
    let flags = [
        "-n",
        "--temp",
        "--top-k",
        "--top-p",
        "--seed",
        "--ctx",
        "--threads",
        "--system",
    ];
    let mut i = 0;
    while i < args.len() {
        if flags.contains(&args[i].as_str()) {
            i += 2;
            continue;
        }
        match args[i].as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            arg if !arg.starts_with('-') && model_arg.is_none() => {
                model_arg = Some(arg.to_string());
            }
            other => return Err(invalid_input(format!("unknown flag: {other}"))),
        }
        i += 1;
    }

    let model_str =
        model_arg.ok_or_else(|| invalid_input("missing <model> argument for `ferrite chat`"))?;
    let path = lmstudio::resolve(&model_str)?;
    let gguf = Gguf::open(&path)?;
    let model = Model::load(&gguf)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    let mut state = State::with_context(&model.config, context);
    if threads > 0 {
        state.threads = threads;
    }
    let mut sampler = Sampler::new(temperature, top_k, top_p, seed);
    let mut pos = 0usize;

    let sys_text = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n");
    let sys_tokens = tokenizer.encode(&sys_text, true);
    if !sys_tokens.is_empty() {
        model.prefill(&mut state, &sys_tokens, pos)?;
        pos += sys_tokens.len();
    }

    println!("Ferrite Interactive Chat ({})", path.display());
    println!("Commands: /clear, /system <prompt>, /stats, /exit");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut line = String::new();
    let mut last_tok_s = 0.0f64;

    loop {
        print!("\n> ");
        stdout.flush()?;
        line.clear();
        let n = stdin.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "/exit" || trimmed == "/quit" {
            break;
        }
        if trimmed == "/clear" {
            state = State::with_context(&model.config, context);
            if threads > 0 {
                state.threads = threads;
            }
            pos = 0;
            let formatted = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n");
            let st = tokenizer.encode(&formatted, true);
            if !st.is_empty() {
                model.prefill(&mut state, &st, pos)?;
                pos += st.len();
            }
            println!("Conversation context cleared.");
            continue;
        }
        if let Some(new_sys) = trimmed.strip_prefix("/system ") {
            system_prompt = new_sys.trim().to_string();
            state = State::with_context(&model.config, context);
            if threads > 0 {
                state.threads = threads;
            }
            pos = 0;
            let formatted = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n");
            let st = tokenizer.encode(&formatted, true);
            if !st.is_empty() {
                model.prefill(&mut state, &st, pos)?;
                pos += st.len();
            }
            println!("System prompt updated and context reset.");
            continue;
        }
        if trimmed == "/stats" {
            println!(
                "KV cache position: {pos} / {context} tokens | Last decode speed: {last_tok_s:.1} tok/s"
            );
            continue;
        }

        let turn_text = format!("<|im_start|>user\n{trimmed}<|im_end|>\n<|im_start|>assistant\n");
        let turn_tokens = tokenizer.encode(&turn_text, false);
        let t0 = Instant::now();
        model.prefill(&mut state, &turn_tokens, pos)?;
        pos += turn_tokens.len();

        let mut next_token = sampler.sample(&mut state.logits);
        let mut gen_count = 0usize;
        while gen_count < limit && pos < context {
            if Some(next_token) == tokenizer.eos {
                break;
            }
            let piece = tokenizer.piece(next_token);
            stdout.write_all(&piece)?;
            stdout.flush()?;
            model.forward(&mut state, next_token, pos)?;
            pos += 1;
            gen_count += 1;
            next_token = sampler.sample(&mut state.logits);
        }
        println!();
        let dt = t0.elapsed().as_secs_f64().max(1e-9);
        last_tok_s = gen_count as f64 / dt;
    }

    Ok(())
}

fn cmd_split(args: &[String]) -> std::io::Result<()> {
    let shards: usize = flag(args, "--shards", 2)?;
    let max_bytes: usize = flag(args, "--max-bytes", 0)?;

    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--shards" | "--max-bytes" => {
                i += 2;
                continue;
            }
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            a if !a.starts_with('-') => positional.push(a.to_string()),
            other => return Err(invalid_input(format!("unknown flag: {other}"))),
        }
        i += 1;
    }

    if positional.len() != 2 {
        return Err(invalid_input(
            "usage: ferrite split <src.gguf> <out_prefix> [--shards <n>] [--max-bytes <bytes>]",
        ));
    }

    let src_path = lmstudio::resolve(&positional[0])?;
    let out_prefix = std::path::Path::new(&positional[1]);
    let options = ferrite::split::SplitOptions {
        shards,
        max_bytes_per_shard: if max_bytes > 0 { Some(max_bytes) } else { None },
    };
    let written = ferrite::split::split_gguf(&src_path, out_prefix, &options)?;
    for p in &written {
        println!("Wrote shard {}", p.display());
    }
    Ok(())
}

fn cmd_prune(args: &[String]) -> std::io::Result<()> {
    let sparsity: f32 = flag(args, "--sparsity", 0.2)?;
    let tensor_opt: String = flag(args, "--tensor", String::new())?;

    let mut model_arg: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--sparsity" | "--tensor" => {
                i += 2;
                continue;
            }
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            a if !a.starts_with('-') && model_arg.is_none() => {
                model_arg = Some(a.to_string());
            }
            other => return Err(invalid_input(format!("unknown flag: {other}"))),
        }
        i += 1;
    }

    let model_str = model_arg.ok_or_else(|| {
        invalid_input("usage: ferrite prune <model.gguf> --sparsity <0..1> [--tensor <name>]")
    })?;
    let path = lmstudio::resolve(&model_str)?;
    let names_vec: Vec<&str> = if tensor_opt.is_empty() {
        Vec::new()
    } else {
        vec![tensor_opt.as_str()]
    };
    let filter = if names_vec.is_empty() {
        None
    } else {
        Some(names_vec.as_slice())
    };
    let stats = ferrite::prune::ModelPruner::prune_unstructured_inplace(&path, sparsity, filter)?;
    println!(
        "Pruned {} / {} parameters ({:.2}% sparsity across {} tensors)",
        stats.pruned_parameters,
        stats.total_parameters,
        stats.sparsity * 100.0,
        stats.tensors_pruned
    );
    Ok(())
}

fn cmd_edit(args: &[String]) -> std::io::Result<()> {
    if args.len() < 3 {
        return Err(invalid_input(
            "usage: ferrite edit <model.gguf> <info|scale> <tensor> [<factor>]",
        ));
    }
    let path = lmstudio::resolve(&args[0])?;
    let op = args[1].as_str();
    let tensor_name = args[2].as_str();

    let mut editor = ferrite::edit::GgufEditor::open(&path)?;
    match op {
        "info" => {
            let info = editor
                .tensor(tensor_name)
                .ok_or_else(|| invalid_input(format!("tensor {tensor_name} not found")))?;
            println!(
                "Tensor {}: type={:?}, dims={:?}, offset={}",
                info.name, info.ty, info.dims, info.offset
            );
            Ok(())
        }
        "scale" => {
            let factor: f32 = args
                .get(3)
                .ok_or_else(|| invalid_input("scale requires a <factor>"))?
                .parse()
                .map_err(|_| invalid_input("invalid scale factor"))?;
            let mut vals = editor.read_tensor_f32(tensor_name)?;
            for v in &mut vals {
                *v *= factor;
            }
            editor.write_tensor_f32(tensor_name, &vals)?;
            editor.flush()?;
            println!(
                "Scaled {} elements in tensor {} by {}",
                vals.len(),
                tensor_name,
                factor
            );
            Ok(())
        }
        other => Err(invalid_input(format!(
            "unknown edit operation '{other}', expected 'info' or 'scale'"
        ))),
    }
}

fn cmd_watermark(args: &[String]) -> std::io::Result<()> {
    let key: u64 = flag(args, "--key", 0xCAFE_BABE_DEAD_BEEF)?;
    let gamma: f32 = flag(args, "--gamma", 0.5)?;
    let delta: f32 = flag(args, "--delta", 2.0)?;

    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--key" | "--gamma" | "--delta" => {
                i += 2;
                continue;
            }
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            a if !a.starts_with('-') => positional.push(a.to_string()),
            other => return Err(invalid_input(format!("unknown flag: {other}"))),
        }
        i += 1;
    }

    if positional.len() < 3 {
        return Err(invalid_input(
            "usage: ferrite watermark <detect|generate> <model> <text_or_prompt> [--key <u64>]",
        ));
    }

    let mode = positional[0].as_str();
    let path = lmstudio::resolve(&positional[1])?;
    let text = positional[2..].join(" ");
    let gguf = Gguf::open(&path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let cfg = ferrite::watermark::WatermarkConfig::new(key, gamma, delta, 1);

    match mode {
        "detect" => {
            let tokens = tokenizer.encode(&text, false);
            let score = ferrite::watermark::WatermarkDetector::detect(&tokens, &cfg, 4.0);
            println!(
                "Watermark Analysis ({} evaluated tokens): green={}/{} ({:.1}%), z={:.3}, p={:.5} -> {}",
                score.total_tokens,
                score.green_tokens,
                score.total_tokens,
                score.green_fraction * 100.0,
                score.z_score,
                score.p_value,
                if score.is_watermarked {
                    "WATERMARKED"
                } else {
                    "UNWATERMARKED"
                }
            );
            Ok(())
        }
        "generate" => {
            let model = Model::load(&gguf)?;
            let mut state = State::new(&model.config);
            let mut tokens = tokenizer.encode(&text, true);
            model.prefill(&mut state, &tokens, 0)?;
            let mut sampler = Sampler::new(0.8, 40, 0.95, 1);
            let start_pos = tokens.len();
            let mut stdout = std::io::stdout();
            for pos in (start_pos..).take(64) {
                ferrite::watermark::WatermarkEngine::apply_watermark(
                    &mut state.logits,
                    &tokens,
                    &cfg,
                );
                let next = sampler.sample(&mut state.logits);
                if Some(next) == tokenizer.eos {
                    break;
                }
                let piece = tokenizer.piece(next);
                stdout.write_all(&piece)?;
                stdout.flush()?;
                tokens.push(next);
                model.forward(&mut state, next, pos)?;
            }
            println!();
            Ok(())
        }
        other => Err(invalid_input(format!(
            "unknown watermark subcommand '{other}', expected 'detect' or 'generate'"
        ))),
    }
}
