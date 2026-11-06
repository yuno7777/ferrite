use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use ferrite::gguf::Gguf;
use ferrite::lmstudio;
use ferrite::model::{self, Model, State};
use ferrite::sample::Sampler;
use ferrite::Tokenizer;

const USAGE: &str = "\
ferrite — pure-Rust GGUF inference

usage:
  ferrite list                      models LM Studio has downloaded
  ferrite where                     directories being searched
  ferrite info <model|path> [-t]    header, hyperparameters, quant mix
                                    -t also dumps the tensor table
  ferrite tokenize <model> <text>   encode text, show ids and pieces
  ferrite run <model> <prompt>      generate, streaming to stdout
  ferrite serve <model>             serve HTTP/SSE API daemon

run/serve options:
  -n <count>      tokens to generate (default 128)
  --temp <t>      sampling temperature, 0 for greedy (default 0)
  --top-k <k>     keep only the k most likely tokens (default 40)
  --top-p <p>     nucleus threshold (default 0.95)
  --seed <s>      sampling seed (default 1)
  --ctx <n>       KV cache size in tokens (default 2048)
  --threads <n>   workers per matvec (default: machine parallelism)
  --host <host>   host interface to bind (default: 127.0.0.1)
  --port <port>   port to bind (default: 8080)
  --draft <model> draft model for speculative decoding acceleration
  --spec-k <k>    speculative candidate count (default 3)
  --lora <adapter> apply GGUF LoRA adapter dynamically

<model> is a path to a .gguf, or any substring of an id from `ferrite list`.
Override the search root with FERRITE_MODELS_DIR.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("list") => cmd_list(),
        Some("where") => cmd_where(),
        Some("info") => cmd_info(&args[1..]),
        Some("tokenize") => cmd_tokenize(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("serve") => cmd_serve(&args[1..]),
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
    let seed: u64 = flag(args, "--seed", 1)?;
    let context: usize = flag(args, "--ctx", model::DEFAULT_CONTEXT)?;
    let threads: usize = flag(args, "--threads", 0)?;

    let draft_query: Option<String> = {
        let at = args.iter().position(|a| a == "--draft");
        at.and_then(|i| args.get(i + 1).cloned())
    };
    let spec_k: usize = flag(args, "--spec-k", 3)?;

    let lora_query: Option<String> = {
        let at = args.iter().position(|a| a == "--lora");
        at.and_then(|i| args.get(i + 1).cloned())
    };

    // Positionals are whatever is left once flags and their values are removed.
    let flags = [
        "-n",
        "--temp",
        "--top-k",
        "--top-p",
        "--seed",
        "--ctx",
        "--threads",
        "--draft",
        "--spec-k",
        "--lora",
    ];
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
    let mut state = State::with_context(&model.config, context);
    if threads > 0 {
        state.threads = threads;
    }
    let mut sampler = Sampler::new(temperature, top_k, top_p, seed);

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
        eprintln!(
            "target: {} · draft: {} (k={spec_k}) · {} prompt tokens · {} kv cache · {} threads",
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
    model.prefill(&mut state, &prompt_tokens, 0)?;
    if let Some((draft_model, draft_state)) = &mut draft_bundle {
        draft_model.prefill(draft_state, &prompt_tokens, 0)?;
    }
    let prefill = prefill_start.elapsed();

    // A multi-byte character can straddle two tokens, so bytes accumulate
    // until they form valid UTF-8 rather than printing one piece at a time.
    let mut pending: Vec<u8> = Vec::new();
    let mut position = prompt_tokens.len();
    let mut generated = 0;
    let decode_start = Instant::now();

    if let Some((draft_model, draft_state)) = &mut draft_bundle {
        let mut engine = ferrite::speculative::SpeculativeEngine::new(&model, draft_model, spec_k);
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
        println!();
        let decode = decode_start.elapsed();
        eprintln!(
            "\nprefill {:.2}s ({:.1} tok/s) · decode {generated} tokens in {:.2}s ({:.1} tok/s) · acceptance rate {:.1}% ({:.2} tok/target-step)",
            prefill.as_secs_f64(),
            prompt_tokens.len() as f64 / prefill.as_secs_f64().max(1e-9),
            decode.as_secs_f64(),
            generated as f64 / decode.as_secs_f64().max(1e-9),
            engine.stats.acceptance_rate() * 100.0,
            engine.stats.tokens_per_batch(),
        );
    } else {
        while generated < limit && position < state.context() {
            let token = sampler.sample(&mut state.logits);
            if Some(token) == tokenizer.eos {
                break;
            }
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
