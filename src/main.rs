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

run options:
  -n <count>      tokens to generate (default 128)
  --temp <t>      sampling temperature, 0 for greedy (default 0)
  --top-k <k>     keep only the k most likely tokens (default 40)
  --top-p <p>     nucleus threshold (default 0.95)
  --seed <s>      sampling seed (default 1)
  --ctx <n>       KV cache size in tokens (default 2048)

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

    // Positionals are whatever is left once flags and their values are removed.
    let flags = ["-n", "--temp", "--top-k", "--top-p", "--seed", "--ctx"];
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
    let model = Model::load(&gguf)?;
    let mut state = State::with_context(&model.config, context);
    let mut sampler = Sampler::new(temperature, top_k, top_p, seed);

    let prompt_tokens = tokenizer.encode(&prompt, true);
    if prompt_tokens.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "prompt encoded to no tokens",
        ));
    }
    eprintln!(
        "{} · {} prompt tokens · {} kv cache",
        model.config.arch,
        prompt_tokens.len(),
        human(state.cache_bytes() as u64)
    );

    let mut stdout = std::io::stdout();
    print!("{prompt}");
    stdout.flush().ok();

    let prefill_start = Instant::now();
    for (position, token) in prompt_tokens.iter().enumerate() {
        model.forward(&mut state, *token, position)?;
    }
    let prefill = prefill_start.elapsed();

    // A multi-byte character can straddle two tokens, so bytes accumulate
    // until they form valid UTF-8 rather than printing one piece at a time.
    let mut pending: Vec<u8> = Vec::new();
    let mut position = prompt_tokens.len();
    let mut generated = 0;
    let decode_start = Instant::now();

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
