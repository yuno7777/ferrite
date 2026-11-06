//! Embedded zero-dependency HTTP and Server-Sent Events (SSE) inference daemon.
//!
//! Exposes an OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`,
//! `/v1/models`, `/health`) directly over `std::net::TcpListener` without external crates.

use std::io::{BufRead, BufReader, Read, Result, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::kv::KvCacheType;
use crate::model::{Model, State};
use crate::sample::Sampler;
use crate::template::{ChatMessage, ChatTemplate};
use crate::tokenizer::Tokenizer;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub threads: usize,
    pub context: usize,
    pub kv_cache: KvCacheType,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
            threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4),
            context: 2048,
            kv_cache: KvCacheType::F32,
        }
    }
}

pub struct Server<'a> {
    pub model: &'a Model<'a>,
    pub tokenizer: &'a Tokenizer,
    pub config: ServerConfig,
    pub template: ChatTemplate,
}

#[derive(Debug)]
pub struct RequestParams {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub tfs_z: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: u64,
    pub stream: bool,
    pub tools: Vec<crate::tools::ToolDefinition>,
    pub tool_choice: crate::tools::ToolChoice,
}

impl Default for RequestParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 128,
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.0,
            tfs_z: 1.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 1,
            stream: false,
            tools: Vec::new(),
            tool_choice: crate::tools::ToolChoice::Auto,
        }
    }
}

impl<'a> Server<'a> {
    pub fn new(model: &'a Model<'a>, tokenizer: &'a Tokenizer, config: ServerConfig) -> Self {
        let template = ChatTemplate::default_for_arch(&model.config.arch);
        Self {
            model,
            tokenizer,
            config,
            template,
        }
    }

    pub fn with_template(mut self, template: ChatTemplate) -> Self {
        self.template = template;
        self
    }

    /// Bind to configured address and start listening.
    pub fn listen(&self) -> Result<TcpListener> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        TcpListener::bind(&addr)
    }

    /// Serve connections on `listener` until `shutdown` is signaled.
    pub fn serve_with_shutdown(
        &self,
        listener: &TcpListener,
        shutdown: Arc<AtomicBool>,
    ) -> Result<()> {
        listener.set_nonblocking(true)?;

        while !shutdown.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut stream, _addr)) => {
                    let _ = stream.set_nonblocking(false);
                    let _ = self.handle_client(&mut stream);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Handle one complete HTTP request from a client stream.
    pub fn handle_client(&self, stream: &mut TcpStream) -> Result<()> {
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut request_line = String::new();
        if reader.read_line(&mut request_line)? == 0 {
            return Ok(());
        }

        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 2 {
            return self.write_error(stream, 400, "Bad Request");
        }

        let method = parts[0];
        let path = parts[1];

        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some((k, v)) = trimmed.split_once(':') {
                if k.trim().eq_ignore_ascii_case("content-length") {
                    content_length = v.trim().parse::<usize>().unwrap_or(0);
                }
            }
        }

        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body)?;
        }
        let body_str = String::from_utf8_lossy(&body);

        match (method, path) {
            ("GET", "/health") | ("GET", "/") => {
                let json = format!(
                    "{{\"status\":\"ok\",\"model\":\"{}\"}}\n",
                    escape_json(&self.model.config.arch)
                );
                self.write_json(stream, 200, &json)
            }
            ("GET", "/v1/models") => {
                let json = format!(
                    "{{\"object\":\"list\",\"data\":[{{\"id\":\"{}\",\"object\":\"model\",\"created\":1700000000,\"owned_by\":\"ferrite\"}}]}}\n",
                    escape_json(&self.model.config.arch)
                );
                self.write_json(stream, 200, &json)
            }
            ("POST", "/v1/chat/completions") => {
                let params = parse_request_json_with_template(&body_str, &self.template);
                if params.stream {
                    self.handle_streaming(stream, &params)
                } else {
                    self.handle_non_streaming(stream, &params)
                }
            }
            ("POST", "/v1/completions") => {
                let params = parse_request_json(&body_str);
                if params.stream {
                    self.handle_streaming(stream, &params)
                } else {
                    self.handle_non_streaming(stream, &params)
                }
            }
            ("POST", "/v1/embeddings") => self.handle_embeddings(stream, &body_str),
            _ => self.write_error(stream, 404, "Not Found"),
        }
    }

    fn handle_non_streaming(&self, stream: &mut TcpStream, params: &RequestParams) -> Result<()> {
        let tokens = self.tokenizer.encode(&params.prompt, true);
        if tokens.is_empty() {
            return self.write_error(stream, 400, "Prompt encoded to 0 tokens");
        }

        let mut state = State::with_kv_cache(
            &self.model.config,
            self.config.context,
            self.config.kv_cache,
        );
        state.threads = self.config.threads;

        self.model.prefill(&mut state, &tokens, 0)?;

        let penalties = crate::sample::PenaltyConfig {
            repetition_penalty: params.repetition_penalty,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            last_n: 64,
        };
        let mut sampler = Sampler::new(params.temperature, params.top_k, params.top_p, params.seed)
            .with_min_p(params.min_p)
            .with_tfs(params.tfs_z)
            .with_penalties(penalties);

        let mut generated_tokens = Vec::new();
        let mut completion_text = String::new();
        let mut pending = Vec::new();
        let mut pos = tokens.len();
        let limit = if params.max_tokens == 0 {
            128
        } else {
            params.max_tokens
        };
        let mut history = tokens.clone();

        while generated_tokens.len() < limit && pos < state.context() {
            let tok = sampler.sample_with_history(&mut state.logits, &history);
            if Some(tok) == self.tokenizer.eos {
                break;
            }
            history.push(tok);
            generated_tokens.push(tok);
            pending.extend(self.tokenizer.piece(tok));
            if let Ok(text) = std::str::from_utf8(&pending) {
                completion_text.push_str(text);
                pending.clear();
            }
            pos += 1;
            if pos < state.context() {
                self.model.forward(&mut state, tok, pos)?;
            }
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let (content_json, finish_reason) = if !params.tools.is_empty()
            && params.tool_choice != crate::tools::ToolChoice::None
        {
            if let Some(tool_call) = crate::tools::parse_tool_call(&completion_text, &params.tools)
            {
                (
                    format!(
                        "\"content\":null,\"tool_calls\":[{{\"id\":\"{}\",\"type\":\"function\",\"function\":{{\"name\":\"{}\",\"arguments\":\"{}\"}}}}]",
                        tool_call.id,
                        tool_call.function_name,
                        escape_json(&tool_call.arguments)
                    ),
                    "tool_calls",
                )
            } else {
                (
                    format!("\"content\":\"{}\"", escape_json(&completion_text)),
                    "stop",
                )
            }
        } else {
            (
                format!("\"content\":\"{}\"", escape_json(&completion_text)),
                "stop",
            )
        };

        let json = format!(
            "{{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"created\":{now},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",{content_json}}},\"finish_reason\":\"{finish_reason}\"}}],\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}}}\n",
            escape_json(&self.model.config.arch),
            tokens.len(),
            generated_tokens.len(),
            tokens.len() + generated_tokens.len()
        );

        self.write_json(stream, 200, &json)
    }

    fn handle_streaming(&self, stream: &mut TcpStream, params: &RequestParams) -> Result<()> {
        let tokens = self.tokenizer.encode(&params.prompt, true);
        if tokens.is_empty() {
            return self.write_error(stream, 400, "Prompt encoded to 0 tokens");
        }

        let mut state = State::with_kv_cache(
            &self.model.config,
            self.config.context,
            self.config.kv_cache,
        );
        state.threads = self.config.threads;

        self.model.prefill(&mut state, &tokens, 0)?;

        let penalties = crate::sample::PenaltyConfig {
            repetition_penalty: params.repetition_penalty,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            last_n: 64,
        };
        let mut sampler = Sampler::new(params.temperature, params.top_k, params.top_p, params.seed)
            .with_min_p(params.min_p)
            .with_tfs(params.tfs_z)
            .with_penalties(penalties);

        // Send SSE Header
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        )?;
        stream.flush()?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut pos = tokens.len();
        let mut generated = 0;
        let limit = if params.max_tokens == 0 {
            128
        } else {
            params.max_tokens
        };
        let mut pending = Vec::new();
        let mut history = tokens.clone();
        let mut completion_text = String::new();

        while generated < limit && pos < state.context() {
            let tok = sampler.sample_with_history(&mut state.logits, &history);
            if Some(tok) == self.tokenizer.eos {
                break;
            }
            history.push(tok);
            generated += 1;
            pending.extend(self.tokenizer.piece(tok));
            if let Ok(text) = std::str::from_utf8(&pending) {
                completion_text.push_str(text);
                if params.tools.is_empty() || params.tool_choice == crate::tools::ToolChoice::None {
                    let chunk_json = format!(
                        "data: {{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":{now},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}\n\n",
                        escape_json(&self.model.config.arch),
                        escape_json(text)
                    );
                    stream.write_all(chunk_json.as_bytes())?;
                    stream.flush()?;
                }
                pending.clear();
            }
            pos += 1;
            if pos < state.context() {
                self.model.forward(&mut state, tok, pos)?;
            }
        }

        if !params.tools.is_empty() && params.tool_choice != crate::tools::ToolChoice::None {
            if let Some(tool_call) = crate::tools::parse_tool_call(&completion_text, &params.tools)
            {
                let tool_chunk = format!(
                    "data: {{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":{now},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{{\"index\":0,\"id\":\"{}\",\"type\":\"function\",\"function\":{{\"name\":\"{}\",\"arguments\":\"{}\"}}}}]}},\"finish_reason\":null}}]}}\n\n",
                    escape_json(&self.model.config.arch),
                    tool_call.id,
                    tool_call.function_name,
                    escape_json(&tool_call.arguments)
                );
                stream.write_all(tool_chunk.as_bytes())?;
                stream.flush()?;

                let final_json = format!(
                    "data: {{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":{now},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\n",
                    escape_json(&self.model.config.arch)
                );
                stream.write_all(final_json.as_bytes())?;
                stream.write_all(b"data: [DONE]\n\n")?;
                stream.flush()?;
                return Ok(());
            } else {
                let content_chunk = format!(
                    "data: {{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":{now},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}\n\n",
                    escape_json(&self.model.config.arch),
                    escape_json(&completion_text)
                );
                stream.write_all(content_chunk.as_bytes())?;
                stream.flush()?;
            }
        }

        // Final event
        let final_json = format!(
            "data: {{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":{now},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n",
            escape_json(&self.model.config.arch)
        );
        stream.write_all(final_json.as_bytes())?;
        stream.write_all(b"data: [DONE]\n\n")?;
        stream.flush()?;

        Ok(())
    }

    fn write_json(&self, stream: &mut TcpStream, status: u16, json: &str) -> Result<()> {
        let status_text = match status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            _ => "Error",
        };
        write!(
            stream,
            "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            json.len(),
            json
        )?;
        stream.flush()
    }

    fn write_error(&self, stream: &mut TcpStream, status: u16, msg: &str) -> Result<()> {
        let json = format!(
            "{{\"error\":{{\"message\":\"{}\",\"code\":{status}}}}}\n",
            escape_json(msg)
        );
        self.write_json(stream, status, &json)
    }

    fn handle_embeddings(&self, stream: &mut TcpStream, body: &str) -> Result<()> {
        let inputs = extract_string_or_array(body, "input");
        if inputs.is_empty() {
            return self.write_error(stream, 400, "Missing 'input' in request");
        }

        let engine = crate::embed::EmbedEngine::new(self.model, crate::embed::Pooling::Mean, true);
        let mut state = State::with_kv_cache(
            &self.model.config,
            self.config.context,
            self.config.kv_cache,
        );
        state.threads = self.config.threads;

        let mut data_items = Vec::new();
        let mut total_tokens = 0;

        for (idx, text) in inputs.iter().enumerate() {
            let tokens = self.tokenizer.encode(text, true);
            total_tokens += tokens.len();
            let vector = engine.embed(&mut state, &tokens)?;
            let floats_str = vector
                .iter()
                .map(|f| format!("{f:.6}"))
                .collect::<Vec<_>>()
                .join(",");
            data_items.push(format!(
                r#"{{"object":"embedding","index":{idx},"embedding":[{floats_str}]}}"#
            ));
        }

        let json = format!(
            r#"{{"object":"list","data":[{}],"model":"ferrite-model","usage":{{"prompt_tokens":{total_tokens},"total_tokens":{total_tokens}}}}}"#,
            data_items.join(",")
        );

        self.write_json(stream, 200, &json)
    }
}

/// Escape string for JSON serialization.
pub fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 32 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Minimal zero-dependency JSON extractor for OpenAI chat/completions requests.
pub fn parse_request_json(json: &str) -> RequestParams {
    let mut params = RequestParams {
        max_tokens: 128,
        temperature: 0.0,
        top_p: 0.95,
        top_k: 40,
        min_p: 0.0,
        tfs_z: 1.0,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        seed: 1,
        stream: false,
        prompt: String::new(),
        tools: Vec::new(),
        tool_choice: crate::tools::ToolChoice::Auto,
    };

    if let Some(val) = extract_number(json, "max_tokens") {
        params.max_tokens = val as usize;
    }
    if let Some(val) = extract_number(json, "temperature") {
        params.temperature = val as f32;
    }
    if let Some(val) = extract_number(json, "top_p") {
        params.top_p = val as f32;
    }
    if let Some(val) = extract_number(json, "top_k") {
        params.top_k = val as usize;
    }
    if let Some(val) = extract_number(json, "min_p") {
        params.min_p = val as f32;
    }
    if let Some(val) = extract_number(json, "tfs_z").or_else(|| extract_number(json, "tfs")) {
        params.tfs_z = val as f32;
    }
    if let Some(val) = extract_number(json, "repetition_penalty")
        .or_else(|| extract_number(json, "repeat_penalty"))
    {
        params.repetition_penalty = val as f32;
    }
    if let Some(val) = extract_number(json, "frequency_penalty") {
        params.frequency_penalty = val as f32;
    }
    if let Some(val) = extract_number(json, "presence_penalty") {
        params.presence_penalty = val as f32;
    }
    if let Some(val) = extract_number(json, "seed") {
        params.seed = val as u64;
    }
    if let Some(val) = extract_bool(json, "stream") {
        params.stream = val;
    }

    if let Some(p) = extract_string(json, "prompt") {
        params.prompt = p;
    } else {
        // Extract messages array: [{"role":"...", "content":"..."}]
        let messages = extract_messages(json);
        if !messages.is_empty() {
            let mut assembled = String::new();
            for (role, content) in messages {
                if !assembled.is_empty() {
                    assembled.push('\n');
                }
                assembled.push_str(&format!("{role}: {content}"));
            }
            assembled.push_str("\nassistant:");
            params.prompt = assembled;
        }
    }

    params.tools = crate::tools::parse_tools(json);
    params.tool_choice = crate::tools::parse_tool_choice(json);

    params
}

/// Extract request parameters and format chat messages using a specified chat template.
pub fn parse_request_json_with_template(json: &str, template: &ChatTemplate) -> RequestParams {
    let mut params = parse_request_json(json);
    let messages = extract_messages(json);
    if !messages.is_empty() {
        let chat_msgs: Vec<ChatMessage> = messages
            .into_iter()
            .map(|(r, c)| ChatMessage::new(r, c))
            .collect();
        params.prompt = template.render(&chat_msgs, true, None, None);
    }
    if !params.tools.is_empty() && params.tool_choice != crate::tools::ToolChoice::None {
        params
            .prompt
            .push_str(&crate::tools::format_tools_system_prompt(
                &params.tools,
                &params.tool_choice,
            ));
    }
    params
}

fn extract_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let key_pos = json.find(&pattern)?;
    let after_key = &json[key_pos + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let content = &after_colon[1..];
    let mut result = String::new();
    let mut chars = content.chars();
    while let Some(c) = chars.next() {
        if c == '"' {
            return Some(result);
        }
        if c == '\\' {
            match chars.next()? {
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                other => result.push(other),
            }
        } else {
            result.push(c);
        }
    }
    None
}

fn extract_bool(json: &str, key: &str) -> Option<bool> {
    let pattern = format!("\"{key}\"");
    let key_pos = json.find(&pattern)?;
    let after_key = &json[key_pos + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if after_colon.starts_with("true") {
        Some(true)
    } else if after_colon.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn extract_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{key}\"");
    let key_pos = json.find(&pattern)?;
    let after_key = &json[key_pos + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    let mut end = 0;
    for (i, c) in after_colon.char_indices() {
        if c.is_ascii_digit() || c == '.' || c == '-' {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if end > 0 {
        after_colon[..end].parse::<f64>().ok()
    } else {
        None
    }
}

fn extract_messages(json: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let pattern = "\"messages\"";
    let Some(pos) = json.find(pattern) else {
        return out;
    };
    let after = &json[pos + pattern.len()..];
    let Some(bracket) = after.find('[') else {
        return out;
    };
    let Some(end_bracket) = after.find(']') else {
        return out;
    };
    let array_content = &after[bracket + 1..end_bracket];

    // Split objects by `{`
    for obj in array_content.split('{') {
        let Some(obj_end) = obj.find('}') else {
            continue;
        };
        let obj_slice = &obj[..obj_end];
        let role = extract_string(obj_slice, "role").unwrap_or_else(|| "user".into());
        if let Some(content) = extract_string(obj_slice, "content") {
            out.push((role, content));
        }
    }
    out
}

fn extract_string_or_array(json: &str, key: &str) -> Vec<String> {
    if let Some(s) = extract_string(json, key) {
        return vec![s];
    }
    let pattern = format!("\"{key}\"");
    let Some(key_pos) = json.find(&pattern) else {
        return Vec::new();
    };
    let after_key = &json[key_pos + pattern.len()..];
    let Some(colon_pos) = after_key.find(':') else {
        return Vec::new();
    };
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if !after_colon.starts_with('[') {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut in_str = false;
    let mut cur = String::new();
    let mut chars = after_colon[1..].chars();
    while let Some(c) = chars.next() {
        if in_str {
            if c == '"' {
                in_str = false;
                out.push(cur.clone());
                cur.clear();
            } else if c == '\\' {
                if let Some(escaped) = chars.next() {
                    match escaped {
                        '"' => cur.push('"'),
                        '\\' => cur.push('\\'),
                        'n' => cur.push('\n'),
                        't' => cur.push('\t'),
                        other => cur.push(other),
                    }
                }
            } else {
                cur.push(c);
            }
        } else if c == '"' {
            in_str = true;
        } else if c == ']' {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escape_handles_quotes_and_newlines() {
        assert_eq!(escape_json("hello \"world\"\n"), "hello \\\"world\\\"\\n");
    }

    #[test]
    fn parse_chat_request_extracts_messages_and_parameters() {
        let payload = r#"{
            "model": "llama",
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": 64,
            "temperature": 0.7,
            "stream": true
        }"#;

        let params = parse_request_json(payload);
        assert_eq!(params.max_tokens, 64);
        assert!((params.temperature - 0.7).abs() < 1e-4);
        assert!(params.stream);
        assert!(params.prompt.contains("What is 2+2?"));
    }

    #[test]
    fn parse_completions_extracts_prompt_field() {
        let payload = r#"{"prompt": "Say hi!", "stream": false}"#;
        let params = parse_request_json(payload);
        assert_eq!(params.prompt, "Say hi!");
        assert!(!params.stream);
    }
}
