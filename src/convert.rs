//! SafeTensors to GGUF zero-dependency model transpiler.
//!
//! Converts Hugging Face `.safetensors` files and `config.json` into standardized,
//! compliant GGUF format natively without requiring Python, PyTorch, or external crates.

use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Result};
use std::path::Path;

use crate::gguf::{GgmlType, Value};
use crate::synth::Builder;

/// Minimal zero-dependency JSON value representation.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn parse(s: &str) -> Result<Self> {
        let chars: Vec<char> = s.chars().collect();
        let mut idx = 0;
        skip_whitespace(&chars, &mut idx);
        let val = parse_value(&chars, &mut idx)?;
        skip_whitespace(&chars, &mut idx);
        if idx < chars.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("trailing characters after JSON at char {idx}"),
            ));
        }
        Ok(val)
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(entries) => {
                for (k, v) in entries {
                    if k == key {
                        return Some(v);
                    }
                }
                None
            }
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Number(n) => {
                if *n >= 0.0 && *n <= u64::MAX as f64 {
                    Some(*n as u64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(obj) => Some(obj.as_slice()),
            _ => None,
        }
    }
}

fn skip_whitespace(chars: &[char], idx: &mut usize) {
    while *idx < chars.len() && chars[*idx].is_whitespace() {
        *idx += 1;
    }
}

fn parse_value(chars: &[char], idx: &mut usize) -> Result<JsonValue> {
    skip_whitespace(chars, idx);
    if *idx >= chars.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unexpected EOF in JSON",
        ));
    }

    match chars[*idx] {
        'n' => parse_literal(chars, idx, "null", JsonValue::Null),
        't' => parse_literal(chars, idx, "true", JsonValue::Bool(true)),
        'f' => parse_literal(chars, idx, "false", JsonValue::Bool(false)),
        '"' => parse_string(chars, idx).map(JsonValue::String),
        '[' => parse_array(chars, idx),
        '{' => parse_object(chars, idx),
        '-' | '0'..='9' => parse_number(chars, idx),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unexpected char '{other}' at position {idx}"),
        )),
    }
}

fn parse_literal(
    chars: &[char],
    idx: &mut usize,
    expected: &str,
    val: JsonValue,
) -> Result<JsonValue> {
    for exp_char in expected.chars() {
        if *idx >= chars.len() || chars[*idx] != exp_char {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected literal {expected}"),
            ));
        }
        *idx += 1;
    }
    Ok(val)
}

fn parse_string(chars: &[char], idx: &mut usize) -> Result<String> {
    *idx += 1; // skip leading '"'
    let mut out = String::new();
    while *idx < chars.len() {
        let c = chars[*idx];
        *idx += 1;
        match c {
            '"' => return Ok(out),
            '\\' => {
                if *idx >= chars.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "unexpected EOF in string escape",
                    ));
                }
                let esc = chars[*idx];
                *idx += 1;
                match esc {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000C}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        if *idx + 4 > chars.len() {
                            return Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                "unexpected EOF in unicode escape",
                            ));
                        }
                        let hex: String = chars[*idx..*idx + 4].iter().collect();
                        *idx += 4;
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(ch) = char::from_u32(code) {
                                out.push(ch);
                            } else {
                                out.push('\u{FFFD}');
                            }
                        } else {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "invalid hex in unicode escape",
                            ));
                        }
                    }
                    other => out.push(other),
                }
            }
            other => out.push(other),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "unterminated string",
    ))
}

fn parse_number(chars: &[char], idx: &mut usize) -> Result<JsonValue> {
    let start = *idx;
    if *idx < chars.len() && chars[*idx] == '-' {
        *idx += 1;
    }
    while *idx < chars.len() && (chars[*idx].is_ascii_digit() || chars[*idx] == '.') {
        *idx += 1;
    }
    if *idx < chars.len() && (chars[*idx] == 'e' || chars[*idx] == 'E') {
        *idx += 1;
        if *idx < chars.len() && (chars[*idx] == '+' || chars[*idx] == '-') {
            *idx += 1;
        }
        while *idx < chars.len() && chars[*idx].is_ascii_digit() {
            *idx += 1;
        }
    }
    let s: String = chars[start..*idx].iter().collect();
    let num = s.parse::<f64>().map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid number '{s}': {e}"),
        )
    })?;
    Ok(JsonValue::Number(num))
}

fn parse_array(chars: &[char], idx: &mut usize) -> Result<JsonValue> {
    *idx += 1; // skip '['
    let mut items = Vec::new();
    loop {
        skip_whitespace(chars, idx);
        if *idx >= chars.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unterminated array",
            ));
        }
        if chars[*idx] == ']' {
            *idx += 1;
            return Ok(JsonValue::Array(items));
        }
        let val = parse_value(chars, idx)?;
        items.push(val);
        skip_whitespace(chars, idx);
        if *idx < chars.len() && chars[*idx] == ',' {
            *idx += 1;
        } else if *idx < chars.len() && chars[*idx] == ']' {
            *idx += 1;
            return Ok(JsonValue::Array(items));
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected ',' or ']' in array",
            ));
        }
    }
}

fn parse_object(chars: &[char], idx: &mut usize) -> Result<JsonValue> {
    *idx += 1; // skip '{'
    let mut entries = Vec::new();
    loop {
        skip_whitespace(chars, idx);
        if *idx >= chars.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unterminated object",
            ));
        }
        if chars[*idx] == '}' {
            *idx += 1;
            return Ok(JsonValue::Object(entries));
        }
        if chars[*idx] != '"' {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected string key in object at char {idx}"),
            ));
        }
        let key = parse_string(chars, idx)?;
        skip_whitespace(chars, idx);
        if *idx >= chars.len() || chars[*idx] != ':' {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected ':' after object key",
            ));
        }
        *idx += 1; // skip ':'
        let val = parse_value(chars, idx)?;
        entries.push((key, val));
        skip_whitespace(chars, idx);
        if *idx < chars.len() && chars[*idx] == ',' {
            *idx += 1;
        } else if *idx < chars.len() && chars[*idx] == '}' {
            *idx += 1;
            return Ok(JsonValue::Object(entries));
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected ',' or '}' in object",
            ));
        }
    }
}

/// Metadata for a tensor parsed from a SafeTensors file header.
#[derive(Debug, Clone)]
pub struct SafeTensorInfo {
    pub name: String,
    pub dtype: GgmlType,
    pub shape: Vec<u64>,
    pub data_start: u64,
    pub data_end: u64,
}

/// Parsed SafeTensors header.
#[derive(Debug, Clone)]
pub struct SafeTensorsHeader {
    pub header_len: u64,
    pub tensors: Vec<SafeTensorInfo>,
}

impl SafeTensorsHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "file smaller than 8-byte SafeTensors header length",
            ));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        if header_len as usize > bytes.len().saturating_sub(8) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "header length {header_len} exceeds file size {}",
                    bytes.len()
                ),
            ));
        }

        let header_str = std::str::from_utf8(&bytes[8..8 + header_len as usize]).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("header is not valid UTF-8: {e}"),
            )
        })?;

        let json = JsonValue::parse(header_str)?;
        let obj = json.as_object().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "SafeTensors header must be a JSON object",
            )
        })?;

        let mut tensors = Vec::new();
        for (key, val) in obj {
            if key == "__metadata__" {
                continue;
            }
            let dtype_str = val.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing tensor dtype")
            })?;
            let dtype = match dtype_str {
                "F32" => GgmlType::F32,
                "F16" => GgmlType::F16,
                "BF16" => GgmlType::BF16,
                "I32" => GgmlType::I32,
                "I16" => GgmlType::I16,
                "I8" | "U8" => GgmlType::I8,
                "I64" => GgmlType::I64,
                "F64" => GgmlType::F64,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unsupported SafeTensors dtype {other}"),
                    ))
                }
            };

            let shape_arr = val.get("shape").and_then(|v| v.as_array()).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing tensor shape")
            })?;
            let mut shape = Vec::with_capacity(shape_arr.len());
            for d in shape_arr {
                let dim = d.as_u64().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "tensor shape dim must be integer",
                    )
                })?;
                shape.push(dim);
            }

            let offsets_arr = val
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing tensor data_offsets")
                })?;
            if offsets_arr.len() != 2 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "data_offsets must have exactly 2 elements",
                ));
            }
            let data_start = offsets_arr[0].as_u64().unwrap_or(0);
            let data_end = offsets_arr[1].as_u64().unwrap_or(0);

            tensors.push(SafeTensorInfo {
                name: key.clone(),
                dtype,
                shape,
                data_start,
                data_end,
            });
        }

        Ok(Self {
            header_len,
            tensors,
        })
    }
}

/// Map a standard Hugging Face PyTorch tensor name into GGUF tensor naming convention.
pub fn map_hf_tensor_name(name: &str) -> String {
    if name == "model.embed_tokens.weight" {
        return "token_embd.weight".to_string();
    }
    if name == "model.norm.weight" {
        return "output_norm.weight".to_string();
    }
    if name == "lm_head.weight" {
        return "output.weight".to_string();
    }
    if let Some(after) = name.strip_prefix("model.layers.") {
        if let Some((idx_str, sub)) = after.split_once('.') {
            if let Ok(idx) = idx_str.parse::<usize>() {
                let mapped_sub = match sub {
                    "input_layernorm.weight" => "attn_norm.weight",
                    "self_attn.q_proj.weight" => "attn_q.weight",
                    "self_attn.q_proj.bias" => "attn_q.bias",
                    "self_attn.k_proj.weight" => "attn_k.weight",
                    "self_attn.k_proj.bias" => "attn_k.bias",
                    "self_attn.v_proj.weight" => "attn_v.weight",
                    "self_attn.v_proj.bias" => "attn_v.bias",
                    "self_attn.o_proj.weight" => "attn_output.weight",
                    "post_attention_layernorm.weight" => "ffn_norm.weight",
                    "mlp.gate_proj.weight" => "ffn_gate.weight",
                    "mlp.up_proj.weight" => "ffn_up.weight",
                    "mlp.down_proj.weight" => "ffn_down.weight",
                    other => other,
                };
                return format!("blk.{idx}.{mapped_sub}");
            }
        }
    }
    name.to_string()
}

/// Convert a SafeTensors file or model directory to a compliant GGUF file.
pub fn convert_safetensors_to_gguf(
    src_path: &Path,
    dst_path: &Path,
    arch_override: Option<&str>,
) -> Result<()> {
    let (safetensors_files, config_json) = if src_path.is_dir() {
        let mut st_files = Vec::new();
        for entry in fs::read_dir(src_path)? {
            let entry = entry?;
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                st_files.push(p);
            }
        }
        st_files.sort();
        if st_files.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no .safetensors files found in {}", src_path.display()),
            ));
        }
        let cfg_path = src_path.join("config.json");
        let cfg = if cfg_path.exists() {
            let s = fs::read_to_string(&cfg_path)?;
            Some(JsonValue::parse(&s)?)
        } else {
            None
        };
        (st_files, cfg)
    } else {
        let cfg_path = src_path.parent().map(|p| p.join("config.json"));
        let cfg = if let Some(cp) = cfg_path {
            if cp.exists() {
                let s = fs::read_to_string(&cp)?;
                Some(JsonValue::parse(&s)?)
            } else {
                None
            }
        } else {
            None
        };
        (vec![src_path.to_path_buf()], cfg)
    };

    // Parse all tensors from all SafeTensors shards
    struct LoadedTensor {
        name: String,
        dims: Vec<u64>,
        ty: GgmlType,
        bytes: Vec<u8>,
    }

    let mut loaded_tensors = Vec::new();
    let mut layer_indices = BTreeSet::new();
    let mut detected_embedding = 0u64;
    let mut detected_vocab = 0u64;
    let mut detected_ffn = 0u64;

    for file_path in &safetensors_files {
        let raw_file = fs::read(file_path)?;
        let header = SafeTensorsHeader::parse(&raw_file)?;
        let data_base = 8 + header.header_len as usize;

        for t in header.tensors {
            let tensor_data =
                &raw_file[data_base + t.data_start as usize..data_base + t.data_end as usize];
            let gguf_name = map_hf_tensor_name(&t.name);

            // SafeTensors shapes are PyTorch row-major; GGUF shapes are reversed (column-major)
            let mut gguf_dims = t.shape.clone();
            gguf_dims.reverse();

            if gguf_name == "token_embd.weight" && gguf_dims.len() >= 2 {
                detected_embedding = gguf_dims[0];
                detected_vocab = gguf_dims[1];
            } else if gguf_name.ends_with(".attn_norm.weight") && !gguf_dims.is_empty() {
                if detected_embedding == 0 {
                    detected_embedding = gguf_dims[0];
                }
            } else if gguf_name.ends_with(".ffn_gate.weight") && gguf_dims.len() >= 2 {
                detected_ffn = gguf_dims[1];
            }

            if let Some(rest) = gguf_name.strip_prefix("blk.") {
                if let Some((idx_str, _)) = rest.split_once('.') {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        layer_indices.insert(idx);
                    }
                }
            }

            loaded_tensors.push(LoadedTensor {
                name: gguf_name,
                dims: gguf_dims,
                ty: t.dtype,
                bytes: tensor_data.to_vec(),
            });
        }
    }

    // Determine architecture configuration
    let mut arch = arch_override.unwrap_or("llama").to_string();
    let mut block_count = layer_indices.len().max(1) as u32;
    let mut context_length = 2048u32;
    let mut embedding_length = detected_embedding.max(1) as u32;
    let mut feed_forward_length = detected_ffn.max(embedding_length as u64 * 4) as u32;
    let mut head_count = 32u32;
    let mut head_count_kv = 32u32;
    let mut rms_eps = 1e-5f32;
    let mut rope_freq_base = 10000.0f32;
    let mut vocab_size = detected_vocab.max(32000) as u32;

    if let Some(cfg) = &config_json {
        if let Some(mt) = cfg.get("model_type").and_then(|v| v.as_str()) {
            arch = mt.to_string();
        } else if let Some(archs) = cfg.get("architectures").and_then(|v| v.as_array()) {
            if let Some(first) = archs.first().and_then(|v| v.as_str()) {
                if first.to_lowercase().contains("qwen2") {
                    arch = "qwen2".to_string();
                } else if first.to_lowercase().contains("gemma2") {
                    arch = "gemma2".to_string();
                } else {
                    arch = "llama".to_string();
                }
            }
        }

        if let Some(nl) = cfg.get("num_hidden_layers").and_then(|v| v.as_u64()) {
            block_count = nl as u32;
        }
        if let Some(hs) = cfg.get("hidden_size").and_then(|v| v.as_u64()) {
            embedding_length = hs as u32;
        }
        if let Some(is) = cfg.get("intermediate_size").and_then(|v| v.as_u64()) {
            feed_forward_length = is as u32;
        }
        if let Some(nh) = cfg.get("num_attention_heads").and_then(|v| v.as_u64()) {
            head_count = nh as u32;
            head_count_kv = head_count;
        }
        if let Some(nkv) = cfg.get("num_key_value_heads").and_then(|v| v.as_u64()) {
            head_count_kv = nkv as u32;
        }
        if let Some(ctx) = cfg.get("max_position_embeddings").and_then(|v| v.as_u64()) {
            context_length = ctx as u32;
        }
        if let Some(eps) = cfg.get("rms_norm_eps").and_then(|v| v.as_f64()) {
            rms_eps = eps as f32;
        }
        if let Some(rfb) = cfg.get("rope_theta").and_then(|v| v.as_f64()) {
            rope_freq_base = rfb as f32;
        }
        if let Some(vs) = cfg.get("vocab_size").and_then(|v| v.as_u64()) {
            vocab_size = vs as u32;
        }
    }

    // Build GGUF
    let mut builder = Builder::new()
        .meta("general.architecture", Value::String(arch.clone()))
        .meta(
            format!("{arch}.block_count").as_str(),
            Value::U32(block_count),
        )
        .meta(
            format!("{arch}.context_length").as_str(),
            Value::U32(context_length),
        )
        .meta(
            format!("{arch}.embedding_length").as_str(),
            Value::U32(embedding_length),
        )
        .meta(
            format!("{arch}.feed_forward_length").as_str(),
            Value::U32(feed_forward_length),
        )
        .meta(
            format!("{arch}.attention.head_count").as_str(),
            Value::U32(head_count),
        )
        .meta(
            format!("{arch}.attention.head_count_kv").as_str(),
            Value::U32(head_count_kv),
        )
        .meta(
            format!("{arch}.attention.layer_norm_rms_epsilon").as_str(),
            Value::F32(rms_eps),
        )
        .meta(
            format!("{arch}.rope.freq_base").as_str(),
            Value::F32(rope_freq_base),
        )
        .meta("tokenizer.ggml.model", Value::String("llama".into()))
        .meta(
            "tokenizer.ggml.tokens",
            Value::Array(
                (0..vocab_size)
                    .map(|i| Value::String(format!("<tok_{i}>")))
                    .collect(),
            ),
        )
        .meta(
            "tokenizer.ggml.scores",
            Value::Array((0..vocab_size).map(|i| Value::F32(i as f32)).collect()),
        )
        .meta(
            "tokenizer.ggml.token_type",
            Value::Array((0..vocab_size).map(|_| Value::I32(1)).collect()),
        );

    for tensor in loaded_tensors {
        builder = builder.tensor(&tensor.name, &tensor.dims, tensor.ty, tensor.bytes);
    }

    let gguf_bytes = builder.build();
    fs::write(dst_path, gguf_bytes)?;

    Ok(())
}

/// Create a synthetic SafeTensors file for testing.
pub fn create_synthetic_safetensors(
    path: &Path,
    tensors: &[(&str, GgmlType, &[u64], &[u8])],
) -> Result<()> {
    let mut header_obj = Vec::new();
    let mut current_offset = 0u64;
    let mut body = Vec::new();

    for (name, ty, shape, data) in tensors {
        let dtype_str = match ty {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::BF16 => "BF16",
            GgmlType::I32 => "I32",
            GgmlType::I16 => "I16",
            GgmlType::I8 => "I8",
            GgmlType::I64 => "I64",
            GgmlType::F64 => "F64",
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "unsupported type for safetensors",
                ))
            }
        };

        let shape_vals: Vec<JsonValue> =
            shape.iter().map(|&d| JsonValue::Number(d as f64)).collect();
        let start = current_offset;
        let end = current_offset + data.len() as u64;
        let offsets_vals = vec![
            JsonValue::Number(start as f64),
            JsonValue::Number(end as f64),
        ];

        let tensor_entry = JsonValue::Object(vec![
            (
                "dtype".to_string(),
                JsonValue::String(dtype_str.to_string()),
            ),
            ("shape".to_string(), JsonValue::Array(shape_vals)),
            ("data_offsets".to_string(), JsonValue::Array(offsets_vals)),
        ]);

        header_obj.push((name.to_string(), tensor_entry));
        body.extend_from_slice(data);
        current_offset = end;
    }

    let header_json = serialize_json(&JsonValue::Object(header_obj));
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + body.len());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&body);

    fs::write(path, out)
}

fn serialize_json(val: &JsonValue) -> String {
    match val {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => {
            if n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                format!("{}", *n as i64)
            } else {
                format!("{n}")
            }
        }
        JsonValue::String(s) => format!("\"{}\"", crate::server::escape_json(s)),
        JsonValue::Array(arr) => {
            let items: Vec<String> = arr.iter().map(serialize_json).collect();
            format!("[{}]", items.join(","))
        }
        JsonValue::Object(entries) => {
            let kvs: Vec<String> = entries
                .iter()
                .map(|(k, v)| {
                    format!(
                        "\"{}\":{}",
                        crate::server::escape_json(k),
                        serialize_json(v)
                    )
                })
                .collect();
            format!("{{{}}}", kvs.join(","))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_parse_primitives() {
        assert_eq!(JsonValue::parse("null").unwrap(), JsonValue::Null);
        assert_eq!(JsonValue::parse("true").unwrap(), JsonValue::Bool(true));
        assert_eq!(JsonValue::parse("false").unwrap(), JsonValue::Bool(false));
        assert_eq!(JsonValue::parse("123").unwrap(), JsonValue::Number(123.0));
        assert_eq!(JsonValue::parse("-45.5").unwrap(), JsonValue::Number(-45.5));
        assert_eq!(
            JsonValue::parse(r#""hello \"world\"\n""#).unwrap(),
            JsonValue::String("hello \"world\"\n".to_string())
        );
    }

    #[test]
    fn test_json_parse_object_and_array() {
        let json_str = r#"
        {
            "dtype": "F32",
            "shape": [10, 20],
            "data_offsets": [0, 800],
            "valid": true,
            "nested": {
                "key": "value"
            }
        }
        "#;
        let v = JsonValue::parse(json_str).expect("parse json");
        assert_eq!(v.get("dtype").and_then(|x| x.as_str()), Some("F32"));
        assert_eq!(
            v.get("valid").and_then(|x| match x {
                JsonValue::Bool(b) => Some(*b),
                _ => None,
            }),
            Some(true)
        );

        let shape = v.get("shape").and_then(|x| x.as_array()).expect("shape");
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0].as_u64(), Some(10));
        assert_eq!(shape[1].as_u64(), Some(20));

        let nested = v.get("nested").expect("nested");
        assert_eq!(nested.get("key").and_then(|x| x.as_str()), Some("value"));
    }

    #[test]
    fn test_safetensors_header_parse() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_header.safetensors");

        let dummy_data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = dummy_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        create_synthetic_safetensors(
            &file_path,
            &[("model.embed_tokens.weight", GgmlType::F32, &[2, 2], &bytes)],
        )
        .expect("create safetensors");

        let raw = fs::read(&file_path).expect("read");
        let header = SafeTensorsHeader::parse(&raw).expect("parse header");
        assert_eq!(header.tensors.len(), 1);
        assert_eq!(header.tensors[0].name, "model.embed_tokens.weight");
        assert_eq!(header.tensors[0].dtype, GgmlType::F32);
        assert_eq!(header.tensors[0].shape, vec![2, 2]);
        assert_eq!(
            header.tensors[0].data_end - header.tensors[0].data_start,
            16
        );

        fs::remove_file(file_path).ok();
    }
}
