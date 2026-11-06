//! Structured Function Calling / Tool Use Engine for OpenAI-compatible tools schema.
//!
//! Provides zero-dependency parsing of `tools` and `tool_choice` in `/v1/chat/completions`,
//! system prompt synthesis, tool call detection from model output, and JSON validation.

use crate::grammar::JsonGrammar;

/// A tool definition in OpenAI format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDefinition {
    pub tool_type: String,
    pub name: String,
    pub description: String,
    pub parameters: String,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: impl Into<String>,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            name: name.into(),
            description: description.into(),
            parameters: parameters.into(),
        }
    }
}

/// Tool choice mode from request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

/// A parsed tool call emitted by the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub tool_type: String,
    pub function_name: String,
    pub arguments: String,
}

/// Generate a unique pseudo-random tool call ID.
pub fn generate_tool_call_id(name: &str, seed: u64) -> String {
    let mut h = 0xcbf29ce484222325u64 ^ seed;
    for b in name.bytes() {
        h = (h ^ (b as u64)).wrapping_mul(0x100000001b3);
    }
    format!("call_{:012x}", h & 0xffffffffffff)
}

/// Extract tools from the request JSON string.
pub fn parse_tools(json: &str) -> Vec<ToolDefinition> {
    let mut tools = Vec::new();
    let Some(tools_pos) = json.find("\"tools\"") else {
        return tools;
    };
    let after_key = &json[tools_pos + 7..];
    let Some(colon_pos) = after_key.find(':') else {
        return tools;
    };
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if !after_colon.starts_with('[') {
        return tools;
    }

    // Split array elements approximately by objects at depth 1
    let array_str = extract_balanced(after_colon, '[', ']').unwrap_or("");
    for obj_str in extract_array_objects(array_str) {
        let name = extract_nested_string(&obj_str, "function", "name")
            .or_else(|| extract_field_string(&obj_str, "name"))
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let desc = extract_nested_string(&obj_str, "function", "description")
            .or_else(|| extract_field_string(&obj_str, "description"))
            .unwrap_or_default();
        let params = extract_nested_raw_object(&obj_str, "function", "parameters")
            .or_else(|| extract_raw_object(&obj_str, "parameters"))
            .unwrap_or_else(|| "{}".to_string());

        tools.push(ToolDefinition {
            tool_type: "function".to_string(),
            name,
            description: desc,
            parameters: params,
        });
    }

    tools
}

/// Extract tool choice from request JSON.
pub fn parse_tool_choice(json: &str) -> ToolChoice {
    let Some(pos) = json.find("\"tool_choice\"") else {
        return ToolChoice::Auto;
    };
    let after_key = &json[pos + 13..];
    let Some(colon_pos) = after_key.find(':') else {
        return ToolChoice::Auto;
    };
    let val_str = after_key[colon_pos + 1..].trim_start();

    if val_str.starts_with('"') {
        let text = extract_field_string_from(val_str).unwrap_or_default();
        match text.as_str() {
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            "auto" => ToolChoice::Auto,
            named => ToolChoice::Named(named.to_string()),
        }
    } else if val_str.starts_with('{') {
        if let Some(func_name) = extract_nested_string(val_str, "function", "name") {
            ToolChoice::Named(func_name)
        } else {
            ToolChoice::Auto
        }
    } else {
        ToolChoice::Auto
    }
}

/// Format tool instructions to inject into system prompt.
pub fn format_tools_system_prompt(tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String {
    if tools.is_empty() || *tool_choice == ToolChoice::None {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("\n\n[AVAILABLE_TOOLS]\n");
    for t in tools {
        out.push_str(&format!(
            "- Function: {}\n  Description: {}\n  Parameters: {}\n",
            t.name, t.description, t.parameters
        ));
    }
    out.push_str("[/AVAILABLE_TOOLS]\n\n");
    out.push_str("To invoke a tool, respond with a JSON object: {\"name\": \"<function_name>\", \"arguments\": {<args>}}");
    match tool_choice {
        ToolChoice::Required => {
            out.push_str("\nYou MUST call one of the available functions.");
        }
        ToolChoice::Named(name) => {
            out.push_str(&format!("\nYou MUST call the function '{name}'."));
        }
        _ => {}
    }
    out
}

/// Attempt to parse a tool call from the generated text.
pub fn parse_tool_call(output: &str, tools: &[ToolDefinition]) -> Option<ToolCall> {
    if tools.is_empty() {
        return None;
    }

    // Try finding JSON object in text
    let trimmed = output.trim();
    let start_idx = trimmed.find('{')?;
    let json_candidate = &trimmed[start_idx..];
    let balanced = extract_balanced(json_candidate, '{', '}')?;

    // Check for "name" field
    let name = extract_field_string(balanced, "name")?;
    let tool_match = tools.iter().find(|t| t.name == name)?;

    // Extract arguments
    let args = if let Some(args_raw) = extract_raw_object(balanced, "arguments") {
        args_raw
    } else if let Some(args_str) = extract_field_string(balanced, "arguments") {
        args_str
    } else {
        "{}".to_string()
    };

    // Validate that arguments is valid JSON using JsonGrammar
    let valid_args = if is_valid_json(&args) {
        args
    } else {
        "{}".to_string()
    };

    Some(ToolCall {
        id: generate_tool_call_id(&tool_match.name, 42),
        tool_type: "function".to_string(),
        function_name: tool_match.name.clone(),
        arguments: valid_args,
    })
}

/// Validate that a string conforms to JSON syntax using the zero-dependency JsonGrammar.
pub fn is_valid_json(s: &str) -> bool {
    let mut grammar = JsonGrammar::new();
    grammar.advance(s.as_bytes()) && grammar.is_accepting()
}

// ---------------------------------------------------------------------------
// String and JSON Extraction Helpers
// ---------------------------------------------------------------------------

fn extract_balanced(s: &str, open: char, close: char) -> Option<&str> {
    let mut depth = 0;
    let mut start = None;
    for (i, c) in s.char_indices() {
        if c == open {
            if depth == 0 {
                start = Some(i);
            }
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                if let Some(st) = start {
                    return Some(&s[st..=i]);
                }
            }
        }
    }
    None
}

fn extract_array_objects(s: &str) -> Vec<String> {
    let mut objects = Vec::new();
    let mut depth = 0;
    let mut start = None;

    for (i, c) in s.char_indices() {
        if c == '{' {
            if depth == 0 {
                start = Some(i);
            }
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                if let Some(st) = start {
                    objects.push(s[st..=i].to_string());
                    start = None;
                }
            }
        }
    }
    objects
}

fn extract_field_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let colon = after.find(':')?;
    extract_field_string_from(&after[colon + 1..])
}

fn extract_field_string_from(after_colon: &str) -> Option<String> {
    let trimmed = after_colon.trim_start();
    if !trimmed.starts_with('"') {
        return None;
    }
    let content = &trimmed[1..];
    let mut res = String::new();
    let mut chars = content.chars();
    while let Some(c) = chars.next() {
        if c == '"' {
            return Some(res);
        }
        if c == '\\' {
            match chars.next()? {
                '"' => res.push('"'),
                '\\' => res.push('\\'),
                'n' => res.push('\n'),
                'r' => res.push('\r'),
                't' => res.push('\t'),
                other => {
                    res.push('\\');
                    res.push(other);
                }
            }
        } else {
            res.push(c);
        }
    }
    None
}

fn extract_nested_string(json: &str, parent: &str, child: &str) -> Option<String> {
    let raw = extract_raw_object(json, parent)?;
    extract_field_string(&raw, child)
}

fn extract_raw_object(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let colon = after.find(':')?;
    let candidate = after[colon + 1..].trim_start();
    if candidate.starts_with('{') {
        extract_balanced(candidate, '{', '}').map(|s| s.to_string())
    } else {
        None
    }
}

fn extract_nested_raw_object(json: &str, parent: &str, child: &str) -> Option<String> {
    let raw = extract_raw_object(json, parent)?;
    extract_raw_object(&raw, child)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tools_and_tool_choice() {
        let json = r#"{
            "model": "ferrite-model",
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        }"#;

        let tools = parse_tools(json);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].description, "Get current weather for location");
        assert!(tools[0].parameters.contains("\"location\""));

        let choice = parse_tool_choice(json);
        assert_eq!(choice, ToolChoice::Auto);
    }

    #[test]
    fn test_parse_tool_call_from_output() {
        let tools = vec![ToolDefinition::new(
            "get_weather",
            "Get weather",
            r#"{"type":"object"}"#,
        )];

        let output = r#"I will check the weather. {"name": "get_weather", "arguments": {"location": "Tokyo"}}"#;
        let call = parse_tool_call(output, &tools).expect("should parse tool call");
        assert_eq!(call.function_name, "get_weather");
        assert!(call.arguments.contains("Tokyo"));
        assert!(call.id.starts_with("call_"));
    }

    #[test]
    fn test_is_valid_json() {
        assert!(is_valid_json(r#"{"location": "Tokyo", "days": 3}"#));
        assert!(is_valid_json(r#"{"nested": {"valid": true}}"#));
        assert!(!is_valid_json(r#"{"invalid": "#));
    }
}
