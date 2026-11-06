//! Zero-dependency Jinja/Mini-Template Chat Engine for GGUF models.
//!
//! Evaluates GGUF `tokenizer.chat_template` definitions to convert OpenAI-format
//! message arrays (`system`, `user`, `assistant`) into exact model prompts
//! for LLaMA-3, ChatML, Mistral, Gemma, Qwen, and custom architectures.

use std::collections::HashMap;

/// A single message in a chat conversation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Evaluated Jinja AST nodes.
#[derive(Clone, Debug)]
enum Node {
    Text(String),
    Output {
        expr: Expr,
        trim_left: bool,
        trim_right: bool,
    },
    If {
        branches: Vec<(Expr, Vec<Node>)>,
        else_branch: Option<Vec<Node>>,
    },
    For {
        var_name: String,
        iterable: Expr,
        body: Vec<Node>,
    },
    Set {
        var_name: String,
        expr: Expr,
    },
}

#[derive(Clone, Debug)]
enum Expr {
    LiteralString(String),
    LiteralBool(bool),
    LiteralInt(i64),
    Variable(String),
    Member(Box<Expr>, String),
    Concat(Vec<Expr>),
    Equal(Box<Expr>, Box<Expr>),
    NotEqual(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    IsDefined(Box<Expr>),
    IsNotDefined(Box<Expr>),
    Trim(Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Message(ChatMessage),
    List(Vec<ChatMessage>),
    Loop {
        first: bool,
        last: bool,
        index: usize,
        index0: usize,
    },
}

impl Value {
    fn is_truthy(&self) -> bool {
        match self {
            Value::None => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Str(s) => !s.is_empty(),
            Value::Message(_) => true,
            Value::List(l) => !l.is_empty(),
            Value::Loop { .. } => true,
        }
    }

    fn as_str(&self) -> String {
        match self {
            Value::None => String::new(),
            Value::Bool(b) => b.to_string(),
            Value::Int(i) => i.to_string(),
            Value::Str(s) => s.clone(),
            Value::Message(m) => m.content.clone(),
            Value::List(_) => "[list]".into(),
            Value::Loop { .. } => "[loop]".into(),
        }
    }
}

/// Zero-dependency compiled chat template renderer.
#[derive(Clone, Debug)]
pub struct ChatTemplate {
    nodes: Vec<Node>,
    raw_template: String,
}

impl ChatTemplate {
    /// Parse a Jinja2 chat template string into an executable AST.
    pub fn parse(template_str: &str) -> Self {
        let mut tokens = tokenize(template_str);
        let nodes = parse_nodes(&mut tokens, &[]);
        Self {
            nodes,
            raw_template: template_str.to_string(),
        }
    }

    /// Default built-in ChatML template (Qwen, Yi, DeepSeek).
    pub fn chatml() -> Self {
        Self::parse(
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
        )
    }

    /// Default built-in LLaMA-3 / 3.1 / 3.2 template.
    pub fn llama3() -> Self {
        Self::parse(
            "{{- bos_token }}{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{'<|start_header_id|>assistant<|end_header_id|>\n\n'}}{% endif %}"
        )
    }

    /// Default built-in LLaMA-2 template (`[INST]`).
    pub fn llama2() -> Self {
        Self::parse(
            "{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\n' + message['content'] + '\n<</SYS>>\n\n' }}{% endif %}{% endfor %}"
        )
    }

    /// Default built-in Gemma 1 & 2 template (`<start_of_turn>`).
    pub fn gemma() -> Self {
        Self::parse(
            "{% for message in messages %}{{'<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>assistant\n'}}{% endif %}"
        )
    }

    /// Select an optimal template based on architecture name and optional special tokens.
    pub fn default_for_arch(arch: &str) -> Self {
        match arch.to_lowercase().as_str() {
            "llama" | "llama2" => Self::llama3(),
            "qwen" | "qwen2" | "qwen2.5" => Self::chatml(),
            "gemma" | "gemma2" => Self::gemma(),
            _ => Self::chatml(),
        }
    }

    /// Access the underlying raw template string.
    pub fn raw(&self) -> &str {
        &self.raw_template
    }

    /// Render a list of chat messages into a formatted prompt.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
    ) -> String {
        let mut env = HashMap::new();
        env.insert("messages".to_string(), Value::List(messages.to_vec()));
        env.insert(
            "add_generation_prompt".to_string(),
            Value::Bool(add_generation_prompt),
        );
        if let Some(bos) = bos_token {
            env.insert("bos_token".to_string(), Value::Str(bos.to_string()));
        }
        if let Some(eos) = eos_token {
            env.insert("eos_token".to_string(), Value::Str(eos.to_string()));
        }

        let mut out = String::new();
        let mut trim_leading = false;
        render_nodes(&self.nodes, &mut env, &mut out, &mut trim_leading);
        out
    }
}

// -------------------------------------------------------------------- AST Execution

fn render_nodes(
    nodes: &[Node],
    env: &mut HashMap<String, Value>,
    out: &mut String,
    trim_leading: &mut bool,
) {
    for node in nodes {
        match node {
            Node::Text(txt) => {
                let piece = if *trim_leading {
                    *trim_leading = false;
                    txt.trim_start()
                } else {
                    txt.as_str()
                };
                out.push_str(piece);
            }
            Node::Output {
                expr,
                trim_left,
                trim_right,
            } => {
                if *trim_left {
                    let trimmed_len = out.trim_end().len();
                    out.truncate(trimmed_len);
                }
                let val = eval_expr(expr, env);
                let rendered = val.as_str();
                let piece = if *trim_leading {
                    *trim_leading = false;
                    rendered.trim_start()
                } else {
                    rendered.as_str()
                };
                out.push_str(piece);
                if *trim_right {
                    let trimmed_len = out.trim_end().len();
                    out.truncate(trimmed_len);
                    *trim_leading = true;
                }
            }
            Node::If {
                branches,
                else_branch,
            } => {
                let mut matched = false;
                for (cond, body) in branches {
                    if eval_expr(cond, env).is_truthy() {
                        render_nodes(body, env, out, trim_leading);
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    if let Some(eb) = else_branch {
                        render_nodes(eb, env, out, trim_leading);
                    }
                }
            }
            Node::For {
                var_name,
                iterable,
                body,
            } => {
                let list = match eval_expr(iterable, env) {
                    Value::List(l) => l,
                    _ => Vec::new(),
                };
                let total = list.len();
                for (i, msg) in list.into_iter().enumerate() {
                    env.insert(var_name.clone(), Value::Message(msg));
                    env.insert(
                        "loop".to_string(),
                        Value::Loop {
                            first: i == 0,
                            last: i + 1 == total,
                            index: i + 1,
                            index0: i,
                        },
                    );
                    render_nodes(body, env, out, trim_leading);
                }
                env.remove(var_name);
                env.remove("loop");
            }
            Node::Set { var_name, expr } => {
                let val = eval_expr(expr, env);
                env.insert(var_name.clone(), val);
            }
        }
    }
}

fn eval_expr(expr: &Expr, env: &HashMap<String, Value>) -> Value {
    match expr {
        Expr::LiteralString(s) => Value::Str(s.clone()),
        Expr::LiteralBool(b) => Value::Bool(*b),
        Expr::LiteralInt(i) => Value::Int(*i),
        Expr::Variable(var) => env.get(var).cloned().unwrap_or(Value::None),
        Expr::Member(parent, member) => {
            let p = eval_expr(parent, env);
            match p {
                Value::Message(msg) => match member.as_str() {
                    "role" => Value::Str(msg.role),
                    "content" => Value::Str(msg.content),
                    _ => Value::None,
                },
                Value::Loop {
                    first,
                    last,
                    index,
                    index0,
                } => match member.as_str() {
                    "first" => Value::Bool(first),
                    "last" => Value::Bool(last),
                    "index" => Value::Int(index as i64),
                    "index0" => Value::Int(index0 as i64),
                    _ => Value::None,
                },
                _ => Value::None,
            }
        }
        Expr::Concat(parts) => {
            let mut s = String::new();
            for part in parts {
                s.push_str(&eval_expr(part, env).as_str());
            }
            Value::Str(s)
        }
        Expr::Equal(a, b) => {
            let va = eval_expr(a, env);
            let vb = eval_expr(b, env);
            Value::Bool(va == vb)
        }
        Expr::NotEqual(a, b) => {
            let va = eval_expr(a, env);
            let vb = eval_expr(b, env);
            Value::Bool(va != vb)
        }
        Expr::And(a, b) => {
            let va = eval_expr(a, env).is_truthy();
            let vb = eval_expr(b, env).is_truthy();
            Value::Bool(va && vb)
        }
        Expr::Or(a, b) => {
            let va = eval_expr(a, env).is_truthy();
            let vb = eval_expr(b, env).is_truthy();
            Value::Bool(va || vb)
        }
        Expr::Not(a) => Value::Bool(!eval_expr(a, env).is_truthy()),
        Expr::IsDefined(a) => match eval_expr(a, env) {
            Value::None => Value::Bool(false),
            _ => Value::Bool(true),
        },
        Expr::IsNotDefined(a) => match eval_expr(a, env) {
            Value::None => Value::Bool(true),
            _ => Value::Bool(false),
        },
        Expr::Trim(a) => {
            let s = eval_expr(a, env).as_str();
            Value::Str(s.trim().to_string())
        }
    }
}

// -------------------------------------------------------------------- Tokenizer & Parser

#[derive(Clone, Debug, PartialEq)]
enum RawToken {
    Text(String),
    Output {
        content: String,
        trim_left: bool,
        trim_right: bool,
    },
    Tag {
        content: String,
        trim_left: bool,
        trim_right: bool,
    },
}

fn tokenize(template: &str) -> Vec<RawToken> {
    let mut tokens = Vec::new();
    let mut idx = 0;
    let bytes = template.as_bytes();
    let len = bytes.len();
    let mut trim_next_leading = false;

    while idx < len {
        if idx + 1 < len && bytes[idx] == b'{' && (bytes[idx + 1] == b'{' || bytes[idx + 1] == b'%')
        {
            let is_output = bytes[idx + 1] == b'{';
            let close_tag = if is_output { b"}}" } else { b"%}" };
            let start = idx + 2;

            let trim_left = start < len && bytes[start] == b'-';

            // Look for closing tag
            let mut close_idx = None;
            let mut scan = start;
            while scan + 1 < len {
                if bytes[scan] == close_tag[0] && bytes[scan + 1] == close_tag[1] {
                    close_idx = Some(scan);
                    break;
                }
                scan += 1;
            }

            if let Some(end) = close_idx {
                let trim_right = end > start && bytes[end - 1] == b'-';

                let slice_start = if trim_left { start + 1 } else { start };
                let slice_end = if trim_right { end - 1 } else { end };
                let content = template[slice_start..slice_end].trim().to_string();

                if trim_left {
                    if let Some(RawToken::Text(ref mut prev)) = tokens.last_mut() {
                        let trimmed_len = prev.trim_end().len();
                        prev.truncate(trimmed_len);
                    }
                }
                if trim_right {
                    trim_next_leading = true;
                }

                if is_output {
                    tokens.push(RawToken::Output {
                        content,
                        trim_left,
                        trim_right,
                    });
                } else {
                    tokens.push(RawToken::Tag {
                        content,
                        trim_left,
                        trim_right,
                    });
                }
                idx = end + 2;
                continue;
            }
        }

        // Collect literal text up to the next `{`
        let mut end = idx + 1;
        while end < len {
            if end + 1 < len
                && bytes[end] == b'{'
                && (bytes[end + 1] == b'{' || bytes[end + 1] == b'%')
            {
                break;
            }
            end += 1;
        }
        let raw = template[idx..end].to_string();
        let text = if trim_next_leading {
            trim_next_leading = false;
            raw.trim_start().to_string()
        } else {
            raw
        };
        if !text.is_empty() {
            tokens.push(RawToken::Text(text));
        }
        idx = end;
    }

    tokens
}

fn parse_nodes(tokens: &mut Vec<RawToken>, stop_tags: &[&str]) -> Vec<Node> {
    let mut nodes = Vec::new();

    while !tokens.is_empty() {
        if let RawToken::Tag { content, .. } = &tokens[0] {
            let first_word = content.split_whitespace().next().unwrap_or("");
            if stop_tags.contains(&first_word) {
                break;
            }
        }

        let tok = tokens.remove(0);
        match tok {
            RawToken::Text(txt) => nodes.push(Node::Text(txt)),
            RawToken::Output {
                content,
                trim_left,
                trim_right,
            } => {
                let expr = parse_expression(&content);
                nodes.push(Node::Output {
                    expr,
                    trim_left,
                    trim_right,
                });
            }
            RawToken::Tag { content, .. } => {
                let mut words = content.split_whitespace();
                let cmd = words.next().unwrap_or("");
                match cmd {
                    "for" => {
                        let var = words.next().unwrap_or("item");
                        let _in = words.next();
                        let iter_expr_str = words.collect::<Vec<_>>().join(" ");
                        let iterable = parse_expression(&iter_expr_str);

                        let body = parse_nodes(tokens, &["endfor"]);
                        if !tokens.is_empty() {
                            tokens.remove(0); // consume endfor
                        }
                        nodes.push(Node::For {
                            var_name: var.to_string(),
                            iterable,
                            body,
                        });
                    }
                    "if" => {
                        let cond_str = content["if".len()..].trim();
                        let mut branches = Vec::new();
                        branches.push((
                            parse_expression(cond_str),
                            parse_nodes(tokens, &["elif", "else", "endif"]),
                        ));

                        let mut else_branch = None;
                        while !tokens.is_empty() {
                            if let RawToken::Tag { content, .. } = &tokens[0] {
                                let w = content.split_whitespace().next().unwrap_or("");
                                if w == "elif" {
                                    let tag = tokens.remove(0);
                                    if let RawToken::Tag { content, .. } = tag {
                                        let elif_cond = content["elif".len()..].trim();
                                        let branch_body =
                                            parse_nodes(tokens, &["elif", "else", "endif"]);
                                        branches.push((parse_expression(elif_cond), branch_body));
                                    }
                                } else if w == "else" {
                                    tokens.remove(0);
                                    else_branch = Some(parse_nodes(tokens, &["endif"]));
                                } else if w == "endif" {
                                    tokens.remove(0);
                                    break;
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }

                        nodes.push(Node::If {
                            branches,
                            else_branch,
                        });
                    }
                    "set" => {
                        let rest = content["set".len()..].trim();
                        if let Some((lhs, rhs)) = rest.split_once('=') {
                            nodes.push(Node::Set {
                                var_name: lhs.trim().to_string(),
                                expr: parse_expression(rhs.trim()),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    nodes
}

fn parse_expression(s: &str) -> Expr {
    let s = s.trim();

    // Check for filter `| trim`
    if let Some((lhs, filter)) = s.rsplit_once('|') {
        if filter.trim() == "trim" {
            return Expr::Trim(Box::new(parse_expression(lhs.trim())));
        }
    }

    // Check for `is defined` / `is not defined`
    if let Some((lhs, _)) = s.split_once("is not defined") {
        return Expr::IsNotDefined(Box::new(parse_expression(lhs.trim())));
    }
    if let Some((lhs, _)) = s.split_once("is defined") {
        return Expr::IsDefined(Box::new(parse_expression(lhs.trim())));
    }

    // Check for `==` and `!=`
    if let Some((lhs, rhs)) = s.split_once("==") {
        return Expr::Equal(
            Box::new(parse_expression(lhs.trim())),
            Box::new(parse_expression(rhs.trim())),
        );
    }
    if let Some((lhs, rhs)) = s.split_once("!=") {
        return Expr::NotEqual(
            Box::new(parse_expression(lhs.trim())),
            Box::new(parse_expression(rhs.trim())),
        );
    }

    // Check for `and` / `or`
    if let Some((lhs, rhs)) = s.split_once(" and ") {
        return Expr::And(
            Box::new(parse_expression(lhs.trim())),
            Box::new(parse_expression(rhs.trim())),
        );
    }
    if let Some((lhs, rhs)) = s.split_once(" or ") {
        return Expr::Or(
            Box::new(parse_expression(lhs.trim())),
            Box::new(parse_expression(rhs.trim())),
        );
    }

    // Check for string concatenation `+`
    let parts = split_concatenation(s);
    if parts.len() > 1 {
        return Expr::Concat(parts.into_iter().map(|p| parse_expression(&p)).collect());
    }

    if let Some(rest) = s.strip_prefix("not ") {
        return Expr::Not(Box::new(parse_expression(rest.trim())));
    }

    // Base primitives
    if (s.starts_with('\'') && s.ends_with('\'')) || (s.starts_with('"') && s.ends_with('"')) {
        let inner = &s[1..s.len() - 1];
        return Expr::LiteralString(unescape_jinja_str(inner));
    }
    if s == "true" || s == "True" {
        return Expr::LiteralBool(true);
    }
    if s == "false" || s == "False" {
        return Expr::LiteralBool(false);
    }
    if let Ok(i) = s.parse::<i64>() {
        return Expr::LiteralInt(i);
    }

    // Member access: `foo['bar']` or `foo.bar`
    if let Some((parent, member)) = parse_member(s) {
        return Expr::Member(Box::new(parse_expression(&parent)), member);
    }

    Expr::Variable(s.to_string())
}

fn split_concatenation(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_quote = None;
    let bytes = s.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\'' || b == b'"' {
            if in_quote == Some(b) {
                in_quote = None;
            } else if in_quote.is_none() {
                in_quote = Some(b);
            }
        } else if b == b'+' && in_quote.is_none() {
            parts.push(s[start..i].trim().to_string());
            start = i + 1;
        }
    }
    if start < s.len() {
        parts.push(s[start..].trim().to_string());
    }
    parts
}

fn parse_member(s: &str) -> Option<(String, String)> {
    if s.ends_with(']') {
        if let Some(open) = s.find('[') {
            let parent = s[..open].trim();
            let mut member = s[open + 1..s.len() - 1].trim();
            if (member.starts_with('\'') && member.ends_with('\''))
                || (member.starts_with('"') && member.ends_with('"'))
            {
                member = &member[1..member.len() - 1];
            }
            return Some((parent.to_string(), member.to_string()));
        }
    }
    if let Some((parent, member)) = s.split_once('.') {
        return Some((parent.trim().to_string(), member.trim().to_string()));
    }
    None
}

fn unescape_jinja_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('\'') => out.push('\''),
                Some('"') => out.push('"'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_chatml_template() {
        let template = ChatTemplate::chatml();
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello!"),
        ];
        let rendered = template.render(&messages, true, None, None);
        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn render_llama3_template() {
        let template = ChatTemplate::llama3();
        let messages = vec![
            ChatMessage::user("What is 2+2?"),
            ChatMessage::assistant("4"),
        ];
        let rendered = template.render(&messages, true, Some("<|begin_of_text|>"), None);
        assert_eq!(
            rendered,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n4<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn render_gemma_template() {
        let template = ChatTemplate::gemma();
        let messages = vec![ChatMessage::user("Hi")];
        let rendered = template.render(&messages, true, None, None);
        assert_eq!(
            rendered,
            "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>assistant\n"
        );
    }

    #[test]
    fn render_loop_helpers_and_conditionals() {
        let tmpl = ChatTemplate::parse(
            "{% for message in messages %}{% if loop.first %}BEGIN\n{% endif %}{{ message.role }}: {{ message.content }}\n{% if loop.last %}END{% endif %}{% endfor %}"
        );
        let messages = vec![
            ChatMessage::user("Question 1"),
            ChatMessage::assistant("Answer 1"),
        ];
        let rendered = tmpl.render(&messages, false, None, None);
        assert_eq!(
            rendered,
            "BEGIN\nuser: Question 1\nassistant: Answer 1\nEND"
        );
    }
}
