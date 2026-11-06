//! Zero-dependency grammar-constrained decoding engine.
//!
//! Enforces exact structural conformance (JSON schemas, regular expressions,
//! bounded choice sets) during autoregressive token generation by dynamically
//! masking invalid vocabulary logits before sampling.

use std::collections::HashSet;

use crate::tokenizer::Tokenizer;

/// Error returned when parsing an invalid regular expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegexError(pub String);

impl std::fmt::Display for RegexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RegexError: {}", self.0)
    }
}

impl std::error::Error for RegexError {}

/// An NFA transition condition on a byte.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ByteCondition {
    Byte(u8),
    Range(u8, u8),
    Class(Vec<(u8, u8)>),
    Any,
    Epsilon,
}

impl ByteCondition {
    fn matches(&self, b: u8) -> bool {
        match self {
            Self::Byte(val) => *val == b,
            Self::Range(start, end) => b >= *start && b <= *end,
            Self::Class(ranges) => ranges.iter().any(|(s, e)| b >= *s && b <= *e),
            Self::Any => true,
            Self::Epsilon => false,
        }
    }
}

/// A node/state in the NFA.
#[derive(Clone, Debug)]
struct NfaState {
    transitions: Vec<(ByteCondition, usize)>,
    is_accept: bool,
}

/// Thompson NFA representation of a compiled regular expression.
#[derive(Clone, Debug)]
pub struct RegexMatcher {
    states: Vec<NfaState>,
    start_state: usize,
    current_states: HashSet<usize>,
}

impl RegexMatcher {
    /// Compile a regex pattern into an NFA matcher.
    pub fn new(pattern: &str) -> std::result::Result<Self, RegexError> {
        let mut compiler = RegexCompiler::new(pattern);
        let (start, accept) = compiler.compile()?;
        compiler.states[accept].is_accept = true;

        let mut matcher = Self {
            states: compiler.states,
            start_state: start,
            current_states: HashSet::new(),
        };
        matcher.reset();
        Ok(matcher)
    }

    /// Reset matcher to start state.
    pub fn reset(&mut self) {
        self.current_states.clear();
        let mut initial = HashSet::new();
        initial.insert(self.start_state);
        self.current_states = self.epsilon_closure(&initial);
    }

    fn epsilon_closure(&self, states: &HashSet<usize>) -> HashSet<usize> {
        let mut closure = states.clone();
        let mut stack: Vec<usize> = states.iter().copied().collect();

        while let Some(state_idx) = stack.pop() {
            if let Some(state) = self.states.get(state_idx) {
                for (cond, target) in &state.transitions {
                    if *cond == ByteCondition::Epsilon && closure.insert(*target) {
                        stack.push(*target);
                    }
                }
            }
        }
        closure
    }

    /// Step state set by one byte.
    fn step_set(&self, from: &HashSet<usize>, byte: u8) -> HashSet<usize> {
        let mut next = HashSet::new();
        for &state_idx in from {
            if let Some(state) = self.states.get(state_idx) {
                for (cond, target) in &state.transitions {
                    if cond.matches(byte) {
                        next.insert(*target);
                    }
                }
            }
        }
        self.epsilon_closure(&next)
    }

    /// Check if a byte slice can be consumed from the current state set.
    pub fn accepts_prefix(&self, bytes: &[u8]) -> bool {
        let mut cur = self.current_states.clone();
        for &b in bytes {
            cur = self.step_set(&cur, b);
            if cur.is_empty() {
                return false;
            }
        }
        true
    }

    /// Advance the current state set by consuming `bytes`.
    /// Returns true if successful, false if input cannot be consumed.
    pub fn advance(&mut self, bytes: &[u8]) -> bool {
        let mut cur = self.current_states.clone();
        for &b in bytes {
            cur = self.step_set(&cur, b);
            if cur.is_empty() {
                return false;
            }
        }
        self.current_states = cur;
        true
    }

    /// Check if current state contains any accepting state.
    pub fn is_accepting(&self) -> bool {
        self.current_states
            .iter()
            .any(|&s| self.states.get(s).is_some_and(|st| st.is_accept))
    }
}

struct RegexCompiler {
    chars: Vec<char>,
    pos: usize,
    states: Vec<NfaState>,
}

impl RegexCompiler {
    fn new(pattern: &str) -> Self {
        Self {
            chars: pattern.chars().collect(),
            pos: 0,
            states: Vec::new(),
        }
    }

    fn add_state(&mut self) -> usize {
        let id = self.states.len();
        self.states.push(NfaState {
            transitions: Vec::new(),
            is_accept: false,
        });
        id
    }

    fn add_transition(&mut self, from: usize, to: usize, cond: ByteCondition) {
        self.states[from].transitions.push((cond, to));
    }

    fn compile(&mut self) -> std::result::Result<(usize, usize), RegexError> {
        if self.chars.is_empty() {
            let s = self.add_state();
            return Ok((s, s));
        }
        let res = self.parse_expr()?;
        if self.pos < self.chars.len() {
            return Err(RegexError(format!(
                "Unexpected character at position {}: '{}'",
                self.pos, self.chars[self.pos]
            )));
        }
        Ok(res)
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }

    // Expr = Term ('|' Term)*
    fn parse_expr(&mut self) -> std::result::Result<(usize, usize), RegexError> {
        let (mut start_a, mut end_a) = self.parse_term()?;

        while self.peek() == Some('|') {
            self.next();
            let (start_b, end_b) = self.parse_term()?;
            let new_start = self.add_state();
            let new_end = self.add_state();

            self.add_transition(new_start, start_a, ByteCondition::Epsilon);
            self.add_transition(new_start, start_b, ByteCondition::Epsilon);
            self.add_transition(end_a, new_end, ByteCondition::Epsilon);
            self.add_transition(end_b, new_end, ByteCondition::Epsilon);

            start_a = new_start;
            end_a = new_end;
        }

        Ok((start_a, end_a))
    }

    // Term = Factor+
    fn parse_term(&mut self) -> std::result::Result<(usize, usize), RegexError> {
        let (start, mut end) = self.parse_factor()?;

        while let Some(ch) = self.peek() {
            if ch == '|' || ch == ')' {
                break;
            }
            let (next_start, next_end) = self.parse_factor()?;
            self.add_transition(end, next_start, ByteCondition::Epsilon);
            end = next_end;
        }

        Ok((start, end))
    }

    // Factor = Atom Quantifier?
    fn parse_factor(&mut self) -> std::result::Result<(usize, usize), RegexError> {
        let (mut start, mut end) = self.parse_atom()?;

        match self.peek() {
            Some('*') => {
                self.next();
                let new_start = self.add_state();
                let new_end = self.add_state();
                self.add_transition(new_start, start, ByteCondition::Epsilon);
                self.add_transition(new_start, new_end, ByteCondition::Epsilon);
                self.add_transition(end, start, ByteCondition::Epsilon);
                self.add_transition(end, new_end, ByteCondition::Epsilon);
                start = new_start;
                end = new_end;
            }
            Some('+') => {
                self.next();
                let new_start = self.add_state();
                let new_end = self.add_state();
                self.add_transition(new_start, start, ByteCondition::Epsilon);
                self.add_transition(end, start, ByteCondition::Epsilon);
                self.add_transition(end, new_end, ByteCondition::Epsilon);
                start = new_start;
                end = new_end;
            }
            Some('?') => {
                self.next();
                let new_start = self.add_state();
                let new_end = self.add_state();
                self.add_transition(new_start, start, ByteCondition::Epsilon);
                self.add_transition(new_start, new_end, ByteCondition::Epsilon);
                self.add_transition(end, new_end, ByteCondition::Epsilon);
                start = new_start;
                end = new_end;
            }
            _ => {}
        }

        Ok((start, end))
    }

    // Atom = '(' Expr ')' | '[' Class ']' | Escape | Literal
    fn parse_atom(&mut self) -> std::result::Result<(usize, usize), RegexError> {
        let Some(ch) = self.next() else {
            return Err(RegexError("Unexpected end of pattern".into()));
        };

        match ch {
            '(' => {
                let res = self.parse_expr()?;
                if self.next() != Some(')') {
                    return Err(RegexError("Missing closing parenthesis ')'".into()));
                }
                Ok(res)
            }
            '[' => self.parse_class(),
            '.' => {
                let s = self.add_state();
                let e = self.add_state();
                self.add_transition(s, e, ByteCondition::Any);
                Ok((s, e))
            }
            '\\' => {
                let escaped = self
                    .next()
                    .ok_or_else(|| RegexError("Trailing escape '\\'".into()))?;
                let s = self.add_state();
                let e = self.add_state();
                match escaped {
                    'd' => self.add_transition(s, e, ByteCondition::Range(b'0', b'9')),
                    'w' => self.add_transition(
                        s,
                        e,
                        ByteCondition::Class(vec![
                            (b'a', b'z'),
                            (b'A', b'Z'),
                            (b'0', b'9'),
                            (b'_', b'_'),
                        ]),
                    ),
                    's' => self.add_transition(
                        s,
                        e,
                        ByteCondition::Class(vec![
                            (b' ', b' '),
                            (b'\t', b'\t'),
                            (b'\n', b'\n'),
                            (b'\r', b'\r'),
                        ]),
                    ),
                    'n' => self.add_transition(s, e, ByteCondition::Byte(b'\n')),
                    't' => self.add_transition(s, e, ByteCondition::Byte(b'\t')),
                    'r' => self.add_transition(s, e, ByteCondition::Byte(b'\r')),
                    other => {
                        let mut buf = [0u8; 4];
                        let bytes = other.encode_utf8(&mut buf).as_bytes();
                        if bytes.len() == 1 {
                            self.add_transition(s, e, ByteCondition::Byte(bytes[0]));
                        } else {
                            let mut cur = s;
                            for (i, &b) in bytes.iter().enumerate() {
                                let nxt = if i + 1 == bytes.len() {
                                    e
                                } else {
                                    self.add_state()
                                };
                                self.add_transition(cur, nxt, ByteCondition::Byte(b));
                                cur = nxt;
                            }
                        }
                    }
                }
                Ok((s, e))
            }
            literal => {
                let s = self.add_state();
                let e = self.add_state();
                let mut buf = [0u8; 4];
                let bytes = literal.encode_utf8(&mut buf).as_bytes();
                if bytes.len() == 1 {
                    self.add_transition(s, e, ByteCondition::Byte(bytes[0]));
                } else {
                    let mut cur = s;
                    for (i, &b) in bytes.iter().enumerate() {
                        let nxt = if i + 1 == bytes.len() {
                            e
                        } else {
                            self.add_state()
                        };
                        self.add_transition(cur, nxt, ByteCondition::Byte(b));
                        cur = nxt;
                    }
                }
                Ok((s, e))
            }
        }
    }

    fn parse_class(&mut self) -> std::result::Result<(usize, usize), RegexError> {
        let negated = if self.peek() == Some('^') {
            self.next();
            true
        } else {
            false
        };

        let mut ranges = Vec::new();

        while let Some(ch) = self.next() {
            if ch == ']' {
                break;
            }
            if self.peek() == Some('-')
                && self.pos + 1 < self.chars.len()
                && self.chars[self.pos + 1] != ']'
            {
                self.next(); // skip '-'
                let end_ch = self.next().unwrap();
                ranges.push((ch as u8, end_ch as u8));
            } else {
                ranges.push((ch as u8, ch as u8));
            }
        }

        let s = self.add_state();
        let e = self.add_state();

        if negated {
            let mut allowed = vec![true; 256];
            for (st, en) in ranges {
                for b in st..=en {
                    allowed[b as usize] = false;
                }
            }
            let mut inv_ranges = Vec::new();
            let mut start = None;
            for b in 0..=255u8 {
                if allowed[b as usize] {
                    if start.is_none() {
                        start = Some(b);
                    }
                } else if let Some(s_b) = start {
                    inv_ranges.push((s_b, b - 1));
                    start = None;
                }
            }
            if let Some(s_b) = start {
                inv_ranges.push((s_b, 255));
            }
            self.add_transition(s, e, ByteCondition::Class(inv_ranges));
        } else {
            self.add_transition(s, e, ByteCondition::Class(ranges));
        }

        Ok((s, e))
    }
}

/// JSON parser stack element for streaming syntax verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JsonScope {
    Value,
    ObjectKey,
    ObjectColon,
    ObjectValue,
    ObjectCommaOrClose,
    ArrayValue,
    ArrayCommaOrClose,
    String { escape: bool },
    Number,
    Boolean { expected: &'static [u8], pos: usize },
    Null { pos: usize },
}

/// Streaming JSON grammar validator enforcing RFC 8259 syntax.
#[derive(Clone, Debug)]
pub struct JsonGrammar {
    stack: Vec<JsonScope>,
    has_started: bool,
    has_completed: bool,
}

impl Default for JsonGrammar {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGrammar {
    pub fn new() -> Self {
        Self {
            stack: vec![JsonScope::Value],
            has_started: false,
            has_completed: false,
        }
    }

    pub fn is_accepting(&self) -> bool {
        self.has_completed || (self.has_started && self.stack.is_empty())
    }

    pub fn accepts_prefix(&self, bytes: &[u8]) -> bool {
        let mut sim = self.clone();
        sim.advance(bytes)
    }

    pub fn advance(&mut self, bytes: &[u8]) -> bool {
        for &b in bytes {
            if !self.step_byte(b) {
                return false;
            }
        }
        true
    }

    fn step_byte(&mut self, b: u8) -> bool {
        if self.has_completed {
            return b.is_ascii_whitespace();
        }

        let Some(scope) = self.stack.last().copied() else {
            if b.is_ascii_whitespace() {
                return true;
            }
            return false;
        };

        match scope {
            JsonScope::Value => {
                if b.is_ascii_whitespace() {
                    return true;
                }
                self.has_started = true;
                self.stack.pop();
                match b {
                    b'{' => {
                        self.stack.push(JsonScope::ObjectCommaOrClose);
                        self.stack.push(JsonScope::ObjectKey);
                        true
                    }
                    b'[' => {
                        self.stack.push(JsonScope::ArrayCommaOrClose);
                        self.stack.push(JsonScope::ArrayValue);
                        true
                    }
                    b'"' => {
                        self.stack.push(JsonScope::String { escape: false });
                        true
                    }
                    b't' => {
                        self.stack.push(JsonScope::Boolean {
                            expected: b"true",
                            pos: 1,
                        });
                        true
                    }
                    b'f' => {
                        self.stack.push(JsonScope::Boolean {
                            expected: b"false",
                            pos: 1,
                        });
                        true
                    }
                    b'n' => {
                        self.stack.push(JsonScope::Null { pos: 1 });
                        true
                    }
                    b'-' | b'0'..=b'9' => {
                        self.stack.push(JsonScope::Number);
                        true
                    }
                    _ => false,
                }
            }
            JsonScope::ObjectKey => {
                if b.is_ascii_whitespace() {
                    return true;
                }
                if b == b'}' {
                    self.stack.pop(); // pop ObjectKey
                    self.stack.pop(); // pop ObjectCommaOrClose
                    if self.stack.is_empty() {
                        self.has_completed = true;
                    }
                    return true;
                }
                if b == b'"' {
                    self.stack.pop();
                    self.stack.push(JsonScope::ObjectColon);
                    self.stack.push(JsonScope::String { escape: false });
                    return true;
                }
                false
            }
            JsonScope::ObjectColon => {
                if b.is_ascii_whitespace() {
                    return true;
                }
                if b == b':' {
                    self.stack.pop();
                    self.stack.push(JsonScope::ObjectValue);
                    self.stack.push(JsonScope::Value);
                    return true;
                }
                false
            }
            JsonScope::ObjectValue => {
                self.stack.pop();
                self.step_byte(b)
            }
            JsonScope::ObjectCommaOrClose => {
                if b.is_ascii_whitespace() {
                    return true;
                }
                if b == b',' {
                    self.stack.pop();
                    self.stack.push(JsonScope::ObjectCommaOrClose);
                    self.stack.push(JsonScope::ObjectKey);
                    return true;
                }
                if b == b'}' {
                    self.stack.pop();
                    if self.stack.is_empty() {
                        self.has_completed = true;
                    }
                    return true;
                }
                false
            }
            JsonScope::ArrayValue => {
                if b.is_ascii_whitespace() {
                    return true;
                }
                if b == b']' {
                    self.stack.pop(); // pop ArrayValue
                    self.stack.pop(); // pop ArrayCommaOrClose
                    if self.stack.is_empty() {
                        self.has_completed = true;
                    }
                    return true;
                }
                self.stack.pop();
                self.stack.push(JsonScope::Value);
                self.step_byte(b)
            }
            JsonScope::ArrayCommaOrClose => {
                if b.is_ascii_whitespace() {
                    return true;
                }
                if b == b',' {
                    self.stack.pop();
                    self.stack.push(JsonScope::ArrayCommaOrClose);
                    self.stack.push(JsonScope::ArrayValue);
                    return true;
                }
                if b == b']' {
                    self.stack.pop();
                    if self.stack.is_empty() {
                        self.has_completed = true;
                    }
                    return true;
                }
                false
            }
            JsonScope::String { escape } => {
                if escape {
                    self.stack.pop();
                    self.stack.push(JsonScope::String { escape: false });
                    matches!(
                        b,
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u'
                    )
                } else if b == b'\\' {
                    self.stack.pop();
                    self.stack.push(JsonScope::String { escape: true });
                    true
                } else if b == b'"' {
                    self.stack.pop();
                    if self.stack.is_empty() {
                        self.has_completed = true;
                    }
                    true
                } else {
                    b >= 0x20
                }
            }
            JsonScope::Number => {
                if b.is_ascii_digit()
                    || b == b'.'
                    || b == b'e'
                    || b == b'E'
                    || b == b'+'
                    || b == b'-'
                {
                    true
                } else {
                    self.stack.pop();
                    if self.stack.is_empty() {
                        self.has_completed = true;
                    }
                    self.step_byte(b)
                }
            }
            JsonScope::Boolean { expected, pos } => {
                if pos < expected.len() && b == expected[pos] {
                    self.stack.pop();
                    if pos + 1 == expected.len() {
                        if self.stack.is_empty() {
                            self.has_completed = true;
                        }
                    } else {
                        self.stack.push(JsonScope::Boolean {
                            expected,
                            pos: pos + 1,
                        });
                    }
                    true
                } else {
                    false
                }
            }
            JsonScope::Null { pos } => {
                const EXPECTED: &[u8] = b"null";
                if pos < EXPECTED.len() && b == EXPECTED[pos] {
                    self.stack.pop();
                    if pos + 1 == EXPECTED.len() {
                        if self.stack.is_empty() {
                            self.has_completed = true;
                        }
                    } else {
                        self.stack.push(JsonScope::Null { pos: pos + 1 });
                    }
                    true
                } else {
                    false
                }
            }
        }
    }
}

/// Exact choice grammar: restricts generation to one of a finite set of strings.
#[derive(Clone, Debug)]
pub struct ChoiceGrammar {
    options: Vec<Vec<u8>>,
    current_prefix: Vec<u8>,
}

impl ChoiceGrammar {
    pub fn new(options: &[&str]) -> Self {
        Self {
            options: options.iter().map(|s| s.as_bytes().to_vec()).collect(),
            current_prefix: Vec::new(),
        }
    }

    pub fn is_accepting(&self) -> bool {
        self.options.iter().any(|opt| opt == &self.current_prefix)
    }

    pub fn accepts_prefix(&self, bytes: &[u8]) -> bool {
        let mut test = self.current_prefix.clone();
        test.extend_from_slice(bytes);
        self.options
            .iter()
            .any(|opt| opt.starts_with(&test) || test.starts_with(opt))
    }

    pub fn advance(&mut self, bytes: &[u8]) -> bool {
        if self.accepts_prefix(bytes) {
            self.current_prefix.extend_from_slice(bytes);
            true
        } else {
            false
        }
    }
}

/// AST representation of a compiled JSON schema.
#[derive(Clone, Debug, PartialEq)]
pub enum SchemaNode {
    String,
    Integer,
    Number,
    Boolean,
    Enum(Vec<String>),
    Array(Box<SchemaNode>),
    Object {
        properties: Vec<(String, SchemaNode)>,
        required: Vec<String>,
    },
    Any,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatchResult {
    Complete(usize),
    Partial,
    Mismatch,
}

fn skip_ascii_ws(bytes: &[u8]) -> &[u8] {
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    &bytes[i..]
}

fn check_schema_node(node: &SchemaNode, input: &[u8]) -> MatchResult {
    let bytes = skip_ascii_ws(input);
    if bytes.is_empty() {
        return MatchResult::Partial;
    }
    let lead_ws = input.len() - bytes.len();

    match node {
        SchemaNode::String => {
            if bytes[0] != b'"' {
                return MatchResult::Mismatch;
            }
            let mut i = 1;
            let mut escape = false;
            while i < bytes.len() {
                let b = bytes[i];
                if escape {
                    escape = false;
                } else if b == b'\\' {
                    escape = true;
                } else if b == b'"' {
                    return MatchResult::Complete(lead_ws + i + 1);
                }
                i += 1;
            }
            MatchResult::Partial
        }
        SchemaNode::Enum(choices) => {
            if bytes[0] != b'"' {
                return MatchResult::Mismatch;
            }
            let mut i = 1;
            let mut escape = false;
            let mut closed = false;
            while i < bytes.len() {
                let b = bytes[i];
                if escape {
                    escape = false;
                } else if b == b'\\' {
                    escape = true;
                } else if b == b'"' {
                    closed = true;
                    break;
                }
                i += 1;
            }
            let text = &bytes[1..i];
            if closed {
                if let Ok(s) = std::str::from_utf8(text) {
                    if choices.iter().any(|c| c == s) {
                        return MatchResult::Complete(lead_ws + i + 1);
                    }
                }
                MatchResult::Mismatch
            } else if let Ok(s) = std::str::from_utf8(text) {
                if choices.iter().any(|c| c.starts_with(s)) {
                    MatchResult::Partial
                } else {
                    MatchResult::Mismatch
                }
            } else {
                MatchResult::Partial
            }
        }
        SchemaNode::Integer => {
            let mut i = 0;
            if bytes[0] == b'-' {
                i += 1;
            }
            if i >= bytes.len() {
                return MatchResult::Partial;
            }
            if !bytes[i].is_ascii_digit() {
                return MatchResult::Mismatch;
            }
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            MatchResult::Complete(lead_ws + i)
        }
        SchemaNode::Number => {
            let mut i = 0;
            if bytes[0] == b'-' {
                i += 1;
            }
            if i >= bytes.len() {
                return MatchResult::Partial;
            }
            if !bytes[i].is_ascii_digit() {
                return MatchResult::Mismatch;
            }
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'.' {
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            MatchResult::Complete(lead_ws + i)
        }
        SchemaNode::Boolean => {
            for &expected in &[&b"true"[..], &b"false"[..]] {
                if bytes.starts_with(expected) {
                    return MatchResult::Complete(lead_ws + expected.len());
                } else if expected.starts_with(bytes) {
                    return MatchResult::Partial;
                }
            }
            MatchResult::Mismatch
        }
        SchemaNode::Array(elem_node) => {
            if bytes[0] != b'[' {
                return MatchResult::Mismatch;
            }
            let mut cur = &bytes[1..];
            let mut total_consumed = 1;
            loop {
                let rest = skip_ascii_ws(cur);
                total_consumed += cur.len() - rest.len();
                cur = rest;
                if cur.is_empty() {
                    return MatchResult::Partial;
                }
                if cur[0] == b']' {
                    return MatchResult::Complete(lead_ws + total_consumed + 1);
                }
                match check_schema_node(elem_node, cur) {
                    MatchResult::Complete(consumed) => {
                        total_consumed += consumed;
                        cur = &cur[consumed..];
                        let after = skip_ascii_ws(cur);
                        total_consumed += cur.len() - after.len();
                        cur = after;
                        if cur.is_empty() {
                            return MatchResult::Partial;
                        }
                        if cur[0] == b',' {
                            total_consumed += 1;
                            cur = &cur[1..];
                        } else if cur[0] == b']' {
                            return MatchResult::Complete(lead_ws + total_consumed + 1);
                        } else {
                            return MatchResult::Mismatch;
                        }
                    }
                    MatchResult::Partial => return MatchResult::Partial,
                    MatchResult::Mismatch => return MatchResult::Mismatch,
                }
            }
        }
        SchemaNode::Object {
            properties,
            required,
        } => {
            if bytes[0] != b'{' {
                return MatchResult::Mismatch;
            }
            let mut cur = &bytes[1..];
            let mut total_consumed = 1;
            let mut seen_keys = std::collections::HashSet::new();

            loop {
                let rest = skip_ascii_ws(cur);
                total_consumed += cur.len() - rest.len();
                cur = rest;
                if cur.is_empty() {
                    return MatchResult::Partial;
                }
                if cur[0] == b'}' {
                    let all_req = required.iter().all(|k| seen_keys.contains(k));
                    return if all_req {
                        MatchResult::Complete(lead_ws + total_consumed + 1)
                    } else {
                        MatchResult::Mismatch
                    };
                }

                if cur[0] != b'"' {
                    return MatchResult::Mismatch;
                }
                let mut k_end = 1;
                let mut closed_key = false;
                while k_end < cur.len() {
                    if cur[k_end] == b'"' && cur[k_end - 1] != b'\\' {
                        closed_key = true;
                        break;
                    }
                    k_end += 1;
                }
                if !closed_key {
                    return MatchResult::Partial;
                }
                let key_str = match std::str::from_utf8(&cur[1..k_end]) {
                    Ok(s) => s.to_string(),
                    Err(_) => return MatchResult::Mismatch,
                };
                seen_keys.insert(key_str.clone());
                total_consumed += k_end + 1;
                cur = &cur[k_end + 1..];

                let rest = skip_ascii_ws(cur);
                total_consumed += cur.len() - rest.len();
                cur = rest;
                if cur.is_empty() {
                    return MatchResult::Partial;
                }
                if cur[0] != b':' {
                    return MatchResult::Mismatch;
                }
                total_consumed += 1;
                cur = &cur[1..];

                let prop_schema = properties
                    .iter()
                    .find(|(k, _)| k == &key_str)
                    .map(|(_, node)| node)
                    .unwrap_or(&SchemaNode::Any);

                match check_schema_node(prop_schema, cur) {
                    MatchResult::Complete(consumed) => {
                        total_consumed += consumed;
                        cur = &cur[consumed..];
                        let after = skip_ascii_ws(cur);
                        total_consumed += cur.len() - after.len();
                        cur = after;
                        if cur.is_empty() {
                            return MatchResult::Partial;
                        }
                        if cur[0] == b',' {
                            total_consumed += 1;
                            cur = &cur[1..];
                        } else if cur[0] == b'}' {
                            let all_req = required.iter().all(|k| seen_keys.contains(k));
                            return if all_req {
                                MatchResult::Complete(lead_ws + total_consumed + 1)
                            } else {
                                MatchResult::Mismatch
                            };
                        } else {
                            return MatchResult::Mismatch;
                        }
                    }
                    MatchResult::Partial => return MatchResult::Partial,
                    MatchResult::Mismatch => return MatchResult::Mismatch,
                }
            }
        }
        SchemaNode::Any => {
            if bytes[0] == b'"' {
                return check_schema_node(&SchemaNode::String, input);
            }
            if bytes[0] == b'{' {
                return check_schema_node(
                    &SchemaNode::Object {
                        properties: Vec::new(),
                        required: Vec::new(),
                    },
                    input,
                );
            }
            if bytes[0] == b'[' {
                return check_schema_node(&SchemaNode::Array(Box::new(SchemaNode::Any)), input);
            }
            if bytes[0] == b't' || bytes[0] == b'f' {
                return check_schema_node(&SchemaNode::Boolean, input);
            }
            if bytes[0] == b'-' || bytes[0].is_ascii_digit() {
                return check_schema_node(&SchemaNode::Number, input);
            }
            MatchResult::Partial
        }
    }
}

fn parse_json_schema(src: &str) -> std::result::Result<SchemaNode, String> {
    let trimmed = src.trim();
    if !trimmed.starts_with('{') {
        return Ok(SchemaNode::Any);
    }

    let ty = if let Some(type_idx) = trimmed.find("\"type\"") {
        let rest = &trimmed[type_idx + 6..];
        let after_colon = rest.trim_start().trim_start_matches(':').trim_start();
        if let Some(val_end) = after_colon.find(|c: char| c == ',' || c == '}' || c.is_whitespace())
        {
            after_colon[..val_end].trim_matches('"').to_string()
        } else {
            after_colon.trim_matches('"').to_string()
        }
    } else if trimmed.contains("\"properties\"") {
        "object".to_string()
    } else if trimmed.contains("\"enum\"") {
        "string".to_string()
    } else {
        "object".to_string()
    };

    match ty.as_str() {
        "string" => {
            if let Some(enum_idx) = trimmed.find("\"enum\"") {
                if let Some(arr_start) = trimmed[enum_idx..].find('[') {
                    let start = enum_idx + arr_start + 1;
                    if let Some(arr_end) = trimmed[start..].find(']') {
                        let items_str = &trimmed[start..start + arr_end];
                        let mut choices = Vec::new();
                        for part in items_str.split(',') {
                            let p = part.trim().trim_matches('"');
                            if !p.is_empty() {
                                choices.push(p.to_string());
                            }
                        }
                        return Ok(SchemaNode::Enum(choices));
                    }
                }
            }
            Ok(SchemaNode::String)
        }
        "integer" => Ok(SchemaNode::Integer),
        "number" => Ok(SchemaNode::Number),
        "boolean" => Ok(SchemaNode::Boolean),
        "array" => {
            let item_node = if let Some(items_idx) = trimmed.find("\"items\"") {
                let rest = &trimmed[items_idx + 7..];
                let after_colon = rest.trim_start().trim_start_matches(':').trim_start();
                parse_json_schema(after_colon)?
            } else {
                SchemaNode::Any
            };
            Ok(SchemaNode::Array(Box::new(item_node)))
        }
        "object" => {
            let mut required = Vec::new();
            if let Some(req_idx) = trimmed.find("\"required\"") {
                if let Some(arr_start) = trimmed[req_idx..].find('[') {
                    let start = req_idx + arr_start + 1;
                    if let Some(arr_end) = trimmed[start..].find(']') {
                        let items_str = &trimmed[start..start + arr_end];
                        for part in items_str.split(',') {
                            let p = part.trim().trim_matches('"');
                            if !p.is_empty() {
                                required.push(p.to_string());
                            }
                        }
                    }
                }
            }

            let mut properties = Vec::new();
            if let Some(prop_idx) = trimmed.find("\"properties\"") {
                let rest = &trimmed[prop_idx + 12..];
                if let Some(obj_start) = rest.find('{') {
                    let props_slice = &rest[obj_start + 1..];
                    let mut i = 0;
                    let bytes = props_slice.as_bytes();
                    while i < bytes.len() {
                        if bytes[i] == b'}' {
                            break;
                        }
                        if bytes[i] == b'"' {
                            let k_start = i + 1;
                            let mut k_end = k_start;
                            while k_end < bytes.len() && bytes[k_end] != b'"' {
                                k_end += 1;
                            }
                            let key = std::str::from_utf8(&bytes[k_start..k_end])
                                .unwrap_or("")
                                .to_string();
                            if let Some(sub_start) = props_slice[k_end..].find('{') {
                                let abs_sub = k_end + sub_start;
                                let mut depth = 0;
                                let mut sub_end = abs_sub;
                                for (idx, &b) in bytes[abs_sub..].iter().enumerate() {
                                    if b == b'{' {
                                        depth += 1;
                                    } else if b == b'}' {
                                        depth -= 1;
                                        if depth == 0 {
                                            sub_end = abs_sub + idx + 1;
                                            break;
                                        }
                                    }
                                }
                                let child_schema =
                                    parse_json_schema(&props_slice[abs_sub..sub_end])?;
                                properties.push((key, child_schema));
                                i = sub_end;
                                continue;
                            }
                        }
                        i += 1;
                    }
                }
            }

            Ok(SchemaNode::Object {
                properties,
                required,
            })
        }
        _ => Ok(SchemaNode::Any),
    }
}

/// Context-Free Grammar byte-state automaton enforcing JSON schema structure.
#[derive(Clone, Debug)]
pub struct SchemaGrammar {
    pub root: SchemaNode,
    pub current_prefix: Vec<u8>,
}

impl SchemaGrammar {
    pub fn new(root: SchemaNode) -> Self {
        Self {
            root,
            current_prefix: Vec::new(),
        }
    }

    /// Compiles a JSON Schema string into a schema grammar automaton.
    pub fn compile(schema_json: &str) -> std::result::Result<Self, String> {
        let node = parse_json_schema(schema_json)?;
        Ok(Self::new(node))
    }

    pub fn is_accepting(&self) -> bool {
        match check_schema_node(&self.root, &self.current_prefix) {
            MatchResult::Complete(len) => {
                let rest = skip_ascii_ws(&self.current_prefix[len..]);
                rest.is_empty()
            }
            _ => false,
        }
    }

    pub fn accepts_prefix(&self, bytes: &[u8]) -> bool {
        let mut test = self.current_prefix.clone();
        test.extend_from_slice(bytes);
        matches!(
            check_schema_node(&self.root, &test),
            MatchResult::Complete(_) | MatchResult::Partial
        )
    }

    pub fn advance(&mut self, bytes: &[u8]) -> bool {
        if self.accepts_prefix(bytes) {
            self.current_prefix.extend_from_slice(bytes);
            true
        } else {
            false
        }
    }
}

/// Unified grammar constraint for logit masking.
#[derive(Clone, Debug)]
pub enum Grammar {
    Json(JsonGrammar),
    Regex(RegexMatcher),
    Choice(ChoiceGrammar),
    Schema(SchemaGrammar),
}

impl Grammar {
    /// Create a grammar enforcing standard RFC 8259 JSON syntax.
    pub fn json() -> Self {
        Self::Json(JsonGrammar::new())
    }

    /// Create a grammar enforcing a regular expression.
    pub fn regex(pattern: &str) -> std::result::Result<Self, RegexError> {
        Ok(Self::Regex(RegexMatcher::new(pattern)?))
    }

    /// Create a grammar enforcing one of an exact set of string choices.
    pub fn choice(options: &[&str]) -> Self {
        Self::Choice(ChoiceGrammar::new(options))
    }

    /// Create a grammar enforcing a JSON Schema specification.
    pub fn schema(schema_json: &str) -> std::result::Result<Self, String> {
        Ok(Self::Schema(SchemaGrammar::compile(schema_json)?))
    }

    /// Check if current state is in an accepting / completed configuration.
    pub fn is_accepting(&self) -> bool {
        match self {
            Self::Json(g) => g.is_accepting(),
            Self::Regex(g) => g.is_accepting(),
            Self::Choice(g) => g.is_accepting(),
            Self::Schema(g) => g.is_accepting(),
        }
    }

    /// Check if appending byte slice is compatible with grammar.
    pub fn accepts_prefix(&self, bytes: &[u8]) -> bool {
        match self {
            Self::Json(g) => g.accepts_prefix(bytes),
            Self::Regex(g) => g.accepts_prefix(bytes),
            Self::Choice(g) => g.accepts_prefix(bytes),
            Self::Schema(g) => g.accepts_prefix(bytes),
        }
    }

    /// Advance grammar state with consumed byte slice.
    pub fn advance(&mut self, bytes: &[u8]) -> bool {
        match self {
            Self::Json(g) => g.advance(bytes),
            Self::Regex(g) => g.advance(bytes),
            Self::Choice(g) => g.advance(bytes),
            Self::Schema(g) => g.advance(bytes),
        }
    }

    /// Filter next-token logits by setting incompatible tokens to -INFINITY.
    pub fn filter_logits(&self, logits: &mut [f32], tokenizer: &Tokenizer) {
        let is_accepting = self.is_accepting();
        let eos_id = tokenizer.eos;

        for (token_id, logit) in logits.iter_mut().enumerate() {
            let id = token_id as u32;

            if Some(id) == eos_id {
                if !is_accepting {
                    *logit = -f32::INFINITY;
                }
                continue;
            }

            if tokenizer.kind_of(id).is_special() {
                *logit = -f32::INFINITY;
                continue;
            }

            let piece = tokenizer.piece(id);
            if piece.is_empty() || !self.accepts_prefix(&piece) {
                *logit = -f32::INFINITY;
            }
        }
    }

    /// Advance grammar state using the token decoded by the tokenizer.
    pub fn advance_token(&mut self, token_id: u32, tokenizer: &Tokenizer) -> bool {
        let piece = tokenizer.piece(token_id);
        if piece.is_empty() {
            true
        } else {
            self.advance(&piece)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_matcher_simple_digits() {
        let mut regex = RegexMatcher::new(r"[0-9]+").unwrap();
        assert!(!regex.is_accepting());
        assert!(regex.accepts_prefix(b"123"));
        assert!(!regex.accepts_prefix(b"abc"));

        assert!(regex.advance(b"42"));
        assert!(regex.is_accepting());
        assert!(regex.advance(b"7"));
        assert!(regex.is_accepting());
        assert!(!regex.advance(b"x"));
    }

    #[test]
    fn test_regex_matcher_alternation() {
        let mut regex = RegexMatcher::new(r"true|false").unwrap();
        assert!(!regex.is_accepting());
        assert!(regex.accepts_prefix(b"tr"));
        assert!(regex.accepts_prefix(b"fal"));
        assert!(!regex.accepts_prefix(b"yes"));

        assert!(regex.advance(b"true"));
        assert!(regex.is_accepting());
    }

    #[test]
    fn test_json_grammar_object() {
        let mut json = JsonGrammar::new();
        assert!(!json.is_accepting());
        assert!(json.advance(b"{\"key\": 123}"));
        assert!(json.is_accepting());
    }

    #[test]
    fn test_json_grammar_array() {
        let mut json = JsonGrammar::new();
        assert!(!json.is_accepting());
        assert!(json.advance(b"[\"hello\", true, null]"));
        assert!(json.is_accepting());
    }

    #[test]
    fn test_choice_grammar() {
        let mut choice = ChoiceGrammar::new(&["apple", "banana", "apricot"]);
        assert!(!choice.is_accepting());
        assert!(choice.accepts_prefix(b"ap"));
        assert!(choice.accepts_prefix(b"ban"));
        assert!(!choice.accepts_prefix(b"orange"));

        assert!(choice.advance(b"apple"));
        assert!(choice.is_accepting());
    }

    #[test]
    fn test_schema_grammar_object_validation() {
        let schema_json = r#"{
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name"]
        }"#;

        let mut grammar = SchemaGrammar::compile(schema_json).expect("compile schema");
        assert!(!grammar.is_accepting());

        assert!(grammar.accepts_prefix(b"{\"name\": \"Alice\""));
        assert!(grammar.advance(b"{\"name\": \"Alice\""));
        assert!(!grammar.is_accepting());

        assert!(grammar.advance(b"}"));
        assert!(grammar.is_accepting());
    }

    #[test]
    fn test_schema_grammar_enum_and_array() {
        let schema_json = r#"{
            "type": "object",
            "properties": {
                "status": { "type": "string", "enum": ["active", "pending", "closed"] }
            },
            "required": ["status"]
        }"#;

        let mut grammar = SchemaGrammar::compile(schema_json).expect("compile");
        assert!(grammar.accepts_prefix(b"{\"status\": \"act"));
        assert!(grammar.accepts_prefix(b"{\"status\": \"active\"}"));
        assert!(!grammar.accepts_prefix(b"{\"status\": \"deleted\"}"));

        assert!(grammar.advance(b"{\"status\": \"active\"}"));
        assert!(grammar.is_accepting());
    }
}
