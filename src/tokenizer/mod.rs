//! Tokenizers reconstructed from GGUF metadata.
//!
//! A GGUF file carries its own vocabulary, so nothing has to be shipped
//! alongside the weights and there is no `tokenizer.json` to keep in sync. Two
//! families cover essentially every model in circulation:
//!
//! - `gpt2`: byte-level BPE with an explicit merge list. Llama 3, Qwen, Phi.
//! - `llama`: SentencePiece, driven by per-token scores. Llama 2, Mistral, Gemma.
//!
//! Both are implemented here. Whichever the file declares is what gets used.

use std::collections::HashMap;
use std::io::{Error, ErrorKind, Result};

use crate::gguf::Gguf;

pub mod bytes;

fn bad(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// SentencePiece: merge by score, spaces become U+2581.
    Spm,
    /// Byte-level BPE: merge by rank from an explicit merge list.
    Bpe,
}

/// GGUF's `tokenizer.ggml.token_type` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenKind {
    fn from_id(id: u64) -> Self {
        match id {
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }

    /// Tokens that must never be produced by merging text — only by an exact
    /// match against the literal token, or by the caller asking for them.
    pub fn is_special(self) -> bool {
        matches!(self, Self::Control | Self::UserDefined)
    }
}

pub struct Tokenizer {
    pub kind: Kind,
    pub tokens: Vec<String>,
    pub kinds: Vec<TokenKind>,
    pub scores: Vec<f32>,
    ids: HashMap<String, u32>,
    /// `<0x0A>` style tokens, for text SentencePiece cannot segment.
    byte_fallback: HashMap<u8, u32>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub unknown: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
}

impl Tokenizer {
    pub fn from_gguf(model: &Gguf) -> Result<Self> {
        let kind = match model.meta_str("tokenizer.ggml.model") {
            Some("llama") => Kind::Spm,
            Some("gpt2") => Kind::Bpe,
            Some(other) => {
                return Err(bad(format!(
                    "tokenizer model {other:?} is not supported (llama and gpt2 vocabs only)"
                )))
            }
            None => return Err(bad("file has no tokenizer.ggml.model")),
        };

        let tokens: Vec<String> = model
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_strings())
            .ok_or_else(|| bad("file has no tokenizer.ggml.tokens"))?
            .into_iter()
            .map(str::to_owned)
            .collect();
        if tokens.is_empty() {
            return Err(bad("vocabulary is empty"));
        }

        let scores = model
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_f32s())
            .unwrap_or_else(|| vec![0.0; tokens.len()]);
        if scores.len() != tokens.len() {
            return Err(bad("tokenizer scores and tokens have different lengths"));
        }

        let kinds: Vec<TokenKind> = match model
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
        {
            Some(values) => {
                if values.len() != tokens.len() {
                    return Err(bad("token_type and tokens have different lengths"));
                }
                values
                    .iter()
                    .map(|v| TokenKind::from_id(v.as_u64().unwrap_or(1)))
                    .collect()
            }
            None => vec![TokenKind::Normal; tokens.len()],
        };

        // First id wins: a duplicate token later in the vocab is unreachable,
        // which matches how the reference implementations behave.
        let mut ids = HashMap::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            ids.entry(token.clone()).or_insert(id as u32);
        }

        let mut byte_fallback = HashMap::new();
        for byte in 0..=255u8 {
            if let Some(id) = ids.get(&format!("<0x{byte:02X}>")) {
                byte_fallback.insert(byte, *id);
            }
        }

        let special = |key: &str| model.meta_u64(key).and_then(|v| u32::try_from(v).ok());
        let bos = special("tokenizer.ggml.bos_token_id");
        let eos = special("tokenizer.ggml.eos_token_id");
        let unknown = special("tokenizer.ggml.unknown_token_id");

        // SentencePiece models prepend BOS unless told otherwise; byte-level
        // ones are explicit about it.
        let add_bos = model
            .meta_u64("tokenizer.ggml.add_bos_token")
            .map(|v| v != 0)
            .unwrap_or(kind == Kind::Spm);
        let add_eos = model
            .meta_u64("tokenizer.ggml.add_eos_token")
            .map(|v| v != 0)
            .unwrap_or(false);

        Ok(Self {
            kind,
            tokens,
            kinds,
            scores,
            ids,
            byte_fallback,
            bos,
            eos,
            unknown,
            add_bos,
            add_eos,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn kind_of(&self, id: u32) -> TokenKind {
        self.kinds
            .get(id as usize)
            .copied()
            .unwrap_or(TokenKind::Normal)
    }

    pub fn score(&self, id: u32) -> f32 {
        self.scores.get(id as usize).copied().unwrap_or(0.0)
    }

    /// Id of the `<0xNN>` token for a byte, if the vocab has one.
    pub fn byte_id(&self, byte: u8) -> Option<u32> {
        self.byte_fallback.get(&byte).copied()
    }

    /// Raw bytes for one token. Not necessarily valid UTF-8 on its own — a
    /// multi-byte character can be split across tokens, which is exactly why
    /// streaming output has to accumulate bytes rather than decode per token.
    pub fn piece(&self, id: u32) -> Vec<u8> {
        let Some(token) = self.token(id) else {
            return Vec::new();
        };
        match self.kind {
            Kind::Bpe => bytes::decode_str(token),
            Kind::Spm => {
                if self.kind_of(id) == TokenKind::Byte {
                    // "<0x0A>" -> 0x0A
                    if let Some(hex) = token
                        .strip_prefix("<0x")
                        .and_then(|rest| rest.strip_suffix('>'))
                    {
                        if let Ok(byte) = u8::from_str_radix(hex, 16) {
                            return vec![byte];
                        }
                    }
                }
                token.replace('\u{2581}', " ").into_bytes()
            }
        }
    }

    /// Ids back to text. Control tokens are dropped rather than printed.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out: Vec<u8> = Vec::new();
        for (position, id) in ids.iter().enumerate() {
            if self.kind_of(*id) == TokenKind::Control {
                continue;
            }
            let piece = self.piece(*id);
            // SentencePiece encodes with a dummy space in front of the text;
            // decoding has to take it back off.
            if position == 0 && self.kind == Kind::Spm && piece.first() == Some(&b' ') {
                out.extend_from_slice(&piece[1..]);
            } else {
                out.extend_from_slice(&piece);
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}
