//! Continuous streaming detokenizer with UTF-8 byte boundary preservation.

use super::{TokenKind, Tokenizer};

/// Configuration options for streaming detokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetokenizerOptions {
    /// Whether to omit special and control tokens from decoded output.
    pub skip_special_tokens: bool,
}

impl Default for DetokenizerOptions {
    fn default() -> Self {
        Self {
            skip_special_tokens: true,
        }
    }
}

/// Continuous streaming detokenizer that preserves UTF-8 code point boundaries
/// across arbitrary token boundaries and byte fallbacks.
#[derive(Debug, Clone)]
pub struct StreamingDetokenizer<'a> {
    tokenizer: &'a Tokenizer,
    options: DetokenizerOptions,
    buffer: Vec<u8>,
}

impl<'a> StreamingDetokenizer<'a> {
    /// Creates a new streaming detokenizer with default options.
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self::with_options(tokenizer, DetokenizerOptions::default())
    }

    /// Creates a new streaming detokenizer with custom options.
    pub fn with_options(tokenizer: &'a Tokenizer, options: DetokenizerOptions) -> Self {
        Self {
            tokenizer,
            options,
            buffer: Vec::new(),
        }
    }

    /// Clears any pending bytes without emitting them.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Returns the number of currently buffered incomplete bytes.
    pub fn buffered_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Feeds a single token ID into the detokenizer and returns any newly
    /// completed valid UTF-8 string chunk.
    pub fn step(&mut self, token_id: u32) -> Option<String> {
        if self.options.skip_special_tokens {
            let kind = self.tokenizer.kind_of(token_id);
            if matches!(kind, TokenKind::Control | TokenKind::Unknown) {
                return None;
            }
            if Some(token_id) == self.tokenizer.bos || Some(token_id) == self.tokenizer.eos {
                return None;
            }
        }

        let piece = self.tokenizer.piece(token_id);
        if piece.is_empty() {
            return None;
        }

        self.feed_bytes(&piece)
    }

    /// Feeds raw bytes into the buffer and extracts the longest valid prefix.
    pub fn feed_bytes(&mut self, bytes: &[u8]) -> Option<String> {
        self.buffer.extend_from_slice(bytes);
        self.extract_valid_chunk()
    }

    /// Flushes any remaining bytes in the buffer as lossy UTF-8.
    pub fn finish(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            None
        } else {
            let text = String::from_utf8_lossy(&self.buffer).into_owned();
            self.buffer.clear();
            Some(text)
        }
    }

    /// Detokenizes an entire slice of token IDs in one pass.
    pub fn detokenize(&mut self, tokens: &[u32]) -> String {
        let mut out = String::new();
        for &id in tokens {
            if let Some(chunk) = self.step(id) {
                out.push_str(&chunk);
            }
        }
        if let Some(tail) = self.finish() {
            out.push_str(&tail);
        }
        out
    }

    fn extract_valid_chunk(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }

        let mut out = String::new();
        loop {
            if self.buffer.is_empty() {
                break;
            }
            match std::str::from_utf8(&self.buffer) {
                Ok(_) => {
                    let s = String::from_utf8(std::mem::take(&mut self.buffer)).unwrap();
                    out.push_str(&s);
                    break;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to > 0 {
                        let valid_slice = self.buffer.drain(..valid_up_to).collect::<Vec<u8>>();
                        let s = String::from_utf8(valid_slice).unwrap();
                        out.push_str(&s);
                    }
                    if let Some(err_len) = err.error_len() {
                        self.buffer.drain(..err_len);
                        out.push('\u{FFFD}');
                    } else {
                        // Incomplete sequence at the tail of the buffer; wait for more bytes.
                        break;
                    }
                }
            }
        }

        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}
