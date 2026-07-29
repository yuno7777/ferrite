//! The GPT-2 byte alphabet.
//!
//! Byte-level BPE vocabs cannot store raw bytes as token text, because a token
//! has to be valid UTF-8 to live in a JSON or GGUF string. So every byte is
//! mapped to a printable code point: byte 0x20 is stored as U+0120 `Ġ`, which
//! is why tokens in those vocabs look like `Ġhello`.
//!
//! Printable ASCII and most of Latin-1 map to themselves; the remaining 68
//! bytes are assigned to U+0100 upward in byte order.

use std::collections::HashMap;
use std::sync::OnceLock;

fn tables() -> &'static ([char; 256], HashMap<char, u8>) {
    static TABLES: OnceLock<([char; 256], HashMap<char, u8>)> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut forward = ['\0'; 256];
        let mut reverse = HashMap::with_capacity(256);
        let mut spill = 0u32;
        for byte in 0u32..256 {
            let printable = (0x21..=0x7E).contains(&byte)
                || (0xA1..=0xAC).contains(&byte)
                || (0xAE..=0xFF).contains(&byte);
            let code = if printable {
                byte
            } else {
                let code = 256 + spill;
                spill += 1;
                code
            };
            let ch = char::from_u32(code).expect("code point is valid");
            forward[byte as usize] = ch;
            reverse.insert(ch, byte as u8);
        }
        (forward, reverse)
    })
}

/// Byte to its stored code point.
pub fn encode_byte(byte: u8) -> char {
    tables().0[byte as usize]
}

/// Stored code point back to the byte, or `None` for a character that is not
/// part of the alphabet at all — a control token like `<|eot_id|>` contains
/// only ASCII, so this still succeeds there.
pub fn decode_char(ch: char) -> Option<u8> {
    tables().1.get(&ch).copied()
}

/// Text to the byte-mapped form a byte-level vocab stores.
pub fn encode_str(text: &str) -> String {
    text.bytes().map(encode_byte).collect()
}

/// Byte-mapped form back to raw bytes. Characters outside the alphabet are
/// passed through as their own UTF-8 encoding, which is what makes decoding a
/// vocab that mixes byte tokens and literal special tokens work.
pub fn decode_str(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        match decode_char(ch) {
            Some(byte) => out.push(byte),
            None => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_is_g_with_dot_above() {
        // The single most recognizable fact about these vocabs.
        assert_eq!(encode_byte(b' '), '\u{0120}');
        assert_eq!(decode_char('\u{0120}'), Some(b' '));
        assert_eq!(encode_str(" hello"), "\u{0120}hello");
    }

    #[test]
    fn every_byte_round_trips_and_is_unique() {
        let mut seen = std::collections::HashSet::new();
        for byte in 0..=255u8 {
            let ch = encode_byte(byte);
            assert!(seen.insert(ch), "byte {byte} collides on {ch:?}");
            assert_eq!(decode_char(ch), Some(byte));
        }
    }

    #[test]
    fn utf8_survives_the_round_trip() {
        for text in ["héllo", "日本語", "a\nb\tc", "🦀"] {
            assert_eq!(decode_str(&encode_str(text)), text.as_bytes());
        }
    }
}
