//! SentencePiece-style tokenization for `llama` vocabs.
//!
//! Where BPE ranks merges by position in a list, SentencePiece scores every
//! token and merges the highest-scoring adjacent pair first. The algorithm is
//! the one in llama.cpp: a doubly linked list of symbols plus a max-heap of
//! candidate merges, re-seeded around each merge.
//!
//! Matching the reference matters more than elegance here — a tokenizer that
//! disagrees with the one a model was trained with degrades output in ways that
//! look like a bad model rather than a bug.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::Tokenizer;

/// SentencePiece's stand-in for a space, U+2581 LOWER ONE EIGHTH BLOCK.
pub const SPACE: char = '\u{2581}';

struct Symbol {
    prev: i32,
    next: i32,
    start: usize,
    /// Zero once the symbol has been merged into its left neighbour.
    len: usize,
}

struct Candidate {
    left: i32,
    right: i32,
    score: f32,
    /// Combined byte length when queued, used to detect stale entries.
    size: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    /// Highest score first; ties broken toward the leftmost pair, matching the
    /// reference implementation's ordering.
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

/// Prepend the dummy prefix and swap spaces for U+2581, the form the vocab was
/// built against.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + SPACE.len_utf8());
    out.push(SPACE);
    for ch in text.chars() {
        out.push(if ch == ' ' { SPACE } else { ch });
    }
    out
}

fn enqueue(
    heap: &mut BinaryHeap<Candidate>,
    symbols: &[Symbol],
    text: &str,
    tok: &Tokenizer,
    left: i32,
    right: i32,
) {
    if left < 0 || right < 0 {
        return;
    }
    let (l, r) = (&symbols[left as usize], &symbols[right as usize]);
    if l.len == 0 || r.len == 0 {
        return;
    }
    let piece = &text[l.start..r.start + r.len];
    if let Some(id) = tok.id(piece) {
        heap.push(Candidate {
            left,
            right,
            score: tok.score(id),
            size: piece.len(),
        });
    }
}

pub fn encode(tok: &Tokenizer, text: &str) -> Vec<u32> {
    let normalized = normalize(text);
    if normalized.is_empty() {
        return Vec::new();
    }

    let mut symbols: Vec<Symbol> = Vec::new();
    for (start, ch) in normalized.char_indices() {
        let index = symbols.len() as i32;
        let len = ch.len_utf8();
        let last = start + len >= normalized.len();
        symbols.push(Symbol {
            prev: index - 1,
            next: if last { -1 } else { index + 1 },
            start,
            len,
        });
    }

    let mut heap = BinaryHeap::new();
    for index in 1..symbols.len() as i32 {
        enqueue(&mut heap, &symbols, &normalized, tok, index - 1, index);
    }

    while let Some(candidate) = heap.pop() {
        let (left, right) = (candidate.left as usize, candidate.right as usize);
        if symbols[left].len == 0 || symbols[right].len == 0 {
            continue;
        }
        // Both symbols are still the size they were when this pair was queued;
        // otherwise a different merge got there first and this entry is stale.
        if symbols[left].len + symbols[right].len != candidate.size {
            continue;
        }

        symbols[left].len += symbols[right].len;
        symbols[right].len = 0;
        symbols[left].next = symbols[right].next;
        if symbols[right].next >= 0 {
            let next = symbols[right].next as usize;
            symbols[next].prev = candidate.left;
        }

        let (prev, next) = (symbols[left].prev, symbols[left].next);
        enqueue(&mut heap, &symbols, &normalized, tok, prev, candidate.left);
        enqueue(&mut heap, &symbols, &normalized, tok, candidate.left, next);
    }

    let mut out = Vec::new();
    let mut index = 0i32;
    while index >= 0 {
        let symbol = &symbols[index as usize];
        if symbol.len > 0 {
            let piece = &normalized[symbol.start..symbol.start + symbol.len];
            match tok.id(piece) {
                Some(id) => out.push(id),
                // Unsegmentable text falls back to one token per byte, which is
                // why these vocabs carry <0x00>..<0xFF>.
                None => {
                    for byte in piece.bytes() {
                        if let Some(id) = tok.byte_id(byte) {
                            out.push(id);
                        } else if let Some(unknown) = tok.unknown {
                            out.push(unknown);
                        }
                    }
                }
            }
        }
        index = symbol.next;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_adds_prefix_and_replaces_spaces() {
        assert_eq!(normalize("a b"), "\u{2581}a\u{2581}b");
        assert_eq!(normalize(""), "\u{2581}");
    }

    #[test]
    fn candidate_order_prefers_high_score_then_left() {
        let mut heap = BinaryHeap::new();
        heap.push(Candidate { left: 5, right: 6, score: 1.0, size: 2 });
        heap.push(Candidate { left: 0, right: 1, score: 9.0, size: 2 });
        heap.push(Candidate { left: 2, right: 3, score: 9.0, size: 2 });
        assert_eq!(heap.pop().unwrap().left, 0, "highest score, leftmost");
        assert_eq!(heap.pop().unwrap().left, 2);
        assert_eq!(heap.pop().unwrap().left, 5);
    }
}
