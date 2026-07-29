//! Byte-level BPE: pretokenization and the merge loop.
//!
//! Merges are ranked by their position in the file's merge list. Encoding
//! repeatedly applies the lowest-ranked applicable merge until none is left,
//! which is what makes the result reproducible across implementations.

use std::collections::HashMap;

/// `(left id, right id) -> (rank, merged id)`.
pub type Merges = HashMap<(u32, u32), (u32, u32)>;

/// Build the merge table from `tokenizer.ggml.merges`, whose entries look like
/// `"Ġt he"` — two token strings separated by a space. Space itself is stored
/// as `Ġ` in these vocabs, so splitting on the first literal space is
/// unambiguous.
pub fn load_merges(list: &[&str], ids: &HashMap<String, u32>) -> Merges {
    let mut merges = Merges::with_capacity(list.len());
    for (rank, entry) in list.iter().enumerate() {
        let Some((left, right)) = entry.split_once(' ') else {
            continue;
        };
        let (Some(&l), Some(&r)) = (ids.get(left), ids.get(right)) else {
            continue;
        };
        // A merge whose result is not itself in the vocab is unusable.
        let Some(&merged) = ids.get(&format!("{left}{right}")) else {
            continue;
        };
        merges.entry((l, r)).or_insert((rank as u32, merged));
    }
    merges
}

/// Collapse `symbols` in place, lowest rank first.
pub fn apply_merges(merges: &Merges, symbols: &mut Vec<u32>) {
    // Rescanning is O(n^2) in the length of one pretoken, and a pretoken is a
    // word. A priority queue here measures slower than the rescan because n is
    // single digits in almost every case.
    loop {
        let mut best: Option<(usize, u32, u32)> = None;
        for i in 0..symbols.len().saturating_sub(1) {
            if let Some(&(rank, merged)) = merges.get(&(symbols[i], symbols[i + 1])) {
                if best.is_none_or(|(_, best_rank, _)| rank < best_rank) {
                    best = Some((i, rank, merged));
                }
            }
        }
        let Some((at, _, merged)) = best else { return };
        symbols[at] = merged;
        symbols.remove(at + 1);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Digit,
    Other,
}

fn class_of(ch: char) -> Class {
    if ch.is_alphabetic() {
        Class::Letter
    } else if ch.is_numeric() {
        Class::Digit
    } else {
        Class::Other
    }
}

/// Contractions the GPT-2 pattern splits off explicitly.
const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

/// Split text into the chunks BPE merges within. Merges never cross a chunk
/// boundary, so this decides tokenization as much as the merge table does.
///
/// This reproduces the GPT-2 pattern's *behavior* — a single leading space
/// attaches to the following word, runs of letters, digits and punctuation
/// split apart, and a whitespace run keeps all but its last character — without
/// a regex engine. Exact id-for-id parity with a specific HuggingFace tokenizer
/// needs that model's own pattern; if a model ever turns out to care, this is
/// the function to specialize.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let end_of = |index: usize| chars.get(index).map_or(text.len(), |(at, _)| *at);

    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let (start, ch) = chars[i];

        if ch == '\'' {
            if let Some(found) = CONTRACTIONS
                .iter()
                .find(|c| text[start..].starts_with(**c))
            {
                out.push(&text[start..start + found.len()]);
                i += found.chars().count();
                continue;
            }
        }

        if ch.is_whitespace() {
            let mut run_end = i;
            while run_end < chars.len() && chars[run_end].1.is_whitespace() {
                run_end += 1;
            }
            let trailing_text = run_end < chars.len();
            if trailing_text {
                // Everything but the final space is its own chunk; the final
                // space belongs to the word that follows.
                if run_end - i > 1 {
                    out.push(&text[start..chars[run_end - 1].0]);
                }
                let attach = chars[run_end - 1].0;
                i = run_end;
                let class = class_of(chars[i].1);
                while i < chars.len()
                    && !chars[i].1.is_whitespace()
                    && class_of(chars[i].1) == class
                {
                    i += 1;
                }
                out.push(&text[attach..end_of(i)]);
            } else {
                out.push(&text[start..end_of(run_end)]);
                i = run_end;
            }
            continue;
        }

        let class = class_of(ch);
        while i < chars.len() && !chars[i].1.is_whitespace() && class_of(chars[i].1) == class {
            i += 1;
        }
        out.push(&text[start..end_of(i)]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_space_attaches_to_the_word() {
        assert_eq!(pretokenize("hello world"), vec!["hello", " world"]);
        assert_eq!(pretokenize(" hi"), vec![" hi"]);
    }

    #[test]
    fn classes_split_apart() {
        assert_eq!(pretokenize("abc123!?"), vec!["abc", "123", "!?"]);
        assert_eq!(pretokenize("x = 42;"), vec!["x", " =", " 42", ";"]);
    }

    #[test]
    fn whitespace_runs_keep_all_but_the_last() {
        assert_eq!(pretokenize("a   b"), vec!["a", "  ", " b"]);
        // Trailing whitespace has no word to attach to, so it stands alone.
        assert_eq!(pretokenize("a  "), vec!["a", "  "]);
        assert_eq!(pretokenize("\n\nx"), vec!["\n", "\nx"]);
    }

    #[test]
    fn contractions_split_off() {
        assert_eq!(pretokenize("don't"), vec!["don", "'t"]);
        assert_eq!(pretokenize("I'll go"), vec!["I", "'ll", " go"]);
    }

    #[test]
    fn pretokens_partition_the_input() {
        // The property that matters: nothing is lost or duplicated, so
        // decode(encode(x)) can round-trip.
        for text in ["hello world", "a   b", "don't stop", "日本 語", "x=1;\n\ty"] {
            assert_eq!(pretokenize(text).concat(), text, "lost bytes in {text:?}");
        }
    }

    #[test]
    fn merges_apply_by_rank() {
        let mut ids = HashMap::new();
        for (i, t) in ["a", "b", "c", "ab", "bc", "abc"].iter().enumerate() {
            ids.insert(t.to_string(), i as u32);
        }
        // "b c" is listed first, so it outranks "a b".
        let merges = load_merges(&["b c", "a b", "ab c"], &ids);

        let mut symbols = vec![0, 1, 2]; // a b c
        apply_merges(&merges, &mut symbols);
        // bc wins first -> [a, bc]; no merge exists for (a, bc), so it stops.
        assert_eq!(symbols, vec![0, 4]);

        let merges = load_merges(&["a b", "ab c"], &ids);
        let mut symbols = vec![0, 1, 2];
        apply_merges(&merges, &mut symbols);
        assert_eq!(symbols, vec![5], "should collapse to abc");
    }
}
