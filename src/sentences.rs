//! Sentence segmentation for memory typing.
//!
//! The memory-type head (`intent::IntentClassifier::classify_memory_sentences`) was
//! trained on single user sentences, so a memory is classified sentence by sentence.
//! This is a port of the splitter used to build the training data
//! (`evals/intent/wildchat/segment.py`); keep the two in step:
//! - fenced code blocks are dropped; blank lines, list items and lines ending in sentence
//!   punctuation are breaks;
//! - a prose line is rejoined with the next one when the next line starts in lower case
//!   (a hard wrap inside a sentence), but never short lines, code or CJK text;
//! - a sentence ends after `.`, `!` or `?` (plus closing quotes or brackets) before an
//!   upper-case letter, digit or opening quote, but not after a common abbreviation or a
//!   single initial; a lower-case word running straight into a capital ("word.Next") is
//!   also split;
//! - a quote mark or bracket left unbalanced at either end is dropped.
//! Only sentences of 3 to 60 words that are mostly letters are kept.

use regex::Regex;
use std::sync::OnceLock;

pub const MIN_WORDS: usize = 3;
pub const MAX_WORDS: usize = 60;
const WRAP_MIN_CHARS: usize = 28;
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "e.g", "i.e", "fig",
    "figs", "inc", "ltd", "co", "corp", "approx", "dept", "est", "vol", "ref", "refs", "al",
    "cf", "ca", "no", "nos", "p", "pp", "sec", "ch", "u.s", "u.k", "a.m", "p.m", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
];
const CLOSERS: &[char] = &['"', '\'', '”', '’', ')', ']'];
const OPENERS: &[char] = &['"', '\'', '“', '‘', '(', '['];

fn fence() -> &'static Regex {
    static FENCE: OnceLock<Regex> = OnceLock::new();
    FENCE.get_or_init(|| Regex::new(r"```[\s\S]*?```").expect("valid regex"))
}

fn paragraph_break() -> &'static Regex {
    static BREAK: OnceLock<Regex> = OnceLock::new();
    BREAK.get_or_init(|| Regex::new(r"\n\s*\n").expect("valid regex"))
}

fn list_item() -> &'static Regex {
    static LIST: OnceLock<Regex> = OnceLock::new();
    LIST.get_or_init(|| Regex::new(r"^([-*•·>#]|\d+[.)]|[a-zA-Z][.)]\s)").expect("valid regex"))
}

fn code_like() -> &'static Regex {
    static CODE: OnceLock<Regex> = OnceLock::new();
    CODE.get_or_init(|| {
        Regex::new(r"[{};=<>\[\]]|//|\w\(|^(#|def |class |import |return |if |for |while )")
            .expect("valid regex")
    })
}

fn is_cjk(c: char) -> bool {
    matches!(c, '\u{3000}'..='\u{9fff}' | '\u{ac00}'..='\u{d7af}')
}

fn ends_with_sentence_punctuation(line: &str) -> bool {
    let trimmed = line.trim_end_matches(CLOSERS);
    matches!(trimmed.chars().last(), Some('.' | '!' | '?' | ':' | ';'))
}

/// A hard wrap: a prose line runs on without sentence punctuation and the next line
/// continues it in lower case.
fn joins(previous: &str, line: &str) -> bool {
    line.chars().next().is_some_and(char::is_lowercase)
        && !ends_with_sentence_punctuation(previous)
        && previous.chars().count() >= WRAP_MIN_CHARS
        && previous.split_whitespace().count() >= 4
        && !list_item().is_match(line)
        && !code_like().is_match(previous)
        && !code_like().is_match(line)
        && !previous.chars().any(is_cjk)
}

/// Byte offsets after which `block` should be cut.
fn boundaries(block: &str) -> Vec<usize> {
    let chars: Vec<(usize, char)> = block.char_indices().collect();
    let mut cuts = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !matches!(chars[i].1, '.' | '!' | '?') {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && matches!(chars[i].1, '.' | '!' | '?') {
            i += 1;
        }
        while i < chars.len() && CLOSERS.contains(&chars[i].1) {
            i += 1;
        }
        let end = i; // first char after the stop and its closers
        let mut next = end;
        while next < chars.len() && chars[next].1.is_whitespace() {
            next += 1;
        }
        if next >= chars.len() {
            continue;
        }
        let spaced = next > end;
        let head = chars[next].1;
        let before: Vec<char> = chars[..start].iter().map(|(_, c)| *c).collect();
        if spaced {
            if !(head.is_uppercase() || head.is_ascii_digit() || OPENERS.contains(&head)) {
                continue;
            }
        } else {
            let two_lower = before.len() >= 2
                && before[before.len() - 1].is_ascii_lowercase()
                && before[before.len() - 2].is_ascii_lowercase();
            if !(head.is_uppercase() && two_lower) {
                continue;
            }
        }
        if chars[start].1 == '.' {
            // The word right before the stop, if it starts a token: an abbreviation or
            // a single initial keeps the sentence going.
            let mut j = before.len();
            while j > 0 && (before[j - 1].is_ascii_alphabetic() || before[j - 1] == '.') {
                j -= 1;
            }
            let starts_token =
                j == 0 || before[j - 1].is_whitespace() || matches!(before[j - 1], '(' | '[' | '"' | '“');
            if j < before.len() && before[j].is_ascii_alphabetic() && starts_token {
                let token: String = before[j..].iter().collect::<String>().to_ascii_lowercase();
                let abbreviation = ABBREVIATIONS.contains(&token.as_str())
                    && (token != "no" || head.is_ascii_digit());
                if token.chars().count() == 1 || abbreviation {
                    continue;
                }
            }
        }
        cuts.push(if end < chars.len() { chars[end].0 } else { block.len() });
    }
    cuts
}

/// Drops a quote mark or bracket left unbalanced at either end.
fn tidy(sentence: &str) -> String {
    let mut s = sentence.trim().to_string();
    for _ in 0..3 {
        let before = s.clone();
        for (opening, closing) in [('"', '"'), ('“', '”'), ('(', ')'), ('[', ']')] {
            let opens = s.matches(opening).count();
            let closes = s.matches(closing).count();
            if opening == closing {
                if opens % 2 == 1 {
                    if s.starts_with(opening) {
                        s = s[opening.len_utf8()..].trim_start().to_string();
                    } else if s.ends_with(closing) {
                        s = s[..s.len() - closing.len_utf8()].trim_end().to_string();
                    }
                }
            } else {
                if s.starts_with(opening) && opens > closes {
                    s = s[opening.len_utf8()..].trim_start().to_string();
                }
                let opens = s.matches(opening).count();
                let closes = s.matches(closing).count();
                if s.ends_with(closing) && closes > opens {
                    s = s[..s.len() - closing.len_utf8()].trim_end().to_string();
                }
            }
        }
        if s == before {
            break;
        }
    }
    s
}

/// Every sentence of `text`, in order (no length filter).
pub fn split(text: &str) -> Vec<String> {
    let text = fence().replace_all(text, "\n\n");
    let mut out = Vec::new();
    for paragraph in paragraph_break().split(&text) {
        let mut blocks: Vec<String> = Vec::new();
        for raw in paragraph.split('\n') {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            match blocks.last_mut() {
                Some(previous) if joins(previous, line) => {
                    previous.push(' ');
                    previous.push_str(line);
                }
                _ => blocks.push(line.to_string()),
            }
        }
        for block in blocks {
            let mut start = 0;
            for cut in boundaries(&block) {
                out.push(tidy(&block[start..cut]));
                start = cut;
            }
            out.push(tidy(&block[start..]));
        }
    }
    out.retain(|sentence| !sentence.is_empty());
    out
}

/// Sentences worth classifying: 3 to 60 words, mostly letters.
pub fn sentences(text: &str) -> Vec<String> {
    split(text)
        .into_iter()
        .filter(|sentence| {
            let words = sentence.split_whitespace().count();
            let letters = sentence.chars().filter(|c| c.is_alphabetic()).count();
            (MIN_WORDS..=MAX_WORDS).contains(&words) && letters * 10 >= sentence.chars().count() * 6
        })
        .collect()
}

#[cfg(test)]
#[path = "../tests/unit/sentences.rs"]
mod tests;
