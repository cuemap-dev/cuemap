use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

// Simple stopword list
static STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_stopwords() -> &'static HashSet<&'static str> {
    STOPWORDS.get_or_init(|| {
        ["the", "is", "at", "which", "on", "in", "a", "an", "and", "or", "for", "to", "of", "it", "this", "that"].into_iter().collect()
    })
}

fn get_token_regex() -> &'static Regex {
    TOKEN_REGEX.get_or_init(|| {
        Regex::new(r"[a-z0-9]+").unwrap()
    })
}

pub fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn tokenize_to_cues(text: &str) -> Vec<String> {
    let normalized = normalize_text(text);
    let mut cues = Vec::new();
    let mut tokens = Vec::new();
    
    // Extract tokens
    for token in get_token_regex().find_iter(&normalized) {
        let t = token.as_str();
        if !get_stopwords().contains(t) && t.len() > 1 {
            tokens.push(t.to_string());
            cues.push(format!("tok:{}", t));
        }
    }
    
    // Extract bigrams (phrases)
    if tokens.len() >= 2 {
        for windows in tokens.windows(2) {
            cues.push(format!("phr:{}_{}", windows[0], windows[1]));
        }
    }
    
    cues
}
