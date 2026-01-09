use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

// Simple stopword list
static STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();

pub fn get_stopwords() -> &'static HashSet<&'static str> {
    STOPWORDS.get_or_init(|| {
        [
  "a", "about", "above", "am", "an", "and", "any", "are", "aren't", "as", "at", 
  "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
  "can't", "cannot", "could", "couldn't", 
  "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
  "each", "few", "for", "from", "further", 
  "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", 
  "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
  "let's", 
  "me", "more", "most", "mustn't", "my", "myself",
  "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
  "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", 
  "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", 
  "under", "until", "up", "us", 
  "very", 
  "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "will", "with", "won't", "would", "wouldn't", 
  "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
].into_iter().collect()
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
            cues.push(t.to_string());
        }
    }
    
    // Extract bigrams (phrases)
    if tokens.len() >= 2 {
        for windows in tokens.windows(2) {
            cues.push(format!("{}_{}", windows[0], windows[1]));
        }
    }
    
    cues
}
