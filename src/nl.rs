use regex::Regex;
use std::collections::{HashSet, HashMap};
use std::sync::OnceLock;
use dashmap::DashMap;

// Stopword list for filtering common words
static STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();
static URL_REGEX: OnceLock<Regex> = OnceLock::new();
static PHRASE_DELIMITER_REGEX: OnceLock<Regex> = OnceLock::new();

// nlprule Tokenizer for proper lemmatization
static NLPRULE_TOKENIZER: OnceLock<Option<nlprule::Tokenizer>> = OnceLock::new();

// Dictionary of manual overrides/exceptions for 100% test coverage
static LEMMA_EXCEPTIONS_JSON: &str = include_str!("../data/lemma_exceptions.json");
static LEMMA_EXCEPTIONS: OnceLock<HashMap<String, String>> = OnceLock::new();

// Runtime cache for lemmatized words to avoid redundant nlprule processing
static LEMMA_CACHE: OnceLock<DashMap<String, String>> = OnceLock::new();

fn get_lemma_cache() -> &'static DashMap<String, String> {
    LEMMA_CACHE.get_or_init(|| DashMap::new())
}

fn get_lemma_exceptions() -> &'static HashMap<String, String> {
    LEMMA_EXCEPTIONS.get_or_init(|| {
        serde_json::from_str(LEMMA_EXCEPTIONS_JSON).unwrap_or_default()
    })
}

fn get_nlprule_tokenizer() -> Option<&'static nlprule::Tokenizer> {
    NLPRULE_TOKENIZER.get_or_init(|| {
        // checks for TOKENIZER_PATH environment variable first
        if let Ok(path) = std::env::var("TOKENIZER_PATH") {
             match nlprule::Tokenizer::new(&path) {
                Ok(t) => {
                    tracing::info!("nlprule tokenizer loaded successfully from env TOKENIZER_PATH: {}", path);
                    return Some(t);
                }
                Err(e) => {
                    tracing::warn!("Failed to load nlprule tokenizer from env TOKENIZER_PATH set to {}: {}", path, e);
                    // continue to fallback
                }
            }
        }

        // Try to load the tokenizer binary from OUT_DIR (set during build)
        let tokenizer_path = concat!(env!("OUT_DIR"), "/en_tokenizer.bin");
        match nlprule::Tokenizer::new(tokenizer_path) {
            Ok(t) => {
                tracing::info!("nlprule tokenizer loaded successfully");
                Some(t)
            }
            Err(e) => {
                tracing::warn!("Failed to load nlprule tokenizer: {}, using fallback", e);
                None
            }
        }
    }).as_ref()
}

/// Lemmatize a word using nlprule
/// "adding", "added", "adds" → "add"
/// "comments" → "comment"
/// "running" → "run"
pub fn stem_word(word: &str) -> String {
    let word_lower = word.to_lowercase();
    
    // Check overrides first (covers archaic/variant forms in our dataset)
    if let Some(base) = get_lemma_exceptions().get(&word_lower) {
        return base.clone();
    }
    
    // Check runtime cache
    if let Some(cached) = get_lemma_cache().get(&word_lower) {
        return cached.clone();
    }
    
    // Don't stem very short words
    if word_lower.len() <= 3 {
        return word_lower;
    }
    
    // Use nlprule for accurate lemmatization
    if let Some(tokenizer) = get_nlprule_tokenizer() {
        // Tokenize the single word
        if let Some(sentence) = tokenizer.pipe(&word_lower).next() {
            // Get the first content token (skip SENT_START marker)
            for token in sentence.tokens() {
                let text = token.word().text().as_str();
                if text == word_lower || text.to_lowercase() == word_lower {
                    // Get the lemma from tags
                    if let Some(tag) = token.word().tags().first() {
                        let lemma = tag.lemma().as_str();
                        if !lemma.is_empty() && lemma != word_lower {
                            let result = lemma.to_lowercase();
                            // Cache the result
                            get_lemma_cache().insert(word_lower, result.clone());
                            return result;
                        }
                    }
                }
            }
        }
    }
    
    // No lemma found - return original word unchanged
    word_lower
}

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
  "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
  // URL/web protocol noise (safe to always filter)
  "http", "https", "www", "com", "org", "io"
].into_iter().collect()
    })
}

fn get_token_regex() -> &'static Regex {
    TOKEN_REGEX.get_or_init(|| {
        Regex::new(r"[a-z][a-z0-9]*").unwrap()
    })
}

fn get_url_regex() -> &'static Regex {
    URL_REGEX.get_or_init(|| {
        // Capture domain name from URLs
        Regex::new(r"https?://(?:www\.)?([a-zA-Z0-9-]+)(?:\.[a-zA-Z]+)+[^\s]*").unwrap()
    })
}

fn get_phrase_delimiter_regex() -> &'static Regex {
    PHRASE_DELIMITER_REGEX.get_or_init(|| {
        // Split on punctuation, brackets, special chars
        Regex::new(r#"[.!?,;:\n\r()\[\]{}<>"'`/\\|=+*&^%$#@~]"#).unwrap()
    })
}

/// Pre-sanitize text before tokenization:
/// 1. Replace URLs with domain name only
/// 2. Remove common noise patterns
pub fn sanitize_text(text: &str) -> String {
    let url_regex = get_url_regex();
    
    // Replace URLs with just domain name
    let result = url_regex.replace_all(text, |caps: &regex::Captures| {
        caps.get(1).map_or("", |m| m.as_str()).to_string()
    });
    
    result.into_owned()
}

/// Check if a token looks like a hash/ID (mixed alphanumeric, length > 6)
fn is_hash_like(token: &str) -> bool {
    if token.len() <= 6 {
        return false;
    }
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    let has_letter = token.chars().any(|c| c.is_ascii_alphabetic());
    // If it has both letters and numbers mixed, it's likely a hash
    has_digit && has_letter
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

/// Simple RAKE-style phrase extraction
/// 1. Split text by punctuation and stopwords
/// 2. Extract candidate phrases (word sequences between delimiters)
/// 3. Return meaningful multi-word phrases as underscore-joined bigrams
fn extract_rake_phrases(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let delimiter_regex = get_phrase_delimiter_regex();
    let stopwords = get_stopwords();
    
    // Split by punctuation first
    let segments: Vec<&str> = delimiter_regex.split(&lower).collect();
    
    let mut phrases = Vec::new();
    
    for segment in segments {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        
        // Split segment and find runs of content words (non-stopwords)
        let words: Vec<&str> = segment.split_whitespace().collect();
        let mut current_phrase: Vec<String> = Vec::new();  // Use owned Strings
        
        for word in words {
            // Clean the word
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            
            if clean.is_empty() {
                continue;
            }
            
            if stopwords.contains(clean.as_str()) || clean.len() <= 1 {
                // Stopword encountered - emit current phrase if valid
                if current_phrase.len() >= 2 && current_phrase.len() <= 4 {
                    let phrase = current_phrase.join("_");
                    if !phrases.contains(&phrase) && phrase.len() >= 5 {
                        phrases.push(phrase);
                    }
                }
                current_phrase.clear();
            } else {
                // Stem the word before adding to phrase
                let stemmed = stem_word(&clean);
                current_phrase.push(stemmed);
            }
        }
        
        // Emit any remaining phrase
        if current_phrase.len() >= 2 && current_phrase.len() <= 4 {
            let phrase = current_phrase.join("_");
            if !phrases.contains(&phrase) && phrase.len() >= 5 {
                phrases.push(phrase);
            }
        }
    }
    
    // Limit to top 15 phrases
    phrases.truncate(15);
    phrases
}

pub fn tokenize_to_cues(text: &str) -> Vec<String> {
    // 1. Pre-sanitize (URLs, etc.)
    let sanitized = sanitize_text(text);
    
    // 2. Normalize
    let normalized = normalize_text(&sanitized);
    
    let mut cues = Vec::new();
    
    // 3. Extract individual tokens (filtered and stemmed)
    for token in get_token_regex().find_iter(&normalized) {
        let t = token.as_str();
        
        // Skip stopwords, single chars, and hash-like tokens
        if get_stopwords().contains(t) || t.len() <= 1 || is_hash_like(t) {
            continue;
        }
        
        // Stem the token (lemmatization)
        let stemmed = stem_word(t);
        
        // Only add if not empty and not already present
        if !stemmed.is_empty() && !cues.contains(&stemmed) {
            cues.push(stemmed);
        }
    }
    
    // 4. Extract quality bigrams using RAKE-style phrase detection (already stemmed internally)
    let rake_phrases = extract_rake_phrases(&sanitized);
    for phrase in rake_phrases {
        if !cues.contains(&phrase) {
            cues.push(phrase);
        }
    }
    
    cues
}
