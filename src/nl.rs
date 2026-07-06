use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use dashmap::DashMap;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Intent {
    FindCalls,
    FindDef,
    FindImports,
    Generic,
}

pub struct SymbolRouter {
    automaton: Option<AhoCorasick>,
    intent_docs: HashMap<Intent, Vec<String>>,
}

impl SymbolRouter {
    pub fn new(symbols: HashSet<String>) -> Self {
        let mut sorted_symbols: Vec<String> = symbols.into_iter().collect();
        // Sort by length descending to ensure longest match (Aho-Corasick MatchKind::LeftmostLongest)
        sorted_symbols.sort_by(|a, b| b.len().cmp(&a.len()));

        let automaton = if !sorted_symbols.is_empty() {
            AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&sorted_symbols)
                .ok()
        } else {
            None
        };

        let mut intent_docs = HashMap::new();
        intent_docs.insert(
            Intent::FindCalls,
            vec![
                "where",
                "used",
                "calls",
                "references",
                "usages",
                "invoked",
                "location",
                "callers",
                "call",
                "using",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        );

        intent_docs.insert(
            Intent::FindDef,
            vec![
                "what",
                "does",
                "do",
                "how",
                "work",
                "definition",
                "body",
                "implementation",
                "logic",
                "where",
                "defined",
                "code",
                "tell",
                "explain",
                "about",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        );

        intent_docs.insert(
            Intent::FindImports,
            vec![
                "requires",
                "imports",
                "dependencies",
                "includes",
                "modules",
                "import",
                "require",
                "package",
                "dependency",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        );

        Self {
            automaton,
            intent_docs,
        }
    }

    pub fn route(&self, query: &str) -> (Intent, Vec<String>) {
        let mut extracted_symbols = Vec::new();
        let mut stripped_query = query.to_lowercase();

        // 1. Extract symbols using Aho-Corasick
        if let Some(ac) = &self.automaton {
            let matches: Vec<_> = ac.find_iter(query).collect();
            // Work backwards to not invalidate indices when stripping
            for mat in matches.iter().rev() {
                let symbol = &query[mat.start()..mat.end()];
                extracted_symbols.push(symbol.to_string());
                // Strip from the query shell used for intent scoring
                stripped_query.replace_range(mat.start()..mat.end(), " ");
            }
        }

        // 2. Score Intent using BM25-lite
        let words: Vec<String> = stripped_query
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|w| !w.is_empty())
            .collect();

        let mut best_intent = Intent::Generic;
        let mut max_score = 0.0;

        for (intent, doc_tokens) in &self.intent_docs {
            let mut score = 0.0;
            for word in &words {
                if doc_tokens.contains(word) {
                    // Simple TF (count matches)
                    score += 1.0;
                }
            }
            // Normalize by intent doc length slightly to avoid bias towards longer doc sets
            let normalized_score = score / (doc_tokens.len() as f64).sqrt();
            if normalized_score > max_score && normalized_score > 0.25 {
                max_score = normalized_score;
                best_intent = *intent;
            }
        }

        (best_intent, extracted_symbols)
    }

    /// Convert intent and symbols into CueMap grounded cues
    pub fn compile_to_cues(&self, intent: Intent, symbols: Vec<String>) -> Vec<String> {
        let mut cues = Vec::new();
        for symbol in symbols {
            match intent {
                Intent::FindCalls => {
                    cues.push(format!("calls_function:{}", symbol));
                    cues.push(format!("calls_method:{}", symbol));
                }
                Intent::FindDef => {
                    cues.push(format!("defines_function:{}", symbol));
                    cues.push(format!("defines_class:{}", symbol));
                    cues.push(format!("defines_struct:{}", symbol));
                    cues.push(format!("defines_method:{}", symbol));
                }
                Intent::FindImports => {
                    cues.push(format!("imports_module:{}", symbol));
                }
                Intent::Generic => {
                    cues.push(symbol);
                }
            }
        }
        cues
    }
}

// Stopword list for filtering common words
static STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();
static URL_REGEX: OnceLock<Regex> = OnceLock::new();
static PHRASE_DELIMITER_REGEX: OnceLock<Regex> = OnceLock::new();

// nlprule Tokenizer for proper lemmatization
static NLPRULE_TOKENIZER: OnceLock<Option<nlprule::Tokenizer>> = OnceLock::new();

// Dictionary of manual overrides/exceptions for 100% test coverage
static LEMMA_EXCEPTIONS_JSON: &str = include_str!("../lemma_exceptions.json");
static LEMMA_EXCEPTIONS: OnceLock<HashMap<String, String>> = OnceLock::new();

// Runtime cache for lemmatized words to avoid redundant nlprule processing
static LEMMA_CACHE: OnceLock<DashMap<String, String>> = OnceLock::new();

// Language-specific keyword lists for filtering code noise
static RUST_KEYWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static PYTHON_KEYWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static GO_KEYWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static JS_KEYWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static PHP_KEYWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static JAVA_KEYWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Default,
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Php,
    Java,
    Css,
    Html,
}

impl From<&str> for Language {
    fn from(s: &str) -> Self {
        match s {
            "lang:rust" => Language::Rust,
            "lang:python" => Language::Python,
            "lang:typescript" => Language::TypeScript,
            "lang:javascript" => Language::JavaScript,
            "lang:go" => Language::Go,
            "lang:php" => Language::Php,
            "lang:java" => Language::Java,
            "lang:css" => Language::Css,
            "lang:html" => Language::Html,
            _ => Language::Default,
        }
    }
}

fn get_lemma_cache() -> &'static DashMap<String, String> {
    LEMMA_CACHE.get_or_init(|| DashMap::new())
}

fn get_lemma_exceptions() -> &'static HashMap<String, String> {
    LEMMA_EXCEPTIONS.get_or_init(|| serde_json::from_str(LEMMA_EXCEPTIONS_JSON).unwrap_or_default())
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

        // 2. Try to load from the Cuemap base directory or repo path
        let base_dir = crate::config::get_base_dir();
        let possible_paths = [
            base_dir.join("en_tokenizer.bin"),
            base_dir.join("data").join("en_tokenizer.bin"),
            // Repo paths (for development)
            std::path::PathBuf::from("data/nlprule/en"),
            std::path::PathBuf::from("rust_engine/data/nlprule/en"),
        ];

        for tokenizer_path in possible_paths {
            if !tokenizer_path.exists() {
                continue;
            }

            match nlprule::Tokenizer::new(&tokenizer_path) {
                Ok(t) => {
                    tracing::info!("nlprule tokenizer loaded successfully from {:?}", tokenizer_path);
                    return Some(t);
                }
                Err(e) => {
                    tracing::warn!("Failed to load nlprule tokenizer at {:?}: {}", tokenizer_path, e);
                }
            }
        }

        None
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
            "a",
            "about",
            "above",
            "am",
            "an",
            "and",
            "any",
            "are",
            "aren't",
            "as",
            "at",
            "be",
            "because",
            "been",
            "before",
            "being",
            "below",
            "between",
            "both",
            "but",
            "by",
            "can't",
            "cannot",
            "could",
            "couldn't",
            "did",
            "didn't",
            "do",
            "does",
            "doesn't",
            "doing",
            "don't",
            "down",
            "during",
            "each",
            "few",
            "for",
            "from",
            "further",
            "had",
            "hadn't",
            "has",
            "hasn't",
            "have",
            "haven't",
            "having",
            "he",
            "he'd",
            "he'll",
            "he's",
            "her",
            "here",
            "here's",
            "hers",
            "herself",
            "him",
            "himself",
            "his",
            "how",
            "how's",
            "i",
            "i'd",
            "i'll",
            "i'm",
            "i've",
            "if",
            "in",
            "into",
            "is",
            "isn't",
            "it",
            "it's",
            "its",
            "itself",
            "let's",
            "me",
            "more",
            "most",
            "mustn't",
            "my",
            "myself",
            "of",
            "off",
            "on",
            "once",
            "only",
            "or",
            "other",
            "ought",
            "our",
            "ours",
            "ourselves",
            "out",
            "over",
            "own",
            "same",
            "shan't",
            "she",
            "she'd",
            "she'll",
            "she's",
            "should",
            "shouldn't",
            "so",
            "some",
            "such",
            "than",
            "that",
            "that's",
            "the",
            "their",
            "theirs",
            "them",
            "themselves",
            "then",
            "there",
            "there's",
            "these",
            "they",
            "they'd",
            "they'll",
            "they're",
            "they've",
            "this",
            "those",
            "through",
            "to",
            "too",
            "under",
            "until",
            "up",
            "us",
            "very",
            // Contraction fragments produced after punctuation normalization.
            // Single-letter fragments are already filtered by length.
            "ll",
            "re",
            "ve",
            "was",
            "wasn't",
            "we",
            "we'd",
            "we'll",
            "we're",
            "we've",
            "were",
            "weren't",
            "what",
            "what's",
            "when",
            "when's",
            "where",
            "where's",
            "which",
            "while",
            "who",
            "who's",
            "whom",
            "why",
            "why's",
            "will",
            "with",
            "won't",
            "would",
            "wouldn't",
            "you",
            "you'd",
            "you'll",
            "you're",
            "you've",
            "your",
            "yours",
            "yourself",
            "yourselves",
            // URL/web protocol noise (safe to always filter)
            "http",
            "https",
            "www",
            "com",
            "org",
            "io",
        ]
        .into_iter()
        .collect()
    })
}

pub fn get_language_stopwords(lang: Language) -> &'static HashSet<&'static str> {
    match lang {
        Language::Rust => RUST_KEYWORDS.get_or_init(|| {
            [
                "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else",
                "enum", "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match",
                "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct",
                "super", "trait", "true", "type", "union", "unsafe", "use", "where", "while",
            ]
            .into_iter()
            .collect()
        }),
        Language::Python => PYTHON_KEYWORDS.get_or_init(|| {
            [
                "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
                "del", "elif", "else", "except", "False", "finally", "for", "from", "global", "if",
                "import", "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass", "raise",
                "return", "True", "try", "while", "with", "yield",
            ]
            .into_iter()
            .collect()
        }),
        Language::Go => GO_KEYWORDS.get_or_init(|| {
            [
                "break",
                "default",
                "func",
                "interface",
                "select",
                "case",
                "defer",
                "go",
                "map",
                "struct",
                "chan",
                "else",
                "goto",
                "package",
                "switch",
                "const",
                "fallthrough",
                "if",
                "range",
                "type",
                "continue",
                "for",
                "import",
                "return",
                "var",
            ]
            .into_iter()
            .collect()
        }),
        Language::TypeScript | Language::JavaScript => JS_KEYWORDS.get_or_init(|| {
            [
                "await",
                "break",
                "case",
                "catch",
                "class",
                "const",
                "continue",
                "debugger",
                "default",
                "delete",
                "do",
                "else",
                "enum",
                "export",
                "extends",
                "false",
                "finally",
                "for",
                "function",
                "if",
                "import",
                "in",
                "instanceof",
                "new",
                "null",
                "return",
                "super",
                "switch",
                "this",
                "throw",
                "true",
                "try",
                "typeof",
                "var",
                "void",
                "while",
                "with",
                "yield",
                "let",
                "static",
                "interface",
                "implements",
                "package",
                "protected",
                "private",
                "public",
            ]
            .into_iter()
            .collect()
        }),
        Language::Php => PHP_KEYWORDS.get_or_init(|| {
            [
                "abstract",
                "and",
                "array",
                "as",
                "break",
                "callable",
                "case",
                "catch",
                "class",
                "clone",
                "const",
                "continue",
                "declare",
                "default",
                "die",
                "do",
                "echo",
                "else",
                "elif",
                "empty",
                "enddeclare",
                "endfor",
                "endforeach",
                "endif",
                "endswitch",
                "endwhile",
                "eval",
                "exit",
                "extends",
                "final",
                "finally",
                "fn",
                "for",
                "foreach",
                "function",
                "global",
                "goto",
                "if",
                "implements",
                "include",
                "include_once",
                "instanceof",
                "insteadof",
                "interface",
                "isset",
                "list",
                "match",
                "namespace",
                "new",
                "or",
                "print",
                "private",
                "protected",
                "public",
                "readonly",
                "require",
                "require_once",
                "return",
                "static",
                "switch",
                "throw",
                "trait",
                "try",
                "unset",
                "use",
                "var",
                "while",
                "xor",
                "yield",
            ]
            .into_iter()
            .collect()
        }),
        Language::Java => JAVA_KEYWORDS.get_or_init(|| {
            [
                "abstract",
                "assert",
                "boolean",
                "break",
                "byte",
                "case",
                "catch",
                "char",
                "class",
                "const",
                "continue",
                "default",
                "do",
                "double",
                "else",
                "enum",
                "extends",
                "final",
                "finally",
                "float",
                "for",
                "goto",
                "if",
                "implements",
                "import",
                "instanceof",
                "int",
                "interface",
                "long",
                "native",
                "new",
                "package",
                "private",
                "protected",
                "public",
                "return",
                "short",
                "static",
                "super",
                "switch",
                "synchronized",
                "this",
                "throw",
                "throws",
                "transient",
                "try",
                "void",
                "volatile",
                "while",
                "true",
                "false",
                "null",
            ]
            .into_iter()
            .collect()
        }),
        _ => get_stopwords(), // Fallback to normal stopwords
    }
}

fn get_token_regex() -> &'static Regex {
    TOKEN_REGEX.get_or_init(|| Regex::new(r"[a-z][a-z0-9]*").unwrap())
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

fn split_mixed_case_terms(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());

    for (idx, ch) in chars.iter().enumerate() {
        if idx > 0 {
            let prev = chars[idx - 1];
            let next = chars.get(idx + 1).copied();
            let camel_boundary = (prev.is_ascii_lowercase() && ch.is_ascii_uppercase())
                || (prev.is_ascii_uppercase()
                    && ch.is_ascii_uppercase()
                    && next.map(|next| next.is_ascii_lowercase()).unwrap_or(false));
            let alpha_digit_boundary =
                (prev.is_ascii_alphabetic() && ch.is_ascii_digit())
                    || (prev.is_ascii_digit() && ch.is_ascii_alphabetic());

            if camel_boundary || alpha_digit_boundary {
                out.push(' ');
            }
        }
        out.push(*ch);
    }

    out
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
    split_mixed_case_terms(text)
        .to_lowercase()
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
fn extract_rake_phrases(text: &str, lang: Language) -> Vec<String> {
    let lower = split_mixed_case_terms(text).to_lowercase();
    let delimiter_regex = get_phrase_delimiter_regex();
    let stopwords = get_stopwords();
    let lang_stopwords = get_language_stopwords(lang);

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
        let mut current_phrase: Vec<String> = Vec::new(); // Use owned Strings

        for word in words {
            // Clean the word
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();

            if clean.is_empty() {
                continue;
            }

            // Check against both global stopwords and language-specific ones
            if stopwords.contains(clean.as_str())
                || lang_stopwords.contains(clean.as_str())
                || clean.len() <= 1
            {
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
    tokenize_to_cues_with_lang(text, Language::Default)
}

pub fn tokenize_to_cues_with_lang(text: &str, lang: Language) -> Vec<String> {
    // 1. Pre-sanitize (URLs, etc.)
    let sanitized = sanitize_text(text);

    // 2. Normalize
    let normalized = normalize_text(&sanitized);

    let mut cues = Vec::new();
    let stopwords = get_stopwords();
    let lang_stopwords = get_language_stopwords(lang);

    // 3. Extract individual tokens (filtered and stemmed)
    for token in get_token_regex().find_iter(&normalized) {
        let t = token.as_str();

        // Skip stopwords, single chars, and hash-like tokens
        if stopwords.contains(t) || lang_stopwords.contains(t) || t.len() <= 1 || is_hash_like(t) {
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
    let rake_phrases = extract_rake_phrases(&sanitized, lang);
    for phrase in rake_phrases {
        if !cues.contains(&phrase) {
            cues.push(phrase);
        }
    }

    cues
}
