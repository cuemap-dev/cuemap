use cuemap_rust::nl::*;

#[test]
fn test_tokenizer_basic() {
    let tokens = tokenize_to_cues("The quick brown fox");
    assert!(tokens.contains(&"quick".to_string()));
    assert!(tokens.contains(&"fox".to_string()));
    // Bigram "quick_brown" may or may not be generated depending on RAKE extraction
}

#[test]
fn test_tokenizer_edge_cases() {
    assert!(tokenize_to_cues("").is_empty());
    assert!(tokenize_to_cues("   ").is_empty());
    
    let special = tokenize_to_cues("!!! @@@ ###");
    // Should be empty or only contains non-alphanumeric tokens if they are allowed
    // Looking at common tokenizers, they usually filter punctuation.
    assert!(special.is_empty());
}

#[test]
fn test_normalize_text() {
    assert_eq!(normalize_text("  HELLO   WORLD  "), "hello world");
    assert_eq!(normalize_text("Mixed-Case_With_Dots.com"), "mixed case with dots com");
}

#[test]
fn test_url_sanitization() {
    let text = "Check https://github.com/user/repo for details";
    let sanitized = sanitize_text(text);
    assert!(sanitized.contains("github"));
    assert!(!sanitized.contains("https://"));
    assert!(!sanitized.contains("/user/repo"));
}

#[test]
fn test_rake_phrases() {
    let text = "Directly export a function expression instead of using a declaration";
    let cues = tokenize_to_cues(text);
    
    // Individual tokens should be present (lemmatized but not over-stemmed)
    assert!(cues.contains(&"directly".to_string()));
    assert!(cues.contains(&"export".to_string()));
    assert!(cues.contains(&"expression".to_string()));
    assert!(cues.contains(&"declaration".to_string()));
}

#[test]
fn test_code_tokens() {
    let text = "const result = await fetch(url)";
    let cues = tokenize_to_cues(text);
    
    // Code keywords are NOT globally filtered (they have meaning in natural language)
    assert!(cues.contains(&"result".to_string()));
    assert!(cues.contains(&"fetch".to_string()));
    assert!(cues.contains(&"url".to_string()));
}

#[test]
fn test_stemming() {
    // Test that different forms of words stem to the same root
    let text1 = "added comment";
    let text2 = "adding comments";
    let text3 = "this line adds a comment";
    
    let cues1 = tokenize_to_cues(text1);
    let cues2 = tokenize_to_cues(text2);
    let cues3 = tokenize_to_cues(text3);
    
    // All forms should produce "add" and "comment"
    assert!(cues1.contains(&"add".to_string()));
    assert!(cues1.contains(&"comment".to_string()));
    
    assert!(cues2.contains(&"add".to_string()));
    assert!(cues2.contains(&"comment".to_string()));
    
    assert!(cues3.contains(&"add".to_string()));
    assert!(cues3.contains(&"comment".to_string()));
    
    // Phrases should also be stemmed consistently
    assert!(cues1.contains(&"add_comment".to_string()));
    assert!(cues2.contains(&"add_comment".to_string()));
}
