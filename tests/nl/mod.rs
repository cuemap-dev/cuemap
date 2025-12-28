use cuemap_rust::nl::*;

#[test]
fn test_tokenizer_basic() {
    let tokens = tokenize_to_cues("The quick brown fox");
    assert!(tokens.contains(&"tok:quick".to_string()));
    assert!(tokens.contains(&"tok:fox".to_string()));
    assert!(tokens.contains(&"phr:quick_brown".to_string()));
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
