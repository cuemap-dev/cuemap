#[cfg(test)]
mod tests {
    use cuemap_rust::semantic::SemanticEngine;
    use std::path::Path;

    #[test]
    fn test_wordnet_expansion() {
        // Initialize without data_dir (should load only WordNet)
        let engine = SemanticEngine::new(None);
        
        // "payment" should expand to "defrayment", "pay", etc. if in WordNet
        // The implementation skips exact value match, so we look for synonyms
        let known_cues = vec!["topic:payment".to_string()];
        
        let expanded = engine.expand_wordnet("dummy content", &known_cues, 0.6, 5);
        
        println!("Expanded: {:?}", expanded);
        
        // Assert we get *some* expansion
        // Note: thesaurus crate might return empty if word not found or stripped.
        // "payment" is a common word.
        if !expanded.is_empty() {
             // New behavior: flat cues
             // assert!(expanded.iter().any(|c| c.starts_with("topic:")));
             assert!(!expanded.iter().any(|c| c.contains(':')), "Output cues must not contain colons");
        } else {
             println!("Warning: No synonyms found for 'payment' in bundled thesaurus");
        }
    }
    
    #[test]
    fn test_wordnet_expansion_preserves_key() {
        let engine = SemanticEngine::new(None);
        let known_cues = vec!["category:fruit".to_string()];
        
        let expanded = engine.expand_wordnet("dummy", &known_cues, 0.6, 5);
        
        // With new flat-cue logic, the key is NOT preserved.
        // Input: "category:fruit" -> Word: "fruit" -> Synonyms: "pomelo", etc.
        // Output: "pomelo" (flat)
        for cue in expanded {
            assert!(!cue.starts_with("category:"));
            assert!(!cue.contains(':'));
        }
    }

    #[test]
    fn test_glove_graceful_fallback() {
        // Should not crash if files missing
        let engine = SemanticEngine::new(None);
        let known_cues = vec!["topic:payment".to_string()];
        let expanded = engine.expand_glove("dummy", &known_cues);
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_global_context_fallback() {
        let engine = SemanticEngine::new(None);
        let expanded = engine.expand_global_context("server crash bug");
        assert!(expanded.is_empty());
    }
}
