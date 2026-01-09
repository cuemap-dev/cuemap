#[cfg(test)]
mod tests {
    use cuemap_rust::semantic::SemanticEngine;
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn test_glove_quality() {
        // Point to data dir
        let data_dir = Path::new("./data");
        let engine = SemanticEngine::new(Some(data_dir));
        
        let inputs = vec!["pistachio", "cheesecake", "dessert", "favorite"];
        
        println!("Loaded engine. Testing GloVe expansions (Threshold 0.60)...");
        
        for input in inputs {
            // Simulate ProposeCues logic
            // We need to access expand_glove logic directly or simulate it
            // expand_glove takes content, known_cues
            // But here we just want to test ONE word's expansion
            
            // Wait, expand_glove in semantic.rs takes (content, known_cues)
            // It iterates known_cues.
            
            let candidates = vec![input.to_string()];
            let expanded = engine.expand_glove("dummy content", &candidates);
            
            println!("Input: '{}' -> Expanded: {:?}", input, expanded);
            
            // Also print similarity of "taught" if input is "pistachio"
            // We can't access `search` directly unless public.
            // But we can infer.
        }
    }
}
