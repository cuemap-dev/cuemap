#[cfg(test)]
mod tests {
    use cuemap_rust::engine::CueMapEngine;
    use std::collections::HashMap;

    #[test]
    fn test_double_indexing_recall() {
        let engine = CueMapEngine::new();
        
        // 1. Add memory with structured cue
        engine.add_memory(
            "Function definition content".to_string(),
            vec!["type:function".to_string(), "name:ComputeTax".to_string()],
            None,
            false
        );
        
        // 2. Recall using full cue (legacy/precise)
        let results_full = engine.recall(vec!["name:ComputeTax".to_string()], 10, false);
        assert!(!results_full.is_empty(), "Should find memory by full cue");
        assert_eq!(results_full[0].content, "Function definition content");

        // 3. Recall using value only (natural language)
        let results_val = engine.recall(vec!["ComputeTax".to_string()], 10, false);
        assert!(!results_val.is_empty(), "Should find memory by value only");
        assert_eq!(results_val[0].content, "Function definition content");

        // 4. Recall using another value
        let results_val2 = engine.recall(vec!["function".to_string()], 10, false);
        assert!(!results_val2.is_empty(), "Should find memory by 'function'");
    }

    #[test]
    fn test_double_indexing_deletion() {
        let engine = CueMapEngine::new();
        let mem_id = engine.add_memory(
            "Specific content".to_string(),
            vec!["category:secret".to_string()],
            None,
            false
        );

        // Verify indexing
        assert!(!engine.recall(vec!["category:secret".to_string()], 1, false).is_empty());
        assert!(!engine.recall(vec!["secret".to_string()], 1, false).is_empty());

        // Delete
        engine.delete_memory(&mem_id);

        // Verify gone from both
        assert!(engine.recall(vec!["category:secret".to_string()], 1, false).is_empty(), "Should be gone from full index");
        assert!(engine.recall(vec!["secret".to_string()], 1, false).is_empty(), "Should be gone from value index");
    }
}
