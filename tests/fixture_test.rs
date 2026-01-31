use cuemap_rust::persistence::PersistenceManager;
use cuemap_rust::engine::CueMapEngine;
use std::path::PathBuf;
use std::fs;

#[test]
fn test_fixture_loading_and_recall() {
    // 1. Create a dummy snapshot file
    let fixture_path = PathBuf::from("tests/fixtures_test.bin");
    
    {
        // Scope to drop engine
        let engine = CueMapEngine::new();
        engine.add_memory("Test Content".to_string(), vec!["test_cue".to_string()], None, false);
        PersistenceManager::save_to_path(&engine, &fixture_path).expect("Failed to save fixture");
    }
    
    // 2. Load it back
    let (memories, cue_index) = PersistenceManager::load_from_path(&fixture_path).expect("Failed to load fixture");
    let loaded_engine = CueMapEngine::from_state(memories, cue_index);
    
    // 3. Verify state
    assert_eq!(loaded_engine.get_memories().len(), 1);
    
    // 4. Run recall
    let results = loaded_engine.recall(vec!["test_cue".to_string()], 5, false, None);
    assert!(!results.is_empty());
    assert_eq!(results[0].content, "Test Content");
    
    // Cleanup
    let _ = fs::remove_file(fixture_path);
}
