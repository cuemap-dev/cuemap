use cuemap::engine::CueMapEngine;
use cuemap::persistence::PersistenceManager;
use cuemap::structures::MainStats;
use cuemap::config::ServerConfig;

#[test]
fn test_fixture_loading_and_recall() {
    // 1. Create a dummy snapshot file
    let fixture_dir = tempfile::tempdir().expect("Failed to create fixture directory");
    let fixture_path = fixture_dir.path().join("cuemap_fixture_test.bin");

    {
        // Scope to drop engine
        let engine = CueMapEngine::new();
        let memory_id = engine.upsert_memory_with_source_key(
            "fixture:test".to_string(),
            "Test Content".to_string(),
            vec!["test_cue".to_string()],
            None,
            Some(MainStats::default()),
            false,
            true,
        );
        assert_eq!(engine.memory_id_for_source_key("fixture:test"), Some(memory_id));
        PersistenceManager::save_to_path(&engine, &fixture_path).expect("Failed to save fixture");
    }

    // 2. Load it back
    let (memories, source_key_to_id, cue_index, next_memory_id, counts) =
        PersistenceManager::load_from_path::<MainStats>(&fixture_path)
        .expect("Failed to load fixture");
    let loaded_engine = CueMapEngine::<MainStats>::from_state(
        memories,
        source_key_to_id,
        cue_index,
        next_memory_id,
        counts,
        ServerConfig::default(),
        "fixture".to_string(),
    );

    // 3. Verify state
    assert_eq!(loaded_engine.get_memories().len(), 1);
    assert_eq!(loaded_engine.memory_id_for_source_key("fixture:test"), Some(1));
    assert_eq!(loaded_engine.next_memory_id(), 2);

    // 4. Run recall
    let results = loaded_engine.recall(vec!["test_cue".to_string()], 5, false, None);
    assert!(!results.is_empty());
    assert_eq!(results[0].content, "Test Content");
}
