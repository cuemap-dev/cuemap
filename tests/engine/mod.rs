use cuemap_rust::engine::CueMapEngine;

#[test]
fn test_memory_cues_storage() {
    let engine = CueMapEngine::new();
    let cues = vec!["a".to_string(), "b".to_string()];
    let memory_id = engine.add_memory("test content".to_string(), cues.clone(), None);

    let memory = engine.get_memory(&memory_id).unwrap();
    assert_eq!(memory.cues, cues);
}

#[test]
fn test_attach_cues() {
    let engine = CueMapEngine::new();
    let initial_cues = vec!["a".to_string()];
    let memory_id = engine.add_memory("test content".to_string(), initial_cues.clone(), None);

    // Attach new cues
    let new_cues = vec!["b".to_string(), "c".to_string()];
    let attached = engine.attach_cues(&memory_id, new_cues.clone());
    assert!(attached);

    // Verify memory has all cues
    let memory = engine.get_memory(&memory_id).unwrap();
    let expected_cues = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    assert_eq!(memory.cues, expected_cues);

    // Verify recall works with new cues
    let results = engine.recall(vec!["b".to_string()], 10, false);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_id, memory_id);

    // Verify attaching existing cues returns false (no change)
    let attached_again = engine.attach_cues(&memory_id, vec!["a".to_string(), "b".to_string()]);
    assert!(!attached_again);
}

#[test]
fn test_freshness_boost() {
    let engine = CueMapEngine::new();
    
    // Add two memories with the same cue
    let _id1 = engine.add_memory("oldest".to_string(), vec!["topic".to_string()], None);
    let id2 = engine.add_memory("newest".to_string(), vec!["topic".to_string()], None);
    
    let results = engine.recall(vec!["topic".to_string()], 10, true);
    
    assert_eq!(results.len(), 2);
    // Position 0 gets +1.0 freshness boost.
    // Recency score for pos 0 = 1/(0+1) + 1.0 = 2.0
    // Recency score for pos 1 = 1/(1+1) = 0.5
    
    assert_eq!(results[0].memory_id, id2); // id2 is more recent
    assert!(results[0].recency_score > 1.5);
    assert!(results[1].recency_score < 1.0);
}

#[test]
fn test_scoring_gradient() {
    let engine = CueMapEngine::new();
    let cue = "grad".to_string();
    
    // Add many memories to create a deep list
    let mut ids = Vec::new();
    for i in 0..10 {
        ids.push(engine.add_memory(format!("content {}", i), vec![cue.clone()], None));
    }
    
    let results = engine.recall(vec![cue], 10, false);
    
    // Scores should be strictly decreasing
    for i in 0..results.len()-1 {
        assert!(results[i].score > results[i+1].score, "Score for {} should be > than {}", i, i+1);
    }
}

#[test]
fn test_log_frequency_scaling() {
    let engine = CueMapEngine::new();
    let id1 = engine.add_memory("frequent".to_string(), vec!["cue".to_string()], None);
    let id2 = engine.add_memory("rare".to_string(), vec!["cue".to_string()], None);
    
    // id1 gets 100 reinforcements
    for _ in 0..100 {
        engine.upsert_memory_with_id(id1.clone(), "frequent".to_string(), vec!["cue".to_string()], None, true);
    }
    
    // id2 gets 10 reinforcements
    for _ in 0..10 {
        engine.upsert_memory_with_id(id2.clone(), "rare".to_string(), vec!["cue".to_string()], None, true);
    }
    
    let results = engine.recall(vec!["cue".to_string()], 10, false);
    
    // log10(100) = 2.0
    // log10(10) = 1.0
    let res1 = results.iter().find(|r| r.memory_id == id1).unwrap();
    let res2 = results.iter().find(|r| r.memory_id == id2).unwrap();
    
    assert_eq!(res1.reinforcement_score, 2.0);
    assert_eq!(res2.reinforcement_score, 1.0);
}
