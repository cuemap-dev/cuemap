use cuemap_rust::engine::CueMapEngine;
use std::collections::HashMap;

#[test]
fn test_pattern_completion() {
    let engine = CueMapEngine::new();
    
    // Memory 1: {A, B}
    engine.add_memory("content 1".to_string(), vec!["cue:a".to_string(), "cue:b".to_string()], None, false);
    // Memory 2: {A, C}
    engine.add_memory("content 2".to_string(), vec!["cue:a".to_string(), "cue:c".to_string()], None, false);
    
    // Recall with {B}
    let results = engine.recall(vec!["cue:b".to_string()], 10, false);
    
    // Check if "cue:a" was inferred. 
    // Since "cue:b" co-occurs with "cue:a" in Memory 1, "cue:a" should be injected.
    // If "cue:a" is injected, Memory 2 should rank because it has "cue:a".
    assert!(results.iter().any(|r| r.content == "content 2"), "Memory 2 should be recalled via inferred cue:a");
}

#[test]
fn test_temporal_chunking() {
    let engine = CueMapEngine::new();
    let mut metadata = HashMap::new();
    metadata.insert("project_id".to_string(), serde_json::json!("p1"));
    
    let id1 = engine.add_memory("event 1".to_string(), vec!["topic:coding".to_string()], Some(metadata.clone()), false);
    let id2 = engine.add_memory("event 2".to_string(), vec!["topic:coding".to_string()], Some(metadata), false);
    
    let mem2 = engine.get_memory(&id2).unwrap();
    let episode_cue = format!("episode:{}", id1);
    assert!(mem2.cues.contains(&episode_cue), "Second memory should have episode cue pointing to the first");
}

#[test]
fn test_salience_bias() {
    let engine = CueMapEngine::new();
    
    // High cue density memory
    let id_salient = engine.add_memory("short".to_string(), vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string(), "e".to_string()], None, false);
    // Normal memory
    let id_normal = engine.add_memory("this is a much longer content for a normal memory".to_string(), vec!["a".to_string()], None, false);
    
    let results = engine.recall(vec!["a".to_string()], 10, false);
    
    assert_eq!(results[0].memory_id, id_salient, "Salient memory should rank first even if newer memory exists if it has much higher salience");
}

#[test]
fn test_match_integrity_scores() {
    let engine = CueMapEngine::new();
    
    engine.add_memory("exact match".to_string(), vec!["a".to_string(), "b".to_string()], None, false);
    engine.add_memory("partial match".to_string(), vec!["a".to_string(), "c".to_string(), "d".to_string(), "e".to_string()], None, false);
    
    let results = engine.recall(vec!["a".to_string(), "b".to_string()], 10, false);
    
    assert!(results[0].match_integrity > results[1].match_integrity, "Exact match should have higher match integrity than partial match");
}

#[test]
fn test_systems_consolidation() {
    let engine = CueMapEngine::new();
    
    engine.add_memory("report part 1".to_string(), vec!["type:report".to_string(), "month:jan".to_string()], None, false);
    engine.add_memory("report part 2".to_string(), vec!["type:report".to_string(), "month:jan".to_string()], None, false);
    
    let initial_count = engine.get_stats().get("total_memories").unwrap().as_u64().unwrap();
    assert_eq!(initial_count, 2);
    
    // Lower threshold because temporal chunking adds an episode cue
    let consolidated = engine.consolidate_memories(0.6);
    assert_eq!(consolidated.len(), 1);
    
    let final_count = engine.get_stats().get("total_memories").unwrap().as_u64().unwrap();
    assert_eq!(final_count, 3); // 2 original + 1 summary (additive)
    
    let mem = engine.get_memory(&consolidated[0].0).unwrap();
    assert!(mem.metadata.contains_key("consolidated"));
    assert!(mem.content.contains("report part 1"));
    assert!(mem.content.contains("report part 2"));
}
