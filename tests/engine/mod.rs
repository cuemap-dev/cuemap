use cuemap::engine::CueMapEngine;
use cuemap::structures::{MainStats, INVALID_MEMORY_ID};
use serde_json::json;
use std::collections::HashMap;

#[test]
fn test_memory_cues_storage() {
    let engine = CueMapEngine::new();
    let cues = vec!["a".to_string(), "b".to_string()];
    let memory_id = engine.add_memory(
        "test content".to_string(),
        cues.clone(),
        None,
        MainStats::default(),
        false,
    );

    let memory = engine.get_memory(memory_id).unwrap();
    assert_eq!(memory.cues, cues);
}

#[test]
fn test_numeric_memory_ids_are_monotonic() {
    let engine = CueMapEngine::new();
    let first = engine.add_memory(
        "first".to_string(),
        vec!["alpha".to_string()],
        None,
        MainStats::default(),
        false,
    );
    let second = engine.add_memory(
        "second".to_string(),
        vec!["beta".to_string()],
        None,
        MainStats::default(),
        false,
    );

    assert_ne!(first, INVALID_MEMORY_ID);
    assert_eq!(second, first + 1);
    assert_eq!(engine.get_memory(first).unwrap().id, first);
    assert_eq!(engine.get_memory(second).unwrap().id, second);
}

#[test]
fn test_source_key_upsert_reuses_numeric_id() {
    let engine = CueMapEngine::new();
    let first = engine.upsert_memory_with_source_key(
        "file:/notes.md:1-3".to_string(),
        "initial content".to_string(),
        vec!["alpha".to_string()],
        None,
        Some(MainStats::default()),
        false,
        true,
    );
    let second = engine.upsert_memory_with_source_key(
        "file:/notes.md:1-3".to_string(),
        "updated content".to_string(),
        vec!["beta".to_string()],
        None,
        Some(MainStats::default()),
        false,
        true,
    );

    assert_eq!(first, second);
    assert_eq!(engine.total_memories(), 1);
    assert_eq!(engine.memory_id_for_source_key("file:/notes.md:1-3"), Some(first));
    assert!(engine.recall(vec!["alpha".to_string()], 10, false, None).is_empty());
    let results = engine.recall(vec!["beta".to_string()], 10, false, None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_id, first);
}

#[test]
fn test_source_order_index_tracks_add_upsert_and_delete() {
    let engine = CueMapEngine::new();
    let mut metadata = HashMap::new();
    metadata.insert("source_session_id".to_string(), json!("thread-1"));
    metadata.insert("source_turn_index".to_string(), json!(2));

    let first = engine.upsert_memory_with_source_key(
        "thread-1:2".to_string(),
        "translation service latency improved".to_string(),
        vec!["translation".to_string(), "service".to_string()],
        Some(metadata),
        Some(MainStats::default()),
        false,
        true,
    );

    let entries = engine.ordered_entries_for_session("thread-1", 10);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].memory_id, first);
    assert_eq!(entries[0].order, 2);

    let mut updated_metadata = HashMap::new();
    updated_metadata.insert("source_session_id".to_string(), json!("thread-1"));
    updated_metadata.insert("source_turn_index".to_string(), json!(5));
    let second = engine.upsert_memory_with_source_key(
        "thread-1:2".to_string(),
        "translation service latency improved again".to_string(),
        vec!["translation".to_string(), "service".to_string()],
        Some(updated_metadata),
        Some(MainStats::default()),
        false,
        true,
    );

    assert_eq!(first, second);
    let entries = engine.ordered_entries_for_session("thread-1", 10);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].memory_id, first);
    assert_eq!(entries[0].order, 5);

    assert!(engine.delete_memory(first));
    assert!(engine.ordered_entries_for_session("thread-1", 10).is_empty());
}

#[test]
fn test_source_order_index_rebuilds_deterministically() {
    let engine = CueMapEngine::new();
    for order in [3, 1, 2] {
        let mut metadata = HashMap::new();
        metadata.insert("source_session_id".to_string(), json!("thread-2"));
        metadata.insert("source_turn_index".to_string(), json!(order));
        engine.add_memory(
            format!("turn {}", order),
            vec!["timeline".to_string()],
            Some(metadata),
            MainStats::default(),
            false,
        );
    }

    engine.rebuild_source_order_index();
    let orders: Vec<i64> = engine
        .ordered_entries_for_session("thread-2", 10)
        .into_iter()
        .map(|entry| entry.order)
        .collect();
    assert_eq!(orders, vec![1, 2, 3]);
}

#[test]
fn test_expansion_depth_uses_source_order_neighbors() {
    let engine = CueMapEngine::new();
    for (order, content, cue) in [
        (1, "previous setup turn", "setup"),
        (2, "target middle turn", "middle"),
        (3, "following resolution turn", "resolution"),
    ] {
        let mut metadata = HashMap::new();
        metadata.insert("source_session_id".to_string(), json!("thread-neighbors"));
        metadata.insert("source_turn_index".to_string(), json!(order));
        engine.add_memory(
            content.to_string(),
            vec![cue.to_string()],
            Some(metadata),
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        vec![("middle".to_string(), 1.0)],
        1,
        false,
        None,
        2,
        false,
        false,
        None,
        None,
    );

    assert_eq!(results.len(), 1);
    let content = &results[0].content;
    let previous = content.find("previous setup turn").unwrap();
    let target = content.find("target middle turn").unwrap();
    let following = content.find("following resolution turn").unwrap();
    assert!(previous < target);
    assert!(target < following);
}

#[test]
fn test_delete_removes_source_key_and_numeric_postings() {
    let engine = CueMapEngine::new();
    let memory_id = engine.upsert_memory_with_source_key(
        "source:delete-me".to_string(),
        "delete me".to_string(),
        vec!["delete".to_string()],
        None,
        Some(MainStats::default()),
        false,
        true,
    );

    assert!(engine.delete_memory(memory_id));
    assert!(engine.get_memory(memory_id).is_none());
    assert_eq!(engine.memory_id_for_source_key("source:delete-me"), None);
    assert!(engine.recall(vec!["delete".to_string()], 10, false, None).is_empty());
}

#[test]
fn test_attach_cues() {
    let engine = CueMapEngine::new();
    let initial_cues = vec!["a".to_string()];
    let memory_id = engine.add_memory(
        "test content".to_string(),
        initial_cues.clone(),
        None,
        MainStats::default(),
        false,
    );

    // Attach new cues
    let new_cues = vec!["b".to_string(), "c".to_string()];
    let attached = engine.attach_cues(memory_id, new_cues.clone());
    assert!(attached);

    // Verify memory has all cues
    let memory = engine.get_memory(memory_id).unwrap();
    let expected_cues = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    assert_eq!(memory.cues, expected_cues);

    // Verify recall works with new cues
    let results = engine.recall(vec!["b".to_string()], 10, false, None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_id, memory_id);

    // Verify attaching existing cues returns false (no change)
    let attached_again = engine.attach_cues(memory_id, vec!["a".to_string(), "b".to_string()]);
    assert!(!attached_again);
}

#[test]
fn test_freshness_boost() {
    let engine = CueMapEngine::new();

    // Add two memories with the same cue
    let _id1 = engine.add_memory(
        "oldest".to_string(),
        vec!["topic".to_string()],
        None,
        MainStats::default(),
        false,
    );
    let id2 = engine.add_memory(
        "newest".to_string(),
        vec!["topic".to_string()],
        None,
        MainStats::default(),
        false,
    );

    let results = engine.recall(vec!["topic".to_string()], 10, true, None);

    assert_eq!(results.len(), 2);
    // Position 0 gets +1.0 freshness boost.
    // Recency score for pos 0 = 1/(0+1) + 1.0 = 2.0
    // Recency score for pos 1 = 1/(1+1) = 0.5

    assert_eq!(results[0].memory_id, id2); // id2 is more recent
                                           //assert!(results[0].recency_score > 1.5);
    assert!(results[1].recency_score < 1.0);
}

#[test]
fn test_scoring_gradient() {
    let engine = CueMapEngine::new();
    let cue = "grad".to_string();

    // Add many memories to create a deep list
    let mut ids = Vec::new();
    for i in 0..10 {
        ids.push(engine.add_memory(
            format!("content {}", i),
            vec![cue.clone()],
            None,
            MainStats::default(),
            false,
        ));
    }

    let results = engine.recall(vec![cue], 10, false, None);

    // Scores should be strictly decreasing
    for i in 0..results.len() - 1 {
        assert!(
            results[i].score > results[i + 1].score,
            "Score for {} should be > than {}",
            i,
            i + 1
        );
    }
}

#[test]
fn test_log_frequency_scaling() {
    let engine = CueMapEngine::new();
    let id1 = engine.add_memory(
        "frequent".to_string(),
        vec!["cue".to_string()],
        None,
        MainStats::default(),
        false,
    );
    let id2 = engine.add_memory(
        "rare".to_string(),
        vec!["cue".to_string()],
        None,
        MainStats::default(),
        false,
    );

    // id1 gets 100 reinforcements
    for _ in 0..100 {
        engine.reinforce_memory(id1, vec!["cue".to_string()]);
    }

    // id2 gets 10 reinforcements
    for _ in 0..10 {
        engine.reinforce_memory(id2, vec!["cue".to_string()]);
    }

    let results = engine.recall(vec!["cue".to_string()], 10, false, None);

    // log10(100) = 2.0
    // log10(10) = 1.0
    let res1 = results.iter().find(|r| r.memory_id == id1).unwrap();
    let res2 = results.iter().find(|r| r.memory_id == id2).unwrap();

    assert_eq!(res1.reinforcement_score, 2.0);
    assert_eq!(res2.reinforcement_score, 1.0);
}

#[test]
fn test_trending_cues_are_derived_from_reinforced_main_memories() {
    let engine = CueMapEngine::<MainStats>::new();
    let first = engine.add_memory(
        "favorite dessert".to_string(),
        vec![
            "dessert".to_string(),
            "sweet".to_string(),
            "path:/tmp/ignored".to_string(),
        ],
        None,
        MainStats::default(),
        false,
    );
    let second = engine.add_memory(
        "dessert recipe".to_string(),
        vec!["dessert".to_string(), "recipe".to_string()],
        None,
        MainStats::default(),
        false,
    );
    let _unreinforced = engine.add_memory(
        "unrelated".to_string(),
        vec!["unrelated".to_string()],
        None,
        MainStats::default(),
        false,
    );

    engine.reinforce_dynamic(first, 1.0);
    engine.reinforce_dynamic(second, 1.0);

    let trending = engine.get_trending_cues(10);
    let dessert = trending.iter().find(|(cue, _)| cue == "dessert").unwrap();
    let recipe = trending.iter().find(|(cue, _)| cue == "recipe").unwrap();

    assert!(dessert.1 > recipe.1);
    assert!(trending.iter().any(|(cue, _)| cue == "sweet"));
    assert!(!trending.iter().any(|(cue, _)| cue == "unrelated"));
    assert!(!trending.iter().any(|(cue, _)| cue.starts_with("path:")));
}
