use cuemap::engine::CueMapEngine;
use cuemap::structures::MainStats;
use std::collections::HashMap;

#[test]
fn test_temporal_chunking() {
    let engine = CueMapEngine::new();
    let mut metadata = HashMap::new();
    metadata.insert("project_id".to_string(), serde_json::json!("p1"));

    let id1 = engine.add_memory(
        "event 1".to_string(),
        vec!["topic:coding".to_string()],
        Some(metadata.clone()),
        MainStats::default(),
        false,
    );
    let id2 = engine.add_memory(
        "event 2".to_string(),
        vec!["topic:coding".to_string()],
        Some(metadata),
        MainStats::default(),
        false,
    );

    let mem2 = engine.get_memory(id2).unwrap();
    let episode_cue = format!("episode:{}", id1);
    assert!(
        mem2.cues.contains(&episode_cue),
        "Second memory should have episode cue pointing to the first"
    );
    let episode_results = engine.recall(vec![episode_cue], 5, false, None);
    assert!(
        episode_results.iter().any(|r| r.memory_id == id2),
        "Episode cue should be indexed for recall"
    );
}

#[test]
fn test_salience_bias() {
    let engine = CueMapEngine::new();

    // High cue density memory
    let mut salient_stats = MainStats::default();
    salient_stats.intrinsic_salience = 50.0;
    let id_salient = engine.add_memory(
        "short".to_string(),
        vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ],
        None,
        salient_stats,
        false,
    );
    // Normal memory
    let _id_normal = engine.add_memory(
        "this is a much longer content for a normal memory".to_string(),
        vec!["a".to_string()],
        None,
        MainStats::default(),
        false,
    );

    let results = engine.recall(vec!["a".to_string()], 10, false, None);

    assert_eq!(results[0].memory_id, id_salient, "Salient memory should rank first even if newer memory exists if it has much higher salience");
}

#[test]
fn test_match_integrity_scores() {
    let engine = CueMapEngine::new();

    let id_exact = engine.add_memory(
        "exact match".to_string(),
        vec!["a".to_string(), "b".to_string()],
        None,
        MainStats::default(),
        false,
    );
    let id_partial = engine.add_memory(
        "partial match".to_string(),
        vec![
            "a".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ],
        None,
        MainStats::default(),
        false,
    );

    let results = engine.recall(vec!["a".to_string(), "b".to_string()], 10, false, None);

    let res_exact = results.iter().find(|r| r.memory_id == id_exact).unwrap();
    let res_partial = results.iter().find(|r| r.memory_id == id_partial).unwrap();

    assert!(
        res_exact.match_integrity > res_partial.match_integrity,
        "Exact match should have higher match integrity than partial match"
    );
}

#[test]
fn test_systems_consolidation() {
    let engine = CueMapEngine::new();

    engine.add_memory(
        "report part 1".to_string(),
        vec!["type:report".to_string(), "month:jan".to_string()],
        None,
        MainStats::default(),
        false,
    );
    engine.add_memory(
        "report part 2".to_string(),
        vec!["type:report".to_string(), "month:jan".to_string()],
        None,
        MainStats::default(),
        false,
    );

    let initial_count = engine
        .get_stats()
        .get("total_memories")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(initial_count, 2);

    // Lower threshold because temporal chunking adds an episode cue
    let consolidated = engine.consolidate_memories(0.6);
    assert_eq!(consolidated.len(), 1);

    let final_count = engine
        .get_stats()
        .get("total_memories")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(final_count, 3); // 2 original + 1 summary (additive)

    let mem = engine.get_memory(consolidated[0].0).unwrap();
    let content = mem.access_content(None).unwrap();
    assert!(mem.metadata.contains_key("consolidated"));
    assert!(content.contains("report part 1"));
    assert!(content.contains("report part 2"));
}
