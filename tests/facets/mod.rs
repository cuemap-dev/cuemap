use cuemap::facets::{compile_query_intent, extract_memory_facets};
use cuemap::config::CueGenStrategy;
use cuemap::engine::CueMapEngine;
use cuemap::structures::MainStats;
use serde_json::json;
use std::collections::HashMap;

#[test]
fn extracts_general_source_and_evidence_facets() {
    let mut metadata = HashMap::new();
    metadata.insert("role".to_string(), json!("Doctor"));
    metadata.insert("channel".to_string(), json!("support inbox"));

    let facets = extract_memory_facets(
        "Doctor: I currently take 20 mg daily for 2 weeks and paid $15 last week.",
        Some(&metadata),
        &[],
    );

    assert!(facets.contains(&"source_role:doctor".to_string()));
    assert!(facets.contains(&"source_channel:support_inbox".to_string()));
    assert!(facets.contains(&"has:number".to_string()));
    assert!(facets.contains(&"has:money".to_string()));
    assert!(facets.contains(&"has:duration".to_string()));
    assert!(facets.contains(&"temporal:current".to_string()));
    assert!(facets.contains(&"temporal:last_week".to_string()));
}

#[test]
fn extracts_type_and_entity_facets_without_benchmark_roles() {
    let facets = extract_memory_facets(
        "Chef: I prefer Sony A7R IV photos, dislike cinnamon, and bought \"Peak Design Bag\".",
        None,
        &[],
    );

    assert!(facets.contains(&"source_role:chef".to_string()));
    assert!(facets.contains(&"type:preference".to_string()));
    assert!(facets.contains(&"type:dislike".to_string()));
    assert!(facets.contains(&"type:ownership".to_string()));
    assert!(facets.contains(&"entity:sony_a7r_iv".to_string()));
    assert!(facets.contains(&"entity:peak_design_bag".to_string()));
}

#[test]
fn does_not_extract_question_words_or_source_labels_as_entities() {
    let facets = extract_memory_facets(
        "User: What breed is Max? Assistant: Max is a Golden Retriever.",
        None,
        &[],
    );

    assert!(facets.contains(&"source_role:user".to_string()));
    assert!(facets.contains(&"entity:max".to_string()));
    assert!(facets.contains(&"entity:golden_retriever".to_string()));
    assert!(!facets.contains(&"entity:what".to_string()));
    assert!(!facets.contains(&"entity:user".to_string()));
    assert!(!facets.contains(&"entity:assistant".to_string()));
}

#[test]
fn compiles_query_intent_only_for_available_facets() {
    let available = |cue: &str| matches!(cue, "has:number" | "has:money" | "type:preference");
    let intent = compile_query_intent("How much money did I spend on my favorite camera?", available);

    assert!(intent.labels.contains(&"money".to_string()));
    assert!(intent.labels.contains(&"preference".to_string()));
    assert!(intent.suppress_generic);
    assert!(intent.weighted_cues.iter().any(|(cue, weight)| cue == "has:money" && *weight > 3.0));
    assert!(intent.weighted_cues.iter().any(|(cue, _)| cue == "type:preference"));
    assert!(!intent.weighted_cues.iter().any(|(cue, _)| cue == "has:duration"));
}

#[test]
fn deprecated_cuegen_values_still_parse_for_compatibility() {
    let glove: CueGenStrategy = serde_json::from_str("\"glove\"").unwrap();
    let ollama: CueGenStrategy = serde_json::from_str("\"ollama\"").unwrap();

    assert!(matches!(glove, CueGenStrategy::Glove));
    assert!(matches!(ollama, CueGenStrategy::Ollama));
}

#[test]
fn weighted_facet_query_reranks_structured_evidence() {
    let engine = CueMapEngine::new();
    engine.add_memory(
        "Camera maintenance notes and lens cleaning checklist.".to_string(),
        vec!["camera".to_string()],
        None,
        MainStats::default(),
        false,
    );
    engine.add_memory(
        "I prefer Fuji cameras for street photography.".to_string(),
        vec!["camera".to_string()],
        None,
        MainStats::default(),
        false,
    );

    let results = engine.recall_weighted(
        vec![("camera".to_string(), 1.0), ("type:preference".to_string(), 3.0)],
        2,
        false,
        None,
        1,
        true,
        true,
        true,
        true,
        None,
        None,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some("I prefer Fuji cameras for street photography."));
}
