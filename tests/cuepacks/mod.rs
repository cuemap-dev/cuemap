use cuemap::cuepacks::{default_registry, CuePackRegistry};
use cuemap::facets::{
    compile_query_intent_with_cuepacks, extract_memory_facets, extract_memory_facets_core,
    extract_memory_facets_with_cuepacks,
};
use std::fs;

#[test]
fn bundled_memory_general_pack_loads_by_default() {
    let infos = default_registry().infos();
    let pack = infos
        .iter()
        .find(|info| info.name == "memory-general")
        .expect("memory-general CuePack should be bundled");

    assert!(pack.enabled_by_default);
    assert!(pack.memory_rules > 0);
    assert!(pack.query_rules > 0);
}

#[test]
fn core_extractor_stays_structural_while_default_pack_restores_domain_facets() {
    let content = "I downloaded Google Maps for directions to the airport station by train.";

    let core = extract_memory_facets_core(content, None, &[]);
    assert!(!core.contains(&"type:navigation".to_string()));
    assert!(!core.contains(&"travel:route".to_string()));
    assert!(!core.contains(&"travel:transit".to_string()));

    let default = extract_memory_facets(content, None, &[]);
    assert!(default.contains(&"type:navigation".to_string()));
    assert!(default.contains(&"travel:route".to_string()));
    assert!(default.contains(&"travel:transit".to_string()));
}

#[test]
fn cuepack_selection_can_disable_default_memory_facets() {
    let content = "I started using a music streaming service for playlists.";
    let off = vec!["off".to_string()];
    let facets =
        extract_memory_facets_with_cuepacks(content, None, &[], default_registry(), Some(&off));

    assert!(!facets.contains(&"media:music_streaming".to_string()));
    assert!(!facets.contains(&"media:streaming".to_string()));
}

#[test]
fn bundled_cuepack_extracts_standing_instruction_facets_and_trigger_cues() {
    let content =
        "Always provide fallback strategies when I ask about error handling in API services.";
    let facets = extract_memory_facets(content, None, &[]);

    assert!(facets.contains(&"type:standing_instruction".to_string()));
    assert!(facets.contains(&"instruction:conditional".to_string()));
    assert!(facets.contains(&"instruction:always".to_string()));
    assert!(facets.contains(&"instruction_trigger:api".to_string()));
    assert!(facets.contains(&"instruction_action:fallback".to_string()));

    let off = vec!["off".to_string()];
    let core_only =
        extract_memory_facets_with_cuepacks(content, None, &[], default_registry(), Some(&off));
    assert!(!core_only.contains(&"type:standing_instruction".to_string()));
    assert!(!core_only.contains(&"instruction_trigger:api".to_string()));
}

#[test]
fn bundled_cuepack_labels_advice_queries_as_instruction_applicable() {
    let intent = compile_query_intent_with_cuepacks(
        "What are some ways I can manage problems that come up when my API calls fail?",
        None,
        |_| false,
        default_registry(),
        None,
    );

    assert!(intent
        .labels
        .contains(&"instruction_applicable".to_string()));
    assert!(intent
        .cuepack_rules
        .iter()
        .any(|rule| rule == "memory-general:memory.instruction_applicable"));
}

#[test]
fn bundled_cuepack_labels_when_and_if_queries_as_instruction_applicable() {
    for query in [
        "When building an application that talks to an API, what should I watch for?",
        "If I'm creating a page layout, how should I structure the markup?",
    ] {
        let intent =
            compile_query_intent_with_cuepacks(query, None, |_| false, default_registry(), None);

        assert!(
            intent
                .labels
                .contains(&"instruction_applicable".to_string()),
            "query should be instruction-applicable: {query}"
        );
        assert!(
            intent
                .cuepack_rules
                .iter()
                .any(|rule| rule == "memory-general:memory.instruction_applicable"),
            "query should be labeled by memory-general instruction rule: {query}"
        );
    }
}

#[test]
fn bundled_cuepack_extracts_preference_facets_and_value_cues() {
    let content = "I prefer geometric vector methods over purely trigonometric formulas for clarity, so can you explain how to use vector algebra to calculate geodesic length between two points on a sphere?";
    let facets = extract_memory_facets(content, None, &[]);

    assert!(facets.contains(&"type:preference".to_string()));
    assert!(facets.contains(&"preference:explicit".to_string()));
    assert!(facets.contains(&"preference_value:geometric".to_string()));
    assert!(facets.contains(&"preference_value:vector".to_string()));
    assert!(facets.contains(&"preference_contrast:trigonometric".to_string()));

    let off = vec!["off".to_string()];
    let core_only =
        extract_memory_facets_with_cuepacks(content, None, &[], default_registry(), Some(&off));
    assert!(core_only.contains(&"type:preference".to_string()));
    assert!(!core_only.contains(&"preference:explicit".to_string()));
    assert!(!core_only.contains(&"preference_value:vector".to_string()));
}

#[test]
fn bundled_cuepack_labels_task_queries_as_preference_applicable() {
    let intent = compile_query_intent_with_cuepacks(
        "Can you show me how to find the shortest path between two points on a sphere?",
        None,
        |_| false,
        default_registry(),
        None,
    );

    assert!(intent
        .labels
        .contains(&"preference_applicable".to_string()));
    assert!(intent
        .cuepack_rules
        .iter()
        .any(|rule| rule == "memory-general:memory.preference_applicable"));
}

#[test]
fn bundled_cuepack_labels_near_future_task_queries_as_preference_applicable() {
    let intent = compile_query_intent_with_cuepacks(
        "I'm about to start editing a long draft; what steps would you suggest?",
        None,
        |_| false,
        default_registry(),
        None,
    );

    assert!(intent
        .labels
        .contains(&"preference_applicable".to_string()));
    assert!(intent
        .cuepack_rules
        .iter()
        .any(|rule| rule == "memory-general:memory.preference_applicable"));
}

#[test]
fn bundled_cuepack_labels_summary_queries_as_multi_evidence() {
    let intent = compile_query_intent_with_cuepacks(
        "Can you provide a detailed summary of everything we covered about deployment planning?",
        None,
        |_| false,
        default_registry(),
        None,
    );

    assert!(intent
        .labels
        .contains(&"multi_evidence_summary".to_string()));
    assert!(intent
        .cuepack_rules
        .iter()
        .any(|rule| rule == "memory-general:memory.multi_evidence_summary"));
}

#[test]
fn bundled_cuepack_labels_overview_queries_as_multi_evidence() {
    let intent = compile_query_intent_with_cuepacks(
        "Can you give me a comprehensive overview of the key details from the project?",
        None,
        |_| false,
        default_registry(),
        None,
    );

    assert!(intent
        .labels
        .contains(&"multi_evidence_summary".to_string()));
    assert!(intent
        .cuepack_rules
        .iter()
        .any(|rule| rule == "memory-general:memory.multi_evidence_summary"));
}

#[test]
fn bundled_cuepack_labels_collection_queries_as_multi_evidence_collection() {
    let intent = compile_query_intent_with_cuepacks(
        "What activities has Melanie done with her family?",
        None,
        |cue| cue == "has:list",
        default_registry(),
        None,
    );

    assert!(intent
        .labels
        .contains(&"multi_evidence_collection".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "has:list"));
    assert!(intent
        .cuepack_rules
        .iter()
        .any(|rule| rule == "memory-general:memory.multi_evidence_collection"));
}

#[test]
fn bundled_cuepack_extracts_inline_enumeration_as_list_evidence() {
    let facets = extract_memory_facets(
        "I like eating greens such as lettuce, spinach, and arugula.",
        None,
        &[],
    );

    assert!(facets.contains(&"has:list".to_string()));
}

#[test]
fn cuepack_query_rules_emit_available_weighted_cues_with_provenance() {
    let available = |cue: &str| {
        matches!(
            cue,
            "type:navigation" | "travel:route" | "travel:transit" | "media:streaming"
        )
    };
    let intent = compile_query_intent_with_cuepacks(
        "What transit app did I use to get around?",
        None,
        available,
        default_registry(),
        None,
    );

    assert!(intent.labels.contains(&"navigation".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:navigation"));
    assert!(intent
        .cuepack_rules
        .iter()
        .any(|rule| rule == "memory-general:navigation.query"));
}

#[test]
fn bundled_cuepack_maps_aquarium_queries_to_tank_without_core_rule() {
    let available = |cue: &str| cue == "tank";
    let intent = compile_query_intent_with_cuepacks(
        "How many fish are there in total in both of my aquariums?",
        None,
        available,
        default_registry(),
        None,
    );

    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "tank" && *weight > 3.0));
    assert!(intent
        .cuepack_rules
        .iter()
        .any(|rule| rule == "memory-general:aquarium.tank_alias"));
}

#[test]
fn core_query_intent_does_not_hardcode_aquarium_alias() {
    let intent = cuemap::facets::compile_query_intent(
        "How many fish are there in total in both of my aquariums?",
        |cue| cue == "tank",
    );

    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "tank"));
}

#[test]
fn bundled_cuepack_maps_room_furniture_queries_to_available_furniture_items() {
    let available = |cue: &str| cue == "dresser";
    let intent = compile_query_intent_with_cuepacks(
        "Any tips for rearranging the furniture in my bedroom?",
        None,
        available,
        default_registry(),
        None,
    );

    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "dresser" && *weight > 4.0));
    assert!(intent
        .cuepack_rules
        .iter()
        .any(|rule| rule == "memory-general:home.bedroom_furniture_alias"));
}

#[test]
fn core_query_intent_does_not_hardcode_room_furniture_aliases() {
    let intent = cuemap::facets::compile_query_intent(
        "Any tips for rearranging the furniture in my bedroom?",
        |cue| cue == "dresser",
    );

    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "dresser"));
}

#[test]
fn custom_cuepack_toml_validates_and_overrides_loaded_pack_by_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("memory-general.toml");
    fs::write(
        &path,
        r#"
name = "memory-general"
version = "9.9.9"
description = "test override"
enabled_by_default = true

[[memory_rules]]
id = "custom"
contains_any = ["cardiology"]
emits = ["domain:cardiology"]
"#,
    )
    .expect("write cuepack");

    let info = CuePackRegistry::validate_file(&path).expect("valid cuepack");
    assert_eq!(info.name, "memory-general");
    assert_eq!(info.version, "9.9.9");

    let registry = CuePackRegistry::load(true, &[dir.path().to_path_buf()]);
    let facets = registry.extract_memory_facets("I track cardiology notes.", None);
    assert!(facets.facets.contains(&"domain:cardiology".to_string()));
    assert!(facets
        .matched_rules
        .contains(&"memory-general:custom".to_string()));
}
