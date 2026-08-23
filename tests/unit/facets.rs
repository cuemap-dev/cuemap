    use super::{compile_query_plan, extract_memory_facets_core};
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn structural_facets_keep_evidence_and_source_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), json!("user"));
        metadata.insert("source_date".to_string(), json!("2023-04-21"));
        let facets = extract_memory_facets_core(
            "The meeting is next Friday at 7:30 PM and costs $20.",
            Some(&metadata),
            &[],
        );
        for expected in [
            "source_role:user",
            "source_time:dated",
            "source_date:2023_04_21",
            "source_week:2023_w16",
            "has:number",
            "has:money",
            "has:time",
            "time_of_day:evening",
            "temporal:relative",
        ] {
            assert!(facets.iter().any(|facet| facet == expected), "missing {expected}: {facets:?}");
        }
    }

    #[test]
    fn semantic_language_does_not_create_ontology_facets() {
        let facets = extract_memory_facets_core(
            "I prefer tea, bought a new mug, and always want recommendations.",
            None,
            &[],
        );
        assert!(!facets.iter().any(|facet| facet.starts_with("type:")));
        assert!(!facets.iter().any(|facet| facet.starts_with("preference:")));
        assert!(!facets.iter().any(|facet| facet.starts_with("purchase:")));
    }

    #[test]
    fn query_plan_only_emits_structural_or_retrieval_shape_signals() {
        let intent = compile_query_plan("Summarize the events from yesterday", |cue| {
            cue == "has:list" || cue.starts_with("temporal:")
        });
        assert!(intent.labels.iter().any(|label| label == "multi_evidence_summary"));
        assert!(intent.labels.iter().any(|label| label == "temporal_yesterday"));
        assert!(!intent.labels.iter().any(|label| label.contains("preference")));
    }

    #[test]
    fn may_modal_is_not_seen_as_a_temporal_month() {
        assert!(!super::date_re().is_match("May I ask whether this may improve recall."));
        assert!(!super::has_temporal_month("May I ask whether this may improve recall."));
    }

    #[test]
    fn quantity_regex_captures_short_units_and_ranges() {
        let measurement = super::measurement_re().captures("5 ms").expect("measurement");
        assert_eq!(measurement.name("value").map(|value| value.as_str()), Some("5"));
        assert_eq!(measurement.name("unit").map(|unit| unit.as_str()), Some("ms"));
        assert_eq!(super::canonical_quantity_unit("ms"), Some("ms"));
        assert!(super::between_range_re().is_match("between 4 and 6 kg"));
    }
