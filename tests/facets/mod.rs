use cuemap::facets::{compile_query_plan, extract_memory_facets};
use serde_json::json;
use std::collections::HashMap;

#[test]
fn extracts_structural_evidence_and_source_cues() {
    let mut metadata = HashMap::new();
    metadata.insert("source_role".to_string(), json!("user"));
    metadata.insert("source_date".to_string(), json!("2023-04-21"));

    let facets = extract_memory_facets(
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
fn keeps_surface_entities_without_classifying_them() {
    let facets = extract_memory_facets("Maya reviewed \"Project Atlas\" on Tuesday.", None, &[]);

    assert!(facets.iter().any(|facet| facet == "entity:maya"));
    assert!(facets.iter().any(|facet| facet == "entity:project_atlas"));
    assert!(facets.iter().any(|facet| facet == "has:date"));
    assert!(!facets.iter().any(|facet| facet.starts_with("type:")));
    assert!(!facets.iter().any(|facet| facet.starts_with("preference:")));
    assert!(!facets.iter().any(|facet| facet.starts_with("purchase:")));
}

#[test]
fn semantic_language_does_not_create_ontology_facets() {
    let facets = extract_memory_facets(
        "I prefer tea, bought a new mug, and always want recommendations.",
        None,
        &[],
    );

    assert!(!facets.iter().any(|facet| facet.starts_with("type:")));
    assert!(!facets.iter().any(|facet| facet.starts_with("preference:")));
    assert!(!facets.iter().any(|facet| facet.starts_with("instruction:")));
    assert!(!facets.iter().any(|facet| facet.starts_with("purchase:")));
}

#[test]
fn structural_evidence_detects_surface_formats_without_topic_classification() {
    let facets = extract_memory_facets(
        "Email me at kaan@example.com, open https://cuemap.dev/docs, and edit src/facets.rs.\n\n```rust\nlet answer = 42;\n```\nThe label is \"semantic\".",
        None,
        &[],
    );

    for expected in [
        "has:email",
        "has:url",
        "has:file_path",
        "has:code",
        "has:quote",
    ] {
        assert!(facets.iter().any(|facet| facet == expected), "missing {expected}: {facets:?}");
    }
    assert!(!facets.iter().any(|facet| facet.starts_with("preference:")));
    assert!(!facets.iter().any(|facet| facet.starts_with("purchase:")));
}

#[test]
fn url_facet_detects_http_urls_and_ignores_incomplete_schemes() {
    let facets = extract_memory_facets("Read https://cuemap.dev/docs#facets.", None, &[]);
    assert!(facets.iter().any(|facet| facet == "has:url"));

    let local_endpoints = extract_memory_facets(
        "Use http://localhost:3000 or http://127.0.0.1:8080; the UI is at localhost:5173.",
        None,
        &[],
    );
    assert!(local_endpoints.iter().any(|facet| facet == "has:url"));

    let incomplete = extract_memory_facets("The value is https://.", None, &[]);
    assert!(!incomplete.iter().any(|facet| facet == "has:url"));
}

#[test]
fn email_facet_detects_address_shapes() {
    let facets = extract_memory_facets("Contact kaan+cuemap@example.co.uk.", None, &[]);
    assert!(facets.iter().any(|facet| facet == "has:email"));

    let incomplete = extract_memory_facets("The address is kaan@example.", None, &[]);
    assert!(!incomplete.iter().any(|facet| facet == "has:email"));
}

#[test]
fn quote_facet_detects_bounded_quoted_spans() {
    let facets = extract_memory_facets("The exact label is \"semantic rerank\".", None, &[]);
    assert!(facets.iter().any(|facet| facet == "has:quote"));

    let incomplete = extract_memory_facets("The apostrophe is in don't.", None, &[]);
    assert!(!incomplete.iter().any(|facet| facet == "has:quote"));
}

#[test]
fn code_facet_detects_inline_and_fenced_code() {
    let inline = extract_memory_facets("Call `compile_query_plan` here.", None, &[]);
    assert!(inline.iter().any(|facet| facet == "has:code"));

    let fenced = extract_memory_facets("```rust\nlet answer = 42;\n```", None, &[]);
    assert!(fenced.iter().any(|facet| facet == "has:code"));

    let truncated = extract_memory_facets("The chunk ends at:\n```rust\nlet answer = 42;", None, &[]);
    assert!(truncated.iter().any(|facet| facet == "has:code"));

    let prose = extract_memory_facets("Use ordinary prose without delimiters.", None, &[]);
    assert!(!prose.iter().any(|facet| facet == "has:code"));
}

#[test]
fn file_path_facet_detects_paths_and_excludes_urls() {
    let relative = extract_memory_facets("Edit src/facets.rs.", None, &[]);
    assert!(relative.iter().any(|facet| facet == "has:file_path"));

    let absolute = extract_memory_facets("The file is /tmp/cuemap/state.json.", None, &[]);
    assert!(absolute.iter().any(|facet| facet == "has:file_path"));

    for file_name in ["Cargo.toml", "facets.rs", ".env", "model.safetensors"] {
        let facets = extract_memory_facets(file_name, None, &[]);
        assert!(facets.iter().any(|facet| facet == "has:file_name"), "missing file name cue for {file_name}: {facets:?}");
        assert!(!facets.iter().any(|facet| facet == "has:file_path"), "bare file name became a path: {file_name}: {facets:?}");
    }

    for path in ["/usr/bin", "../config", "src/components"] {
        let facets = extract_memory_facets(path, None, &[]);
        assert!(facets.iter().any(|facet| facet == "has:file_path"), "missing path cue for {path}: {facets:?}");
    }

    let url = extract_memory_facets("Open https://cuemap.dev/docs/facets.rs.", None, &[]);
    assert!(!url.iter().any(|facet| facet == "has:file_path"));
}

#[test]
fn clock_facets_reject_impossible_meridiem_hours_and_accept_24_hour_times() {
    let impossible = extract_memory_facets("The event is at 17 pm, not 23 am.", None, &[]);
    assert!(!impossible.iter().any(|facet| facet == "has:time"));
    assert!(!impossible.iter().any(|facet| facet.starts_with("time_of_day:")));

    let valid = extract_memory_facets("The event is at 17:00.", None, &[]);
    assert!(valid.iter().any(|facet| facet == "has:time"));
    assert!(valid.iter().any(|facet| facet == "time_of_day:evening"));
}

#[test]
fn may_is_only_a_month_in_temporal_context() {
    let modal = extract_memory_facets(
        "May I ask whether this may improve recall?",
        None,
        &[],
    );
    assert!(
        !modal.iter().any(|facet| facet == "has:date"),
        "modal facets: {modal:?}"
    );
    assert!(
        !modal.iter().any(|facet| facet == "content_month:05"),
        "modal facets: {modal:?}"
    );

    let date = extract_memory_facets(
        "The deployment was in May 2024 and the trip followed on May 3.",
        None,
        &[],
    );
    assert!(date.iter().any(|facet| facet == "has:date"));
    assert!(date.iter().any(|facet| facet == "content_month:05"));
}

#[test]
fn ambiguous_textual_months_require_calendar_context() {
    for prose in [
        "We march forward.",
        "March toward the entrance.",
        "An august presence filled the room.",
        "August joined the team.",
        "April joined the call.",
        "June joined the call.",
    ] {
        let facets = extract_memory_facets(prose, None, &[]);
        assert!(!facets.iter().any(|facet| facet == "has:date"), "false date for {prose:?}: {facets:?}");
        assert!(!facets.iter().any(|facet| facet.starts_with("content_month:")), "false month for {prose:?}: {facets:?}");
    }

    for dated in [
        "in March 2024",
        "march 3",
        "August 2024",
        "April 2024",
        "It is June.",
    ] {
        let facets = extract_memory_facets(dated, None, &[]);
        assert!(facets.iter().any(|facet| facet == "has:date"), "missing date for {dated:?}: {facets:?}");
        assert!(facets.iter().any(|facet| facet.starts_with("content_month:")), "missing month for {dated:?}: {facets:?}");
    }

    for modal in ["It may in fact work.", "Version 5 may fail."] {
        let facets = extract_memory_facets(modal, None, &[]);
        assert!(!facets.iter().any(|facet| facet == "has:date"), "false date for {modal:?}: {facets:?}");
        assert!(!facets.iter().any(|facet| facet == "content_month:05"), "false May for {modal:?}: {facets:?}");
    }
}

#[test]
fn query_shape_matching_uses_tokens_not_substrings() {
    let accidental = compile_query_plan("What is the smallest callback value?", |cue| {
        cue == "has:list"
    });
    assert!(!accidental
        .labels
        .iter()
        .any(|label| label == "multi_evidence_collection"));

    let explicit = compile_query_plan("List all callback values", |cue| cue == "has:list");
    assert!(explicit
        .labels
        .iter()
        .any(|label| label == "multi_evidence_collection"));
}

#[test]
fn query_plan_only_emits_structural_or_query_shape_signals() {
    let intent = compile_query_plan("Summarize the events from yesterday", |cue| {
        cue == "has:list" || cue.starts_with("temporal:")
    });

    assert!(intent.labels.iter().any(|label| label == "multi_evidence_summary"));
    assert!(intent.labels.iter().any(|label| label == "temporal_yesterday"));
    assert!(!intent.labels.iter().any(|label| label.contains("preference")));
    assert!(!intent.labels.iter().any(|label| label.contains("purchase")));
}

#[test]
fn assistant_authored_first_person_text_keeps_assistant_source_role() {
    let mut metadata = HashMap::new();
    metadata.insert("source_role".to_string(), json!("assistant"));

    let facets = extract_memory_facets(
        "I recommend using Flask routes for this project.",
        Some(&metadata),
        &[],
    );

    assert!(facets.iter().any(|facet| facet == "source_role:assistant"));
    assert!(!facets.iter().any(|facet| facet == "source_role:user"));

    let prefixed_facets = extract_memory_facets(
        "assistant: I recommend using Flask routes for this project.",
        None,
        &[],
    );
    assert!(prefixed_facets
        .iter()
        .any(|facet| facet == "source_role:assistant"));
    assert!(!prefixed_facets
        .iter()
        .any(|facet| facet == "source_role:user"));
}

#[test]
fn query_perspective_is_grammar_only_and_supports_first_person_variants() {
    for query in [
        "Have I worked with Flask routes?",
        "Did I build the project?",
        "What do I need to do?",
        "Didn't I mention the project?",
    ] {
        let intent = compile_query_plan(query, |cue| cue == "source_role:user");

        assert!(
            intent
                .labels
                .iter()
                .any(|label| label == "query_perspective_first_person"),
            "missing first-person perspective for {query:?}: {intent:?}"
        );
        assert!(intent.labels.iter().any(|label| label == "source_user"));
        assert!(intent
            .weighted_cues
            .iter()
            .any(|(cue, weight)| cue == "source_role:user" && *weight == 2.0));
    }
}

#[test]
fn query_perspective_keeps_second_and_third_person_separate_from_source_role() {
    let second_person = compile_query_plan("What did you recommend?", |cue| {
        cue == "source_role:assistant"
    });
    assert!(second_person
        .labels
        .iter()
        .any(|label| label == "query_perspective_second_person"));
    assert!(second_person
        .labels
        .iter()
        .any(|label| label == "source_assistant"));
    assert!(second_person
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_role:assistant" && *weight == 2.0));

    let third_person = compile_query_plan("What did they recommend?", |cue| {
        cue == "source_role:user" || cue == "source_role:assistant"
    });
    assert!(third_person
        .labels
        .iter()
        .any(|label| label == "query_perspective_third_person"));
    assert!(!third_person.labels.iter().any(|label| {
        label == "source_user" || label == "source_assistant"
    }));
    assert!(!third_person
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "source_role:user" || cue == "source_role:assistant"));
}

#[test]
fn request_wrappers_use_embedded_perspective_instead_of_outer_you() {
    let event_ordering = compile_query_plan(
        "Can you list the order in which I brought up different aspects?",
        |cue| cue == "source_role:user" || cue == "source_role:assistant",
    );
    assert!(event_ordering
        .labels
        .iter()
        .any(|label| label == "query_perspective_first_person"));
    assert!(event_ordering.labels.iter().any(|label| label == "source_user"));
    assert!(!event_ordering
        .labels
        .iter()
        .any(|label| label == "source_assistant"));
    assert!(event_ordering
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_role:user" && *weight == 2.0));

    let assistant_target = compile_query_plan("Can you tell me what you recommended?", |cue| {
        cue == "source_role:assistant"
    });
    assert!(assistant_target
        .labels
        .iter()
        .any(|label| label == "query_perspective_second_person"));
    assert!(assistant_target
        .labels
        .iter()
        .any(|label| label == "source_assistant"));

    let plain_request = compile_query_plan("Can you recommend a dessert?", |cue| {
        cue == "source_role:user" || cue == "source_role:assistant"
    });
    assert!(!plain_request.labels.iter().any(|label| {
        label == "query_perspective_first_person"
            || label == "query_perspective_second_person"
            || label == "query_perspective_third_person"
            || label == "source_user"
            || label == "source_assistant"
    }));
}

#[test]
fn embedded_question_perspective_beats_outer_wrapper_and_conflicts_are_unset() {
    let user_target = compile_query_plan("Do you remember what I bought?", |cue| {
        cue == "source_role:user" || cue == "source_role:assistant"
    });
    assert!(user_target
        .labels
        .iter()
        .any(|label| label == "query_perspective_first_person"));
    assert!(user_target.labels.iter().any(|label| label == "source_user"));
    assert!(!user_target
        .labels
        .iter()
        .any(|label| label == "source_assistant"));

    let assistant_target = compile_query_plan("Do you remember what you recommended?", |cue| {
        cue == "source_role:user" || cue == "source_role:assistant"
    });
    assert!(assistant_target
        .labels
        .iter()
        .any(|label| label == "query_perspective_second_person"));
    assert!(assistant_target
        .labels
        .iter()
        .any(|label| label == "source_assistant"));

    let conflicting = compile_query_plan(
        "Do you remember what I bought and what you recommended?",
        |cue| cue == "source_role:user" || cue == "source_role:assistant",
    );
    assert!(!conflicting.labels.iter().any(|label| {
        label == "source_user" || label == "source_assistant"
    }));
    assert!(!conflicting.weighted_cues.iter().any(|(cue, _)| {
        cue == "source_role:user" || cue == "source_role:assistant"
    }));
}

#[test]
fn non_wrapper_embedded_perspective_disagreement_is_unweighted() {
    for query in [
        "Who did I meet when you visited?",
        "What did I say about what she bought?",
    ] {
        let plan = compile_query_plan(query, |cue| {
            cue == "source_role:user" || cue == "source_role:assistant"
        });
        assert!(!plan.labels.iter().any(|label| {
            label == "source_user" || label == "source_assistant"
        }), "unexpected source preference for {query:?}: {plan:?}");
        assert!(!plan.weighted_cues.iter().any(|(cue, _)| {
            cue == "source_role:user" || cue == "source_role:assistant"
        }), "unexpected source cue for {query:?}: {plan:?}");
    }

    let matching = compile_query_plan("What did I say about what I bought?", |cue| {
        cue == "source_role:user"
    });
    assert!(matching.labels.iter().any(|label| label == "source_user"));
}

#[test]
fn extracts_quantities_percentages_ranges_and_comparisons() {
    let facets = extract_memory_facets(
        "Latency was 5 ms, the payload was 512 MB, accuracy was 20%, and the safe range was between 4 and 6 kg.",
        None,
        &[],
    );

    for expected in [
        "has:measurement",
        "quantity_unit:ms",
        "measurement:5_ms",
        "quantity_unit:mb",
        "measurement:512_mb",
        "has:percentage",
        "percentage:20",
        "has:numeric_range",
        "has:comparator",
        "comparison:between",
        "range_min:4",
        "range_max:6",
        "quantity_unit:kg",
        "range:4_6_kg",
    ] {
        assert!(facets.iter().any(|facet| facet == expected), "missing {expected}: {facets:?}");
    }

    let comparison = extract_memory_facets("Keep the response under 5 ms, or at least 90% accurate.", None, &[]);
    assert!(comparison.iter().any(|facet| facet == "comparison:less_than"));
    assert!(comparison.iter().any(|facet| facet == "comparison:greater_than"));
    assert!(comparison.iter().any(|facet| facet == "has:comparator"));
}

#[test]
fn extracts_technical_identifiers_without_reusing_agent_namespaces() {
    let facets = extract_memory_facets(
        "UUID 550e8400-e29b-41d4-a716-446655440000, version v0.7.2, PR #142, GH-143, commit a91f72c, endpoint 127.0.0.1:8080, cuemap.dev, CUEMAP_INDEX_PATH, @kaan, and #retrieval.",
        None,
        &[],
    );

    for expected in [
        "has:uuid",
        "uuid:550e8400_e29b_41d4_a716_446655440000",
        "has:semver",
        "version:0_7_2",
        "has:issue_reference",
        "issue:142",
        "issue:143",
        "has:commit_hash",
        "commit:a91f72c",
        "has:ip_address",
        "ip:127_0_0_1",
        "has:port",
        "port:8080",
        "has:domain",
        "domain_name:cuemap_dev",
        "has:environment_variable",
        "env:cuemap_index_path",
        "has:user_mention",
        "mention:kaan",
        "has:hashtag",
        "hashtag:retrieval",
    ] {
        assert!(facets.iter().any(|facet| facet == expected), "missing {expected}: {facets:?}");
    }
    assert!(!facets.iter().any(|facet| facet.starts_with("lang:")));
    assert!(!facets.iter().any(|facet| facet.starts_with("type:")));
}

#[test]
fn extracts_file_names_extensions_and_directory_segments() {
    let facets = extract_memory_facets(
        "Edit src/facets.rs, then update Cargo.toml, .env, and model.safetensors.",
        None,
        &[],
    );

    for expected in [
        "has:file_name",
        "has:file_path",
        "has:directory_path",
        "path_segment:src",
        "file_name:facets_rs",
        "file_extension:rs",
        "file_name:cargo_toml",
        "file_extension:toml",
        "file_name:env",
        "file_name:model_safetensors",
        "file_extension:safetensors",
    ] {
        assert!(facets.iter().any(|facet| facet == expected), "missing {expected}: {facets:?}");
    }
    assert!(!facets.iter().any(|facet| facet.starts_with("file:")));
    assert!(!facets.iter().any(|facet| facet.starts_with("path:")));
}

#[test]
fn extracts_document_and_code_structure_markers() {
    let content = r#"{"name":"cuemap"}
name: cuemap
enabled: true
<root><item>value</item></root>
a,b
c,d

| Name | Value |
| --- | --- |
| mode | fast |

Traceback (most recent call last):
  at src/main.rs:42

diff --git a/a.rs b/a.rs
@@ -1 +1 @@

## Install
- [x] Build the engine
> Preserve this note.
[documentation](https://cuemap.dev/docs)
```rust
let answer = 42;
```"#;
    let facets = extract_memory_facets(content, None, &[]);

    for expected in [
        "has:json",
        "has:key_value_pairs",
        "has:yaml",
        "has:xml",
        "has:csv",
        "has:markdown_table",
        "has:stack_trace",
        "has:diff",
        "has:heading",
        "heading_level:2",
        "has:checklist",
        "has:block_quote",
        "has:markdown_link",
        "code_language:rust",
    ] {
        assert!(facets.iter().any(|facet| facet == expected), "missing {expected}: {facets:?}");
    }
}

#[test]
fn extracts_negation_contrast_correction_and_supersession() {
    let facets = extract_memory_facets(
        "I never chose the old option, but actually changed my mind and used to prefer it instead of the new one.",
        None,
        &[],
    );

    for expected in [
        "has:negation",
        "has:contrast",
        "has:correction",
        "has:supersession",
    ] {
        assert!(facets.iter().any(|facet| facet == expected), "missing {expected}: {facets:?}");
    }
}

#[test]
fn extracts_emoji_without_script_noise() {
    let facets = extract_memory_facets("Hello мир 世界 مرحبا नमस्ते 👋", None, &[]);

    assert!(facets.iter().any(|facet| facet == "has:emoji"), "missing emoji facet: {facets:?}");
    assert!(facets.iter().all(|facet| !facet.starts_with("script:") && facet != "has:script"), "unexpected script facets: {facets:?}");
}

#[test]
fn query_plan_emits_bounded_answer_shape_labels() {
    for (query, expected) in [
        ("Who did I meet?", "answer_shape_person"),
        ("Where did I go?", "answer_shape_location"),
        ("When did it happen?", "answer_shape_time"),
        ("How many memories mention Flask?", "answer_shape_count"),
        ("How much did it cost?", "answer_shape_amount"),
        ("Why did I choose it?", "answer_shape_reason"),
        ("How long did it take?", "answer_shape_duration"),
        ("Which option did I select?", "answer_shape_selection"),
        ("What kind of file is this?", "answer_shape_category"),
        ("Did I deploy it?", "answer_shape_boolean"),
    ] {
        let plan = compile_query_plan(query, |_| false);
        assert!(plan.labels.iter().any(|label| label == expected), "missing {expected} for {query:?}: {plan:?}");
    }
}
