use cuemap::config::ServerConfig;
use cuemap::cuebridge::CueBridgeArtifacts;
use cuemap::projects::ProjectContext;
use cuemap::structures::MainStats;
use cuemap::{normalization::NormalizationConfig, taxonomy::Taxonomy};
use std::fs;
use std::sync::Arc;

fn write_artifact(dir: &std::path::Path, name: &str, content: &str) {
    fs::create_dir_all(dir).expect("create artifact dir");
    fs::write(dir.join(name), content).expect("write artifact");
}

fn context_with_data_dir(project_id: &str, data_dir: &std::path::Path) -> ProjectContext {
    let mut config = ServerConfig::default();
    config.server.data_dir = data_dir.to_string_lossy().to_string();
    ProjectContext::new(
        NormalizationConfig::default(),
        Taxonomy::default(),
        Arc::new(Default::default()),
        config,
        project_id.to_string(),
    )
}

#[test]
fn cuebridge_loads_gap_and_alias_packs_for_project() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_dir = tmp.path().join("artifacts").join("proj");
    write_artifact(
        &artifact_dir,
        "gap.json",
        r#"{
          "schema_version": 1,
          "artifact_type": "gap_pack",
          "name": "test-gap",
          "entries": [{
            "id": "gap_1",
            "query_signature": {"required_any": ["car"]},
            "expansions": [{"cue": "automobile", "weight": 1.5}],
            "confidence": 0.9,
            "max_fanout": 1,
            "provenance": {}
          }]
        }"#,
    );
    write_artifact(
        &artifact_dir,
        "alias.json",
        r#"{
          "schema_version": 1,
          "artifact_type": "alias_pack",
          "name": "test-alias",
          "entries": [{
            "id": "alias_1",
            "from": "todo",
            "to": "task",
            "weight": 0.8,
            "confidence": 0.95,
            "provenance": {}
          }]
        }"#,
    );

    let artifacts = CueBridgeArtifacts::load_for_project(&tmp.path().to_string_lossy(), "proj");
    let summary = artifacts.summary();

    assert_eq!(summary.artifact_count, 2);
    assert_eq!(summary.gap_entry_count, 1);
    assert_eq!(summary.alias_entry_count, 1);
    assert!(summary.load_errors.is_empty());
    assert!(summary.artifacts.iter().all(|artifact| artifact.sha256.len() == 64));
}

#[test]
fn alias_pack_expands_only_available_query_cues() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project_id = "alias_proj";
    let artifact_dir = tmp.path().join("artifacts").join(project_id);
    write_artifact(
        &artifact_dir,
        "alias.json",
        r#"{
          "schema_version": 1,
          "artifact_type": "alias_pack",
          "name": "test-alias",
          "entries": [
            {
              "id": "alias_1",
              "from": "todo",
              "to": "task",
              "weight": 0.8,
              "confidence": 0.9,
              "provenance": {}
            },
            {
              "id": "alias_2",
              "from": "todo",
              "to": "missing_target",
              "weight": 0.8,
              "confidence": 0.9,
              "provenance": {}
            }
          ]
        }"#,
    );
    let ctx = context_with_data_dir(project_id, tmp.path());
    ctx.main.add_memory(
        "finish the tax prep task".to_string(),
        vec!["task".to_string()],
        None,
        MainStats::default(),
        false,
    );

    let (expanded, trace) =
        ctx.expand_query_cues_with_trace(vec!["todo".to_string()], &["todo".to_string()]);

    assert!(expanded.iter().any(|(cue, _)| cue == "task"));
    assert!(!expanded.iter().any(|(cue, _)| cue == "missing_target"));
    assert_eq!(trace.len(), 1);
    assert_eq!(trace[0].entry_id, "alias_1");
}

#[test]
fn gap_pack_expands_only_when_query_signature_matches() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_dir = tmp.path().join("artifacts").join("gap_proj");
    write_artifact(
        &artifact_dir,
        "gap.json",
        r#"{
          "schema_version": 1,
          "artifact_type": "gap_pack",
          "name": "test-gap",
          "entries": [
            {
              "id": "gap_car",
              "query_signature": {"required_any": ["car"]},
              "expansions": [{"cue": "automobile", "weight": 1.5}],
              "confidence": 0.9,
              "max_fanout": 1,
              "provenance": {}
            },
            {
              "id": "gap_food",
              "query_signature": {"required_any": ["pie"]},
              "expansions": [{"cue": "dessert", "weight": 1.5}],
              "confidence": 0.9,
              "max_fanout": 1,
              "provenance": {}
            }
          ]
        }"#,
    );
    let artifacts =
        CueBridgeArtifacts::load_for_project(&tmp.path().to_string_lossy(), "gap_proj");
    let expansions = artifacts.gap_expansions(
        &[("car".to_string(), 1.0)],
        None,
        &["car".to_string()],
        |cue| cue == "automobile",
        6,
    );

    assert_eq!(expansions.len(), 1);
    assert_eq!(expansions[0].cue, "automobile");
    assert_eq!(expansions[0].entry_id, "gap_car");
}
