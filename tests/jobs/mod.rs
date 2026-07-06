use cuemap::api::{RecallGroundedRequest, RecallRequest};
use cuemap::config::{JobsConfig, TuningConfig};
use cuemap::jobs::*;
use cuemap::multi_tenant::MultiTenantEngine;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::time::{sleep, Duration};

#[test]
fn test_default_jobs_are_lean_core() {
    let config = JobsConfig::default();

    assert!(config.background_processing);
    assert_eq!(config.market_heatmap_interval_seconds, 60);
}

#[test]
fn test_recall_requests_keep_extra_passes_off_by_default() {
    let recall: RecallRequest = serde_json::from_value(serde_json::json!({
        "query_text": "favorite dessert"
    }))
    .unwrap();
    assert_eq!(
        recall.ordered_reconstruction,
        cuemap::api::OrderedReconstructionMode::Off
    );

    let grounded: RecallGroundedRequest = serde_json::from_value(serde_json::json!({
        "query_text": "favorite dessert"
    }))
    .unwrap();
    assert!(grounded.auto_reinforce);
}

#[tokio::test]
async fn extract_and_ingest_preserves_metadata_for_ordered_recall() {
    let dir = tempdir().unwrap();
    let snapshots_dir = dir.path().join("snapshots");
    std::fs::create_dir_all(&snapshots_dir).unwrap();
    let engine = Arc::new(MultiTenantEngine::with_snapshots_dir(
        &snapshots_dir,
        TuningConfig::default(),
    ));
    let project_id = "metadata_job_test".to_string();
    let ctx = engine.get_or_create_project(project_id.clone()).unwrap();
    let queue = JobQueue::new(engine.clone(), None, false);
    let session = queue.session_manager.get_or_create(&project_id);
    session.expect_write();

    let mut metadata = HashMap::new();
    metadata.insert("source_session_id".to_string(), json!("thread-job"));
    metadata.insert("source_turn_index".to_string(), json!(7));
    metadata.insert("source_role".to_string(), json!("assistant"));

    queue
        .enqueue(Job::ExtractAndIngest {
            project_id: project_id.clone(),
            source_key: "thread-job:7".to_string(),
            content: "assistant: Then we optimized translation latency.".to_string(),
            file_path: "thread-job".to_string(),
            structural_cues: vec!["source_type:chat_message".to_string()],
            metadata: Some(metadata),
            category: cuemap::agent::chunker::ChunkCategory::Prose,
        })
        .await;

    for _ in 0..250 {
        if let Some(memory_id) = ctx.main.memory_id_for_source_key("thread-job:7") {
            let memory = ctx.main.get_memory(memory_id).unwrap();
            assert_eq!(
                memory
                    .metadata
                    .get("source_session_id")
                    .and_then(|value| value.as_str()),
                Some("thread-job")
            );
            assert_eq!(ctx.main.ordered_entries_for_session("thread-job", 10).len(), 1);
            return;
        }
        sleep(Duration::from_millis(20)).await;
    }

    panic!("metadata-preserving ExtractAndIngest job did not complete");
}
