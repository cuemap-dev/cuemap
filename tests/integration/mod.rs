use cuemap_rust::engine::CueMapEngine;
use cuemap_rust::projects::ProjectContext;
use cuemap_rust::normalization::NormalizationConfig;
use cuemap_rust::taxonomy::Taxonomy;
use cuemap_rust::jobs::{Job, JobQueue, SingleTenantProvider};
use std::sync::Arc;
use std::time::Duration;
use serde_json::Value;

#[tokio::test]
async fn test_lexicon_resolution() {
    let ctx = Arc::new(ProjectContext::new(
        NormalizationConfig::default(),
        Taxonomy::default(),
    ));
    
    // Create a job queue for this project
    let provider = Arc::new(SingleTenantProvider { project: ctx.clone() });
    let job_queue = JobQueue::new(provider);

    // 1. Add memory with cues
    let content = "The payments service is experiencing high latency.".to_string();
    let cues = vec!["service:payments".to_string(), "status:slow".to_string()];
    let memory_id = ctx.main.add_memory(content.clone(), cues.clone(), None);

    // 2. Manually trigger Lexicon training job (usually triggered by API)
    job_queue.enqueue(Job::TrainLexiconFromMemory {
        project_id: "default".to_string(),
        memory_id: memory_id.clone(),
    }).await;

    // 3. Wait for job processing (lexicon training is fast)
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 4. Resolve cues from natural language text
    // "payments" should map to "service:payments" if trained correctly
    let resolved = ctx.resolve_cues_from_text("payments latency slow");
    
    assert!(resolved.contains(&"service:payments".to_string()));
    assert!(resolved.contains(&"status:slow".to_string()));
}

#[tokio::test]
async fn test_alias_expansion_and_weighting() {
    let ctx = Arc::new(ProjectContext::new(
        NormalizationConfig::default(),
        Taxonomy::default(),
    ));

    // 1. Setup an alias: "pay" -> "service:payments" (weight 0.85)
    let alias_content = serde_json::json!({
        "from": "pay",
        "to": "service:payments",
        "downweight": 0.85,
        "status": "active"
    }).to_string();
    
    let alias_cues = vec![
        "type:alias".to_string(),
        "from:pay".to_string(),
        "status:active".to_string(),
    ];
    
    ctx.aliases.add_memory(alias_content, alias_cues, None);

    // 2. Expand cues
    let query_cues = vec!["pay".to_string()];
    let expanded = ctx.expand_query_cues(query_cues);

    // Verify original and alias are present
    assert_eq!(expanded.len(), 2);
    assert!(expanded.iter().any(|(c, w)| c == "pay" && *w == 1.0));
    assert!(expanded.iter().any(|(c, w)| c == "service:payments" && *w == 0.85));

    // 3. Verify weighting in recall
    // Add two memories: one for "pay", one for "service:payments"
    let id_exact = ctx.main.add_memory("Direct pay".to_string(), vec!["pay".to_string()], None);
    let id_aliased = ctx.main.add_memory("Payments service".to_string(), vec!["service:payments".to_string()], None);

    let results = ctx.main.recall_weighted(expanded, 10, false, None, true);

    assert_eq!(results.len(), 2);
    // Exact match (weight 1.0) should be first
    assert_eq!(results[0].memory_id, id_exact);
    assert_eq!(results[1].memory_id, id_aliased);
    
    // Check explanation
    if let Some(explain) = &results[1].explain {
        assert_eq!(explain["intersection_weighted"].as_f64().unwrap(), 0.85);
    }
}

#[tokio::test]
async fn test_explain_output_structure() {
    let engine = CueMapEngine::new();
    
    engine.add_memory("test".to_string(), vec!["a".to_string()], None);
    
    let results = engine.recall_weighted(vec![("a".to_string(), 1.0)], 10, false, None, true);
    
    assert!(!results.is_empty());
    let explain = results[0].explain.as_ref().expect("Explain should be present");
    
    assert!(explain.get("intersection_score").is_some());
    assert!(explain.get("recency_component").is_some());
    assert!(explain.get("weights").is_some());
}

#[tokio::test]
async fn test_alias_proposal_job() {
    let ctx = Arc::new(ProjectContext::new(
        NormalizationConfig::default(),
        Taxonomy::default(),
    ));
    
    let provider = Arc::new(SingleTenantProvider { project: ctx.clone() });
    let job_queue = JobQueue::new(provider);

    // Create a scenario for alias discovery:
    // "prod" and "production" share 100% of memories
    for i in 0..25 {
        ctx.main.add_memory(format!("Mem {}", i), vec!["prod".to_string(), "production".to_string()], None);
    }

    // Trigger alias proposal job
    job_queue.enqueue(Job::ProposeAliases {
        project_id: "default".to_string(),
    }).await;

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Check aliases engine for proposed alias
    let results = ctx.aliases.recall(vec!["type:alias".to_string()], 10, false);
    assert!(!results.is_empty(), "Should have proposed at least one alias");
    
    let content: Value = serde_json::from_str(&results[0].content).unwrap();
    assert_eq!(content["status"], "proposed");
    assert!(
        (content["from"] == "prod" && content["to"] == "production") ||
        (content["from"] == "production" && content["to"] == "prod")
    );
}
