use cuemap_rust::jobs::{Job, JobQueue, SingleTenantProvider};
use cuemap_rust::projects::ProjectContext;
use cuemap_rust::llm::{LlmConfig, setup::ensure_ollama_running};
use cuemap_rust::normalization::NormalizationConfig;
use cuemap_rust::taxonomy::Taxonomy;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

async fn setup_live_system() -> (cuemap_rust::engine::CueMapEngine, Arc<JobQueue>, Arc<ProjectContext>) {
    // 1. Configure LLM (Local Ollama)
    let config = LlmConfig {
        provider: "ollama".to_string(),
        model: "mistral".to_string(), 
        api_key: None,
        ollama_url: "http://localhost:11434".to_string(),
    };

    // 2. Ensure Ollama is running
    let llm_ready = ensure_ollama_running(&config).await;
    if !llm_ready {
        eprintln!("WARNING: Ollama is not reachable. Live tests might fail or be skipped.");
        // We panic here because the user explicitly asked for real evals
        panic!("Ollama unreachable. Please run 'ollama serve' or check connection.");
    }

    // 3. Setup Project Context
    let normalization = NormalizationConfig::default();
    let taxonomy = Taxonomy::default();

    // ProjectContext::new() creates internal engines (main, lexicon, aliases)
    let project = Arc::new(ProjectContext::new(normalization, taxonomy, cuemap_rust::config::CueGenStrategy::default(), cuemap_rust::semantic::SemanticEngine::new(None)));
    
    // 4. Setup Job Queue
    let provider = Arc::new(SingleTenantProvider {
        project: project.clone(),
    });
    let job_queue = Arc::new(JobQueue::new(provider));

    (project.main.clone(), job_queue, project)
}

#[tokio::test]
async fn test_live_async_ingestion_stability() {
    let (engine, queue, _) = setup_live_system().await;

    // Rapidly ingest memories
    for i in 0..2 { // Reduced to 2 for speed
        let content = format!("Database connection failed for user {}. The SQL query timed out after 3000ms. Severity: High.", i);
        let id = format!("mem_{}", i);
        
        // Add minimal memory directly
        engine.upsert_memory_with_id(
            id.clone(), 
            content.clone(), 
            vec!["test:live".to_string()], 
            None, 
            false
        );

        // Enqueue LLM job
        queue.enqueue(Job::ProposeCues {
            project_id: "default".to_string(), // SingleTenantProvider ignores project_id, but we pass "default"
            memory_id: id,
            content,
        }).await;
    }

    // Wait for processing
    let mut success = false;
    for _ in 0..60 { // Poll every second for 60s
        sleep(Duration::from_secs(1)).await;
        
        let mem = engine.get_memory("mem_0").unwrap();
        // Check if cues were added. "test:live" is 1. We expect more.
        if mem.cues.len() > 1 {
            success = true;
            println!("Memory mem_0 enriched! Cues ({}) : {:?}", mem.cues.len(), mem.cues);
            break;
        }
    }

    assert!(success, "Background LLM job failed to enrich memory within timeout");
}

#[tokio::test]
async fn test_live_llm_integration() {
    let (engine, queue, _) = setup_live_system().await;

    // Incident report text
    let content = "The payment gateway is returning 500 errors during checkout process. Users are unable to complete purchases.";
    let id = "incident_real_llm";
    
    engine.upsert_memory_with_id(id.to_string(), content.to_string(), vec!["type:incident".to_string()], None, false);

    queue.enqueue(Job::ProposeCues {
        project_id: "default".to_string(),
        memory_id: id.to_string(),
        content: content.to_string(),
    }).await;

    // Verify semantic cues
    let mut cues_found = false;
    for _ in 0..60 {
        sleep(Duration::from_secs(1)).await;
        if let Some(mem) = engine.get_memory(id) {
            // Check for domain specific cues likely to be extracted by Mistral
            // e.g., "topic:payments", "service:checkout", "status:error"
            let has_payment = mem.cues.iter().any(|c| c.contains("payment") || c.contains("checkout"));
            let has_error = mem.cues.iter().any(|c| c.contains("error") || c.contains("fail"));
            
            if has_payment || has_error {
                cues_found = true;
                println!("LLM Integration Success. Extracted cues: {:?}", mem.cues);
                break;
            }
        }
    }

    assert!(cues_found, "LLM failed to extract relevant semantic cues from incident text");
}

#[tokio::test]
async fn test_live_pattern_completion() {
    // E15: Pattern Completion (Live Co-occurrence)
    let (engine, _, _) = setup_live_system().await;

    // 1. Establish strong link between "apple" and "banana"
    // We ingest multiple memories containing both to boost co-occurrence count
    for i in 0..5 {
        engine.upsert_memory_with_id(
            format!("link_{}", i),
            "Fruit salad mix.".to_string(),
            vec!["apple".to_string(), "banana".to_string()],
            None,
            false // no reinforce, just initial add updates matrix
        );
    }
    
    // 2. Ingest a target memory that ONLY has "banana"
    let target_id = "target_banana_only";
    engine.upsert_memory_with_id(
        target_id.to_string(),
        "I am a yellow curved fruit.".to_string(),
        vec!["banana".to_string()],
        None,
        false
    );

    // 3. Query for "apple"
    // If pattern completion works, "apple" should trigger "banana" via co-occurrence,
    // and thus retrieve the "banana"-only memory.
    let results = engine.recall(vec!["apple".to_string()], 10, false);
    
    // 4. Check if target is in results
    let found = results.iter().any(|r| r.memory_id == target_id);
    
    // Debug print
    if !found {
        println!("Pattern Completion Failed. Results:");
        for r in &results {
            println!(" - {} (Score: {:.2})", r.memory_id, r.score);
        }
    } else {
        println!("Pattern Completion Success: 'apple' retrieved 'banana'-only memory.");
    }

    assert!(found, "Query for 'apple' failed to retrieve 'banana' memory despite strong co-occurrence.");
}
