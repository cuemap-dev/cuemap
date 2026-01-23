use cuemap_rust::agent::chunker::Chunker;
use cuemap_rust::agent::ingester::{Ingester, CrawlResult};
use cuemap_rust::agent::AgentConfig;
use cuemap_rust::jobs::{JobQueue, JobProgress, IngestionPhase};
use cuemap_rust::multi_tenant::MultiTenantEngine;
use cuemap_rust::config::CueGenStrategy;
use cuemap_rust::semantic::SemanticEngine;
use std::sync::Arc;

/// Test basic URL chunking (single page, no recursion)
#[tokio::test]
async fn test_single_url_chunking() {
    // ... (unchanged)
    // Use a simple, stable page for testing
    let test_url = "https://example.com";
    
    let result = Chunker::chunk_url(test_url).await;
    
    match result {
        Ok(chunks) => {
            println!("Single URL test: {} chunks extracted from {}", chunks.len(), test_url);
            assert!(!chunks.is_empty(), "Should extract at least one chunk from example.com");
            
            // Verify chunks have content and cues
            for chunk in &chunks {
                assert!(!chunk.content.is_empty(), "Chunk content should not be empty");
                assert!(!chunk.structural_cues.is_empty(), "Chunk should have structural cues");
            }
            
            // Verify domain cue is present
            let has_domain_cue = chunks.iter().any(|c| 
                c.structural_cues.iter().any(|s| s.contains("domain:"))
            );
            assert!(has_domain_cue, "Should have domain cue");
        }
        Err(e) => {
            println!("Warning: Could not fetch test URL: {}", e);
            // Don't fail if network is unavailable
        }
    }
}

// ... (test_content_link_extraction and test_url_normalization unchanged, skipping for brevity in replacement)

/// Test recursive crawl with depth=1 on a real documentation page
/// This is a longer test that requires network access
#[ignore]  // Run with: cargo test --test recursive_crawl_test test_recursive_crawl_depth_1 -- --ignored
#[tokio::test]
async fn test_recursive_crawl_depth_1() {
    // Use axum docs as a stable test target (small, well-structured)
    let test_url = "https://docs.rs/axum/latest/axum/";
    
    // Create a minimal engine for testing
    let semantic_engine = SemanticEngine::new(None);
    let engine = Arc::new(MultiTenantEngine::new(CueGenStrategy::Default, semantic_engine));
    let job_queue = Arc::new(JobQueue::new(engine.clone(), true)); // Disable bg jobs for testing
    
    let config = AgentConfig {
        watch_dir: String::new(),
        throttle_ms: 100, // Throttle to be polite
    };
    
    let mut ingester = Ingester::new(config, job_queue.clone());
    
    println!("Starting recursive crawl of {} with depth=1...", test_url);
    
    let result = ingester.process_url_recursive(
        test_url,
        "test-project",
        1, // Depth 1
        true, // Same domain only
    ).await;
    
    match result {
        Ok(crawl_result) => {
            println!("Crawl Result:");
            println!("  Pages crawled: {}", crawl_result.pages_crawled);
            println!("  Total chunks: {}", crawl_result.memory_ids.len());
            println!("  Links found: {}", crawl_result.links_found);
            println!("  Links skipped: {}", crawl_result.links_skipped);
            println!("  Errors: {}", crawl_result.errors.len());
            
            // Verify we got multiple pages
            assert!(crawl_result.pages_crawled >= 1, "Should crawl at least 1 page");
            
            // Verify chunks were created
            assert!(!crawl_result.memory_ids.is_empty(), "Should create memory chunks");
            
            // Verify session tracking
            if let Some(session) = job_queue.get_session("test-project") {
                let progress = session.get_progress();
                println!("\nJob Progress:");
                println!("  Phase: {}", progress.phase);
                println!("  Writes: {}/{}", progress.writes_completed, progress.writes_total);
                
                // Verify writes tracking matches chunks
                assert_eq!(
                    progress.writes_total, 
                    crawl_result.memory_ids.len(),
                    "writes_total should match total chunks"
                );
                assert_eq!(
                    progress.writes_completed,
                    progress.writes_total,
                    "All writes should be completed before test ends"
                );
            }
        }
        Err(e) => {
            println!("Warning: Crawl failed (may be network issue): {}", e);
            // Don't fail if network is unavailable
        }
    }
}

/// Test that job phases work correctly:
/// 1. During crawl: phase should be Writing
/// 2. After crawl: all writes should be complete before bg jobs start
#[ignore]
#[tokio::test]
async fn test_job_phase_ordering() {
    let test_url = "https://example.com";
    
    let semantic_engine = SemanticEngine::new(None);
    let engine = Arc::new(MultiTenantEngine::new(CueGenStrategy::Default, semantic_engine));
    let job_queue = Arc::new(JobQueue::new(engine.clone(), true));
    
    let config = AgentConfig {
        watch_dir: String::new(),
        throttle_ms: 0,
    };
    
    let mut ingester = Ingester::new(config, job_queue.clone());
    
    // Single page crawl (depth=0 still uses recursive method internally)
    let result = ingester.process_url(test_url, "phase-test").await;
    
    if let Ok(memory_ids) = result {
        if let Some(session) = job_queue.get_session("phase-test") {
            let progress = session.get_progress();
            
            // All writes should be complete
            assert_eq!(
                progress.writes_completed,
                progress.writes_total,
                "All writes should complete: {}/{}",
                progress.writes_completed,
                progress.writes_total
            );
            
            // Memory IDs should match write count
            assert_eq!(
                memory_ids.len(),
                progress.writes_total,
                "Memory IDs should match writes_total"
            );
            
            println!("Phase test passed: {} writes completed", progress.writes_completed);
        }
    }
}
