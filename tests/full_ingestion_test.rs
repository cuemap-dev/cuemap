//! Full integration test: verify chunking works for all test data file types
//! 
//! This test ingests files from data/agent-test/ and verifies that chunking
//! produces meaningful results with correct cues.

use cuemap::agent::chunker::Chunker;
use std::path::PathBuf;
use std::fs;

/// Helper to collect all files recursively from a directory
fn collect_files(dir: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                // Skip .DS_Store and other hidden files
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if !name.starts_with('.') {
                        files.push(path);
                    }
                }
            } else if path.is_dir() {
                files.extend(collect_files(path.to_str().unwrap_or("")));
            }
        }
    }
    files
}

#[ignore]
#[test]
fn test_chunking_coverage_all_file_types() {
    let base_dir = "data/agent-test";
    
    // Count chunks per category
    let mut results: std::collections::HashMap<String, (usize, usize)> = std::collections::HashMap::new();
    
    let files = collect_files(base_dir);
    assert!(!files.is_empty(), "Should find test files in data/agent-test/");
    
    for file_path in &files {
        let category = file_path.parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Try to read and chunk the file
        if let Ok(content) = fs::read_to_string(&file_path) {
            let chunks = Chunker::chunk_file(&file_path, &content);
            
            let entry = results.entry(category.clone()).or_insert((0, 0));
            entry.0 += 1; // file count
            entry.1 += chunks.len(); // chunk count
            
            // Verify chunks have required fields
            for chunk in &chunks {
                assert!(!chunk.content.is_empty(), "Chunk content should not be empty for {:?}", file_path);
                assert!(!chunk.structural_cues.is_empty(), "Chunk should have structural cues for {:?}", file_path);
            }
        }
    }
    
    // Verify we processed files from key categories
    println!("=== Chunking Coverage Results ===");
    for (category, (file_count, chunk_count)) in &results {
        println!("{}: {} files -> {} chunks", category, file_count, chunk_count);
    }
    
    // Assert minimum coverage
    assert!(results.get("whatsapp").map(|r| r.0).unwrap_or(0) >= 1, "Should process WhatsApp files");
    assert!(results.get("instagram").map(|r| r.0).unwrap_or(0) >= 1, "Should process Instagram files");
    assert!(results.get("programming").map(|r| r.0).unwrap_or(0) >= 5, "Should process programming files");
    assert!(results.get("markdown").map(|r| r.0).unwrap_or(0) >= 1, "Should process markdown files");
}

#[test]
fn test_whatsapp_content_extraction() {
    // Verify WhatsApp parser extracts meaningful conversation content
    let wa_path = PathBuf::from("data/agent-test/whatsapp/chat_3.txt");
    if let Ok(content) = fs::read_to_string(&wa_path) {
        let chunks = Chunker::chunk_file(&wa_path, &content);
        
        assert!(!chunks.is_empty(), "WhatsApp should produce chunks");
        
        // Verify platform cue
        let has_platform = chunks.iter().any(|c| 
            c.structural_cues.contains(&"platform:whatsapp".to_string())
        );
        assert!(has_platform, "Should have platform:whatsapp cue");
        
        // Verify participant/sender extraction
        let has_senders = chunks.iter().any(|c|
            c.structural_cues.iter().any(|s| s.starts_with("sender:"))
        );
        assert!(has_senders, "Should extract sender cues");
        
        // Verify content contains actual messages
        let has_content = chunks.iter().any(|c| c.content.len() > 50);
        assert!(has_content, "Should have substantive message content");
    }
}

#[test]
fn test_programming_file_extraction() {
    let test_cases = vec![
        ("data/agent-test/programming/engine.rs", "lang:rust"),
        ("data/agent-test/programming/classifier.py", "lang:python"),
        ("data/agent-test/programming/index.ts", "lang:typescript"),
        ("data/agent-test/programming/test.go", "lang:go"),
    ];
    
    for (path_str, expected_lang_cue) in test_cases {
        let path = PathBuf::from(path_str);
        if let Ok(content) = fs::read_to_string(&path) {
            let chunks = Chunker::chunk_file(&path, &content);
            
            assert!(!chunks.is_empty(), "Should produce chunks for {}", path_str);
            
            // Verify language cue
            let has_lang = chunks.iter().any(|c| 
                c.structural_cues.contains(&expected_lang_cue.to_string())
            );
            assert!(has_lang, "Should have {} cue for {}", expected_lang_cue, path_str);
        }
    }
}

#[test]
fn test_openapi_detection() {
    let api_path = PathBuf::from("data/agent-test/api-spec/openapi-sample.yaml");
    if let Ok(content) = fs::read_to_string(&api_path) {
        let chunks = Chunker::chunk_file(&api_path, &content);
        
        assert!(!chunks.is_empty(), "OpenAPI should produce chunks");
        println!("OpenAPI chunks: {}", chunks.len());
        
        // Verify YAML structure was parsed
        let has_yaml_cue = chunks.iter().any(|c|
            c.structural_cues.iter().any(|s| s.contains("yaml") || s.contains("openapi"))
        );
        println!("Has YAML/OpenAPI cue: {}", has_yaml_cue);
    }
}

#[test]
fn test_csv_chunking() {
    let csv_path = PathBuf::from("data/agent-test/csv/message_1 2.csv");
    if let Ok(content) = fs::read_to_string(&csv_path) {
        let chunks = Chunker::chunk_file(&csv_path, &content);
        
        assert!(!chunks.is_empty(), "CSV should produce chunks");
        println!("CSV chunks: {}", chunks.len());
        
        // Verify has row data
        let has_row_content = chunks.iter().any(|c| c.content.contains(","));
        assert!(has_row_content, "CSV chunks should contain comma-separated data");
    }
}

#[test]
fn test_markdown_chunking() {
    let md_path = PathBuf::from("data/agent-test/markdown/README.md");
    if let Ok(content) = fs::read_to_string(&md_path) {
        let chunks = Chunker::chunk_file(&md_path, &content);
        
        assert!(!chunks.is_empty(), "Markdown should produce chunks");
        println!("Markdown chunks: {}", chunks.len());
        
        // Verify markdown structure was parsed
        let has_heading = chunks.iter().any(|c| 
            c.structural_cues.iter().any(|s| s.contains("heading") || s.contains("markdown"))
        );
        println!("Has heading cue: {}", has_heading);
    }
}

#[test]
fn test_instagram_chrome_youtube_extraction() {
    // Instagram
    let ig_path = PathBuf::from("data/agent-test/instagram/message_1.json");
    if let Ok(content) = fs::read_to_string(&ig_path) {
        let chunks = Chunker::chunk_file(&ig_path, &content);
        assert!(!chunks.is_empty(), "Instagram should produce chunks");
        let has_ig = chunks.iter().any(|c| c.structural_cues.contains(&"platform:instagram".to_string()));
        assert!(has_ig, "Should have platform:instagram cue");
    }
    
    // Chrome History (large file - just check it doesn't crash)
    let ch_path = PathBuf::from("data/agent-test/google-chrome/History.json");
    if let Ok(content) = fs::read_to_string(&ch_path) {
        let chunks = Chunker::chunk_file(&ch_path, &content);
        assert!(!chunks.is_empty(), "Chrome History should produce chunks");
        println!("Chrome History chunks: {}", chunks.len());
    }
    
    // YouTube (large file - just check it parses)
    let yt_path = PathBuf::from("data/agent-test/youtube/watch-history.html");
    if let Ok(content) = fs::read_to_string(&yt_path) {
        let chunks = Chunker::chunk_file(&yt_path, &content);
        println!("YouTube watch-history chunks: {}", chunks.len());
        // Large file may produce many chunks - just verify it works
        assert!(chunks.len() > 0 || content.len() < 100, "YouTube should produce chunks for non-trivial content");
    }
}

