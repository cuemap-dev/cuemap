use cuemap_rust::engine::CueMapEngine;
use cuemap_rust::persistence::PersistenceManager;
use std::path::{Path};
use std::env;
use std::fs;
use walkdir::WalkDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: create_fixture <source_dir> <output_snapshot_path>");
        std::process::exit(1);
    }

    let source_dir = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);

    println!("Ingesting from {:?}...", source_dir);
    
    let engine = CueMapEngine::new();
    
    // Simple ingestion: walk directory, read files, add as memories
    for entry in WalkDir::new(source_dir).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let path = entry.path();
            // Skip hidden files or target dir
            if path.to_string_lossy().contains("/.") || path.to_string_lossy().contains("target/") {
                continue;
            }
            
            if let Ok(content) = fs::read_to_string(path) {
                // Naive chunking: file as one memory
                let rel_path = path.strip_prefix(source_dir).unwrap_or(path).to_string_lossy().to_string();
                let file_name = path.file_name().unwrap().to_string_lossy().to_string();
                
                // Extract simple cues (tokens)
                let cues: Vec<String> = file_name
                    .replace(".", " ")
                    .replace("_", " ")
                    .split_whitespace()
                    .map(|s| s.to_lowercase())
                    .collect();
                
                println!("Adding memory: {}", rel_path);
                // Use relative path as the deterministic ID
                engine.upsert_memory_with_id(rel_path, content, cues, None, false);
            }
        }
    }

    println!("Saving snapshot to {:?}...", output_path);
    
    // Use PersistenceManager static method if possible, or instance
    // PersistenceManager::save_to_path(&engine, output_path)?;
    // Wait, save_to_path is struct method or static? Checked file: it is `pub fn save_to_path`.
    PersistenceManager::save_to_path(&engine, output_path)?;

    println!("Done. Memories: {}", engine.get_memories().len());
    Ok(())
}
