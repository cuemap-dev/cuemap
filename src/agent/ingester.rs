use crate::agent::chunker::Chunker;
use crate::agent::AgentConfig;
use crate::jobs::{Job, JobQueue};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{info, warn, debug};
use ignore::WalkBuilder;

pub struct Ingester {
    config: AgentConfig,
    job_queue: Arc<JobQueue>,
    file_hashes: HashMap<String, String>, // path -> sha256
}

impl Ingester {
    pub fn new(config: AgentConfig, job_queue: Arc<JobQueue>) -> Self {
        Self {
            config,
            job_queue,
            file_hashes: HashMap::new(),
        }
    }

    pub async fn scan_all(&mut self) -> Result<(), String> {
        info!("Starting full scan of {}", self.config.watch_dir);
        
        let path_str = self.config.watch_dir.clone();
        
        // Use ignore crate to respect .gitignore
        let walker = WalkBuilder::new(&path_str)
            .hidden(true)
            .git_ignore(true)
            .build();

        for result in walker {
            match result {
                Ok(entry) => {
                    let path = entry.path();
                    if path.is_file() {
                        if let Err(_e) = self.process_file_path(path.to_path_buf()).await {
                            // warn!("Failed to process {:?}: {}", path, e);
                        }
                        // Throttle
                        if self.config.throttle_ms > 0 {
                            sleep(Duration::from_millis(self.config.throttle_ms)).await;
                        }
                    }
                }
                Err(err) => warn!("Walk error: {}", err),
            }
        }
        
        info!("Scan complete. Tracking {} files.", self.file_hashes.len());
        Ok(())
    }

    pub async fn process_file_path(&mut self, path: PathBuf) -> Result<(), String> {
        let path_str = path.to_string_lossy().to_string();
        // Standardize casing for case-insensitive filesystems (MacOS/Windows)
        let path_norm = path_str.to_lowercase();
        
        // 1. Read file as bytes first (works for both text and binary)
        let bytes = fs::read(&path)
            .map_err(|e| format!("Read error: {}", e))?;
            
        // 2. Hash check
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());
        
        if let Some(old_hash) = self.file_hashes.get(&path_norm) {
            if old_hash == &hash {
                debug!("Skipping unchanged file: {}", path_norm);
                return Ok(());
            }
        }
        
        // Update hash
        self.file_hashes.insert(path_norm.clone(), hash.clone());
        info!("Ingesting: {}", path_str);
        
        // 3. Chunk
        // Try to convert to UTF-8 for text-based chunking, otherwise pass empty string
        // The chunker will use the path for binary formats (PDF, Office)
        let content_str = String::from_utf8(bytes).ok();
        let chunks = Chunker::chunk_file(&path, content_str.as_deref().unwrap_or(""));
        
        // 4. Send to Job Queue
        let project_id = "main".to_string();
        let mut valid_memory_ids = Vec::new();
        
        for chunk in chunks.iter() {
            let mut chunk_hasher = Sha256::new();
            chunk_hasher.update(chunk.content.as_bytes());
            let chunk_hash = format!("{:x}", chunk_hasher.finalize());
            // Use normalized path for ID consistency
            let memory_id = format!("file:{}:{}", path_norm, chunk_hash); 
            
            let full_content = format!(
                "File: {}\nContext: {}\nLines: {}-{}\n\n{}", 
                path_str, chunk.context, chunk.start_line, chunk.end_line, chunk.content
            );
            
            self.job_queue.enqueue(Job::ExtractAndIngest {
                project_id: project_id.clone(),
                memory_id: memory_id.clone(),
                content: full_content,
                file_path: path_norm.clone(),
            }).await;
            
            valid_memory_ids.push(memory_id);
        }
        
        // 5. Verification: Prune stale memories
        self.job_queue.enqueue(Job::VerifyFile {
            project_id,
            file_path: path_norm,
            valid_memory_ids,
        }).await;

        Ok(())
    }

    pub async fn delete_file_path(&mut self, path: PathBuf) -> Result<(), String> {
        let path_str = path.to_string_lossy().to_string();
        let path_norm = path_str.to_lowercase();
        info!("Processing deletion: {}", path_str);

        // Remove from tracking
        self.file_hashes.remove(&path_norm);

        // Enqueue Verification with EMPTY valid_ids to prune all associated memories
        self.job_queue.enqueue(Job::VerifyFile {
            project_id: "main".to_string(),
            file_path: path_norm,
            valid_memory_ids: Vec::new(),
        }).await;

        Ok(())
    }
}
