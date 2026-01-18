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
            
            // Store only raw content - no metadata prefix
            // Context info is captured in structural_cues (path, context, category)
            self.job_queue.enqueue(Job::ExtractAndIngest {
                project_id: project_id.clone(),
                memory_id: memory_id.clone(),
                content: chunk.content.clone(),
                file_path: path_norm.clone(),
                structural_cues: chunk.structural_cues.clone(),
                category: chunk.category,
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

    /// Process content from a URL - fetches, chunks, and ingests
    pub async fn process_url(&mut self, url: &str, project_id: &str) -> Result<Vec<String>, String> {
        use crate::agent::chunker::Chunker;
        
        info!("Ingesting URL: {}", url);
        
        let chunks = Chunker::chunk_url(url).await?;
        let source = format!("url:{}", url);
        
        self.process_chunks(chunks, project_id, &source).await
    }

    /// Process raw content (text/json/yaml/etc) without a file path
    pub async fn process_content(&mut self, content: &str, filename: &str, project_id: &str) -> Result<Vec<String>, String> {
        use crate::agent::chunker::Chunker;
        
        info!("Ingesting content: {} ({} bytes)", filename, content.len());
        
        // Create a virtual path for the chunker to determine content type
        let virtual_path = PathBuf::from(filename);
        let chunks = Chunker::chunk_file(&virtual_path, content);
        let source = format!("api:{}", filename);
        
        self.process_chunks(chunks, project_id, &source).await
    }

    /// Shared chunk processing logic - returns memory IDs of ingested chunks
    async fn process_chunks(
        &mut self, 
        chunks: Vec<crate::agent::chunker::Chunk>, 
        project_id: &str, 
        source: &str
    ) -> Result<Vec<String>, String> {
        let mut memory_ids = Vec::new();
        
        // Track session for progress reporting
        let session = self.job_queue.session_manager.get_or_create(project_id);
        for _ in &chunks {
            session.expect_write();
        }
        
        for chunk in chunks.iter() {
            let mut chunk_hasher = Sha256::new();
            chunk_hasher.update(chunk.content.as_bytes());
            let chunk_hash = format!("{:x}", chunk_hasher.finalize());
            
            // Use source for ID consistency
            let memory_id = format!("{}:{}", source, chunk_hash);
            
            // ExtractAndIngest does the write - enqueue immediately  
            self.job_queue.enqueue(Job::ExtractAndIngest {
                project_id: project_id.to_string(),
                memory_id: memory_id.clone(),
                content: chunk.content.clone(),
                file_path: source.to_string(),
                structural_cues: chunk.structural_cues.clone(),
                category: chunk.category,
            }).await;
            
            // Buffer downstream jobs for phased processing
            self.job_queue.buffer(project_id, Job::ProposeCues {
                project_id: project_id.to_string(),
                memory_id: memory_id.clone(),
                content: chunk.content.clone(),
            }).await;
            
            self.job_queue.buffer(project_id, Job::TrainLexiconFromMemory {
                project_id: project_id.to_string(),
                memory_id: memory_id.clone(),
            }).await;
            
            self.job_queue.buffer(project_id, Job::UpdateGraph {
                project_id: project_id.to_string(),
                memory_id: memory_id.clone(),
            }).await;
            
            session.write_complete();
            
            memory_ids.push(memory_id);
        }
        
        info!("Enqueued {} chunks from {}", memory_ids.len(), source);
        Ok(memory_ids)
    }
}

