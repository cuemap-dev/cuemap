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

    /// Process URL with recursive crawling up to specified depth
    /// Uses BFS traversal, extracts links only from main content (not nav/footer)
    /// 
    /// Phase 1: Crawl all pages and collect chunks (no writes yet)
    /// Phase 2: Write all chunks as memories
    /// Phase 3: Buffer bg jobs (auto-flush will process them after writes complete)
    pub async fn process_url_recursive(
        &mut self,
        start_url: &str,
        project_id: &str,
        max_depth: u8,
        same_domain_only: bool,
    ) -> Result<CrawlResult, String> {
        use std::collections::{HashSet, VecDeque};
        use crate::agent::chunker::{Chunker, Chunk};
        use scraper::Html;

        let base_url = url::Url::parse(start_url)
            .map_err(|e| format!("Invalid start URL: {}", e))?;
        let base_domain = base_url.host_str().unwrap_or("").to_string();

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, u8)> = VecDeque::new(); // (url, depth)
        let mut result = CrawlResult {
            memory_ids: Vec::new(),
            pages_crawled: 0,
            links_found: 0,
            links_skipped: 0,
            errors: Vec::new(),
        };

        // Collect all chunks across all pages before writing
        let mut all_chunks: Vec<(String, Chunk)> = Vec::new(); // (source, chunk)

        // Start with the initial URL at depth 0
        queue.push_back((start_url.to_string(), 0));
        visited.insert(Self::normalize_url(start_url));

        // HTTP client for fetching pages
        let client = reqwest::Client::builder()
            .user_agent("CueMap/0.6 (https://cuemap.dev; bot)")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        // ========== PHASE 1: Crawl and collect chunks ==========
        info!("Crawl Phase 1: Fetching pages and collecting chunks...");
        
        while let Some((current_url, depth)) = queue.pop_front() {
            info!("Crawling [depth={}]: {}", depth, current_url);

            // Fetch the page
            let html_content = match client.get(&current_url).send().await {
                Ok(response) => match response.text().await {
                    Ok(text) => text,
                    Err(e) => {
                        result.errors.push((current_url.clone(), format!("Read error: {}", e)));
                        continue;
                    }
                },
                Err(e) => {
                    result.errors.push((current_url.clone(), format!("Fetch error: {}", e)));
                    continue;
                }
            };

            // Parse and chunk the content
            match Chunker::chunk_url(&current_url).await {
                Ok(chunks) => {
                    let source = format!("url:{}", current_url);
                    for chunk in chunks {
                        all_chunks.push((source.clone(), chunk));
                    }
                    result.pages_crawled += 1;
                }
                Err(e) => {
                    result.errors.push((current_url.clone(), format!("Chunk error: {}", e)));
                    continue;
                }
            }

            // If we haven't reached max depth, extract and queue links
            if depth < max_depth {
                let parsed_current = match url::Url::parse(&current_url) {
                    Ok(u) => u,
                    Err(_) => continue,
                };

                let document = Html::parse_document(&html_content);
                let links = Chunker::extract_content_links(&document, &parsed_current);
                result.links_found += links.len();

                for link in links {
                    let normalized = Self::normalize_url(&link);

                    // Skip if already visited
                    if visited.contains(&normalized) {
                        result.links_skipped += 1;
                        continue;
                    }

                    // Domain check if same_domain_only is enabled
                    if same_domain_only {
                        if let Ok(link_url) = url::Url::parse(&link) {
                            let link_domain = link_url.host_str().unwrap_or("");
                            if link_domain != base_domain {
                                result.links_skipped += 1;
                                continue;
                            }
                        } else {
                            result.links_skipped += 1;
                            continue;
                        }
                    }

                    // Skip non-HTML resources
                    if Self::is_non_html_resource(&link) {
                        result.links_skipped += 1;
                        continue;
                    }

                    visited.insert(normalized);
                    queue.push_back((link, depth + 1));
                }
            }
        }

        info!("Crawl Phase 1 complete: {} pages, {} total chunks collected", 
              result.pages_crawled, all_chunks.len());

        // ========== PHASE 2: Write all chunks as memories ==========
        info!("Crawl Phase 2: Writing {} chunks as memories...", all_chunks.len());
        
        // Set up session tracking for the entire batch
        let session = self.job_queue.session_manager.get_or_create(project_id);
        for _ in &all_chunks {
            session.expect_write();
        }
        
        // Write all chunks
        for (source, chunk) in &all_chunks {
            let mut chunk_hasher = Sha256::new();
            chunk_hasher.update(chunk.content.as_bytes());
            let chunk_hash = format!("{:x}", chunk_hasher.finalize());
            let memory_id = format!("{}:{}", source, chunk_hash);
            
            // Write immediately
            self.job_queue.enqueue(Job::ExtractAndIngest {
                project_id: project_id.to_string(),
                memory_id: memory_id.clone(),
                content: chunk.content.clone(),
                file_path: source.clone(),
                structural_cues: chunk.structural_cues.clone(),
                category: chunk.category,
            }).await;
            
            session.write_complete();
            result.memory_ids.push(memory_id);
        }
        
        info!("Crawl Phase 2 complete: {} memories written", result.memory_ids.len());

        // ========== PHASE 3: Buffer background jobs ==========
        info!("Crawl Phase 3: Buffering {} background jobs...", all_chunks.len() * 3);
        
        for (source, chunk) in all_chunks {
            let mut chunk_hasher = Sha256::new();
            chunk_hasher.update(chunk.content.as_bytes());
            let chunk_hash = format!("{:x}", chunk_hasher.finalize());
            let memory_id = format!("{}:{}", source, chunk_hash);
            
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
        }

        info!(
            "Crawl complete: {} pages, {} chunks, {} links skipped, {} errors",
            result.pages_crawled,
            result.memory_ids.len(),
            result.links_skipped,
            result.errors.len()
        );

        Ok(result)
    }

    /// Normalize URL for deduplication (remove fragments, trailing slashes, etc.)
    fn normalize_url(url: &str) -> String {
        if let Ok(mut parsed) = url::Url::parse(url) {
            parsed.set_fragment(None);
            let mut s = parsed.to_string();
            // Remove trailing slash for consistency
            if s.ends_with('/') && s.len() > 1 {
                s.pop();
            }
            s.to_lowercase()
        } else {
            url.to_lowercase()
        }
    }

    /// Check if URL points to a non-HTML resource (pdf, image, etc.)
    fn is_non_html_resource(url: &str) -> bool {
        let skip_extensions = [
            ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
            ".mp3", ".mp4", ".wav", ".avi", ".mov",
            ".zip", ".tar", ".gz", ".rar",
            ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".css", ".js", ".json", ".xml", ".rss", ".atom",
        ];
        let lower = url.to_lowercase();
        skip_extensions.iter().any(|ext| lower.contains(ext))
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

/// Result of a recursive URL crawl
#[derive(Debug, Clone)]
pub struct CrawlResult {
    pub memory_ids: Vec<String>,
    pub pages_crawled: usize,
    pub links_found: usize,
    pub links_skipped: usize,
    pub errors: Vec<(String, String)>, // (url, error message)
}

/// Progress update during crawling
#[derive(Debug, Clone)]
pub struct CrawlProgress {
    pub current_url: String,
    pub depth_level: u8,
    pub pages_done: usize,
    pub pages_queued: usize,
    pub total_chunks: usize,
    pub links_found: usize,
    pub links_skipped: usize,
}
