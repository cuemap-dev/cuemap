use crate::agent::chunker::Chunker;
use crate::agent::AgentConfig;
use crate::jobs::{Job, JobQueue};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{info, warn, debug};
use ignore::WalkBuilder;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use serde::{Deserialize, Serialize};

pub struct Ingester {
    config: AgentConfig,
    job_queue: Arc<JobQueue>,
    file_hashes: HashMap<String, String>, // path -> sha256
    gitignore: Option<Gitignore>,
    memory_hashes: HashMap<String, String>,    // memory_id -> content_hash
    path_to_memories: HashMap<String, HashSet<String>>, // path -> set of current memory_ids
}

#[derive(Serialize, Deserialize, Default)]
struct IngesterState {
    file_hashes: HashMap<String, String>,
    memory_hashes: HashMap<String, String>,
    path_to_memories: HashMap<String, HashSet<String>>,
}

impl Ingester {
    pub fn new(config: AgentConfig, job_queue: Arc<JobQueue>) -> Self {
        // Canonicalize watch_dir to ensure absolute path matching works across the engine
        let watch_path = fs::canonicalize(&config.watch_dir)
            .unwrap_or_else(|_| PathBuf::from(&config.watch_dir));
        debug!("Agent initializing with watch root: {:?}", watch_path);

        // Prepare gitignore
        let mut gitignore = None;
        let mut builder = GitignoreBuilder::new(&watch_path);
        
        // Search for .gitignore in watch_dir AND its parents up to the filesystem root
        // This is important for monorepos where the root .gitignore is in a parent directory.
        let mut current = Some(watch_path.as_path());
        let mut found_any = false;
        while let Some(p) = current {
            let p_gi = p.join(".gitignore");
            if p_gi.exists() {
                 debug!("Loading .gitignore from {:?}", p_gi);
                 if let Some(err) = builder.add(&p_gi) {
                    warn!("Error loading .gitignore at {:?}: {}", p_gi, err);
                } else {
                    found_any = true;
                }
            }
            current = p.parent();
        }

        if found_any {
            match builder.build() {
                Ok(gi) => gitignore = Some(gi),
                Err(e) => warn!("Failed to build gitignore: {}", e),
            }
        } else {
            debug!("No .gitignore files found in {:?} or its parents", watch_path);
        }

        let mut config = config;
        config.watch_dir = watch_path.to_string_lossy().to_string();

        Self {
            config,
            job_queue,
            file_hashes: HashMap::new(),
            gitignore,
            memory_hashes: HashMap::new(),
            path_to_memories: HashMap::new(),
        }
    }

    pub fn load_state(&mut self, state_path: &std::path::Path) -> Result<(), String> {
        if !state_path.exists() {
            return Ok(());
        }

        let content = fs::read_to_string(state_path)
            .map_err(|e| format!("Failed to read agent state: {}", e))?;
        
        let state: IngesterState = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse agent state: {}", e))?;

        self.file_hashes = state.file_hashes;
        self.memory_hashes = state.memory_hashes;
        self.path_to_memories = state.path_to_memories;

        debug!("Loaded agent state: {} files tracked", self.file_hashes.len());
        Ok(())
    }

    pub fn save_state(&self, state_path: &std::path::Path) -> Result<(), String> {
        let state = IngesterState {
            file_hashes: self.file_hashes.clone(),
            memory_hashes: self.memory_hashes.clone(),
            path_to_memories: self.path_to_memories.clone(),
        };

        let content = serde_json::to_string_pretty(&state)
            .map_err(|e| format!("Failed to serialize agent state: {}", e))?;

        fs::write(state_path, content)
            .map_err(|e| format!("Failed to write agent state: {}", e))?;

        debug!("Saved agent state: {} files tracked", self.file_hashes.len());
        Ok(())
    }

    pub async fn scan_all(&mut self) -> Result<(), String> {
        debug!("Starting full scan of {}", self.config.watch_dir);
        
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
        
        debug!("Scan complete. Tracking {} files.", self.file_hashes.len());
        Ok(())
    }

    pub async fn process_file_path(&mut self, path: PathBuf) -> Result<(), String> {
        let path = fs::canonicalize(&path)
            .map_err(|e| format!("Failed to canonicalize path {:?}: {}", path, e))?;
        let path_str = path.to_string_lossy().to_string();
        
        // 0. Ignore state file
        if let Some(ref state_path) = self.config.state_file {
            if let Ok(abs_path) = std::fs::canonicalize(&path) {
                if let Ok(abs_state) = std::fs::canonicalize(state_path) {
                    if abs_path == abs_state {
                        debug!("Skipping agent state file: {}", path_str);
                        return Ok(());
                    }
                } else {
                     // If state file doesn't exist yet but paths match string-wise
                     if path == *state_path {
                         debug!("Skipping agent state file: {}", path_str);
                         return Ok(());
                     }
                }
            }
        }

        // 0.1 Hidden file check (matches behavior of scan_all)
        // Check if any component starts with a dot (excluding '.' and '..')
        if path.components().any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s.starts_with('.') && s != "." && s != ".."
        }) {
            debug!("Skipping hidden path: {}", path_str);
            return Ok(());
        }

        // 0.1 Check Gitignore
        if let Some(gi) = &self.gitignore {
            // gi.matched handles absolute paths by making them relative to the builder's root.
            if gi.matched(&path, path.is_dir()).is_ignore() {
                debug!("Skipping gitignored file: {}", path_str);
                return Ok(());
            }
        }

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
        debug!("Ingesting: {}", path_str);
        
        // 3. Chunk
        let content_str = String::from_utf8(bytes).ok();
        let chunks = Chunker::chunk_file(&path, content_str.as_deref().unwrap_or(""));
        
        // 4. Send to Job Queue
        let project_id = "main".to_string();
        let mut valid_memory_ids = Vec::new();
        
        let session = self.job_queue.session_manager.get_or_create(&project_id);
        
        // Track which memories are new/updated vs unchanged
        let old_memories = self.path_to_memories.get(&path_norm).cloned().unwrap_or_default();
        let mut new_memories = HashSet::new();

        for chunk in chunks.iter() {
            let mut memory_id = format!("file:{}:{}-{}", path_norm, chunk.start_line, chunk.end_line);
            let mut suffix = 1;
            while new_memories.contains(&memory_id) {
                memory_id = format!("file:{}:{}-{}:{}", path_norm, chunk.start_line, chunk.end_line, suffix);
                suffix += 1;
            }
            new_memories.insert(memory_id.clone());

            let mut chunk_hasher = Sha256::new();
            chunk_hasher.update(chunk.content.as_bytes());
            let chunk_hash = format!("{:x}", chunk_hasher.finalize());
            
            // Optimization: Skip ingestion if ID and content haven't changed
            if let Some(old_hash) = self.memory_hashes.get(&memory_id) {
                if old_hash == &chunk_hash {
                    debug!("Skipping unchanged memory: {}", memory_id);
                    valid_memory_ids.push(memory_id);
                    continue;
                }
            }

            self.memory_hashes.insert(memory_id.clone(), chunk_hash);
            session.expect_write();

            self.job_queue.enqueue(Job::ExtractAndIngest {
                project_id: project_id.clone(),
                memory_id: memory_id.clone(),
                content: chunk.content.clone(),
                file_path: path_norm.clone(),
                structural_cues: chunk.structural_cues.clone(),
                category: chunk.category,
            }).await;
            
            self.job_queue.buffer(&project_id, Job::ProposeCues {
                project_id: project_id.clone(),
                memory_id: memory_id.clone(),
                content: chunk.content.clone(),
            }).await;

            self.job_queue.buffer(&project_id, Job::TrainLexiconFromMemory {
                project_id: project_id.clone(),
                memory_id: memory_id.clone(),
            }).await;
            
            self.job_queue.buffer(&project_id, Job::UpdateGraph {
                project_id: project_id.clone(),
                memory_id: memory_id.clone(),
            }).await;

            valid_memory_ids.push(memory_id);
        }
        
        // Cleanup memories that no longer exist in this file (e.g. after code shift or deletion)
        for old_id in old_memories {
            if !new_memories.contains(&old_id) {
                self.memory_hashes.remove(&old_id);
                // Explicitly delete from engine
                self.job_queue.enqueue(Job::DeleteMemory {
                    project_id: project_id.clone(),
                    memory_id: old_id,
                }).await;
            }
        }
        self.path_to_memories.insert(path_norm.clone(), new_memories);

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
        debug!("Processing deletion: {}", path_str);

        // Remove from tracking
        self.file_hashes.remove(&path_norm);
        if let Some(mems) = self.path_to_memories.remove(&path_norm) {
            for m_id in mems {
                self.memory_hashes.remove(&m_id);
                // Explicitly delete from engine
                self.job_queue.enqueue(Job::DeleteMemory {
                    project_id: "main".to_string(),
                    memory_id: m_id,
                }).await;
            }
        }

        Ok(())
    }

    /// Process content from a URL - fetches, chunks, and ingests
    pub async fn process_url(&mut self, url: &str, project_id: &str) -> Result<Vec<String>, String> {
        use crate::agent::chunker::Chunker;
        
        debug!("Ingesting URL: {}", url);
        
        // Standard ingestion uses sequential chunking
        let chunks = Chunker::chunk_url(url, false).await?;
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
        debug!("Crawl Phase 1: Fetching pages and collecting chunks...");
        
        while let Some((current_url, depth)) = queue.pop_front() {
            debug!("Crawling [depth={}]: {}", depth, current_url);

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
            // Recursive crawler uses sequential chunking
            match Chunker::chunk_url(&current_url, false).await {
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

        debug!("Crawl Phase 1 complete: {} pages, {} total chunks collected", 
              result.pages_crawled, all_chunks.len());

        // ========== PHASE 2: Write all chunks as memories ==========
        debug!("Crawl Phase 2: Writing {} chunks as memories...", all_chunks.len());
        
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
            
            result.memory_ids.push(memory_id);
        }
        
        info!("Crawl Phase 2 complete: {} memories written", result.memory_ids.len());

        // ========== PHASE 3: Buffer background jobs ==========
        debug!("Crawl Phase 3: Buffering {} background jobs...", all_chunks.len() * 3);
        
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

        debug!(
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
        
        debug!("Ingesting content: {} ({} bytes)", filename, content.len());
        
        // Create a virtual path for the chunker to determine content type
        let virtual_path = PathBuf::from(filename);
        let chunks = Chunker::chunk_file(&virtual_path, content);
        let source = format!("api:{}", filename);
        
        self.process_chunks(chunks, project_id, &source).await
    }

    /// Publicly expose processing of chunks for external callers (like API immediate recall)
    pub async fn process_chunks(
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
            
            memory_ids.push(memory_id);
        }
        
        debug!("Enqueued {} chunks from {}", memory_ids.len(), source);
        Ok(memory_ids)
    }

    /// Fetch and chunk a URL without persisting (for immediate recall)
    pub async fn fetch_and_chunk_url(&self, url: &str) -> Result<Vec<crate::agent::chunker::Chunk>, String> {
        use crate::agent::chunker::Chunker;
        
        debug!("Fetching and chunking URL: {}", url);
        // Immediate recall uses parallel chunking for speed
        Chunker::chunk_url(url, true).await
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
