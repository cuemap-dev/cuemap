use crate::agent::chunker::Chunker;
use crate::agent::AgentConfig;
use crate::jobs::{Job, JobQueue, MemoryRef};
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use ignore::Match;
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{debug, info, warn};

const DEFAULT_NOISE_PATTERNS: &[&str] = &[
    // JS/TS
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
    "tsconfig.json",
    "node_modules/",
    ".npmrc",
    ".eslintcache",
    ".next/",
    ".nuxt/",
    "bower_components/",
    "__snapshots__/",
    // Rust
    "Cargo.lock",
    "target/",
    // Python
    "poetry.lock",
    "Pipfile.lock",
    "__pycache__/",
    "venv/",
    ".venv/",
    "env/",
    ".env/",
    ".pytest_cache/",
    ".ipynb_checkpoints/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    // Go
    "go.sum",
    // Java/JVM
    ".gradle/",
    ".m2/",
    "build/",
    // PHP
    "composer.lock",
    // Ruby
    "Gemfile.lock",
    // iOS/macOS
    "Pods/",
    "DerivedData/",
    "*.xcodeproj/",
    "*.xcworkspace/",
    // System & IDEs
    ".DS_Store",
    "Thumbs.db",
    ".idea/",
    ".vscode/",
    ".history/",
    ".git/",
    ".svn/",
    ".hg/",
];
const REPOSITORY_IGNORE_FILENAMES: &[&str] =
    &[".gitignore", ".antigravityignore", ".cuemapignore"];
const INGESTER_STATE_VERSION: u32 = 2;

pub struct Ingester {
    config: AgentConfig,
    job_queue: Arc<JobQueue>,
    file_hashes: HashMap<String, String>, // path -> sha256
    policy_ignore: Option<Gitignore>,
    repository_ignores: Vec<Gitignore>,
    memory_hashes: HashMap<String, String>, // memory_id -> content_hash
    path_to_memories: HashMap<String, HashSet<String>>, // path -> set of current memory_ids
}

#[derive(Serialize, Deserialize, Default)]
struct IngesterState {
    #[serde(default)]
    schema_version: u32,
    file_hashes: HashMap<String, String>,
    memory_hashes: HashMap<String, String>,
    path_to_memories: HashMap<String, HashSet<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DirectoryPreviewEntry {
    pub path: String,
    pub kind: String,
    pub supported_files: usize,
    pub bytes: u64,
    pub categories: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DirectoryPreview {
    pub watch_dir: String,
    pub supported_files: usize,
    pub bytes: u64,
    pub entries: Vec<DirectoryPreviewEntry>,
    pub scan_errors: usize,
}

#[derive(Default)]
struct PreviewEntryAccumulator {
    kind: String,
    supported_files: usize,
    bytes: u64,
    categories: BTreeMap<String, usize>,
}

impl Ingester {
    pub fn new(config: AgentConfig, job_queue: Arc<JobQueue>) -> Self {
        // Canonicalize watch_dir to ensure absolute path matching works across the engine
        let watch_path = fs::canonicalize(&config.watch_dir)
            .unwrap_or_else(|_| PathBuf::from(&config.watch_dir));
        debug!("Agent initializing with watch root: {:?}", watch_path);

        let policy_ignore = Self::build_policy_ignore(&watch_path, &config.ignored_patterns);
        let repository_ignores = Self::build_repository_ignores(&watch_path);

        let mut config = config;
        config.watch_dir = watch_path.to_string_lossy().to_string();

        Self {
            config,
            job_queue,
            file_hashes: HashMap::new(),
            policy_ignore,
            repository_ignores,
            memory_hashes: HashMap::new(),
            path_to_memories: HashMap::new(),
        }
    }

    fn is_ignore_file(path: &Path) -> bool {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(|name| REPOSITORY_IGNORE_FILENAMES.contains(&name))
            .unwrap_or(false)
    }

    pub fn is_ignore_config_path(path: &Path) -> bool {
        Self::is_ignore_file(path)
    }

    fn build_policy_ignore(watch_path: &Path, ignored_patterns: &[String]) -> Option<Gitignore> {
        let mut builder = GitignoreBuilder::new(watch_path);

        for pattern in DEFAULT_NOISE_PATTERNS {
            let _ = builder.add_line(None, pattern);
        }
        for pattern in ignored_patterns {
            let _ = builder.add_line(None, pattern);
        }

        match builder.build() {
            Ok(gitignore) => Some(gitignore),
            Err(error) => {
                warn!("Failed to build gitignore: {}", error);
                None
            }
        }
    }

    fn build_repository_ignores(watch_path: &Path) -> Vec<Gitignore> {
        let mut directories = HashSet::new();

        let mut current = Some(watch_path);
        while let Some(directory) = current {
            if REPOSITORY_IGNORE_FILENAMES
                .iter()
                .any(|name| directory.join(name).is_file())
            {
                directories.insert(directory.to_path_buf());
            }
            current = directory.parent();
        }

        for result in WalkBuilder::new(watch_path)
            .hidden(false)
            .git_ignore(true)
            .build()
        {
            if let Ok(entry) = result {
                let ignore_path = entry.path();
                if Self::is_ignore_file(ignore_path) {
                    if let Some(directory) = ignore_path.parent() {
                        directories.insert(directory.to_path_buf());
                    }
                }
            }
        }

        let mut directories: Vec<PathBuf> = directories.into_iter().collect();
        directories.sort_by_key(|path| path.components().count());

        directories
            .into_iter()
            .filter_map(|directory| {
                let mut builder = GitignoreBuilder::new(&directory);
                for filename in REPOSITORY_IGNORE_FILENAMES {
                    let ignore_path = directory.join(filename);
                    if ignore_path.is_file() {
                        if let Some(error) = builder.add(&ignore_path) {
                            warn!("Error loading ignore file at {:?}: {}", ignore_path, error);
                        }
                    }
                }
                match builder.build() {
                    Ok(ignore) if !ignore.is_empty() => Some(ignore),
                    Ok(_) => None,
                    Err(error) => {
                        warn!("Failed to build repository ignore matcher: {}", error);
                        None
                    }
                }
            })
            .collect()
    }

    fn path_is_ignored(&self, path: &Path) -> bool {
        if let Some(policy_ignore) = &self.policy_ignore {
            if policy_ignore
                .matched_path_or_any_parents(path, false)
                .is_ignore()
            {
                return true;
            }
        }

        for repository_ignore in self.repository_ignores.iter().rev() {
            if !path.starts_with(repository_ignore.path()) {
                continue;
            }
            match repository_ignore.matched_path_or_any_parents(path, false) {
                Match::Ignore(_) => return true,
                Match::Whitelist(_) => return false,
                Match::None => {}
            }
        }

        false
    }

    pub fn load_state(&mut self, state_path: &std::path::Path) -> Result<(), String> {
        if !state_path.exists() {
            return Ok(());
        }

        let content = fs::read_to_string(state_path)
            .map_err(|e| format!("Failed to read agent state: {}", e))?;

        let state: IngesterState = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse agent state: {}", e))?;

        if state.schema_version < INGESTER_STATE_VERSION {
            debug!(
                "Upgrading agent state from schema {} to {}; tracked files will be reingested",
                state.schema_version, INGESTER_STATE_VERSION
            );
            self.file_hashes = HashMap::new();
            self.memory_hashes = HashMap::new();
        } else {
            self.file_hashes = state.file_hashes;
            self.memory_hashes = state.memory_hashes;
        }
        self.path_to_memories = state.path_to_memories;

        debug!(
            "Loaded agent state: {} files tracked",
            self.file_hashes.len()
        );
        Ok(())
    }

    pub fn save_state(&self, state_path: &std::path::Path) -> Result<(), String> {
        let state = IngesterState {
            schema_version: INGESTER_STATE_VERSION,
            file_hashes: self.file_hashes.clone(),
            memory_hashes: self.memory_hashes.clone(),
            path_to_memories: self.path_to_memories.clone(),
        };

        let content = serde_json::to_string_pretty(&state)
            .map_err(|e| format!("Failed to serialize agent state: {}", e))?;

        fs::write(state_path, content)
            .map_err(|e| format!("Failed to write agent state: {}", e))?;

        debug!(
            "Saved agent state: {} files tracked",
            self.file_hashes.len()
        );
        Ok(())
    }

    fn path_is_selected(&self, path: &Path) -> bool {
        if self.config.included_paths.is_empty() {
            return true;
        }

        let Ok(relative) = path.strip_prefix(&self.config.watch_dir) else {
            return false;
        };

        self.config.included_paths.iter().any(|included| {
            let included_path = Path::new(included);
            relative == included_path || relative.starts_with(included_path)
        })
    }

    fn path_is_allowed(&self, path: &Path) -> bool {
        if !path.is_file() || Chunker::detect_type(path).is_none() {
            return false;
        }

        if Self::is_ignore_file(path) {
            return false;
        }

        if let Some(ref state_path) = self.config.state_file {
            let matches_state = fs::canonicalize(state_path)
                .map(|canonical_state| canonical_state == path)
                .unwrap_or_else(|_| state_path == path);
            if matches_state {
                return false;
            }
        }

        let Ok(relative) = path.strip_prefix(&self.config.watch_dir) else {
            return false;
        };

        if relative.components().any(|component| {
            let name = component.as_os_str().to_string_lossy();
            name.starts_with('.')
                && name != "."
                && name != ".."
                && name != ".gitignore"
                && name != ".cuemapignore"
                && name != ".antigravityignore"
        }) {
            return false;
        }

        if !self.path_is_selected(path) {
            return false;
        }

        if self.path_is_ignored(path) {
            return false;
        }

        if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
            if self
                .config
                .ignored_extensions
                .iter()
                .any(|ignored| ignored.eq_ignore_ascii_case(extension))
            {
                return false;
            }
        }

        true
    }

    pub fn preview_scope(&self) -> Result<DirectoryPreview, String> {
        let mut entries: BTreeMap<String, PreviewEntryAccumulator> = BTreeMap::new();
        let mut supported_files = 0usize;
        let mut total_bytes = 0u64;
        let mut scan_errors = 0usize;

        let walker = WalkBuilder::new(&self.config.watch_dir)
            .hidden(false)
            .git_ignore(true)
            .build();

        for result in walker {
            let entry = match result {
                Ok(entry) => entry,
                Err(error) => {
                    scan_errors += 1;
                    warn!("Directory preview walk error: {}", error);
                    continue;
                }
            };
            let path = match fs::canonicalize(entry.path()) {
                Ok(path) if self.path_is_allowed(&path) => path,
                _ => continue,
            };
            let Ok(relative) = path.strip_prefix(&self.config.watch_dir) else {
                continue;
            };

            let mut components = relative.components();
            let Some(first_component) = components.next() else {
                continue;
            };
            let nested = components.next().is_some();
            let key = if nested {
                first_component.as_os_str().to_string_lossy().to_string()
            } else {
                relative.to_string_lossy().replace('\\', "/")
            };
            let kind = if nested { "directory" } else { "file" };
            let bytes = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
            let category = format!("{:?}", Chunker::get_category_for_file(&path)).to_lowercase();

            let aggregate = entries.entry(key).or_default();
            aggregate.kind = kind.to_string();
            aggregate.supported_files += 1;
            aggregate.bytes += bytes;
            *aggregate.categories.entry(category).or_default() += 1;
            supported_files += 1;
            total_bytes += bytes;
        }

        Ok(DirectoryPreview {
            watch_dir: self.config.watch_dir.clone(),
            supported_files,
            bytes: total_bytes,
            entries: entries
                .into_iter()
                .map(|(path, entry)| DirectoryPreviewEntry {
                    path,
                    kind: entry.kind,
                    supported_files: entry.supported_files,
                    bytes: entry.bytes,
                    categories: entry.categories,
                })
                .collect(),
            scan_errors,
        })
    }

    pub async fn reload_filters_and_rescan(&mut self) -> Result<(), String> {
        let watch_path = PathBuf::from(&self.config.watch_dir);
        self.policy_ignore =
            Self::build_policy_ignore(&watch_path, &self.config.ignored_patterns);
        self.repository_ignores = Self::build_repository_ignores(&watch_path);
        self.scan_all().await
    }

    pub async fn scan_all(&mut self) -> Result<(), String> {
        debug!("Starting full scan of {}", self.config.watch_dir);

        let path_str = self.config.watch_dir.clone();
        let mut eligible_paths = HashSet::new();
        let mut walk_failed = false;

        // Use ignore crate to respect .gitignore natively for early pruning.
        // CueMap-specific ignore files are evaluated with directory-scoped matchers.
        let walker = WalkBuilder::new(&path_str)
            .hidden(false)
            .git_ignore(true)
            .build();

        for result in walker {
            match result {
                Ok(entry) => {
                    let path = match fs::canonicalize(entry.path()) {
                        Ok(path) if self.path_is_allowed(&path) => path,
                        _ => continue,
                    };
                    eligible_paths.insert(path.to_string_lossy().to_lowercase());
                    if let Err(error) = self.process_file_path(path.clone()).await {
                        debug!("Skipping file {:?}: {}", path, error);
                    }
                    if self.config.throttle_ms > 0 {
                        sleep(Duration::from_millis(self.config.throttle_ms)).await;
                    }
                }
                Err(err) => {
                    walk_failed = true;
                    warn!("Walk error: {}", err);
                }
            }
        }

        if !walk_failed {
            let stale_paths: Vec<String> = self
                .file_hashes
                .keys()
                .filter(|path| !eligible_paths.contains(*path))
                .cloned()
                .collect();
            for stale_path in stale_paths {
                self.delete_tracked_path_key(&stale_path).await;
            }
        }

        debug!("Scan complete. Tracking {} files.", self.file_hashes.len());
        Ok(())
    }

    pub async fn process_file_path(&mut self, path: PathBuf) -> Result<(), String> {
        let path = fs::canonicalize(&path)
            .map_err(|e| format!("Failed to canonicalize path {:?}: {}", path, e))?;
        let path_str = path.to_string_lossy().to_string();

        if !self.path_is_allowed(&path) {
            debug!("Skipping out-of-scope or unsupported file: {}", path_str);
            return Ok(());
        }

        // Standardize casing for case-insensitive filesystems (MacOS/Windows)
        let path_norm = path_str.to_lowercase();

        // 1. Read file as bytes first (works for both text and binary)
        let bytes = fs::read(&path).map_err(|e| format!("Read error: {}", e))?;

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
        let project_id = self.config.project_id.clone();
        let mut valid_memory_ids = Vec::new();

        let session = self.job_queue.session_manager.get_or_create(&project_id);

        // Track which memories are new/updated vs unchanged
        let old_memories = self
            .path_to_memories
            .get(&path_norm)
            .cloned()
            .unwrap_or_default();
        let mut new_memories = HashSet::new();

        for chunk in chunks.iter() {
            let mut memory_id =
                format!("file:{}:{}-{}", path_norm, chunk.start_line, chunk.end_line);
            let mut suffix = 1;
            while new_memories.contains(&memory_id) {
                memory_id = format!(
                    "file:{}:{}-{}:{}",
                    path_norm, chunk.start_line, chunk.end_line, suffix
                );
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

            let category = format!("{:?}", chunk.category).to_lowercase();
            let mut structural_cues = chunk.structural_cues.clone();
            structural_cues.push("source_type:repository_file".to_string());
            structural_cues.push("source_channel:filesystem".to_string());

            let mut metadata = HashMap::new();
            metadata.insert(
                "source_type".to_string(),
                serde_json::json!("repository_file"),
            );
            metadata.insert(
                "source_channel".to_string(),
                serde_json::json!("filesystem"),
            );
            metadata.insert(
                "source_path".to_string(),
                serde_json::json!(path_str.clone()),
            );
            metadata.insert("source_category".to_string(), serde_json::json!(category));

            self.job_queue
                .enqueue(Job::ExtractAndIngest {
                    project_id: project_id.clone(),
                    source_key: memory_id.clone(),
                    content: chunk.content.clone(),
                    file_path: path_norm.clone(),
                    structural_cues,
                    metadata: Some(metadata),
                    embedding: None,
                    category: chunk.category,
                })
                .await;

            valid_memory_ids.push(memory_id);
        }

        // Cleanup memories that no longer exist in this file (e.g. after code shift or deletion)
        for old_id in old_memories {
            if !new_memories.contains(&old_id) {
                self.memory_hashes.remove(&old_id);
                // Explicitly delete from engine
                self.job_queue
                    .enqueue(Job::DeleteMemory {
                        project_id: project_id.clone(),
                        memory_ref: MemoryRef::SourceKey(old_id),
                    })
                    .await;
            }
        }
        self.path_to_memories
            .insert(path_norm.clone(), new_memories);

        // 5. Verification: Prune stale memories
        self.job_queue
            .enqueue(Job::VerifyFile {
                project_id,
                file_path: path_norm,
                valid_source_keys: valid_memory_ids,
            })
            .await;

        Ok(())
    }

    async fn delete_tracked_path_key(&mut self, path_norm: &str) {
        self.file_hashes.remove(path_norm);
        if let Some(memories) = self.path_to_memories.remove(path_norm) {
            for memory_id in memories {
                self.memory_hashes.remove(&memory_id);
                self.job_queue
                    .enqueue(Job::DeleteMemory {
                        project_id: self.config.project_id.clone(),
                        memory_ref: MemoryRef::SourceKey(memory_id),
                    })
                    .await;
            }
        }
    }

    pub async fn delete_file_path(&mut self, path: PathBuf) -> Result<(), String> {
        // Deletion events arrive after the file is gone, so canonicalize the
        // parent directory as a fallback to keep the key consistent with
        // process_file_path on symlinked temporary roots (notably macOS /var).
        let path = fs::canonicalize(&path).or_else(|_| {
            let parent = path
                .parent()
                .ok_or_else(|| std::io::Error::other("missing parent directory"))?;
            let file_name = path
                .file_name()
                .ok_or_else(|| std::io::Error::other("missing file name"))?;
            Ok::<PathBuf, std::io::Error>(fs::canonicalize(parent)?.join(file_name))
        }).map_err(|error| format!("Failed to canonicalize deleted path {:?}: {}", path, error))?;
        let path_str = path.to_string_lossy().to_string();
        let path_norm = path_str.to_lowercase();
        debug!("Processing deletion: {}", path_str);

        self.delete_tracked_path_key(&path_norm).await;

        Ok(())
    }

    /// Process content from a URL - fetches, chunks, and ingests
    pub async fn process_url(
        &mut self,
        url: &str,
        project_id: &str,
    ) -> Result<Vec<String>, String> {
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
        use crate::agent::chunker::{Chunk, Chunker};
        use scraper::Html;
        use std::collections::{HashSet, VecDeque};

        let base_url =
            url::Url::parse(start_url).map_err(|e| format!("Invalid start URL: {}", e))?;
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
            .user_agent(concat!(
                "CueMap/",
                env!("CARGO_PKG_VERSION"),
                " (https://cuemap.dev; bot)"
            ))
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
                        result
                            .errors
                            .push((current_url.clone(), format!("Read error: {}", e)));
                        continue;
                    }
                },
                Err(e) => {
                    result
                        .errors
                        .push((current_url.clone(), format!("Fetch error: {}", e)));
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
                    result
                        .errors
                        .push((current_url.clone(), format!("Chunk error: {}", e)));
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

        debug!(
            "Crawl Phase 1 complete: {} pages, {} total chunks collected",
            result.pages_crawled,
            all_chunks.len()
        );

        // ========== PHASE 2: Write all chunks as memories ==========
        debug!(
            "Crawl Phase 2: Writing {} chunks as memories...",
            all_chunks.len()
        );

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
            self.job_queue
                .enqueue(Job::ExtractAndIngest {
                    project_id: project_id.to_string(),
                    source_key: memory_id.clone(),
                    content: chunk.content.clone(),
                    file_path: source.clone(),
                    structural_cues: chunk.structural_cues.clone(),
                    metadata: None,
                    embedding: None,
                    category: chunk.category,
                })
                .await;

            result.memory_ids.push(memory_id);
        }

        info!(
            "Crawl Phase 2 complete: {} memories written",
            result.memory_ids.len()
        );

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
            ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".mp3", ".mp4", ".wav",
            ".avi", ".mov", ".zip", ".tar", ".gz", ".rar", ".doc", ".docx", ".xls", ".xlsx",
            ".ppt", ".pptx", ".css", ".js", ".json", ".xml", ".rss", ".atom",
        ];
        let lower = url.to_lowercase();
        skip_extensions.iter().any(|ext| lower.contains(ext))
    }

    /// Process raw content (text/json/yaml/etc) without a file path
    pub async fn process_content(
        &mut self,
        content: &str,
        filename: &str,
        project_id: &str,
    ) -> Result<Vec<String>, String> {
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
        source: &str,
    ) -> Result<Vec<String>, String> {
        self.process_chunks_with_metadata(chunks, project_id, source, None)
            .await
    }

    pub async fn process_chunks_with_metadata(
        &mut self,
        chunks: Vec<crate::agent::chunker::Chunk>,
        project_id: &str,
        source: &str,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<String>, String> {
        self.process_chunks_with_metadata_and_embeddings(chunks, project_id, source, metadata, None)
            .await
    }

    pub async fn process_chunks_with_metadata_and_embeddings(
        &mut self,
        chunks: Vec<crate::agent::chunker::Chunk>,
        project_id: &str,
        source: &str,
        metadata: Option<HashMap<String, serde_json::Value>>,
        embeddings: Option<Vec<Vec<f32>>>,
    ) -> Result<Vec<String>, String> {
        let mut memory_ids = Vec::new();

        // Track session for progress reporting
        let session = self.job_queue.session_manager.get_or_create(project_id);
        for _ in &chunks {
            session.expect_write();
        }

        for (chunk_index, chunk) in chunks.iter().enumerate() {
            let mut chunk_hasher = Sha256::new();
            chunk_hasher.update(chunk.content.as_bytes());
            let chunk_hash = format!("{:x}", chunk_hasher.finalize());

            // Use source for ID consistency
            let memory_id = format!("{}:{}", source, chunk_hash);

            // ExtractAndIngest does the write - enqueue immediately
            self.job_queue
                .enqueue(Job::ExtractAndIngest {
                    project_id: project_id.to_string(),
                    source_key: memory_id.clone(),
                    content: chunk.content.clone(),
                    file_path: source.to_string(),
                    structural_cues: chunk.structural_cues.clone(),
                    metadata: metadata.clone(),
                    embedding: embeddings
                        .as_ref()
                        .and_then(|vectors| vectors.get(chunk_index).cloned()),
                    category: chunk.category,
                })
                .await;

            memory_ids.push(memory_id);
        }

        debug!("Enqueued {} chunks from {}", memory_ids.len(), source);
        Ok(memory_ids)
    }

    /// Fetch and chunk a URL without persisting (for immediate recall)
    pub async fn fetch_and_chunk_url(
        &self,
        url: &str,
    ) -> Result<Vec<crate::agent::chunker::Chunk>, String> {
        use crate::agent::chunker::Chunker;

        debug!("Fetching and chunking URL: {}", url);
        // Immediate recall uses parallel chunking for speed
        Chunker::chunk_url(url, true).await
    }

    pub fn get_file_hashes(&self) -> &HashMap<String, String> {
        &self.file_hashes
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TuningConfig;
    use crate::jobs::JobQueue;
    use crate::multi_tenant::MultiTenantEngine;

    fn test_ingester(dir: &Path, state_file: Option<PathBuf>) -> Ingester {
        let provider = Arc::new(MultiTenantEngine::with_snapshots_dir(
            dir.join("snapshots"),
            TuningConfig::default(),
        ));
        let queue = Arc::new(JobQueue::new(provider, None, true));
        Ingester::new(
            AgentConfig {
                project_id: "ingester-tests".to_string(),
                watch_dir: dir.to_string_lossy().to_string(),
                throttle_ms: 0,
                state_file,
                included_paths: Vec::new(),
                ignored_patterns: Vec::new(),
                ignored_extensions: Vec::new(),
            },
            queue,
        )
    }

    #[tokio::test]
    async fn state_round_trip_and_legacy_upgrade_are_safe() {
        let dir = tempfile::tempdir().unwrap();
        let state_path = dir.path().join("state.json");
        let mut ingester = test_ingester(dir.path(), Some(state_path.clone()));
        ingester
            .file_hashes
            .insert("/tmp/note.md".to_string(), "hash".to_string());
        ingester
            .memory_hashes
            .insert("memory-1".to_string(), "chunk-hash".to_string());
        ingester.path_to_memories.insert(
            "/tmp/note.md".to_string(),
            ["memory-1".to_string()].into_iter().collect(),
        );
        ingester.save_state(&state_path).unwrap();

        let mut restored = test_ingester(dir.path(), Some(state_path.clone()));
        restored.load_state(&state_path).unwrap();
        assert_eq!(restored.get_file_hashes().get("/tmp/note.md"), Some(&"hash".to_string()));
        assert!(restored.memory_hashes.contains_key("memory-1"));

        std::fs::write(
            &state_path,
            serde_json::json!({
                "schema_version": 1,
                "file_hashes": {"/tmp/old.md": "old"},
                "memory_hashes": {"old-memory": "old"},
                "path_to_memories": {}
            })
            .to_string(),
        )
        .unwrap();
        restored.load_state(&state_path).unwrap();
        assert!(restored.get_file_hashes().is_empty());
        assert!(restored.memory_hashes.is_empty());

        std::fs::write(&state_path, "not-json").unwrap();
        assert!(restored.load_state(&state_path).is_err());
    }

    #[tokio::test]
    async fn preview_scope_reports_supported_files_and_scope_filters() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("README.md"), "# readme").unwrap();
        std::fs::write(dir.path().join("notes.log"), "ignored").unwrap();

        let mut ingester = test_ingester(dir.path(), None);
        ingester.config.included_paths = vec!["src".to_string()];
        ingester.config.ignored_extensions = vec!["log".to_string()];
        let preview = ingester.preview_scope().unwrap();
        assert_eq!(preview.supported_files, 1);
        assert_eq!(preview.entries[0].path, "src");
        assert_eq!(preview.entries[0].kind, "directory");
        assert_eq!(preview.entries[0].categories.len(), 1);
    }

    #[tokio::test]
    async fn file_processing_skips_unchanged_content_and_deletes_tracking() {
        let dir = tempfile::tempdir().unwrap();
        let note = dir.path().join("note.md");
        std::fs::write(&note, "first").unwrap();
        let mut ingester = test_ingester(dir.path(), None);
        ingester.process_file_path(note.clone()).await.unwrap();
        let first_hashes = ingester.file_hashes.clone();
        ingester.process_file_path(note.clone()).await.unwrap();
        assert_eq!(ingester.file_hashes, first_hashes);
        ingester.delete_file_path(note.clone()).await.unwrap();
        assert!(ingester.file_hashes.is_empty());
        assert!(ingester.path_to_memories.is_empty());
    }
}
