use crate::config::*;
use crate::metrics::MetricsCollector;
use crate::multi_tenant::MultiTenantEngine;
use crate::projects::ProjectContext;
use crate::structures::{MainStats, MemoryId};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

// Alias Job Constants
pub const ALIAS_MIN_CUE_MEMORIES: usize = 3;
pub const ALIAS_MAX_CUE_MEMORIES: usize = 5000;
pub const ALIAS_MAX_CANDIDATES: usize = 500;
pub const ALIAS_SAMPLE_SIZE: usize = 50;
pub const ALIAS_SIZE_SIMILARITY_MAX_RATIO: f64 = 0.5;
pub const ALIAS_OVERLAP_THRESHOLD: f64 = 0.65;

#[derive(Debug)]
pub enum MemoryRef {
    Id(MemoryId),
    SourceKey(String),
}

impl Clone for MemoryRef {
    fn clone(&self) -> Self {
        match self {
            Self::Id(id) => Self::Id(*id),
            Self::SourceKey(source_key) => Self::SourceKey(source_key.clone()),
        }
    }
}

impl std::fmt::Display for MemoryRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Id(id) => write!(f, "{}", id),
            Self::SourceKey(source_key) => write!(f, "source_key:{}", source_key),
        }
    }
}

impl MemoryRef {
    fn resolve_main(&self, ctx: &ProjectContext) -> Option<MemoryId> {
        match self {
            Self::Id(id) => Some(*id),
            Self::SourceKey(source_key) => ctx.main.memory_id_for_source_key(source_key),
        }
    }
}

#[derive(Debug)]
pub enum Job {
    ProposeAliases {
        project_id: String,
    },
    ExtractAndIngest {
        project_id: String,
        source_key: String,
        content: String,
        file_path: String,
        structural_cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        category: crate::agent::chunker::ChunkCategory,
    },
    VerifyFile {
        project_id: String,
        file_path: String,
        valid_source_keys: Vec<String>,
    },
    ReinforceMemories {
        project_id: String,
        memory_ids: Vec<MemoryId>,
        cues: Vec<String>,
    },
    ReinforceLexicon {
        project_id: String,
        memory_ids: Vec<MemoryId>,
        cues: Vec<String>,
    },
    UpdateMarketHeatmap {
        project_id: String,
    },
    DeleteMemory {
        project_id: String,
        memory_ref: MemoryRef,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestionPhase {
    Writing,    // Accepting writes, buffering jobs
    Processing, // Processing buffered jobs
    Done,       // All jobs complete
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct JobProgress {
    pub phase: String,
    pub writes_completed: usize,
    pub writes_total: usize,
}

/// Tracks a bulk ingestion session with buffered jobs
pub struct IngestionSession {
    pub project_id: String,
    pub phase: std::sync::atomic::AtomicU8, // 0=Writing, 1=Processing, 2=Done
    pub writes_completed: std::sync::atomic::AtomicUsize,
    pub writes_total: std::sync::atomic::AtomicUsize,
    last_write: tokio::sync::Mutex<std::time::Instant>,
}

impl IngestionSession {
    pub fn new(project_id: String) -> Self {
        Self {
            project_id,
            phase: std::sync::atomic::AtomicU8::new(0),
            writes_completed: std::sync::atomic::AtomicUsize::new(0),
            writes_total: std::sync::atomic::AtomicUsize::new(0),
            last_write: tokio::sync::Mutex::new(std::time::Instant::now()),
        }
    }

    pub fn get_phase(&self) -> IngestionPhase {
        match self.phase.load(std::sync::atomic::Ordering::Relaxed) {
            0 => IngestionPhase::Writing,
            1 => IngestionPhase::Processing,
            _ => IngestionPhase::Done,
        }
    }

    pub fn get_progress(&self) -> JobProgress {
        let phase = match self.get_phase() {
            IngestionPhase::Writing => "writing",
            IngestionPhase::Processing => "processing",
            IngestionPhase::Done => "done",
        };
        JobProgress {
            phase: phase.to_string(),
            writes_completed: self
                .writes_completed
                .load(std::sync::atomic::Ordering::Relaxed),
            writes_total: self.writes_total.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Buffer a job for later processing
    pub async fn buffer_job(&self, job: Job) {
        *self.last_write.lock().await = std::time::Instant::now();

        let _ = job;
    }

    /// Mark a write as complete
    pub fn write_complete(&self) {
        self.writes_completed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Increment expected write count
    pub fn expect_write(&self) {
        // Reactivate session if it was done or processing
        self.phase.store(0, std::sync::atomic::Ordering::Relaxed);
        self.writes_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Check if we should auto-flush (no writes for 2 seconds)
    pub async fn should_auto_flush(&self) -> bool {
        let last = *self.last_write.lock().await;
        let writes_done = self
            .writes_completed
            .load(std::sync::atomic::Ordering::Relaxed);
        let writes_expected = self.writes_total.load(std::sync::atomic::Ordering::Relaxed);

        // Only auto-flush if all expected writes are done AND 2 seconds have passed
        writes_done >= writes_expected && writes_expected > 0 && last.elapsed().as_secs() >= 2
    }

    pub fn is_stale(&self) -> bool {
        let phase = self.phase.load(std::sync::atomic::Ordering::Relaxed);
        // If done/idle for more than 5 minutes
        phase == 2 && self.writes_total.load(std::sync::atomic::Ordering::Relaxed) > 0
    }

    /// Flush and process all buffered jobs in order
    pub async fn flush(
        &self,
        provider: &Arc<dyn ProjectProvider>,
        metrics: &Option<Arc<MetricsCollector>>,
        jobs_config: &JobsConfig,
    ) {
        use std::sync::atomic::Ordering;

        // Try to transition Writing -> Processing
        // If phase is not Writing (e.g. already Processing), skip
        if self
            .phase
            .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        let _ = (provider, metrics, jobs_config);

        // Try to transition Processing -> Done
        // If phase changed back to Writing during processing (via expect_write), this will fail,
        // leaving the session in Writing mode (which checks out, as we have new work).
        let _ = self
            .phase
            .compare_exchange(1, 2, Ordering::Relaxed, Ordering::Relaxed);
    }
}

/// Manages ingestion sessions per project
pub struct SessionManager {
    sessions: dashmap::DashMap<String, Arc<IngestionSession>>,
    provider: Arc<dyn ProjectProvider>,
    metrics: Option<Arc<MetricsCollector>>,
    jobs_config: JobsConfig,
}

impl SessionManager {
    pub fn new(
        provider: Arc<dyn ProjectProvider>,
        metrics: Option<Arc<MetricsCollector>>,
        jobs_config: JobsConfig,
    ) -> Self {
        Self {
            sessions: dashmap::DashMap::new(),
            provider,
            metrics,
            jobs_config,
        }
    }

    /// Get aggregated progress across all sessions
    pub fn get_global_progress(&self) -> JobProgress {
        let mut global = JobProgress {
            phase: "idle".to_string(), // Default
            writes_completed: 0,
            writes_total: 0,
        };

        let mut active_count = 0;

        for entry in self.sessions.iter() {
            let p = entry.value().get_progress();

            global.writes_completed += p.writes_completed;
            global.writes_total += p.writes_total;

            if p.phase != "idle" && p.phase != "done" {
                active_count += 1;
            }
        }

        if active_count > 0 {
            global.phase = format!("processing ({} projects)", active_count);
        }

        global
    }

    /// Get or create a session for a project
    pub fn get_or_create(&self, project_id: &str) -> Arc<IngestionSession> {
        self.sessions
            .entry(project_id.to_string())
            .or_insert_with(|| Arc::new(IngestionSession::new(project_id.to_string())))
            .clone()
    }

    /// Get session if it exists
    pub fn get(&self, project_id: &str) -> Option<Arc<IngestionSession>> {
        self.sessions.get(project_id).map(|r| r.clone())
    }

    /// Flush a specific session
    pub async fn flush_session(&self, project_id: &str) {
        if let Some(session) = self.get(project_id) {
            session
                .flush(&self.provider, &self.metrics, &self.jobs_config)
                .await;
        }
    }

    /// Start auto-flush background task
    pub fn start_auto_flush(self: Arc<Self>) {
        let manager = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
            let mut cleanup_interval = 0;

            loop {
                interval.tick().await;

                // Flush sessions
                // 1. Collect sessions to flush (avoid holding DashMap lock during flush)
                let mut sessions_to_flush = Vec::new();
                for entry in manager.sessions.iter() {
                    let session = entry.value().clone();
                    if session.get_phase() == IngestionPhase::Writing
                        && session.should_auto_flush().await
                    {
                        sessions_to_flush.push(session);
                    }
                }

                // 2. Flush sessions outside the lock
                for session in sessions_to_flush {
                    debug!(
                        "[Jobs] Auto-flushing session for project: {}",
                        session.project_id
                    );
                    session
                        .flush(&manager.provider, &manager.metrics, &manager.jobs_config)
                        .await;
                }

                // Cleanup stale sessions every 30 iterations (60 seconds)
                cleanup_interval += 1;
                if cleanup_interval >= 30 {
                    cleanup_interval = 0;
                    // We need to collect keys to remove to avoid deadlock on DashMap if removing during iteration?
                    // DashMap is safe for concurrent removal, but retain() is easier.
                    manager.sessions.retain(|_, session| !session.is_stale());
                }
            }
        });
    }
}

pub struct JobQueue {
    sender: mpsc::Sender<Job>,
    pub session_manager: Arc<SessionManager>,
    pub metrics: Option<Arc<MetricsCollector>>,
    background_processing: bool,
}

// Abstraction to access projects regardless of mode
pub trait ProjectProvider: Send + Sync + 'static {
    fn get_project(&self, project_id: &str) -> Option<Arc<ProjectContext>>;
    fn save_project(&self, project_id: &str) -> Result<(), String>;
    fn list_active_projects(&self) -> Vec<String>;
}

impl ProjectProvider for MultiTenantEngine {
    fn get_project(&self, project_id: &str) -> Option<Arc<ProjectContext>> {
        self.get_or_create_project(project_id.to_string()).ok()
    }

    fn save_project(&self, project_id: &str) -> Result<(), String> {
        self.save_project(&project_id.to_string()).map(|_| ())
    }

    fn list_active_projects(&self) -> Vec<String> {
        self.list_projects()
            .into_iter()
            .map(|p| p.project_id)
            .collect()
    }
}

impl JobQueue {
    pub fn new(
        provider: Arc<dyn ProjectProvider>,
        metrics: Option<Arc<MetricsCollector>>,
        disable_bg_jobs: bool,
    ) -> Self {
        let mut jobs_config = JobsConfig::default();
        jobs_config.background_processing = !disable_bg_jobs;
        Self::new_with_config(provider, metrics, jobs_config)
    }

    pub fn new_with_config(
        provider: Arc<dyn ProjectProvider>,
        metrics: Option<Arc<MetricsCollector>>,
        jobs_config: JobsConfig,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel(1000);
        let provider_clone = provider.clone();
        let session_manager = Arc::new(SessionManager::new(
            provider.clone(),
            metrics.clone(),
            jobs_config.clone(),
        ));
        let session_manager_clone = session_manager.clone();
        let metrics_clone = metrics.clone();
        let background_processing = jobs_config.background_processing;
        let jobs_config_for_worker = jobs_config.clone();

        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                // Determine if this job should signal a session write completion
                let project_for_completion = match &job {
                    Job::ExtractAndIngest { project_id, .. } => Some(project_id.clone()),
                    _ => None,
                };

                if background_processing {
                    process_job(
                        job,
                        &provider_clone,
                        &metrics_clone,
                        &jobs_config_for_worker,
                    )
                    .await;
                }

                // If it was a write job, signal completion to the session
                if let Some(pid) = project_for_completion {
                    if let Some(session) = session_manager_clone.get(&pid) {
                        debug!("[Jobs] Async write job complete for project: {}", pid);
                        session.write_complete();
                    }
                }
            }
        });

        if jobs_config.background_processing {
            session_manager.clone().start_auto_flush();

            // Background Task: Market Heatmap Sync (Every 60s)
            let tx_sync = tx.clone();
            let provider_sync = provider.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
                debug!("JobQueue: Market Heatmap Sync background task started");
                loop {
                    interval.tick().await;
                    let projects = provider_sync.list_active_projects();
                    debug!(
                        "JobQueue: Ticking Market Heatmap Sync ({} projects)",
                        projects.len()
                    );
                    // Trigger update for all active projects
                    for pid in projects {
                        let _ = tx_sync
                            .send(Job::UpdateMarketHeatmap { project_id: pid })
                            .await;
                    }
                }
            });
        }

        Self {
            sender: tx,
            session_manager,
            metrics,
            background_processing: jobs_config.background_processing,
        }
    }

    pub fn has_buffered_ingest_jobs(&self) -> bool {
        false
    }

    pub fn should_buffer_job(&self, job: &Job) -> bool {
        if !self.background_processing {
            return false;
        }

        let _ = job;
        false
    }

    /// Enqueue a job immediately (for non-buffered jobs like Reinforce)
    pub async fn enqueue(&self, job: Job) {
        if let Err(e) = self.sender.send(job).await {
            warn!("Failed to enqueue job: {}", e);
        }
    }

    /// Buffer a job for phased processing
    pub async fn buffer(&self, project_id: &str, job: Job) {
        if !self.should_buffer_job(&job) {
            return;
        }
        let session = self.session_manager.get_or_create(project_id);
        session.buffer_job(job).await;
    }

    /// Get session for a project
    pub fn get_session(&self, project_id: &str) -> Option<Arc<IngestionSession>> {
        self.session_manager.get(project_id)
    }

    /// Get total pending job count across all sessions (for metrics)
    pub fn pending_count(&self) -> usize {
        let mut count = 0;
        for entry in self.session_manager.sessions.iter() {
            let session = entry.value();
            let progress = session.get_progress();
            // Count jobs that haven't completed yet
            let pending = progress
                .writes_total
                .saturating_sub(progress.writes_completed);
            count += pending;
        }
        count
    }

    pub fn get_global_progress(&self) -> JobProgress {
        self.session_manager.get_global_progress()
    }
}

struct CueCandidate {
    cue: String,
    len: usize,
    sample: HashSet<MemoryId>, // Hashed set for fast lookups in stage 1
}

// --- Helper Functions ---

/// Split cue into significant tokens
fn cue_tokens(cue: &str) -> SmallVec<[String; 8]> {
    let mut tokens = SmallVec::new();
    let parts = cue.split(|c| c == ':' || c == '-' || c == '_');

    for part in parts {
        let lower = part.to_lowercase();
        if lower.len() >= 3 {
            tokens.push(lower);
        }
    }
    tokens
}

/// Check if two cues share at least one significant token
fn lexical_gate(a: &str, b: &str) -> bool {
    // 1. Check if one contains the other (simple rewrite)
    if a.contains(b) || b.contains(a) {
        return true;
    }

    // 2. Token overlap
    let tokens_a = cue_tokens(a);
    if tokens_a.is_empty() {
        return false;
    }

    let tokens_b = cue_tokens(b);
    if tokens_b.is_empty() {
        return false;
    }

    for ta in &tokens_a {
        for tb in &tokens_b {
            if ta == tb {
                return true;
            }
        }
    }

    false
}

/// Check if cue is in canonical key:value format
fn is_canonical_format(cue: &str) -> bool {
    match cue.split_once(':') {
        Some((k, v)) => !k.is_empty() && !v.is_empty(),
        None => false,
    }
}

/// Deterministically choose (canonical, alias)
fn choose_canonical(a: &str, b: &str) -> (String, String) {
    let a_canon = is_canonical_format(a);
    let b_canon = is_canonical_format(b);

    if a_canon && !b_canon {
        (a.to_string(), b.to_string())
    } else if !a_canon && b_canon {
        (b.to_string(), a.to_string())
    } else {
        // Tie-breaker: lexicographical
        if a < b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        }
    }
}

async fn process_job(
    job: Job,
    provider: &Arc<dyn ProjectProvider>,
    metrics: &Option<Arc<MetricsCollector>>,
    _jobs_config: &JobsConfig,
) {
    match job {
        Job::ProposeAliases { project_id } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                let ctx_clone = ctx.clone();
                let project_id_clone = project_id.clone();

                tokio::task::spawn_blocking(move || {
                    let cue_index = ctx_clone.main.get_cue_index();

                    // 1. Filter and Select Mid-Frequency Cues
                    let mut stats: Vec<(String, usize)> = cue_index
                        .iter()
                        .map(|entry| (entry.key().clone(), entry.value().len()))
                        .filter(|(k, cnt)| {
                            k.len() >= 3
                                && *cnt >= ALIAS_MIN_CUE_MEMORIES
                                && *cnt <= ALIAS_MAX_CUE_MEMORIES
                        })
                        .collect();

                    stats.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                    let drop_count = (stats.len() as f64 * 0.01) as usize;
                    let stats = stats
                        .into_iter()
                        .skip(drop_count)
                        .take(ALIAS_MAX_CANDIDATES)
                        .collect::<Vec<_>>();

                    if stats.is_empty() {
                        return;
                    }

                    // 2. Build Candidates
                    let candidates: Vec<CueCandidate> = stats
                        .into_iter()
                        .filter_map(|(key, len)| {
                            if let Some(entry) = cue_index.get(&key) {
                                let sample_vec = entry.get_recent_owned(Some(ALIAS_SAMPLE_SIZE));
                                let sample_set: HashSet<MemoryId> = sample_vec.into_iter().collect();
                                Some(CueCandidate {
                                    cue: key,
                                    len,
                                    sample: sample_set,
                                })
                            } else {
                                None
                            }
                        })
                        .collect();

                    info!(
                        "Job: Analyzing {} candidates for aliases in project {}",
                        candidates.len(),
                        project_id_clone
                    );

                    // 3. Parallel Comparison
                    let proposals: Vec<(String, String, f64, String)> = candidates
                        .par_iter()
                        .enumerate()
                        .fold(Vec::new, |mut acc, (i, cand_a)| {
                            for cand_b in candidates.iter().skip(i + 1) {
                                let diff = (cand_a.len as isize - cand_b.len as isize).abs();
                                let max_len = std::cmp::max(cand_a.len, cand_b.len);
                                if (diff as f64 / max_len as f64) > ALIAS_SIZE_SIMILARITY_MAX_RATIO
                                {
                                    continue;
                                }

                                if !lexical_gate(&cand_a.cue, &cand_b.cue) {
                                    continue;
                                }

                                let intersection =
                                    cand_a.sample.intersection(&cand_b.sample).count();
                                let min_sample_len =
                                    std::cmp::min(cand_a.sample.len(), cand_b.sample.len());
                                if min_sample_len == 0 {
                                    continue;
                                }

                                let sample_score = intersection as f64 / min_sample_len as f64;
                                if sample_score < (ALIAS_OVERLAP_THRESHOLD - 0.15) {
                                    continue;
                                }

                                if let Some(entry_a) = cue_index.get(&cand_a.cue) {
                                    if let Some(entry_b) = cue_index.get(&cand_b.cue) {
                                        let (smaller, larger) = if entry_a.len() < entry_b.len() {
                                            (&entry_a.items, &entry_b.items)
                                        } else {
                                            (&entry_b.items, &entry_a.items)
                                        };

                                        let exact_intersection = smaller
                                            .iter()
                                            .filter(|id| larger.contains(*id))
                                            .count();
                                        let min_len = smaller.len();
                                        if min_len == 0 {
                                            continue;
                                        }

                                        let exact_score =
                                            exact_intersection as f64 / min_len as f64;

                                        if exact_score >= ALIAS_OVERLAP_THRESHOLD {
                                            let (canon, alias) =
                                                choose_canonical(&cand_a.cue, &cand_b.cue);
                                            let alias_id_str = format!("{}->{}", alias, canon);
                                            let alias_uuid = Uuid::new_v5(
                                                &Uuid::NAMESPACE_OID,
                                                alias_id_str.as_bytes(),
                                            );
                                            acc.push((
                                                alias,
                                                canon,
                                                exact_score,
                                                alias_uuid.to_string(),
                                            ));
                                        }
                                    }
                                }
                            }
                            acc
                        })
                        .reduce(Vec::new, |mut a, b| {
                            a.extend(b);
                            a
                        });

                    // 4. Register Proposals
                    for (from, to, score, alias_id) in proposals {
                        let id_cue = format!("alias_id:{}", alias_id);
                        if !ctx_clone.aliases.get_cue_index().contains_key(&id_cue) {
                            let content = serde_json::json!({
                                "from": from,
                                "to": to,
                                "downweight": score,
                                "status": "proposed",
                                "reason": "overlap_analysis"
                            })
                            .to_string();

                            let cues = vec![
                                "type:alias".to_string(),
                                "status:proposed".to_string(),
                                "reason:overlap_analysis".to_string(),
                                id_cue,
                            ];

                            ctx_clone.aliases.upsert_memory_with_source_key(
                                alias_id.clone(),
                                content,
                                cues,
                                None,
                                Some(MainStats::default()),
                                false,
                                false,
                            );
                            info!(
                                "Job: Proposed alias {} -> {} (score: {:.2})",
                                from, to, score
                            );
                        }
                    }
                })
                .await
                .unwrap();
            }
        }
        Job::ExtractAndIngest {
            project_id,
            source_key,
            content,
            file_path,
            structural_cues,
            metadata,
            category,
        } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                let ctx_clone = ctx.clone();
                let source_key_clone = source_key.clone();
                let content_clone = content.clone();
                let file_path_clone = file_path.clone();
                let structural_cues_clone = structural_cues.clone();
                let metadata_clone = metadata.clone();

                tokio::task::spawn_blocking(move || {
                    debug!(
                        "Agent: Fast extraction starting for {} (category: {:?})",
                        source_key_clone, category
                    );

                    // 1. Resolve raw content cues (tokens only, no expansion)
                    let lang = structural_cues_clone
                        .iter()
                        .find(|c| c.starts_with("lang:"))
                        .map(|c| crate::nl::Language::from(c.as_str()))
                        .unwrap_or(crate::nl::Language::Default);

                    let mut resolved_cues = structural_cues_clone;
                    let (normalized_tokens, _, _) =
                        ctx_clone.resolve_cues_from_text_with_lang(&content_clone, true, lang);
                    for token in normalized_tokens {
                        if !resolved_cues.contains(&token) {
                            resolved_cues.push(token);
                        }
                    }

                    // 2. Add metadata cues
                    resolved_cues.push(format!("path:{}", file_path_clone));
                    resolved_cues.push("source:agent".to_string());
                    resolved_cues.push(format!("category:{:?}", category).to_lowercase());

                    // 3. Upsert memory (Lean cues only)
                    let memory_id = ctx_clone.main.upsert_memory_with_source_key(
                        source_key_clone.clone(),
                        content_clone,
                        resolved_cues.clone(),
                        metadata_clone,
                        Some(MainStats::default()),
                        false,
                        true,
                    );

                    debug!(
                        "Agent: Ingested {} as memory {} ({:?}, {} cues)",
                        source_key_clone,
                        memory_id,
                        category,
                        resolved_cues.len()
                    );
                })
                .await
                .unwrap();

                // Record ingestion metric
                if let Some(m) = metrics {
                    m.record_ingestion();
                }
            }
        }

        Job::VerifyFile {
            project_id,
            file_path,
            valid_source_keys,
        } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                // Strategy:
                // 1. Look up all memories associated with "path:{file_path}"
                // 2. Filter for those that are NOT in valid_memory_ids
                // 3. Delete them

                let path_cue = format!("path:{}", file_path);
                if let Some(ordered_set) = ctx.main.get_cue_index().get(&path_cue) {
                    // Get all memory IDs associated with this file
                    let current_memories = ordered_set.get_recent_owned(None);
                    let valid_set: HashSet<String> = valid_source_keys.into_iter().collect();

                    let mut deleted_count = 0;
                    for mem_id in current_memories {
                        let is_stale = ctx
                            .main
                            .get_memory(mem_id)
                            .and_then(|memory| memory.source_key)
                            .map(|source_key| !valid_set.contains(&source_key))
                            .unwrap_or(false);
                        if is_stale {
                            if ctx.main.delete_memory(mem_id) {
                                deleted_count += 1;
                            }
                        }
                    }

                    if deleted_count > 0 {
                        info!(
                            "Agent: Verified {}. Pruned {} stale memories.",
                            file_path, deleted_count
                        );
                    } else {
                        debug!("Agent: Verified {}. No stale memories found.", file_path);
                    }
                }
            }
        }
        Job::DeleteMemory {
            project_id,
            memory_ref,
        } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                let Some(memory_id) = memory_ref.resolve_main(&ctx) else {
                    debug!("Job: DeleteMemory skipped unresolved memory {}", memory_ref);
                    return;
                };
                if ctx.main.delete_memory(memory_id) {
                    debug!("Job: Deleted stale memory {}", memory_id);
                }
            }
        }
        Job::ReinforceMemories {
            project_id,
            memory_ids,
            cues,
        } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                // 1. Primary Reinforcement
                debug!(
                    "Job: Starting reinforcement for {} memories",
                    memory_ids.len()
                );
                for memory_id in &memory_ids {
                    // Use dynamic reinforcement logic (logarithmic saturation)
                    ctx.main.reinforce_dynamic(*memory_id, 1.0);
                }

                // 2. Ripple Effect (Retrieval-Induced Activation)
                // If this reinforcement was triggered by a Recall (indicated by presence of cues?),
                // we "prime" related memories.
                if !cues.is_empty() && !memory_ids.is_empty() {
                    let ctx_clone = ctx.clone();
                    // Take top 5 memory contents (limiter)
                    let source_memories: Vec<MemoryId> = memory_ids.iter().take(5).copied().collect();

                    // let project_id_clone = project_id.clone(); // Unused

                    tokio::task::spawn_blocking(move || {
                        let mut ripple_targets: HashSet<MemoryId> = HashSet::new();

                        // For each source memory, perform a lightweight recall
                        for mem_id in source_memories {
                            if let Some(mem) = ctx_clone.main.get_memory(mem_id) {
                                let lang = mem
                                    .cues
                                    .iter()
                                    .find(|c| c.starts_with("lang:"))
                                    .map(|c| crate::nl::Language::from(c.as_str()))
                                    .unwrap_or(crate::nl::Language::Default);
                                // Use content as query
                                let content =
                                    ctx_clone.main.read_memory_content(&mem).unwrap_or_default();
                                let ripple_results = ctx_clone.main.recall_fast(
                                    crate::nl::tokenize_to_cues_with_lang(&content, lang),
                                    10,
                                );

                                for res in ripple_results {
                                    // Don't boost the source itself again
                                    if res.memory_id != mem_id {
                                        ripple_targets.insert(res.memory_id);
                                    }
                                }
                            }
                        }

                        // Boost Tier 2 (Ripple) memories
                        if !ripple_targets.is_empty() {
                            debug!(
                                "Job: [Ripple Effect] Priming {} related memories",
                                ripple_targets.len()
                            );
                            for target_id in ripple_targets {
                                // Smaller boost (0.5) for priming
                                ctx_clone.main.reinforce_dynamic(target_id, 0.5);
                            }
                        }
                    });
                }
            }
        }
        Job::ReinforceLexicon {
            project_id,
            memory_ids,
            cues,
        } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                for memory_id in &memory_ids {
                    // Use tiered reinforcement logic (buckets)
                    ctx.lexicon.reinforce_tiered(*memory_id, 1);
                }
                debug!(
                    "Job: Reinforced {} lexicon entries with {} cues",
                    memory_ids.len(),
                    cues.len()
                );
            }
        }
        Job::UpdateMarketHeatmap { project_id } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                // Sync recently reinforced main-memory cues -> Market Heatmap.
                // This is driven by recall/reinforcement events, not by identity lexicon entries.
                let trending = ctx.main.get_trending_cues(1000);

                if !trending.is_empty() {
                    let mut map = ctx.market_heatmap.write().unwrap();
                    map.clear();

                    for (cue, velocity) in trending {
                        // Normalize velocity to 0.0 - 2.0 range
                        // Log10(1 + v) is a good start.
                        let score = (1.0 + velocity).log10() as f32;

                        // Cap at 2.0 to prevent runaway market override
                        let final_score = score.min(2.0);

                        if final_score > 0.1 {
                            map.insert(cue.clone(), final_score);
                            // Log top 10 cues contributing to heatmap
                            if map.len() <= 10 {
                                debug!("Job: [Heatmap] Cue '{}' added with lift {:.2} (velocity={:.2})", cue, final_score, velocity);
                            }
                        }
                    }

                    let total_cues = map.len();
                    let avg_lift = if total_cues > 0 {
                        map.values().sum::<f32>() / total_cues as f32
                    } else {
                        0.0
                    };

                    debug!("Job: Updated Market Heatmap for '{}' with {} active cues (avg lift: {:.2})", project_id, total_cues, avg_lift);
                } else {
                    debug!(
                        "Job: No trending cues found for project '{}', heatmap unchanged",
                        project_id
                    );
                }
            }
        }
    }
}
