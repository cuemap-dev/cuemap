use crate::multi_tenant::MultiTenantEngine;
use crate::projects::ProjectContext;
use crate::llm::{LlmConfig, propose_cues};
use crate::normalization::normalize_cue;
use crate::taxonomy::validate_cues;
use crate::config::*;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};
use std::collections::HashSet;
use rayon::prelude::*;
use smallvec::SmallVec;
use uuid::Uuid;

#[derive(Debug)]
pub enum Job {
    ProposeCues { project_id: String, memory_id: String, content: String },
    TrainLexiconFromMemory { project_id: String, memory_id: String },
    ProposeAliases { project_id: String },
    ExtractAndIngest { project_id: String, memory_id: String, content: String, file_path: String, structural_cues: Vec<String>, category: crate::agent::chunker::ChunkCategory },
    VerifyFile { project_id: String, file_path: String, valid_memory_ids: Vec<String> },
    UpdateGraph { project_id: String, memory_id: String },
    ReinforceMemories { project_id: String, memory_ids: Vec<String>, cues: Vec<String> },
    ReinforceLexicon { project_id: String, memory_ids: Vec<String>, cues: Vec<String> },
    ConsolidateMemories { project_id: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestionPhase {
    Writing,      // Accepting writes, buffering jobs
    Processing,   // Processing buffered jobs
    Done,         // All jobs complete
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct JobProgress {
    pub phase: String,
    pub writes_completed: usize,
    pub writes_total: usize,
    pub propose_cues_completed: usize,
    pub propose_cues_total: usize,
    pub train_lexicon_completed: usize,
    pub train_lexicon_total: usize,
    pub update_graph_completed: usize,
    pub update_graph_total: usize,
}

/// Tracks a bulk ingestion session with buffered jobs
pub struct IngestionSession {
    pub project_id: String,
    pub phase: std::sync::atomic::AtomicU8,  // 0=Writing, 1=Processing, 2=Done
    pub writes_completed: std::sync::atomic::AtomicUsize,
    pub writes_total: std::sync::atomic::AtomicUsize,
    pending_propose_cues: tokio::sync::Mutex<Vec<(String, String, String)>>,  // (project_id, memory_id, content)
    pending_train_lexicon: tokio::sync::Mutex<Vec<(String, String)>>,         // (project_id, memory_id)
    pending_update_graph: tokio::sync::Mutex<Vec<(String, String)>>,          // (project_id, memory_id)
    pub propose_cues_completed: std::sync::atomic::AtomicUsize,
    pub train_lexicon_completed: std::sync::atomic::AtomicUsize,
    pub update_graph_completed: std::sync::atomic::AtomicUsize,
    last_write: tokio::sync::Mutex<std::time::Instant>,
}

impl IngestionSession {
    pub fn new(project_id: String) -> Self {
        Self {
            project_id,
            phase: std::sync::atomic::AtomicU8::new(0),
            writes_completed: std::sync::atomic::AtomicUsize::new(0),
            writes_total: std::sync::atomic::AtomicUsize::new(0),
            pending_propose_cues: tokio::sync::Mutex::new(Vec::new()),
            pending_train_lexicon: tokio::sync::Mutex::new(Vec::new()),
            pending_update_graph: tokio::sync::Mutex::new(Vec::new()),
            propose_cues_completed: std::sync::atomic::AtomicUsize::new(0),
            train_lexicon_completed: std::sync::atomic::AtomicUsize::new(0),
            update_graph_completed: std::sync::atomic::AtomicUsize::new(0),
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
            writes_completed: self.writes_completed.load(std::sync::atomic::Ordering::Relaxed),
            writes_total: self.writes_total.load(std::sync::atomic::Ordering::Relaxed),
            propose_cues_completed: self.propose_cues_completed.load(std::sync::atomic::Ordering::Relaxed),
            propose_cues_total: 0, // Will be set after flush
            train_lexicon_completed: self.train_lexicon_completed.load(std::sync::atomic::Ordering::Relaxed),
            train_lexicon_total: 0,
            update_graph_completed: self.update_graph_completed.load(std::sync::atomic::Ordering::Relaxed),
            update_graph_total: 0,
        }
    }
    
    /// Buffer a job for later processing
    pub async fn buffer_job(&self, job: Job) {
        *self.last_write.lock().await = std::time::Instant::now();
        
        match job {
            Job::ProposeCues { project_id, memory_id, content } => {
                self.pending_propose_cues.lock().await.push((project_id, memory_id, content));
            }
            Job::TrainLexiconFromMemory { project_id, memory_id } => {
                self.pending_train_lexicon.lock().await.push((project_id, memory_id));
            }
            Job::UpdateGraph { project_id, memory_id } => {
                self.pending_update_graph.lock().await.push((project_id, memory_id));
            }
            _ => {} // Other jobs are not buffered
        }
    }
    
    /// Mark a write as complete
    pub fn write_complete(&self) {
        self.writes_completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Increment expected write count
    pub fn expect_write(&self) {
        // Reactivate session if it was done or processing
        self.phase.store(0, std::sync::atomic::Ordering::Relaxed);
        self.writes_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Check if we should auto-flush (no writes for 2 seconds)
    pub async fn should_auto_flush(&self) -> bool {
        let last = *self.last_write.lock().await;
        let writes_done = self.writes_completed.load(std::sync::atomic::Ordering::Relaxed);
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
    pub async fn flush(&self, provider: &Arc<dyn ProjectProvider>) {
        use std::sync::atomic::Ordering;
        
        // Try to transition Writing -> Processing
        // If phase is not Writing (e.g. already Processing), skip
        if self.phase.compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed).is_err() {
            return;
        }
        
        // Get job counts for progress reporting
        let propose_cues = std::mem::take(&mut *self.pending_propose_cues.lock().await);
        let train_lexicon = std::mem::take(&mut *self.pending_train_lexicon.lock().await);
        let update_graph = std::mem::take(&mut *self.pending_update_graph.lock().await);
        
        let total_propose = propose_cues.len();
        let total_train = train_lexicon.len();
        let total_graph = update_graph.len();
        
        if total_propose > 0 || total_train > 0 || total_graph > 0 {
            info!("[Jobs] Phase 2: Processing {} ProposeCues, {} TrainLexicon, {} UpdateGraph", 
                  total_propose, total_train, total_graph);
            
            // Process ProposeCues first
            for (_i, (project_id, memory_id, content)) in propose_cues.into_iter().enumerate() {

                process_job(Job::ProposeCues { project_id, memory_id, content }, provider).await;
                self.propose_cues_completed.fetch_add(1, Ordering::Relaxed);
            }
            
            // Then TrainLexicon
            for (_i, (project_id, memory_id)) in train_lexicon.into_iter().enumerate() {

                process_job(Job::TrainLexiconFromMemory { project_id, memory_id }, provider).await;
                self.train_lexicon_completed.fetch_add(1, Ordering::Relaxed);
            }
            
            // Finally UpdateGraph
            for (_i, (project_id, memory_id)) in update_graph.into_iter().enumerate() {

                process_job(Job::UpdateGraph { project_id, memory_id }, provider).await;
                self.update_graph_completed.fetch_add(1, Ordering::Relaxed);
            }
            
            info!("[Jobs] All background jobs complete ✓");
        }
        
        // Try to transition Processing -> Done
        // If phase changed back to Writing during processing (via expect_write), this will fail,
        // leaving the session in Writing mode (which checks out, as we have new work).
        let _ = self.phase.compare_exchange(1, 2, Ordering::Relaxed, Ordering::Relaxed);
    }
}

/// Manages ingestion sessions per project
pub struct SessionManager {
    sessions: dashmap::DashMap<String, Arc<IngestionSession>>,
    provider: Arc<dyn ProjectProvider>,
}

impl SessionManager {
    pub fn new(provider: Arc<dyn ProjectProvider>) -> Self {
        Self {
            sessions: dashmap::DashMap::new(),
            provider,
        }
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
            session.flush(&self.provider).await;
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
                    if session.get_phase() == IngestionPhase::Writing && session.should_auto_flush().await {
                        sessions_to_flush.push(session);
                    }
                }
                
                // 2. Flush sessions outside the lock
                for session in sessions_to_flush {
                    // debug!("[Jobs] Auto-flushing session for project: {}", session.project_id);
                    session.flush(&manager.provider).await;
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
}

// Abstraction to access projects regardless of mode
pub trait ProjectProvider: Send + Sync + 'static {
    fn get_project(&self, project_id: &str) -> Option<Arc<ProjectContext>>;
    fn save_project(&self, project_id: &str) -> Result<(), String>;
}

impl ProjectProvider for MultiTenantEngine {
    fn get_project(&self, project_id: &str) -> Option<Arc<ProjectContext>> {
        self.get_project(&project_id.to_string())
    }
    
    fn save_project(&self, project_id: &str) -> Result<(), String> {
        self.save_project(&project_id.to_string()).map(|_| ())
    }
}



impl JobQueue {
    pub fn new(provider: Arc<dyn ProjectProvider>, disable_bg_jobs: bool) -> Self {
        let (tx, mut rx) = mpsc::channel(1000);
        let provider_clone = provider.clone();
        
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                if !disable_bg_jobs {
                    process_job(job, &provider_clone).await;
                }
            }
        });
        
        let session_manager = Arc::new(SessionManager::new(provider));
        session_manager.clone().start_auto_flush();
        
        Self { 
            sender: tx,
            session_manager,
        }
    }
    
    /// Enqueue a job immediately (for non-buffered jobs like Reinforce)
    pub async fn enqueue(&self, job: Job) {
        if let Err(e) = self.sender.send(job).await {
            warn!("Failed to enqueue job: {}", e);
        }
    }
    
    /// Buffer a job for phased processing
    pub async fn buffer(&self, project_id: &str, job: Job) {
        let session = self.session_manager.get_or_create(project_id);
        session.buffer_job(job).await;
    }
    
    /// Get session for a project
    pub fn get_session(&self, project_id: &str) -> Option<Arc<IngestionSession>> {
        self.session_manager.get(project_id)
    }
}

struct CueCandidate {
    cue: String,
    len: usize,
    sample: HashSet<String>, // Hashed set for fast lookups in stage 1
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
    if tokens_a.is_empty() { return false; }
    
    let tokens_b = cue_tokens(b);
    if tokens_b.is_empty() { return false; }
    
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

/// Check if a cue is suitable for lexicon training (excluding high-cardinality cues)
pub fn is_lexicon_trainable(cue: &str) -> bool {
    let lower = cue.to_lowercase();
    !lower.starts_with("path:") && 
    !lower.starts_with("id:") && 
    !lower.starts_with("memory_id:") && 
    !lower.starts_with("file:") && 
    !lower.starts_with("alias_id:") &&
    !lower.starts_with("source:")
}

// Shared logic for training lexicon from memory content (Identity + WordNet Synonyms)
fn train_lexicon_impl(ctx: &ProjectContext, memory_id: &str, content: &str) {
    // Tokenize content
    let tokens = crate::nl::tokenize_to_cues(content);

    
    if tokens.is_empty() {
        return;
    }
    
    let mut identity_count = 0;
    let mut synonym_count = 0;
    let mut sample_synonyms: Vec<String> = Vec::new();
    
    // REFACTOR: Avoid global N^2 association.
    // 1. Associate each token with ITSELF (Identity).
    // 2. Associate each token with its DIRECT synonyms (WordNet).
    
    for token in &tokens {
        if !is_lexicon_trainable(&token) {
            continue;
        }

        // 1. Train Identity: Token -> Token
        let lex_id = format!("cue:{}", token);
        ctx.lexicon.upsert_memory_with_id(
            lex_id.clone(),
            token.clone(),
            vec![token.clone()], 
            None,
            false
        );
        identity_count += 1;

        // 2. Train Synonyms: Token -> Synonym (WordNet)
        let expanded = ctx.semantic_engine.expand_wordnet(&token, &[token.clone()], 0.65, 3);
        
        for synonym in expanded {
            if !is_lexicon_trainable(&synonym) {
                continue;
            }
            // Upsert: Synonym triggered by Token
            let syn_id = format!("cue:{}", synonym);
            ctx.lexicon.upsert_memory_with_id(
                syn_id,
                synonym.clone(),
                vec![token.clone()],
                None,
                false
            );
            synonym_count += 1;
            if sample_synonyms.len() < 5 {
                sample_synonyms.push(format!("{}->{}", token, synonym));
            }
        }
    }
    
    if identity_count > 0 || synonym_count > 0 {
        let sample_str = if !sample_synonyms.is_empty() {
            format!(" (e.g. {})", sample_synonyms.join(", "))
        } else {
            String::new()
        };
        debug!("Job: Lexicon trained {} identity + {} synonym mappings for memory {}{}", 
            identity_count, synonym_count, memory_id, sample_str);
    }
}

async fn process_job(job: Job, provider: &Arc<dyn ProjectProvider>) {
    match job {
        Job::TrainLexiconFromMemory { project_id, memory_id } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                let ctx_clone = ctx.clone();
                let memory_id_clone = memory_id.clone();
                
                tokio::task::spawn_blocking(move || {
                     // Fetch memory from main engine
                     if let Some(memory) = ctx_clone.main.get_memory(&memory_id_clone) {
                         train_lexicon_impl(&ctx_clone, &memory_id_clone, &memory.content);
                     }
                }).await.unwrap();
            }
        }

        Job::ProposeCues { project_id, memory_id, content } => {
             if let Some(ctx) = provider.get_project(&project_id) {
                 let ctx_clone = ctx.clone();
                 let memory_id_clone = memory_id.clone();
                 let content_clone = content.clone();
                 let project_id_clone = project_id.clone();
                 
                 tokio::task::spawn_blocking(move || {
                     let ctx = ctx_clone;
                     let memory_id = memory_id_clone;
                     let content = content_clone;
                     let project_id = project_id_clone;
                     let rt_handle = tokio::runtime::Handle::current();

                     debug!("Job: Proposing cues for memory {} in project {} (strategy: {:?})", memory_id, project_id, ctx.cuegen_strategy);
                 
                 // 1. Resolve known cues (Lexicon recall)
                 let (mut known_cues, _) = ctx.resolve_cues_from_text(&content, false);
                 
                 // 2. Bootstrap if needed (for static strategies to have something to expand)
                 // If Lexicon found very few cues, add raw tokens as seed cues for expansion.
                 // Limit to 10 seeds because expansion multiplies them (each seed → multiple synonyms).
                 if known_cues.len() < 3 {
                     let tokens = crate::nl::tokenize_to_cues(&content);
                     for token in tokens.into_iter().take(10) {
                         if !known_cues.contains(&token) {
                             known_cues.push(token);
                         }
                     }
                 }
                 
                 // Track cues by source for detailed logging
                 let mut wordnet_cues: Vec<String> = Vec::new();
                 let mut glove_cues: Vec<String> = Vec::new();
                 let mut context_cues: Vec<String> = Vec::new();
                 let mut llm_cues: Vec<String> = Vec::new();
                 
                 // IDF Filtering: Identify expansion candidates (rare cues only)
                 let total = ctx.total_memories();
                 // Threshold: 10% of corpus, minimum 20 memories.
                 let threshold = (total as f64 * 0.1).max(20.0) as usize;

                 // PERF/QUALITY: Use raw tokens for expansion to avoid Lexicon Pollution loop.
                 // We only expand what is explicitly in the content.
                 // Filter by IDF to skip common words (e.g. "the").
                 let tokens = crate::nl::tokenize_to_cues(&content);
                 let expansion_candidates: Vec<String> = tokens.iter()
                     .filter(|c| ctx.get_cue_frequency(c) <= threshold)
                     .cloned()
                     .collect();
                 
                 
                 // 3. Static Semantic Expansion (Always on - WordNet)
                 let wn_result = ctx.semantic_engine.expand_wordnet(&content, &expansion_candidates, 0.65, 3);
                 wordnet_cues.extend(wn_result);
                 
                // 4. Strategy Specific Expansion
                match ctx.cuegen_strategy {
                    CueGenStrategy::Default => {
                        // Minimal strategy: Only WordNet (handled below always-on)
                        // No extra expansion.
                    },
                    CueGenStrategy::Glove => {
                        // GloVe Expansion (Nearest Neighbors of Cues)
                        let glove_result = ctx.semantic_engine.expand_glove(&content, &expansion_candidates);
                        glove_cues.extend(glove_result);
                        
                        // Global Context Expansion (Nearest Neighbors of Context Vector)
                        let context_result = ctx.semantic_engine.expand_global_context(&content);
                        context_cues.extend(context_result);
                    },
                     CueGenStrategy::Ollama => {
                         // LLM Expansion
                         if let Some(config) = LlmConfig::from_strategy(&ctx.cuegen_strategy) {
                             let content_ref = content.clone();
                             let known_cues_ref = known_cues.clone();
                             let config_clone = config.clone();
                             match rt_handle.block_on(async move {
                                 propose_cues(&content_ref, &config_clone, &known_cues_ref).await
                             }) {
                                 Ok(result) => llm_cues.extend(result),
                                 Err(e) => error!("Job: LLM failed: {}", e),
                             }
                         }
                     }
                 }
                 
                 // Log source breakdown before normalization
                 let log_sample = |name: &str, cues: &[String]| {
                     if !cues.is_empty() {
                         let sample: Vec<_> = cues.iter().take(5).collect();
                         let suffix = if cues.len() > 5 { format!(" (+{} more)", cues.len() - 5) } else { String::new() };
                         debug!("  └─ {}: {:?}{}", name, sample, suffix);
                     }
                 };
                 
                 log_sample("WordNet", &wordnet_cues);
                 log_sample("GloVe", &glove_cues);
                 log_sample("Context", &context_cues);
                 log_sample("LLM", &llm_cues);
                 

                 
                 // Merge all proposed cues with deduplication and filtering
                 let mut seen = HashSet::new();
                 let mut proposed_cues = Vec::new();
                 
                 let filter_and_add = |cues: Vec<String>, seen: &mut HashSet<String>, out: &mut Vec<String>| {
                     for cue in cues {
                         let lower = cue.to_lowercase();

                         // Skip very short cues
                         if lower.len() < 3 {
                             continue;
                         }
                         // Skip duplicates
                         if seen.contains(&lower) {
                             continue;
                         }
                         seen.insert(lower);
                         out.push(cue);
                     }
                 };
                 
                 filter_and_add(wordnet_cues, &mut seen, &mut proposed_cues);
                 filter_and_add(glove_cues, &mut seen, &mut proposed_cues);
                 filter_and_add(context_cues, &mut seen, &mut proposed_cues);
                 filter_and_add(llm_cues, &mut seen, &mut proposed_cues);
                 
                 // Cap total proposed cues to prevent explosion
                 const MAX_PROPOSED_CUES: usize = 10;
                 if proposed_cues.len() > MAX_PROPOSED_CUES {
                     proposed_cues.truncate(MAX_PROPOSED_CUES);
                     debug!("Job: Truncated proposed cues to {}", MAX_PROPOSED_CUES);
                 }
                 
                 // 5. Merge, Normalize & Validate
                 let mut normalized_cues = Vec::new();
                 for cue in proposed_cues {
                     let (normalized, _) = normalize_cue(&cue, &ctx.normalization);
                     normalized_cues.push(normalized);
                 }
                 
                 let report = validate_cues(normalized_cues, &ctx.taxonomy);
                 
                 // 6. Attach accepted cues
                 if !report.accepted.is_empty() {
                     ctx.main.attach_cues(&memory_id, report.accepted.clone());
                     let sample: Vec<_> = report.accepted.iter().take(8).collect();
                     let suffix = if report.accepted.len() > 8 { format!(" (+{} more)", report.accepted.len() - 8) } else { String::new() };
                     debug!("Job: Attached {} cues to memory {}: {:?}{}", report.accepted.len(), memory_id, sample, suffix);
                     
                     // 7. Retrain lexicon with new cues
                     let tokens = crate::nl::tokenize_to_cues(&content);
                     if !tokens.is_empty() {
                         for canonical_cue in report.accepted {
                             if !is_lexicon_trainable(&canonical_cue) {
                                 continue;
                             }
                             
                              let lex_id = format!("cue:{}", canonical_cue);
                              
                              // Filter out identity mappings
                              let filtered_tokens: Vec<String> = tokens.iter()
                                  .filter(|t| t.as_str() != canonical_cue.as_str() && !canonical_cue.contains(t.as_str()))
                                  .cloned()
                                  .collect();
                                  
                              if filtered_tokens.is_empty() {
                                  continue;
                              }

                              ctx.lexicon.upsert_memory_with_id(
                                  lex_id, 
                                  canonical_cue, 
                                  filtered_tokens, 
                                  None,
                                  false
                              );
                         }
                     }
                 }
                 }).await.unwrap();
             }
        }
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
                        .filter(|(k, cnt)| k.len() >= 3 && *cnt >= ALIAS_MIN_CUE_MEMORIES && *cnt <= ALIAS_MAX_CUE_MEMORIES)
                        .collect();
                    
                    stats.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                    let drop_count = (stats.len() as f64 * 0.01) as usize;
                    let stats = stats.into_iter().skip(drop_count).take(ALIAS_MAX_CANDIDATES).collect::<Vec<_>>();
                    
                    if stats.is_empty() {
                        return;
                    }
                    
                    // 2. Build Candidates
                    let candidates: Vec<CueCandidate> = stats
                        .into_iter()
                        .filter_map(|(key, len)| {
                            if let Some(entry) = cue_index.get(&key) {
                                let sample_vec = entry.get_recent_owned(Some(ALIAS_SAMPLE_SIZE));
                                let sample_set: HashSet<String> = sample_vec.into_iter().collect();
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
                    
                    debug!("Job: Analyzing {} candidates for aliases in project {}", candidates.len(), project_id_clone);
                    
                    // 3. Parallel Comparison
                    let proposals: Vec<(String, String, f64, String)> = candidates
                        .par_iter()
                        .enumerate()
                        .fold(Vec::new, |mut acc, (i, cand_a)| {
                            for cand_b in candidates.iter().skip(i + 1) {
                                let diff = (cand_a.len as isize - cand_b.len as isize).abs();
                                let max_len = std::cmp::max(cand_a.len, cand_b.len);
                                if (diff as f64 / max_len as f64) > ALIAS_SIZE_SIMILARITY_MAX_RATIO {
                                    continue;
                                }
                                
                                if !lexical_gate(&cand_a.cue, &cand_b.cue) {
                                    continue;
                                }
                                
                                let intersection = cand_a.sample.intersection(&cand_b.sample).count();
                                let min_sample_len = std::cmp::min(cand_a.sample.len(), cand_b.sample.len());
                                if min_sample_len == 0 { continue; }
                                
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
                                        
                                        let exact_intersection = smaller.iter().filter(|id| larger.contains(*id)).count();
                                        let min_len = smaller.len();
                                        if min_len == 0 { continue; }
                                        
                                        let exact_score = exact_intersection as f64 / min_len as f64;
                                        
                                        if exact_score >= ALIAS_OVERLAP_THRESHOLD {
                                            let (canon, alias) = choose_canonical(&cand_a.cue, &cand_b.cue);
                                            let alias_id_str = format!("{}->{}", alias, canon);
                                            let alias_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, alias_id_str.as_bytes());
                                            acc.push((alias, canon, exact_score, alias_uuid.to_string()));
                                        }
                                    }
                                }
                            }
                            acc
                        })
                        .reduce(Vec::new, |mut a, b| { a.extend(b); a });
                    
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
                            }).to_string();
                            
                            let cues = vec![
                                "type:alias".to_string(),
                                format!("from:{}", from),
                                format!("to:{}", to),
                                "status:proposed".to_string(),
                                "reason:overlap_analysis".to_string(),
                                id_cue
                            ];
                            
                            ctx_clone.aliases.upsert_memory_with_id(alias_id.clone(), content, cues, None, false);
                            debug!("Job: Proposed alias {} -> {} (score: {:.2})", from, to, score);
                        }
                    }
                }).await.unwrap();
            }

        }
        Job::ExtractAndIngest { project_id, memory_id, content, file_path, structural_cues, category } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                let ctx_clone = ctx.clone();
                let memory_id_clone = memory_id.clone();
                let content_clone = content.clone();
                let file_path_clone = file_path.clone();
                let structural_cues_clone = structural_cues.clone();

                tokio::task::spawn_blocking(move || {
                    debug!("Agent: Fast extraction starting for {} (category: {:?})", memory_id_clone, category);
                    
                    use crate::agent::chunker::ChunkCategory;
                    
                    let mut resolved_cues: Vec<String>;
                    
                    // 1. Resolve raw content cues (tokens only, no expansion)
                    match category {
                        ChunkCategory::Conversation => {
                            resolved_cues = structural_cues_clone;
                            let (normalized_tokens, _) = ctx_clone.resolve_cues_from_text(&content_clone, true);
                            for token in normalized_tokens {
                                if !resolved_cues.contains(&token) {
                                    resolved_cues.push(token);
                                }
                            }
                        },
                        // Treat all other categories similarly: Just get tokens.
                        // Prose/WebContent getting WordNet expansion is now handled by Lexicon Training below.
                        _ => {
                             let (normalized_tokens, _) = ctx_clone.resolve_cues_from_text(&content_clone, true);
                             resolved_cues = normalized_tokens;
                        }
                    }
                    
                    // 2. Add metadata cues
                    resolved_cues.push(format!("path:{}", file_path_clone));
                    resolved_cues.push("source:agent".to_string());
                    resolved_cues.push(format!("category:{:?}", category).to_lowercase());
                    
                    // 3. Upsert memory (Lean cues only)
                    ctx_clone.main.upsert_memory_with_id(
                        memory_id_clone.clone(),
                        content_clone,
                        resolved_cues.clone(),
                        None,
                        false
                    );
                    
                    // Note: Lexicon training is now handled by buffered TrainLexiconFromMemory jobs
                    // to ensure all writes complete before background processing starts.
                    
                    debug!("Agent: Ingested {} ({:?}, {} cues)", memory_id_clone, category, resolved_cues.len());
                }).await.unwrap();
            }
        }

        Job::VerifyFile { project_id, file_path, valid_memory_ids } => {
             if let Some(ctx) = provider.get_project(&project_id) {
                  // Strategy:
                  // 1. Look up all memories associated with "path:{file_path}"
                  // 2. Filter for those that are NOT in valid_memory_ids
                  // 3. Delete them
                  
                  let path_cue = format!("path:{}", file_path);
                  if let Some(ordered_set) = ctx.main.get_cue_index().get(&path_cue) {
                      // Get all memory IDs associated with this file
                      let current_memories = ordered_set.get_recent_owned(None);
                      let valid_set: HashSet<String> = valid_memory_ids.into_iter().collect();
                      
                      let mut deleted_count = 0;
                      for mem_id in current_memories {
                          // Only delete if it's an agent-managed memory (check prefix "file:")
                          // and not in the valid set.
                          if mem_id.starts_with("file:") && !valid_set.contains(&mem_id) {
                               if ctx.main.delete_memory(&mem_id) {
                                   deleted_count += 1;
                               }
                          }
                      }
                      
                      if deleted_count > 0 {
                          debug!("Agent: Verified {}. Pruned {} stale memories.", file_path, deleted_count);
                      } else {
                          debug!("Agent: Verified {}. No stale memories found.", file_path);
                      }
                  }
             }
        }
        Job::UpdateGraph { project_id, memory_id } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                let ctx_clone = ctx.clone();
                let memory_id_clone = memory_id.clone();
                tokio::task::spawn_blocking(move || {
                    if let Some(memory) = ctx_clone.main.get_memories().get(&memory_id_clone) {
                        let cues = memory.cues.clone();
                        // Update of the co-occurrence matrix
                        ctx_clone.main.update_cue_co_occurrence(&cues);
                        debug!("Job: Updated graph connectivity for {} cues (memory: {})", cues.len(), memory_id_clone);
                    }
                }).await.unwrap();
            }
        }

        Job::ReinforceMemories { project_id, memory_ids, cues } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                for memory_id in &memory_ids {
                    ctx.main.reinforce_memory(memory_id, cues.clone());
                }
                debug!("Job: Reinforced {} memories with {} cues", memory_ids.len(), cues.len());
            }
        }
        Job::ReinforceLexicon { project_id, memory_ids, cues } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                for memory_id in &memory_ids {
                    ctx.lexicon.reinforce_memory(memory_id, cues.clone());
                }
                debug!("Job: Reinforced {} lexicon entries with {} cues", memory_ids.len(), cues.len());
            }
        }
        Job::ConsolidateMemories { project_id } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                info!("Starting autonomous consolidation for project '{}'", project_id);
                let merged = ctx.main.consolidate_memories(0.9); // 90% overlap threshold
                if !merged.is_empty() {
                        info!("Consolidation: Merged {} overlapping groups in project '{}'", merged.len(), project_id);
                        // Save snapshot after significant change
                        if let Err(e) = provider.save_project(&project_id) {
                            error!("Failed to save project '{}' after consolidation: {}", project_id, e);
                        }
                } else {
                    debug!("Consolidation: No overlapping memories found for '{}'", project_id);
                }
            }
        }
        }
    }

