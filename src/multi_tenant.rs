//! Multi-tenant engine supporting project isolation.

use crate::config::TuningConfig;
use crate::crypto::EncryptionKey;
use crate::engine::CueMapEngine;
use crate::normalization::NormalizationConfig;
use crate::persistence::PersistenceManager;
use crate::projects::ProjectContext;
use crate::semantic::SemanticEncoder;
use crate::structures::{LexiconStats, MainStats};
use crate::taxonomy::Taxonomy;
use ahash::RandomState;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub type ProjectId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStats {
    pub project_id: ProjectId,
    pub total_memories: usize,
    pub total_cues: usize,
    pub created_at: f64,
    pub last_activity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMeta {
    pub project_id: ProjectId,
    pub created_at: u64,
    pub watch_dir: Option<String>,
    pub agent_enabled: bool,
    #[serde(default)]
    pub included_paths: Vec<String>,
    #[serde(default)]
    pub ignored_patterns: Vec<String>,
    #[serde(default)]
    pub ignored_extensions: Vec<String>,
}

impl ProjectMeta {
    pub fn new(project_id: ProjectId) -> Self {
        Self {
            project_id,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            watch_dir: None,
            agent_enabled: false,
            included_paths: Vec::new(),
            ignored_patterns: Vec::new(),
            ignored_extensions: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct MultiTenantEngine {
    projects: Arc<DashMap<ProjectId, Arc<ProjectContext>, RandomState>>,
    snapshots_dir: PathBuf,
    master_key: Option<Arc<EncryptionKey>>,
    tuning: Arc<TuningConfig>,
    config: crate::config::ServerConfig,
    semantic_encoder: Arc<OnceLock<Result<Option<Arc<dyn SemanticEncoder>>, String>>>,
}

impl MultiTenantEngine {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::with_snapshots_dir("./snapshots", TuningConfig::default())
    }

    pub fn with_snapshots_dir<P: AsRef<Path>>(
        dir: P,
        tuning: TuningConfig,
    ) -> Self {
        let snapshots_dir = dir.as_ref().to_path_buf();

        // Create snapshots directory if it doesn't exist
        if let Err(e) = fs::create_dir_all(&snapshots_dir) {
            eprintln!("Warning: Failed to create snapshots directory: {}", e);
        }

        Self {
            projects: Arc::new(DashMap::with_hasher(RandomState::new())),
            snapshots_dir,
            master_key: None,
            tuning: Arc::new(tuning),
            config: crate::config::ServerConfig::default(),
            semantic_encoder: Arc::new(OnceLock::new()),
        }
    }

    pub fn with_config(
        mut config: crate::config::ServerConfig,
        snapshots_dir: PathBuf,
    ) -> Self {
        if let Err(e) = fs::create_dir_all(&snapshots_dir) {
            eprintln!("Warning: Failed to create snapshots directory: {}", e);
        }

        config.semantic = config.semantic.resolved();

        let engine = Self {
            projects: Arc::new(DashMap::with_hasher(RandomState::new())),
            snapshots_dir,
            master_key: None,
            tuning: Arc::new(config.tuning.clone()),
            config,
            semantic_encoder: Arc::new(OnceLock::new()),
        };
        if engine.config.semantic.encoder_enabled {
            if let Err(error) = engine.configured_semantic_encoder() {
                tracing::warn!(
                    error = %error,
                    "Semantic encoder unavailable; continuing without automatic text embeddings"
                );
            }
        }
        engine
    }

    fn configured_semantic_encoder(&self) -> Result<Option<Arc<dyn SemanticEncoder>>, String> {
        self.semantic_encoder
            .get_or_init(|| crate::semantic::load_configured_encoder(&self.config.semantic))
            .clone()
    }

    pub fn set_master_key(&mut self, key: Option<Arc<EncryptionKey>>) {
        self.master_key = key;
    }

    pub fn get_or_create_project(
        &self,
        project_id: ProjectId,
    ) -> Result<Arc<ProjectContext>, String> {
        if let Some(ctx) = self.projects.get(&project_id) {
            ctx.touch();
            Ok(ctx.clone())
        } else {
            // Create new project with default config
            let semantic_encoder = match self.configured_semantic_encoder() {
                Ok(encoder) => encoder,
                Err(error) => {
                    tracing::warn!(
                        project_id = %project_id,
                        error = %error,
                        "Semantic encoder unavailable; continuing without automatic text embeddings"
                    );
                    None
                }
            };
            let mut ctx_obj = ProjectContext::new_with_encoder(
                NormalizationConfig::default(),
                Taxonomy::default(),
                self.tuning.clone(),
                self.config.clone(),
                project_id.clone(),
                semantic_encoder,
            );

            // Set master key on engines
            ctx_obj.main.set_master_key(self.master_key.clone());
            ctx_obj.aliases.set_master_key(self.master_key.clone());
            ctx_obj.lexicon.set_master_key(self.master_key.clone());

            let ctx = Arc::new(ctx_obj);
            self.projects.insert(project_id.clone(), ctx.clone());

            // Ensure meta exists
            if let Ok(meta) = self.load_project_meta(&project_id) {
                let _ = self.save_project_meta(&meta);
            }

            Ok(ctx)
        }
    }

    /// Spawns a background thread to periodically save all project snapshots
    pub fn start_periodic_snapshots(&self, interval: Duration) {
        let engine = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                let results = engine.save_all();
                let saved = results.iter().filter(|(_, r)| r.is_ok()).count();
                let failed = results.iter().filter(|(_, r)| r.is_err()).count();

                if saved > 0 {
                    tracing::debug!("Periodic snapshot: saved {} projects", saved);
                }
                if failed > 0 {
                    tracing::warn!("Periodic snapshot: failed to save {} projects", failed);
                }
            }
        });
    }

    pub fn get_project(&self, project_id: &ProjectId) -> Option<Arc<ProjectContext>> {
        self.projects.get(project_id).map(|e| e.clone())
    }

    pub fn list_projects(&self) -> Vec<ProjectStats> {
        self.projects
            .iter()
            .map(|entry| {
                let project_id = entry.key().clone();
                let ctx = entry.value();
                let stats = ctx.main.get_stats();

                ProjectStats {
                    project_id,
                    total_memories: stats
                        .get("total_memories")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                    total_cues: stats
                        .get("total_cues")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                    created_at: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs_f64(),
                    last_activity: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs_f64(),
                }
            })
            .collect()
    }

    pub fn delete_project(&self, project_id: &ProjectId) -> bool {
        self.projects.remove(project_id).is_some()
    }

    /// Save a project snapshot to disk (main, aliases, lexicon)
    pub fn save_project(&self, project_id: &ProjectId) -> Result<PathBuf, String> {
        let ctx = self
            .get_project(project_id)
            .ok_or_else(|| format!("Project '{}' not found", project_id))?;

        // Save all 3 engines with suffixes
        let main_path = self.snapshots_dir.join(format!("{}.bin", project_id));
        let aliases_path = self
            .snapshots_dir
            .join(format!("{}_aliases.bin", project_id));
        let lexicon_path = self
            .snapshots_dir
            .join(format!("{}_lexicon.bin", project_id));

        PersistenceManager::save_to_path(&ctx.main, &main_path)
            .map_err(|e| format!("Failed to save main engine: {}", e))?;

        PersistenceManager::save_to_path(&ctx.aliases, &aliases_path)
            .map_err(|e| format!("Failed to save aliases engine: {}", e))?;

        PersistenceManager::save_to_path(&ctx.lexicon, &lexicon_path)
            .map_err(|e| format!("Failed to save lexicon engine: {}", e))?;

        tracing::info!("Saved project '{}' (main + aliases + lexicon)", project_id);

        Ok(main_path)
    }

    /// Load a project snapshot from disk (main, aliases, lexicon)
    pub fn load_project(&self, project_id: &ProjectId) -> Result<Arc<ProjectContext>, String> {
        let main_path = self.snapshots_dir.join(format!("{}.bin", project_id));
        let aliases_path = self
            .snapshots_dir
            .join(format!("{}_aliases.bin", project_id));
        let lexicon_path = self
            .snapshots_dir
            .join(format!("{}_lexicon.bin", project_id));

        if !main_path.exists() {
            return Err(format!("Snapshot for project '{}' not found", project_id));
        }

        // Load main engine (required)
        let (memories, source_key_to_id, cue_index, next_memory_id, main_counts) =
            PersistenceManager::load_from_path::<MainStats>(&main_path)
                .map_err(|e| format!("Failed to load main engine: {}", e))?;
        let mut main_engine = CueMapEngine::from_state(
            memories,
            source_key_to_id,
            cue_index,
            next_memory_id,
            main_counts,
            self.config.clone(),
            project_id.clone(),
        );
        main_engine.set_master_key(self.master_key.clone());
        main_engine.set_tuning_config(self.tuning.as_ref().clone());
        match self.configured_semantic_encoder() {
            Ok(semantic_encoder) => main_engine.set_semantic_encoder(semantic_encoder),
            Err(error) => {
                tracing::warn!(
                    project_id = %project_id,
                    error = %error,
                    "Semantic encoder unavailable for loaded project; continuing without automatic text embeddings"
                );
            }
        }

        // Load aliases engine (optional - may not exist for older snapshots)
        let mut aliases_engine = if aliases_path.exists() {
            match PersistenceManager::load_from_path::<MainStats>(&aliases_path) {
                Ok((
                    memories,
                    source_key_to_id,
                    cue_index,
                    next_memory_id,
                    aliases_counts,
                )) => {
                    tracing::debug!("Loaded aliases for project '{}'", project_id);
                    let mut local_config = self.config.clone();
                    local_config.server.store_content_on_disk = false;
                    local_config.semantic = crate::semantic::SemanticConfig::default();
                    let engine = CueMapEngine::from_state(
                        memories,
                        source_key_to_id,
                        cue_index,
                        next_memory_id,
                        aliases_counts,
                        local_config,
                        project_id.clone(),
                    );
                    engine
                }
                Err(e) => {
                    tracing::warn!("Failed to load aliases for '{}': {}", project_id, e);
                    CueMapEngine::new()
                }
            }
        } else {
            CueMapEngine::new()
        };
        aliases_engine.set_master_key(self.master_key.clone());
        aliases_engine.set_tuning_config(self.tuning.as_ref().clone());

        // Load lexicon engine (optional - may not exist for older snapshots)
        let mut lexicon_engine = if lexicon_path.exists() {
            match PersistenceManager::load_from_path::<LexiconStats>(&lexicon_path) {
                Ok((
                    memories,
                    source_key_to_id,
                    cue_index,
                    next_memory_id,
                    lex_counts,
                )) => {
                    tracing::debug!("Loaded lexicon for project '{}'", project_id);
                    let mut local_config = self.config.clone();
                    local_config.server.store_content_on_disk = false;
                    local_config.semantic = crate::semantic::SemanticConfig::default();
                    let engine = CueMapEngine::from_state(
                        memories,
                        source_key_to_id,
                        cue_index,
                        next_memory_id,
                        lex_counts,
                        local_config,
                        project_id.clone(),
                    );
                    engine
                }
                Err(e) => {
                    tracing::warn!("Failed to load lexicon for '{}': {}", project_id, e);
                    CueMapEngine::new()
                }
            }
        } else {
            CueMapEngine::new()
        };
        lexicon_engine.set_master_key(self.master_key.clone());
        lexicon_engine.set_tuning_config(self.tuning.as_ref().clone());

        let ctx = Arc::new(ProjectContext {
            main: main_engine,
            aliases: aliases_engine,
            lexicon: lexicon_engine,
            query_cache: DashMap::with_hasher(RandomState::new()),
            symbol_router_cache: RwLock::new(Default::default()),
            normalization: NormalizationConfig::default(),
            taxonomy: Taxonomy::default(),
            last_activity: std::sync::atomic::AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            ),
            market_heatmap: Arc::new(RwLock::new(HashMap::new())),
            tuning: self.tuning.clone(),
            cuebridge_artifacts: RwLock::new(crate::cuebridge::CueBridgeArtifacts::load_for_project(
                &self.config.server.data_dir,
                project_id,
            )),
        });

        self.projects.insert(project_id.clone(), ctx.clone());

        Ok(ctx)
    }

    /// Save all projects to disk
    pub fn save_all(&self) -> HashMap<String, Result<PathBuf, String>> {
        let mut results = HashMap::new();

        // Collect IDs to avoid holding lock during save (prevent re-entrancy deadlock)
        let project_ids: Vec<String> = self.projects.iter().map(|e| e.key().clone()).collect();

        for project_id in project_ids {
            let result = self.save_project(&project_id);
            results.insert(project_id, result);
        }

        results
    }

    /// Load all available snapshots from disk
    pub fn load_all(&self) -> HashMap<String, Result<(), String>> {
        let mut results = HashMap::new();
        let snapshots = self.list_snapshots();

        for project_id in snapshots {
            let result = self
                .load_project(&project_id)
                .map(|_| ())
                .map_err(|e| format!("Failed to load: {}", e));
            results.insert(project_id, result);
        }

        results
    }

    /// List available snapshots on disk
    pub fn list_snapshots(&self) -> Vec<String> {
        PersistenceManager::list_snapshots_in_dir(&self.snapshots_dir)
    }

    /// Delete a project snapshot from disk
    #[allow(dead_code)]
    pub fn delete_snapshot(&self, project_id: &ProjectId) -> Result<(), String> {
        let snapshot_path = self.snapshots_dir.join(format!("{}.bin", project_id));
        let meta_path = self.snapshots_dir.join(format!("{}.meta.json", project_id));

        // Try to delete meta if exists
        if meta_path.exists() {
            let _ = fs::remove_file(meta_path);
        }

        PersistenceManager::delete_snapshot(&snapshot_path)
    }

    /// Load project metadata
    pub fn load_project_meta(&self, project_id: &ProjectId) -> Result<ProjectMeta, String> {
        let meta_path = self.snapshots_dir.join(format!("{}.meta.json", project_id));
        if meta_path.exists() {
            let content = fs::read_to_string(&meta_path).map_err(|e| e.to_string())?;
            let meta: ProjectMeta = serde_json::from_str(&content).map_err(|e| e.to_string())?;
            Ok(meta)
        } else {
            // Return default if not found (legacy projects)
            Ok(ProjectMeta::new(project_id.clone()))
        }
    }

    /// Save project metadata
    pub fn save_project_meta(&self, meta: &ProjectMeta) -> Result<(), String> {
        let meta_path = self
            .snapshots_dir
            .join(format!("{}.meta.json", meta.project_id));
        let content = serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?;
        fs::write(meta_path, content).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Set watch directory for a project
    pub fn set_project_watch_dir(
        &self,
        project_id: &str,
        watch_dir: Option<String>,
    ) -> Result<(), String> {
        // Validation
        if let Some(dir) = &watch_dir {
            let path = Path::new(dir);
            if !path.exists() {
                return Err(format!("Directory '{}' does not exist", dir));
            }
        }

        // Get current meta
        let mut meta = self.load_project_meta(&project_id.to_string())?;

        meta.watch_dir = watch_dir;
        // Auto-enable agent if watch dir is set
        meta.agent_enabled = meta.watch_dir.is_some();

        self.save_project_meta(&meta)?;

        Ok(())
    }

    /// Persist the complete repository ingestion scope for a project.
    pub fn set_project_watch_config(
        &self,
        project_id: &str,
        watch_dir: String,
        included_paths: Vec<String>,
        ignored_patterns: Vec<String>,
        ignored_extensions: Vec<String>,
    ) -> Result<ProjectMeta, String> {
        let path = Path::new(&watch_dir);
        if !path.is_dir() {
            return Err(format!("Directory '{}' does not exist", watch_dir));
        }

        let mut meta = self.load_project_meta(&project_id.to_string())?;
        meta.watch_dir = Some(watch_dir);
        meta.agent_enabled = true;
        meta.included_paths = included_paths;
        meta.ignored_patterns = ignored_patterns;
        meta.ignored_extensions = ignored_extensions;
        self.save_project_meta(&meta)?;
        Ok(meta)
    }

    pub fn get_global_stats(&self) -> HashMap<String, serde_json::Value> {
        let projects = self.list_projects();

        let total_memories: usize = projects.iter().map(|p| p.total_memories).sum();
        let total_cues: usize = projects.iter().map(|p| p.total_cues).sum();

        let mut stats = HashMap::new();
        stats.insert(
            "total_projects".to_string(),
            serde_json::json!(projects.len()),
        );
        stats.insert(
            "total_memories".to_string(),
            serde_json::json!(total_memories),
        );
        stats.insert("total_cues".to_string(), serde_json::json!(total_cues));
        stats.insert("projects".to_string(), serde_json::json!(projects));

        stats
    }

    pub fn project_artifact_summary(
        &self,
        project_id: &ProjectId,
    ) -> Result<crate::cuebridge::CueBridgeArtifactSummary, String> {
        let ctx = self
            .get_project(project_id)
            .ok_or_else(|| format!("Project '{}' not found", project_id))?;
        Ok(ctx.cuebridge_artifact_summary())
    }

    pub fn reload_project_artifacts(
        &self,
        project_id: &ProjectId,
    ) -> Result<crate::cuebridge::CueBridgeArtifactSummary, String> {
        let ctx = self.get_or_create_project(project_id.clone())?;
        Ok(ctx.reload_cuebridge_artifacts(&self.config.server.data_dir, project_id))
    }
}

/// Validate project ID format
pub fn validate_project_id(project_id: &str) -> bool {
    // Allow alphanumeric, hyphens, underscores
    // Length between 3 and 64 characters
    if project_id.len() < 3 || project_id.len() > 64 {
        return false;
    }

    project_id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
}
