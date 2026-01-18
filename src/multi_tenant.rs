//! Multi-tenant engine supporting project isolation.

use crate::engine::CueMapEngine;
use crate::persistence::PersistenceManager;
use crate::projects::ProjectContext;
use crate::normalization::NormalizationConfig;
use crate::taxonomy::Taxonomy;
use crate::config::CueGenStrategy;
use crate::semantic::SemanticEngine;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Duration};

pub type ProjectId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStats {
    pub project_id: ProjectId,
    pub total_memories: usize,
    pub total_cues: usize,
    pub created_at: f64,
    pub last_activity: f64,
}

#[derive(Clone)]
pub struct MultiTenantEngine {
    projects: Arc<DashMap<ProjectId, Arc<ProjectContext>>>,
    snapshots_dir: PathBuf,
    cuegen_strategy: CueGenStrategy,
    semantic_engine: SemanticEngine,
}

impl MultiTenantEngine {
    #[allow(dead_code)]
    pub fn new(cuegen_strategy: CueGenStrategy, semantic_engine: SemanticEngine) -> Self {
        Self::with_snapshots_dir("./snapshots", cuegen_strategy, semantic_engine)
    }
    
    pub fn with_snapshots_dir<P: AsRef<Path>>(dir: P, cuegen_strategy: CueGenStrategy, semantic_engine: SemanticEngine) -> Self {
        let snapshots_dir = dir.as_ref().to_path_buf();
        
        // Create snapshots directory if it doesn't exist
        if let Err(e) = fs::create_dir_all(&snapshots_dir) {
            eprintln!("Warning: Failed to create snapshots directory: {}", e);
        }
        
        Self {
            projects: Arc::new(DashMap::new()),
            snapshots_dir,
            cuegen_strategy,
            semantic_engine,
        }
    }
    
    pub fn get_or_create_project(&self, project_id: ProjectId) -> Result<Arc<ProjectContext>, String> {
        if let Some(ctx) = self.projects.get(&project_id) {
            ctx.touch();
            Ok(ctx.clone())
        } else {


            // Create new project with default config
            let ctx = Arc::new(ProjectContext::new(
                NormalizationConfig::default(),
                Taxonomy::default(),
                self.cuegen_strategy.clone(),
                self.semantic_engine.clone(),
            ));
            self.projects.insert(project_id, ctx.clone());
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
                    total_memories: stats.get("total_memories")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                    total_cues: stats.get("total_cues")
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
        let ctx = self.get_project(project_id)
            .ok_or_else(|| format!("Project '{}' not found", project_id))?;
        
        // Save all 3 engines with suffixes
        let main_path = self.snapshots_dir.join(format!("{}.bin", project_id));
        let aliases_path = self.snapshots_dir.join(format!("{}_aliases.bin", project_id));
        let lexicon_path = self.snapshots_dir.join(format!("{}_lexicon.bin", project_id));
        
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
        let aliases_path = self.snapshots_dir.join(format!("{}_aliases.bin", project_id));
        let lexicon_path = self.snapshots_dir.join(format!("{}_lexicon.bin", project_id));
        
        if !main_path.exists() {
            return Err(format!("Snapshot for project '{}' not found", project_id));
        }
        
        // Load main engine (required)
        let (memories, cue_index) = PersistenceManager::load_from_path(&main_path)
            .map_err(|e| format!("Failed to load main engine: {}", e))?;
        let main_engine = CueMapEngine::from_state(memories, cue_index);
        
        // Load aliases engine (optional - may not exist for older snapshots)
        let aliases_engine = if aliases_path.exists() {
            match PersistenceManager::load_from_path(&aliases_path) {
                Ok((memories, cue_index)) => {
                    tracing::debug!("Loaded aliases for project '{}'", project_id);
                    CueMapEngine::from_state(memories, cue_index)
                }
                Err(e) => {
                    tracing::warn!("Failed to load aliases for '{}': {}", project_id, e);
                    CueMapEngine::new()
                }
            }
        } else {
            CueMapEngine::new()
        };
        
        // Load lexicon engine (optional - may not exist for older snapshots)
        let lexicon_engine = if lexicon_path.exists() {
            match PersistenceManager::load_from_path(&lexicon_path) {
                Ok((memories, cue_index)) => {
                    tracing::debug!("Loaded lexicon for project '{}'", project_id);
                    CueMapEngine::from_state(memories, cue_index)
                }
                Err(e) => {
                    tracing::warn!("Failed to load lexicon for '{}': {}", project_id, e);
                    CueMapEngine::new()
                }
            }
        } else {
            CueMapEngine::new()
        };
        
        let ctx = Arc::new(ProjectContext {
            main: main_engine,
            aliases: aliases_engine,
            lexicon: lexicon_engine,
            query_cache: DashMap::new(),
            normalization: NormalizationConfig::default(),
            taxonomy: Taxonomy::default(),
            cuegen_strategy: self.cuegen_strategy.clone(),
            semantic_engine: self.semantic_engine.clone(),
            last_activity: std::sync::atomic::AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
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
            let result = self.load_project(&project_id)
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
        PersistenceManager::delete_snapshot(&snapshot_path)
    }
    
    #[allow(dead_code)]
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
        stats.insert(
            "total_cues".to_string(),
            serde_json::json!(total_cues),
        );
        stats.insert(
            "projects".to_string(),
            serde_json::json!(projects),
        );
        
        stats
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
