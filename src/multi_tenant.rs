//! Multi-tenant engine supporting project isolation.

use crate::engine::CueMapEngine;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

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
    projects: Arc<DashMap<ProjectId, Arc<CueMapEngine>>>,
}

impl MultiTenantEngine {
    pub fn new() -> Self {
        Self {
            projects: Arc::new(DashMap::new()),
        }
    }
    
    pub fn get_or_create_project(&self, project_id: ProjectId) -> Arc<CueMapEngine> {
        if let Some(engine) = self.projects.get(&project_id) {
            engine.clone()
        } else {
            let engine = Arc::new(CueMapEngine::new());
            self.projects.insert(project_id, engine.clone());
            engine
        }
    }
    
    pub fn get_project(&self, project_id: &ProjectId) -> Option<Arc<CueMapEngine>> {
        self.projects.get(project_id).map(|e| e.clone())
    }
    
    pub fn list_projects(&self) -> Vec<ProjectStats> {
        self.projects
            .iter()
            .map(|entry| {
                let project_id = entry.key().clone();
                let engine = entry.value();
                let stats = engine.get_stats();
                
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
