use crate::engine::CueMapEngine;
use crate::normalization::NormalizationConfig;
use crate::taxonomy::Taxonomy;
use dashmap::DashMap;
use std::sync::Arc;
use serde_json::Value;

pub struct ProjectContext {
    pub main: CueMapEngine,
    pub aliases: CueMapEngine,
    pub lexicon: CueMapEngine,
    pub query_cache: DashMap<String, Vec<String>>,
    pub normalization: NormalizationConfig,
    pub taxonomy: Taxonomy,
}

impl ProjectContext {
    pub fn new(normalization: NormalizationConfig, taxonomy: Taxonomy) -> Self {
        Self {
            main: CueMapEngine::new(),
            aliases: CueMapEngine::new(),
            lexicon: CueMapEngine::new(),
            query_cache: DashMap::new(),
            normalization,
            taxonomy,
        }
    }
    
    pub fn resolve_cues_from_text(&self, text: &str) -> Vec<String> {
        let normalized_text = crate::nl::normalize_text(text);
        
        // Check cache
        if let Some(cues) = self.query_cache.get(&normalized_text) {
            return cues.clone();
        }
        
        // Tokenize
        let tokens = crate::nl::tokenize_to_cues(text);
        
        if tokens.is_empty() {
            return Vec::new();
        }
        
        // Query lexicon (limit 8, auto_reinforce true)
        let lexicon_results = self.lexicon.recall(tokens, 8, true);
        
        let mut canonical_cues = Vec::new();
        for result in lexicon_results {
            // result.content is the canonical cue
            let (normalized, _) = crate::normalization::normalize_cue(&result.content, &self.normalization);
            canonical_cues.push(normalized);
        }
        
        // Validate list
        let report = crate::taxonomy::validate_cues(canonical_cues, &self.taxonomy);
        let accepted = report.accepted;
        
        // Cache
        self.query_cache.insert(normalized_text, accepted.clone());
        
        accepted
    }
    
    pub fn expand_query_cues(&self, cues: Vec<String>) -> Vec<(String, f64)> {
        let mut expanded: Vec<(String, f64)> = Vec::new();
        
        for cue in cues {
            // 1. Add original cue with weight 1.0
            expanded.push((cue.clone(), 1.0));
            
            // 2. Query aliases
            let alias_query = vec![
                "type:alias".to_string(),
                format!("from:{}", cue),
                "status:active".to_string(),
            ];
            
            // Recall aliases (limit 8, auto_reinforce false to avoid noise)
            let aliases = self.aliases.recall(alias_query, 8, false);
            
            for alias in aliases {
                // Parse alias content to get target cue and weight
                if let Ok(data) = serde_json::from_str::<Value>(&alias.content) {
                     // STRICT FILTER: Check if 'from' matches the current cue exactly
                     if let Some(from_val) = data.get("from").and_then(|v| v.as_str()) {
                         if from_val != cue {
                             continue;
                         }
                     }

                     if let Some(to_cue) = data.get("to").and_then(|v| v.as_str()) {
                         // Default downweight 0.85 if not specified
                         let downweight = data.get("downweight").and_then(|v| v.as_f64()).unwrap_or(0.85);
                         
                         // The "to" field in content is the actual cue, e.g., "service:payments"
                         expanded.push((to_cue.to_string(), downweight));
                     }
                }
            }
        }
        
        expanded
    }
}

pub struct ProjectStore {
    pub projects: DashMap<String, Arc<ProjectContext>>,
}

impl ProjectStore {
    pub fn new() -> Self {
        Self {
            projects: DashMap::new(),
        }
    }

    pub fn get_or_create(&self, project_id: &str) -> Arc<ProjectContext> {
        if let Some(ctx) = self.projects.get(project_id) {
            return ctx.clone();
        }

        // Create new project with default config
        // In a real app, we might load config from DB/disk here
        let ctx = Arc::new(ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
        ));

        self.projects.insert(project_id.to_string(), ctx.clone());
        ctx
    }
}

