use crate::engine::CueMapEngine;
use crate::normalization::NormalizationConfig;
use crate::taxonomy::Taxonomy;
use crate::config::CueGenStrategy;
use crate::semantic::SemanticEngine;
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
    pub cuegen_strategy: CueGenStrategy,
    pub semantic_engine: SemanticEngine,
}

impl ProjectContext {
    pub fn new(normalization: NormalizationConfig, taxonomy: Taxonomy, cuegen_strategy: CueGenStrategy, semantic_engine: SemanticEngine) -> Self {
        Self {
            main: CueMapEngine::new(),
            aliases: CueMapEngine::new(),
            lexicon: CueMapEngine::new(),
            query_cache: DashMap::new(),
            normalization,
            taxonomy,
            cuegen_strategy,
            semantic_engine,
        }
    }
    
    // IDF-based filtering helpers
    pub fn get_cue_frequency(&self, cue: &str) -> usize {
        self.main.get_cue_frequency(cue)
    }

    pub fn total_memories(&self) -> usize {
        self.main.total_memories()
    }
    
    /// Resolves cues from text using the Lexicon.
    /// Returns (resolved_cues, lexicon_memory_ids) - the memory IDs can be used for async reinforcement.
    pub fn resolve_cues_from_text(&self, text: &str) -> (Vec<String>, Vec<String>) {
        use std::time::Instant;
        let t_start = Instant::now();
        
        let normalized_text = crate::nl::normalize_text(text);
        
        // Check cache (cache only stores cues, not memory IDs)
        if let Some(cues) = self.query_cache.get(&normalized_text) {
            return (cues.clone(), Vec::new());  // No memory IDs from cache
        }
        
        // Tokenize
        let t_tok = Instant::now();
        let tokens = crate::nl::tokenize_to_cues(text);
        let tok_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
        
        if tokens.is_empty() {
            return (Vec::new(), Vec::new());
        }
        
        // Fast lexicon lookup - O(1) per cue, no scoring overhead
        let t_lex = Instant::now();
        let lexicon_results = self.lexicon.recall_fast(tokens, 64);
        let lex_ms = t_lex.elapsed().as_secs_f64() * 1000.0;
        
        let t_norm = Instant::now();
        let mut canonical_cues = Vec::new();
        let mut lexicon_memory_ids = Vec::new();
        for result in lexicon_results {
            // result.content is the canonical cue
            let (normalized, _) = crate::normalization::normalize_cue(&result.content, &self.normalization);
            canonical_cues.push(normalized);
            lexicon_memory_ids.push(result.memory_id.clone());
        }
        let norm_ms = t_norm.elapsed().as_secs_f64() * 1000.0;
        
        // Validate list
        let t_val = Instant::now();
        let report = crate::taxonomy::validate_cues(canonical_cues, &self.taxonomy);
        let accepted = report.accepted;
        let val_ms = t_val.elapsed().as_secs_f64() * 1000.0;
        
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        
        // Log timing breakdown if slow (>1ms)
        if total_ms > 1.0 {
            tracing::info!(
                "resolve_cues_from_text: tok={:.2}ms lex_recall={:.2}ms norm={:.2}ms val={:.2}ms | total={:.2}ms",
                tok_ms, lex_ms, norm_ms, val_ms, total_ms
            );
        }
        
        // Cache
        self.query_cache.insert(normalized_text, accepted.clone());
        
        (accepted, lexicon_memory_ids)
    }
    
    pub fn expand_query_cues(&self, cues: Vec<String>, original_tokens: &[String]) -> Vec<(String, f64)> {
        let mut expanded: Vec<(String, f64)> = Vec::new();
        
        for cue in cues {
            // 1. Add original cue with weight 1.0
            expanded.push((cue.clone(), 1.0));
            
            // 2. ONLY expand aliases for original tokens (not Lexicon synonyms)
            // This avoids expensive alias lookups for all 67 Lexicon results
            if !original_tokens.contains(&cue) {
                continue;  // Skip alias expansion for synonyms
            }
            
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
        
        // Deduplicate
        let mut seen = std::collections::HashSet::new();
        expanded.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        expanded.into_iter()
            .filter(|(cue, _)| {
                // PERFORMANCE: Corpus Restriction
                // Only keep cues that exist in the index.
                self.main.get_cue_index().contains_key(cue) && seen.insert(cue.clone())
            })
            .collect()
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
            CueGenStrategy::default(),
            SemanticEngine::new(None),
        ));

        self.projects.insert(project_id.to_string(), ctx.clone());
        ctx
    }
}

