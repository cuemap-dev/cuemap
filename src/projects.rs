use crate::structures::{MainStats, LexiconStats};
use std::collections::HashMap;
use crate::engine::CueMapEngine;
use crate::normalization::NormalizationConfig;
use crate::taxonomy::Taxonomy;
use crate::config::{CueGenStrategy, TuningConfig, LlmConfig};
use crate::semantic::SemanticEngine;
use dashmap::DashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use serde_json::Value;
use ahash::RandomState;

pub struct ProjectContext {
    pub main: CueMapEngine<MainStats>,
    pub aliases: CueMapEngine<MainStats>,
    pub lexicon: CueMapEngine<LexiconStats>,
    pub query_cache: DashMap<String, Vec<String>, RandomState>,
    pub normalization: NormalizationConfig,
    pub taxonomy: Taxonomy,
    pub cuegen_strategy: CueGenStrategy,
    pub semantic_engine: SemanticEngine,
    pub last_activity: AtomicU64,
    // Shared Context (holds top 10k cues)
    pub market_heatmap: Arc<RwLock<HashMap<String, f32>>>,
    pub tuning: Arc<TuningConfig>,
    pub llm_config: Arc<LlmConfig>,
}

impl ProjectContext {
    pub fn new(normalization: NormalizationConfig, taxonomy: Taxonomy, cuegen_strategy: CueGenStrategy, semantic_engine: SemanticEngine, tuning: Arc<TuningConfig>, llm_config: Arc<LlmConfig>, config: crate::config::ServerConfig, project_id: String) -> Self {
        let mut main = CueMapEngine::with_tuning(tuning.as_ref().clone());
        main.config = config.clone();
        main.project_id = project_id.clone();
        
        let mut aliases = CueMapEngine::with_tuning(tuning.as_ref().clone());
        aliases.config = config.clone();
        aliases.config.server.store_content_on_disk = false; // Disable for aliases (tiny memories, arbitrary IDs)
        aliases.project_id = project_id.clone();
        
        let mut lexicon = CueMapEngine::with_tuning(tuning.as_ref().clone());
        lexicon.config = config.clone();
        lexicon.config.server.store_content_on_disk = false; // Disable for lexicon (tiny memories, arbitrary IDs)
        lexicon.project_id = project_id.clone();

        Self {
            main,
            aliases,
            lexicon,
            query_cache: DashMap::with_hasher(RandomState::new()),
            normalization,
            taxonomy,
            cuegen_strategy,
            semantic_engine,
            last_activity: AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            market_heatmap: Arc::new(RwLock::new(HashMap::new())),
            tuning,
            llm_config,
        }
    }
    
    pub fn touch(&self) {
        self.last_activity.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::Relaxed
        );
    }
    
    pub fn get_last_activity(&self) -> u64 {
        self.last_activity.load(Ordering::Relaxed)
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
    /// Resolve cues from text with tokenization, normalization, and validation.
    /// 
    /// When `skip_lexicon` is true, skips the lexicon lookup but still performs:
    /// - Tokenization
    /// - Normalization
    /// - Taxonomy validation
    pub fn resolve_cues_from_text(&self, text: &str, skip_lexicon: bool) -> (Vec<String>, Vec<String>, Vec<String>) {
        self.resolve_cues_from_text_with_lang(text, skip_lexicon, crate::nl::Language::Default)
    }

    pub fn resolve_cues_from_text_with_lang(&self, text: &str, skip_lexicon: bool, lang: crate::nl::Language) -> (Vec<String>, Vec<String>, Vec<String>) {
        use std::time::Instant;
        let t_start = Instant::now();
        
        let mut canonical_cues = Vec::new();
        let mut lexicon_memory_ids = Vec::new();
        let mut tokens = Vec::new();

        // 1. Symbol-First Intent Routing (Aho-Corasick + BM25)
        let symbols = self.main.get_all_symbols();
        if !symbols.is_empty() {
            let router = crate::nl::SymbolRouter::new(symbols);
            let (intent, extracted_symbols) = router.route(text);
            
            if !extracted_symbols.is_empty() && intent != crate::nl::Intent::Generic {
                canonical_cues = router.compile_to_cues(intent, extracted_symbols.clone());
                tokens = extracted_symbols;
                tracing::info!("SymbolRouter: Identified intent {:?} with symbols {:?} -> Cues: {:?}", intent, tokens, canonical_cues);
            }
        }

        // 2. Fallback to traditional tokenization and Lexicon if SymbolRouter wasn't decisive
        if canonical_cues.is_empty() {
            let t_tok = Instant::now();
            tokens = crate::nl::tokenize_to_cues_with_lang(text, lang);
            let _tok_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
            if tokens.is_empty() {
                return (Vec::new(), Vec::new(), Vec::new());
            }

            let normalized_text = if lang == crate::nl::Language::Default {
                crate::nl::normalize_text(text)
            } else {
                format!("{:?}:{}", lang, crate::nl::normalize_text(text))
            };
            
            // Check cache
            if !skip_lexicon {
                if let Some(cues) = self.query_cache.get(&normalized_text) {
                    return (cues.clone(), Vec::new(), tokens);
                }
            }
            
            let t_lex = Instant::now();
            if skip_lexicon {
                for token in &tokens {
                    let (normalized, _) = crate::normalization::normalize_cue(token, &self.normalization);
                    if !canonical_cues.contains(&normalized) {
                        canonical_cues.push(normalized);
                    }
                }
            } else {
                let lexicon_results = self.lexicon.recall_fast(tokens.clone(), 64);
                for result in lexicon_results {
                    let (normalized, _) = crate::normalization::normalize_cue(&result.content, &self.normalization);
                    canonical_cues.push(normalized);
                    lexicon_memory_ids.push(result.memory_id.clone());
                }
                
                if canonical_cues.is_empty() {
                    for token in &tokens {
                        let (normalized, _) = crate::normalization::normalize_cue(token, &self.normalization);
                        if !canonical_cues.contains(&normalized) {
                            canonical_cues.push(normalized);
                        }
                    }
                }
            }
            let _lex_ms = t_lex.elapsed().as_secs_f64() * 1000.0;
            
            // Cache (only if lexicon was used)
            if !skip_lexicon {
                self.query_cache.insert(normalized_text, canonical_cues.clone());
            }
        }
        
        // 3. Validate list
        let t_val = Instant::now();
        let report = crate::taxonomy::validate_cues(canonical_cues, &self.taxonomy);
        let accepted = report.accepted;
        let _val_ms = t_val.elapsed().as_secs_f64() * 1000.0;
        
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        
        if total_ms > 1.0 {
            tracing::debug!(
                "resolve_cues_from_text: total={:.2}ms skip_lexicon={}",
                total_ms, skip_lexicon
            );
        }
        
        (accepted, lexicon_memory_ids, tokens)
    }
    
    pub fn expand_query_cues(&self, cues: Vec<String>, original_tokens: &[String]) -> Vec<(String, f64)> {
        let mut expanded: Vec<(String, f64)> = Vec::new();
        
        for cue in cues {
            // 1. Add original cue with weight 1.0
            expanded.push((cue.clone(), 1.0));
            
            // 2. ONLY expand aliases for original tokens (not Lexicon synonyms)
            if !original_tokens.contains(&cue) {
                continue;
            }
            
            // 2. Query aliases
            let alias_query = vec![
                "type:alias".to_string(),
                format!("from:{}", cue),
                "status:active".to_string(),
            ];
            
            // Recall aliases (limit 8, auto_reinforce false to avoid noise, no heatmap)
            let aliases = self.aliases.recall(alias_query, 8, false, None);
            
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
                         
                         // The "to" field in content is the actual cue
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
                // Only keep cues that exist in the index.
                self.main.get_cue_index().contains_key(cue) && seen.insert(cue.clone())
            })
            .collect()
    }
}

pub struct ProjectStore {
    pub projects: DashMap<String, Arc<ProjectContext>, RandomState>,
}

impl ProjectStore {
    pub fn new() -> Self {
        Self {
            projects: DashMap::with_hasher(RandomState::new()),
        }
    }

    pub fn get_or_create(&self, project_id: &str) -> Arc<ProjectContext> {
        if let Some(ctx) = self.projects.get(project_id) {
            return ctx.clone();
        }

        // Create new project with default config
        let ctx = Arc::new(ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            CueGenStrategy::default(),
            SemanticEngine::new(None),
            Arc::new(TuningConfig::default()),
            Arc::new(LlmConfig::default()),
            crate::config::ServerConfig::default(),
            project_id.to_string(),
        ));

        self.projects.insert(project_id.to_string(), ctx.clone());
        ctx
    }
}

