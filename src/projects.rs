use crate::config::TuningConfig;
use crate::cuebridge::{CueBridgeArtifactSummary, CueBridgeArtifacts, CueBridgeAliasExpansion};
use crate::engine::CueMapEngine;
use crate::normalization::NormalizationConfig;
use crate::structures::{LexiconStats, MainStats, MemoryId};
use crate::taxonomy::Taxonomy;
use ahash::RandomState;
use dashmap::DashMap;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) struct SymbolRouterCache {
    cue_index_version: usize,
    symbol_count: usize,
    router: Option<crate::nl::SymbolRouter>,
}

impl Default for SymbolRouterCache {
    fn default() -> Self {
        Self {
            cue_index_version: 0,
            symbol_count: 0,
            router: None,
        }
    }
}

pub struct ProjectContext {
    pub main: CueMapEngine<MainStats>,
    pub aliases: CueMapEngine<MainStats>,
    pub lexicon: CueMapEngine<LexiconStats>,
    pub query_cache: DashMap<String, Vec<String>, RandomState>,
    pub(crate) symbol_router_cache: RwLock<SymbolRouterCache>,
    pub normalization: NormalizationConfig,
    pub taxonomy: Taxonomy,
    pub last_activity: AtomicU64,
    // Shared Context (holds top 10k cues)
    pub market_heatmap: Arc<RwLock<HashMap<String, f32>>>,
    pub tuning: Arc<TuningConfig>,
    pub cuebridge_artifacts: RwLock<CueBridgeArtifacts>,
}

impl ProjectContext {
    pub fn new(
        normalization: NormalizationConfig,
        taxonomy: Taxonomy,
        tuning: Arc<TuningConfig>,
        config: crate::config::ServerConfig,
        project_id: String,
    ) -> Self {
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

        let cuebridge_artifacts =
            CueBridgeArtifacts::load_for_project(&config.server.data_dir, &project_id);

        Self {
            main,
            aliases,
            lexicon,
            query_cache: DashMap::with_hasher(RandomState::new()),
            symbol_router_cache: RwLock::new(SymbolRouterCache::default()),
            normalization,
            taxonomy,
            last_activity: AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            ),
            market_heatmap: Arc::new(RwLock::new(HashMap::new())),
            tuning,
            cuebridge_artifacts: RwLock::new(cuebridge_artifacts),
        }
    }

    pub fn touch(&self) {
        self.last_activity.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::Relaxed,
        );
    }

    pub fn get_last_activity(&self) -> u64 {
        self.last_activity.load(Ordering::Relaxed)
    }

    pub fn reload_cuebridge_artifacts(
        &self,
        data_dir: &str,
        project_id: &str,
    ) -> CueBridgeArtifactSummary {
        let artifacts = CueBridgeArtifacts::load_for_project(data_dir, project_id);
        let summary = artifacts.summary();
        if let Ok(mut guard) = self.cuebridge_artifacts.write() {
            *guard = artifacts;
        }
        summary
    }

    pub fn cuebridge_artifact_summary(&self) -> CueBridgeArtifactSummary {
        self.cuebridge_artifacts
            .read()
            .map(|artifacts| artifacts.summary())
            .unwrap_or_default()
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
    pub fn resolve_cues_from_text(
        &self,
        text: &str,
        skip_lexicon: bool,
    ) -> (Vec<String>, Vec<MemoryId>, Vec<String>) {
        self.resolve_cues_from_text_with_lang(text, skip_lexicon, crate::nl::Language::Default)
    }

    pub fn resolve_cues_from_text_with_lang(
        &self,
        text: &str,
        skip_lexicon: bool,
        lang: crate::nl::Language,
    ) -> (Vec<String>, Vec<MemoryId>, Vec<String>) {
        use std::time::Instant;
        let t_start = Instant::now();

        let mut canonical_cues = Vec::new();
        let mut lexicon_memory_ids = Vec::new();
        let mut tokens = Vec::new();

        // 1. Symbol-First Intent Routing (Aho-Corasick + BM25)
        if let Some((routed_cues, routed_tokens)) = self.route_symbol_query(text) {
            canonical_cues = routed_cues;
            tokens = routed_tokens;
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
                    let (normalized, _) =
                        crate::normalization::normalize_cue(token, &self.normalization);
                    if !canonical_cues.contains(&normalized) {
                        canonical_cues.push(normalized);
                    }
                }
            } else {
                let lexicon_results = self.lexicon.recall_fast(tokens.clone(), 64);
                for result in lexicon_results {
                    let (normalized, _) =
                        crate::normalization::normalize_cue(&result.content, &self.normalization);
                    canonical_cues.push(normalized);
                    lexicon_memory_ids.push(result.memory_id);
                }

                if canonical_cues.is_empty() {
                    for token in &tokens {
                        let (normalized, _) =
                            crate::normalization::normalize_cue(token, &self.normalization);
                        if !canonical_cues.contains(&normalized) {
                            canonical_cues.push(normalized);
                        }
                    }
                }
            }
            let _lex_ms = t_lex.elapsed().as_secs_f64() * 1000.0;

            // Cache (only if lexicon was used)
            if !skip_lexicon {
                self.query_cache
                    .insert(normalized_text, canonical_cues.clone());
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
                total_ms,
                skip_lexicon
            );
        }

        (accepted, lexicon_memory_ids, tokens)
    }

    fn route_symbol_query(&self, text: &str) -> Option<(Vec<String>, Vec<String>)> {
        let cue_index_version = self.main.cue_index_version();

        if let Ok(cache) = self.symbol_router_cache.read() {
            if cache.cue_index_version == cue_index_version {
                return Self::route_with_symbol_cache(text, &cache);
            }
        }

        let mut cache = self.symbol_router_cache.write().ok()?;
        if cache.cue_index_version != cue_index_version {
            let symbols = self.main.get_all_symbols();
            cache.symbol_count = symbols.len();
            cache.router = if symbols.is_empty() {
                None
            } else {
                Some(crate::nl::SymbolRouter::new(symbols))
            };
            cache.cue_index_version = cue_index_version;
        }

        Self::route_with_symbol_cache(text, &cache)
    }

    fn route_with_symbol_cache(
        text: &str,
        cache: &SymbolRouterCache,
    ) -> Option<(Vec<String>, Vec<String>)> {
        if cache.symbol_count == 0 {
            return None;
        }
        let router = cache.router.as_ref()?;
        let (intent, extracted_symbols) = router.route(text);

        if extracted_symbols.is_empty() || intent == crate::nl::Intent::Generic {
            return None;
        }

        let canonical_cues = router.compile_to_cues(intent, extracted_symbols.clone());
        tracing::debug!(
            "SymbolRouter: Identified intent {:?} with symbols {:?} -> Cues: {:?}",
            intent,
            extracted_symbols,
            canonical_cues
        );
        Some((canonical_cues, extracted_symbols))
    }

    pub fn expand_query_cues(
        &self,
        cues: Vec<String>,
        original_tokens: &[String],
    ) -> Vec<(String, f64)> {
        self.expand_query_cues_with_trace(cues, original_tokens).0
    }

    pub fn expand_query_cues_with_trace(
        &self,
        cues: Vec<String>,
        original_tokens: &[String],
    ) -> (Vec<(String, f64)>, Vec<CueBridgeAliasExpansion>) {
        let mut expanded: Vec<(String, f64)> = Vec::new();
        let mut cuebridge_aliases = Vec::new();
        let all_query_cues = cues.clone();

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
                        let downweight = data
                            .get("downweight")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.85);

                        // The "to" field in content is the actual cue
                        expanded.push((to_cue.to_string(), downweight));
                    }
                }
            }

            if let Ok(artifacts) = self.cuebridge_artifacts.read() {
                for alias in artifacts.alias_expansions(&cue, &all_query_cues, |target| {
                    self.main.get_cue_index().contains_key(target)
                }) {
                    expanded.push((alias.to.clone(), alias.weight * alias.confidence.max(0.1)));
                    cuebridge_aliases.push(alias);
                }
            }
        }

        // Deduplicate
        let mut seen = std::collections::HashSet::new();
        expanded.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let deduped = expanded
            .into_iter()
            .filter(|(cue, _)| {
                // Only keep cues that exist in the index.
                self.main.get_cue_index().contains_key(cue) && seen.insert(cue.clone())
            })
            .collect::<Vec<_>>();
        (deduped, cuebridge_aliases)
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
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            project_id.to_string(),
        ));

        self.projects.insert(project_id.to_string(), ctx.clone());
        ctx
    }
}
