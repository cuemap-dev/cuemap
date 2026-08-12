use crate::config::TuningConfig;
use crate::crypto::EncryptionKey;
use crate::intent::{
    intent_compatibility, IntentClassification, IntentClassifier, IntentTarget,
    INTENT_TAXONOMY_VERSION,
};
use crate::semantic::{LinearReranker, SemanticEncoder, SemanticIndex, StoredSemanticVector};
use crate::structures::{
    LexiconStats, MainStats, Memory, MemoryId, MemoryScoringFeatures, MemoryStats, OrderedSet,
    INVALID_MEMORY_ID,
};
use ahash::RandomState;
use dashmap::DashMap;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::num::NonZeroUsize;

const MEMORY_SCORING_FEATURES_VERSION: u8 = 1;
const FAMILY_PERSON: u64 = 1 << 0;
const FAMILY_QUANTITY: u64 = 1 << 1;
const FAMILY_INVENTORY: u64 = 1 << 2;
const FAMILY_TRAVEL: u64 = 1 << 3;
const FAMILY_AGE: u64 = 1 << 4;
const FAMILY_EDUCATION: u64 = 1 << 5;
const FAMILY_FAMILY: u64 = 1 << 6;
const FAMILY_FAMILY_COUNT: u64 = 1 << 7;
const FAMILY_SOURCE_ROLE: u64 = 1 << 8;
const FAMILY_SOURCE_TIME: u64 = 1 << 9;
const FAMILY_UPDATE: u64 = 1 << 10;
const FAMILY_NON_SOURCE_STRUCTURED: u64 = 1 << 11;
const FAMILY_STRUCTURED: u64 = 1 << 12;

#[derive(Debug, Clone, Serialize)]
pub struct RecallResult {
    pub memory_id: MemoryId,
    pub content: String,
    pub score: f64,
    pub match_integrity: f64,
    pub intersection_count: usize,
    pub recency_score: f64,
    pub reinforcement_score: f64,
    pub salience_score: f64,
    pub created_at: f64, // Timestamp when memory was created
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explain: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RecallTimingBreakdown {
    pub total_ms: f64,
    pub normalize_filter_ms: f64,
    pub consolidated_search_ms: f64,
    pub candidate_generation_ms: f64,
    pub candidate_scoring_ms: f64,
    pub scoring_filter_ms: f64,
    pub scoring_position_ms: f64,
    pub scoring_salience_ms: f64,
    pub scoring_structured_ms: f64,
    pub scoring_finalize_ms: f64,
    pub min_intersection_ms: f64,
    pub auto_reinforce_ms: f64,
    pub sort_truncate_ms: f64,
    pub materialize_ms: f64,
    pub active_cue_count: usize,
    pub initial_active_cue_count: usize,
    pub candidate_count: usize,
    pub scored_candidate_count: usize,
    pub returned_count: usize,
    pub cue_count_with_postings: usize,
    pub scanned_posting_count: usize,
    pub adaptive_scan_limit: usize,
    pub max_posting_len: usize,
    pub semantic_rerank_candidate_limit: usize,
    pub semantic_rerank_candidate_count: usize,
}

#[derive(Debug, Clone, Default)]
struct ConsolidatedSearchTiming {
    candidate_generation_ms: f64,
    candidate_scoring_ms: f64,
    scoring_filter_ms: f64,
    scoring_position_ms: f64,
    scoring_salience_ms: f64,
    scoring_structured_ms: f64,
    scoring_finalize_ms: f64,
    candidate_count: usize,
    scored_candidate_count: usize,
    cue_count_with_postings: usize,
    scanned_posting_count: usize,
    adaptive_scan_limit: usize,
    max_posting_len: usize,
}

#[derive(Debug, Clone, Default)]
struct CandidateScoringTiming {
    filter_ms: f64,
    position_ms: f64,
    salience_ms: f64,
    structured_ms: f64,
    finalize_ms: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceOrderEntry {
    pub order: i64,
    pub memory_id: MemoryId,
}

#[derive(Debug, Clone)]
pub struct ScoredMemoryCandidate {
    pub memory_id: MemoryId,
    pub score: f64,
    pub match_integrity: f64,
    pub intersection_count: usize,
    pub recency_score: f64,
    pub reinforcement_score: f64,
    pub salience_score: f64,
    pub created_at: f64,
    // Raw values for late materialization
    pub intersection_weighted: f64,
    pub match_count: f64,
    pub rerank_bonus: f64,
    pub generic_penalty: f64,
    pub semantic_similarity: f32,
    pub intent_compatibility: f64,
    pub intent_rerank_bonus: f64,
}

struct QueryScoringCue<'a> {
    cue: &'a str,
    weight: f64,
    generated: bool,
    semantic_generated: bool,
    prefix: Option<&'a str>,
    family_mask: u64,
    is_family_prefix: bool,
    is_source_time_prefix: bool,
    rerank_multiplier: f64,
}

struct QueryScoringProfile<'a> {
    cues: Vec<QueryScoringCue<'a>>,
    lexical_query_seen: bool,
    strong_lexical_query_seen: bool,
    strong_structured_query_seen: bool,
    structured_query_seen: bool,
    person_structured_query_seen: bool,
    quantity_structured_query_seen: bool,
    inventory_structured_query_seen: bool,
    travel_structured_query_seen: bool,
    age_structured_query_seen: bool,
    education_structured_query_seen: bool,
    family_structured_query_seen: bool,
    family_count_structured_query_seen: bool,
    source_role_structured_query_seen: bool,
    source_time_structured_query_seen: bool,
    update_structured_query_seen: bool,
}

fn is_generated_memory_facet(cue: &str) -> bool {
    if cue.starts_with("source_role:")
        || cue.starts_with("source_channel:")
        || cue.starts_with("source_type:")
        || cue.starts_with("source_session:")
        || cue.starts_with("source_time:")
        || cue.starts_with("source_date:")
        || cue.starts_with("source_week:")
        || cue.starts_with("source_month:")
        || cue.starts_with("has:")
        || cue.starts_with("completion_count:")
        || cue.starts_with("completed_action:")
        || cue.starts_with("instruction:")
        || cue.starts_with("instruction_trigger:")
        || cue.starts_with("instruction_action:")
        || cue.starts_with("preference:")
        || cue.starts_with("preference_value:")
        || cue.starts_with("preference_topic:")
        || cue.starts_with("preference_contrast:")
        || cue.starts_with("temporal:")
        || cue.starts_with("co_residence:")
        || cue.starts_with("entity:")
        || cue.starts_with("person_title:")
        || cue.starts_with("person_role_phrase:")
        || cue.starts_with("person_ref:")
        || cue.starts_with("quantity_object:")
        || cue.starts_with("quantity_unit:")
        || cue.starts_with("quantity_unit_object:")
        || cue.starts_with("quantity_count:")
        || cue.starts_with("inventory_object:")
        || cue.starts_with("inventory_count:")
        || cue.starts_with("purchase:")
        || cue.starts_with("companion:")
        || cue.starts_with("age:")
        || cue.starts_with("education:")
        || cue.starts_with("travel:")
        || cue.starts_with("media:")
        || cue.starts_with("reading:")
        || cue.starts_with("transport_mode:")
        || cue.starts_with("transport_event:")
        || cue.starts_with("activity_domain:")
        || cue.starts_with("topic:")
        || cue.starts_with("attribute:")
        || cue.starts_with("family_relation:")
        || cue.starts_with("family_scope:")
        || cue.starts_with("family_count:")
        || cue.starts_with("sibling_kind:")
    {
        return true;
    }

    matches!(
        cue,
        "type:preference"
            | "type:dislike"
            | "type:ownership"
            | "type:recommendation"
            | "type:recipe"
            | "type:answer"
            | "type:routine"
            | "type:update"
            | "type:navigation"
            | "type:activity"
            | "type:event"
            | "type:milestone"
            | "type:decision"
            | "type:selection"
            | "type:naming"
            | "type:entity_attribute"
            | "type:expertise"
            | "type:interest"
            | "type:ingredient"
            | "type:homegrown"
            | "type:usage"
            | "type:standing_instruction"
    )
}

fn cue_structured_family_mask(cue: &str) -> u64 {
    let Some((prefix, _)) = cue.split_once(':') else {
        return 0;
    };

    let mut mask = FAMILY_STRUCTURED;
    if prefix.starts_with("person_") {
        mask |= FAMILY_PERSON;
    }
    if prefix.starts_with("quantity_") {
        mask |= FAMILY_QUANTITY;
    }
    if prefix.starts_with("inventory_") {
        mask |= FAMILY_INVENTORY;
    }
    if prefix == "travel" {
        mask |= FAMILY_TRAVEL;
    }
    if prefix == "age" || cue == "has:age" {
        mask |= FAMILY_AGE;
    }
    if prefix == "education" {
        mask |= FAMILY_EDUCATION;
    }
    if matches!(
        prefix,
        "family_relation" | "family_scope" | "family_count" | "sibling_kind"
    ) {
        mask |= FAMILY_FAMILY;
    }
    if prefix == "family_count" {
        mask |= FAMILY_FAMILY_COUNT;
    }
    if prefix == "source_role" {
        mask |= FAMILY_SOURCE_ROLE;
    }
    let is_source_time_prefix = matches!(
        prefix,
        "source_time" | "source_date" | "source_week" | "source_month" | "source_year"
    );
    if is_source_time_prefix {
        mask |= FAMILY_SOURCE_TIME;
    } else {
        mask |= FAMILY_NON_SOURCE_STRUCTURED;
    }
    if cue == "type:update" {
        mask |= FAMILY_UPDATE;
    }

    mask
}

fn compute_memory_scoring_features(cues: &[String]) -> MemoryScoringFeatures {
    let mut features = MemoryScoringFeatures {
        version: MEMORY_SCORING_FEATURES_VERSION,
        scored_cue_len: 0,
        has_summary_type: false,
        structured_family_mask: 0,
    };

    for cue in cues {
        if cue == "type:summary" {
            features.has_summary_type = true;
        }
        if !is_generated_memory_facet(cue) {
            features.scored_cue_len += 1;
        }
        features.structured_family_mask |= cue_structured_family_mask(cue);
    }
    features.scored_cue_len = features.scored_cue_len.max(1);
    features
}

fn memory_scoring_features<T>(memory: &Memory<T>) -> MemoryScoringFeatures {
    if memory.scoring_features.version == MEMORY_SCORING_FEATURES_VERSION {
        memory.scoring_features.clone()
    } else {
        compute_memory_scoring_features(&memory.cues)
    }
}

fn rerank_multiplier_for_prefix(prefix: &str) -> f64 {
    match prefix {
        "source_role" | "source_channel" | "source_type" | "source_time" | "source_date"
        | "source_week" | "source_month" | "source_year" => 12.0,
        "type" => 10.0,
        "has" => 9.0,
        "temporal" => 8.0,
        "entity" => 7.0,
        "person_title" | "person_role_phrase" | "person_ref" => 8.0,
        "quantity_object" | "quantity_unit" | "quantity_unit_object" => 10.0,
        "completion_count" => 12.0,
        "completed_action" => 10.0,
        "instruction" | "instruction_trigger" | "instruction_action" => 10.0,
        "preference" | "preference_value" | "preference_topic" | "preference_contrast" => 10.0,
        "companion" => 10.0,
        "co_residence" => 10.0,
        "inventory_object" => 10.0,
        "purchase" => 10.0,
        "travel" => 9.0,
        "age" => 10.0,
        "education" => 9.0,
        "family_count" => 12.0,
        "family_scope" => 10.0,
        "family_relation" | "sibling_kind" => 9.0,
        _ => 4.0,
    }
}

fn build_query_scoring_profile(query_cues: &[(String, f64)]) -> QueryScoringProfile<'_> {
    let mut profile = QueryScoringProfile {
        cues: Vec::with_capacity(query_cues.len()),
        lexical_query_seen: false,
        strong_lexical_query_seen: false,
        strong_structured_query_seen: false,
        structured_query_seen: false,
        person_structured_query_seen: false,
        quantity_structured_query_seen: false,
        inventory_structured_query_seen: false,
        travel_structured_query_seen: false,
        age_structured_query_seen: false,
        education_structured_query_seen: false,
        family_structured_query_seen: false,
        family_count_structured_query_seen: false,
        source_role_structured_query_seen: false,
        source_time_structured_query_seen: false,
        update_structured_query_seen: false,
    };

    for (cue, weight) in query_cues {
        let generated = is_generated_memory_facet(cue);
        let semantic_generated = is_semantic_generated_facet(cue);
        let prefix = cue.split_once(':').map(|(prefix, _)| prefix);
        let family_mask = cue_structured_family_mask(cue);
        let is_family_prefix = prefix.is_some_and(|prefix| {
            matches!(
                prefix,
                "family_relation" | "family_scope" | "family_count" | "sibling_kind"
            )
        });
        let is_source_time_prefix = prefix.is_some_and(|prefix| {
            matches!(
                prefix,
                "source_time" | "source_date" | "source_week" | "source_month" | "source_year"
            )
        });
        let rerank_multiplier = prefix.map(rerank_multiplier_for_prefix).unwrap_or(0.0);

        if !generated {
            profile.lexical_query_seen = true;
            if *weight >= 0.9 {
                profile.strong_lexical_query_seen = true;
            }
        }
        if semantic_generated && *weight >= 3.0 {
            profile.strong_structured_query_seen = true;
        }
        if prefix.is_some() {
            profile.structured_query_seen = true;
        }
        if family_mask & FAMILY_PERSON != 0 {
            profile.person_structured_query_seen = true;
        }
        if family_mask & FAMILY_QUANTITY != 0 {
            profile.quantity_structured_query_seen = true;
        }
        if family_mask & FAMILY_INVENTORY != 0 {
            profile.inventory_structured_query_seen = true;
        }
        if family_mask & FAMILY_TRAVEL != 0 {
            profile.travel_structured_query_seen = true;
        }
        if family_mask & FAMILY_AGE != 0 {
            profile.age_structured_query_seen = true;
        }
        if family_mask & FAMILY_EDUCATION != 0 {
            profile.education_structured_query_seen = true;
        }
        if family_mask & FAMILY_FAMILY != 0 {
            profile.family_structured_query_seen = true;
        }
        if family_mask & FAMILY_FAMILY_COUNT != 0 {
            profile.family_count_structured_query_seen = true;
        }
        if prefix == Some("source_role") && *weight >= 3.0 {
            profile.source_role_structured_query_seen = true;
        }
        if family_mask & FAMILY_SOURCE_TIME != 0 {
            profile.source_time_structured_query_seen = true;
        }
        if cue == "type:update" {
            profile.update_structured_query_seen = true;
        }

        profile.cues.push(QueryScoringCue {
            cue,
            weight: *weight,
            generated,
            semantic_generated,
            prefix,
            family_mask,
            is_family_prefix,
            is_source_time_prefix,
            rerank_multiplier,
        });
    }

    profile
}

fn is_semantic_generated_facet(cue: &str) -> bool {
    cue.starts_with("type:")
        || cue.starts_with("media:")
        || cue.starts_with("reading:")
        || cue.starts_with("travel:")
        || cue.starts_with("transport_mode:")
        || cue.starts_with("transport_event:")
        || cue.starts_with("activity_domain:")
        || cue.starts_with("topic:")
        || cue.starts_with("purchase:")
        || cue.starts_with("completion_count:")
        || cue.starts_with("completed_action:")
        || cue.starts_with("instruction:")
        || cue.starts_with("instruction_trigger:")
        || cue.starts_with("instruction_action:")
        || cue.starts_with("preference:")
        || cue.starts_with("preference_value:")
        || cue.starts_with("preference_topic:")
        || cue.starts_with("preference_contrast:")
        || cue.starts_with("companion:")
        || cue.starts_with("co_residence:")
}

fn should_index_value_alias(cue: &str) -> bool {
    !is_generated_memory_facet(cue)
}

#[derive(Clone)]
pub struct CueMapEngine<T>
where
    T: Serialize
        + for<'de> Deserialize<'de>
        + Clone
        + Default
        + Send
        + Sync
        + MemoryStats
        + 'static,
{
    memories: Arc<DashMap<MemoryId, Memory<T>, RandomState>>,
    source_key_to_id: Arc<DashMap<String, MemoryId, RandomState>>,
    cue_index: Arc<DashMap<String, OrderedSet, RandomState>>,
    source_order_index: Arc<DashMap<String, Vec<SourceOrderEntry>, RandomState>>,
    // Temporal Chunking: track last event per session/project
    last_events: Arc<DashMap<String, (MemoryId, f64, Vec<String>), RandomState>>,

    memory_count: Arc<AtomicUsize>,
    cue_count: Arc<AtomicUsize>,
    next_memory_id: Arc<AtomicU32>,
    master_key: Option<Arc<EncryptionKey>>,
    tuning: Arc<TuningConfig>,
    /// Global occurrences of each cue (for IDF weighting)
    pub cue_global_counts: Arc<DashMap<String, u64, RandomState>>,

    /// Optional vector index. The index is rebuilt from persisted vectors and
    /// remains empty unless semantic retrieval is explicitly enabled.
    semantic_index: Arc<RwLock<SemanticIndex>>,
    /// Optional local text encoder. It is loaded only when the binary includes
    /// the encoder feature and the project explicitly enables it in config.
    semantic_encoder: Arc<RwLock<Option<Arc<dyn SemanticEncoder>>>>,
    /// Intent classifier built from the configured local encoder and the
    /// versioned CueKey taxonomy.
    intent_classifier: Arc<RwLock<Option<Arc<IntentClassifier>>>>,
    /// Bounded cache for repeated query text embeddings. The lock is held
    /// only while accessing the cache, never while running the encoder.
    query_embedding_cache: Arc<Mutex<LruCache<String, Vec<f32>>>>,

    // Storage context
    pub config: crate::config::ServerConfig,
    pub project_id: String,
}

impl<T> CueMapEngine<T>
where
    T: Serialize
        + for<'de> Deserialize<'de>
        + Clone
        + Default
        + Send
        + Sync
        + MemoryStats
        + 'static,
{
    pub fn new() -> Self {
        Self {
            memories: Arc::new(DashMap::with_hasher(RandomState::new())),
            source_key_to_id: Arc::new(DashMap::with_hasher(RandomState::new())),
            cue_index: Arc::new(DashMap::with_hasher(RandomState::new())),
            source_order_index: Arc::new(DashMap::with_hasher(RandomState::new())),
            last_events: Arc::new(DashMap::with_hasher(RandomState::new())),
            memory_count: Arc::new(AtomicUsize::new(0)),
            cue_count: Arc::new(AtomicUsize::new(0)),
            next_memory_id: Arc::new(AtomicU32::new(1)),
            master_key: None,
            tuning: Arc::new(TuningConfig::default()),
            cue_global_counts: Arc::new(DashMap::with_hasher(RandomState::new())),
            semantic_index: Arc::new(RwLock::new(SemanticIndex::new(
                crate::semantic::SemanticConfig::default(),
            ))),
            semantic_encoder: Arc::new(RwLock::new(None)),
            intent_classifier: Arc::new(RwLock::new(None)),
            query_embedding_cache: Self::new_query_embedding_cache(
                &crate::semantic::SemanticConfig::default(),
            ),
            config: crate::config::ServerConfig::default(),
            project_id: "default".to_string(),
        }
    }

    pub fn with_tuning(tuning: TuningConfig) -> Self {
        let mut engine = Self::new();
        engine.tuning = Arc::new(tuning);
        engine
    }

    pub fn with_key(key: Option<EncryptionKey>) -> Self {
        let mut engine = Self::new();
        engine.master_key = key.map(Arc::new);
        engine
    }

    pub fn set_master_key(&mut self, key: Option<Arc<EncryptionKey>>) {
        self.master_key = key;
    }

    pub fn set_tuning_config(&mut self, tuning: TuningConfig) {
        self.tuning = Arc::new(tuning);
    }

    pub fn set_semantic_config(&mut self, config: crate::semantic::SemanticConfig) {
        let config = config.resolved();
        self.config.semantic = config.clone();
        self.query_embedding_cache = Self::new_query_embedding_cache(&config);
        let mut index = SemanticIndex::new(config);
        index.rebuild(self.memories.iter().filter_map(|entry| {
            entry
                .semantic_vector
                .as_ref()
                .map(|vector| (entry.id, vector.clone()))
        }));
        self.semantic_index = Arc::new(RwLock::new(index));
    }

    pub fn set_semantic_encoder(&mut self, encoder: Option<Arc<dyn SemanticEncoder>>) {
        let classifier = encoder.as_ref().and_then(|encoder| {
            match IntentClassifier::new(
                encoder.clone(),
                self.config.semantic.model_version.clone(),
            ) {
                Ok(classifier) => Some(Arc::new(classifier)),
                Err(error) => {
                    tracing::warn!(error = %error, "Intent classifier unavailable");
                    None
                }
            }
        });
        self.semantic_encoder = Arc::new(RwLock::new(encoder));
        self.intent_classifier = Arc::new(RwLock::new(classifier));
        if let Ok(mut cache) = self.query_embedding_cache.lock() {
            cache.clear();
        }
    }

    pub fn configure_semantic_encoder(&mut self) -> Result<(), String> {
        let encoder = crate::semantic::load_configured_encoder(&self.config.semantic)?;
        self.set_semantic_encoder(encoder);
        Ok(())
    }

    fn new_query_embedding_cache(
        config: &crate::semantic::SemanticConfig,
    ) -> Arc<Mutex<LruCache<String, Vec<f32>>>> {
        let capacity = NonZeroUsize::new(config.resolved().query_embedding_cache_capacity.max(1))
            .expect("query embedding cache capacity is always non-zero");
        Arc::new(Mutex::new(LruCache::new(capacity)))
    }

    /// Encode text with the bundled local encoder when enabled. Callers can
    /// still provide an embedding directly for a single memory/query, and
    /// can disable automatic encoding through `SemanticConfig`.
    pub fn encode_semantic_text(&self, text: &str) -> Option<Vec<f32>> {
        let config = self.config.semantic.resolved();
        if !config.enabled || !config.encoder_enabled {
            return None;
        }
        let cache_enabled = config.query_embedding_cache_capacity > 0;
        if cache_enabled {
            if let Ok(mut cache) = self.query_embedding_cache.lock() {
                if let Some(vector) = cache.get(text) {
                    return Some(vector.clone());
                }
            }
        }
        let encoder = self.semantic_encoder.read().ok()?.clone()?;
        let vector = match encoder.encode(text) {
            Ok(vector) => vector,
            Err(error) => {
                tracing::debug!(error = %error, "Semantic text encoding skipped");
                return None;
            }
        };
        if vector.len() != encoder.dimensions() {
            tracing::debug!(
                expected = encoder.dimensions(),
                received = vector.len(),
                "Semantic text encoder returned incompatible dimensions"
            );
            return None;
        }
        if cache_enabled {
            if let Ok(mut cache) = self.query_embedding_cache.lock() {
                cache.put(text.to_owned(), vector.clone());
            }
        }
        Some(vector)
    }

    pub fn classify_intent(
        &self,
        text: &str,
        target: IntentTarget,
    ) -> Result<IntentClassification, String> {
        let classifier = self
            .intent_classifier
            .read()
            .map_err(|_| "intent classifier lock poisoned".to_string())?
            .clone()
            .ok_or_else(|| "intent classifier unavailable".to_string())?;
        classifier.classify(text, target)
    }

    pub fn classify_intent_with_embedding(
        &self,
        text: &str,
        target: IntentTarget,
        embedding: &[f32],
    ) -> Result<IntentClassification, String> {
        let classifier = self
            .intent_classifier
            .read()
            .map_err(|_| "intent classifier lock poisoned".to_string())?
            .clone()
            .ok_or_else(|| "intent classifier unavailable".to_string())?;
        classifier.classify_with_embedding(text, embedding, target)
    }

    pub fn attach_intent_classification(
        &self,
        memory_id: MemoryId,
        classification: IntentClassification,
    ) -> bool {
        if let Some(mut memory) = self.memories.get_mut(&memory_id) {
            memory.intent_classification = Some(classification);
            true
        } else {
            false
        }
    }

    /// Return `(total, annotated, missing_current_version, stale_version)` for
    /// memory intent metadata. A stale annotation is treated as missing for
    /// readiness because changing the model or taxonomy invalidates its score.
    pub fn intent_coverage(&self) -> (usize, usize, usize, usize) {
        let expected_model_version = self.config.semantic.model_version.as_str();
        let mut total = 0;
        let mut annotated = 0;
        let mut current = 0;
        let mut stale = 0;

        for entry in self.memories.iter() {
            total += 1;
            match entry.intent_classification.as_ref() {
                Some(classification) => {
                    annotated += 1;
                    if classification.taxonomy_version == INTENT_TAXONOMY_VERSION
                        && classification.model_version == expected_model_version
                    {
                        current += 1;
                    } else {
                        stale += 1;
                    }
                }
                None => {}
            }
        }

        (total, annotated, total.saturating_sub(current), stale)
    }

    pub fn semantic_index_stats(&self) -> (bool, usize, Option<usize>) {
        self.semantic_index
            .read()
            .map(|index| {
                (
                    index.config().enabled,
                    index.len(),
                    index.dimensions(),
                )
            })
            .unwrap_or((false, 0, None))
    }

    fn prepare_semantic_vector(&self, vector: Option<Vec<f32>>) -> Option<StoredSemanticVector> {
        let vector = vector?;
        let config = self.config.semantic.resolved();
        if !config.enabled {
            return None;
        }
        if config.dimensions != 0 && vector.len() != config.dimensions {
            tracing::debug!(
                expected = config.dimensions,
                received = vector.len(),
                "Skipping semantic vector with incompatible dimensions"
            );
            return None;
        }
        let memory_count = self.memory_count.load(Ordering::Relaxed).saturating_add(1);
        if !config.within_memory_budget_for_dimensions(vector.len(), memory_count) {
            tracing::debug!(
                memory_count,
                max_memory_mb = config.max_memory_mb,
                "Skipping semantic vector because the configured memory budget is full"
            );
            return None;
        }
        match StoredSemanticVector::from_f32(&vector, config.storage) {
            Ok(vector) => Some(vector),
            Err(error) => {
                tracing::debug!(error = %error, "Skipping invalid semantic vector");
                None
            }
        }
    }

    pub fn get_master_key(&self) -> Option<Arc<EncryptionKey>> {
        self.master_key.clone()
    }

    pub fn from_state(
        memories: DashMap<MemoryId, Memory<T>, RandomState>,
        source_key_to_id: DashMap<String, MemoryId, RandomState>,
        cue_index: DashMap<String, OrderedSet, RandomState>,
        next_memory_id: MemoryId,
        loaded_global_counts: Option<DashMap<String, u64, RandomState>>,
        mut config: crate::config::ServerConfig,
        project_id: String,
    ) -> Self {
        config.semantic = config.semantic.resolved();
        let cue_global_counts = loaded_global_counts
            .map(Arc::new)
            .unwrap_or_else(|| Arc::new(DashMap::with_hasher(RandomState::new())));

        let count = memories.len();
        let cue_count = cue_index.len();
        let engine = Self {
            memories: Arc::new(memories),
            source_key_to_id: Arc::new(source_key_to_id),
            cue_index: Arc::new(cue_index),
            source_order_index: Arc::new(DashMap::with_hasher(RandomState::new())),
            cue_global_counts,
            last_events: Arc::new(DashMap::with_hasher(RandomState::new())),
            memory_count: Arc::new(AtomicUsize::new(count)),
            cue_count: Arc::new(AtomicUsize::new(cue_count)),
            next_memory_id: Arc::new(AtomicU32::new(next_memory_id.max(1))),
            master_key: None,
            tuning: Arc::new(TuningConfig::default()),
            semantic_index: Arc::new(RwLock::new(SemanticIndex::new(
                config.semantic.clone(),
            ))),
            semantic_encoder: Arc::new(RwLock::new(None)),
            intent_classifier: Arc::new(RwLock::new(None)),
            query_embedding_cache: Self::new_query_embedding_cache(&config.semantic),
            config: config.clone(),
            project_id: project_id.clone(),
        };
        if let Ok(mut index) = engine.semantic_index.write() {
            index.rebuild(engine.memories.iter().filter_map(|entry| {
                entry
                    .semantic_vector
                    .as_ref()
                    .map(|vector| (entry.id, vector.clone()))
            }));
        }
        engine.rebuild_source_order_index();

        // Migration logic: Sync RAM/Disk state with config
        if config.server.store_content_on_disk {
            // Move loaded memories to disk
            let disk_dir = engine.get_disk_content_dir();
            for mut entry in engine.memories.iter_mut() {
                let memory = entry.value_mut();
                if !memory.disk_backed && !memory.content.is_empty() {
                    let path = disk_dir.join(format!("{}.bin", memory.id));
                    if let Err(e) = std::fs::write(&path, &memory.content) {
                        tracing::error!("Migration (RAM -> Disk): Failed for {}: {}", memory.id, e);
                    } else {
                        memory.content = vec![];
                        memory.disk_backed = true;
                    }
                }
            }
        } else {
            // Move disk-backed memories back to RAM
            let disk_dir = engine.get_disk_content_dir();
            let mut migrated_count = 0;
            for mut entry in engine.memories.iter_mut() {
                let memory = entry.value_mut();
                if memory.disk_backed {
                    let path = disk_dir.join(format!("{}.bin", memory.id));
                    if path.exists() {
                        match std::fs::read(&path) {
                            Ok(bytes) => {
                                memory.content = bytes;
                                memory.disk_backed = false;
                                migrated_count += 1;
                                // Clean up disk file
                                let _ = std::fs::remove_file(&path);
                            }
                            Err(e) => tracing::error!(
                                "Migration (Disk -> RAM): Failed to read {}: {}",
                                memory.id,
                                e
                            ),
                        }
                    }
                }
            }
            if migrated_count > 0 {
                tracing::info!(
                    "Migration (Disk -> RAM): Successfully restored {} memories to RAM",
                    migrated_count
                );
            }
        }

        engine
    }

    // Expose internal state for persistence
    pub fn get_memories(&self) -> &Arc<DashMap<MemoryId, Memory<T>, RandomState>> {
        &self.memories
    }

    pub fn get_source_key_to_id(&self) -> &Arc<DashMap<String, MemoryId, RandomState>> {
        &self.source_key_to_id
    }

    pub fn get_cue_index(&self) -> &Arc<DashMap<String, OrderedSet, RandomState>> {
        &self.cue_index
    }

    pub fn next_memory_id(&self) -> MemoryId {
        self.next_memory_id.load(Ordering::Relaxed)
    }

    pub fn memory_id_for_source_key(&self, source_key: &str) -> Option<MemoryId> {
        self.source_key_to_id.get(source_key).map(|entry| *entry)
    }

    fn allocate_memory_id(&self) -> Option<MemoryId> {
        loop {
            let current = self.next_memory_id.load(Ordering::Relaxed);
            if current == INVALID_MEMORY_ID || current == MemoryId::MAX {
                tracing::error!("Memory ID space exhausted for project {}", self.project_id);
                return None;
            }
            let next = current + 1;
            if self
                .next_memory_id
                .compare_exchange(current, next, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Some(current);
            }
        }
    }

    fn index_memory_cues(&self, memory_id: MemoryId, cues: &[String]) {
        for cue in cues {
            let cue_lower = cue.to_lowercase().trim().to_string();
            if cue_lower.is_empty() {
                continue;
            }

            if !self.cue_index.contains_key(&cue_lower) {
                self.cue_count.fetch_add(1, Ordering::Relaxed);
            }
            self.cue_index
                .entry(cue_lower.clone())
                .or_insert_with(OrderedSet::new)
                .add(memory_id);

            if should_index_value_alias(&cue_lower) {
                if let Some((_, value)) = cue_lower.split_once(':') {
                    if !value.is_empty() {
                        let value = value.to_string();
                        if !self.cue_index.contains_key(&value) {
                            self.cue_count.fetch_add(1, Ordering::Relaxed);
                        }
                        self.cue_index
                            .entry(value)
                            .or_insert_with(OrderedSet::new)
                            .add(memory_id);
                    }
                }
            }
        }
    }

    /// Helper to determine the storage path for disk-backed contents
    pub fn get_disk_content_dir(&self) -> std::path::PathBuf {
        let path = std::path::PathBuf::from(&self.config.server.data_dir)
            .join("contents")
            .join(&self.project_id);

        if !path.exists() {
            let _ = std::fs::create_dir_all(&path);
        }
        path
    }

    /// Read memory content dynamically, handling disk-backed resolution
    pub fn read_memory_content(&self, memory: &Memory<T>) -> Result<String, String> {
        if memory.disk_backed {
            let path = self
                .get_disk_content_dir()
                .join(format!("{}.bin", memory.id));
            let bytes = std::fs::read(&path).map_err(|e| format!("Disk read error: {}", e))?;
            Memory::<T>::decode_content_bytes(&bytes, self.master_key.as_deref())
        } else {
            memory.access_content(self.master_key.as_deref())
        }
    }

    fn metadata_string_value(
        metadata: &HashMap<String, serde_json::Value>,
        keys: &[&str],
    ) -> Option<String> {
        for key in keys {
            let Some(value) = metadata.get(*key) else {
                continue;
            };
            match value {
                serde_json::Value::String(text) if !text.trim().is_empty() => {
                    return Some(text.trim().to_string());
                }
                serde_json::Value::Number(number) => return Some(number.to_string()),
                serde_json::Value::Bool(value) => return Some(value.to_string()),
                _ => {}
            }
        }
        None
    }

    fn normalize_source_order_value(value: &str) -> Option<String> {
        let normalized = value
            .trim()
            .to_lowercase()
            .chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
            .collect::<String>()
            .split('_')
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("_");
        if normalized.is_empty() {
            None
        } else {
            Some(normalized)
        }
    }

    fn source_order_session_from(
        metadata: &HashMap<String, serde_json::Value>,
        cues: &[String],
    ) -> Option<String> {
        if let Some(session) = Self::metadata_string_value(
            metadata,
            &[
                "source_session_id",
                "session_id",
                "conversation_id",
                "thread_id",
            ],
        )
        .and_then(|value| Self::normalize_source_order_value(&value))
        {
            return Some(session);
        }

        for cue in cues {
            if let Some(value) = cue.strip_prefix("source_session:") {
                if let Some(normalized) = Self::normalize_source_order_value(value) {
                    return Some(normalized);
                }
            }
        }
        None
    }

    fn source_order_value_from(
        metadata: &HashMap<String, serde_json::Value>,
        cues: &[String],
    ) -> Option<i64> {
        for key in [
            "source_turn_index",
            "source_order",
            "turn_index",
            "message_index",
            "sequence_index",
        ] {
            let Some(value) = metadata.get(key) else {
                continue;
            };
            match value {
                serde_json::Value::Number(number) => {
                    if let Some(value) = number.as_i64() {
                        return Some(value);
                    }
                    if let Some(value) = number.as_u64().and_then(|v| i64::try_from(v).ok()) {
                        return Some(value);
                    }
                }
                serde_json::Value::String(text) => {
                    if let Ok(value) = text.trim().parse::<i64>() {
                        return Some(value);
                    }
                }
                _ => {}
            }
        }

        for cue in cues {
            for prefix in [
                "source_turn_index:",
                "source_order:",
                "turn_index:",
                "message_index:",
                "sequence_index:",
            ] {
                if let Some(value) = cue.strip_prefix(prefix) {
                    if let Ok(parsed) = value.trim().parse::<i64>() {
                        return Some(parsed);
                    }
                }
            }
        }
        None
    }

    fn source_order_link_for_memory(memory: &Memory<T>) -> Option<(String, i64)> {
        let session = Self::source_order_session_from(&memory.metadata, &memory.cues)?;
        let order = Self::source_order_value_from(&memory.metadata, &memory.cues)?;
        Some((session, order))
    }

    fn add_source_order_entry(&self, session: String, order: i64, memory_id: MemoryId) {
        let mut entries = self
            .source_order_index
            .entry(session)
            .or_insert_with(Vec::new);
        entries.retain(|entry| entry.memory_id != memory_id);
        entries.push(SourceOrderEntry { order, memory_id });
        entries.sort_unstable_by(|a, b| {
            a.order
                .cmp(&b.order)
                .then_with(|| a.memory_id.cmp(&b.memory_id))
        });
    }

    fn remove_source_order_entry(&self, memory_id: MemoryId) {
        let sessions: Vec<String> = self
            .source_order_index
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        for session in sessions {
            let mut empty = false;
            if let Some(mut entries) = self.source_order_index.get_mut(&session) {
                entries.retain(|entry| entry.memory_id != memory_id);
                empty = entries.is_empty();
            }
            if empty {
                self.source_order_index.remove(&session);
            }
        }
    }

    pub fn rebuild_source_order_index(&self) {
        self.source_order_index.clear();
        for memory in self.memories.iter() {
            if let Some((session, order)) = Self::source_order_link_for_memory(memory.value()) {
                self.add_source_order_entry(session, order, memory.value().id);
            }
        }
    }

    pub fn source_order_for_memory(&self, memory_id: MemoryId) -> Option<(String, i64)> {
        let memory = self.memories.get(&memory_id)?;
        Self::source_order_link_for_memory(memory.value())
    }

    pub fn ordered_entries_for_session(
        &self,
        session: &str,
        scan_limit: usize,
    ) -> Vec<SourceOrderEntry> {
        let Some(normalized) = Self::normalize_source_order_value(session) else {
            return Vec::new();
        };
        self.source_order_index
            .get(&normalized)
            .map(|entries| entries.iter().take(scan_limit).copied().collect())
            .unwrap_or_default()
    }

    fn source_order_window(
        &self,
        session: &str,
        center_order: i64,
        radius: usize,
    ) -> Vec<SourceOrderEntry> {
        let Some(normalized) = Self::normalize_source_order_value(session) else {
            return Vec::new();
        };
        let radius = i64::try_from(radius).unwrap_or(i64::MAX);
        let start_order = center_order.saturating_sub(radius);
        let end_order = center_order.saturating_add(radius);
        self.source_order_index
            .get(&normalized)
            .map(|entries| {
                entries
                    .iter()
                    .copied()
                    .filter(|entry| entry.order >= start_order && entry.order <= end_order)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn with_synchronous_facets(
        &self,
        content: &str,
        metadata: Option<&HashMap<String, serde_json::Value>>,
        cues: Vec<String>,
    ) -> Vec<String> {
        if TypeId::of::<T>() != TypeId::of::<MainStats>() {
            return cues;
        }

        let mut enriched = cues;
        let mut seen: HashSet<String> = enriched.iter().map(|cue| cue.to_lowercase()).collect();
        for facet in crate::facets::extract_memory_facets(content, metadata, &enriched) {
            if seen.insert(facet.to_lowercase()) {
                enriched.push(facet);
            }
        }
        enriched
    }

    pub fn add_memory(
        &self,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        stats: T,
        disable_temporal_chunking: bool,
    ) -> MemoryId {
        self.add_memory_with_source_key_and_event_time(
            content,
            cues,
            metadata,
            stats,
            disable_temporal_chunking,
            None,
            None,
        )
    }

    pub fn add_memory_with_source_key(
        &self,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        stats: T,
        disable_temporal_chunking: bool,
        source_key: Option<String>,
    ) -> MemoryId {
        self.add_memory_with_source_key_and_event_time(
            content,
            cues,
            metadata,
            stats,
            disable_temporal_chunking,
            source_key,
            None,
        )
    }

    pub fn add_memory_with_event_time(
        &self,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        stats: T,
        disable_temporal_chunking: bool,
        event_time: Option<f64>,
    ) -> MemoryId {
        self.add_memory_with_source_key_and_event_time(
            content,
            cues,
            metadata,
            stats,
            disable_temporal_chunking,
            None,
            event_time,
        )
    }

    pub fn add_memory_with_event_time_and_vector(
        &self,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        stats: T,
        disable_temporal_chunking: bool,
        event_time: Option<f64>,
        semantic_vector: Option<Vec<f32>>,
    ) -> MemoryId {
        self.add_memory_with_source_key_and_event_time_and_vector(
            content,
            cues,
            metadata,
            stats,
            disable_temporal_chunking,
            None,
            event_time,
            semantic_vector,
        )
    }

    fn add_memory_with_source_key_and_event_time(
        &self,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        stats: T,
        disable_temporal_chunking: bool,
        source_key: Option<String>,
        event_time: Option<f64>,
    ) -> MemoryId {
        self.add_memory_with_source_key_and_event_time_and_vector(
            content,
            cues,
            metadata,
            stats,
            disable_temporal_chunking,
            source_key,
            event_time,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn add_memory_with_source_key_and_event_time_and_vector(
        &self,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        stats: T,
        disable_temporal_chunking: bool,
        source_key: Option<String>,
        event_time: Option<f64>,
        semantic_vector: Option<Vec<f32>>,
    ) -> MemoryId {
        let semantic_vector = semantic_vector
            .or_else(|| self.encode_semantic_text(&content));
        let semantic_vector = self.prepare_semantic_vector(semantic_vector);
        let cues = self.with_synchronous_facets(&content, metadata.as_ref(), cues);

        // Create payload (Compressed or Encrypted)
        let payload = match Memory::<T>::create_payload(&content, self.master_key.as_deref()) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Failed to create memory payload: {}", e);
                return INVALID_MEMORY_ID;
            }
        };

        let Some(memory_id) = self.allocate_memory_id() else {
            return INVALID_MEMORY_ID;
        };
        let mut memory = Memory::new(payload, metadata);
        memory.id = memory_id;
        memory.source_key = source_key.clone();
        memory.semantic_vector = semantic_vector.clone();
        if let Some(event_time) = event_time {
            memory.created_at = event_time;
        }

        // Store cues in memory
        memory.cues = cues.clone();
        memory.stats = stats;

        // Handle disk-backed contents
        if self.config.server.store_content_on_disk {
            let path = self
                .get_disk_content_dir()
                .join(format!("{}.bin", memory.id));
            if let Err(e) = std::fs::write(&path, &memory.content) {
                tracing::error!("Failed to write memory content to disk: {}", e);
            } else {
                memory.content = vec![]; // Clear from RAM
                memory.disk_backed = true;
            }
        }

        // 1. Temporal Chunking
        let project_id = memory
            .metadata
            .get("project_id")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();

        if let Some(last_event) = self.last_events.get(&project_id) {
            let (last_id, last_time, last_cues) = last_event.clone();
            let now = memory.created_at;

            // Time proximity (< 5 mins) and High cue overlap (> 50%)
            let time_diff = now - last_time;
            let overlap = memory.cues.iter().filter(|c| last_cues.contains(c)).count();
            let overlap_ratio = if !memory.cues.is_empty() {
                (overlap as f64) / (memory.cues.len() as f64)
            } else {
                0.0
            };

            if (0.0..300.0).contains(&time_diff)
                && overlap_ratio > 0.5
                && !disable_temporal_chunking
            {
                let episode_cue = format!("episode:{}", last_id);
                memory.cues.push(episode_cue.clone());
            }
        }
        memory.scoring_features = compute_memory_scoring_features(&memory.cues);
        self.last_events.insert(
            project_id,
            (memory_id, memory.created_at, memory.cues.clone()),
        );
        let indexed_cues = memory.cues.clone();
        let source_order_link = Self::source_order_link_for_memory(&memory);
        if self.memories.insert(memory_id, memory).is_none() {
            self.memory_count.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(source_key) = source_key {
            self.source_key_to_id.insert(source_key, memory_id);
        }
        self.index_memory_cues(memory_id, &indexed_cues);
        if let Some(vector) = semantic_vector.as_ref() {
            if let Ok(mut index) = self.semantic_index.write() {
                if let Err(error) = index.insert(memory_id, vector) {
                    tracing::debug!(
                        memory_id,
                        error = %error,
                        "Skipping invalid semantic vector"
                    );
                }
            }
        }
        if let Some((session, order)) = source_order_link {
            self.add_source_order_entry(session, order, memory_id);
        }

        memory_id
    }

    pub fn reinforce_memory(&self, memory_id: MemoryId, cues: Vec<String>) -> bool {
        if let Some(mut memory) = self.memories.get_mut(&memory_id) {
            memory.touch();
            memory.stats.manual_boost(); // Manual reinforcement boost
        } else {
            return false;
        }

        // Move to front for each cue (Double Indexing)
        for cue in cues {
            let cue_lower = cue.to_lowercase().trim().to_string();
            if cue_lower.is_empty() {
                continue;
            }

            // 1. Move full cue
            if let Some(mut entry) = self.cue_index.get_mut(&cue_lower) {
                entry.move_to_front(memory_id);
            }

            // 2. Move value
            if should_index_value_alias(&cue_lower) {
                if let Some((_, value)) = cue_lower.split_once(':') {
                    if !value.is_empty() {
                        if let Some(mut entry) = self.cue_index.get_mut(value) {
                            entry.move_to_front(memory_id);
                        }
                    }
                }
            }
        }

        true
    }

    pub fn delete_memory(&self, memory_id: MemoryId) -> bool {
        if let Some((_, memory)) = self.memories.remove(&memory_id) {
            self.memory_count.fetch_sub(1, Ordering::Relaxed);
            if let Ok(mut index) = self.semantic_index.write() {
                index.remove(memory_id);
            }
            self.remove_source_order_entry(memory_id);
            if let Some(source_key) = memory.source_key {
                self.source_key_to_id.remove(&source_key);
            }
            // Remove from cue index (Double Indexing)
            for cue in memory.cues {
                let cue_lower = cue.to_lowercase().trim().to_string();
                if cue_lower.is_empty() {
                    continue;
                }

                // 1. Remove from full cue entry
                if let Some(mut entry) = self.cue_index.get_mut(&cue_lower) {
                    entry.remove(memory_id);
                    if entry.is_empty() {
                        drop(entry); // Release RefMut to allow removal
                        if self.cue_index.remove(&cue_lower).is_some() {
                            self.cue_count.fetch_sub(1, Ordering::Relaxed);
                        }
                    }
                }

                // 2. Remove from value entry
                if should_index_value_alias(&cue_lower) {
                    if let Some((_, value)) = cue_lower.split_once(':') {
                        if !value.is_empty() {
                            if let Some(mut entry) = self.cue_index.get_mut(value) {
                                entry.remove(memory_id);
                                if entry.is_empty() {
                                    drop(entry);
                                    if self.cue_index.remove(value).is_some() {
                                        self.cue_count.fetch_sub(1, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            true
        } else {
            false
        }
    }

    pub fn get_cue_frequency(&self, cue: &str) -> usize {
        let cue_lower = cue.to_lowercase();
        let cue_trimmed = cue_lower.trim();
        if let Some(set) = self.cue_index.get(cue_trimmed) {
            set.len()
        } else {
            0
        }
    }

    pub fn total_memories(&self) -> usize {
        self.memory_count.load(Ordering::Relaxed)
    }

    pub fn cue_index_version(&self) -> usize {
        self.cue_count.load(Ordering::Relaxed)
    }

    pub fn upsert_memory_with_source_key(
        &self,
        source_key: String,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        stats: Option<T>,
        reinforce: bool,
        overwrite_cues: bool,
    ) -> MemoryId {
        self.upsert_memory_with_source_key_and_options(
            source_key,
            content,
            cues,
            metadata,
            stats,
            reinforce,
            overwrite_cues,
            true,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsert_memory_with_source_key_and_options(
        &self,
        source_key: String,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        stats: Option<T>,
        reinforce: bool,
        overwrite_cues: bool,
        disable_temporal_chunking: bool,
        event_time: Option<f64>,
    ) -> MemoryId {
        self.upsert_memory_with_source_key_and_options_and_vector(
            source_key,
            content,
            cues,
            metadata,
            stats,
            reinforce,
            overwrite_cues,
            disable_temporal_chunking,
            event_time,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsert_memory_with_source_key_and_options_and_vector(
        &self,
        source_key: String,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        stats: Option<T>,
        reinforce: bool,
        overwrite_cues: bool,
        disable_temporal_chunking: bool,
        event_time: Option<f64>,
        semantic_vector: Option<Vec<f32>>,
    ) -> MemoryId {
        let semantic_vector = semantic_vector
            .or_else(|| self.encode_semantic_text(&content));
        let semantic_vector = self.prepare_semantic_vector(semantic_vector);
        let cues = self.with_synchronous_facets(&content, metadata.as_ref(), cues);

        if let Some(existing_id) = self.source_key_to_id.get(&source_key).map(|entry| *entry) {
            if let Ok(mut index) = self.semantic_index.write() {
                index.remove(existing_id);
            }
            self.remove_source_order_entry(existing_id);
            {
                if let Some(mut memory) = self.memories.get_mut(&existing_id) {
                    // Update content ALWAYS
                    match Memory::<T>::create_payload(&content, self.master_key.as_deref()) {
                        Ok(p) => {
                            if self.config.server.store_content_on_disk {
                                let path = self
                                    .get_disk_content_dir()
                                    .join(format!("{}.bin", existing_id));
                                if let Err(e) = std::fs::write(&path, &p) {
                                    tracing::error!(
                                        "Failed to update memory content on disk: {}",
                                        e
                                    );
                                    memory.content = p;
                                } else {
                                    memory.content = vec![];
                                    memory.disk_backed = true;
                                }
                            } else {
                                memory.content = p;
                                memory.disk_backed = false;
                            }
                        }
                        Err(e) => tracing::error!("Failed to update content: {}", e),
                    }

                    if let Some(m) = metadata {
                        memory.metadata = m;
                    }
                    if let Some(s) = stats.clone() {
                        memory.stats = s;
                    }
                    if let Some(event_time) = event_time {
                        memory.created_at = event_time;
                    }
                    memory.semantic_vector = semantic_vector.clone();
                    memory.source_key = Some(source_key.clone());
                    // We need to drop lock before attach/overwrite ops to avoid deadlocks
                    // (though attach_cues re-acquires check, better safe)
                }
            }

            if overwrite_cues {
                // Remove old cues from index + Replace cues
                // We need to get old cues first
                let old_cues = if let Some(mem) = self.memories.get(&existing_id) {
                    mem.cues.clone()
                } else {
                    Vec::new()
                };

                self.remove_cues_from_index(existing_id, &old_cues);

                if let Some(mut mem) = self.memories.get_mut(&existing_id) {
                    mem.cues = Vec::new(); // Clear
                }
                // Now attach new cues (effectively replacing)
                self.attach_cues(existing_id, cues.clone());
            } else {
                // Merge mode
                self.attach_cues(existing_id, cues.clone());
            }

            if reinforce {
                self.reinforce_memory(existing_id, cues);
            }
            if let Some(memory) = self.memories.get(&existing_id) {
                if let Some((session, order)) = Self::source_order_link_for_memory(memory.value()) {
                    self.add_source_order_entry(session, order, existing_id);
                }
            }
            if let Some(vector) = semantic_vector.as_ref() {
                if let Ok(mut index) = self.semantic_index.write() {
                    if let Err(error) = index.insert(existing_id, vector) {
                        tracing::debug!(
                            memory_id = existing_id,
                            error = %error,
                            "Skipping invalid semantic vector"
                        );
                    }
                }
            }
            return existing_id;
        }

        // Insert new
        let payload = match Memory::<T>::create_payload(&content, self.master_key.as_deref()) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Failed to create memory payload: {}", e);
                return INVALID_MEMORY_ID;
            }
        };

        let Some(memory_id) = self.allocate_memory_id() else {
            return INVALID_MEMORY_ID;
        };
        let mut memory = Memory::new(payload, metadata);
        memory.id = memory_id;
        memory.source_key = Some(source_key.clone());
        memory.semantic_vector = semantic_vector.clone();
        if let Some(event_time) = event_time {
            memory.created_at = event_time;
        }
        memory.cues = cues.clone();

        if self.config.server.store_content_on_disk {
            let path = self
                .get_disk_content_dir()
                .join(format!("{}.bin", memory.id));
            if let Err(error) = std::fs::write(&path, &memory.content) {
                tracing::error!("Failed to write memory content to disk: {}", error);
            } else {
                memory.content = Vec::new();
                memory.disk_backed = true;
            }
        }

        let project_id = memory
            .metadata
            .get("project_id")
            .and_then(|value| value.as_str())
            .unwrap_or("default")
            .to_string();
        if let Some(last_event) = self.last_events.get(&project_id) {
            let (last_id, last_time, last_cues) = last_event.clone();
            let time_diff = memory.created_at - last_time;
            let overlap = memory.cues.iter().filter(|cue| last_cues.contains(cue)).count();
            let overlap_ratio = if memory.cues.is_empty() {
                0.0
            } else {
                (overlap as f64) / (memory.cues.len() as f64)
            };

            if (0.0..300.0).contains(&time_diff)
                && overlap_ratio > 0.5
                && !disable_temporal_chunking
            {
                memory.cues.push(format!("episode:{}", last_id));
            }
        }
        memory.scoring_features = compute_memory_scoring_features(&memory.cues);
        if let Some(s) = stats {
            memory.stats = s;
        }
        let indexed_cues = memory.cues.clone();
        let source_order_link = Self::source_order_link_for_memory(&memory);
        self.last_events.insert(
            project_id,
            (memory_id, memory.created_at, memory.cues.clone()),
        );

        if self.memories.insert(memory_id, memory).is_none() {
            self.memory_count.fetch_add(1, Ordering::Relaxed);
        }
        self.source_key_to_id.insert(source_key, memory_id);
        self.index_memory_cues(memory_id, &indexed_cues);
        if let Some(vector) = semantic_vector.as_ref() {
            if let Ok(mut index) = self.semantic_index.write() {
                if let Err(error) = index.insert(memory_id, vector) {
                    tracing::debug!(
                        memory_id,
                        error = %error,
                        "Skipping invalid semantic vector"
                    );
                }
            }
        }
        if let Some((session, order)) = source_order_link {
            self.add_source_order_entry(session, order, memory_id);
        }
        memory_id
    }

    pub fn attach_cues(&self, memory_id: MemoryId, cues: Vec<String>) -> bool {
        // 1. Get memory and check if it exists
        if let Some(mut memory) = self.memories.get_mut(&memory_id) {
            // 2. Identify new cues (deduplication)
            let mut new_cues = Vec::new();
            for cue in cues {
                let cue_lower = cue.to_lowercase().trim().to_string();
                if cue_lower.is_empty() {
                    continue;
                }

                // Check against existing cues (case-insensitive check technically needed, but we store as-is)
                // Assuming existing cues were normalized or we just check strict equality
                if !memory.cues.contains(&cue) {
                    new_cues.push(cue);
                }
            }

            if new_cues.is_empty() {
                return false;
            }

            // 3. Update memory.cues
            memory.cues.extend(new_cues.clone());
            memory.scoring_features = compute_memory_scoring_features(&memory.cues);

            // 4. Update index for new cues (Double Indexing)
            for cue in new_cues {
                let cue_lower = cue.to_lowercase().trim().to_string();

                // 1. Index full cue
                let cue_lower_clone = cue_lower.clone();
                if !self.cue_index.contains_key(&cue_lower_clone) {
                    self.cue_count.fetch_add(1, Ordering::Relaxed);
                }
                self.cue_index
                    .entry(cue_lower_clone)
                    .or_insert_with(OrderedSet::new)
                    .add(memory_id);

                // 2. Index value
                if should_index_value_alias(&cue_lower) {
                    if let Some((_, value)) = cue_lower.split_once(':') {
                        if !value.is_empty() {
                            let val_str = value.to_string();
                            if !self.cue_index.contains_key(&val_str) {
                                self.cue_count.fetch_add(1, Ordering::Relaxed);
                            }
                            self.cue_index
                                .entry(val_str)
                                .or_insert_with(OrderedSet::new)
                                .add(memory_id);
                        }
                    }
                }
            }

            let source_order_link = Self::source_order_link_for_memory(&memory);
            drop(memory);
            self.remove_source_order_entry(memory_id);
            if let Some((session, order)) = source_order_link {
                self.add_source_order_entry(session, order, memory_id);
            }
            return true;
        } else {
            false
        }
    }

    pub fn remove_cues_from_index(&self, memory_id: MemoryId, cues: &[String]) {
        for cue in cues {
            let cue_lower = cue.to_lowercase().trim().to_string();
            if cue_lower.is_empty() {
                continue;
            }

            // 1. Remove from full cue entry
            if let Some(mut entry) = self.cue_index.get_mut(&cue_lower) {
                entry.remove(memory_id);
                if entry.is_empty() {
                    drop(entry);
                    if self.cue_index.remove(&cue_lower).is_some() {
                        self.cue_count.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            }

            // 2. Remove from value entry
            if should_index_value_alias(&cue_lower) {
                if let Some((_, value)) = cue_lower.split_once(':') {
                    if !value.is_empty() {
                        if let Some(mut entry) = self.cue_index.get_mut(value) {
                            entry.remove(memory_id);
                            if entry.is_empty() {
                                drop(entry);
                                if self.cue_index.remove(value).is_some() {
                                    self.cue_count.fetch_sub(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn recall(
        &self,
        query_cues: Vec<String>,
        limit: usize,
        auto_reinforce: bool,
        heatmap: Option<&HashMap<String, f32>>,
    ) -> Vec<RecallResult> {
        self.recall_with_min_intersection(query_cues, limit, auto_reinforce, None, heatmap)
    }

    pub fn recall_with_min_intersection(
        &self,
        query_cues: Vec<String>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        heatmap: Option<&HashMap<String, f32>>,
    ) -> Vec<RecallResult> {
        if query_cues.is_empty() {
            return Vec::new();
        }

        // Default weight of 1.0 for standard recall
        let weighted_cues: Vec<(String, f64)> = query_cues.into_iter().map(|c| (c, 1.0)).collect();

        self.recall_weighted(
            weighted_cues,
            limit,
            auto_reinforce,
            min_intersection,
            1,
            false,
            false,
            heatmap,
            None,
        )
    }

    /// O(limit) recall using intersection-first strategy.
    /// Only scans the smallest cue list up to `limit` items, probes others in O(1).
    /// Returns results in recency order - no expensive sorting.
    /// Best for: simple keyword queries where speed > perfect ranking.
    pub fn recall_intersection(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
    ) -> Vec<RecallResult> {
        if query_cues.is_empty() || limit == 0 {
            return Vec::new();
        }

        // 1. Normalize and collect cue sets with sizes
        let mut cue_sets = Vec::new();
        for (cue, weight) in &query_cues {
            let cue_lower = cue.to_lowercase();
            let cue_trimmed = cue_lower.trim().to_string();
            if cue_trimmed.is_empty() {
                continue;
            }

            if let Some(ordered_set) = self.cue_index.get(&cue_trimmed) {
                cue_sets.push((cue_trimmed, *weight, ordered_set));
            }
        }

        if cue_sets.is_empty() {
            return Vec::new();
        }

        // 2. Sort by set size - smallest (most selective) first
        cue_sets.sort_by(|a, b| a.2.len().cmp(&b.2.len()));

        // 3. Iterate ONLY the smallest cue's list, up to limit items
        let (_driver_cue, driver_weight, driver_set) = &cue_sets[0];
        let other_sets = &cue_sets[1..];

        let mut results = Vec::with_capacity(limit);
        let scan_limit = driver_set.len().min(limit * 10); // Scan 10x limit to find enough intersections

        for memory_id in driver_set.get_recent(Some(scan_limit)) {
            // 4. O(1) probes into other cue sets
            let mut total_weight = *driver_weight;
            let mut match_count = 1;

            for (_other_cue, other_weight, other_set) in other_sets {
                if other_set.get_index_of(memory_id).is_some() {
                    total_weight += other_weight;
                    match_count += 1;
                }
            }

            // 5. Fetch memory and build result
            if let Some(memory) = self.memories.get(&memory_id) {
                let decrypted_content = self
                    .read_memory_content(memory.value())
                    .unwrap_or_else(|_| "<decryption failed>".to_string());

                results.push(RecallResult {
                    memory_id,
                    content: decrypted_content,
                    score: total_weight * 100.0, // Simple intersection-based score
                    match_integrity: (match_count as f64) / (cue_sets.len() as f64),
                    intersection_count: match_count,
                    recency_score: 1.0,
                    reinforcement_score: memory.stats.get_reinforcement_count() as f64,
                    salience_score: memory.stats.get_salience(),
                    created_at: memory.created_at,
                    metadata: memory.metadata.clone(),
                    explain: None,
                });

                // 6. Early termination when limit reached
                if results.len() >= limit {
                    break;
                }
            }
        }

        results
    }

    /// Fast O(1) lookup for lexicon-style queries.
    /// Returns memories that match ANY query cue, ordered by recency.
    /// No scoring - just direct index lookup.
    pub fn recall_fast(&self, query_cues: Vec<String>, limit: usize) -> Vec<RecallResult> {
        if query_cues.is_empty() {
            return Vec::new();
        }

        // We need to collect ALL candidates first to sort them
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();

        for cue in query_cues {
            let cue_lower = cue.to_lowercase();
            let cue_trimmed = cue_lower.trim();
            if cue_trimmed.is_empty() {
                continue;
            }

            if let Some(ordered_set) = self.cue_index.get(cue_trimmed) {
                // Grab more than the limit initially (2x limit) to allow for sorting
                for memory_id in ordered_set.get_recent(Some(limit * 2)) {
                    if seen.contains(&memory_id) {
                        continue;
                    }
                    seen.insert(memory_id);

                    if let Some(memory) = self.memories.get(&memory_id) {
                        let decrypted_content = self
                            .read_memory_content(memory.value())
                            .unwrap_or_else(|_| "<decryption failed>".to_string());

                        candidates.push(RecallResult {
                            memory_id,
                            content: decrypted_content,
                            score: 1.0,
                            match_integrity: 1.0,
                            intersection_count: 1,
                            recency_score: 1.0,
                            reinforcement_score: memory.stats.get_reinforcement_count() as f64,
                            salience_score: memory.stats.get_salience(),
                            created_at: memory.created_at,
                            metadata: memory.metadata.clone(),
                            explain: None,
                        });
                    }
                }
            }
        }

        // Sort by Hierarchy of Signals (Cascading Sort)
        candidates.sort_by(|a, b| {
            // 1. Primary: Learned Relevance (Hebbian) - "What have I successfully recalled before?"
            b.reinforcement_score
                .partial_cmp(&a.reinforcement_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                // 2. Secondary: Intrinsic Value (Amygdala) - "Which memory has rarer/richer cues?"
                // This SOLVES the Cold Start. "Lemon Cheesecake" (rare) > "Food" (common).
                .then_with(|| {
                    b.salience_score
                        .partial_cmp(&a.salience_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                // 3. Tertiary: Freshness (Temporal) - "If both are unreinforced and equally salient, show the new one."
                .then_with(|| {
                    b.created_at
                        .partial_cmp(&a.created_at)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        candidates.into_iter().take(limit).collect()
    }

    pub fn recall_weighted(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        expansion_depth: usize,
        explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
    ) -> Vec<RecallResult> {
        self.recall_weighted_profiled(
            query_cues,
            limit,
            auto_reinforce,
            min_intersection,
            expansion_depth,
            explain,
            disable_salience_bias,
            heatmap,
            mandatory_cues,
            None,
            false,
        )
        .0
    }

    pub fn recall_weighted_with_timing(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        expansion_depth: usize,
        explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
    ) -> (Vec<RecallResult>, RecallTimingBreakdown) {
        self.recall_weighted_profiled(
            query_cues,
            limit,
            auto_reinforce,
            min_intersection,
            expansion_depth,
            explain,
            disable_salience_bias,
            heatmap,
            mandatory_cues,
            None,
            true,
        )
    }

    pub fn recall_weighted_with_query_embedding(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        expansion_depth: usize,
        explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
        query_embedding: Option<&[f32]>,
    ) -> Vec<RecallResult> {
        self.recall_weighted_profiled(
            query_cues,
            limit,
            auto_reinforce,
            min_intersection,
            expansion_depth,
            explain,
            disable_salience_bias,
            heatmap,
            mandatory_cues,
            query_embedding,
            false,
        )
        .0
    }

    /// Runs semantic scoring only over the lexical candidates produced by the
    /// same request. This is the bounded hybrid path: it does not query the
    /// semantic index or add semantic-only candidates.
    pub fn recall_weighted_with_query_embedding_rerank_only(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        expansion_depth: usize,
        explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
        query_embedding: Option<&[f32]>,
    ) -> Vec<RecallResult> {
        self.recall_weighted_with_query_embedding_rerank_only_and_intent(
            query_cues,
            limit,
            auto_reinforce,
            min_intersection,
            expansion_depth,
            explain,
            disable_salience_bias,
            heatmap,
            mandatory_cues,
            query_embedding,
            None,
        )
    }

    pub fn recall_weighted_with_query_embedding_rerank_only_and_intent(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        expansion_depth: usize,
        explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
        query_embedding: Option<&[f32]>,
        query_intent: Option<&IntentClassification>,
    ) -> Vec<RecallResult> {
        self.recall_weighted_profiled_with_options(
            query_cues,
            limit,
            auto_reinforce,
            min_intersection,
            expansion_depth,
            explain,
            disable_salience_bias,
            heatmap,
            mandatory_cues,
            query_embedding,
            query_intent,
            false,
            true,
        )
        .0
    }

    pub fn recall_weighted_with_query_embedding_with_timing(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        expansion_depth: usize,
        explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
        query_embedding: Option<&[f32]>,
    ) -> (Vec<RecallResult>, RecallTimingBreakdown) {
        self.recall_weighted_profiled(
            query_cues,
            limit,
            auto_reinforce,
            min_intersection,
            expansion_depth,
            explain,
            disable_salience_bias,
            heatmap,
            mandatory_cues,
            query_embedding,
            true,
        )
    }

    /// Timing variant of the bounded hybrid path. Semantic scoring is
    /// restricted to the lexical result set and cannot introduce new IDs.
    pub fn recall_weighted_with_query_embedding_rerank_only_with_timing(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        expansion_depth: usize,
        explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
        query_embedding: Option<&[f32]>,
    ) -> (Vec<RecallResult>, RecallTimingBreakdown) {
        self.recall_weighted_with_query_embedding_rerank_only_with_intent_with_timing(
            query_cues,
            limit,
            auto_reinforce,
            min_intersection,
            expansion_depth,
            explain,
            disable_salience_bias,
            heatmap,
            mandatory_cues,
            query_embedding,
            None,
        )
    }

    pub fn recall_weighted_with_query_embedding_rerank_only_with_intent_with_timing(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        expansion_depth: usize,
        explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
        query_embedding: Option<&[f32]>,
        query_intent: Option<&IntentClassification>,
    ) -> (Vec<RecallResult>, RecallTimingBreakdown) {
        self.recall_weighted_profiled_with_options(
            query_cues,
            limit,
            auto_reinforce,
            min_intersection,
            expansion_depth,
            explain,
            disable_salience_bias,
            heatmap,
            mandatory_cues,
            query_embedding,
            query_intent,
            true,
            true,
        )
    }

    fn recall_weighted_profiled(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        expansion_depth: usize,
        explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
        query_embedding: Option<&[f32]>,
        collect_detailed_timing: bool,
    ) -> (Vec<RecallResult>, RecallTimingBreakdown) {
        self.recall_weighted_profiled_with_options(
            query_cues,
            limit,
            auto_reinforce,
            min_intersection,
            expansion_depth,
            explain,
            disable_salience_bias,
            heatmap,
            mandatory_cues,
            query_embedding,
            None,
            collect_detailed_timing,
            false,
        )
    }

    fn recall_weighted_profiled_with_options(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        expansion_depth: usize,
        explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
        query_embedding: Option<&[f32]>,
        query_intent: Option<&IntentClassification>,
        collect_detailed_timing: bool,
        semantic_rerank_only: bool,
    ) -> (Vec<RecallResult>, RecallTimingBreakdown) {
        let total_start = Instant::now();
        let mut timing = RecallTimingBreakdown::default();
        if query_cues.is_empty() && query_embedding.is_none() {
            timing.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            return (Vec::new(), timing);
        }

        // Normalize primary cues
        let phase_start = Instant::now();
        let active_cues: Vec<(String, f64)> = query_cues
            .iter()
            .map(|(c, w)| (c.to_lowercase().trim().to_string(), *w))
            .filter(|(c, _)| !c.is_empty() && self.cue_index.contains_key(c))
            .collect();
        timing.normalize_filter_ms = phase_start.elapsed().as_secs_f64() * 1000.0;
        timing.initial_active_cue_count = active_cues.len();

        if active_cues.is_empty() && query_embedding.is_none() {
            timing.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            return (Vec::new(), timing);
        }

        timing.active_cue_count = active_cues.len();

        // Consolidated search using Selective Set Intersection
        let phase_start = Instant::now();
        let (mut results, search_timing) = self.consolidated_search(
            &active_cues,
            limit,
            explain,
            disable_salience_bias,
            heatmap,
            mandatory_cues,
            collect_detailed_timing,
        );
        timing.consolidated_search_ms = phase_start.elapsed().as_secs_f64() * 1000.0;
        timing.candidate_generation_ms = search_timing.candidate_generation_ms;
        timing.candidate_scoring_ms = search_timing.candidate_scoring_ms;
        timing.scoring_filter_ms = search_timing.scoring_filter_ms;
        timing.scoring_position_ms = search_timing.scoring_position_ms;
        timing.scoring_salience_ms = search_timing.scoring_salience_ms;
        timing.scoring_structured_ms = search_timing.scoring_structured_ms;
        timing.scoring_finalize_ms = search_timing.scoring_finalize_ms;
        timing.candidate_count = search_timing.candidate_count;
        timing.scored_candidate_count = search_timing.scored_candidate_count;
        timing.cue_count_with_postings = search_timing.cue_count_with_postings;
        timing.scanned_posting_count = search_timing.scanned_posting_count;
        timing.adaptive_scan_limit = search_timing.adaptive_scan_limit;
        timing.max_posting_len = search_timing.max_posting_len;

        if semantic_rerank_only {
            let config = self.config.semantic.resolved();
            let rerank_limit = config.semantic_rerank_candidate_limit.max(limit);
            results.sort_unstable_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(rerank_limit);
            timing.semantic_rerank_candidate_limit = rerank_limit;
            timing.semantic_rerank_candidate_count = results.len();
        }

        // Apply intent before semantic fusion so the semantic pass operates on
        // the intent-aware lexical slate. The intent bonus is kept separately
        // and reintroduced by rerank_existing_semantic_candidates after
        // semantic normalization, so normalization cannot wash it out.
        if let Some(query_intent) = query_intent {
            self.apply_intent_reranker(&mut results, query_intent);
        }

        if let Some(query_embedding) = query_embedding {
            let semantic_start = Instant::now();
            let (new_candidates, semantic_candidates) = if semantic_rerank_only {
                (
                    0,
                    self.rerank_existing_semantic_candidates(&mut results, query_embedding),
                )
            } else {
                self.merge_semantic_candidates(&mut results, query_embedding, limit)
            };
            timing.candidate_generation_ms += semantic_start.elapsed().as_secs_f64() * 1000.0;
            timing.candidate_count += new_candidates;
            timing.scored_candidate_count += semantic_candidates;
            self.apply_semantic_reranker(&mut results, query_embedding);
        }

        // Filter by minimum intersection if specified (on primary cues only?)
        // For now, simple retention.
        let phase_start = Instant::now();
        if let Some(min_int) = min_intersection {
            results.retain(|r| r.intersection_count >= min_int);
        }
        timing.min_intersection_ms = phase_start.elapsed().as_secs_f64() * 1000.0;

        // 3. Auto-reinforce if enabled (only primary cues)
        let phase_start = Instant::now();
        if auto_reinforce {
            let primary_cues: Vec<String> = query_cues.iter().map(|(c, _)| c.clone()).collect();
            for result in &results {
                self.reinforce_memory(result.memory_id, primary_cues.clone());
            }
        }
        timing.auto_reinforce_ms = phase_start.elapsed().as_secs_f64() * 1000.0;

        // Global sort by score
        let phase_start = Instant::now();
        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit);
        timing.sort_truncate_ms = phase_start.elapsed().as_secs_f64() * 1000.0;

        // Finalize results by accessing content only for the top K
        let phase_start = Instant::now();
        let mut final_results = Vec::with_capacity(results.len());

        for candidate in results {
            if let Some(memory) = self.memories.get(&candidate.memory_id) {
                let explain_data = if explain {
                    Some(serde_json::json!({
                        "intersection_weighted": candidate.intersection_weighted,
                        "score": candidate.score,
                        "match_integrity": candidate.match_integrity,
                        "intersection_count": candidate.intersection_count,
                        "recency_score": candidate.recency_score,
                        "reinforcement_score": candidate.reinforcement_score,
                        "salience_score": candidate.salience_score,
                        "semantic_similarity": candidate.semantic_similarity,
                        "rerank_bonus": candidate.rerank_bonus,
                        "generic_penalty": candidate.generic_penalty,
                        "intent_compatibility": candidate.intent_compatibility,
                        "intent_rerank_bonus": candidate.intent_rerank_bonus,
                    }))
                } else {
                    None
                };

                let mut content = self
                    .read_memory_content(memory.value())
                    .unwrap_or_else(|_| "<decryption failed>".to_string());

                // Look for linkages for context expansion
                let mut parent_id = None;
                let mut chunk_idx = None;
                for cue in &memory.cues {
                    if cue.starts_with("parent:") {
                        parent_id = Some(cue.clone());
                    } else if cue.starts_with("chunk_idx:") {
                        chunk_idx = cue.split(':').nth(1).and_then(|v| v.parse::<usize>().ok());
                    }
                }
                let source_order_link = Self::source_order_link_for_memory(memory.value());
                let memory_metadata = memory.metadata.clone();
                drop(memory); // Release guard before potentially looking up siblings

                if expansion_depth > 1 {
                    let mut expanded_from_parent = false;
                    if let (Some(pid), Some(idx)) = (parent_id, chunk_idx) {
                        let mut neighbors = Vec::new();
                        let start_idx = idx.saturating_sub(expansion_depth - 1);
                        let end_idx = idx + expansion_depth - 1;

                        if let Some(parent_set) = self.cue_index.get(&pid) {
                            // Limit search to prevent massive documents from blowing up memory
                            for sibling_id in parent_set.items.iter().take(500) {
                                if sibling_id == &candidate.memory_id {
                                    continue;
                                }
                                if let Some(sibling) = self.memories.get(sibling_id) {
                                    for c in &sibling.cues {
                                        if c.starts_with("chunk_idx:") {
                                            if let Some(s_idx) = c
                                                .split(':')
                                                .nth(1)
                                                .and_then(|v| v.parse::<usize>().ok())
                                            {
                                                if s_idx >= start_idx && s_idx <= end_idx {
                                                    if let Ok(s_content) =
                                                        self.read_memory_content(sibling.value())
                                                    {
                                                        neighbors.push((s_idx, s_content));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if !neighbors.is_empty() {
                            neighbors.push((idx, content));
                            neighbors.sort_by_key(|(i, _)| *i);
                            content = neighbors
                                .into_iter()
                                .map(|(_, c)| c)
                                .collect::<Vec<String>>()
                                .join("\n\n");
                            expanded_from_parent = true;
                        }
                    }

                    if !expanded_from_parent {
                        if let Some((session, order)) = source_order_link {
                            let mut neighbors = Vec::new();
                            let mut includes_current = false;
                            for entry in
                                self.source_order_window(&session, order, expansion_depth - 1)
                            {
                                if let Some(sibling) = self.memories.get(&entry.memory_id) {
                                    if let Ok(s_content) = self.read_memory_content(sibling.value())
                                    {
                                        if entry.memory_id == candidate.memory_id {
                                            includes_current = true;
                                        }
                                        neighbors.push((entry.order, entry.memory_id, s_content));
                                    }
                                }
                            }

                            if !includes_current {
                                neighbors.push((order, candidate.memory_id, content.clone()));
                            }

                            if neighbors.len() > 1 {
                                neighbors.sort_by(|a, b| {
                                    a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1))
                                });
                                content = neighbors
                                    .into_iter()
                                    .map(|(_, _, c)| c)
                                    .collect::<Vec<String>>()
                                    .join("\n\n");
                            }
                        }
                    }
                }

                final_results.push(RecallResult {
                    memory_id: candidate.memory_id,
                    content,
                    score: candidate.score,
                    match_integrity: candidate.match_integrity,
                    intersection_count: candidate.intersection_count,
                    recency_score: candidate.recency_score,
                    reinforcement_score: candidate.reinforcement_score,
                    salience_score: candidate.salience_score,
                    created_at: candidate.created_at,
                    metadata: memory_metadata,
                    explain: explain_data,
                });
            }
        }

        timing.materialize_ms = phase_start.elapsed().as_secs_f64() * 1000.0;
        timing.returned_count = final_results.len();
        timing.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        (final_results, timing)
    }

    fn rerank_existing_semantic_candidates(
        &self,
        results: &mut [ScoredMemoryCandidate],
        query_embedding: &[f32],
    ) -> usize {
        let config = self.config.semantic.resolved();
        let normalized_query = match StoredSemanticVector::normalized_query(query_embedding) {
            Ok(query) => query,
            Err(error) => {
                tracing::debug!(error = %error, "Semantic reranking skipped");
                return 0;
            }
        };

        let mut scored = Vec::new();
        for (index, candidate) in results.iter_mut().enumerate() {
            let Some(memory) = self.memories.get(&candidate.memory_id) else {
                continue;
            };
            let Some(vector) = memory.semantic_vector.as_ref() else {
                continue;
            };
            let Ok(similarity) = vector.cosine_similarity_normalized(&normalized_query) else {
                continue;
            };

            candidate.semantic_similarity = similarity;
            scored.push((index, similarity));
        }

        if scored.len() < 2 {
            return scored.len();
        }

        // Intent is applied before this method, so recover the original
        // lexical score from the separately tracked intent delta. Semantic
        // normalization must compare lexical evidence on its own, then add
        // the strong intent prior back to the fused result.
        let lexical_min = results
            .iter()
            .map(|candidate| candidate.score - candidate.intent_rerank_bonus)
            .fold(f64::INFINITY, f64::min);
        let lexical_max = results
            .iter()
            .map(|candidate| candidate.score - candidate.intent_rerank_bonus)
            .fold(f64::NEG_INFINITY, f64::max);
        let lexical_range = lexical_max - lexical_min;
        let semantic_min = scored
            .iter()
            .map(|(_, similarity)| *similarity)
            .fold(f32::INFINITY, f32::min);
        let semantic_max = scored
            .iter()
            .map(|(_, similarity)| *similarity)
            .fold(f32::NEG_INFINITY, f32::max);
        let semantic_range = semantic_max - semantic_min;
        let semantic_weight = config.semantic_rerank_weight.clamp(0.0, 1.0);

        if !lexical_range.is_finite() || lexical_range <= f64::EPSILON
            || !semantic_range.is_finite()
            || semantic_range <= f32::EPSILON
            || semantic_weight <= f64::EPSILON
        {
            return scored.len();
        }

        let lexical_weight = 1.0 - semantic_weight;
        for (index, similarity) in scored.iter().copied() {
            let candidate = &mut results[index];
            let lexical_score = candidate.score - candidate.intent_rerank_bonus;
            let lexical_quality = ((lexical_score - lexical_min) / lexical_range).clamp(0.0, 1.0);
            let semantic_quality =
                ((similarity - semantic_min) / semantic_range).clamp(0.0, 1.0) as f64;
            let fused_quality =
                lexical_quality * lexical_weight + semantic_quality * semantic_weight;
            let fused_score = lexical_min + fused_quality * lexical_range;
            let final_score = fused_score + candidate.intent_rerank_bonus;
            let delta = final_score - candidate.score;
            candidate.score = final_score;
            candidate.rerank_bonus += delta;
        }

        scored.len()
    }

    fn merge_semantic_candidates(
        &self,
        results: &mut Vec<ScoredMemoryCandidate>,
        query_embedding: &[f32],
        limit: usize,
    ) -> (usize, usize) {
        let config = self.config.semantic.resolved();
        let candidate_limit = config.candidate_limit.max(limit);
        let normalized_query = match StoredSemanticVector::normalized_query(query_embedding) {
            Ok(query) => query,
            Err(error) => {
                tracing::debug!(error = %error, "Semantic query skipped");
                return (0, 0);
            }
        };
        let candidate_ids = match self.semantic_index.read() {
            Ok(index) => match index.query_candidate_ids(query_embedding, candidate_limit) {
                Ok(candidate_ids) => candidate_ids,
                Err(error) => {
                    tracing::debug!(error = %error, "Semantic query skipped");
                    return (0, 0);
                }
            },
            Err(_) => return (0, 0),
        };
        let mut semantic_candidates = candidate_ids
            .into_iter()
            .filter_map(|memory_id| {
                let memory = self.memories.get(&memory_id)?;
                let vector = memory.semantic_vector.as_ref()?;
                let similarity = vector.cosine_similarity_normalized(&normalized_query).ok()?;
                Some((memory_id, similarity))
            })
            .collect::<Vec<_>>();
        semantic_candidates.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        semantic_candidates.truncate(candidate_limit);

        let mut existing = results
            .iter()
            .map(|candidate| candidate.memory_id)
            .collect::<HashSet<_>>();
        let mut new_count = 0;

        for (memory_id, similarity) in semantic_candidates.iter().copied() {
            if let Some(candidate) = results
                .iter_mut()
                .find(|candidate| candidate.memory_id == memory_id)
            {
                candidate.semantic_similarity = similarity;
                candidate.score +=
                    similarity as f64 * config.semantic_score_multiplier;
                continue;
            }

            if !existing.insert(memory_id) {
                continue;
            }
            let Some(memory) = self.memories.get(&memory_id) else {
                continue;
            };
            let reinforcement_score = if memory.stats.get_reinforcement_count() > 0 {
                (memory.stats.get_reinforcement_count() as f64).log10()
            } else {
                0.0
            };
            results.push(ScoredMemoryCandidate {
                memory_id,
                score: similarity as f64 * config.semantic_score_multiplier,
                match_integrity: similarity.max(0.0) as f64,
                intersection_count: 0,
                recency_score: 0.0,
                reinforcement_score,
                salience_score: 0.0,
                created_at: memory.created_at,
                intersection_weighted: 0.0,
                match_count: 0.0,
                rerank_bonus: 0.0,
                generic_penalty: 1.0,
                semantic_similarity: similarity,
                intent_compatibility: 0.0,
                intent_rerank_bonus: 0.0,
            });
            new_count += 1;
        }

        (new_count, semantic_candidates.len())
    }

    fn apply_semantic_reranker(
        &self,
        results: &mut [ScoredMemoryCandidate],
        _query_embedding: &[f32],
    ) {
        let config = &self.config.semantic;
        if !config.reranker_enabled || config.reranker_weights.is_empty() {
            return;
        }
        let model = LinearReranker::from_config(config);
        for candidate in results {
            let features = [
                (candidate.score / self.tuning.intersection_score_multiplier)
                    .clamp(-10.0, 10.0) as f32,
                candidate.semantic_similarity,
                candidate.match_integrity.clamp(0.0, 1.0) as f32,
                (candidate.intersection_count as f32 / 8.0).min(1.0),
                candidate.recency_score.clamp(0.0, 1.0) as f32,
                (candidate.salience_score / 10.0).clamp(0.0, 1.0) as f32,
            ];
            let delta = model.score(&features) as f64 * config.reranker_scale;
            candidate.rerank_bonus += delta;
            candidate.score += delta;
        }
    }

    fn apply_intent_reranker(
        &self,
        results: &mut [ScoredMemoryCandidate],
        query_intent: &IntentClassification,
    ) {
        let config = self.config.semantic.resolved();
        if !config.intent_rerank_enabled
            || !query_intent.is_recall_intent()
            || results.len() < 2
        {
            return;
        }
        let lexical_min = results
            .iter()
            .map(|candidate| candidate.score - candidate.intent_rerank_bonus)
            .fold(f64::INFINITY, f64::min);
        let lexical_max = results
            .iter()
            .map(|candidate| candidate.score - candidate.intent_rerank_bonus)
            .fold(f64::NEG_INFINITY, f64::max);
        let lexical_range = lexical_max - lexical_min;
        if !lexical_range.is_finite() || lexical_range <= f64::EPSILON {
            return;
        }

        let query_weight = f64::from(query_intent.confidence_weight);
        for candidate in results {
            let Some(memory) = self.memories.get(&candidate.memory_id) else {
                continue;
            };
            let Some(memory_intent) = memory.intent_classification.as_ref() else {
                continue;
            };
            if memory_intent.taxonomy_version != crate::intent::INTENT_TAXONOMY_VERSION
                || memory_intent.model_version != config.model_version
            {
                continue;
            }
            let compatibility = intent_compatibility(query_intent, memory_intent);
            let memory_weight = f64::from(memory_intent.confidence_weight);
            let positive = compatibility
                * query_weight
                * memory_weight
                * config.intent_rerank_weight;
            let no_recall_penalty = if memory_intent.memory_eligible {
                0.0
            } else {
                // Suppression should be strong only when both sides are
                // decisive. A low-margin query must not apply a categorical
                // penalty to every action/chitchat memory.
                query_weight * memory_weight * config.intent_no_recall_penalty
            };
            let delta = ((positive - no_recall_penalty) * lexical_range)
                .clamp(-config.intent_rerank_max_delta, config.intent_rerank_max_delta);
            candidate.intent_compatibility = compatibility;
            candidate.intent_rerank_bonus = delta;
            candidate.score += delta;
            candidate.rerank_bonus += delta;
        }
    }

    fn consolidated_search(
        &self,
        query_cues: &[(String, f64)],
        limit: usize,
        explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
        collect_detailed_timing: bool,
    ) -> (Vec<ScoredMemoryCandidate>, ConsolidatedSearchTiming) {
        let mut timing = ConsolidatedSearchTiming::default();
        if query_cues.is_empty() {
            return (Vec::new(), timing);
        }

        // 1. Gather cue data with set sizes for sorting
        let mut cue_data = Vec::with_capacity(query_cues.len());
        let total_memories = self.memories.len() as f64;

        for (cue, weight) in query_cues {
            if let Some(ordered_set) = self.cue_index.get(cue) {
                // IDF Weighting (BM25 variant): Penalize common cues, boost rare ones
                // BM25's IDF accounts for the complement (memories WITHOUT this cue),
                // making it much more aggressive at demoting high-frequency cues.
                // e.g. at df=40% of corpus: old formula gave 0.91, BM25 gives 0.40
                let df = ordered_set.len() as f64;
                let idf = ((total_memories - df + 0.5) / (df + 0.5))
                    .ln()
                    .max(self.tuning.idf_threshold_percent);
                let adjusted_weight = weight * idf;

                cue_data.push((cue.clone(), adjusted_weight, ordered_set));
            }
        }

        if cue_data.is_empty() {
            return (Vec::new(), timing);
        }
        timing.cue_count_with_postings = cue_data.len();
        timing.max_posting_len = cue_data
            .iter()
            .map(|(_, _, set)| set.len())
            .max()
            .unwrap_or(0);

        // OPTIMIZATION 1: Sort by set size (smallest first)
        // Processing rarer cues first produces fewer candidates to probe
        cue_data.sort_by(|a, b| a.2.len().cmp(&b.2.len()));

        // OPTIMIZATION 2: Adaptive scan limit based on requested limit
        // For limit=5, we don't need to scan 10k items per cue
        // Scale: limit * factor, capped at max for safety
        let adaptive_scan_limit =
            (limit * self.tuning.adaptive_scan_factor).min(self.tuning.adaptive_scan_max);
        timing.adaptive_scan_limit = adaptive_scan_limit;

        // 2. Perform Union-based search with O(1) Probing
        let phase_start = Instant::now();
        let mut candidates = Vec::new();
        let mut seen_memories = HashSet::new();

        for (cue_idx, (cue, _weight, set)) in cue_data.iter().enumerate() {
            let scan_limit = std::cmp::min(set.len(), adaptive_scan_limit);
            timing.scanned_posting_count += scan_limit;
            let items = set.get_recent(Some(scan_limit));

            for (pos_rev, memory_id) in items.iter().enumerate() {
                // If we've already processed this memory from a previous cue, skip it
                if seen_memories.contains(memory_id) {
                    continue;
                }
                seen_memories.insert(*memory_id);

                let mut total_weight = 0.0;
                let mut positions_info = Vec::with_capacity(cue_data.len());

                // 3. For each NEW candidate, probe ALL query cue lists to get full intersection data
                for (other_idx, (_other_cue, other_weight, other_set)) in
                    cue_data.iter().enumerate()
                {
                    // Optimization: if it's the current set we're iterating, we know it's there
                    if other_idx == cue_idx {
                        total_weight += *other_weight;
                        positions_info.push((
                            pos_rev,
                            other_set.len(),
                            *other_weight,
                            is_generated_memory_facet(cue),
                        ));
                        continue;
                    }

                    // O(1) probe into other sets
                    if let Some(oldest_idx) = other_set.get_index_of(*memory_id) {
                        total_weight += *other_weight;
                        let recency_pos = (other_set.len() - 1) - oldest_idx;
                        positions_info.push((
                            recency_pos,
                            other_set.len(),
                            *other_weight,
                            is_generated_memory_facet(_other_cue),
                        ));
                    }
                }

                // 4. Collect candidate
                candidates.push((*memory_id, positions_info, total_weight));
            }
        }
        timing.candidate_generation_ms = phase_start.elapsed().as_secs_f64() * 1000.0;
        timing.candidate_count = candidates.len();

        // 5. Score candidates
        let phase_start = Instant::now();
        let results = self.score_consolidated_candidates(
            candidates,
            query_cues,
            explain,
            disable_salience_bias,
            heatmap,
            mandatory_cues,
            collect_detailed_timing,
        );
        timing.candidate_scoring_ms = phase_start.elapsed().as_secs_f64() * 1000.0;
        timing.scored_candidate_count = results.0.len();
        timing.scoring_filter_ms = results.1.filter_ms;
        timing.scoring_position_ms = results.1.position_ms;
        timing.scoring_salience_ms = results.1.salience_ms;
        timing.scoring_structured_ms = results.1.structured_ms;
        timing.scoring_finalize_ms = results.1.finalize_ms;

        (results.0, timing)
    }

    fn score_consolidated_candidates(
        &self,
        candidates: Vec<(MemoryId, Vec<(usize, usize, f64, bool)>, f64)>,
        query_cues: &[(String, f64)],
        _explain: bool,
        disable_salience_bias: bool,
        heatmap: Option<&HashMap<String, f32>>,
        mandatory_cues: Option<&Vec<String>>,
        collect_detailed_timing: bool,
    ) -> (Vec<ScoredMemoryCandidate>, CandidateScoringTiming) {
        let max_rec_weight = self.tuning.max_rec_weight;
        let max_freq_weight = self.tuning.max_freq_weight;
        let query_profile = build_query_scoring_profile(query_cues);
        let active_heatmap = heatmap.and_then(|map| if map.is_empty() { None } else { Some(map) });
        let salience_now = if disable_salience_bias {
            0
        } else {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        };

        let mut results = Vec::with_capacity(candidates.len());
        let mut timing = CandidateScoringTiming::default();

        for (memory_id, positions_info, total_weight) in candidates {
            if let Some(memory) = self.memories.get(&memory_id) {
                let mut detail_start = if collect_detailed_timing {
                    Some(Instant::now())
                } else {
                    None
                };
                let features = memory_scoring_features(memory.value());
                // Strict Mandatory Filter
                if let Some(mandatory) = mandatory_cues {
                    let mut missing_mandatory = false;
                    for m_cue in mandatory {
                        // Strict check, MUST exist literally in memory's cues
                        if !memory.cues.contains(m_cue) {
                            missing_mandatory = true;
                            break;
                        }
                    }
                    if missing_mandatory {
                        continue;
                    }
                }

                if let Some(start) = detail_start.as_mut() {
                    timing.filter_ms += start.elapsed().as_secs_f64() * 1000.0;
                    *start = Instant::now();
                }
                let mut total_recency = 0.0;
                let mut total_w_rec = 0.0;
                let mut total_w_freq = 0.0;
                let match_count = positions_info.len() as f64;

                for (pos, list_len, _weight, _) in &positions_info {
                    let pos_f64 = *pos as f64;
                    let list_len_f64 = *list_len as f64;
                    let sigma = list_len_f64.sqrt();
                    let ratio = pos_f64 / sigma;

                    let w_rec = max_rec_weight / (ratio + 1.0);
                    let w_freq = 1.0 + (max_freq_weight * (1.0 - (1.0 / (ratio + 1.0))));

                    let recency_component = 1.0 / (pos_f64 + 1.0);

                    total_recency += recency_component; // Independent of IDF weight
                    total_w_rec += w_rec;
                    total_w_freq += w_freq;
                }

                let avg_w_rec = total_w_rec / match_count;
                let avg_w_freq = total_w_freq / match_count;
                let recency_score = total_recency / match_count;
                if let Some(start) = detail_start.as_mut() {
                    timing.position_ms += start.elapsed().as_secs_f64() * 1000.0;
                    *start = Instant::now();
                }

                let frequency_score = if memory.stats.get_reinforcement_count() > 0 {
                    (memory.stats.get_reinforcement_count() as f64).log10()
                } else {
                    0.0
                };

                let (_salience_score, _effective_salience, _market_lift) = if disable_salience_bias
                {
                    (0.0, 0.0, 0.0)
                } else {
                    let eff = memory.stats.get_effective_salience(salience_now);

                    let mut lift = 0.0;
                    if let Some(map) = active_heatmap {
                        for cue in &memory.cues {
                            if let Some(val) = map.get(cue) {
                                lift += *val as f64;
                            }
                        }
                    }

                    (eff + lift, eff, lift)
                };
                if let Some(start) = detail_start.as_mut() {
                    timing.salience_ms += start.elapsed().as_secs_f64() * 1000.0;
                    *start = Instant::now();
                }

                // BM25-lite Length Normalization
                let b = 1.0;
                let k1 = 2.0;
                let avg_cues = 40.0; // Estimated average cue list length
                let scored_cue_len = features.scored_cue_len;
                let len_f64 = scored_cue_len as f64;

                let bm25_len_penalty = 1.0 - b + b * (len_f64 / avg_cues);
                // With TF=1 for all matching cues, the BM25 formula simplifies to a single tf_component
                // Note: we're applying this to total_weight (which is sum of IDFs)
                let bm25_tf_component = (k1 + 1.0) / (1.0 + k1 * bm25_len_penalty);

                let intersection_score =
                    total_weight * bm25_tf_component * self.tuning.intersection_score_multiplier;

                let mut structured_match_seen = false;
                let mut person_structured_match_seen = false;
                let mut quantity_structured_match_seen = false;
                let mut inventory_structured_match_seen = false;
                let mut travel_structured_match_seen = false;
                let mut age_structured_match_seen = false;
                let mut education_structured_match_seen = false;
                let mut family_structured_match_seen = false;
                let mut family_count_structured_match_seen = false;
                let mut source_role_structured_match_seen = false;
                let mut source_time_structured_match_seen = false;
                let mut update_structured_match_seen = false;
                let mut non_source_structured_match_seen = false;
                let mut rerank_bonus = 0.0;
                let lexical_match_count = positions_info
                    .iter()
                    .filter(|(_, _, _, generated)| !*generated)
                    .count() as f64;
                let mut strong_lexical_match_seen = false;
                let mut strong_structured_match_seen = false;
                let allow_structured_rerank = !query_profile.lexical_query_seen
                    || lexical_match_count > 0.0
                    || query_profile.strong_structured_query_seen;
                for query_cue in &query_profile.cues {
                    if query_cue.prefix.is_none() {
                        continue;
                    }
                    if query_cue.family_mask != 0
                        && features.structured_family_mask & query_cue.family_mask == 0
                    {
                        continue;
                    }
                    let exact_match = memory
                        .cues
                        .iter()
                        .any(|memory_cue| memory_cue == query_cue.cue);
                    if !exact_match {
                        continue;
                    }

                    structured_match_seen = true;
                    if query_cue.weight >= 0.9 && !query_cue.generated {
                        strong_lexical_match_seen = true;
                    }
                    if query_cue.weight >= 3.0 && query_cue.semantic_generated {
                        strong_structured_match_seen = true;
                    }
                    if query_cue.family_mask & FAMILY_PERSON != 0 {
                        person_structured_match_seen = true;
                    }
                    if query_cue.family_mask & FAMILY_QUANTITY != 0 {
                        quantity_structured_match_seen = true;
                    }
                    if query_cue.family_mask & FAMILY_INVENTORY != 0 {
                        inventory_structured_match_seen = true;
                    }
                    if query_cue.family_mask & FAMILY_TRAVEL != 0 {
                        travel_structured_match_seen = true;
                    }
                    if query_cue.family_mask & FAMILY_AGE != 0 {
                        age_structured_match_seen = true;
                    }
                    if query_cue.family_mask & FAMILY_EDUCATION != 0 {
                        education_structured_match_seen = true;
                    }
                    if query_cue.family_mask & FAMILY_FAMILY != 0 {
                        family_structured_match_seen = true;
                    }
                    if query_cue.family_mask & FAMILY_FAMILY_COUNT != 0 {
                        family_count_structured_match_seen = true;
                    }
                    if query_cue.prefix == Some("source_role") && query_cue.weight >= 3.0 {
                        source_role_structured_match_seen = true;
                    }
                    if query_cue.family_mask & FAMILY_SOURCE_TIME != 0 {
                        source_time_structured_match_seen = true;
                    }
                    if query_cue.family_mask & FAMILY_NON_SOURCE_STRUCTURED != 0 {
                        non_source_structured_match_seen = true;
                    }
                    if query_cue.cue == "type:update" {
                        update_structured_match_seen = true;
                    }
                    if allow_structured_rerank
                        || query_cue.is_family_prefix
                        || query_cue.is_source_time_prefix
                    {
                        rerank_bonus += query_cue.weight * query_cue.rerank_multiplier;
                    }
                }
                if let Some(start) = detail_start.as_mut() {
                    timing.structured_ms += start.elapsed().as_secs_f64() * 1000.0;
                    *start = Instant::now();
                }
                let generic_penalty = if query_profile.update_structured_query_seen
                    && !update_structured_match_seen
                {
                    0.5
                } else if query_profile.family_count_structured_query_seen
                    && !family_count_structured_match_seen
                    && family_structured_match_seen
                {
                    0.55
                } else if query_profile.family_structured_query_seen && family_structured_match_seen {
                    1.0
                } else if query_profile.source_role_structured_query_seen && !source_role_structured_match_seen {
                    0.03
                } else if query_profile.source_time_structured_query_seen && source_time_structured_match_seen {
                    if lexical_match_count == 0.0 && !non_source_structured_match_seen {
                        0.35
                    } else {
                        1.0
                    }
                } else if query_profile.source_time_structured_query_seen
                    && !source_time_structured_match_seen
                    && lexical_match_count == 0.0
                {
                    0.35
                } else if query_profile.strong_structured_query_seen
                    && !strong_structured_match_seen
                {
                    0.45
                } else if query_profile.lexical_query_seen
                    && lexical_match_count == 0.0
                    && structured_match_seen
                    && !strong_structured_match_seen
                {
                    0.25
                } else if query_profile.strong_lexical_query_seen
                    && !strong_lexical_match_seen
                    && structured_match_seen
                    && match_count <= 4.0
                {
                    0.55
                } else if query_profile.person_structured_query_seen
                    && !person_structured_match_seen
                    && match_count <= 2.0
                {
                    0.45
                } else if query_profile.quantity_structured_query_seen
                    && !quantity_structured_match_seen
                    && match_count <= 2.0
                {
                    0.55
                } else if query_profile.inventory_structured_query_seen
                    && !inventory_structured_match_seen
                    && match_count <= 3.0
                {
                    0.55
                } else if query_profile.travel_structured_query_seen
                    && !travel_structured_match_seen
                    && match_count <= 2.0
                {
                    0.55
                } else if query_profile.age_structured_query_seen
                    && !age_structured_match_seen
                    && match_count <= 3.0
                {
                    0.25
                } else if query_profile.education_structured_query_seen
                    && !education_structured_match_seen
                    && !age_structured_match_seen
                    && match_count <= 3.0
                {
                    0.6
                } else if query_profile.family_structured_query_seen
                    && !family_structured_match_seen
                    && match_count <= 3.0
                {
                    0.45
                } else if query_profile.source_time_structured_query_seen
                    && !source_time_structured_match_seen
                    && match_count <= 3.0
                {
                    0.45
                } else if query_profile.structured_query_seen && !structured_match_seen && match_count <= 2.0 {
                    0.85
                } else {
                    1.0
                };

                // Final score includes salience plus deterministic facet reranking.
                // We use salience_score (Effective + Market) here.
                let base_score = intersection_score
                    + (recency_score * avg_w_rec)
                    + (frequency_score * avg_w_freq)
                    + (_salience_score * self.tuning.salience_score_multiplier);
                let score = (base_score + rerank_bonus) * generic_penalty;

                // Match integrity calculation
                // 1. Intersection strength (relative to match count)
                let intersection_strength = total_weight / match_count.max(1.0);
                // 2. Context agreement: how many of the memory's cues matched the query
                let context_agreement = if !memory.cues.is_empty() {
                    (match_count / (scored_cue_len as f64)).min(1.0)
                } else {
                    0.0
                };
                // 3. Reinforcement boost (capped)
                let reinforcement_boost = (frequency_score / 2.0).min(1.0);
                let match_integrity = (intersection_strength * 0.5
                    + context_agreement * 0.3
                    + reinforcement_boost * 0.2)
                    .min(1.0);

                results.push(ScoredMemoryCandidate {
                    memory_id,
                    score,
                    match_integrity,
                    intersection_count: match_count as usize,
                    recency_score,
                    reinforcement_score: frequency_score,
                    salience_score: _salience_score,
                    created_at: memory.created_at,
                    intersection_weighted: total_weight,
                    match_count,
                    rerank_bonus,
                    generic_penalty,
                    semantic_similarity: 0.0,
                    intent_compatibility: 0.0,
                    intent_rerank_bonus: 0.0,
                });
                if let Some(start) = detail_start.as_mut() {
                    timing.finalize_ms += start.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }

        (results, timing)
    }

    pub fn get_memory(&self, memory_id: MemoryId) -> Option<Memory<T>> {
        self.memories.get(&memory_id).map(|m| m.clone())
    }

    // Consolidate Memory function removed from generic implementation
    // It requires specific knowledge of how to merge T
    // Will be re-implemented in specialized impl blocks if needed

    pub fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        stats.insert(
            "total_memories".to_string(),
            serde_json::json!(self.memory_count.load(Ordering::Relaxed)),
        );
        stats.insert(
            "total_cues".to_string(),
            serde_json::json!(self.cue_count.load(Ordering::Relaxed)),
        );
        stats.insert(
            "cue_global_counts".to_string(),
            serde_json::json!(self.cue_global_counts.len()),
        );

        stats
    }

}

// ==================================================================================
// Specialized Implementation for "Brain" (MainStats)
// ==================================================================================

impl CueMapEngine<MainStats> {
    /// Aggregate recently reinforced memory activity into cue-level heat.
    /// This drives the market heatmap from actual recalled/boosted memories,
    /// not from lexicon identity entries.
    pub fn get_trending_cues(&self, limit: usize) -> Vec<(String, f64)> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let window_start = now.saturating_sub(15 * 60);
        let mut cue_heat: HashMap<String, f64> = HashMap::new();

        for entry in self.memories.iter() {
            let memory = entry.value();
            let stats = &memory.stats;
            if stats.last_boosted_at < window_start || stats.dynamic_salience <= 0.0 {
                continue;
            }

            let age_seconds = now.saturating_sub(stats.last_boosted_at) as f64;
            let recency = (-age_seconds / (15.0 * 60.0)).exp();
            let heat = stats.dynamic_salience * recency;
            if heat <= 0.0 {
                continue;
            }

            for cue in &memory.cues {
                if cue.starts_with("id:")
                    || cue.starts_with("memory_id:")
                    || cue.starts_with("path:")
                    || cue.starts_with("file:")
                    || cue.starts_with("episode:")
                    || cue.starts_with("chunk_idx:")
                {
                    continue;
                }
                *cue_heat.entry(cue.clone()).or_insert(0.0) += heat;
            }
        }

        let mut trending: Vec<(String, f64)> = cue_heat.into_iter().collect();
        trending.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        trending.truncate(limit);
        trending
    }

    /// Decays dynamic salience for all memories and updates generic salience proxy
    pub fn decay_salience(&self, decay_rate: f64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for mut memory in self.memories.iter_mut() {
            let stats = &mut memory.value_mut().stats;
            let time_delta = now.saturating_sub(stats.last_boosted_at);

            // Simple exponential decay: N(t) = N0 * e^(-lambda * t)
            // We use hours as time unit
            let hours_passed = (time_delta as f64) / 3600.0;
            if hours_passed > 0.1 {
                let decay_factor = (-decay_rate * hours_passed).exp();
                stats.dynamic_salience *= decay_factor;

                // Clamp near zero
                if stats.dynamic_salience < 0.01 {
                    stats.dynamic_salience = 0.0;
                }
            }
        }
    }

    /// Reinforces memory by adding dynamic heat (Brain logic)
    /// Algorithm: Inverse Proportional Boosting
    /// NewScore = Current + (Amount / (1.0 + Current))
    /// This prevents "Context Poisoning" (log explosion) from high-frequency events.
    pub fn reinforce_dynamic(&self, memory_id: MemoryId, amount: f64) {
        if let Some(mut memory) = self.memories.get_mut(&memory_id) {
            memory.touch();
            let stats = &mut memory.stats;

            // Logarithmic Saturation
            stats.dynamic_salience += amount / (1.0 + stats.dynamic_salience);

            stats.reinforcement_count += 1;
            stats.last_boosted_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
    }

    /// Calculates "Effective Importance" by combining Intrinsic + Decayed Dynamic + Market Heatmap
    pub fn score_with_decay_and_market(
        &self,
        candidate_ids: Vec<MemoryId>,
        heatmap: &HashMap<String, f32>,
    ) -> Vec<RecallResult> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut results = Vec::new();

        for id in candidate_ids {
            if let Some(mem) = self.memories.get(&id) {
                let stats = &mem.stats;

                // 1. Effective Salience (Intrinsic + Decayed Dynamic) - centralized in MemoryStats trait
                let effective_salience = stats.get_effective_salience(now);

                // 2. Market Heatmap Logic
                // Logic: Sum of heatmap values for cues found in this memory
                let mut market_lift = 0.0;
                for cue in &mem.cues {
                    if let Some(score) = heatmap.get(cue) {
                        market_lift += *score as f64;
                    }
                }

                // 3. Aggregate Total Salience
                let total_salience = effective_salience + market_lift;

                // Construct result
                results.push(RecallResult {
                    memory_id: mem.id,
                    content: self
                        .read_memory_content(mem.value())
                        .unwrap_or_else(|_| "<decryption failed>".to_string()),
                    score: total_salience,
                    match_integrity: 1.0,
                    intersection_count: 0,
                    recency_score: effective_salience,
                    reinforcement_score: stats.reinforcement_count as f64,
                    salience_score: total_salience,
                    created_at: mem.created_at,
                    metadata: mem.metadata.clone(),
                    explain: Some(serde_json::json!({
                        "intrinsic": stats.intrinsic_salience,
                        "effective_salience": effective_salience,
                        "market_lift": market_lift,
                        "total_salience": total_salience
                    })),
                });
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Prune memories with low salience (Brain Plasticity)
    pub fn prune_low_salience(&self, threshold: f64) -> usize {
        let mut to_remove = Vec::new();

        for entry in self.memories.iter() {
            let stats = &entry.value().stats;
            // Total effective salience
            let total_salience = stats.intrinsic_salience + stats.dynamic_salience;

            // Protect high reinforcement memories from pruning even if cold?
            // Maybe not, if unused for a LONG time.

            if total_salience < threshold && stats.reinforcement_count < 5 {
                to_remove.push(entry.key().clone());
            }
        }

        let count = to_remove.len();
        for id in to_remove {
            self.delete_memory(id);
        }

        count
    }

    /// Consolidate memories - specialized for MainStats
    pub fn consolidate_memories(&self, cue_overlap_threshold: f64) -> Vec<(MemoryId, Vec<MemoryId>)> {
        let mut to_merge = Vec::new();
        let mut seen = HashSet::new();

        // 1. Find overlapping memories (Naive)
        for entry in self.memories.iter() {
            let (id_a, mem_a) = entry.pair();
            if seen.contains(id_a) {
                continue;
            }

            // Skip already consolidated memories to avoid recursion
            if mem_a
                .metadata
                .get("consolidated")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                continue;
            }

            let mut group = vec![*id_a];

            if let Some(first_cue) = mem_a.cues.first() {
                if let Some(ordered_set) = self.cue_index.get(first_cue) {
                    for id_b in ordered_set.get_recent(None) {
                        if *id_a == id_b || seen.contains(&id_b) {
                            continue;
                        }

                        if let Some(mem_b) = self.memories.get(&id_b) {
                            if mem_b
                                .metadata
                                .get("consolidated")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                            {
                                continue;
                            }

                            let cues_a: HashSet<_> = mem_a.cues.iter().collect();
                            let cues_b: HashSet<_> = mem_b.cues.iter().collect();

                            let intersection = cues_a.intersection(&cues_b).count();
                            let union = cues_a.union(&cues_b).count();

                            if union > 0 {
                                let similarity = (intersection as f64) / (union as f64);
                                if similarity >= cue_overlap_threshold {
                                    group.push(id_b);
                                }
                            }
                        }
                    }
                }
            }

            if group.len() > 1 {
                for id in &group {
                    seen.insert(*id);
                }
                to_merge.push(group);
            }
        }

        let mut results = Vec::new();

        // 2. Merge
        for group in to_merge {
            let mut combined_content = String::new();
            let mut combined_cues = HashSet::new();

            // MainStats aggregation
            let mut total_intrinsic = 0.0;
            let mut max_dynamic: f64 = 0.0;
            let mut total_reinforcement = 0;

            for id in &group {
                if let Some(mem) = self.memories.get(id) {
                    if !combined_content.is_empty() {
                        combined_content.push_str("\n---\n");
                    }
                    if let Ok(c) = self.read_memory_content(mem.value()) {
                        combined_content.push_str(&c);
                    }
                    for cue in &mem.cues {
                        combined_cues.insert(cue.clone());
                    }

                    total_intrinsic += mem.stats.intrinsic_salience;
                    max_dynamic = max_dynamic.max(mem.stats.dynamic_salience);
                    total_reinforcement += mem.stats.reinforcement_count;
                }
            }

            let mut summary_content = format!("[Consolidated Memory]\n{}", combined_content);
            if summary_content.len() > 1000 {
                summary_content.truncate(1000);
                summary_content.push_str("... [truncated]");
            }

            let mut metadata = HashMap::new();
            metadata.insert("consolidated".to_string(), serde_json::json!(true));
            metadata.insert("original_count".to_string(), serde_json::json!(group.len()));

            let mut cues_vec: Vec<String> = combined_cues.into_iter().collect();
            cues_vec.push("type:summary".to_string());

            // Create stats
            let new_stats = MainStats {
                intrinsic_salience: (total_intrinsic / group.len() as f64) * 1.2, // Boost consolidated intrinsic
                dynamic_salience: max_dynamic, // Keep urgency of most urgent part
                last_boosted_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                reinforcement_count: total_reinforcement,
            };

            let new_id =
                self.add_memory(summary_content, cues_vec, Some(metadata), new_stats, false);
            results.push((new_id, group));
        }

        results
    }

    /// Extract all unique symbols (un-prefixed values) from structural cues in the index.
    /// E.g., "defines_function:my_func" -> "my_func"
    pub fn get_all_symbols(&self) -> HashSet<String> {
        let mut symbols = HashSet::new();
        let prefixes = [
            "defines_function:",
            "defines_class:",
            "defines_method:",
            "defines_struct:",
            "defines_enum:",
            "defines_trait:",
            "defines_type:",
            "defines_interface:",
            "calls_function:",
            "calls_method:",
            "imports_module:",
            "creates_object:",
        ];

        for entry in self.cue_index.iter() {
            let cue = entry.key();
            for prefix in prefixes {
                if cue.starts_with(prefix) {
                    let symbol = &cue[prefix.len()..];
                    if !symbol.is_empty() {
                        symbols.insert(symbol.to_string());
                    }
                    break;
                }
            }
        }
        symbols
    }
}

// ==================================================================================
// Specialized Implementation for "Dictionary" (LexiconStats)
// ==================================================================================

impl CueMapEngine<LexiconStats> {
    /// Tiered Reinforcement for Dictionary (Minute/Daily Buckets)
    pub fn reinforce_tiered(&self, memory_id: MemoryId, amount: u64) {
        if let Some(mut memory) = self.memories.get_mut(&memory_id) {
            memory.touch();
            let stats = &mut memory.stats;
            stats.total_count += amount;
            stats.last_reinforced = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Calculate buckets
            let now_mins = (stats.last_reinforced / 60) as u32;
            let now_day = (stats.last_reinforced / 86400) as u32;

            *stats.minute_stats.entry(now_mins).or_insert(0) += amount as u16;
            *stats.daily_stats.entry(now_day).or_insert(0) += amount as u32;

            // Cleanup old minute buckets (keep last 60 mins)
            let min_threshold = now_mins.saturating_sub(60);
            stats.minute_stats.retain(|&k, _| k >= min_threshold);
        }
    }

    /// Trending identification (Spike detection)
    /// Sums bucket counts in window to identify trending cues
    pub fn get_trending_items(&self, limit: usize) -> Vec<(MemoryId, f64)> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let current_min = (now / 60) as u32;
        // Look at last 15 mins
        let window_start = current_min.saturating_sub(15);

        let mut trending = Vec::new();

        for entry in self.memories.iter() {
            let stats = &entry.value().stats;

            // Calculate velocity in window
            let mut recent_velocity = 0.0;
            for (min, count) in &stats.minute_stats {
                if *min >= window_start {
                    recent_velocity += *count as f64;
                }
            }

            if recent_velocity >= 1.0 {
                // Normalize by baseline (total count / age?) or simply raw velocity for "Trending Now"
                trending.push((*entry.key(), recent_velocity));
            }
        }

        trending.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Log top 5 items with bucket breakdown
        if !trending.is_empty() {
            tracing::info!(
                "Trending: Identification finished. Found {} candidates >= 1.0 velocity",
                trending.len()
            );
            for i in 0..trending.len().min(5) {
                let (cue, velocity) = &trending[i];
                if let Some(entry) = self.memories.get(cue) {
                    let stats = &entry.stats;
                    let m_count = stats.minute_stats.len();
                    let d_count = stats.daily_stats.len();
                    let total = stats.total_count;
                    tracing::info!("Trending: [Rank {}] Cue '{}' velocity={:.2}. Stats: totals={}, min_buckets={}, day_buckets={}, last_ref={}",
                        i+1, cue, velocity, total, m_count, d_count, stats.last_reinforced);
                }
            }
        }

        trending.truncate(limit);

        trending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::SemanticConfig;
    use std::collections::BTreeMap;

    fn classification(
        primary_intent: &str,
        confidence_weight: f32,
        memory_eligible: bool,
    ) -> IntentClassification {
        let mut scores = BTreeMap::new();
        for label in crate::intent::INTENT_LABELS {
            scores.insert(label.to_string(), if label == primary_intent { 1.0 } else { 0.0 });
        }
        IntentClassification {
            primary_intent: primary_intent.to_string(),
            scores,
            top_intents: vec![primary_intent.to_string()],
            top_score: 1.0,
            margin: 1.0,
            confidence_weight,
            recall_eligible: memory_eligible,
            recall_action: if memory_eligible {
                "recall".to_string()
            } else {
                "no_recall".to_string()
            },
            memory_eligible,
            model_version: SemanticConfig::default().model_version,
            taxonomy_version: INTENT_TAXONOMY_VERSION.to_string(),
        }
    }

    fn candidate(memory_id: MemoryId, score: f64) -> ScoredMemoryCandidate {
        ScoredMemoryCandidate {
            memory_id,
            score,
            match_integrity: 1.0,
            intersection_count: 1,
            recency_score: 0.0,
            reinforcement_score: 0.0,
            salience_score: 0.0,
            created_at: 0.0,
            intersection_weighted: score,
            match_count: 1.0,
            rerank_bonus: 0.0,
            generic_penalty: 1.0,
            semantic_similarity: 0.0,
            intent_compatibility: 0.0,
            intent_rerank_bonus: 0.0,
        }
    }

    #[test]
    fn intent_rerank_penalty_is_confidence_scaled_and_bounded() {
        let mut engine = CueMapEngine::<MainStats>::new();
        let mut config = SemanticConfig::default();
        config.intent_rerank_weight = 1.0;
        config.intent_no_recall_penalty = 1.0;
        config.intent_rerank_max_delta = 32.0;
        engine.set_semantic_config(config);

        let memory_id = engine.add_memory(
            "Run the deployment command".to_string(),
            vec!["deployment".to_string()],
            None,
            MainStats::default(),
            true,
        );
        engine.attach_intent_classification(
            memory_id,
            classification("action_or_command", 1.0, false),
        );

        let query = classification("event_or_plan", 1.0, true);
        let mut results = vec![candidate(memory_id, 1000.0), candidate(INVALID_MEMORY_ID, 0.0)];
        engine.apply_intent_reranker(&mut results, &query);
        assert_eq!(results[0].intent_rerank_bonus, -32.0);

        let uncertain_query = classification("event_or_plan", 0.1, true);
        let mut results = vec![candidate(memory_id, 100.0), candidate(INVALID_MEMORY_ID, 0.0)];
        engine.apply_intent_reranker(&mut results, &uncertain_query);
        assert!(results[0].intent_rerank_bonus > -32.0);
    }

    #[test]
    fn intent_bonus_survives_semantic_normalization() {
        let mut engine = CueMapEngine::<MainStats>::new();
        let mut config = SemanticConfig::default();
        config.enabled = true;
        config.dimensions = 3;
        config.storage = crate::semantic::SemanticStorage::F32;
        config.index = crate::semantic::SemanticIndexMode::Exact;
        config.semantic_rerank_weight = 0.60;
        config.intent_rerank_weight = 1.0;
        config.intent_rerank_max_delta = 64.0;
        engine.set_semantic_config(config);

        let matching_id = engine.add_memory_with_event_time_and_vector(
            "A preference memory".to_string(),
            vec!["preference".to_string()],
            None,
            MainStats::default(),
            true,
            None,
            Some(vec![1.0, 0.0, 0.0]),
        );
        let unrelated_id = engine.add_memory_with_event_time_and_vector(
            "An unrelated event memory".to_string(),
            vec!["event".to_string()],
            None,
            MainStats::default(),
            true,
            None,
            Some(vec![0.0, 1.0, 0.0]),
        );
        engine.attach_intent_classification(
            matching_id,
            classification("preference", 1.0, true),
        );
        engine.attach_intent_classification(
            unrelated_id,
            classification("event_or_plan", 1.0, true),
        );

        let query = classification("preference", 1.0, true);
        let mut results = vec![
            candidate(matching_id, 1000.0),
            candidate(unrelated_id, 0.0),
        ];
        engine.apply_intent_reranker(&mut results, &query);
        assert_eq!(results[0].intent_rerank_bonus, 64.0);

        let semantic_count =
            engine.rerank_existing_semantic_candidates(&mut results, &[1.0, 0.0, 0.0]);

        assert_eq!(semantic_count, 2);
        assert_eq!(results[0].score, 1064.0);
        assert_eq!(results[0].intent_rerank_bonus, 64.0);
        assert_eq!(results[0].semantic_similarity, 1.0);
    }

    #[test]
    fn structured_helpers_cover_facet_families_and_fallbacks() {
        let generated = [
            "source_role:user", "source_channel:chat", "source_type:note",
            "source_session:s1", "source_time:morning", "source_date:2024",
            "source_week:1", "source_month:1", "has:age", "completion_count:2",
            "completed_action:ship", "instruction:do", "instruction_trigger:now",
            "instruction_action:run", "preference:tea", "preference_value:green",
            "preference_topic:drink", "preference_contrast:coffee", "temporal:today",
            "co_residence:home", "entity:person", "person_title:dr", "person_role_phrase:lead",
            "person_ref:kaan", "quantity_object:items", "quantity_unit:kg",
            "quantity_unit_object:bag", "quantity_count:2", "inventory_object:book",
            "inventory_count:3", "purchase:item", "companion:friend", "age:42",
            "education:college", "travel:trip", "media:book", "reading:novel",
            "transport_mode:train", "transport_event:arrival", "activity_domain:work",
            "topic:rust", "attribute:fast", "family_relation:sibling", "family_scope:home",
            "family_count:2", "sibling_kind:brother", "type:preference",
        ];
        for cue in generated {
            assert!(is_generated_memory_facet(cue), "{cue} should be generated");
            assert!(cue_structured_family_mask(cue) != 0);
        }
        for cue in ["type:preference", "media:book", "travel:trip", "topic:rust", "purchase:item"] {
            assert!(is_semantic_generated_facet(cue));
        }
        assert!(!is_generated_memory_facet("plain lexical cue"));
        assert!(!is_semantic_generated_facet("plain lexical cue"));
        assert_eq!(cue_structured_family_mask("plain lexical cue"), 0);
        assert_eq!(rerank_multiplier_for_prefix("source_time"), 12.0);
        assert_eq!(rerank_multiplier_for_prefix("unknown"), 4.0);

        let cues = generated.iter().map(|cue| cue.to_string()).collect::<Vec<_>>();
        let features = compute_memory_scoring_features(&cues);
        assert_eq!(features.version, MEMORY_SCORING_FEATURES_VERSION);
        assert!(features.has_summary_type == false);
        assert_eq!(features.scored_cue_len, 1);
        assert!(features.structured_family_mask & FAMILY_PERSON != 0);
        assert!(features.structured_family_mask & FAMILY_SOURCE_TIME != 0);

        let profile_cues = [
            ("plain".to_string(), 1.0),
            ("person_role_phrase:lead".to_string(), 1.0),
            ("quantity_object:items".to_string(), 1.0),
            ("inventory_object:book".to_string(), 1.0),
            ("travel:trip".to_string(), 1.0),
            ("age:42".to_string(), 1.0),
            ("education:college".to_string(), 1.0),
            ("family_count:2".to_string(), 3.0),
            ("source_role:user".to_string(), 3.0),
            ("source_time:morning".to_string(), 1.0),
            ("type:update".to_string(), 1.0),
            ("type:preference".to_string(), 3.0),
        ];
        let profile = build_query_scoring_profile(&profile_cues);
        assert!(profile.lexical_query_seen);
        assert!(profile.strong_lexical_query_seen);
        assert!(profile.strong_structured_query_seen);
        assert!(profile.structured_query_seen);
        assert!(profile.person_structured_query_seen);
        assert!(profile.quantity_structured_query_seen);
        assert!(profile.inventory_structured_query_seen);
        assert!(profile.travel_structured_query_seen);
        assert!(profile.age_structured_query_seen);
        assert!(profile.education_structured_query_seen);
        assert!(profile.family_structured_query_seen);
        assert!(profile.family_count_structured_query_seen);
        assert!(profile.source_role_structured_query_seen);
        assert!(profile.source_time_structured_query_seen);
        assert!(profile.update_structured_query_seen);

        let mut stale = Memory::<MainStats>::new(Vec::new(), None);
        stale.cues = vec!["plain".to_string()];
        let fresh = memory_scoring_features(&stale);
        assert_eq!(fresh.scored_cue_len, 1);
        stale.scoring_features.version = MEMORY_SCORING_FEATURES_VERSION;
        stale.scoring_features.scored_cue_len = 9;
        assert_eq!(memory_scoring_features(&stale).scored_cue_len, 9);
    }

    #[test]
    fn metadata_and_source_order_parsers_cover_types_and_invalid_values() {
        let mut metadata = HashMap::new();
        metadata.insert("number".to_string(), serde_json::json!(7));
        metadata.insert("flag".to_string(), serde_json::json!(true));
        metadata.insert("blank".to_string(), serde_json::json!("  "));
        assert_eq!(CueMapEngine::<MainStats>::metadata_string_value(&metadata, &["blank", "number"]), Some("7".to_string()));
        assert_eq!(CueMapEngine::<MainStats>::metadata_string_value(&metadata, &["flag"]), Some("true".to_string()));
        assert_eq!(CueMapEngine::<MainStats>::metadata_string_value(&metadata, &["missing"]), None);

        assert_eq!(CueMapEngine::<MainStats>::normalize_source_order_value(" Hello, World! "), Some("hello_world".to_string()));
        assert_eq!(CueMapEngine::<MainStats>::normalize_source_order_value("---"), None);
        metadata.insert("session_id".to_string(), serde_json::json!(" Session A "));
        metadata.insert("source_turn_index".to_string(), serde_json::json!("12"));
        assert_eq!(CueMapEngine::<MainStats>::source_order_session_from(&metadata, &[]), Some("session_a".to_string()));
        assert_eq!(CueMapEngine::<MainStats>::source_order_value_from(&metadata, &[]), Some(12));

        let mut numeric = HashMap::new();
        numeric.insert("source_order".to_string(), serde_json::json!(9));
        assert_eq!(CueMapEngine::<MainStats>::source_order_value_from(&numeric, &[]), Some(9));
        let cues = vec!["source_session:Fallback Session".to_string(), "turn_index:4".to_string()];
        assert_eq!(CueMapEngine::<MainStats>::source_order_session_from(&HashMap::new(), &cues), Some("fallback_session".to_string()));
        assert_eq!(CueMapEngine::<MainStats>::source_order_value_from(&HashMap::new(), &cues), Some(4));
        assert_eq!(CueMapEngine::<MainStats>::source_order_value_from(&HashMap::new(), &["turn_index:nope".to_string()]), None);

        let engine = CueMapEngine::<MainStats>::new();
        assert!(engine.ordered_entries_for_session("!!!", 10).is_empty());
        assert!(engine.source_order_window("!!!", 0, 2).is_empty());
    }

    #[test]
    fn constructors_semantic_setup_and_encoding_error_paths_are_safe() {
        struct UnitEncoder;
        impl SemanticEncoder for UnitEncoder {
            fn dimensions(&self) -> usize { 3 }
            fn encode(&self, text: &str) -> Result<Vec<f32>, String> {
                if text == "bad" { return Err("bad input".to_string()); }
                Ok(vec![1.0, 0.0, 0.0])
            }
        }
        let key = crate::crypto::EncryptionKey::new(vec![7; 32]);
        let keyed = CueMapEngine::<MainStats>::with_key(Some(key.clone()));
        assert_eq!(keyed.get_master_key().unwrap().as_bytes(), &[7; 32]);
        let unkeyed = CueMapEngine::<MainStats>::with_key(None);
        assert!(unkeyed.get_master_key().is_none());

        let mut engine = CueMapEngine::<MainStats>::new();
        assert!(engine.encode_semantic_text("disabled").is_none());
        let mut config = SemanticConfig::default();
        config.enabled = true;
        config.encoder_enabled = true;
        config.dimensions = 3;
        engine.set_semantic_config(config);
        assert!(engine.encode_semantic_text("no encoder").is_none());
        assert!(matches!(engine.classify_intent("anything", IntentTarget::Memory), Err(error) if error == "intent classifier unavailable"));
        engine.set_semantic_encoder(Some(Arc::new(UnitEncoder)));
        assert!(engine.encode_semantic_text("works").is_some());
        assert!(engine.encode_semantic_text("bad").is_none());
        let _ = engine.classify_intent("anything", IntentTarget::Memory);
        let _ = engine.classify_intent_with_embedding("anything", IntentTarget::Memory, &[1.0, 0.0, 0.0]);
        assert!(engine.configure_semantic_encoder().is_ok());

        let mut bad = SemanticConfig::default();
        bad.enabled = true;
        bad.dimensions = 3;
        bad.storage = crate::semantic::SemanticStorage::F32;
        engine.set_semantic_config(bad);
        let id = engine.add_memory_with_event_time_and_vector("bad vector".to_string(), vec!["v".to_string()], None, MainStats::default(), true, Some(42.0), Some(vec![1.0, 2.0]));
        assert_ne!(id, INVALID_MEMORY_ID);
        assert!(engine.get_memory(id).unwrap().semantic_vector.is_none());
        assert_eq!(engine.semantic_index_stats(), (true, 0, None));
    }

    #[test]
    fn recall_edge_paths_and_maintenance_helpers_are_covered() {
        let engine = CueMapEngine::<MainStats>::new();
        assert!(engine.recall(Vec::new(), 10, false, None).is_empty());
        assert!(engine.recall_intersection(Vec::new(), 10).is_empty());
        assert!(engine.recall_intersection(vec![("missing".to_string(), 1.0)], 10).is_empty());
        assert!(engine.recall_fast(Vec::new(), 10).is_empty());
        assert!(engine.recall_weighted(Vec::new(), 10, false, None, 1, false, false, None, None).is_empty());
        assert!(engine.recall_weighted_with_query_embedding(Vec::new(), 10, false, None, 1, false, false, None, None, Some(&[1.0, 0.0])).is_empty());

        let id = engine.add_memory("alpha beta".to_string(), vec!["alpha".to_string(), "type:update".to_string()], None, MainStats::default(), true);
        assert_eq!(engine.get_cue_frequency(" ALPHA "), 1);
        assert!(engine.recall_intersection(vec![("alpha".to_string(), 2.0), ("missing".to_string(), 1.0)], 2).len() == 1);
        assert_eq!(engine.recall_fast(vec!["alpha".to_string(), "".to_string()], 1).len(), 1);
        let mandatory = vec!["not-present".to_string()];
        assert!(engine.recall_weighted(vec![("alpha".to_string(), 1.0)], 10, false, None, 1, true, false, None, Some(&mandatory)).is_empty());
        let mut heatmap = HashMap::new();
        heatmap.insert("alpha".to_string(), 2.0);
        let (results, timing) = engine.recall_weighted_with_timing(vec![("alpha".to_string(), 1.0)], 2, true, Some(1), 1, true, false, Some(&heatmap), None);
        assert_eq!(results.len(), 1);
        assert!(timing.total_ms >= 0.0);
        assert!(engine.attach_cues(INVALID_MEMORY_ID, vec!["x".to_string()]) == false);
        engine.remove_cues_from_index(id, &[" alpha ".to_string(), "missing".to_string(), "".to_string()]);
        assert_eq!(engine.get_cue_frequency("alpha"), 0);
    }

    #[test]
    fn mainstats_and_lexicon_specialized_paths_are_exercised() {
        let engine = CueMapEngine::<MainStats>::new();
        let id = engine.add_memory("hot memory".to_string(), vec!["hot".to_string()], None, MainStats { intrinsic_salience: 0.2, dynamic_salience: 0.0, last_boosted_at: 0, reinforcement_count: 0 }, true);
        engine.reinforce_dynamic(id, 4.0);
        let mut heatmap = HashMap::new();
        heatmap.insert("hot".to_string(), 2.5);
        let scored = engine.score_with_decay_and_market(vec![id, INVALID_MEMORY_ID], &heatmap);
        assert_eq!(scored.len(), 1);
        assert!(scored[0].score > 2.5);
        assert!(scored[0].explain.is_some());
        engine.decay_salience(0.5);
        assert!(!engine.get_trending_cues(10).is_empty());
        assert_eq!(engine.prune_low_salience(100.0), 1);

        let dict = CueMapEngine::<LexiconStats>::new();
        let dict_id = dict.add_memory("word".to_string(), vec!["word".to_string()], None, LexiconStats::default(), true);
        dict.reinforce_tiered(dict_id, 3);
        let trending = dict.get_trending_items(10);
        assert_eq!(trending, vec![(dict_id, 3.0)]);
        assert!(dict.get_trending_items(0).is_empty());
    }

    #[test]
    fn from_state_rebuilds_indexes_and_disk_migration_roundtrips() {
        let source = CueMapEngine::<MainStats>::new();
        let id = source.add_memory("persisted".to_string(), vec!["persist".to_string()], None, MainStats::default(), true);
        let mut config = crate::config::ServerConfig::default();
        config.server.data_dir = std::env::temp_dir().join(format!("cuemap-engine-{}", std::process::id())).to_string_lossy().to_string();
        config.server.store_content_on_disk = true;
        let disk = CueMapEngine::from_state((**source.get_memories()).clone(), (**source.get_source_key_to_id()).clone(), (**source.get_cue_index()).clone(), source.next_memory_id(), None, config.clone(), "disk-test".to_string());
        let memory = disk.get_memory(id).unwrap();
        assert!(memory.disk_backed);
        assert_eq!(disk.read_memory_content(&memory).unwrap(), "persisted");
        let mut ram_config = config;
        ram_config.server.store_content_on_disk = false;
        let restored = CueMapEngine::from_state((**disk.get_memories()).clone(), (**disk.get_source_key_to_id()).clone(), (**disk.get_cue_index()).clone(), disk.next_memory_id(), None, ram_config, "disk-test".to_string());
        let restored_memory = restored.get_memory(id).unwrap();
        assert!(!restored_memory.disk_backed);
        assert_eq!(restored.read_memory_content(&restored_memory).unwrap(), "persisted");
        assert_eq!(restored.get_cue_frequency("persist"), 1);
        let _ = std::fs::remove_dir_all(restored.get_disk_content_dir().parent().unwrap().parent().unwrap());
    }

    #[test]
    fn temporal_chunking_disk_storage_and_source_aliases_work() {
        let mut engine = CueMapEngine::<MainStats>::new();
        let mut config = engine.config.clone();
        config.server.data_dir = std::env::temp_dir().join(format!("cuemap-engine-direct-{}", std::process::id())).to_string_lossy().to_string();
        config.server.store_content_on_disk = true;
        engine.config = config;
        let first = engine.add_memory_with_event_time(
            "first event".to_string(),
            vec!["topic:rust".to_string()],
            None,
            MainStats::default(),
            false,
            Some(1000.0),
        );
        let second = engine.add_memory_with_event_time(
            "second event".to_string(),
            vec!["topic:rust".to_string()],
            None,
            MainStats::default(),
            false,
            Some(1100.0),
        );
        let first_memory = engine.get_memory(first).unwrap();
        let second_memory = engine.get_memory(second).unwrap();
        assert!(first_memory.disk_backed && second_memory.disk_backed);
        assert!(second_memory.cues.iter().any(|cue| cue == &format!("episode:{first}")));
        assert_eq!(engine.read_memory_content(&second_memory).unwrap(), "second event");
        assert_eq!(engine.get_cue_frequency("topic:rust"), 2);
        assert_eq!(engine.source_order_for_memory(first), None);
        let _ = std::fs::remove_dir_all(engine.get_disk_content_dir().parent().unwrap().parent().unwrap());
    }

    #[test]
    fn semantic_budget_and_reranker_invalid_paths_are_safe() {
        let mut engine = CueMapEngine::<MainStats>::new();
        let mut config = SemanticConfig::default();
        config.enabled = true;
        config.dimensions = 3;
        config.max_memory_mb = 1;
        config.storage = crate::semantic::SemanticStorage::F32;
        config.reranker_enabled = true;
        config.reranker_weights = vec![0.1; 6];
        config.reranker_scale = 0.5;
        engine.set_semantic_config(config);
        let id = engine.add_memory_with_event_time_and_vector("vector".to_string(), vec!["vector".to_string()], None, MainStats::default(), true, None, Some(vec![1.0, 0.0, 0.0]));
        let mut candidates = vec![candidate(id, 1.0), candidate(INVALID_MEMORY_ID, 0.0)];
        assert_eq!(engine.rerank_existing_semantic_candidates(&mut candidates, &[]), 0);
        assert_eq!(engine.merge_semantic_candidates(&mut candidates, &[], 2), (0, 0));
        engine.apply_semantic_reranker(&mut candidates, &[]);
        let invalid_query_results = engine.recall_weighted_with_query_embedding(vec![("vector".to_string(), 1.0)], 2, false, None, 1, false, false, None, None, Some(&[]));
        assert_eq!(invalid_query_results.len(), 1);
        assert!(engine.get_memory(id).unwrap().semantic_vector.is_some());

        let mut budget_engine = CueMapEngine::<MainStats>::new();
        let mut budget_config = SemanticConfig::default();
        budget_config.enabled = true;
        budget_config.dimensions = 1_000_000;
        budget_config.max_memory_mb = 1;
        budget_engine.set_semantic_config(budget_config);
        let huge = budget_engine.add_memory_with_event_time_and_vector("budget".to_string(), vec!["budget".to_string()], None, MainStats::default(), true, None, Some(vec![0.0; 1_000_000]));
        assert!(budget_engine.get_memory(huge).unwrap().semantic_vector.is_none());
    }

    #[test]
    fn intent_coverage_tracks_current_and_stale_annotations() {
        let engine = CueMapEngine::<MainStats>::new();
        let current_id = engine.add_memory("current".to_string(), vec!["current".to_string()], None, MainStats::default(), true);
        let stale_id = engine.add_memory("stale".to_string(), vec!["stale".to_string()], None, MainStats::default(), true);
        engine.attach_intent_classification(current_id, classification("preference", 1.0, true));
        let mut stale = classification("event_or_plan", 1.0, true);
        stale.model_version = "old-model".to_string();
        engine.attach_intent_classification(stale_id, stale);
        assert_eq!(engine.intent_coverage(), (2, 2, 1, 1));
        assert!(engine.attach_intent_classification(INVALID_MEMORY_ID, classification("preference", 1.0, true)) == false);
    }

    #[test]
    fn parent_expansion_symbols_and_consolidation_are_exercised() {
        let engine = CueMapEngine::<MainStats>::new();
        let mut parent = HashMap::new();
        parent.insert("parent:doc".to_string(), serde_json::json!(true));
        let _first = engine.add_memory("chunk one".to_string(), vec!["parent:doc".to_string(), "chunk_idx:0".to_string(), "needle".to_string()], None, MainStats::default(), true);
        let _middle = engine.add_memory("chunk two".to_string(), vec!["parent:doc".to_string(), "chunk_idx:1".to_string(), "needle".to_string()], None, MainStats::default(), true);
        let _third = engine.add_memory("chunk three".to_string(), vec!["parent:doc".to_string(), "chunk_idx:2".to_string(), "needle".to_string()], None, MainStats::default(), true);
        let expanded = engine.recall_weighted(vec![("needle".to_string(), 1.0)], 1, false, None, 2, false, false, None, None);
        assert!(expanded.iter().any(|result| result.content.contains("chunk two") && result.content.contains("chunk three")));

        engine.add_memory("fn".to_string(), vec!["defines_function:run".to_string(), "calls_method:save".to_string(), "defines_function:".to_string()], Some(parent), MainStats::default(), true);
        let symbols = engine.get_all_symbols();
        assert!(symbols.contains("run") && symbols.contains("save"));

        let mut m1 = MainStats::default();
        m1.intrinsic_salience = 2.0;
        let mut m2 = MainStats::default();
        m2.intrinsic_salience = 4.0;
        engine.add_memory("merge one".to_string(), vec!["shared".to_string(), "one".to_string()], None, m1, true);
        engine.add_memory("merge two".to_string(), vec!["shared".to_string(), "two".to_string()], None, m2, true);
        let merged = engine.consolidate_memories(0.3);
        assert!(!merged.is_empty());
        let summary = engine.get_memory(merged[0].0).unwrap();
        assert_eq!(summary.metadata.get("consolidated").and_then(|v| v.as_bool()), Some(true));
        assert!(summary.cues.iter().any(|cue| cue == "type:summary"));
    }

    #[test]
    fn id_exhaustion_and_stats_snapshot_are_safe() {
        let engine = CueMapEngine::<MainStats>::new();
        engine.next_memory_id.store(MemoryId::MAX, Ordering::Relaxed);
        assert_eq!(engine.add_memory("too late".to_string(), vec!["x".to_string()], None, MainStats::default(), true), INVALID_MEMORY_ID);
        let stats = engine.get_stats();
        assert_eq!(stats.get("total_memories").and_then(|v| v.as_u64()), Some(0));
        assert_eq!(stats.get("total_cues").and_then(|v| v.as_u64()), Some(0));
    }

    #[test]
    fn public_constructor_and_recall_wrappers_are_smoke_tested() {
        let mut engine = CueMapEngine::<MainStats>::with_tuning(crate::config::TuningConfig::default());
        engine.set_tuning_config(crate::config::TuningConfig::default());
        engine.set_master_key(Some(Arc::new(crate::crypto::EncryptionKey::new(vec![3; 32]))));
        let id = engine.add_memory_with_source_key(
            "wrapper memory".to_string(),
            vec!["wrapper".to_string()],
            None,
            MainStats::default(),
            true,
            Some("wrapper-key".to_string()),
        );
        assert_ne!(id, INVALID_MEMORY_ID);
        assert_eq!(engine.memory_id_for_source_key("wrapper-key"), Some(id));
        assert!(engine.get_memories().contains_key(&id));
        assert!(engine.get_source_key_to_id().contains_key("wrapper-key"));
        assert!(engine.get_cue_index().contains_key("wrapper"));
        assert!(engine.next_memory_id() > id);
        let reranked = engine.recall_weighted_with_query_embedding_rerank_only(vec![("wrapper".to_string(), 1.0)], 1, false, None, 1, false, false, None, None, Some(&[1.0]));
        assert_eq!(reranked.len(), 1);
        let (_results, timing) = engine.recall_weighted_with_query_embedding_rerank_only_with_timing(vec![("wrapper".to_string(), 1.0)], 1, false, None, 1, false, false, None, None, Some(&[1.0]));
        assert!(timing.total_ms >= 0.0);
    }

    #[test]
    fn generic_engine_paths_are_instantiated_for_lexicon_and_main_stats() {
        let mut metadata = HashMap::new();
        metadata.insert("session_id".to_string(), serde_json::json!("lex-session"));
        metadata.insert("source_order".to_string(), serde_json::json!(1));
        let mut lexicon = CueMapEngine::<LexiconStats>::new();
        let lex_id = lexicon.add_memory("lexicon word".to_string(), vec!["word".to_string()], Some(metadata), LexiconStats::default(), true);
        let lex_mem = lexicon.get_memory(lex_id).unwrap();
        assert_eq!(lexicon.read_memory_content(&lex_mem).unwrap(), "lexicon word");
        assert_eq!(lexicon.recall_fast(vec!["word".to_string()], 1).len(), 1);
        assert_eq!(lexicon.recall_intersection(vec![("word".to_string(), 1.0)], 1).len(), 1);
        assert_eq!(lexicon.source_order_window("lex-session", 1, 2).len(), 1);
        let _ = lexicon.set_semantic_config(SemanticConfig::default());
        let restored = CueMapEngine::from_state((**lexicon.get_memories()).clone(), (**lexicon.get_source_key_to_id()).clone(), (**lexicon.get_cue_index()).clone(), lexicon.next_memory_id(), None, crate::config::ServerConfig::default(), "lexicon".to_string());
        assert_eq!(restored.get_memory(lex_id).unwrap().id, lex_id);

        let main = CueMapEngine::<MainStats>::new();
        let main_id = main.add_memory("main word".to_string(), vec!["word".to_string()], None, MainStats::default(), true);
        assert_eq!(main.recall_fast(vec!["word".to_string()], 1).len(), 1);
        assert_eq!(main.recall_intersection(vec![("word".to_string(), 1.0)], 1).len(), 1);
        let main_mem = main.get_memory(main_id).unwrap();
        assert_eq!(main.read_memory_content(&main_mem).unwrap(), "main word");
    }

    #[test]
    fn structured_reranking_penalty_matrix_exercises_all_families() {
        let cases: [(&str, f64, &str); 15] = [
            ("type:update", 1.0, "family_relation:sibling"),
            ("family_count:2", 1.0, "family_relation:sibling"),
            ("source_role:user", 3.0, "family_relation:sibling"),
            ("source_time:morning", 1.0, "family_relation:sibling"),
            ("source_time:morning", 1.0, "source_time:evening"),
            ("type:preference", 3.0, "family_relation:sibling"),
            ("person_role_phrase:lead", 1.0, "quantity_object:item"),
            ("quantity_object:item", 1.0, "inventory_object:book"),
            ("inventory_object:book", 1.0, "travel:trip"),
            ("travel:trip", 1.0, "age:42"),
            ("age:42", 1.0, "education:college"),
            ("education:college", 1.0, "family_relation:sibling"),
            ("family_relation:sibling", 1.0, "family_scope:home"),
            ("family_scope:home", 1.0, "family_relation:sibling"),
            ("source_time:morning", 1.0, "topic:rust"),
        ];
        for (query_cue, weight, candidate_structured) in cases {
            let engine = CueMapEngine::<MainStats>::new();
            engine.add_memory(
                "structured seed".to_string(),
                vec!["seed".to_string(), query_cue.to_string()],
                None,
                MainStats::default(),
                true,
            );
            engine.add_memory(
                "lexical candidate".to_string(),
                vec!["lexical".to_string(), candidate_structured.to_string()],
                None,
                MainStats::default(),
                true,
            );
            let results = engine.recall_weighted(
                vec![("lexical".to_string(), 1.0), (query_cue.to_string(), weight)],
                5,
                false,
                None,
                1,
                false,
                true,
                None,
                None,
            );
            assert!(!results.is_empty(), "no result for {query_cue}");
        }
    }

    #[test]
    fn salience_decay_and_consolidation_truncation_paths_are_verified() {
        let engine = CueMapEngine::<MainStats>::new();
        let cold = engine.add_memory("cold".to_string(), vec!["cold".to_string()], None, MainStats::default(), true);
        let warm = engine.add_memory("warm".to_string(), vec!["warm".to_string()], None, MainStats::default(), true);
        if let Some(mut memory) = engine.get_memories().get_mut(&cold) {
            memory.stats.dynamic_salience = 0.005;
            memory.stats.last_boosted_at = 1;
        }
        if let Some(mut memory) = engine.get_memories().get_mut(&warm) {
            memory.stats.dynamic_salience = 3.0;
            memory.stats.last_boosted_at = 1;
        }
        engine.decay_salience(0.5);
        assert_eq!(engine.get_memory(cold).unwrap().stats.dynamic_salience, 0.0);
        assert!(engine.get_memory(warm).unwrap().stats.dynamic_salience < 3.0);
        assert!(engine.get_trending_cues(10).is_empty());

        let long = "x".repeat(700);
        engine.add_memory(long.clone(), vec!["merge-long".to_string()], None, MainStats::default(), true);
        engine.add_memory(long, vec!["merge-long".to_string()], None, MainStats::default(), true);
        let merged = engine.consolidate_memories(0.5);
        assert!(!merged.is_empty());
        let summary = engine.get_memory(merged[0].0).unwrap();
        assert!(engine.read_memory_content(&summary).unwrap().contains("[truncated]"));
    }

    #[test]
    fn recall_intersection_and_fast_paths_cover_duplicates_and_matches() {
        let engine = CueMapEngine::<MainStats>::new();
        let id = engine.add_memory("both cues".to_string(), vec!["first".to_string(), "second".to_string()], None, MainStats::default(), true);
        let other = engine.add_memory("first only".to_string(), vec!["first".to_string()], None, MainStats::default(), true);
        let intersection = engine.recall_intersection(vec![("first".to_string(), 2.0), ("second".to_string(), 3.0)], 10);
        assert_eq!(intersection[0].memory_id, id);
        assert_eq!(intersection[0].intersection_count, 2);
        let fast = engine.recall_fast(vec!["first".to_string(), "second".to_string(), "first".to_string()], 10);
        assert_eq!(fast.len(), 2);
        assert!(fast.iter().any(|result| result.memory_id == other));
    }
}
