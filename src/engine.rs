use crate::config::TuningConfig;
use crate::crypto::EncryptionKey;
use crate::structures::{
    LexiconStats, MainStats, Memory, MemoryId, MemoryScoringFeatures, MemoryStats, OrderedSet,
    INVALID_MEMORY_ID,
};
use ahash::RandomState;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

    pub fn get_master_key(&self) -> Option<Arc<EncryptionKey>> {
        self.master_key.clone()
    }

    pub fn from_state(
        memories: DashMap<MemoryId, Memory<T>, RandomState>,
        source_key_to_id: DashMap<String, MemoryId, RandomState>,
        cue_index: DashMap<String, OrderedSet, RandomState>,
        next_memory_id: MemoryId,
        loaded_global_counts: Option<DashMap<String, u64, RandomState>>,
        config: crate::config::ServerConfig,
        project_id: String,
    ) -> Self {
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
            config: config.clone(),
            project_id: project_id.clone(),
        };
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
        cuepacks: &crate::cuepacks::CuePackRegistry,
        cuepack_selection: Option<&[String]>,
    ) -> Vec<String> {
        if TypeId::of::<T>() != TypeId::of::<MainStats>() {
            return cues;
        }

        let mut enriched = cues;
        let mut seen: HashSet<String> = enriched.iter().map(|cue| cue.to_lowercase()).collect();
        for facet in crate::facets::extract_memory_facets_with_cuepacks(
            content,
            metadata,
            &enriched,
            cuepacks,
            cuepack_selection,
        ) {
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
        self.add_memory_with_cuepacks(
            content,
            cues,
            metadata,
            stats,
            disable_temporal_chunking,
            crate::cuepacks::default_registry(),
            None,
        )
    }

    pub fn add_memory_with_cuepacks(
        &self,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        stats: T,
        disable_temporal_chunking: bool,
        cuepacks: &crate::cuepacks::CuePackRegistry,
        cuepack_selection: Option<&[String]>,
    ) -> MemoryId {
        self.add_memory_with_cuepacks_and_source_key(
            content,
            cues,
            metadata,
            stats,
            disable_temporal_chunking,
            cuepacks,
            cuepack_selection,
            None,
        )
    }

    pub fn add_memory_with_cuepacks_and_source_key(
        &self,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        stats: T,
        disable_temporal_chunking: bool,
        cuepacks: &crate::cuepacks::CuePackRegistry,
        cuepack_selection: Option<&[String]>,
        source_key: Option<String>,
    ) -> MemoryId {
        let cues = self.with_synchronous_facets(
            &content,
            metadata.as_ref(),
            cues,
            cuepacks,
            cuepack_selection,
        );

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

            if time_diff < 300.0 && overlap_ratio > 0.5 && !disable_temporal_chunking {
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
        let cues = self.with_synchronous_facets(
            &content,
            metadata.as_ref(),
            cues,
            crate::cuepacks::default_registry(),
            None,
        );

        if let Some(existing_id) = self.source_key_to_id.get(&source_key).map(|entry| *entry) {
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
        memory.cues = cues.clone();
        memory.scoring_features = compute_memory_scoring_features(&memory.cues);
        if let Some(s) = stats {
            memory.stats = s;
        }
        let source_order_link = Self::source_order_link_for_memory(&memory);

        if self.memories.insert(memory_id, memory).is_none() {
            self.memory_count.fetch_add(1, Ordering::Relaxed);
        }
        self.source_key_to_id.insert(source_key, memory_id);
        self.index_memory_cues(memory_id, &cues);
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
        collect_detailed_timing: bool,
    ) -> (Vec<RecallResult>, RecallTimingBreakdown) {
        let total_start = Instant::now();
        let mut timing = RecallTimingBreakdown::default();
        if query_cues.is_empty() {
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

        if active_cues.is_empty() {
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
                        "rerank_bonus": candidate.rerank_bonus,
                        "generic_penalty": candidate.generic_penalty,
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
