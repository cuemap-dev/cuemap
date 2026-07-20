use crate::auth::AuthConfig;
use crate::jobs::JobQueue;
use crate::metrics::MetricsCollector;
use crate::multi_tenant::{validate_project_id, MultiTenantEngine};
use crate::normalization::normalize_cue;
use crate::persistence::CloudBackupManager;
use crate::structures::{LexiconStats, MainStats, MemoryId, MemoryStats};
use crate::taxonomy::validate_cues;
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    middleware,
    response::IntoResponse,
    routing::{delete, get, patch, post},
    Json, Router,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Deserialize, Serialize)]
pub struct AddMemoryRequest {
    pub content: String,
    #[serde(default)]
    pub cues: Vec<String>,
    #[serde(default)]
    pub source_key: Option<String>,
    /// Original event timestamp as Unix seconds. When omitted, ingestion time is used.
    #[serde(default)]
    pub event_time: Option<f64>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub cuepacks: Option<Vec<String>>,
    #[serde(default)]
    pub disable_temporal_chunking: bool,
    #[serde(default)]
    pub async_ingest: bool,
    #[serde(default)]
    pub minimal_response: bool,
    #[serde(default)]
    pub trace_timing: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddMemoryResponse {
    id: MemoryId,
    status: String,
    cues: Vec<String>,
    latency_ms: f64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AddMemoryBatchRequest {
    pub memories: Vec<AddMemoryRequest>,
    #[serde(default)]
    pub minimal_response: bool,
    #[serde(default)]
    pub trace_timing: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RecallRequest {
    #[serde(default)]
    pub cues: Vec<String>,
    #[serde(default)]
    pub query_text: Option<String>,
    #[serde(default)]
    pub query_time: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_auto_reinforce")]
    pub auto_reinforce: bool,
    #[serde(default)]
    pub projects: Option<Vec<String>>,
    #[serde(default)]
    pub min_intersection: Option<usize>,
    #[serde(default)]
    pub explain: bool,
    #[serde(default)]
    pub trace_timing: bool,
    #[serde(default)]
    pub disable_salience_bias: bool,
    #[serde(default = "default_expansion_depth")]
    pub expansion_depth: usize,
    #[serde(default = "default_true")]
    pub disable_alias_expansion: bool,
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default)]
    pub cuepacks: Option<Vec<String>>,
    #[serde(default)]
    pub parent_fusion: ParentFusionMode,
    #[serde(default = "default_parent_fusion_limit")]
    pub parent_fusion_limit: usize,
    #[serde(default = "default_parent_fusion_min_chunks")]
    pub parent_fusion_min_chunks: usize,
    #[serde(default)]
    pub ordered_reconstruction: OrderedReconstructionMode,
    #[serde(default = "default_ordered_reconstruction_limit")]
    pub ordered_reconstruction_limit: usize,
    #[serde(default = "default_ordered_session_scan_limit")]
    pub ordered_session_scan_limit: usize,
    #[serde(default = "default_ordered_max_sessions")]
    pub ordered_max_sessions: usize,
    #[serde(default)]
    pub evidence_coverage: EvidenceCoverageMode,
    #[serde(default = "default_evidence_coverage_limit")]
    pub evidence_coverage_limit: usize,
    #[serde(default = "default_evidence_coverage_session_scan_limit")]
    pub evidence_coverage_session_scan_limit: usize,
    #[serde(default = "default_evidence_coverage_max_sessions")]
    pub evidence_coverage_max_sessions: usize,
    #[serde(default)]
    pub disable_cuebridge_artifacts: bool,
    #[serde(default = "default_cuebridge_gap_limit")]
    pub cuebridge_gap_limit: usize,
}

#[derive(Debug, Deserialize)]
pub struct ProjectExportQuery {
    #[serde(default)]
    pub cursor: Option<MemoryId>,
    #[serde(default = "default_project_export_limit")]
    pub limit: usize,
    #[serde(default = "default_true")]
    pub include_content: bool,
    #[serde(default = "default_true")]
    pub include_cues: bool,
    #[serde(default = "default_true")]
    pub include_metadata: bool,
}

fn default_depth() -> usize {
    1
}

fn default_project_export_limit() -> usize {
    1000
}

fn default_parent_fusion_limit() -> usize {
    80
}

fn default_parent_fusion_min_chunks() -> usize {
    2
}

fn default_ordered_reconstruction_limit() -> usize {
    80
}

fn default_ordered_session_scan_limit() -> usize {
    4096
}

fn default_ordered_max_sessions() -> usize {
    3
}

fn default_evidence_coverage_limit() -> usize {
    100
}

fn default_evidence_coverage_session_scan_limit() -> usize {
    4096
}

fn default_evidence_coverage_max_sessions() -> usize {
    3
}

fn default_cuebridge_gap_limit() -> usize {
    6
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OrderedReconstructionMode {
    Off,
    Auto,
    Force,
}

impl Default for OrderedReconstructionMode {
    fn default() -> Self {
        Self::Off
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EvidenceCoverageMode {
    Off,
    Auto,
    Force,
}

impl Default for EvidenceCoverageMode {
    fn default() -> Self {
        Self::Off
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ParentFusionMode {
    Off,
    Auto,
    Force,
}

impl Default for ParentFusionMode {
    fn default() -> Self {
        Self::Off
    }
}

fn apply_query_intent(
    ctx: &crate::projects::ProjectContext,
    cuepack_registry: &crate::cuepacks::CuePackRegistry,
    cuepack_selection: Option<&[String]>,
    query_text: Option<&str>,
    query_time: Option<&str>,
    expanded_cues: &mut Vec<(String, f64)>,
) -> Option<crate::facets::QueryIntent> {
    let query_text = query_text?;
    let total_memories = ctx.main.total_memories().max(1);
    let intent = crate::facets::compile_query_intent_with_cuepacks(query_text, query_time, |cue| {
        let df = ctx.main.get_cue_frequency(cue);
        if df == 0 {
            return false;
        }

        if cue.starts_with("source_role:")
            || cue.starts_with("source_channel:")
            || cue.starts_with("source_type:")
            || cue.starts_with("source_session:")
        {
            true
        } else if cue.starts_with("source_date:")
            || cue.starts_with("source_week:")
            || cue.starts_with("source_month:")
            || cue.starts_with("source_year:")
        {
            true
        } else if cue.starts_with("entity:")
            || cue.starts_with("person_")
            || cue.starts_with("quantity_")
            || cue.starts_with("quantity_count:")
            || cue.starts_with("inventory_")
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
            || cue.starts_with("family_")
            || cue.starts_with("sibling_kind:")
            || cue.starts_with("media:")
            || cue.starts_with("reading:")
            || cue.starts_with("transport_")
            || cue.starts_with("schedule:")
            || cue.starts_with("activity_domain:")
            || cue.starts_with("topic:")
        {
            df <= 32 || df * 3 <= total_memories
        } else if cue.starts_with("has:")
            || cue.starts_with("temporal:")
            || cue.starts_with("time_of_day:")
        {
            df <= 8 || df * 8 <= total_memories
        } else {
            df <= 16 || df * 5 <= total_memories
        }
    }, cuepack_registry, cuepack_selection);

    for (cue, multiplier) in &intent.cue_weight_adjustments {
        if let Some((_, weight)) = expanded_cues
            .iter_mut()
            .find(|(existing, _)| existing == cue)
        {
            *weight *= *multiplier;
        }
    }

    for (cue, weight) in &intent.weighted_cues {
        if let Some((_, existing_weight)) = expanded_cues
            .iter_mut()
            .find(|(existing, _)| existing == cue)
        {
            if *existing_weight < *weight {
                *existing_weight = *weight;
            }
        } else {
            expanded_cues.push((cue.clone(), *weight));
        }
    }

    Some(intent)
}

fn merge_cuebridge_gap_results(
    all_results: &mut Vec<crate::engine::RecallResult>,
    mut gap_results: Vec<crate::engine::RecallResult>,
    expansions: &[crate::cuebridge::CueBridgeGapExpansion],
    explain: bool,
) {
    let cues: Vec<String> = expansions.iter().map(|expansion| expansion.cue.clone()).collect();
    let provenance = serde_json::json!(expansions);

    for result in &mut gap_results {
        result
            .metadata
            .insert("cuebridge_gap_pack".to_string(), serde_json::json!(true));
        result.metadata.insert(
            "cuebridge_gap_pack_cues".to_string(),
            serde_json::json!(cues.clone()),
        );
        if explain {
            let payload = serde_json::json!({
                "cuebridge_gap_pack": true,
                "expansions": provenance,
            });
            match result.explain.as_mut().and_then(|value| value.as_object_mut()) {
                Some(obj) => {
                    obj.insert("cuebridge_gap_pack".to_string(), payload);
                }
                None => {
                    result.explain = Some(serde_json::json!({ "cuebridge_gap_pack": payload }));
                }
            }
        }
    }

    for result in gap_results {
        if let Some(existing) = all_results
            .iter_mut()
            .find(|existing| existing.memory_id == result.memory_id)
        {
            if result.score > existing.score {
                existing.score = result.score;
                existing.match_integrity = result.match_integrity;
                existing.intersection_count = result.intersection_count;
                existing.recency_score = result.recency_score;
                existing.reinforcement_score = result.reinforcement_score;
                existing.salience_score = result.salience_score;
            }
            existing
                .metadata
                .insert("cuebridge_gap_pack".to_string(), serde_json::json!(true));
            existing.metadata.insert(
                "cuebridge_gap_pack_cues".to_string(),
                serde_json::json!(cues.clone()),
            );
            if let Some(explain_value) = result.explain {
                existing.explain = Some(explain_value);
            }
        } else {
            all_results.push(result);
        }
    }
}

const MAX_ORDERED_RECONSTRUCTION_LIMIT: usize = 160;
const ORDERED_RECONSTRUCTION_PIVOT_SCAN: usize = 30;

#[derive(Clone)]
struct OrderedReconstructionCandidate {
    memory_id: MemoryId,
    session: String,
    order: i64,
    score: f64,
    match_count: usize,
    matched_cues: Vec<String>,
}

fn should_run_ordered_reconstruction(
    mode: OrderedReconstructionMode,
    query_intent: Option<&crate::facets::QueryIntent>,
) -> bool {
    match mode {
        OrderedReconstructionMode::Off => false,
        OrderedReconstructionMode::Force => true,
        OrderedReconstructionMode::Auto => query_intent
            .map(|intent| {
                intent.labels.iter().any(|label| {
                    label == "ordered_reconstruction"
                        || label == "multi_evidence_summary"
                        || label == "multi_evidence_collection"
                        || label == "source_instruction"
                })
            })
            .unwrap_or(false),
    }
}

fn ordered_cue_is_weak(cue: &str) -> bool {
    cue.len() <= 2
        || cue.starts_with("source_")
        || cue.starts_with("parent:")
        || cue.starts_with("chunk_idx:")
        || cue == "user"
        || cue == "assistant"
        || cue == "question"
        || cue == "answer"
}

fn ordered_reconstruction_results(
    ctx: &crate::projects::ProjectContext,
    expanded_cues: &[(String, f64)],
    all_results: &[crate::engine::RecallResult],
    limit: usize,
    session_scan_limit: usize,
    max_sessions: usize,
    explain: bool,
) -> Vec<crate::engine::RecallResult> {
    if expanded_cues.is_empty() || all_results.is_empty() || limit == 0 || max_sessions == 0 {
        return Vec::new();
    }

    let mut query_weights: HashMap<String, f64> = HashMap::new();
    for (cue, weight) in expanded_cues {
        let cue = cue.trim().to_lowercase();
        if cue.is_empty() || ordered_cue_is_weak(&cue) {
            continue;
        }
        query_weights
            .entry(cue)
            .and_modify(|existing| {
                if *existing < *weight {
                    *existing = *weight;
                }
            })
            .or_insert(*weight);
    }
    if query_weights.is_empty() {
        return Vec::new();
    }

    let mut sessions = Vec::<(String, f64)>::new();
    let mut seen_sessions = HashSet::new();
    for result in all_results.iter().take(ORDERED_RECONSTRUCTION_PIVOT_SCAN) {
        let Some((session, _order)) = ctx.main.source_order_for_memory(result.memory_id) else {
            continue;
        };
        if seen_sessions.insert(session.clone()) {
            sessions.push((session, result.score));
            if sessions.len() >= max_sessions {
                break;
            }
        }
    }
    if sessions.is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::<OrderedReconstructionCandidate>::new();
    let existing_ids: HashSet<MemoryId> = all_results.iter().map(|result| result.memory_id).collect();
    for (session, pivot_score) in sessions {
        for entry in ctx
            .main
            .ordered_entries_for_session(&session, session_scan_limit.max(1))
        {
            let Some(memory) = ctx.main.get_memory(entry.memory_id) else {
                continue;
            };
            let memory_cues: HashSet<String> =
                memory.cues.iter().map(|cue| cue.to_lowercase()).collect();
            let mut score = 0.0;
            let mut matched = Vec::new();
            for (cue, weight) in &query_weights {
                if memory_cues.contains(cue) {
                    score += weight.max(0.1) * 24.0;
                    matched.push(cue.clone());
                }
            }

            if matched.is_empty() && !existing_ids.contains(&entry.memory_id) {
                continue;
            }

            score += 30.0;
            score += pivot_score.min(200.0) * 0.10;
            if existing_ids.contains(&entry.memory_id) {
                score += 20.0;
            }

            matched.sort();
            matched.dedup();
            candidates.push(OrderedReconstructionCandidate {
                memory_id: entry.memory_id,
                session: session.clone(),
                order: entry.order,
                score,
                match_count: matched.len(),
                matched_cues: matched,
            });
        }
    }

    if candidates.is_empty() {
        return Vec::new();
    }

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.session.cmp(&b.session))
            .then_with(|| a.order.cmp(&b.order))
            .then_with(|| a.memory_id.cmp(&b.memory_id))
    });
    candidates.truncate(limit.min(MAX_ORDERED_RECONSTRUCTION_LIMIT));

    let mut results = Vec::new();
    for candidate in candidates {
        let Some(memory) = ctx.main.get_memory(candidate.memory_id) else {
            continue;
        };
        let content = ctx
            .main
            .read_memory_content(&memory)
            .unwrap_or_else(|_| "<decryption failed>".to_string());
        let mut metadata = memory.metadata.clone();
        metadata.insert(
            "ordered_reconstruction".to_string(),
            serde_json::json!(true),
        );
        metadata.insert(
            "ordered_reconstruction_session".to_string(),
            serde_json::json!(candidate.session),
        );
        metadata.insert(
            "ordered_reconstruction_order".to_string(),
            serde_json::json!(candidate.order),
        );
        metadata.insert(
            "ordered_reconstruction_matched_cues".to_string(),
            serde_json::json!(candidate.matched_cues),
        );

        let explain_payload = if explain {
            Some(serde_json::json!({
                "ordered_reconstruction": true,
                "score": candidate.score,
                "match_count": candidate.match_count,
                "session": metadata.get("ordered_reconstruction_session"),
                "order": metadata.get("ordered_reconstruction_order"),
                "matched_cues": metadata.get("ordered_reconstruction_matched_cues"),
            }))
        } else {
            None
        };

        results.push(crate::engine::RecallResult {
            memory_id: memory.id,
            content,
            score: candidate.score,
            match_integrity: candidate.match_count as f64 / query_weights.len().max(1) as f64,
            intersection_count: candidate.match_count,
            recency_score: 1.0,
            reinforcement_score: memory.stats.get_reinforcement_count() as f64,
            salience_score: memory.stats.get_salience(),
            created_at: memory.created_at,
            metadata,
            explain: explain_payload,
        });
    }

    results
}

fn merge_ordered_reconstruction_results(
    all_results: &mut Vec<crate::engine::RecallResult>,
    ordered_results: Vec<crate::engine::RecallResult>,
) {
    for result in ordered_results {
        if let Some(existing) = all_results
            .iter_mut()
            .find(|existing| existing.memory_id == result.memory_id)
        {
            if result.score > existing.score {
                existing.score = result.score;
                existing.match_integrity = result.match_integrity;
                existing.intersection_count = result.intersection_count;
                existing.recency_score = result.recency_score;
                existing.reinforcement_score = result.reinforcement_score;
                existing.salience_score = result.salience_score;
            }
            existing
                .metadata
                .insert("ordered_reconstruction".to_string(), serde_json::json!(true));
            if let Some(explain) = result.explain {
                existing.explain = Some(explain);
            }
        } else {
            all_results.push(result);
        }
    }
}

const MAX_EVIDENCE_COVERAGE_LIMIT: usize = 160;
const EVIDENCE_COVERAGE_PIVOT_SCAN: usize = 40;
const EVIDENCE_COVERAGE_TOPIC_LIMIT: usize = 12;
const EVIDENCE_COVERAGE_PROTECTED_RESULTS: usize = 5;
const EVIDENCE_COVERAGE_MIN_NEW_MATCHED_CUES: usize = 2;

#[derive(Clone)]
struct EvidenceCoverageCandidate {
    memory_id: MemoryId,
    session: String,
    order: i64,
    base_score: f64,
    selection_rank: usize,
    matched_cues: Vec<String>,
    topics: Vec<String>,
    role: Option<String>,
    source_plan: Option<String>,
}

fn should_run_evidence_coverage(
    mode: EvidenceCoverageMode,
    query_intent: Option<&crate::facets::QueryIntent>,
) -> bool {
    match mode {
        EvidenceCoverageMode::Off => false,
        EvidenceCoverageMode::Force => true,
        EvidenceCoverageMode::Auto => query_intent
            .map(|intent| {
                intent
                    .labels
                    .iter()
                    .any(|label| {
                        label == "multi_evidence_summary"
                            || label == "multi_evidence_collection"
                            || label == "ordered_reconstruction"
                    })
            })
            .unwrap_or(false),
    }
}

fn evidence_coverage_cue_is_weak(cue: &str) -> bool {
    ordered_cue_is_weak(cue)
        || crate::nl::get_stopwords().contains(cue)
        || matches!(
            cue,
            "summary"
                | "summarize"
                | "comprehensive"
                | "detailed"
                | "detail"
                | "thorough"
                | "complete"
                | "overview"
                | "everything"
                | "entire"
                | "full"
                | "scope"
                | "topic"
                | "aspect"
                | "aspects"
                | "key"
                | "important"
                | "capture"
                | "cover"
                | "covered"
                | "covering"
                | "discussion"
                | "discussions"
                | "conversation"
                | "conversations"
                | "session"
                | "sessions"
                | "develop"
                | "developed"
                | "development"
                | "evolve"
                | "evolved"
                | "progress"
                | "progressed"
                | "journey"
                | "process"
                | "approach"
                | "give"
                | "provide"
                | "include"
                | "including"
                | "involved"
                | "related"
                | "various"
        )
}

fn evidence_coverage_topic_is_generated(cue: &str) -> bool {
    cue.starts_with("source_")
        || cue.starts_with("has:")
        || cue.starts_with("type:")
        || cue.starts_with("temporal:")
        || cue.starts_with("time_of_day:")
        || cue.starts_with("content_month:")
        || cue.starts_with("quantity_")
        || cue.starts_with("inventory_")
        || cue.starts_with("person_")
        || cue.starts_with("entity:")
        || cue.starts_with("preference")
        || cue.starts_with("instruction")
        || cue.starts_with("parent:")
        || cue.starts_with("chunk_idx:")
}

fn evidence_coverage_topics(cues: &[String]) -> Vec<String> {
    let mut topics = Vec::new();
    let mut seen = HashSet::new();
    for cue in cues {
        let cue = cue.trim().to_lowercase();
        if cue.len() <= 2
            || evidence_coverage_topic_is_generated(&cue)
            || evidence_coverage_cue_is_weak(&cue)
            || !seen.insert(cue.clone())
        {
            continue;
        }
        topics.push(cue);
        if topics.len() >= EVIDENCE_COVERAGE_TOPIC_LIMIT {
            break;
        }
    }
    topics
}

fn evidence_coverage_role(cues: &[String]) -> Option<String> {
    cues.iter()
        .find_map(|cue| cue.strip_prefix("source_role:").map(|role| role.to_string()))
}

fn evidence_coverage_source_plan(
    metadata: &HashMap<String, serde_json::Value>,
    cues: &[String],
) -> Option<String> {
    for key in ["source_plan_idx", "source_plan", "plan_idx", "plan_id"] {
        let Some(value) = metadata.get(key) else {
            continue;
        };
        match value {
            serde_json::Value::String(text) if !text.trim().is_empty() => {
                return Some(text.trim().to_lowercase());
            }
            serde_json::Value::Number(number) => return Some(number.to_string()),
            _ => {}
        }
    }

    cues.iter()
        .find_map(|cue| cue.strip_prefix("source_plan:").map(|plan| plan.to_string()))
}

fn evidence_coverage_query_weights(expanded_cues: &[(String, f64)]) -> HashMap<String, f64> {
    let mut query_weights = HashMap::new();
    for (cue, weight) in expanded_cues {
        let cue = cue.trim().to_lowercase();
        if cue.is_empty() || evidence_coverage_cue_is_weak(&cue) {
            continue;
        }
        query_weights
            .entry(cue)
            .and_modify(|existing| {
                if *existing < *weight {
                    *existing = *weight;
                }
            })
            .or_insert(*weight);
    }
    query_weights
}

fn evidence_coverage_sessions(
    ctx: &crate::projects::ProjectContext,
    all_results: &[crate::engine::RecallResult],
    max_sessions: usize,
) -> Vec<(String, f64)> {
    let mut sessions = Vec::new();
    let mut seen = HashSet::new();
    for result in all_results.iter().take(EVIDENCE_COVERAGE_PIVOT_SCAN) {
        let Some((session, _order)) = ctx.main.source_order_for_memory(result.memory_id) else {
            continue;
        };
        if seen.insert(session.clone()) {
            sessions.push((session, result.score));
            if sessions.len() >= max_sessions {
                break;
            }
        }
    }
    sessions
}

fn build_evidence_coverage_candidates(
    ctx: &crate::projects::ProjectContext,
    expanded_cues: &[(String, f64)],
    all_results: &[crate::engine::RecallResult],
    session_scan_limit: usize,
    max_sessions: usize,
) -> Vec<EvidenceCoverageCandidate> {
    let query_weights = evidence_coverage_query_weights(expanded_cues);
    if query_weights.is_empty() || all_results.is_empty() || max_sessions == 0 {
        return Vec::new();
    }

    let sessions = evidence_coverage_sessions(ctx, all_results, max_sessions);
    if sessions.is_empty() {
        return Vec::new();
    }

    let existing_ids: HashSet<MemoryId> = all_results.iter().map(|result| result.memory_id).collect();
    let mut candidates = Vec::new();
    let mut seen_memory_ids = HashSet::new();
    for (session, pivot_score) in sessions {
        for entry in ctx
            .main
            .ordered_entries_for_session(&session, session_scan_limit.max(1))
        {
            if !seen_memory_ids.insert(entry.memory_id) {
                continue;
            }
            let Some(memory) = ctx.main.get_memory(entry.memory_id) else {
                continue;
            };
            let memory_cues: HashSet<String> =
                memory.cues.iter().map(|cue| cue.to_lowercase()).collect();
            let mut matched = Vec::new();
            let mut score = 0.0;
            for (cue, weight) in &query_weights {
                if memory_cues.contains(cue) {
                    matched.push(cue.clone());
                    score += weight.max(0.1) * 32.0;
                }
            }
            let is_existing_result = existing_ids.contains(&entry.memory_id);
            if matched.is_empty() && !is_existing_result {
                continue;
            }
            if !is_existing_result && matched.len() < EVIDENCE_COVERAGE_MIN_NEW_MATCHED_CUES {
                continue;
            }

            let topics = evidence_coverage_topics(&memory.cues);
            let has_answer_shape = memory_cues.contains("type:answer")
                || memory_cues.contains("has:list")
                || memory_cues.contains("type:recommendation");
            score += 120.0;
            score += pivot_score.min(250.0) * 0.08;
            score += matched.len() as f64 * 22.0;
            score += topics.len().min(6) as f64 * 5.0;
            if is_existing_result {
                score += 35.0;
            }
            if has_answer_shape {
                score += 18.0;
            }

            matched.sort();
            matched.dedup();
            candidates.push(EvidenceCoverageCandidate {
                memory_id: memory.id,
                session: session.clone(),
                order: entry.order,
                base_score: score,
                selection_rank: usize::MAX,
                matched_cues: matched,
                topics,
                role: evidence_coverage_role(&memory.cues),
                source_plan: evidence_coverage_source_plan(&memory.metadata, &memory.cues),
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.base_score
            .partial_cmp(&a.base_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.session.cmp(&b.session))
            .then_with(|| a.order.cmp(&b.order))
            .then_with(|| a.memory_id.cmp(&b.memory_id))
    });
    candidates
}

fn select_evidence_coverage_candidates(
    mut candidates: Vec<EvidenceCoverageCandidate>,
    limit: usize,
) -> Vec<EvidenceCoverageCandidate> {
    let limit = limit.min(MAX_EVIDENCE_COVERAGE_LIMIT);
    let mut selected = Vec::new();
    let mut covered_query = HashSet::new();
    let mut topic_counts: HashMap<String, usize> = HashMap::new();
    let mut role_counts: HashMap<String, usize> = HashMap::new();
    let mut order_buckets = HashSet::new();
    let mut plan_counts: HashMap<String, usize> = HashMap::new();

    while !candidates.is_empty() && selected.len() < limit {
        let mut best_idx = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (idx, candidate) in candidates.iter().enumerate() {
            let new_query = candidate
                .matched_cues
                .iter()
                .filter(|cue| !covered_query.contains(*cue))
                .count();
            let new_topics = candidate
                .topics
                .iter()
                .filter(|topic| topic_counts.get(*topic).copied().unwrap_or(0) == 0)
                .count();
            let repeated_topic_penalty = candidate
                .topics
                .iter()
                .map(|topic| topic_counts.get(topic).copied().unwrap_or(0).min(4) as f64)
                .sum::<f64>();
            let role_penalty = candidate
                .role
                .as_ref()
                .and_then(|role| role_counts.get(role).copied())
                .unwrap_or(0)
                .min(12) as f64;
            let order_bucket = (candidate.order / 8).max(0);
            let order_bonus = if order_buckets.contains(&(candidate.session.clone(), order_bucket)) {
                0.0
            } else {
                16.0
            };
            let plan_bonus = candidate
                .source_plan
                .as_ref()
                .map(|plan| {
                    let seen = plan_counts.get(plan).copied().unwrap_or(0).min(4) as f64;
                    if seen == 0.0 {
                        32.0
                    } else {
                        -seen * 8.0
                    }
                })
                .unwrap_or(0.0);
            let adjusted = candidate.base_score
                + new_query as f64 * 70.0
                + new_topics.min(8) as f64 * 12.0
                + order_bonus
                + plan_bonus
                - repeated_topic_penalty * 3.0
                - role_penalty * 1.5;
            if adjusted > best_score {
                best_score = adjusted;
                best_idx = idx;
            }
        }

        let mut candidate = candidates.remove(best_idx);
        candidate.base_score = best_score;
        candidate.selection_rank = selected.len();
        for cue in &candidate.matched_cues {
            covered_query.insert(cue.clone());
        }
        for topic in &candidate.topics {
            *topic_counts.entry(topic.clone()).or_insert(0) += 1;
        }
        if let Some(role) = &candidate.role {
            *role_counts.entry(role.clone()).or_insert(0) += 1;
        }
        order_buckets.insert((candidate.session.clone(), (candidate.order / 8).max(0)));
        if let Some(plan) = &candidate.source_plan {
            *plan_counts.entry(plan.clone()).or_insert(0) += 1;
        }
        selected.push(candidate);
    }

    selected.sort_by(|a, b| {
        b.base_score
            .partial_cmp(&a.base_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.session.cmp(&b.session))
            .then_with(|| a.order.cmp(&b.order))
            .then_with(|| a.memory_id.cmp(&b.memory_id))
    });
    selected
}

fn evidence_coverage_results(
    ctx: &crate::projects::ProjectContext,
    expanded_cues: &[(String, f64)],
    all_results: &[crate::engine::RecallResult],
    limit: usize,
    session_scan_limit: usize,
    max_sessions: usize,
    explain: bool,
) -> Vec<crate::engine::RecallResult> {
    if limit == 0 {
        return Vec::new();
    }
    let candidates = build_evidence_coverage_candidates(
        ctx,
        expanded_cues,
        all_results,
        session_scan_limit,
        max_sessions,
    );
    if candidates.is_empty() {
        return Vec::new();
    }
    let selected = select_evidence_coverage_candidates(candidates, limit);
    let query_weight_count = evidence_coverage_query_weights(expanded_cues).len().max(1);
    let protected_score_ceiling = evidence_coverage_score_ceiling(all_results);

    let mut results = Vec::new();
    for candidate in selected {
        let Some(memory) = ctx.main.get_memory(candidate.memory_id) else {
            continue;
        };
        let content = ctx
            .main
            .read_memory_content(&memory)
            .unwrap_or_else(|_| "<decryption failed>".to_string());
        let mut metadata = memory.metadata.clone();
        metadata.insert("evidence_coverage".to_string(), serde_json::json!(true));
        metadata.insert(
            "evidence_coverage_session".to_string(),
            serde_json::json!(candidate.session),
        );
        metadata.insert(
            "evidence_coverage_order".to_string(),
            serde_json::json!(candidate.order),
        );
        metadata.insert(
            "evidence_coverage_matched_cues".to_string(),
            serde_json::json!(candidate.matched_cues),
        );
        metadata.insert(
            "evidence_coverage_topics".to_string(),
            serde_json::json!(candidate.topics),
        );
        if let Some(source_plan) = &candidate.source_plan {
            metadata.insert(
                "evidence_coverage_source_plan".to_string(),
                serde_json::json!(source_plan),
            );
        }
        metadata.insert(
            "evidence_coverage_selection_rank".to_string(),
            serde_json::json!(candidate.selection_rank),
        );

        let explain_payload = if explain {
            Some(serde_json::json!({
                "evidence_coverage": true,
                "score": candidate.base_score,
                "selection_rank": candidate.selection_rank,
                "session": metadata.get("evidence_coverage_session"),
                "order": metadata.get("evidence_coverage_order"),
                "matched_cues": metadata.get("evidence_coverage_matched_cues"),
                "topics": metadata.get("evidence_coverage_topics"),
                "source_plan": metadata.get("evidence_coverage_source_plan"),
            }))
        } else {
            None
        };

        results.push(crate::engine::RecallResult {
            memory_id: memory.id,
            content,
            score: protected_score_ceiling
                .map(|ceiling| candidate.base_score.min(ceiling))
                .unwrap_or(candidate.base_score),
            match_integrity: candidate.matched_cues.len() as f64
                / query_weight_count as f64,
            intersection_count: candidate.matched_cues.len(),
            recency_score: 1.0,
            reinforcement_score: memory.stats.get_reinforcement_count() as f64,
            salience_score: memory.stats.get_salience(),
            created_at: memory.created_at,
            metadata,
            explain: explain_payload,
        });
    }

    results
}

fn evidence_coverage_score_ceiling(
    all_results: &[crate::engine::RecallResult],
) -> Option<f64> {
    if all_results.is_empty() {
        return None;
    }
    let mut scores: Vec<f64> = all_results
        .iter()
        .filter_map(|result| result.score.is_finite().then_some(result.score))
        .collect();
    if scores.is_empty() {
        return None;
    }
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let protected_idx = EVIDENCE_COVERAGE_PROTECTED_RESULTS
        .saturating_sub(1)
        .min(scores.len().saturating_sub(1));
    Some(scores[protected_idx] - 0.000_001)
}

fn merge_evidence_coverage_results(
    all_results: &mut Vec<crate::engine::RecallResult>,
    evidence_results: Vec<crate::engine::RecallResult>,
) {
    for result in evidence_results {
        if let Some(existing) = all_results
            .iter_mut()
            .find(|existing| existing.memory_id == result.memory_id)
        {
            if result.score > existing.score {
                existing.score = result.score;
                existing.match_integrity = result.match_integrity;
                existing.intersection_count = result.intersection_count;
                existing.recency_score = result.recency_score;
                existing.reinforcement_score = result.reinforcement_score;
                existing.salience_score = result.salience_score;
                existing.explain = result.explain.clone();
            }
            existing
                .metadata
                .insert("evidence_coverage".to_string(), serde_json::json!(true));
            if let Some(matched) = result.metadata.get("evidence_coverage_matched_cues").cloned() {
                existing
                    .metadata
                    .insert("evidence_coverage_matched_cues".to_string(), matched);
            }
            if let Some(topics) = result.metadata.get("evidence_coverage_topics").cloned() {
                existing
                    .metadata
                    .insert("evidence_coverage_topics".to_string(), topics);
            }
            if let Some(plan) = result.metadata.get("evidence_coverage_source_plan").cloned() {
                existing
                    .metadata
                    .insert("evidence_coverage_source_plan".to_string(), plan);
            }
            if let Some(rank) = result.metadata.get("evidence_coverage_selection_rank").cloned() {
                existing
                    .metadata
                    .insert("evidence_coverage_selection_rank".to_string(), rank);
            }
        } else {
            all_results.push(result);
        }
    }
}

const SLATE_RERANK_POOL_LIMIT: usize = 100;
const SLATE_RERANK_TARGET_LIMIT: usize = 20;
const SLATE_RERANK_PROTECTED_RESULTS: usize = 3;

#[derive(Clone)]
struct SlateRerankCandidate {
    idx: usize,
    memory_id: MemoryId,
    original_rank: usize,
    matched_cues: Vec<String>,
    topics: Vec<String>,
    session: Option<String>,
    order: Option<i64>,
    role: Option<String>,
    source_plan: Option<String>,
    tagged_evidence: bool,
    tagged_ordered: bool,
    tagged_standing_instruction: bool,
}

#[derive(Clone, Copy, Default)]
struct SlateRerankIntent {
    ordered: bool,
    summary: bool,
    instruction: bool,
    target_role: Option<&'static str>,
}

fn slate_rerank_requested(
    ordered_mode: OrderedReconstructionMode,
    evidence_mode: EvidenceCoverageMode,
    query_intent: Option<&crate::facets::QueryIntent>,
) -> bool {
    if ordered_mode == OrderedReconstructionMode::Off && evidence_mode == EvidenceCoverageMode::Off {
        return false;
    }

    query_intent
        .map(|intent| {
            intent.labels.iter().any(|label| {
                label == "ordered_reconstruction"
                    || label == "multi_evidence_summary"
                    || label == "multi_evidence_collection"
                    || label == "instruction_applicable"
            })
        })
        .unwrap_or(false)
}

fn slate_rerank_intent(
    query_intent: Option<&crate::facets::QueryIntent>,
) -> SlateRerankIntent {
    let Some(intent) = query_intent else {
        return SlateRerankIntent::default();
    };
    let target_role = if intent.labels.iter().any(|label| label == "source_user") {
        Some("user")
    } else if intent.labels.iter().any(|label| label == "source_assistant") {
        Some("assistant")
    } else {
        None
    };

    SlateRerankIntent {
        ordered: intent
            .labels
            .iter()
            .any(|label| label == "ordered_reconstruction"),
        summary: intent.labels.iter().any(|label| {
            label == "multi_evidence_summary" || label == "multi_evidence_collection"
        }),
        instruction: intent
            .labels
            .iter()
            .any(|label| label == "instruction_applicable"),
        target_role,
    }
}

fn slate_rerank_query_weights(expanded_cues: &[(String, f64)]) -> HashMap<String, f64> {
    evidence_coverage_query_weights(expanded_cues)
}

fn build_slate_rerank_candidates(
    ctx: &crate::projects::ProjectContext,
    all_results: &[crate::engine::RecallResult],
    expanded_cues: &[(String, f64)],
) -> Vec<SlateRerankCandidate> {
    let query_weights = slate_rerank_query_weights(expanded_cues);
    if query_weights.is_empty() {
        return Vec::new();
    }

    all_results
        .iter()
        .take(SLATE_RERANK_POOL_LIMIT)
        .enumerate()
        .filter_map(|(idx, result)| {
            let memory = ctx.main.get_memory(result.memory_id)?;
            let memory_cues: HashSet<String> =
                memory.cues.iter().map(|cue| cue.to_lowercase()).collect();
            let mut matched_cues = Vec::new();
            for cue in query_weights.keys() {
                if memory_cues.contains(cue) {
                    matched_cues.push(cue.clone());
                }
            }
            matched_cues.sort();
            matched_cues.dedup();

            let (session, order) = ctx
                .main
                .source_order_for_memory(result.memory_id)
                .map(|(session, order)| (Some(session), Some(order)))
                .unwrap_or((None, None));
            let role = evidence_coverage_role(&memory.cues)
                .or_else(|| source_role_from_metadata(&memory.metadata))
                .or_else(|| source_role_from_metadata(&result.metadata));
            let tagged_standing_instruction = memory_cues.contains("type:standing_instruction")
                || result.metadata.contains_key("standing_instruction_projection");

            Some(SlateRerankCandidate {
                idx,
                memory_id: result.memory_id,
                original_rank: idx,
                matched_cues,
                topics: evidence_coverage_topics(&memory.cues),
                session,
                order,
                role,
                source_plan: evidence_coverage_source_plan(&memory.metadata, &memory.cues),
                tagged_evidence: result.metadata.contains_key("evidence_coverage"),
                tagged_ordered: result.metadata.contains_key("ordered_reconstruction"),
                tagged_standing_instruction,
            })
        })
        .collect()
}

fn slate_rerank_candidate_score(
    candidate: &SlateRerankCandidate,
    selected: &[SlateRerankCandidate],
    covered_cues: &HashSet<String>,
    topic_counts: &HashMap<String, usize>,
    order_buckets: &HashSet<(String, i64)>,
    plan_counts: &HashMap<String, usize>,
    intent: SlateRerankIntent,
    pool_len: usize,
) -> f64 {
    let rank_prior = pool_len.saturating_sub(candidate.original_rank) as f64;
    let new_query = candidate
        .matched_cues
        .iter()
        .filter(|cue| !covered_cues.contains(*cue))
        .count();
    let new_topics = candidate
        .topics
        .iter()
        .filter(|topic| topic_counts.get(*topic).copied().unwrap_or(0) == 0)
        .count();
    let repeated_topic_penalty = candidate
        .topics
        .iter()
        .map(|topic| topic_counts.get(topic).copied().unwrap_or(0).min(4) as f64)
        .sum::<f64>();
    let role_bonus = match (intent.target_role, candidate.role.as_deref()) {
        (Some(target), Some(role)) if target == role => 75.0,
        (Some(_), Some(_)) => -95.0,
        (Some(_), None) => -20.0,
        _ => 0.0,
    };
    let evidence_bonus = if candidate.tagged_evidence { 70.0 } else { 0.0 };
    let ordered_bonus = if intent.ordered && candidate.tagged_ordered {
        95.0
    } else if candidate.tagged_ordered {
        45.0
    } else {
        0.0
    };
    let instruction_bonus = if intent.instruction && candidate.tagged_standing_instruction {
        260.0
    } else {
        0.0
    };
    let summary_signal_bonus =
        if intent.summary && slate_rerank_candidate_has_strong_summary_signal(candidate) {
            45.0
        } else {
            0.0
        };
    let order_bonus = match (&candidate.session, candidate.order) {
        (Some(session), Some(order)) => {
            let bucket = order / 8;
            if order_buckets.contains(&(session.clone(), bucket)) {
                -8.0
            } else {
                32.0
            }
        }
        _ => 0.0,
    };
    let plan_bonus = candidate
        .source_plan
        .as_ref()
        .map(|plan| {
            let seen = plan_counts.get(plan).copied().unwrap_or(0).min(4) as f64;
            if seen == 0.0 {
                28.0
            } else {
                -seen * 10.0
            }
        })
        .unwrap_or(0.0);
    let same_session_bonus = candidate
        .session
        .as_ref()
        .map(|session| {
            if selected
                .iter()
                .any(|selected| selected.session.as_ref() == Some(session))
            {
                18.0
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);

    rank_prior
        + evidence_bonus
        + ordered_bonus
        + instruction_bonus
        + summary_signal_bonus
        + role_bonus
        + order_bonus
        + plan_bonus
        + same_session_bonus
        + new_query as f64 * 60.0
        + candidate.matched_cues.len().min(8) as f64 * 12.0
        + new_topics.min(8) as f64 * 10.0
        - repeated_topic_penalty * 4.0
}

fn slate_rerank_candidate_has_strong_summary_signal(candidate: &SlateRerankCandidate) -> bool {
    candidate.matched_cues.len() >= 3
        || (candidate.matched_cues.len() >= 2 && candidate.source_plan.is_some())
        || (candidate.matched_cues.len() >= 2 && candidate.topics.len() >= 3)
}

fn slate_rerank_candidate_is_eligible(
    candidate: &SlateRerankCandidate,
    intent: SlateRerankIntent,
) -> bool {
    if let (Some(target_role), Some(role)) = (intent.target_role, candidate.role.as_deref()) {
        if role != target_role {
            return false;
        }
    }

    if intent.instruction && candidate.tagged_standing_instruction {
        return true;
    }

    if intent.ordered && candidate.tagged_ordered {
        return !candidate.matched_cues.is_empty() || candidate.source_plan.is_some();
    }

    if intent.summary && (candidate.tagged_evidence || candidate.tagged_ordered) {
        return !candidate.matched_cues.is_empty() || candidate.source_plan.is_some();
    }

    if intent.summary && slate_rerank_candidate_has_strong_summary_signal(candidate) {
        return true;
    }

    false
}

fn order_selected_slate_candidates(
    selected: &mut [SlateRerankCandidate],
    intent: SlateRerankIntent,
) {
    if !intent.ordered || selected.len() <= SLATE_RERANK_PROTECTED_RESULTS + 1 {
        return;
    }

    selected[SLATE_RERANK_PROTECTED_RESULTS..].sort_by(|a, b| {
        a.session
            .cmp(&b.session)
            .then_with(|| a.order.cmp(&b.order))
            .then_with(|| a.original_rank.cmp(&b.original_rank))
            .then_with(|| a.memory_id.cmp(&b.memory_id))
    });
}

fn apply_slate_rerank(
    ctx: &crate::projects::ProjectContext,
    all_results: &mut Vec<crate::engine::RecallResult>,
    expanded_cues: &[(String, f64)],
    ordered_mode: OrderedReconstructionMode,
    evidence_mode: EvidenceCoverageMode,
    query_intent: Option<&crate::facets::QueryIntent>,
    limit: usize,
) -> usize {
    if all_results.len() <= SLATE_RERANK_PROTECTED_RESULTS
        || !slate_rerank_requested(ordered_mode, evidence_mode, query_intent)
    {
        return 0;
    }

    all_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let target_len = limit
        .min(SLATE_RERANK_TARGET_LIMIT)
        .min(all_results.len())
        .max(SLATE_RERANK_PROTECTED_RESULTS);
    if target_len <= SLATE_RERANK_PROTECTED_RESULTS {
        return 0;
    }

    let candidates = build_slate_rerank_candidates(ctx, all_results, expanded_cues);
    if candidates.len() <= target_len {
        return 0;
    }

    let intent = slate_rerank_intent(query_intent);
    let mut selected = candidates
        .iter()
        .take(SLATE_RERANK_PROTECTED_RESULTS)
        .cloned()
        .collect::<Vec<_>>();
    let mut selected_ids: HashSet<MemoryId> =
        selected.iter().map(|candidate| candidate.memory_id).collect();
    let mut covered_cues = HashSet::new();
    let mut topic_counts: HashMap<String, usize> = HashMap::new();
    let mut order_buckets = HashSet::new();
    let mut plan_counts: HashMap<String, usize> = HashMap::new();

    for candidate in &selected {
        for cue in &candidate.matched_cues {
            covered_cues.insert(cue.clone());
        }
        for topic in &candidate.topics {
            *topic_counts.entry(topic.clone()).or_insert(0) += 1;
        }
        if let (Some(session), Some(order)) = (&candidate.session, candidate.order) {
            order_buckets.insert((session.clone(), order / 8));
        }
        if let Some(plan) = &candidate.source_plan {
            *plan_counts.entry(plan.clone()).or_insert(0) += 1;
        }
    }

    while selected.len() < target_len {
        let mut best = None::<(usize, f64)>;
        for (candidate_idx, candidate) in candidates.iter().enumerate() {
            if selected_ids.contains(&candidate.memory_id)
                || !slate_rerank_candidate_is_eligible(candidate, intent)
            {
                continue;
            }
            let score = slate_rerank_candidate_score(
                candidate,
                &selected,
                &covered_cues,
                &topic_counts,
                &order_buckets,
                &plan_counts,
                intent,
                candidates.len(),
            );
            if best
                .as_ref()
                .map(|(_, best_score)| score > *best_score)
                .unwrap_or(true)
            {
                best = Some((candidate_idx, score));
            }
        }

        let Some((candidate_idx, _score)) = best else {
            break;
        };
        let candidate = candidates[candidate_idx].clone();
        selected_ids.insert(candidate.memory_id);
        for cue in &candidate.matched_cues {
            covered_cues.insert(cue.clone());
        }
        for topic in &candidate.topics {
            *topic_counts.entry(topic.clone()).or_insert(0) += 1;
        }
        if let (Some(session), Some(order)) = (&candidate.session, candidate.order) {
            order_buckets.insert((session.clone(), order / 8));
        }
        if let Some(plan) = &candidate.source_plan {
            *plan_counts.entry(plan.clone()).or_insert(0) += 1;
        }
        selected.push(candidate);
    }

    if selected.len() <= SLATE_RERANK_PROTECTED_RESULTS {
        return 0;
    }

    order_selected_slate_candidates(&mut selected, intent);

    let protected_ceiling = all_results
        .get(SLATE_RERANK_PROTECTED_RESULTS.saturating_sub(1))
        .map(|result| result.score)
        .unwrap_or(0.0)
        - 0.000_001;
    let mut moved = 0usize;
    for (slot, candidate) in selected
        .iter()
        .enumerate()
        .skip(SLATE_RERANK_PROTECTED_RESULTS)
    {
        if candidate.original_rank > slot {
            moved += 1;
        }
        if let Some(result) = all_results.get_mut(candidate.idx) {
            result.score = result
                .score
                .max(protected_ceiling - 0.001 * (slot as f64 + 1.0));
            result.metadata.insert(
                "slate_rerank".to_string(),
                serde_json::json!({
                    "slot": slot,
                    "original_rank": candidate.original_rank,
                    "matched_cues": candidate.matched_cues,
                    "topics": candidate.topics,
                    "protected_results": SLATE_RERANK_PROTECTED_RESULTS
                }),
            );
        }
    }

    moved
}

const MAX_PARENT_FUSION_LIMIT: usize = 100;
const MAX_PARENT_FUSION_SIBLINGS: usize = 500;
const MAX_PARENT_FUSION_STITCHED_CHUNKS: usize = 16;

#[derive(Clone)]
struct ParentFusionHit {
    result: crate::engine::RecallResult,
    chunk_idx: usize,
}

fn should_run_parent_fusion(
    results: &[crate::engine::RecallResult],
    mode: ParentFusionMode,
    query_intent: Option<&crate::facets::QueryIntent>,
    query_text: Option<&str>,
) -> bool {
    match mode {
        ParentFusionMode::Off => false,
        ParentFusionMode::Force => true,
        ParentFusionMode::Auto => {
            if !query_supports_parent_fusion(query_intent, query_text) {
                return false;
            }

            let Some(top) = results.iter().max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) else {
                return true;
            };

            results.len() < 5 || top.match_integrity < 0.75 || top.intersection_count < 3
        }
    }
}

fn query_supports_parent_fusion(
    query_intent: Option<&crate::facets::QueryIntent>,
    query_text: Option<&str>,
) -> bool {
    if query_intent
        .is_some_and(|intent| intent.labels.iter().any(|label| label == "temporal_order"))
    {
        return true;
    }

    let Some(query_text) = query_text else {
        return false;
    };
    let lower = query_text.to_lowercase();
    [
        "summary",
        "summarize",
        "summarise",
        "overview",
        "in order",
        "chronological",
        "timeline",
        "sequence",
        "progress",
        "progression",
        "evolution",
        "steps",
        "items",
        "list",
        "mention",
        "compare",
        "main points",
        "key points",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn segment_link_from_cues(cues: &[String]) -> Option<(String, usize)> {
    let mut parent_id = None;
    let mut chunk_idx = None;

    for cue in cues {
        if cue.starts_with("parent:") {
            parent_id = Some(cue.clone());
        } else if cue.starts_with("chunk_idx:") {
            chunk_idx = cue
                .split(':')
                .nth(1)
                .and_then(|value| value.parse::<usize>().ok());
        }
    }

    match (parent_id, chunk_idx) {
        (Some(parent_id), Some(chunk_idx)) => Some((parent_id, chunk_idx)),
        _ => None,
    }
}

fn build_parent_fusion_results(
    ctx: &crate::projects::ProjectContext,
    candidate_results: Vec<crate::engine::RecallResult>,
    min_chunks: usize,
    explain: bool,
) -> Vec<crate::engine::RecallResult> {
    let mut groups: HashMap<String, Vec<ParentFusionHit>> = HashMap::new();
    let mut seen = HashSet::new();

    for result in candidate_results {
        let Some(memory) = ctx.main.get_memory(result.memory_id) else {
            continue;
        };
        let Some((parent_id, chunk_idx)) = segment_link_from_cues(&memory.cues) else {
            continue;
        };
        if !seen.insert((parent_id.clone(), chunk_idx)) {
            continue;
        }
        groups
            .entry(parent_id)
            .or_default()
            .push(ParentFusionHit { result, chunk_idx });
    }

    let min_chunks = min_chunks.max(1);
    let mut fused = Vec::new();
    for (parent_id, mut hits) in groups {
        if hits.len() < min_chunks {
            continue;
        }
        hits.sort_by(|a, b| {
            b.result
                .score
                .partial_cmp(&a.result.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk_idx.cmp(&b.chunk_idx))
        });
        if let Some(result) = stitch_parent_fusion_result(ctx, &parent_id, &hits, explain) {
            fused.push(result);
        }
    }

    fused
}

fn stitch_parent_fusion_result(
    ctx: &crate::projects::ProjectContext,
    parent_id: &str,
    hits: &[ParentFusionHit],
    explain: bool,
) -> Option<crate::engine::RecallResult> {
    let best_hit = hits.first()?.result.clone();
    let hit_indices: HashSet<usize> = hits.iter().map(|hit| hit.chunk_idx).collect();
    let contributing_ids: Vec<MemoryId> = hits.iter().map(|hit| hit.result.memory_id).collect();
    let contributing_indices: Vec<usize> = {
        let mut values: Vec<usize> = hit_indices.iter().copied().collect();
        values.sort_unstable();
        values
    };
    let stitched_chunks = stitched_parent_chunks(ctx, parent_id, &hit_indices);

    if stitched_chunks.is_empty() {
        return None;
    }

    let stitched_indices: Vec<usize> = stitched_chunks.iter().map(|(idx, _, _)| *idx).collect();
    let content = join_stitched_chunk_contents(
        &stitched_chunks
            .iter()
            .map(|(_, _, content)| content.clone())
            .collect::<Vec<_>>(),
    );

    let best_score = hits
        .iter()
        .map(|hit| hit.result.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_top = hits
        .iter()
        .take(5)
        .map(|hit| hit.result.score.max(0.0))
        .sum::<f64>();
    let distinct_bonus = (hits.len().saturating_sub(1).min(5) as f64) * 0.18;
    let contiguous_bonus = if has_adjacent_chunk_hit(&contributing_indices) {
        0.08
    } else {
        0.0
    };
    let fused_score =
        best_score * (1.0 + distinct_bonus + contiguous_bonus) + (sum_top - best_score).max(0.0) * 0.35;

    let mut result = best_hit;
    result.content = content;
    result.score = fused_score;
    result.intersection_count = hits
        .iter()
        .map(|hit| hit.result.intersection_count)
        .max()
        .unwrap_or(result.intersection_count)
        + hits.len().saturating_sub(1);
    result.match_integrity = hits
        .iter()
        .map(|hit| hit.result.match_integrity)
        .fold(result.match_integrity, f64::max)
        .min(1.0);
    result
        .metadata
        .insert("parent_fusion".to_string(), serde_json::json!(true));
    result
        .metadata
        .insert("parent_id".to_string(), serde_json::json!(parent_id));
    result.metadata.insert(
        "parent_fusion_contributing_memory_ids".to_string(),
        serde_json::json!(contributing_ids),
    );
    result.metadata.insert(
        "parent_fusion_contributing_chunk_indices".to_string(),
        serde_json::json!(contributing_indices),
    );
    result.metadata.insert(
        "parent_fusion_stitched_chunk_indices".to_string(),
        serde_json::json!(stitched_indices),
    );

    if explain {
        let payload = serde_json::json!({
            "parent_id": parent_id,
            "contributing_memory_ids": contributing_ids,
            "contributing_chunk_count": hits.len(),
            "stitched_chunk_count": stitched_chunks.len(),
            "score": fused_score,
        });
        match result.explain.as_mut().and_then(|value| value.as_object_mut()) {
            Some(obj) => {
                obj.insert("parent_fusion".to_string(), payload);
            }
            None => {
                result.explain = Some(serde_json::json!({ "parent_fusion": payload }));
            }
        }
    }

    Some(result)
}

fn stitched_parent_chunks(
    ctx: &crate::projects::ProjectContext,
    parent_id: &str,
    hit_indices: &HashSet<usize>,
) -> Vec<(usize, MemoryId, String)> {
    let Some(parent_set) = ctx.main.get_cue_index().get(parent_id) else {
        return Vec::new();
    };

    let mut siblings = Vec::new();
    for sibling_id in parent_set.items.iter().take(MAX_PARENT_FUSION_SIBLINGS) {
        let Some(memory) = ctx.main.get_memory(*sibling_id) else {
            continue;
        };
        let Some((_, chunk_idx)) = segment_link_from_cues(&memory.cues) else {
            continue;
        };
        let Ok(content) = ctx.main.read_memory_content(&memory) else {
            continue;
        };
        siblings.push((chunk_idx, *sibling_id, content));
    }
    drop(parent_set);

    siblings.sort_by_key(|(idx, _, _)| *idx);
    siblings.dedup_by_key(|(idx, _, _)| *idx);
    if siblings.len() <= MAX_PARENT_FUSION_STITCHED_CHUNKS {
        return siblings;
    }

    let min_hit = hit_indices.iter().min().copied().unwrap_or(0);
    let max_hit = hit_indices.iter().max().copied().unwrap_or(min_hit);
    let mut selected: Vec<_> = siblings
        .iter()
        .filter(|(idx, _, _)| *idx >= min_hit.saturating_sub(1) && *idx <= max_hit + 1)
        .cloned()
        .collect();

    if selected.len() <= MAX_PARENT_FUSION_STITCHED_CHUNKS {
        return selected;
    }

    selected = siblings
        .iter()
        .filter(|(idx, _, _)| hit_indices.contains(idx))
        .cloned()
        .collect();
    let mut selected_indices: HashSet<usize> = selected.iter().map(|(idx, _, _)| *idx).collect();
    for radius in 1..=2 {
        if selected.len() >= MAX_PARENT_FUSION_STITCHED_CHUNKS {
            break;
        }
        for (hit_idx, _, _) in siblings
            .iter()
            .filter(|(idx, _, _)| hit_indices.contains(idx))
            .cloned()
            .collect::<Vec<_>>()
        {
            for neighbor_idx in [hit_idx.saturating_sub(radius), hit_idx + radius] {
                if selected.len() >= MAX_PARENT_FUSION_STITCHED_CHUNKS {
                    break;
                }
                if !selected_indices.insert(neighbor_idx) {
                    continue;
                }
                if let Some(chunk) = siblings.iter().find(|(idx, _, _)| *idx == neighbor_idx) {
                    selected.push(chunk.clone());
                }
            }
        }
    }

    selected.sort_by_key(|(idx, _, _)| *idx);
    selected.truncate(MAX_PARENT_FUSION_STITCHED_CHUNKS);
    selected
}

fn join_stitched_chunk_contents(contents: &[String]) -> String {
    let mut sentences = Vec::new();
    let mut seen = HashSet::new();

    for content in contents {
        for sentence in content.unicode_sentences() {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let normalized = sentence
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            if seen.insert(normalized) {
                sentences.push(sentence.to_string());
            }
        }
    }

    if sentences.len() >= contents.len() {
        sentences.join(" ")
    } else {
        contents.join("\n\n")
    }
}

fn has_adjacent_chunk_hit(indices: &[usize]) -> bool {
    indices.windows(2).any(|window| window[1] == window[0] + 1)
}

fn merge_parent_fusion_results(
    all_results: &mut Vec<crate::engine::RecallResult>,
    fused_results: Vec<crate::engine::RecallResult>,
) {
    for fused in fused_results {
        if let Some(existing) = all_results
            .iter_mut()
            .find(|existing| existing.memory_id == fused.memory_id)
        {
            if fused.score > existing.score {
                *existing = fused;
            }
        } else {
            all_results.push(fused);
        }
    }
}

fn normalize_source_value(value: &str) -> Option<String> {
    let normalized = value
        .trim()
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_");

    if normalized.len() < 2 || normalized.len() > 64 {
        None
    } else {
        Some(normalized)
    }
}

fn metadata_string<'a>(
    metadata: &'a HashMap<String, serde_json::Value>,
    keys: &[&str],
) -> Option<&'a str> {
    for key in keys {
        if let Some(value) = metadata.get(*key).and_then(|value| value.as_str()) {
            if !value.trim().is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn source_session_cue_from_metadata(
    metadata: &HashMap<String, serde_json::Value>,
) -> Option<String> {
    metadata_string(
        metadata,
        &[
            "source_session_id",
            "session_id",
            "conversation_id",
            "thread_id",
        ],
    )
    .and_then(normalize_source_value)
    .map(|value| format!("source_session:{}", value))
}

fn source_role_from_metadata(metadata: &HashMap<String, serde_json::Value>) -> Option<String> {
    metadata_string(metadata, &["source_role", "role", "speaker", "author_role"])
        .and_then(normalize_source_value)
}

fn source_answer_projection_requested(
    query_intent: Option<&crate::facets::QueryIntent>,
    query_text: Option<&str>,
) -> bool {
    let Some(intent) = query_intent else {
        return false;
    };
    let has_assistant_source = intent.labels.iter().any(|label| label == "source_assistant");
    if has_assistant_source {
        return true;
    }

    if !intent.labels.iter().any(|label| label == "source_answer") {
        return false;
    }

    let lower = query_text.unwrap_or_default().to_lowercase();
    lower.contains("assistant")
        || lower.contains("your answer")
        || lower.contains("the answer you gave")
}

fn query_wants_list_answer(query_text: Option<&str>) -> bool {
    let lower = query_text.unwrap_or_default().to_lowercase();
    lower.contains(" list")
        || lower.contains("listed")
        || lower.contains("1st")
        || lower.contains("2nd")
        || lower.contains("3rd")
        || lower.contains("4th")
        || lower.contains("5th")
        || lower.contains("6th")
        || lower.contains("7th")
        || lower.contains("8th")
        || lower.contains("9th")
        || lower.contains(" first ")
        || lower.contains(" second ")
        || lower.contains(" third ")
        || lower.contains(" fourth ")
        || lower.contains(" fifth ")
        || lower.contains(" sixth ")
        || lower.contains(" seventh ")
        || lower.contains(" eighth ")
        || lower.contains(" ninth ")
}

fn source_answer_projection_cues(
    ctx: &crate::projects::ProjectContext,
    query_intent: Option<&crate::facets::QueryIntent>,
    query_text: Option<&str>,
    all_results: &[crate::engine::RecallResult],
) -> Vec<(String, f64)> {
    if !source_answer_projection_requested(query_intent, query_text) {
        return Vec::new();
    }
    if ctx.main.get_cue_frequency("source_role:assistant") == 0 {
        return Vec::new();
    }

    let mut seen_sessions = std::collections::HashSet::new();
    let mut cues = Vec::new();
    for result in all_results.iter().take(8) {
        if result.intersection_count < 2 && result.match_integrity < 0.20 {
            continue;
        }
        let Some(session_cue) = source_session_cue_from_metadata(&result.metadata) else {
            continue;
        };
        if !seen_sessions.insert(session_cue.clone()) {
            continue;
        }
        cues.push((session_cue, 4.0));
        if seen_sessions.len() >= 3 {
            break;
        }
    }

    if cues.is_empty() {
        return Vec::new();
    }

    cues.push(("source_role:assistant".to_string(), 3.0));
    if query_wants_list_answer(query_text) {
        if ctx.main.get_cue_frequency("has:list") > 0 {
            cues.push(("has:list".to_string(), 2.8));
        }
        if ctx.main.get_cue_frequency("has:number") > 0 {
            cues.push(("has:number".to_string(), 1.0));
        }
    }
    cues
}

fn source_prompt_projection_cues(
    ctx: &crate::projects::ProjectContext,
    query_intent: Option<&crate::facets::QueryIntent>,
    query_text: Option<&str>,
    all_results: &[crate::engine::RecallResult],
) -> Vec<(String, f64)> {
    let Some(intent) = query_intent else {
        return Vec::new();
    };
    if !intent.labels.iter().any(|label| label == "source_answer")
        || !intent.labels.iter().any(|label| label == "source_assistant")
    {
        return Vec::new();
    }
    if ctx.main.get_cue_frequency("source_role:user") == 0 {
        return Vec::new();
    }

    let anchors = source_prompt_anchor_cues(query_text);
    if anchors.len() < 2 {
        return Vec::new();
    }

    let mut seen_sessions = std::collections::HashSet::new();
    let mut cues = Vec::new();
    for result in all_results.iter().take(8) {
        if source_role_from_metadata(&result.metadata).as_deref() != Some("assistant") {
            continue;
        }
        if result.intersection_count < 2 && result.match_integrity < 0.20 {
            continue;
        }
        let Some(session_cue) = source_session_cue_from_metadata(&result.metadata) else {
            continue;
        };
        if !seen_sessions.insert(session_cue.clone()) {
            continue;
        }
        cues.push((session_cue, 4.0));
        if seen_sessions.len() >= 3 {
            break;
        }
    }

    if cues.is_empty() {
        return Vec::new();
    }

    cues.push(("source_role:user".to_string(), 3.0));
    for anchor in anchors.into_iter().take(8) {
        if ctx.main.get_cue_frequency(&anchor) > 0 {
            cues.push((anchor, 1.8));
        }
    }
    cues
}

fn user_context_projection_requested(
    query_intent: Option<&crate::facets::QueryIntent>,
    query_text: Option<&str>,
) -> bool {
    if query_intent
        .map(|intent| {
            intent.labels.iter().any(|label| {
                matches!(
                    label.as_str(),
                    "source_answer" | "source_assistant" | "source_user" | "decision_selection"
                )
            })
        })
        .unwrap_or(false)
    {
        return false;
    }

    let lower = query_text.unwrap_or_default().to_lowercase();
    let explicit_recommendation = lower.contains("recommend") || lower.contains("suggest");
    let generic_recommendation_request = explicit_recommendation
        && (lower.contains("something")
            || lower.contains("anything")
            || lower.contains("some recommendations")
            || lower.contains("some suggestions"));

    lower.contains("tip")
        || lower.contains("advice")
        || generic_recommendation_request
        || lower.contains("what should")
        || lower.contains("should i")
        || lower.contains("trouble")
        || lower.contains("problem")
        || lower.contains("help with")
        || lower.contains("looking for")
}

fn projection_anchor_is_generic(cue: &str) -> bool {
    if cue.starts_with("type:")
        || cue.starts_with("has:")
        || cue.starts_with("temporal:")
        || cue.starts_with("source_")
    {
        return true;
    }

    if cue.contains('_') {
        let parts = cue.split('_').collect::<Vec<_>>();
        let generic_count = parts
            .iter()
            .filter(|part| projection_anchor_is_generic(part))
            .count();
        let specific_count = parts.len().saturating_sub(generic_count);
        return generic_count == parts.len() || (generic_count > 0 && specific_count <= 1);
    }

    matches!(
        cue,
        "can"
            | "could"
            | "would"
            | "should"
            | "please"
            | "recommend"
            | "recommendation"
            | "suggest"
            | "suggestion"
            | "idea"
            | "ideas"
            | "tip"
            | "tips"
            | "advice"
            | "help"
            | "helpful"
            | "useful"
            | "good"
            | "best"
            | "some"
            | "any"
            | "thing"
            | "something"
            | "bit"
            | "get"
            | "got"
            | "getting"
            | "around"
            | "show"
            | "shows"
            | "movie"
            | "movies"
            | "watch"
            | "tonight"
            | "new"
            | "current"
            | "recent"
            | "lately"
    )
}

fn projection_anchor_cues(query_text: Option<&str>) -> Vec<String> {
    let Some(query_text) = query_text else {
        return Vec::new();
    };

    let mut seen = std::collections::HashSet::new();
    let mut anchors = Vec::new();
    for cue in crate::nl::tokenize_to_cues(query_text) {
        let cue = cue.trim().to_lowercase();
        if cue.len() <= 2 || projection_anchor_is_generic(&cue) || !seen.insert(cue.clone()) {
            continue;
        }
        anchors.push(cue);
        if anchors.len() >= 8 {
            break;
        }
    }
    anchors
}

fn source_prompt_anchor_is_generic(cue: &str) -> bool {
    if cue.starts_with("type:")
        || cue.starts_with("has:")
        || cue.starts_with("temporal:")
        || cue.starts_with("source_")
    {
        return true;
    }

    if cue.contains('_') {
        let parts = cue.split('_').collect::<Vec<_>>();
        return parts.iter().all(|part| source_prompt_anchor_is_generic(part));
    }

    matches!(
        cue,
        "can"
            | "could"
            | "would"
            | "should"
            | "please"
            | "look"
            | "back"
            | "go"
            | "going"
            | "previous"
            | "conversation"
            | "chat"
            | "wonder"
            | "wondering"
            | "remind"
            | "remember"
            | "tell"
            | "ask"
            | "asked"
            | "answer"
            | "answered"
            | "response"
            | "respond"
            | "said"
    )
}

fn source_prompt_anchor_cues(query_text: Option<&str>) -> Vec<String> {
    let Some(query_text) = query_text else {
        return Vec::new();
    };

    let mut seen = std::collections::HashSet::new();
    let mut anchors = Vec::new();
    for cue in crate::nl::tokenize_to_cues(query_text) {
        let cue = cue.trim().to_lowercase();
        if cue.len() <= 2 || source_prompt_anchor_is_generic(&cue) || !seen.insert(cue.clone()) {
            continue;
        }
        anchors.push(cue);
        if anchors.len() >= 12 {
            break;
        }
    }
    anchors
}

fn projection_anchor_match_count(content: &str, anchors: &[String]) -> usize {
    if anchors.is_empty() {
        return 0;
    }
    let content_cues = crate::nl::tokenize_to_cues(content)
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    anchors
        .iter()
        .filter(|anchor| content_cues.contains(anchor.as_str()))
        .count()
}

fn projection_pivot_matches_context(
    content: &str,
    intersection_count: usize,
    anchor_cues: &[String],
    allow_high_confidence_pivot: bool,
) -> bool {
    if anchor_cues.len() >= 2 {
        return projection_anchor_match_count(content, anchor_cues) >= 2;
    }

    allow_high_confidence_pivot && intersection_count >= 4
}

fn suppress_user_context_projection_for_intent(
    query_intent: Option<&crate::facets::QueryIntent>,
) -> bool {
    query_intent
        .map(|intent| {
            intent
                .labels
                .iter()
                .any(|label| label == "vague_interest_recommendation")
        })
        .unwrap_or(false)
}

fn user_context_projection_cues(
    ctx: &crate::projects::ProjectContext,
    query_intent: Option<&crate::facets::QueryIntent>,
    query_text: Option<&str>,
    all_results: &[crate::engine::RecallResult],
) -> Vec<(String, f64)> {
    if !user_context_projection_requested(query_intent, query_text) {
        return Vec::new();
    }
    if ctx.main.get_cue_frequency("source_role:user") == 0 {
        return Vec::new();
    }

    if suppress_user_context_projection_for_intent(query_intent) {
        return Vec::new();
    }

    let allow_high_confidence_pivot = query_intent
        .map(|intent| {
            intent
                .labels
                .iter()
                .any(|label| label == "personal_recommendation_context")
                && !intent
                    .labels
                    .iter()
                    .any(|label| label == "vague_interest_recommendation")
        })
        .unwrap_or(false);
    let anchor_cues = projection_anchor_cues(query_text);
    if anchor_cues.len() < 2 && !allow_high_confidence_pivot {
        return Vec::new();
    }

    let mut seen_sessions = std::collections::HashSet::new();
    let mut cues = Vec::new();
    for result in all_results.iter().take(10) {
        if source_role_from_metadata(&result.metadata).as_deref() != Some("assistant") {
            continue;
        }
        if result.intersection_count < 2 {
            continue;
        }
        if !projection_pivot_matches_context(
            &result.content,
            result.intersection_count,
            &anchor_cues,
            allow_high_confidence_pivot,
        ) {
            continue;
        }
        let Some(session_cue) = source_session_cue_from_metadata(&result.metadata) else {
            continue;
        };
        if !seen_sessions.insert(session_cue.clone()) {
            continue;
        }
        cues.push((session_cue, 4.0));
        if seen_sessions.len() >= 2 {
            break;
        }
    }

    if cues.is_empty() {
        return Vec::new();
    }

    cues.push(("source_role:user".to_string(), 3.0));
    cues
}

fn merge_source_answer_projection_results(
    all_results: &mut Vec<crate::engine::RecallResult>,
    mut projection_results: Vec<crate::engine::RecallResult>,
) {
    for result in &mut projection_results {
        result.metadata.insert(
            "source_answer_projection".to_string(),
            serde_json::json!(true),
        );
    }

    for result in projection_results {
        if let Some(existing) = all_results
            .iter_mut()
            .find(|existing| existing.memory_id == result.memory_id)
        {
            if result.score > existing.score {
                existing.score = result.score;
                existing.match_integrity = result.match_integrity;
                existing.intersection_count = result.intersection_count;
                existing.recency_score = result.recency_score;
                existing.reinforcement_score = result.reinforcement_score;
                existing.salience_score = result.salience_score;
                existing.explain = result.explain.clone();
            }
            existing.metadata.insert(
                "source_answer_projection".to_string(),
                serde_json::json!(true),
            );
        } else {
            all_results.push(result);
        }
    }
}

fn source_prompt_projection_min_intersection(query_text: Option<&str>) -> usize {
    let anchors = source_prompt_anchor_cues(query_text);
    if anchors.len() >= 4 {
        4
    } else {
        3
    }
}

fn merge_source_prompt_projection_results(
    all_results: &mut Vec<crate::engine::RecallResult>,
    mut projection_results: Vec<crate::engine::RecallResult>,
    query_text: Option<&str>,
) {
    let min_intersection = source_prompt_projection_min_intersection(query_text);
    for result in &mut projection_results {
        result.metadata.insert(
            "source_prompt_projection".to_string(),
            serde_json::json!(true),
        );
        let bonus = (result.intersection_count.saturating_sub(1) as f64) * 700.0;
        result.score += bonus;
        result.metadata.insert(
            "source_prompt_projection_boost".to_string(),
            serde_json::json!({
                "min_intersection": min_intersection,
                "bonus": bonus
            }),
        );
    }

    for result in projection_results {
        if source_role_from_metadata(&result.metadata).as_deref() != Some("user") {
            continue;
        }
        if result.intersection_count < min_intersection {
            continue;
        }
        if let Some(existing) = all_results
            .iter_mut()
            .find(|existing| existing.memory_id == result.memory_id)
        {
            if result.score > existing.score {
                existing.score = result.score;
                existing.match_integrity = result.match_integrity;
                existing.intersection_count = result.intersection_count;
                existing.recency_score = result.recency_score;
                existing.reinforcement_score = result.reinforcement_score;
                existing.salience_score = result.salience_score;
                existing.explain = result.explain.clone();
            }
            existing.metadata.insert(
                "source_prompt_projection".to_string(),
                serde_json::json!(true),
            );
        } else {
            all_results.push(result);
        }
    }
}

fn merge_user_context_projection_results(
    all_results: &mut Vec<crate::engine::RecallResult>,
    mut projection_results: Vec<crate::engine::RecallResult>,
) {
    for result in &mut projection_results {
        result.metadata.insert(
            "user_context_projection".to_string(),
            serde_json::json!(true),
        );
    }

    for result in projection_results {
        if source_role_from_metadata(&result.metadata).as_deref() != Some("user") {
            continue;
        }
        if let Some(existing) = all_results
            .iter_mut()
            .find(|existing| existing.memory_id == result.memory_id)
        {
            if result.score > existing.score {
                existing.score = result.score;
                existing.match_integrity = result.match_integrity;
                existing.intersection_count = result.intersection_count;
                existing.recency_score = result.recency_score;
                existing.reinforcement_score = result.reinforcement_score;
                existing.salience_score = result.salience_score;
                existing.explain = result.explain.clone();
            }
            existing.metadata.insert(
                "user_context_projection".to_string(),
                serde_json::json!(true),
            );
        } else {
            all_results.push(result);
        }
    }
}

#[derive(Debug, Clone, Default)]
struct StandingInstructionProjection {
    cues: Vec<(String, f64)>,
    anchors: Vec<String>,
}

fn standing_instruction_projection_requested(
    query_intent: Option<&crate::facets::QueryIntent>,
) -> bool {
    query_intent
        .map(|intent| {
            intent
                .labels
                .iter()
                .any(|label| label == "instruction_applicable")
        })
        .unwrap_or(false)
}

fn standing_instruction_anchor_is_generic(cue: &str) -> bool {
    if cue.starts_with("type:")
        || cue.starts_with("has:")
        || cue.starts_with("temporal:")
        || cue.starts_with("source_")
    {
        return true;
    }

    if cue.contains('_') {
        let parts = cue.split('_').collect::<Vec<_>>();
        return parts.iter().all(|part| standing_instruction_anchor_is_generic(part));
    }

    crate::nl::get_stopwords().contains(cue)
        || matches!(
            cue,
            "what"
                | "which"
                | "how"
                | "should"
                | "could"
                | "would"
                | "can"
                | "explain"
                | "help"
                | "recommend"
                | "show"
                | "tell"
                | "know"
                | "consider"
                | "thing"
                | "things"
                | "way"
                | "ways"
                | "good"
                | "common"
                | "typical"
                | "current"
                | "recent"
                | "problem"
                | "problems"
        )
}

fn projection_anchor_variants(cue: &str) -> Vec<String> {
    let mut variants = Vec::new();

    if matches!(cue, "chance" | "odds" | "likelihood" | "probability") {
        variants.extend(
            ["chance", "odds", "likelihood", "probability"]
                .iter()
                .map(|variant| (*variant).to_string()),
        );
    }

    if cue.len() > 3 {
        if let Some(stem) = cue.strip_suffix('s') {
            variants.push(stem.to_string());
        } else {
            variants.push(format!("{cue}s"));
        }
    }
    if let Some(stem) = cue.strip_suffix("ing") {
        if stem.len() > 2 {
            variants.push(stem.to_string());
            variants.push(format!("{stem}e"));
        }
    }
    if let Some(stem) = cue.strip_suffix("ed") {
        if stem.len() > 2 {
            variants.push(stem.to_string());
            variants.push(format!("{stem}e"));
        }
    }
    if let Some(stem) = cue.strip_suffix("ize") {
        if stem.len() > 2 {
            variants.push(format!("{stem}ization"));
        }
    }
    if let Some(stem) = cue.strip_suffix("ise") {
        if stem.len() > 2 {
            variants.push(format!("{stem}isation"));
        }
    }
    if let Some(stem) = cue.strip_suffix("ate") {
        if stem.len() > 2 {
            variants.push(format!("{stem}ation"));
        }
    }
    if let Some(stem) = cue.strip_suffix("ure") {
        if stem.len() > 2 {
            variants.push(format!("{stem}uration"));
        }
    }
    if cue.len() > 4 && cue.ends_with('t') {
        variants.push(format!("{cue}ation"));
    }
    if let Some(stem) = cue.strip_suffix("ization") {
        if stem.len() > 2 {
            variants.push(format!("{stem}ize"));
        }
    }
    if let Some(stem) = cue.strip_suffix("isation") {
        if stem.len() > 2 {
            variants.push(format!("{stem}ise"));
        }
    }
    if let Some(stem) = cue.strip_suffix("ation") {
        if stem.len() > 2 {
            variants.push(stem.to_string());
            variants.push(format!("{stem}ate"));
        }
    }

    variants.retain(|variant| variant != cue && variant.len() > 2);
    variants.sort();
    variants.dedup();
    variants
}

fn standing_instruction_projection_anchors(query_text: Option<&str>) -> Vec<String> {
    let Some(query_text) = query_text else {
        return Vec::new();
    };

    let mut base_seen = std::collections::HashSet::new();
    let mut base_anchors = Vec::new();
    for cue in crate::nl::tokenize_to_cues(query_text) {
        let cue = cue.trim().to_lowercase();
        if cue.len() <= 2
            || standing_instruction_anchor_is_generic(&cue)
            || !base_seen.insert(cue.clone())
        {
            continue;
        }
        base_anchors.push(cue);
        if base_anchors.len() >= 10 {
            break;
        }
    }

    let mut seen = std::collections::HashSet::new();
    let mut anchors = Vec::new();
    'outer: for cue in base_anchors {
        for candidate in std::iter::once(cue.clone()).chain(projection_anchor_variants(&cue)) {
            if candidate.len() <= 2
                || standing_instruction_anchor_is_generic(&candidate)
                || !seen.insert(candidate.clone())
            {
                continue;
            }
            anchors.push(candidate);
            if anchors.len() >= 18 {
                break 'outer;
            }
        }
    }
    anchors
}

fn standing_instruction_projection_cues(
    ctx: &crate::projects::ProjectContext,
    query_intent: Option<&crate::facets::QueryIntent>,
    query_text: Option<&str>,
) -> StandingInstructionProjection {
    if !standing_instruction_projection_requested(query_intent) {
        return StandingInstructionProjection::default();
    }
    if ctx.main.get_cue_frequency("type:standing_instruction") == 0 {
        return StandingInstructionProjection::default();
    }

    let anchors = standing_instruction_projection_anchors(query_text);
    if anchors.is_empty() {
        return StandingInstructionProjection::default();
    }

    let mut cues = vec![("type:standing_instruction".to_string(), 6.0)];
    if ctx.main.get_cue_frequency("instruction:conditional") > 0 {
        cues.push(("instruction:conditional".to_string(), 2.4));
    }
    if ctx.main.get_cue_frequency("instruction:always") > 0 {
        cues.push(("instruction:always".to_string(), 2.2));
    }

    let mut specific_cues = 0usize;
    let mut seen = std::collections::HashSet::new();
    for anchor in &anchors {
        let trigger_cue = format!("instruction_trigger:{anchor}");
        if ctx.main.get_cue_frequency(&trigger_cue) > 0 && seen.insert(trigger_cue.clone()) {
            cues.push((trigger_cue, 5.0));
            specific_cues += 1;
        }
        if ctx.main.get_cue_frequency(anchor) > 0 && seen.insert(anchor.clone()) {
            cues.push((anchor.clone(), 1.8));
            specific_cues += 1;
        }
    }

    if specific_cues == 0 {
        return StandingInstructionProjection::default();
    }

    StandingInstructionProjection { cues, anchors }
}

fn standing_instruction_projection_matched_anchors(
    memory_cues: &[String],
    anchors: &[String],
) -> Vec<String> {
    let memory_cues = memory_cues
        .iter()
        .map(|cue| cue.as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut matched = Vec::new();
    for anchor in anchors {
        let trigger = format!("instruction_trigger:{anchor}");
        if memory_cues.contains(trigger.as_str()) || memory_cues.contains(anchor.as_str()) {
            matched.push(anchor.clone());
        }
    }
    matched.sort();
    matched.dedup();
    matched
}

fn merge_standing_instruction_projection_results(
    ctx: &crate::projects::ProjectContext,
    all_results: &mut Vec<crate::engine::RecallResult>,
    mut projection_results: Vec<crate::engine::RecallResult>,
    anchors: &[String],
) {
    for result in &mut projection_results {
        let Some(memory) = ctx.main.get_memory(result.memory_id) else {
            continue;
        };
        if !memory
            .cues
            .iter()
            .any(|cue| cue == "type:standing_instruction")
        {
            continue;
        }
        let matched = standing_instruction_projection_matched_anchors(&memory.cues, anchors);
        if matched.is_empty() {
            continue;
        }

        let bonus = 900.0 + (matched.len() as f64 * 220.0);
        result.score += bonus;
        result.metadata.insert(
            "standing_instruction_projection".to_string(),
            serde_json::json!(true),
        );
        result.metadata.insert(
            "standing_instruction_projection_matched_anchors".to_string(),
            serde_json::json!(matched),
        );
        result.metadata.insert(
            "standing_instruction_projection_boost".to_string(),
            serde_json::json!(bonus),
        );
    }

    for result in projection_results {
        if !result
            .metadata
            .contains_key("standing_instruction_projection")
        {
            continue;
        }
        if let Some(existing) = all_results
            .iter_mut()
            .find(|existing| existing.memory_id == result.memory_id)
        {
            if result.score > existing.score {
                existing.score = result.score;
                existing.match_integrity = result.match_integrity;
                existing.intersection_count = result.intersection_count;
                existing.recency_score = result.recency_score;
                existing.reinforcement_score = result.reinforcement_score;
                existing.salience_score = result.salience_score;
                existing.explain = result.explain.clone();
            }
            existing.metadata.insert(
                "standing_instruction_projection".to_string(),
                serde_json::json!(true),
            );
            if let Some(matched) = result
                .metadata
                .get("standing_instruction_projection_matched_anchors")
                .cloned()
            {
                existing.metadata.insert(
                    "standing_instruction_projection_matched_anchors".to_string(),
                    matched,
                );
            }
        } else {
            all_results.push(result);
        }
    }
}

#[derive(Debug, Clone, Default)]
struct PreferenceProjection {
    cues: Vec<(String, f64)>,
    anchors: Vec<String>,
}

fn preference_projection_requested(query_intent: Option<&crate::facets::QueryIntent>) -> bool {
    query_intent
        .map(|intent| {
            intent
                .labels
                .iter()
                .any(|label| label == "preference_applicable")
        })
        .unwrap_or(false)
}

fn preference_projection_anchor_is_generic(cue: &str) -> bool {
    if cue.starts_with("type:")
        || cue.starts_with("has:")
        || cue.starts_with("temporal:")
        || cue.starts_with("source_")
    {
        return true;
    }

    if cue.contains('_') {
        let parts = cue.split('_').collect::<Vec<_>>();
        return parts.iter().all(|part| preference_projection_anchor_is_generic(part));
    }

    crate::nl::get_stopwords().contains(cue)
        || matches!(
            cue,
            "what"
                | "which"
                | "how"
                | "should"
                | "could"
                | "would"
                | "can"
                | "explain"
                | "help"
                | "recommend"
                | "show"
                | "suggest"
                | "walk"
                | "tell"
                | "know"
                | "consider"
                | "thing"
                | "things"
                | "way"
                | "ways"
                | "good"
                | "common"
                | "typical"
                | "current"
                | "recent"
                | "try"
                | "trying"
                | "option"
                | "options"
                | "step"
                | "steps"
                | "plan"
                | "planning"
                | "prepare"
                | "experience"
                | "use"
                | "using"
                | "find"
                | "get"
        )
}

fn preference_projection_anchors(query_text: Option<&str>) -> Vec<String> {
    let Some(query_text) = query_text else {
        return Vec::new();
    };

    let mut base_seen = std::collections::HashSet::new();
    let mut base_anchors = Vec::new();
    for cue in crate::nl::tokenize_to_cues(query_text) {
        let cue = cue.trim().to_lowercase();
        if cue.len() <= 2
            || preference_projection_anchor_is_generic(&cue)
            || !base_seen.insert(cue.clone())
        {
            continue;
        }
        base_anchors.push(cue);
        if base_anchors.len() >= 12 {
            break;
        }
    }

    let mut seen = std::collections::HashSet::new();
    let mut anchors = Vec::new();
    'outer: for cue in base_anchors {
        for candidate in std::iter::once(cue.clone()).chain(projection_anchor_variants(&cue)) {
            if candidate.len() <= 2
                || preference_projection_anchor_is_generic(&candidate)
                || !seen.insert(candidate.clone())
            {
                continue;
            }
            anchors.push(candidate);
            if anchors.len() >= 20 {
                break 'outer;
            }
        }
    }
    anchors
}

fn preference_projection_cues(
    ctx: &crate::projects::ProjectContext,
    query_intent: Option<&crate::facets::QueryIntent>,
    query_text: Option<&str>,
) -> PreferenceProjection {
    if !preference_projection_requested(query_intent) {
        return PreferenceProjection::default();
    }

    let has_preference = ctx.main.get_cue_frequency("type:preference") > 0;
    let has_dislike = ctx.main.get_cue_frequency("type:dislike") > 0;
    if !has_preference && !has_dislike {
        return PreferenceProjection::default();
    }

    let anchors = preference_projection_anchors(query_text);
    if anchors.is_empty() {
        return PreferenceProjection::default();
    }

    let mut cues = Vec::new();
    if has_preference {
        cues.push(("type:preference".to_string(), 4.8));
    }
    if has_dislike {
        cues.push(("type:dislike".to_string(), 2.0));
    }
    if ctx.main.get_cue_frequency("source_role:user") > 0 {
        cues.push(("source_role:user".to_string(), 1.6));
    }
    if ctx.main.get_cue_frequency("preference:explicit") > 0 {
        cues.push(("preference:explicit".to_string(), 2.4));
    }

    let mut specific_cues = 0usize;
    let mut seen = std::collections::HashSet::new();
    for anchor in &anchors {
        for (prefix, weight) in [
            ("preference_value", 4.2),
            ("preference_topic", 4.0),
            ("preference_contrast", 1.4),
        ] {
            let cue = format!("{prefix}:{anchor}");
            if ctx.main.get_cue_frequency(&cue) > 0 && seen.insert(cue.clone()) {
                cues.push((cue, weight));
                specific_cues += 1;
            }
        }
        if ctx.main.get_cue_frequency(anchor) > 0 && seen.insert(anchor.clone()) {
            cues.push((anchor.clone(), 1.7));
            specific_cues += 1;
        }
    }

    if specific_cues == 0 {
        return PreferenceProjection::default();
    }

    PreferenceProjection { cues, anchors }
}

fn preference_projection_matched_anchors(
    memory_cues: &[String],
    anchors: &[String],
) -> (Vec<String>, usize) {
    let memory_cues = memory_cues
        .iter()
        .map(|cue| cue.as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut matched = Vec::new();
    let mut preference_specific = 0usize;
    for anchor in anchors {
        let value = format!("preference_value:{anchor}");
        let topic = format!("preference_topic:{anchor}");
        let contrast = format!("preference_contrast:{anchor}");
        let specific_match = memory_cues.contains(value.as_str())
            || memory_cues.contains(topic.as_str())
            || memory_cues.contains(contrast.as_str());
        if specific_match {
            preference_specific += 1;
        }
        if specific_match || memory_cues.contains(anchor.as_str()) {
            matched.push(anchor.clone());
        }
    }
    matched.sort();
    matched.dedup();
    (matched, preference_specific)
}

fn merge_preference_projection_results(
    ctx: &crate::projects::ProjectContext,
    all_results: &mut Vec<crate::engine::RecallResult>,
    mut projection_results: Vec<crate::engine::RecallResult>,
    anchors: &[String],
) {
    for result in &mut projection_results {
        let Some(memory) = ctx.main.get_memory(result.memory_id) else {
            continue;
        };
        let is_preference_memory = memory
            .cues
            .iter()
            .any(|cue| cue == "type:preference" || cue == "type:dislike");
        if !is_preference_memory {
            continue;
        }

        let (matched, preference_specific) =
            preference_projection_matched_anchors(&memory.cues, anchors);
        if matched.is_empty() || (preference_specific == 0 && matched.len() < 2) {
            continue;
        }

        let bonus = 650.0 + (matched.len() as f64 * 180.0) + (preference_specific as f64 * 120.0);
        result.score += bonus;
        result.metadata.insert(
            "preference_projection".to_string(),
            serde_json::json!(true),
        );
        result.metadata.insert(
            "preference_projection_matched_anchors".to_string(),
            serde_json::json!(matched),
        );
        result.metadata.insert(
            "preference_projection_boost".to_string(),
            serde_json::json!(bonus),
        );
    }

    for result in projection_results {
        if !result.metadata.contains_key("preference_projection") {
            continue;
        }
        if let Some(existing) = all_results
            .iter_mut()
            .find(|existing| existing.memory_id == result.memory_id)
        {
            if result.score > existing.score {
                existing.score = result.score;
                existing.match_integrity = result.match_integrity;
                existing.intersection_count = result.intersection_count;
                existing.recency_score = result.recency_score;
                existing.reinforcement_score = result.reinforcement_score;
                existing.salience_score = result.salience_score;
                existing.explain = result.explain.clone();
            }
            existing.metadata.insert(
                "preference_projection".to_string(),
                serde_json::json!(true),
            );
            if let Some(matched) = result
                .metadata
                .get("preference_projection_matched_anchors")
                .cloned()
            {
                existing.metadata.insert(
                    "preference_projection_matched_anchors".to_string(),
                    matched,
                );
            }
        } else {
            all_results.push(result);
        }
    }
}

fn apply_user_context_adjacency_preference(
    all_results: &mut [crate::engine::RecallResult],
    query_intent: Option<&crate::facets::QueryIntent>,
    query_text: Option<&str>,
) {
    const MAX_ADJACENCY_PIVOTS: usize = 4;

    if !user_context_projection_requested(query_intent, query_text) {
        return;
    }

    let projected_sessions = all_results
        .iter()
        .filter_map(|result| {
            if source_role_from_metadata(&result.metadata).as_deref() != Some("user") {
                return None;
            }
            if !result.metadata.contains_key("user_context_projection") {
                return None;
            }
            source_session_cue_from_metadata(&result.metadata)
        })
        .collect::<std::collections::HashSet<_>>();
    if projected_sessions.is_empty() {
        return;
    }

    let mut pivots = all_results
        .iter()
        .enumerate()
        .filter_map(|(idx, result)| {
            if source_role_from_metadata(&result.metadata).as_deref() != Some("assistant") {
                return None;
            }
            if result.intersection_count < 2 {
                return None;
            }
            let session = source_session_cue_from_metadata(&result.metadata)?;
            if !projected_sessions.contains(&session) {
                return None;
            }
            Some((idx, session, result.created_at, result.score))
        })
        .collect::<Vec<_>>();
    pivots.sort_by(|(_, _, _, left), (_, _, _, right)| {
        right
            .partial_cmp(left)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut boosts = Vec::new();
    for (pivot_idx, pivot_session, _, pivot_score) in
        pivots.into_iter().take(MAX_ADJACENCY_PIVOTS)
    {
        let mut session_positions = all_results
            .iter()
            .enumerate()
            .filter_map(|(idx, result)| {
                if source_session_cue_from_metadata(&result.metadata).as_deref()
                    == Some(pivot_session.as_str())
                {
                    Some((idx, result.created_at))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        session_positions.sort_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        });

        let Some(pivot_position) = session_positions
            .iter()
            .position(|(idx, _)| *idx == pivot_idx)
        else {
            continue;
        };

        for (position, (idx, _)) in session_positions
            .iter()
            .enumerate()
            .take(pivot_position)
            .rev()
            .take(2)
        {
            let distance = pivot_position - position;
            if distance > 4 {
                break;
            }
            if source_role_from_metadata(&all_results[*idx].metadata).as_deref() != Some("user") {
                continue;
            }
            if !all_results[*idx]
                .metadata
                .contains_key("user_context_projection")
            {
                continue;
            }
            let bonus = (pivot_score * 2.0) / (distance * distance) as f64;
            boosts.push((*idx, pivot_session.clone(), distance, bonus));
        }
    }

    let mut best_boosts = std::collections::BTreeMap::new();
    for (idx, pivot_session, distance, bonus) in boosts {
        let replace = best_boosts
            .get(&idx)
            .map(|(_, _, existing_bonus): &(String, usize, f64)| bonus > *existing_bonus)
            .unwrap_or(true);
        if replace {
            best_boosts.insert(idx, (pivot_session, distance, bonus));
        }
    }

    for (idx, (pivot_session, distance, bonus)) in best_boosts {
        all_results[idx].score += bonus;
        all_results[idx].metadata.insert(
            "user_context_adjacency_boost".to_string(),
            serde_json::json!({
                "pivot_session": pivot_session,
                "distance": distance,
                "bonus": bonus
            }),
        );
    }
}

fn decision_projection_requested(query_intent: Option<&crate::facets::QueryIntent>) -> bool {
    query_intent
        .map(|intent| {
            intent
                .labels
                .iter()
                .any(|label| label == "decision_selection")
        })
        .unwrap_or(false)
}

fn decision_projection_cues(
    ctx: &crate::projects::ProjectContext,
    query_intent: Option<&crate::facets::QueryIntent>,
    all_results: &[crate::engine::RecallResult],
) -> Vec<(String, f64)> {
    if !decision_projection_requested(query_intent) {
        return Vec::new();
    }

    let mut seen_sessions = std::collections::HashSet::new();
    let mut cues = Vec::new();
    for result in all_results.iter().take(8) {
        if result.intersection_count < 2 && result.match_integrity < 0.20 {
            continue;
        }
        let Some(session_cue) = source_session_cue_from_metadata(&result.metadata) else {
            continue;
        };
        if !seen_sessions.insert(session_cue.clone()) {
            continue;
        }
        cues.push((session_cue, 4.0));
        if seen_sessions.len() >= 3 {
            break;
        }
    }

    if cues.is_empty() {
        return Vec::new();
    }

    if ctx.main.get_cue_frequency("type:decision") > 0 {
        cues.push(("type:decision".to_string(), 3.0));
    }
    if ctx.main.get_cue_frequency("type:selection") > 0 {
        cues.push(("type:selection".to_string(), 2.6));
    }
    if query_intent
        .map(|intent| intent.labels.iter().any(|label| label == "naming_decision"))
        .unwrap_or(false)
        && ctx.main.get_cue_frequency("type:naming") > 0
    {
        cues.push(("type:naming".to_string(), 2.4));
    }
    cues
}

fn merge_decision_projection_results(
    all_results: &mut Vec<crate::engine::RecallResult>,
    mut projection_results: Vec<crate::engine::RecallResult>,
) {
    for result in &mut projection_results {
        result
            .metadata
            .insert("decision_projection".to_string(), serde_json::json!(true));
    }

    for result in projection_results {
        if let Some(existing) = all_results
            .iter_mut()
            .find(|existing| existing.memory_id == result.memory_id)
        {
            if result.score > existing.score {
                existing.score = result.score;
                existing.match_integrity = result.match_integrity;
                existing.intersection_count = result.intersection_count;
                existing.recency_score = result.recency_score;
                existing.reinforcement_score = result.reinforcement_score;
                existing.salience_score = result.salience_score;
                existing.explain = result.explain.clone();
            }
            existing
                .metadata
                .insert("decision_projection".to_string(), serde_json::json!(true));
        } else {
            all_results.push(result);
        }
    }
}

fn apply_source_role_preference(
    all_results: &mut [crate::engine::RecallResult],
    query_intent: Option<&crate::facets::QueryIntent>,
) {
    let Some(intent) = query_intent else {
        return;
    };

    let target_role = if intent.labels.iter().any(|label| label == "source_assistant") {
        Some("assistant")
    } else if intent.labels.iter().any(|label| label == "source_user") {
        Some("user")
    } else {
        None
    };
    let Some(target_role) = target_role else {
        return;
    };

    let target_exists = all_results
        .iter()
        .any(|result| source_role_from_metadata(&result.metadata).as_deref() == Some(target_role));
    if !target_exists {
        return;
    }

    for result in all_results {
        let Some(role) = source_role_from_metadata(&result.metadata) else {
            continue;
        };
        if role != target_role {
            result.score *= 0.35;
            result.metadata.insert(
                "source_role_mismatch_penalty".to_string(),
                serde_json::json!(target_role),
            );
        }
    }
}

fn apply_decision_adjacency_preference(
    all_results: &mut [crate::engine::RecallResult],
    query_intent: Option<&crate::facets::QueryIntent>,
) {
    if !decision_projection_requested(query_intent) {
        return;
    }

    let Some((pivot_idx, pivot_session, pivot_score)) = all_results
        .iter()
        .enumerate()
        .filter_map(|(idx, result)| {
            if crate::facets::has_decision_selection_language(&result.content) {
                return None;
            }
            let session = source_session_cue_from_metadata(&result.metadata)?;
            Some((idx, session, result.score))
        })
        .max_by(|(_, _, left), (_, _, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
    else {
        return;
    };

    let mut session_positions = all_results
        .iter()
        .enumerate()
        .filter_map(|(idx, result)| {
            if source_session_cue_from_metadata(&result.metadata).as_deref()
                == Some(pivot_session.as_str())
            {
                Some((idx, result.created_at))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    session_positions.sort_by(|(_, left), (_, right)| {
        left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
    });

    let Some(pivot_position) = session_positions
        .iter()
        .position(|(idx, _)| *idx == pivot_idx)
    else {
        return;
    };

    for (position, (idx, _)) in session_positions.iter().enumerate().skip(pivot_position + 1) {
        let distance = position - pivot_position;
        if distance > 8 {
            break;
        }
        if !crate::facets::has_decision_selection_language(&all_results[*idx].content) {
            continue;
        }
        let bonus = pivot_score / (distance * distance) as f64;
        all_results[*idx].score += bonus;
        all_results[*idx].metadata.insert(
            "decision_adjacency_boost".to_string(),
            serde_json::json!({
                "pivot_session": pivot_session,
                "distance": distance,
                "bonus": bonus
            }),
        );
    }
}

fn apply_source_answer_adjacency_preference(
    all_results: &mut [crate::engine::RecallResult],
    query_intent: Option<&crate::facets::QueryIntent>,
) {
    let Some(intent) = query_intent else {
        return;
    };
    if !intent.labels.iter().any(|label| label == "source_answer")
        || !intent.labels.iter().any(|label| label == "source_assistant")
    {
        return;
    }

    let Some((pivot_idx, pivot_session)) = all_results
        .iter()
        .enumerate()
        .filter_map(|(idx, result)| {
            let role = source_role_from_metadata(&result.metadata)?;
            if role == "assistant" {
                return None;
            }
            let session = source_session_cue_from_metadata(&result.metadata)?;
            Some((idx, session, result.score))
        })
        .max_by(|(_, _, left), (_, _, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, session, _)| (idx, session))
    else {
        return;
    };

    let mut session_positions = all_results
        .iter()
        .enumerate()
        .filter_map(|(idx, result)| {
            if source_session_cue_from_metadata(&result.metadata).as_deref()
                == Some(pivot_session.as_str())
            {
                Some((idx, result.created_at))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    session_positions.sort_by(|(_, left), (_, right)| {
        left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
    });

    let Some(pivot_position) = session_positions
        .iter()
        .position(|(idx, _)| *idx == pivot_idx)
    else {
        return;
    };

    for (position, (idx, _)) in session_positions.iter().enumerate().skip(pivot_position + 1) {
        let distance = position - pivot_position;
        if distance > 8 {
            break;
        }
        if source_role_from_metadata(&all_results[*idx].metadata).as_deref() != Some("assistant") {
            continue;
        }
        let bonus = 2000.0 / (distance * distance) as f64;
        all_results[*idx].score += bonus;
        all_results[*idx].metadata.insert(
            "source_answer_adjacency_boost".to_string(),
            serde_json::json!({
                "pivot_session": pivot_session,
                "distance": distance,
                "bonus": bonus
            }),
        );
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RecallGroundedRequest {
    pub query_text: String,
    #[serde(default = "default_token_budget")]
    pub token_budget: u32,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub projects: Option<Vec<String>>,
    #[serde(default = "default_true")]
    pub auto_reinforce: bool, // Default to true - memories should learn from usage
    #[serde(default)]
    pub disable_salience_bias: bool,
    #[serde(default)]
    pub min_intersection: Option<usize>,
    #[serde(default = "default_true")]
    pub disable_alias_expansion: bool,
    #[serde(default = "default_expansion_depth")]
    pub expansion_depth: usize,
    #[serde(default)]
    pub cuepacks: Option<Vec<String>>,
}

fn default_true() -> bool {
    true
}

fn default_token_budget() -> u32 {
    2048
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RecallGroundedResponse {
    pub verified_context: String,
    pub proof: crate::grounding::GroundingProof,
    pub engine_latency_ms: f64,
    pub signature_alg: String,
    pub signature: String,
    pub public_key: Option<String>,
}

fn default_auto_reinforce() -> bool {
    false
}

fn default_expansion_depth() -> usize {
    1
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReinforceRequest {
    pub cues: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AddAliasRequest {
    pub from: String,
    pub to: String,
    pub weight: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GetAliasRequest {
    pub cue: String,
}

#[derive(Debug, Deserialize)]
pub struct MergeAliasRequest {
    pub cues: Vec<String>,
    pub to: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AliasResponse {
    pub id: MemoryId,
    pub from: String,
    pub to: String,
    pub weight: f64,
}

/// Response for /lexicon/inspect/:cue endpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct LexiconInspectResponse {
    pub cue: String,
    pub outgoing: Vec<LexiconEntry>, // What this token maps to
    pub incoming: Vec<LexiconEntry>, // Other tokens that map to the same canonical
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LexiconEntry {
    pub memory_id: MemoryId,
    pub content: String, // The canonical cue
    pub token: String,   // The raw token (from cues)
    pub reinforcement_score: f64,
    pub created_at: f64,
    #[serde(default)]
    pub affected_memories_count: usize, // Main memories that have this token but not the canonical
}

/// Request for POST /lexicon/wire - manually wire a token to a canonical cue
#[derive(Debug, Deserialize, Serialize)]
pub struct WireLexiconRequest {
    pub token: String,
    pub canonical: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct IngestUrlRequest {
    pub url: String,
    /// Crawl depth: 0 = single page (default), 1+ = follow links recursively
    #[serde(default)]
    pub depth: u8,
    /// Only follow links within the same domain (default: true)
    #[serde(default = "default_true")]
    pub same_domain_only: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CreateProjectRequest {
    pub project_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SetWatchDirRequest {
    pub watch_dir: String,
    #[serde(default)]
    pub included_paths: Option<Vec<String>>,
    #[serde(default)]
    pub ignored_patterns: Option<Vec<String>>,
    #[serde(default)]
    pub ignored_extensions: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PreviewDirectoryRequest {
    pub watch_dir: String,
    #[serde(default)]
    pub included_paths: Option<Vec<String>>,
    #[serde(default)]
    pub ignored_patterns: Option<Vec<String>>,
    #[serde(default)]
    pub ignored_extensions: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RecallWebRequest {
    pub url: Option<String>,
    pub query: String,
    #[serde(default)]
    pub persist: bool,
}

#[derive(Debug, Serialize)]
pub struct ReinforceResponse {
    status: String,
    memory_id: MemoryId,
}

#[derive(Clone)]
pub struct EngineState {
    pub mt_engine: Arc<MultiTenantEngine>,
    pub read_only: bool,
    pub job_queue: Arc<JobQueue>,
    pub metrics: Arc<MetricsCollector>,
    pub data_dir: String,
    pub cloud_backup: Option<Arc<CloudBackupManager>>,
    pub context_signer: Option<Arc<crate::crypto::ContextSigner>>,
    pub agent_manager: Arc<crate::agent::manager::AgentManager>,
    pub cuepack_registry: Arc<crate::cuepacks::CuePackRegistry>,
}

struct StoredMemoryOutcome {
    memory_id: MemoryId,
    accepted_cues: Vec<String>,
    rejected_cues: serde_json::Value,
    latency_ms: f64,
    minimal_response: bool,
    timing: Option<serde_json::Map<String, serde_json::Value>>,
}

/// API Routes
pub fn routes(
    mt_engine: Arc<MultiTenantEngine>,
    job_queue: Arc<JobQueue>,
    metrics: Arc<MetricsCollector>,
    auth_config: AuthConfig,
    read_only: bool,
    data_dir: String,
    cloud_backup: Option<Arc<CloudBackupManager>>,
    context_signer: Option<Arc<crate::crypto::ContextSigner>>,
    agent_manager: Arc<crate::agent::manager::AgentManager>,
    cuepack_registry: Arc<crate::cuepacks::CuePackRegistry>,
) -> Router {
    let mut router = Router::new()
        .route("/", get(root))
        .route("/memories", post(add_memory))
        .route("/memories/batch", post(add_memories_batch))
        .route("/recall", post(recall))
        .route("/recall/web", post(recall_web))
        .route("/memories/:id/reinforce", patch(reinforce_memory))
        .route("/memories/:id", get(get_memory).delete(delete_memory))
        .route("/stats", get(get_stats))
        .route("/projects", get(list_projects).post(create_project))
        .route("/recall/grounded", post(recall_grounded))
        .route("/projects/:id", delete(delete_project))
        .route(
            "/projects/:id/artifacts",
            get(project_artifacts).post(reload_project_artifacts),
        )
        .route("/projects/:id/export", get(export_project))
        .route(
            "/projects/:id/watch-dir",
            get(get_project_watch_dir).post(set_project_watch_dir),
        )
        .route("/aliases", post(add_alias).get(get_aliases))
        .route("/aliases/merge", post(merge_aliases))
        .route("/lexicon/inspect/:cue", get(lexicon_inspect))
        .route("/lexicon/entry/:id", delete(lexicon_delete))
        .route("/lexicon/graph", get(lexicon_graph))
        .route("/lexicon/wire", post(lexicon_wire))
        .route("/ingest/url", post(ingest_url))
        .route("/ingest/content", post(ingest_content))
        .route("/ingest/file", post(ingest_file))
        .route("/ingest/directory/preview", post(preview_directory))
        .route("/jobs/status", get(jobs_status))
        .route("/debug/analyze-text", post(debug_analyze_text))
        .route("/metrics", get(prometheus_metrics))
        // Cloud backup endpoints
        .route("/backup/upload", post(backup_upload))
        .route("/backup/download", post(backup_download))
        .route("/backup/list", get(backup_list))
        .route("/backup/:project_id", delete(backup_delete))
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(EngineState {
            mt_engine,
            read_only,
            job_queue,
            metrics,
            data_dir,
            cloud_backup,
            context_signer,
            agent_manager,
            cuepack_registry,
        });

    // Add auth middleware if enabled
    if auth_config.is_enabled() {
        router = router.layer(middleware::from_fn_with_state(
            auth_config,
            crate::auth::auth_middleware,
        ));
    }

    router
}

async fn root() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "CueMap Rust Engine",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "High-performance Temporal-Associative Memory Store",
        "capabilities": ["repository_ingestion_scope_v1"]
    }))
}

// Handlers
fn extract_project_id(
    headers: &HeaderMap,
) -> Result<String, (StatusCode, Json<serde_json::Value>)> {
    let project_id = headers
        .get("X-Project-ID")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing X-Project-ID header"})),
            )
        })?;

    if !validate_project_id(project_id) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid project ID format"})),
        ));
    }

    Ok(project_id.to_string())
}

fn extract_project_id_optional(headers: &HeaderMap) -> Option<String> {
    headers
        .get("X-Project-ID")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .filter(|s| validate_project_id(s))
}

fn source_event_time(
    explicit: Option<f64>,
    metadata: Option<&HashMap<String, serde_json::Value>>,
) -> Option<f64> {
    if explicit.is_some() {
        return explicit;
    }

    let value = metadata?.get("source_timestamp")?;
    if let Some(timestamp) = value.as_f64() {
        return (timestamp.is_finite() && timestamp >= 0.0).then_some(timestamp);
    }

    let parsed = chrono::DateTime::parse_from_rfc3339(value.as_str()?).ok()?;
    Some(
        parsed.timestamp() as f64
            + f64::from(parsed.timestamp_subsec_nanos()) / 1_000_000_000.0,
    )
}

async fn store_memory_request(
    state: &EngineState,
    project_id: &str,
    req: AddMemoryRequest,
) -> Result<StoredMemoryOutcome, (StatusCode, serde_json::Value)> {
    use std::time::Instant;

    let start = Instant::now();
    let trace_timing = req.trace_timing;
    let minimal_response = req.minimal_response;
    let mut timing = serde_json::Map::new();
    let mut phase_start = Instant::now();

    if state.read_only {
        return Err((
            StatusCode::FORBIDDEN,
            serde_json::json!({
                "error": "Read-only mode: modifications are not allowed"
            }),
        ));
    }

    let ctx = match state.mt_engine.get_or_create_project(project_id.to_string()) {
        Ok(c) => c,
        Err(e) => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                serde_json::json!({"error": e}),
            ))
        }
    };
    if trace_timing {
        timing.insert(
            "project_lookup_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
    }

    let AddMemoryRequest {
        content,
        cues,
        source_key,
        event_time,
        metadata,
        cuepacks,
        disable_temporal_chunking,
        async_ingest: _,
        minimal_response: _,
        trace_timing: _,
    } = req;

    if event_time.is_some_and(|timestamp| !timestamp.is_finite() || timestamp < 0.0) {
        return Err((
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "error": "event_time must be a finite, non-negative Unix timestamp in seconds"
            }),
        ));
    }
    let event_time = source_event_time(event_time, metadata.as_ref());

    phase_start = Instant::now();
    let mut initial_cues = cues;
    if initial_cues.is_empty() {
        let tokens = crate::nl::tokenize_to_cues(&content);
        initial_cues.extend(tokens);
    }
    if trace_timing {
        timing.insert(
            "tokenize_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
    }

    phase_start = Instant::now();
    let mut normalized_cues = Vec::with_capacity(initial_cues.len());
    for cue in initial_cues {
        let (normalized, _) = normalize_cue(&cue, &ctx.normalization);
        normalized_cues.push(normalized);
    }
    if trace_timing {
        timing.insert(
            "normalize_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
    }

    phase_start = Instant::now();
    let report = validate_cues(normalized_cues, &ctx.taxonomy);
    if trace_timing {
        timing.insert(
            "validate_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
    }

    let accepted_for_response = if minimal_response {
        Vec::new()
    } else {
        report.accepted.clone()
    };
    let rejected_for_response = if minimal_response {
        serde_json::json!([])
    } else {
        serde_json::json!(report.rejected)
    };

    phase_start = Instant::now();
    let cuepack_selection = cuepacks.as_deref();
    let memory_id = if let Some(source_key) = source_key {
        ctx.main.upsert_memory_with_source_key_and_options(
            source_key,
            content,
            report.accepted,
            metadata,
            None,
            false,
            true,
            disable_temporal_chunking,
            &state.cuepack_registry,
            cuepack_selection,
            event_time,
        )
    } else {
        ctx.main.add_memory_with_cuepacks_and_event_time(
            content,
            report.accepted,
            metadata,
            MainStats::default(),
            disable_temporal_chunking,
            &state.cuepack_registry,
            cuepack_selection,
            event_time,
        )
    };
    if trace_timing {
        timing.insert(
            "engine_add_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
    }

    phase_start = Instant::now();
    state.metrics.record_ingestion();
    if trace_timing {
        timing.insert(
            "metrics_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    if trace_timing {
        timing.insert("total_ms".to_string(), serde_json::json!(latency_ms));
    }

    Ok(StoredMemoryOutcome {
        memory_id,
        accepted_cues: accepted_for_response,
        rejected_cues: rejected_for_response,
        latency_ms,
        minimal_response,
        timing: if trace_timing { Some(timing) } else { None },
    })
}

async fn add_memory(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<AddMemoryRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    match store_memory_request(&state, &project_id, req).await {
        Ok(outcome) => {
            let mut body = if outcome.minimal_response {
                serde_json::json!({
                    "id": outcome.memory_id,
                    "status": "stored",
                    "latency_ms": outcome.latency_ms
                })
            } else {
                serde_json::json!({
                    "id": outcome.memory_id,
                    "status": "stored",
                    "cues": outcome.accepted_cues,
                    "rejected_cues": outcome.rejected_cues,
                    "latency_ms": outcome.latency_ms
                })
            };
            if let Some(timing) = outcome.timing {
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("timing".to_string(), serde_json::Value::Object(timing));
                }
            }
            (StatusCode::OK, Json(body))
        }
        Err((status, body)) => (status, Json(body)),
    }
}

async fn add_memories_batch(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<AddMemoryBatchRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let start = std::time::Instant::now();
    let count = req.memories.len();
    let mut ids = Vec::with_capacity(count);
    let mut per_memory_timings = if req.trace_timing {
        Some(Vec::with_capacity(count))
    } else {
        None
    };

    for (idx, mut memory_req) in req.memories.into_iter().enumerate() {
        if req.minimal_response {
            memory_req.minimal_response = true;
        }
        if req.trace_timing {
            memory_req.trace_timing = true;
        }

        match store_memory_request(&state, &project_id, memory_req).await {
            Ok(outcome) => {
                ids.push(outcome.memory_id);
                if let Some(timings) = per_memory_timings.as_mut() {
                    timings.push(serde_json::json!({
                        "id": outcome.memory_id,
                        "latency_ms": outcome.latency_ms,
                        "timing": outcome.timing.unwrap_or_default()
                    }));
                }
            }
            Err((status, body)) => {
                return (
                    status,
                    Json(serde_json::json!({
                        "error": "batch write failed",
                        "failed_index": idx,
                        "detail": body
                    })),
                );
            }
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let per_memory_latency_ms = if count == 0 {
        0.0
    } else {
        elapsed_ms / count as f64
    };

    let mut body = serde_json::json!({
        "status": "stored",
        "count": count,
        "ids": ids,
        "latency_ms": elapsed_ms,
        "per_memory_latency_ms": per_memory_latency_ms
    });

    if let Some(timings) = per_memory_timings {
        if let Some(obj) = body.as_object_mut() {
            obj.insert("timings".to_string(), serde_json::Value::Array(timings));
        }
    }

    (StatusCode::OK, Json(body))
}

#[axum::debug_handler]
async fn recall(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<RecallRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use std::time::Instant;
    let start = Instant::now();
    let EngineState {
        ref mt_engine,
        ref job_queue,
        ref cuepack_registry,
        ..
    } = &state;

    // --- Path 1: Cross-domain query ---
    if let Some(projects) = req.projects {
        let start = Instant::now();

        // Query all projects in parallel using rayon
        let (all_results, reinforce_tasks): (Vec<serde_json::Value>, Vec<Option<(String, Vec<MemoryId>, Vec<String>)>>) = projects
            .par_iter()
            .map(|project_id| {
                let ctx = match mt_engine.get_or_create_project(project_id.clone()) {
                    Ok(c) => c,
                    Err(_) => return (serde_json::json!({"project_id": project_id, "error": "Capacity reached"}), None),
                };

                // Collect cues
                let mut cues_to_process = req.cues.clone();

                // Extract mandatory constraints from explicit cues
                let mandatory_cues: Vec<String> = req.cues.iter()
                    .map(|c| normalize_cue(c, &ctx.normalization).0)
                    .collect();
                let mandatory_cues_ref = if mandatory_cues.is_empty() { None } else { Some(&mandatory_cues) };

                let (original_tokens, _lexicon_mids) = if let Some(text) = &req.query_text {
                     let (resolved, lex_mids, tokens) = ctx.resolve_cues_from_text(text, false);
                     cues_to_process.extend(resolved);
                     (tokens, lex_mids)
                } else {
                    (req.cues.clone(), Vec::new())
                };

                // Normalize query cues
                let mut normalized_cues = Vec::new();
                for cue in &cues_to_process {
                    let (normalized, _) = normalize_cue(cue, &ctx.normalization);
                    normalized_cues.push(normalized);
                }

                // Expand aliases
                let mut expanded_cues = if req.disable_alias_expansion {
                    normalized_cues.into_iter().map(|c| (c, 1.0)).collect()
                } else {
                    ctx.expand_query_cues(normalized_cues, &original_tokens)
                };
                let query_intent = apply_query_intent(
                    &ctx,
                    &state.cuepack_registry,
                    req.cuepacks.as_deref(),
                    req.query_text.as_deref(),
                    req.query_time.as_deref(),
                    &mut expanded_cues,
                );
                let mut all_results: Vec<crate::engine::RecallResult> = Vec::new();
                let mut used_pivot_memory_ids = std::collections::HashSet::new();
                let limit = req.limit.max(1);
                let depth = req.depth.max(1);

                for hop in 1..=depth {
                    let current_limit = (limit as f64 / hop as f64).ceil() as usize;

                    let mut results = {
                        let heatmap = ctx.market_heatmap.read().ok();
                        let heatmap_ref = heatmap.as_deref();

                        ctx.main.recall_weighted(
                            expanded_cues.clone(),
                            current_limit,
                            false,
                            req.min_intersection,
                            req.expansion_depth,
                            req.explain,
                            req.disable_salience_bias,
                            heatmap_ref,
                            mandatory_cues_ref
                        )
                    };

                    // Add hop metadata
                    for r in &mut results {
                        if !r.metadata.contains_key("hop") {
                            r.metadata.insert("hop".to_string(), serde_json::json!(hop));
                        }
                    }

                    // Merge results, avoiding duplicates
                    for r in results {
                        if !all_results.iter().any(|existing| existing.memory_id == r.memory_id) {
                            all_results.push(r);
                        }
                    }

                    if hop < depth {
                        let mut pivot_memory = None;
                        for r in &all_results {
                            if !used_pivot_memory_ids.contains(&r.memory_id) {
                                pivot_memory = ctx.main.get_memory(r.memory_id);
                                if pivot_memory.is_some() {
                                    used_pivot_memory_ids.insert(r.memory_id.clone());
                                    break;
                                }
                            }
                        }

                        if let Some(mem) = pivot_memory {
                            let existing_cues: std::collections::HashSet<String> = expanded_cues.iter().map(|(c, _)| c.clone()).collect();
                            for cue in mem.cues {
                                if !existing_cues.contains(&cue) {
                                    expanded_cues.push((cue, 0.5f64.powi(hop as i32)));
                                }
                            }
                        } else {
                            break;
                        }
                    }
                }

                let source_answer_projection_expansions =
                    source_answer_projection_cues(
                        &ctx,
                        query_intent.as_ref(),
                        req.query_text.as_deref(),
                        &all_results,
                    );
                if !source_answer_projection_expansions.is_empty() {
                    let heatmap = ctx.market_heatmap.read().ok();
                    let heatmap_ref = heatmap.as_deref();
                    let projection_results = ctx.main.recall_weighted(
                        source_answer_projection_expansions.clone(),
                        limit.max(5).min(20),
                        false,
                        req.min_intersection,
                        1,
                        req.explain,
                        req.disable_salience_bias,
                        heatmap_ref,
                        mandatory_cues_ref
                    );
                    merge_source_answer_projection_results(&mut all_results, projection_results);
                }

                let source_prompt_projection_expansions =
                    source_prompt_projection_cues(
                        &ctx,
                        query_intent.as_ref(),
                        req.query_text.as_deref(),
                        &all_results,
                    );
                if !source_prompt_projection_expansions.is_empty() {
                    let heatmap = ctx.market_heatmap.read().ok();
                    let heatmap_ref = heatmap.as_deref();
                    let projection_results = ctx.main.recall_weighted(
                        source_prompt_projection_expansions.clone(),
                        limit.max(5).min(20),
                        false,
                        req.min_intersection,
                        1,
                        req.explain,
                        req.disable_salience_bias,
                        heatmap_ref,
                        mandatory_cues_ref
                    );
                    merge_source_prompt_projection_results(
                        &mut all_results,
                        projection_results,
                        req.query_text.as_deref(),
                    );
                }

                let user_context_projection_expansions =
                    user_context_projection_cues(
                        &ctx,
                        query_intent.as_ref(),
                        req.query_text.as_deref(),
                        &all_results,
                    );
                if !user_context_projection_expansions.is_empty() {
                    let heatmap = ctx.market_heatmap.read().ok();
                    let heatmap_ref = heatmap.as_deref();
                    let projection_results = ctx.main.recall_weighted(
                        user_context_projection_expansions.clone(),
                        limit.max(5).min(20),
                        false,
                        req.min_intersection,
                        1,
                        req.explain,
                        req.disable_salience_bias,
                        heatmap_ref,
                        mandatory_cues_ref
                    );
                    merge_user_context_projection_results(&mut all_results, projection_results);
                }

                let standing_instruction_projection = standing_instruction_projection_cues(
                    &ctx,
                    query_intent.as_ref(),
                    req.query_text.as_deref(),
                );
                if !standing_instruction_projection.cues.is_empty() {
                    let heatmap = ctx.market_heatmap.read().ok();
                    let heatmap_ref = heatmap.as_deref();
                    let projection_results = ctx.main.recall_weighted(
                        standing_instruction_projection.cues.clone(),
                        limit.max(5).min(30),
                        false,
                        req.min_intersection,
                        1,
                        req.explain,
                        req.disable_salience_bias,
                        heatmap_ref,
                        mandatory_cues_ref
                    );
                    merge_standing_instruction_projection_results(
                        &ctx,
                        &mut all_results,
                        projection_results,
                        &standing_instruction_projection.anchors,
                    );
                }

                let preference_projection = preference_projection_cues(
                    &ctx,
                    query_intent.as_ref(),
                    req.query_text.as_deref(),
                );
                if !preference_projection.cues.is_empty() {
                    let heatmap = ctx.market_heatmap.read().ok();
                    let heatmap_ref = heatmap.as_deref();
                    let projection_results = ctx.main.recall_weighted(
                        preference_projection.cues.clone(),
                        limit.max(5).min(30),
                        false,
                        req.min_intersection,
                        1,
                        req.explain,
                        req.disable_salience_bias,
                        heatmap_ref,
                        mandatory_cues_ref
                    );
                    merge_preference_projection_results(
                        &ctx,
                        &mut all_results,
                        projection_results,
                        &preference_projection.anchors,
                    );
                }

                let decision_projection_expansions =
                    decision_projection_cues(&ctx, query_intent.as_ref(), &all_results);
                if !decision_projection_expansions.is_empty() {
                    let heatmap = ctx.market_heatmap.read().ok();
                    let heatmap_ref = heatmap.as_deref();
                    let projection_results = ctx.main.recall_weighted(
                        decision_projection_expansions.clone(),
                        limit.max(5).min(20),
                        false,
                        req.min_intersection,
                        1,
                        req.explain,
                        req.disable_salience_bias,
                        heatmap_ref,
                        mandatory_cues_ref
                    );
                    merge_decision_projection_results(&mut all_results, projection_results);
                }

                apply_source_role_preference(&mut all_results, query_intent.as_ref());
                apply_source_answer_adjacency_preference(&mut all_results, query_intent.as_ref());
                apply_user_context_adjacency_preference(
                    &mut all_results,
                    query_intent.as_ref(),
                    req.query_text.as_deref(),
                );
                apply_decision_adjacency_preference(&mut all_results, query_intent.as_ref());
                all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                let results = all_results;

                let json_results: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| serde_json::json!({
                        "id": r.memory_id,
                        "content": r.content,
                        "score": r.score,
                        "intersection_count": r.intersection_count,
                        "recency_score": r.recency_score,
                        "metadata": r.metadata,
                        "explain": r.explain
                    }))
                    .collect();

                let mut response_block = serde_json::json!({
                    "project_id": project_id,
                    "results": json_results
                });

                if req.explain {
                    response_block.as_object_mut().unwrap().insert(
                        "explain".to_string(),
                        serde_json::json!({
                            "query_cues": cues_to_process,
                            "expanded_cues": expanded_cues,
                            "query_intent": query_intent,
                            "cuepacks": req.cuepacks.clone().unwrap_or_else(|| vec!["default".to_string()]),
                            "source_answer_projection_cues": source_answer_projection_expansions,
                            "source_prompt_projection_cues": source_prompt_projection_expansions,
                            "user_context_projection_cues": user_context_projection_expansions,
                            "standing_instruction_projection_cues": standing_instruction_projection.cues,
                            "preference_projection_cues": preference_projection.cues,
                            "decision_projection_cues": decision_projection_expansions
                        })
                    );
                }

                // Collect reinforcement task
                let task = if req.auto_reinforce && !results.is_empty() {
                     let memory_ids: Vec<MemoryId> = results.iter().map(|r| r.memory_id).collect();
                     let cues: Vec<String> = expanded_cues.iter().map(|(c, _)| c.clone()).collect();
                     Some((project_id.clone(), memory_ids, cues))
                } else {
                    None
                };

                (response_block, task)
            })
            .unzip();

        // Enqueue reinforcement tasks
        for task in reinforce_tasks {
            if let Some((pid, mids, cues)) = task {
                job_queue
                    .enqueue(crate::jobs::Job::ReinforceMemories {
                        project_id: pid,
                        memory_ids: mids,
                        cues,
                    })
                    .await;
            }
        }

        let elapsed = start.elapsed();
        let engine_latency_ms = elapsed.as_secs_f64() * 1000.0;
        state.metrics.record_recall(engine_latency_ms);

        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "results": all_results,
                "engine_latency": engine_latency_ms
            })),
        );
    }

    // --- Path 2: Single project query ---
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let trace_timing = req.trace_timing;
    let mut timing = serde_json::Map::new();
    let mut phase_start = Instant::now();
    let ctx = match mt_engine.get_or_create_project(project_id.clone()) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };
    if trace_timing {
        timing.insert(
            "project_lookup_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
    }

    // Collect cues
    phase_start = Instant::now();
    let mut cues_to_process = req.cues.clone();

    // Extract mandatory constraints from explicit cues
    let mandatory_cues: Vec<String> = req
        .cues
        .iter()
        .map(|c| normalize_cue(c, &ctx.normalization).0)
        .collect();
    let mandatory_cues_ref = if mandatory_cues.is_empty() {
        None
    } else {
        Some(&mandatory_cues)
    };

    let mut lexicon_memory_ids: Vec<MemoryId> = Vec::new();
    let mut tokens_from_text = Vec::new();
    if let Some(ref text) = req.query_text {
        // 1. Lexicon Recall
        let (resolved, lex_mids, tokens) = ctx.resolve_cues_from_text(text, false);
        cues_to_process.extend(resolved);
        lexicon_memory_ids = lex_mids;

        // 2. Raw Token Fallback
        tokens_from_text = tokens;
        for token in &tokens_from_text {
            if !cues_to_process.contains(token) {
                cues_to_process.push(token.clone());
            }
        }
    }
    if trace_timing {
        timing.insert(
            "query_resolution_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "resolved_cue_count".to_string(),
            serde_json::json!(cues_to_process.len()),
        );
        timing.insert(
            "raw_token_count".to_string(),
            serde_json::json!(tokens_from_text.len()),
        );
    }

    // Normalize query cues
    phase_start = Instant::now();
    let mut normalized_cues = Vec::new();
    for cue in &cues_to_process {
        let (normalized, _) = normalize_cue(cue, &ctx.normalization);
        normalized_cues.push(normalized);
    }
    if trace_timing {
        timing.insert(
            "normalize_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
    }

    // Expand aliases
    phase_start = Instant::now();
    let original_tokens = if req.query_text.is_some() {
        tokens_from_text.clone()
    } else {
        req.cues.clone()
    };
    let (mut expanded_cues, cuebridge_alias_expansions) = if req.disable_alias_expansion {
        (
            normalized_cues.into_iter().map(|c| (c, 1.0)).collect(),
            Vec::new(),
        )
    } else {
        ctx.expand_query_cues_with_trace(normalized_cues, &original_tokens)
    };
    if trace_timing {
        timing.insert(
            "alias_expansion_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "expanded_cue_count_before_intent".to_string(),
            serde_json::json!(expanded_cues.len()),
        );
    }
    phase_start = Instant::now();
    let query_intent = apply_query_intent(
        &ctx,
        cuepack_registry,
        req.cuepacks.as_deref(),
        req.query_text.as_deref(),
        req.query_time.as_deref(),
        &mut expanded_cues,
    );
    if trace_timing {
        timing.insert(
            "query_intent_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "expanded_cue_count_after_intent".to_string(),
            serde_json::json!(expanded_cues.len()),
        );
    }
    let mut all_results: Vec<crate::engine::RecallResult> = Vec::new();
    let mut used_pivot_memory_ids = std::collections::HashSet::new();
    let limit = req.limit.max(1);
    let depth = req.depth.max(1);
    let mut base_recall_timing: Option<crate::engine::RecallTimingBreakdown> = None;
    let mut base_recall_total_ms = 0.0;
    let mut base_recall_calls = 0usize;

    for hop in 1..=depth {
        let current_limit = (limit as f64 / hop as f64).ceil() as usize;

        let mut results = {
            let heatmap = ctx.market_heatmap.read().ok();
            let heatmap_ref = heatmap.as_deref();

            if trace_timing {
                let (results, call_timing) = ctx.main.recall_weighted_with_timing(
                    expanded_cues.clone(),
                    current_limit,
                    false,
                    req.min_intersection,
                    req.expansion_depth,
                    req.explain,
                            req.disable_salience_bias,
                    heatmap_ref,
                    mandatory_cues_ref,
                );
                base_recall_total_ms += call_timing.total_ms;
                base_recall_calls += 1;
                if base_recall_timing.is_none() {
                    base_recall_timing = Some(call_timing);
                }
                results
            } else {
                ctx.main.recall_weighted(
                    expanded_cues.clone(),
                    current_limit,
                    false,
                    req.min_intersection,
                    req.expansion_depth,
                    req.explain,
                            req.disable_salience_bias,
                    heatmap_ref,
                    mandatory_cues_ref,
                )
            }
        };

        // Add hop metadata
        for r in &mut results {
            if !r.metadata.contains_key("hop") {
                r.metadata.insert("hop".to_string(), serde_json::json!(hop));
            }
        }

        // Merge results, avoiding duplicates
        for r in results {
            if !all_results
                .iter()
                .any(|existing| existing.memory_id == r.memory_id)
            {
                all_results.push(r);
            }
        }

        if hop < depth {
            let mut pivot_memory = None;
            for r in &all_results {
                if !used_pivot_memory_ids.contains(&r.memory_id) {
                    pivot_memory = ctx.main.get_memory(r.memory_id);
                    if pivot_memory.is_some() {
                        used_pivot_memory_ids.insert(r.memory_id.clone());
                        break;
                    }
                }
            }

            if let Some(mem) = pivot_memory {
                let existing_cues: std::collections::HashSet<String> =
                    expanded_cues.iter().map(|(c, _)| c.clone()).collect();
                for cue in mem.cues {
                    if !existing_cues.contains(&cue) {
                        expanded_cues.push((cue, 0.5f64.powi(hop as i32)));
                    }
                }
            } else {
                break;
            }
        }
    }

    phase_start = Instant::now();
    let source_answer_projection_expansions = source_answer_projection_cues(
        &ctx,
        query_intent.as_ref(),
        req.query_text.as_deref(),
        &all_results,
    );
    if !source_answer_projection_expansions.is_empty() {
        let heatmap = ctx.market_heatmap.read().ok();
        let heatmap_ref = heatmap.as_deref();
        let projection_results = ctx.main.recall_weighted(
            source_answer_projection_expansions.clone(),
            limit.max(5).min(20),
            false,
            req.min_intersection,
            1,
            req.explain,
                        req.disable_salience_bias,
            heatmap_ref,
            mandatory_cues_ref,
        );
        merge_source_answer_projection_results(&mut all_results, projection_results);
    }
    if trace_timing {
        timing.insert(
            "source_answer_projection_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "source_answer_projection_cue_count".to_string(),
            serde_json::json!(source_answer_projection_expansions.len()),
        );
    }

    phase_start = Instant::now();
    let source_prompt_projection_expansions = source_prompt_projection_cues(
        &ctx,
        query_intent.as_ref(),
        req.query_text.as_deref(),
        &all_results,
    );
    if !source_prompt_projection_expansions.is_empty() {
        let heatmap = ctx.market_heatmap.read().ok();
        let heatmap_ref = heatmap.as_deref();
        let projection_results = ctx.main.recall_weighted(
            source_prompt_projection_expansions.clone(),
            limit.max(5).min(20),
            false,
            req.min_intersection,
            1,
            req.explain,
                        req.disable_salience_bias,
            heatmap_ref,
            mandatory_cues_ref,
        );
        merge_source_prompt_projection_results(
            &mut all_results,
            projection_results,
            req.query_text.as_deref(),
        );
    }
    if trace_timing {
        timing.insert(
            "source_prompt_projection_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "source_prompt_projection_cue_count".to_string(),
            serde_json::json!(source_prompt_projection_expansions.len()),
        );
    }

    phase_start = Instant::now();
    let user_context_projection_expansions = user_context_projection_cues(
        &ctx,
        query_intent.as_ref(),
        req.query_text.as_deref(),
        &all_results,
    );
    if !user_context_projection_expansions.is_empty() {
        let heatmap = ctx.market_heatmap.read().ok();
        let heatmap_ref = heatmap.as_deref();
        let projection_results = ctx.main.recall_weighted(
            user_context_projection_expansions.clone(),
            limit.max(5).min(20),
            false,
            req.min_intersection,
            1,
            req.explain,
                        req.disable_salience_bias,
            heatmap_ref,
            mandatory_cues_ref,
        );
        merge_user_context_projection_results(&mut all_results, projection_results);
    }
    if trace_timing {
        timing.insert(
            "user_context_projection_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "user_context_projection_cue_count".to_string(),
            serde_json::json!(user_context_projection_expansions.len()),
        );
    }

    phase_start = Instant::now();
    let standing_instruction_projection = standing_instruction_projection_cues(
        &ctx,
        query_intent.as_ref(),
        req.query_text.as_deref(),
    );
    if !standing_instruction_projection.cues.is_empty() {
        let heatmap = ctx.market_heatmap.read().ok();
        let heatmap_ref = heatmap.as_deref();
        let projection_results = ctx.main.recall_weighted(
            standing_instruction_projection.cues.clone(),
            limit.max(5).min(30),
            false,
            req.min_intersection,
            1,
            req.explain,
                        req.disable_salience_bias,
            heatmap_ref,
            mandatory_cues_ref,
        );
        merge_standing_instruction_projection_results(
            &ctx,
            &mut all_results,
            projection_results,
            &standing_instruction_projection.anchors,
        );
    }
    if trace_timing {
        timing.insert(
            "standing_instruction_projection_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "standing_instruction_projection_cue_count".to_string(),
            serde_json::json!(standing_instruction_projection.cues.len()),
        );
    }

    phase_start = Instant::now();
    let preference_projection = preference_projection_cues(
        &ctx,
        query_intent.as_ref(),
        req.query_text.as_deref(),
    );
    if !preference_projection.cues.is_empty() {
        let heatmap = ctx.market_heatmap.read().ok();
        let heatmap_ref = heatmap.as_deref();
        let projection_results = ctx.main.recall_weighted(
            preference_projection.cues.clone(),
            limit.max(5).min(30),
            false,
            req.min_intersection,
            1,
            req.explain,
                        req.disable_salience_bias,
            heatmap_ref,
            mandatory_cues_ref,
        );
        merge_preference_projection_results(
            &ctx,
            &mut all_results,
            projection_results,
            &preference_projection.anchors,
        );
    }
    if trace_timing {
        timing.insert(
            "preference_projection_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "preference_projection_cue_count".to_string(),
            serde_json::json!(preference_projection.cues.len()),
        );
    }

    phase_start = Instant::now();
    let decision_projection_expansions =
        decision_projection_cues(&ctx, query_intent.as_ref(), &all_results);
    if !decision_projection_expansions.is_empty() {
        let heatmap = ctx.market_heatmap.read().ok();
        let heatmap_ref = heatmap.as_deref();
        let projection_results = ctx.main.recall_weighted(
            decision_projection_expansions.clone(),
            limit.max(5).min(20),
            false,
            req.min_intersection,
            1,
            req.explain,
                        req.disable_salience_bias,
            heatmap_ref,
            mandatory_cues_ref,
        );
        merge_decision_projection_results(&mut all_results, projection_results);
    }
    if trace_timing {
        timing.insert(
            "decision_projection_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "decision_projection_cue_count".to_string(),
            serde_json::json!(decision_projection_expansions.len()),
        );
    }

    phase_start = Instant::now();
    let mut cuebridge_gap_expansions = Vec::<crate::cuebridge::CueBridgeGapExpansion>::new();
    let mut cuebridge_gap_applied_count = 0usize;
    let cuebridge_artifact_summary = ctx.cuebridge_artifact_summary();
    if !req.disable_cuebridge_artifacts && cuebridge_artifact_summary.gap_entry_count > 0 {
        cuebridge_gap_expansions = ctx
            .cuebridge_artifacts
            .read()
            .map(|artifacts| {
                artifacts.gap_expansions(
                    &expanded_cues,
                    query_intent.as_ref(),
                    &original_tokens,
                    |cue| ctx.main.get_cue_frequency(cue) > 0,
                    req.cuebridge_gap_limit,
                )
            })
            .unwrap_or_default();

        if !cuebridge_gap_expansions.is_empty() {
            let existing_cues: HashSet<String> =
                expanded_cues.iter().map(|(cue, _)| cue.clone()).collect();
            for expansion in &cuebridge_gap_expansions {
                if !existing_cues.contains(&expansion.cue) {
                    expanded_cues.push((expansion.cue.clone(), expansion.score.max(0.01)));
                }
            }

            let heatmap = ctx.market_heatmap.read().ok();
            let heatmap_ref = heatmap.as_deref();
            let gap_results = ctx.main.recall_weighted(
                expanded_cues.clone(),
                limit,
                false,
                req.min_intersection,
                req.expansion_depth,
                req.explain,
                req.disable_salience_bias,
                heatmap_ref,
                mandatory_cues_ref,
            );
            cuebridge_gap_applied_count = gap_results.len();
            merge_cuebridge_gap_results(
                &mut all_results,
                gap_results,
                &cuebridge_gap_expansions,
                req.explain,
            );
        }
    }
    if trace_timing {
        timing.insert(
            "cuebridge_gap_pack_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "cuebridge_gap_pack_cue_count".to_string(),
            serde_json::json!(cuebridge_gap_expansions.len()),
        );
        timing.insert(
            "cuebridge_gap_pack_applied_count".to_string(),
            serde_json::json!(cuebridge_gap_applied_count),
        );
    }

    phase_start = Instant::now();
    let mut ordered_reconstruction_applied_count = 0usize;
    if should_run_ordered_reconstruction(req.ordered_reconstruction, query_intent.as_ref()) {
        let ordered_results = ordered_reconstruction_results(
            &ctx,
            &expanded_cues,
            &all_results,
            req.ordered_reconstruction_limit
                .max(limit)
                .min(MAX_ORDERED_RECONSTRUCTION_LIMIT),
            req.ordered_session_scan_limit,
            req.ordered_max_sessions,
            req.explain,
        );
        ordered_reconstruction_applied_count = ordered_results.len();
        merge_ordered_reconstruction_results(&mut all_results, ordered_results);
    }
    if trace_timing {
        timing.insert(
            "ordered_reconstruction_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "ordered_reconstruction_applied_count".to_string(),
            serde_json::json!(ordered_reconstruction_applied_count),
        );
    }

    phase_start = Instant::now();
    let mut evidence_coverage_applied_count = 0usize;
    if should_run_evidence_coverage(req.evidence_coverage, query_intent.as_ref()) {
        let evidence_results = evidence_coverage_results(
            &ctx,
            &expanded_cues,
            &all_results,
            req.evidence_coverage_limit
                .max(limit)
                .min(MAX_EVIDENCE_COVERAGE_LIMIT),
            req.evidence_coverage_session_scan_limit,
            req.evidence_coverage_max_sessions,
            req.explain,
        );
        evidence_coverage_applied_count = evidence_results.len();
        merge_evidence_coverage_results(&mut all_results, evidence_results);
    }
    if trace_timing {
        timing.insert(
            "evidence_coverage_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "evidence_coverage_applied_count".to_string(),
            serde_json::json!(evidence_coverage_applied_count),
        );
    }

    phase_start = Instant::now();
    let mut parent_fusion_applied_count = 0usize;
    if should_run_parent_fusion(
        &all_results,
        req.parent_fusion,
        query_intent.as_ref(),
        req.query_text.as_deref(),
    ) {
        let fusion_limit = req
            .parent_fusion_limit
            .max(limit)
            .min(MAX_PARENT_FUSION_LIMIT)
            .max(1);
        let heatmap = ctx.market_heatmap.read().ok();
        let heatmap_ref = heatmap.as_deref();
        let fusion_candidates = ctx.main.recall_weighted(
            expanded_cues.clone(),
            fusion_limit,
            false,
            req.min_intersection,
            1,
            req.explain,
                            req.disable_salience_bias,
            heatmap_ref,
            mandatory_cues_ref,
        );
        let fused_results = build_parent_fusion_results(
            &ctx,
            fusion_candidates,
            req.parent_fusion_min_chunks,
            req.explain,
        );
        parent_fusion_applied_count = fused_results.len();
        merge_parent_fusion_results(&mut all_results, fused_results);
    }
    if trace_timing {
        timing.insert(
            "parent_fusion_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "parent_fusion_applied_count".to_string(),
            serde_json::json!(parent_fusion_applied_count),
        );
    }

    phase_start = Instant::now();
    apply_source_role_preference(&mut all_results, query_intent.as_ref());
    apply_source_answer_adjacency_preference(&mut all_results, query_intent.as_ref());
    apply_user_context_adjacency_preference(
        &mut all_results,
        query_intent.as_ref(),
        req.query_text.as_deref(),
    );
    apply_decision_adjacency_preference(&mut all_results, query_intent.as_ref());
    let slate_rerank_applied_count = apply_slate_rerank(
        &ctx,
        &mut all_results,
        &expanded_cues,
        req.ordered_reconstruction,
        req.evidence_coverage,
        query_intent.as_ref(),
        limit,
    );
    all_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let results = all_results;
    if trace_timing {
        timing.insert(
            "postprocess_sort_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "merged_result_count".to_string(),
            serde_json::json!(results.len()),
        );
        timing.insert(
            "slate_rerank_applied_count".to_string(),
            serde_json::json!(slate_rerank_applied_count),
        );
        timing.insert(
            "base_recall_total_ms".to_string(),
            serde_json::json!(base_recall_total_ms),
        );
        timing.insert(
            "base_recall_calls".to_string(),
            serde_json::json!(base_recall_calls),
        );
        if let Some(base) = &base_recall_timing {
            timing.insert(
                "base_recall".to_string(),
                serde_json::json!(base),
            );
        }
    }

    let elapsed = start.elapsed();
    let engine_latency_ms = elapsed.as_secs_f64() * 1000.0;

    // Async reinforcement via background job (doesn't block response)
    phase_start = Instant::now();
    if req.auto_reinforce && !results.is_empty() {
        let memory_ids: Vec<MemoryId> = results.iter().map(|r| r.memory_id).collect();
        let cues: Vec<String> = expanded_cues.iter().map(|(c, _)| c.clone()).collect();
        // This await was causing the conflict because `heatmap` was seen as potentially live
        job_queue
            .enqueue(crate::jobs::Job::ReinforceMemories {
                project_id: project_id.clone(),
                memory_ids,
                cues,
            })
            .await;
    }

    // Reinforce Lexicon memories (async)
    if req.auto_reinforce && !lexicon_memory_ids.is_empty() {
        let tokens = if let Some(ref text) = req.query_text {
            crate::nl::tokenize_to_cues(text)
        } else {
            Vec::new()
        };
        job_queue
            .enqueue(crate::jobs::Job::ReinforceLexicon {
                project_id: project_id.clone(),
                memory_ids: lexicon_memory_ids,
                cues: tokens,
            })
            .await;
    }
    if trace_timing {
        timing.insert(
            "reinforcement_enqueue_ms".to_string(),
            serde_json::json!(phase_start.elapsed().as_secs_f64() * 1000.0),
        );
        timing.insert(
            "engine_latency_ms".to_string(),
            serde_json::json!(engine_latency_ms),
        );
    }

    // Record metrics
    state.metrics.record_recall(engine_latency_ms);

    if trace_timing {
        timing.insert(
            "total_before_response_ms".to_string(),
            serde_json::json!(start.elapsed().as_secs_f64() * 1000.0),
        );
    }

    if req.explain {
        let mut response = serde_json::json!({
            "results": results,
            "engine_latency": engine_latency_ms,
            "explain": {
                "query_cues": cues_to_process,
                "expanded_cues": expanded_cues,
                "query_intent": query_intent,
                "cuepacks": req.cuepacks.clone().unwrap_or_else(|| vec!["default".to_string()]),
                "source_answer_projection_cues": source_answer_projection_expansions,
                "source_prompt_projection_cues": source_prompt_projection_expansions,
                "user_context_projection_cues": user_context_projection_expansions,
                "standing_instruction_projection_cues": standing_instruction_projection.cues,
                "preference_projection_cues": preference_projection.cues,
                "decision_projection_cues": decision_projection_expansions,
                "cuebridge_artifacts": cuebridge_artifact_summary,
                "cuebridge_alias_pack": {
                    "applied_count": cuebridge_alias_expansions.len(),
                    "expansions": cuebridge_alias_expansions
                },
                "cuebridge_gap_pack": {
                    "disabled": req.disable_cuebridge_artifacts,
                    "gap_limit": req.cuebridge_gap_limit,
                    "applied_count": cuebridge_gap_applied_count,
                    "expansions": cuebridge_gap_expansions
                },
                "ordered_reconstruction": {
                    "mode": req.ordered_reconstruction,
                    "limit": req.ordered_reconstruction_limit,
                    "session_scan_limit": req.ordered_session_scan_limit,
                    "max_sessions": req.ordered_max_sessions,
                    "applied_count": ordered_reconstruction_applied_count
                },
                "evidence_coverage": {
                    "mode": req.evidence_coverage,
                    "limit": req.evidence_coverage_limit,
                    "session_scan_limit": req.evidence_coverage_session_scan_limit,
                    "max_sessions": req.evidence_coverage_max_sessions,
                    "applied_count": evidence_coverage_applied_count
                },
                "parent_fusion": {
                    "mode": req.parent_fusion,
                    "limit": req.parent_fusion_limit,
                    "min_chunks": req.parent_fusion_min_chunks,
                    "applied_count": parent_fusion_applied_count
                },
                "slate_rerank": {
                    "applied_count": slate_rerank_applied_count,
                    "pool_limit": SLATE_RERANK_POOL_LIMIT,
                    "target_limit": SLATE_RERANK_TARGET_LIMIT,
                    "protected_results": SLATE_RERANK_PROTECTED_RESULTS
                }
            }
        });
        if trace_timing {
            response
                .as_object_mut()
                .unwrap()
                .insert("timing".to_string(), serde_json::Value::Object(timing));
        }
        return (
            StatusCode::OK,
            Json(response),
        );
    }

    let mut response = serde_json::json!({
        "results": results,
        "engine_latency": engine_latency_ms
    });
    if trace_timing {
        response
            .as_object_mut()
            .unwrap()
            .insert("timing".to_string(), serde_json::Value::Object(timing));
    }
    (
        StatusCode::OK,
        Json(response),
    )
}

async fn reinforce_memory(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Path(memory_id): Path<MemoryId>,
    Json(req): Json<ReinforceRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState {
        ref mt_engine,
        ..
    } = &state;
    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };

    // Normalize cues
    let mut normalized_cues = Vec::new();

    if req.cues.is_empty() {
        if let Some(mem) = ctx.main.get_memories().get(&memory_id) {
            normalized_cues = mem.cues.clone();
        }
    } else {
        for cue in req.cues {
            let (normalized, _) = normalize_cue(&cue, &ctx.normalization);
            normalized_cues.push(normalized);
        }
    }

    let success = ctx.main.reinforce_memory(memory_id, normalized_cues);

    if success {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "reinforced",
                "memory_id": memory_id
            })),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "status": "not_found",
                "memory_id": memory_id
            })),
        )
    }
}

async fn get_memory(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Path(memory_id): Path<MemoryId>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState {
        ref mt_engine,
        ..
    } = &state;
    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };
    match ctx.main.get_memory(memory_id) {
        Some(memory) => (StatusCode::OK, Json(serde_json::json!(memory))),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Memory not found"})),
        ),
    }
}

/// GDPR-compliant delete (multi-tenant)
async fn delete_memory(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Path(memory_id): Path<MemoryId>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState {
        mt_engine,
        read_only,
        ..
    } = state;
    if read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "Read-only mode"})),
        );
    }

    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };
    let deleted = ctx.main.delete_memory(memory_id);
    if deleted {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "deleted",
                "memory_id": memory_id
            })),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "Memory not found",
                "memory_id": memory_id
            })),
        )
    }
}

async fn get_stats(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id_opt = extract_project_id_optional(&headers);
    let EngineState { mt_engine, .. } = state;

    if let Some(project_id) = project_id_opt {
        let ctx = match mt_engine.get_or_create_project(project_id) {
            Ok(c) => c,
            Err(e) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({"error": e})),
                )
            }
        };
        let stats = ctx.main.get_stats();
        (
            StatusCode::OK,
            Json(serde_json::Value::Object(stats.into_iter().collect())),
        )
    } else {
        // Global stats
        let stats = mt_engine.get_global_stats();
        (StatusCode::OK, Json(serde_json::json!(stats)))
    }
}

/// Get job/ingestion progress for a project or globally
async fn jobs_status(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id_opt = extract_project_id_optional(&headers);
    let EngineState { job_queue, .. } = state;

    if let Some(project_id) = project_id_opt {
        if let Some(session) = job_queue.get_session(&project_id) {
            let progress = session.get_progress();
            (StatusCode::OK, Json(serde_json::json!(progress)))
        } else {
            // No active session - return idle status
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "phase": "idle",
                    "writes_completed": 0,
                    "writes_total": 0
                })),
            )
        }
    } else {
        // Global progress
        let progress = job_queue.get_global_progress();
        (StatusCode::OK, Json(serde_json::json!(progress)))
    }
}

async fn recall_grounded(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<RecallGroundedRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use crate::grounding::{create_grounding_proof, GroundingEngine};
    use std::time::Instant;

    let project_id = if let Some(ref projects) = req.projects {
        projects.first().cloned().unwrap_or_else(|| {
            headers
                .get("X-Project-ID")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("default")
                .to_string()
        })
    } else {
        match extract_project_id(&headers) {
            Ok(id) => id,
            Err(e) => return e,
        }
    };

    let EngineState {
        ref mt_engine,
        ref cuepack_registry,
        ..
    } = &state;
    let start = Instant::now();
    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };

    // 1. Standard CueMap Recall
    let (resolved, _lexicon_memory_ids, tokens) =
        ctx.resolve_cues_from_text(&req.query_text, false);
    let mut normalized_cues = Vec::new();
    for cue in &resolved {
        let (normalized, _) = crate::normalization::normalize_cue(cue, &ctx.normalization);
        normalized_cues.push(normalized);
    }

    let mut expanded_cues = if req.disable_alias_expansion {
        normalized_cues.into_iter().map(|c| (c, 1.0)).collect()
    } else {
        // tokens were computed in step 1, reuse them!
        ctx.expand_query_cues(normalized_cues, &tokens)
    };
    let _query_intent = apply_query_intent(
        &ctx,
        cuepack_registry,
        req.cuepacks.as_deref(),
        Some(&req.query_text),
        None,
        &mut expanded_cues,
    );

    let heatmap = ctx.market_heatmap.read().ok();
    let heatmap_ref = heatmap.as_deref();

    let results = ctx.main.recall_weighted(
        expanded_cues.clone(),
        req.limit.max(20),
        req.auto_reinforce,
        req.min_intersection,
        req.expansion_depth,
        true, // explain
        req.disable_salience_bias,
        heatmap_ref,
        None, // mandatory_cues not exposed in RecallGroundedRequest
    );
    drop(heatmap); // Guard must be dropped before async return to satisfy Send (even if implicit)

    // 2. Apply Budgeting Logic
    let (selected, excluded, context_block) = GroundingEngine::select_memories(
        req.query_text.clone(),
        resolved.clone(),
        expanded_cues.clone(),
        results,
        req.token_budget,
    );

    // 3. Create Proof
    let proof = create_grounding_proof(
        uuid::Uuid::new_v4().to_string(),
        req.query_text,
        resolved,
        expanded_cues,
        req.token_budget,
        selected,
        excluded,
    );

    let elapsed = start.elapsed();

    // 4. Sign Context
    let signed_context = if let Some(signer) = state.context_signer {
        signer.sign(&context_block)
    } else {
        crate::crypto::SignedContext {
            algorithm: "none",
            signature: "error: signing key not set".to_string(),
            public_key: None,
        }
    };

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "verified_context": context_block,
            "proof": proof,
            "engine_latency_ms": elapsed.as_secs_f64() * 1000.0,
            "signature_alg": signed_context.algorithm,
            "signature": signed_context.signature,
            "public_key": signed_context.public_key
        })),
    )
}

async fn list_projects(State(state): State<EngineState>) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState { mt_engine, .. } = state;
    let projects = mt_engine.list_projects();
    (StatusCode::OK, Json(serde_json::json!(projects)))
}

async fn create_project(
    State(state): State<EngineState>,
    Json(req): Json<CreateProjectRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState {
        mt_engine,
        read_only,
        ..
    } = state;
    if read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "Read-only mode"})),
        );
    }

    if !validate_project_id(&req.project_id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid project ID format"})),
        );
    }

    // Check if exists first to return 409 Conflict logic if desired, or just idempotent OK
    // get_or_create_project is idempotent, but we might want to be explicit.
    // For now, let's just use get_or_create_project and return 200 OK or 201 Created.
    // Actually, if we want to mimic "create", 201 is good.

    match mt_engine.get_or_create_project(req.project_id.clone()) {
        Ok(_) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "status": "created",
                "project_id": req.project_id
            })),
        ),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": e})),
        ),
    }
}

async fn delete_project(
    State(state): State<EngineState>,
    Path(project_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState { mt_engine, .. } = state;
    let deleted = mt_engine.delete_project(&project_id);
    if deleted {
        (
            StatusCode::OK,
            Json(serde_json::json!({"status": "deleted", "project_id": project_id})),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Project not found"})),
        )
    }
}

async fn project_artifacts(
    State(state): State<EngineState>,
    Path(project_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    if !validate_project_id(&project_id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid project ID format"})),
        );
    }

    match state.mt_engine.project_artifact_summary(&project_id) {
        Ok(summary) => (StatusCode::OK, Json(serde_json::json!(summary))),
        Err(err) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": err})),
        ),
    }
}

async fn reload_project_artifacts(
    State(state): State<EngineState>,
    Path(project_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    if !validate_project_id(&project_id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid project ID format"})),
        );
    }

    match state.mt_engine.reload_project_artifacts(&project_id) {
        Ok(summary) => (StatusCode::OK, Json(serde_json::json!(summary))),
        Err(err) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": err})),
        ),
    }
}

async fn export_project(
    State(state): State<EngineState>,
    Path(project_id): Path<String>,
    Query(query): Query<ProjectExportQuery>,
) -> (StatusCode, Json<serde_json::Value>) {
    if !validate_project_id(&project_id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid project ID format"})),
        );
    }

    let Some(ctx) = state.mt_engine.get_project(&project_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Project not found"})),
        );
    };

    let limit = query.limit.clamp(1, 10_000);
    let cursor = query.cursor.unwrap_or(0);
    let mut ids: Vec<MemoryId> = ctx
        .main
        .get_memories()
        .iter()
        .map(|entry| *entry.key())
        .filter(|id| *id > cursor)
        .collect();
    ids.sort_unstable();

    let has_more = ids.len() > limit;
    ids.truncate(limit);
    let next_cursor = if has_more { ids.last().copied() } else { None };
    let mut exported = Vec::with_capacity(ids.len());

    for id in ids {
        let Some(memory) = ctx.main.get_memory(id) else {
            continue;
        };
        let mut item = serde_json::json!({
            "id": memory.id,
            "source_key": memory.source_key,
            "created_at": memory.created_at,
            "last_accessed": memory.last_accessed,
            "disk_backed": memory.disk_backed,
        });

        if query.include_content {
            let content = ctx
                .main
                .read_memory_content(&memory)
                .unwrap_or_else(|err| format!("<content read failed: {}>", err));
            item["content"] = serde_json::json!(content);
        }
        if query.include_cues {
            item["cues"] = serde_json::json!(memory.cues);
        }
        if query.include_metadata {
            item["metadata"] = serde_json::json!(memory.metadata);
        }
        exported.push(item);
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "project_id": project_id,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "limit": limit,
            "count": exported.len(),
            "has_more": has_more,
            "include_content": query.include_content,
            "include_cues": query.include_cues,
            "include_metadata": query.include_metadata,
            "memories": exported
        })),
    )
}

fn normalize_included_paths(paths: Option<Vec<String>>) -> Result<Vec<String>, String> {
    let mut normalized = Vec::new();
    for raw_path in paths.unwrap_or_default() {
        let candidate = raw_path.trim().replace('\\', "/");
        let candidate = candidate.trim_matches('/');
        if candidate.is_empty() || candidate == "." {
            return Ok(Vec::new());
        }

        let mut components = Vec::new();
        for component in std::path::Path::new(candidate).components() {
            match component {
                std::path::Component::Normal(value) => {
                    components.push(value.to_string_lossy().to_string())
                }
                std::path::Component::CurDir => {}
                _ => {
                    return Err(format!(
                        "Included path '{}' must stay within the watch directory",
                        raw_path
                    ))
                }
            }
        }
        if !components.is_empty() {
            normalized.push(components.join("/"));
        }
    }
    normalized.sort();
    normalized.dedup();
    Ok(normalized)
}

fn normalize_ignored_extensions(extensions: Option<Vec<String>>) -> Vec<String> {
    let mut normalized: Vec<String> = extensions
        .unwrap_or_default()
        .into_iter()
        .map(|extension| extension.trim().trim_start_matches('.').to_lowercase())
        .filter(|extension| !extension.is_empty())
        .collect();
    normalized.sort();
    normalized.dedup();
    normalized
}

async fn preview_directory(
    State(state): State<EngineState>,
    Json(req): Json<PreviewDirectoryRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let watch_dir = match std::fs::canonicalize(&req.watch_dir) {
        Ok(path) if path.is_dir() => path.to_string_lossy().to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Directory '{}' does not exist", req.watch_dir)
                })),
            )
        }
    };
    let included_paths = match normalize_included_paths(req.included_paths) {
        Ok(paths) => paths,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": error})),
            )
        }
    };
    let config = crate::agent::AgentConfig {
        project_id: "directory-preview".to_string(),
        watch_dir,
        throttle_ms: 0,
        state_file: None,
        included_paths,
        ignored_patterns: req.ignored_patterns.unwrap_or_default(),
        ignored_extensions: normalize_ignored_extensions(req.ignored_extensions),
    };
    let ingester = crate::agent::ingester::Ingester::new(config, state.job_queue);

    match ingester.preview_scope() {
        Ok(preview) => (
            StatusCode::OK,
            Json(serde_json::to_value(preview).unwrap_or_else(|error| {
                serde_json::json!({"error": format!("Failed to serialize preview: {}", error)})
            })),
        ),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": error})),
        ),
    }
}

async fn get_project_watch_dir(
    State(state): State<EngineState>,
    Path(project_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    match state.mt_engine.load_project_meta(&project_id) {
        Ok(meta) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "project_id": project_id,
                "initialized": meta.agent_enabled && meta.watch_dir.is_some(),
                "watch_dir": meta.watch_dir,
                "included_paths": meta.included_paths,
                "ignored_patterns": meta.ignored_patterns,
                "ignored_extensions": meta.ignored_extensions,
            })),
        ),
        Err(error) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": error})),
        ),
    }
}

async fn set_project_watch_dir(
    State(state): State<EngineState>,
    Path(project_id): Path<String>,
    Json(req): Json<SetWatchDirRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState {
        mt_engine,
        read_only,
        agent_manager,
        data_dir,
        ..
    } = state;

    if read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": "Read-only mode: modifications are not allowed"
            })),
        );
    }

    let watch_dir = match std::fs::canonicalize(&req.watch_dir) {
        Ok(path) if path.is_dir() => path.to_string_lossy().to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Directory '{}' does not exist", req.watch_dir)
                })),
            )
        }
    };
    let included_paths = match normalize_included_paths(req.included_paths) {
        Ok(paths) => paths,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": error})),
            )
        }
    };
    let ignored_patterns = req.ignored_patterns.unwrap_or_default();
    let ignored_extensions = normalize_ignored_extensions(req.ignored_extensions);

    match mt_engine.set_project_watch_config(
        &project_id,
        watch_dir.clone(),
        included_paths.clone(),
        ignored_patterns.clone(),
        ignored_extensions.clone(),
    ) {
        Ok(_) => {
            // Immediately start/update the agent
            let agent_config = crate::agent::AgentConfig {
                project_id: project_id.clone(),
                watch_dir: watch_dir.clone(),
                throttle_ms: 100, // Small throttle to prevent CPU pinning
                state_file: Some(
                    std::path::PathBuf::from(data_dir)
                        .join("snapshots")
                        .join(format!("{}_agent_state.json", project_id)),
                ),
                included_paths: included_paths.clone(),
                ignored_patterns: ignored_patterns.clone(),
                ignored_extensions: ignored_extensions.clone(),
            };

            // Spawn the starting of the agent securely
            let project_id_clone = project_id.clone();
            tokio::spawn(async move {
                agent_manager
                    .start_agent(&project_id_clone, agent_config)
                    .await;
            });

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "updated",
                    "project_id": project_id,
                    "watch_dir": watch_dir,
                    "included_paths": included_paths,
                    "ignored_patterns": ignored_patterns,
                    "ignored_extensions": ignored_extensions,
                })),
            )
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e})),
        ),
    }
}

// Multi-tenant Alias Handlers

async fn add_alias(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<AddAliasRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState {
        mt_engine,
        read_only,
        ..
    } = state;
    if read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "Read-only"})),
        );
    }

    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };

    let alias_source_key = uuid::Uuid::new_v4().to_string();
    let content = serde_json::json!({
        "from": req.from,
        "to": req.to,
        "downweight": req.weight.unwrap_or(0.85),
        "status": "active",
        "reason": "manual"
    })
    .to_string();

    let cues = vec![
        "type:alias".to_string(),
        format!("from:{}", req.from),
        format!("to:{}", req.to),
        "status:active".to_string(),
        "reason:manual".to_string(),
    ];

    let alias_id = ctx.aliases.upsert_memory_with_source_key(
        alias_source_key,
        content,
        cues,
        None,
        Some(MainStats::default()),
        false,
        false,
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({"id": alias_id, "status": "created"})),
    )
}

async fn get_aliases(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState { mt_engine, .. } = state;
    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };

    let cue = params.get("cue").cloned().unwrap_or_default();
    if cue.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Missing 'cue' query param"})),
        );
    }

    let query_cues = vec![
        "type:alias".to_string(),
        format!("to:{}", cue),
        "status:active".to_string(),
    ];

    let results = ctx.aliases.recall(query_cues, 50, false, None);
    let mut aliases = Vec::new();

    for res in results {
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&res.content) {
            let from_match = data
                .get("from")
                .and_then(|v| v.as_str())
                .map(|v| v == cue)
                .unwrap_or(false);
            let to_match = data
                .get("to")
                .and_then(|v| v.as_str())
                .map(|v| v == cue)
                .unwrap_or(false);

            if from_match || to_match {
                aliases.push(data);
            }
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({"aliases": aliases})),
    )
}

/// Lexicon Surgeon (Multi-tenant): Inspect a cue in the Lexicon
async fn lexicon_inspect(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Path(cue): axum::extract::Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState { mt_engine, .. } = state;
    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };
    let cue_lower = cue.to_lowercase();

    // Helper: count main memories that have token in cues but NOT canonical
    let count_affected = |token: &str, canonical: &str| -> usize {
        let token_lower = token.to_lowercase();
        let canonical_lower = canonical.to_lowercase();
        let mut count = 0;

        for ref_multi in ctx.main.get_memories().iter() {
            let memory = ref_multi.value();
            let cues_lower: Vec<String> = memory.cues.iter().map(|c| c.to_lowercase()).collect();
            if cues_lower.contains(&token_lower) && !cues_lower.contains(&canonical_lower) {
                count += 1;
            }
        }
        count
    };

    // 1. OUTGOING: What does this token trigger?
    let outgoing_results = ctx.lexicon.recall_fast(vec![cue_lower.clone()], 100);
    let outgoing: Vec<LexiconEntry> = outgoing_results
        .iter()
        .map(|r| {
            let token = r
                .metadata
                .get("cues")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str())
                .unwrap_or(&cue_lower)
                .to_string();

            let affected = count_affected(&token, &r.content);

            LexiconEntry {
                memory_id: r.memory_id,
                content: r.content.clone(),
                token,
                reinforcement_score: r.reinforcement_score,
                created_at: r.created_at,
                affected_memories_count: affected,
            }
        })
        .collect();

    // 2. INCOMING: What tokens map to this canonical cue?
    let mut incoming: Vec<LexiconEntry> = Vec::new();
    for ref_multi in ctx.lexicon.get_memories().iter() {
        let memory = ref_multi.value();
        let content = ctx.lexicon.read_memory_content(memory).unwrap_or_default();
        if content.to_lowercase() == cue_lower {
            for token in &memory.cues {
                let affected = count_affected(token, &content);
                incoming.push(LexiconEntry {
                    memory_id: memory.id,
                    content: content.clone(),
                    token: token.clone(),
                    reinforcement_score: memory.stats.get_reinforcement_count() as f64,
                    created_at: memory.created_at,
                    affected_memories_count: affected,
                });
            }
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::json!(LexiconInspectResponse {
            cue: cue_lower,
            outgoing,
            incoming,
        })),
    )
}

/// Delete a lexicon entry (multi-tenant)
async fn lexicon_delete(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Path(memory_id): axum::extract::Path<MemoryId>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState {
        mt_engine,
        read_only,
        ..
    } = state;
    if read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "Read-only mode"})),
        );
    }

    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };
    let deleted = ctx.lexicon.delete_memory(memory_id);
    if deleted {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "deleted",
                "memory_id": memory_id
            })),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "Entry not found",
                "memory_id": memory_id
            })),
        )
    }
}

/// Get full Lexicon as graph data (multi-tenant)
async fn lexicon_graph(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState { mt_engine, .. } = state;
    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };
    let mut nodes = Vec::new();
    let mut links = Vec::new();
    let mut token_to_canonical: HashMap<String, Vec<String>> = HashMap::new();

    // Return all entries (no limit)
    for ref_multi in ctx.lexicon.get_memories().iter() {
        let memory = ref_multi.value();
        let canonical = ctx.lexicon.read_memory_content(memory).unwrap_or_default();
        for token in &memory.cues {
            token_to_canonical
                .entry(token.clone())
                .or_default()
                .push(canonical.clone());
        }
    }

    let mut node_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (token, canonicals) in &token_to_canonical {
        if !node_ids.contains(token) {
            nodes.push(serde_json::json!({
                "id": token,
                "label": token,
                "group": "token"
            }));
            node_ids.insert(token.clone());
        }

        for canonical in canonicals {
            if !node_ids.contains(canonical) {
                nodes.push(serde_json::json!({
                    "id": canonical,
                    "label": canonical,
                    "group": "canonical"
                }));
                node_ids.insert(canonical.clone());
            }

            if token != canonical {
                links.push(serde_json::json!({
                    "source": token,
                    "target": canonical
                }));
            }
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "nodes": nodes,
            "links": links,
            "total_entries": nodes.len()
        })),
    )
}

/// Manually wire a token to a canonical cue (multi-tenant)
async fn lexicon_wire(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<WireLexiconRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState {
        mt_engine,
        read_only,
        ..
    } = state;
    if read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "Read-only mode"})),
        );
    }

    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };
    let token = req.token.to_lowercase();
    let canonical = req.canonical.to_lowercase();
    let lex_source_key = format!("cue:{}", canonical);

    let lex_id = ctx.lexicon.upsert_memory_with_source_key(
        lex_source_key,
        canonical.clone(),
        vec![token.clone()],
        None,
        Some(LexiconStats::default()),
        false,
        false,
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "wired",
            "memory_id": lex_id,
            "token": token,
            "canonical": canonical
        })),
    )
}

async fn merge_aliases(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<MergeAliasRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState {
        mt_engine,
        read_only,
        ..
    } = state;
    if read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "Read-only"})),
        );
    }

    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };
    let mut created_ids = Vec::new();

    for from_cue in req.cues {
        let alias_source_key = uuid::Uuid::new_v4().to_string();
        let content = serde_json::json!({
            "from": from_cue,
            "to": req.to,
            "downweight": 1.0,
            "status": "active",
            "reason": "manual_merge"
        })
        .to_string();

        let cues = vec![
            "type:alias".to_string(),
            format!("from:{}", from_cue),
            format!("to:{}", req.to),
            "status:active".to_string(),
            "reason:manual_merge".to_string(),
        ];

        let alias_id = ctx.aliases.upsert_memory_with_source_key(
            alias_source_key,
            content,
            cues,
            None,
            Some(MainStats::default()),
            false,
            false,
        );
        created_ids.push(alias_id);
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "merged",
            "target": req.to,
            "count": created_ids.len()
        })),
    )
}

/// Ingest content from a URL using the Agent's Ingester
/// Supports recursive crawling when depth > 0

async fn recall_web(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<RecallWebRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use crate::agent::ingester::Ingester;
    use crate::agent::search::search_ddg_lite;
    use crate::agent::AgentConfig;
    use std::time::Instant;

    let EngineState {
        read_only,
        job_queue,
        ..
    } = state;
    if req.persist && read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "Read-only mode: cannot persist"})),
        );
    }

    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    // Ensure project exists (auto-create)
    if let Err(e) = state.mt_engine.get_or_create_project(project_id.clone()) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": e})),
        );
    }

    // Create an ingester for this request
    let config = AgentConfig {
        project_id: project_id.clone(),
        watch_dir: String::new(),
        throttle_ms: 0,
        state_file: None,
        included_paths: Vec::new(),
        ignored_patterns: Vec::new(),
        ignored_extensions: Vec::new(),
    };
    let ingester = Ingester::new(config.clone(), job_queue.clone());
    let ingester = std::sync::Arc::new(ingester); // Arc for sharing across tasks

    let start_time = Instant::now();
    let mut chunks = Vec::new();
    let mut urls_processed = Vec::new();

    // 1. Determine targets: specific URL or Search
    if let Some(url) = &req.url {
        // Direct URL Mode
        urls_processed.push(url.clone());
        match ingester.fetch_and_chunk_url(url).await {
            Ok(c) => chunks.extend(c),
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": format!("Failed to fetch URL: {}", e)})),
                )
            }
        };
    } else {
        // Search Mode
        let search_results = match search_ddg_lite(&req.query, 5).await {
            Ok(res) => res,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": format!("Search failed: {}", e)})),
                )
            }
        };

        urls_processed = search_results.clone();

        // Parallel Fetch & Chunk using JoinSet for async concurrency
        let mut set = tokio::task::JoinSet::new();

        for url in search_results {
            let ingester_clone = ingester.clone();
            set.spawn(async move { (url.clone(), ingester_clone.fetch_and_chunk_url(&url).await) });
        }

        // Collect results
        while let Some(res) = set.join_next().await {
            if let Ok((url, result)) = res {
                match result {
                    Ok(c) => chunks.extend(c),
                    Err(e) => tracing::warn!("Failed to fetch search result {}: {}", url, e),
                }
            }
        }

        if chunks.is_empty() {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "No content found from search results"})),
            );
        }
    }

    let fetch_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    // 2. Immediate Recall (In-Memory)
    let processing_start = Instant::now();
    let query_cues = crate::nl::tokenize_to_cues(&req.query);

    // Simple improved scoring: weighted intersection
    struct ScoredChunk {
        content: String,
        score: f64,
        intersection: usize,
    }

    let mut scored_chunks: Vec<ScoredChunk> = chunks
        .iter()
        .map(|chunk| {
            let chunk_cues = crate::nl::tokenize_to_cues(&chunk.content);

            let intersection = chunk_cues.iter().filter(|c| query_cues.contains(c)).count();

            // Simple scoring: intersection count * 10
            let mut score = (intersection as f64) * 10.0;

            // Boost if query terms appear in structural cues (e.g. title, header)
            for q in &query_cues {
                for s in &chunk.structural_cues {
                    if s.to_lowercase().contains(q) {
                        score += 5.0;
                    }
                }
            }

            ScoredChunk {
                content: chunk.content.clone(),
                score,
                intersection,
            }
        })
        .collect();

    // Sort by score desc
    scored_chunks.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top results (more for search mode to have variety)
    let limit = if req.url.is_none() { 10 } else { 5 };

    let results: Vec<serde_json::Value> = scored_chunks
        .into_iter()
        .take(limit)
        .filter(|r| r.score > 0.0)
        .map(|r| {
            serde_json::json!({
                "content": r.content,
                "score": r.score,
                "intersection": r.intersection,
                // "url": r.url // Optional source info
            })
        })
        .collect();

    let processing_ms = processing_start.elapsed().as_secs_f64() * 1000.0;

    // 3. Optional Persistence (Async)
    if req.persist {
        let project_id_clone = project_id.clone();
        let chunks_clone = chunks.clone();
        let urls_processed_clone = urls_processed.clone();

        // Fix: Use local job_queue variable, not state.job_queue (which is moved)
        let job_queue_clone = job_queue.clone();

        tokio::spawn(async move {
            let config = AgentConfig {
                project_id: project_id_clone.clone(),
                watch_dir: String::new(),
                throttle_ms: 0,
                state_file: None,
                included_paths: Vec::new(),
                ignored_patterns: Vec::new(),
                ignored_extensions: Vec::new(),
            };
            let mut async_ingester = Ingester::new(config, job_queue_clone);

            // For search results, we have mixed sources. `process_chunks` expects a single source?
            // `process_chunks` takes `source: &str`.
            // We should arguably just say "web_search" or iterate and group by source?
            // Actually `process_chunks` uses source to generate ID. If we pass "web_search", valid.
            // But we should differentiate URLs if possible.
            // Ingester implementation:
            // let memory_id = format!("{}:{}", source, chunk_hash);
            // If all share "web_search", they dedup by content hash, which is fine.

            let source = if urls_processed_clone.len() == 1 {
                format!("url:{}", urls_processed_clone[0])
            } else {
                "web_search".to_string()
            };

            if let Err(e) = async_ingester
                .process_chunks(chunks_clone, &project_id_clone, &source)
                .await
            {
                tracing::error!("Async persistence failed for web recall: {}", e);
            }
        });
    }

    let total_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "urls": urls_processed,
            "results": results,
            "latency_ms": total_ms,
            "timings": {
                "fetch_chunk": fetch_ms,
                "search_overhead": if req.url.is_none() { true } else { false },
                "processing": processing_ms
            }
        })),
    )
}

/// Ingest content from a URL using the Agent's Ingester
/// Supports recursive crawling when depth > 0
async fn ingest_url(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<IngestUrlRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use crate::agent::ingester::Ingester;
    use crate::agent::AgentConfig;

    let EngineState {
        read_only,
        job_queue,
        ..
    } = state;
    if read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "Read-only mode"})),
        );
    }

    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    // Ensure project exists (auto-create)
    if let Err(e) = state.mt_engine.get_or_create_project(project_id.clone()) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": e})),
        );
    }

    // Create an ingester for this request
    let config = AgentConfig {
        project_id: project_id.clone(),
        watch_dir: String::new(), // Not used for API-driven ingestion
        throttle_ms: 0,
        state_file: None,
        included_paths: Vec::new(),
        ignored_patterns: Vec::new(),
        ignored_extensions: Vec::new(),
    };
    let mut ingester = Ingester::new(config, job_queue);

    // Check if recursive crawling is requested
    if req.depth > 0 {
        // Recursive crawl
        match ingester
            .process_url_recursive(&req.url, &project_id, req.depth, req.same_domain_only)
            .await
        {
            Ok(result) => (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "crawled",
                    "url": req.url,
                    "depth": req.depth,
                    "pages_crawled": result.pages_crawled,
                    "total_chunks": result.memory_ids.len(),
                    "links_found": result.links_found,
                    "links_skipped": result.links_skipped,
                    "memory_ids": result.memory_ids,
                    "errors": result.errors.iter().map(|(url, err)| {
                        serde_json::json!({"url": url, "error": err})
                    }).collect::<Vec<_>>()
                })),
            ),
            Err(e) => (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Failed to crawl URL: {}", e)
                })),
            ),
        }
    } else {
        // Single page ingestion (original behavior)
        match ingester.process_url(&req.url, &project_id).await {
            Ok(memory_ids) => (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "ingested",
                    "url": req.url,
                    "chunks": memory_ids.len(),
                    "memory_ids": memory_ids
                })),
            ),
            Err(e) => (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Failed to ingest URL: {}", e)
                })),
            ),
        }
    }
}

/// Request for POST /ingest/content - ingest raw content
#[derive(Debug, Deserialize)]
pub struct IngestContentRequest {
    pub content: String,
    #[serde(default = "default_filename")]
    pub filename: String, // Used to determine content type (e.g. "notes.md", "data.json")
    #[serde(default)]
    pub source_key: Option<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub structural_cues: Vec<String>,
    #[serde(default)]
    pub segmenter: TextSegmenterMode,
    #[serde(default)]
    pub segment_window_size: Option<usize>,
    #[serde(default)]
    pub segment_overlap: Option<usize>,
    #[serde(default)]
    pub segment_min_chunk_chars: Option<usize>,
    #[serde(default)]
    pub segment_max_chunk_chars: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct DebugAnalyzeTextRequest {
    pub text: String,
    #[serde(default)]
    pub query_time: Option<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub existing_cues: Vec<String>,
    #[serde(default)]
    pub available_cues: Vec<String>,
    #[serde(default)]
    pub cuepacks: Option<Vec<String>>,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub segmenter: TextSegmenterMode,
    #[serde(default)]
    pub segment_window_size: Option<usize>,
    #[serde(default)]
    pub segment_overlap: Option<usize>,
    #[serde(default)]
    pub segment_min_chunk_chars: Option<usize>,
    #[serde(default)]
    pub segment_max_chunk_chars: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextSegmenterMode {
    SentenceWindow,
    LogicalBlock,
}

impl Default for TextSegmenterMode {
    fn default() -> Self {
        Self::SentenceWindow
    }
}

fn default_filename() -> String {
    "content.txt".to_string()
}

fn segmenter_config_from_request(req: &IngestContentRequest) -> crate::agent::chunker::SegmenterConfig {
    let mut config = crate::agent::chunker::SegmenterConfig::default();
    if let Some(window_size) = req.segment_window_size {
        config.window_size = window_size.max(1);
    }
    if let Some(overlap) = req.segment_overlap {
        config.overlap = overlap.min(config.window_size.saturating_sub(1));
    }
    if let Some(min_chars) = req.segment_min_chunk_chars {
        config.min_chunk_chars = min_chars.max(1);
    }
    if let Some(max_chars) = req.segment_max_chunk_chars {
        config.max_chunk_chars = max_chars.max(config.min_chunk_chars);
    }
    config
}

fn segmenter_config_from_debug_request(
    req: &DebugAnalyzeTextRequest,
) -> crate::agent::chunker::SegmenterConfig {
    let mut config = crate::agent::chunker::SegmenterConfig::default();
    if let Some(window_size) = req.segment_window_size {
        config.window_size = window_size.max(1);
    }
    if let Some(overlap) = req.segment_overlap {
        config.overlap = overlap.min(config.window_size.saturating_sub(1));
    }
    if let Some(min_chars) = req.segment_min_chunk_chars {
        config.min_chunk_chars = min_chars.max(1);
    }
    if let Some(max_chars) = req.segment_max_chunk_chars {
        config.max_chunk_chars = max_chars.max(config.min_chunk_chars);
    }
    config
}

async fn debug_analyze_text(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<DebugAnalyzeTextRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use crate::agent::chunker::Chunker;

    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let ctx = match state.mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
        }
    };

    let raw_cues = crate::nl::tokenize_to_cues(&req.text);
    let normalized_cues: Vec<String> = raw_cues
        .iter()
        .map(|cue| {
            let (normalized, _) = normalize_cue(cue, &ctx.normalization);
            normalized
        })
        .collect();

    let cuepack_selection = req.cuepacks.as_deref();
    let core_facets = crate::facets::extract_memory_facets_core(
        &req.text,
        req.metadata.as_ref(),
        &req.existing_cues,
    );
    let memory_facets = crate::facets::extract_memory_facets_with_cuepacks(
        &req.text,
        req.metadata.as_ref(),
        &req.existing_cues,
        &state.cuepack_registry,
        cuepack_selection,
    );
    let available_cues: HashSet<String> = req
        .available_cues
        .iter()
        .map(|cue| cue.to_lowercase())
        .collect();
    let query_intent = crate::facets::compile_query_intent_with_cuepacks(
        &req.text,
        req.query_time.as_deref(),
        |cue| ctx.main.get_cue_frequency(cue) > 0 || available_cues.contains(&cue.to_lowercase()),
        &state.cuepack_registry,
        cuepack_selection,
    );

    let segmenter_config = segmenter_config_from_debug_request(&req);
    let chunks = match req.segmenter {
        TextSegmenterMode::SentenceWindow => {
            Chunker::chunk_text_with_config(&req.text, &segmenter_config)
        }
        TextSegmenterMode::LogicalBlock => {
            Chunker::chunk_text_logical_blocks(&req.text, &segmenter_config)
        }
    };
    let chunk_summaries: Vec<serde_json::Value> = chunks
        .into_iter()
        .enumerate()
        .map(|(idx, chunk)| {
            let chunk_cues = crate::nl::tokenize_to_cues(&chunk.content);
            serde_json::json!({
                "index": idx,
                "chars": chunk.content.len(),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "context": chunk.context,
                "structural_cues": chunk.structural_cues,
                "cues": chunk_cues,
                "preview": chunk.content.chars().take(500).collect::<String>(),
            })
        })
        .collect();

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "raw_cues": raw_cues,
            "normalized_cues": normalized_cues,
            "core_facets": core_facets,
            "memory_facets": memory_facets,
            "query_intent": query_intent,
            "segmenter": req.segmenter,
            "filename": req.filename.unwrap_or_else(|| "content.txt".to_string()),
            "chunks": chunk_summaries,
        })),
    )
}

/// Ingest raw content using the Agent's Ingester
async fn ingest_content(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<IngestContentRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use crate::agent::chunker::Chunker;
    use crate::agent::ingester::Ingester;
    use crate::agent::AgentConfig;

    let EngineState {
        read_only,
        job_queue,
        ..
    } = state;
    if read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "Read-only mode"})),
        );
    }

    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    // Ensure project exists (auto-create)
    if let Err(e) = state.mt_engine.get_or_create_project(project_id.clone()) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": e})),
        );
    }

    let source = req
        .source_key
        .clone()
        .unwrap_or_else(|| format!("api:{}", req.filename));
    let virtual_path = std::path::PathBuf::from(&req.filename);
    let segmenter_config = segmenter_config_from_request(&req);
    let mut chunks = match req.segmenter {
        TextSegmenterMode::SentenceWindow => {
            if req.segment_window_size.is_some()
                || req.segment_overlap.is_some()
                || req.segment_min_chunk_chars.is_some()
                || req.segment_max_chunk_chars.is_some()
            {
                Chunker::chunk_text_with_config(&req.content, &segmenter_config)
            } else {
                Chunker::chunk_file(&virtual_path, &req.content)
            }
        }
        TextSegmenterMode::LogicalBlock => {
            Chunker::chunk_text_logical_blocks(&req.content, &segmenter_config)
        }
    };
    if req.source_key.is_some() {
        Chunker::attach_parent_links(&mut chunks, &source);
    }
    Chunker::inherit_structural_cues(&mut chunks, &req.structural_cues);

    if chunks.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Failed to ingest content: no chunks produced"
            })),
        );
    }

    // Create an ingester for this request
    let config = AgentConfig {
        project_id: project_id.clone(),
        watch_dir: String::new(),
        throttle_ms: 0,
        state_file: None,
        included_paths: Vec::new(),
        ignored_patterns: Vec::new(),
        ignored_extensions: Vec::new(),
    };
    let mut ingester = Ingester::new(config, job_queue);

    match ingester
        .process_chunks_with_metadata(chunks, &project_id, &source, req.metadata)
        .await
    {
        Ok(memory_ids) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ingested",
                "filename": req.filename,
                "source_key": source,
                "chunks": memory_ids.len(),
                "memory_ids": memory_ids
            })),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("Failed to ingest content: {}", e)
            })),
        ),
    }
}

/// Ingest a binary file via multipart upload (for PDFs, Office docs, etc.)
async fn ingest_file(
    State(state): State<EngineState>,
    headers: HeaderMap,
    mut multipart: axum_extra::extract::Multipart,
) -> (StatusCode, Json<serde_json::Value>) {
    use crate::agent::chunker::Chunker;
    use sha2::{Digest, Sha256};
    use std::io::Write;

    let EngineState {
        read_only,
        job_queue,
        ..
    } = state;
    if read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "Read-only mode"})),
        );
    }

    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    // Ensure project exists (auto-create)
    if let Err(e) = state.mt_engine.get_or_create_project(project_id.clone()) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": e})),
        );
    }

    // Extract file from multipart
    let mut filename = String::new();
    let mut file_bytes: Vec<u8> = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        if name == "file" {
            filename = field.file_name().unwrap_or("upload.bin").to_string();
            if let Ok(bytes) = field.bytes().await {
                file_bytes = bytes.to_vec();
            }
        } else if name == "filename" {
            if let Ok(text) = field.text().await {
                filename = text;
            }
        }
    }

    if file_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "No file data received"
            })),
        );
    }

    // Write to temp file
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join(&filename);

    let mut temp_file = match std::fs::File::create(&temp_path) {
        Ok(f) => f,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Failed to create temp file: {}", e)
                })),
            )
        }
    };

    if let Err(e) = temp_file.write_all(&file_bytes) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to write temp file: {}", e)
            })),
        );
    }
    drop(temp_file);

    // Chunk the file
    let chunks = Chunker::chunk_binary_file(&temp_path);

    // Clean up temp file
    let _ = std::fs::remove_file(&temp_path);

    if chunks.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Failed to extract content from file (0 chunks)"
            })),
        );
    }

    // Track session for progress reporting
    let session = job_queue.session_manager.get_or_create(&project_id);
    for _ in &chunks {
        session.expect_write();
    }

    // Enqueue jobs for each chunk
    let source = format!("file:{}", filename);
    let mut source_keys = Vec::new();

    for chunk in chunks.iter() {
        let mut chunk_hasher = Sha256::new();
        chunk_hasher.update(chunk.content.as_bytes());
        let chunk_hash = format!("{:x}", chunk_hasher.finalize());
        let source_key = format!("{}:{}", source, chunk_hash);

        // ExtractAndIngest does the write - enqueue immediately
        job_queue
            .enqueue(crate::jobs::Job::ExtractAndIngest {
                project_id: project_id.clone(),
                source_key: source_key.clone(),
                content: chunk.content.clone(),
                file_path: source.clone(),
                structural_cues: chunk.structural_cues.clone(),
                metadata: None,
                category: chunk.category,
            })
            .await;

        session.write_complete();

        source_keys.push(source_key);
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ingested",
            "filename": filename,
            "chunks": source_keys.len(),
            "source_keys": source_keys
        })),
    )
}

/// Prometheus metrics endpoint - returns plain text in Prometheus exposition format
async fn prometheus_metrics(State(state): State<EngineState>) -> impl IntoResponse {
    use std::sync::atomic::Ordering;

    let EngineState {
        mt_engine,
        metrics,
        job_queue,
        ..
    } = state;

    // Get global stats from multi-tenant engine
    let global_stats = mt_engine.get_global_stats();
    let total_memories = global_stats
        .get("total_memories")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total_cues = global_stats
        .get("total_cues")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total_projects = global_stats
        .get("total_projects")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    // Get metrics from collector
    let ingestion_count = metrics.ingestion_count.load(Ordering::Relaxed);
    let recall_count = metrics.recall_count.load(Ordering::Relaxed);
    tracing::debug!(
        "Metrics: Found ingestion_count={}, recall_count={}",
        ingestion_count,
        recall_count
    );
    let recall_p99 = metrics.get_p99_latency();
    let recall_avg = metrics.get_avg_latency();

    // Get memory usage
    let memory_bytes = crate::metrics::get_memory_usage_bytes();

    // Get active jobs count
    let active_jobs = job_queue.pending_count();

    // Build Prometheus format output
    let output = format!(
        "# HELP cuemap_ingestion_rate Total memory ingestions since startup
# TYPE cuemap_ingestion_rate counter
cuemap_ingestion_rate {}

# HELP cuemap_recall_requests_total Total recall requests since startup
# TYPE cuemap_recall_requests_total counter
cuemap_recall_requests_total {}

# HELP cuemap_recall_latency_p99 P99 recall latency in milliseconds
# TYPE cuemap_recall_latency_p99 gauge
cuemap_recall_latency_p99 {:.2}

# HELP cuemap_recall_latency_avg Average recall latency in milliseconds
# TYPE cuemap_recall_latency_avg gauge
cuemap_recall_latency_avg {:.2}

# HELP cuemap_memory_usage_bytes Process memory usage in bytes (RSS)
# TYPE cuemap_memory_usage_bytes gauge
cuemap_memory_usage_bytes {}

# HELP cuemap_total_memories Total memories across all projects
# TYPE cuemap_total_memories gauge
cuemap_total_memories {}

# HELP cuemap_lexicon_size Total cues/terms in lexicon
# TYPE cuemap_lexicon_size gauge
cuemap_lexicon_size {}

# HELP cuemap_total_projects Number of active projects
# TYPE cuemap_total_projects gauge
cuemap_total_projects {}

# HELP cuemap_active_jobs Current pending background jobs
# TYPE cuemap_active_jobs gauge
cuemap_active_jobs {}
",
        ingestion_count,
        recall_count,
        recall_p99,
        recall_avg,
        memory_bytes,
        total_memories,
        total_cues,
        total_projects,
        active_jobs,
    );

    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        output,
    )
}

// ============================================================================
// Cloud Backup Endpoints
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct BackupRequest {
    pub project_id: String,
}

#[derive(Debug, Serialize)]
pub struct BackupResponse {
    pub success: bool,
    pub project_id: String,
    pub size_bytes: Option<u64>,
    pub message: String,
}

/// Upload a project snapshot to cloud storage
async fn backup_upload(
    State(state): State<EngineState>,
    Json(req): Json<BackupRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState {
        mt_engine,
        data_dir,
        cloud_backup,
        ..
    } = state;

    // Check if cloud backup is configured
    let backup_manager = match cloud_backup {
        Some(ref manager) => manager,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Cloud backup is not configured"
                })),
            );
        }
    };

    // Validate project ID
    if !validate_project_id(&req.project_id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid project ID format"})),
        );
    }

    // Save project locally first
    if let Err(e) = mt_engine.save_project(&req.project_id) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to save project locally: {}", e)
            })),
        );
    }

    // Read the snapshot files
    let snapshots_dir = mt_engine.list_snapshots();
    if !snapshots_dir.contains(&req.project_id) {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Project snapshot not found"})),
        );
    }

    // Get snapshot data from local files
    let snapshots_path = std::path::Path::new(&data_dir).join("snapshots");
    let main_path = snapshots_path.join(format!("{}.bin", req.project_id));
    let aliases_path = snapshots_path.join(format!("{}_aliases.bin", req.project_id));
    let lexicon_path = snapshots_path.join(format!("{}_lexicon.bin", req.project_id));

    let main_data = match std::fs::read(&main_path) {
        Ok(data) => bytes::Bytes::from(data),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Failed to read main snapshot: {}", e)
                })),
            );
        }
    };

    let aliases_data = std::fs::read(&aliases_path).ok().map(bytes::Bytes::from);
    let lexicon_data = std::fs::read(&lexicon_path).ok().map(bytes::Bytes::from);

    // Upload to cloud
    match backup_manager
        .upload_project_snapshot(&req.project_id, main_data, aliases_data, lexicon_data)
        .await
    {
        Ok(size) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "project_id": req.project_id,
                "size_bytes": size,
                "message": "Snapshot uploaded to cloud storage"
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to upload to cloud: {}", e)
            })),
        ),
    }
}

/// Download a project snapshot from cloud storage
async fn backup_download(
    State(state): State<EngineState>,
    Json(req): Json<BackupRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState {
        mt_engine,
        data_dir,
        cloud_backup,
        ..
    } = state;

    // Check if cloud backup is configured
    let backup_manager = match cloud_backup {
        Some(ref manager) => manager,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Cloud backup is not configured"
                })),
            );
        }
    };

    // Validate project ID
    if !validate_project_id(&req.project_id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid project ID format"})),
        );
    }

    // Download from cloud
    let (main_data, aliases_data, lexicon_data) = match backup_manager
        .download_project_snapshot(&req.project_id)
        .await
    {
        Ok(data) => data,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("Failed to download from cloud: {}", e)
                })),
            );
        }
    };

    // Save to local snapshots directory
    let snapshots_dir = std::path::Path::new(&data_dir).join("snapshots");

    // Create snapshots directory if needed
    if let Err(e) = std::fs::create_dir_all(&snapshots_dir) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to create snapshots directory: {}", e)
            })),
        );
    }

    // Write main snapshot
    let main_path = snapshots_dir.join(format!("{}.bin", req.project_id));
    if let Err(e) = std::fs::write(&main_path, &main_data) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to write main snapshot: {}", e)
            })),
        );
    }

    // Write aliases snapshot if present
    if let Some(data) = aliases_data {
        let aliases_path = snapshots_dir.join(format!("{}_aliases.bin", req.project_id));
        let _ = std::fs::write(&aliases_path, &data);
    }

    // Write lexicon snapshot if present
    if let Some(data) = lexicon_data {
        let lexicon_path = snapshots_dir.join(format!("{}_lexicon.bin", req.project_id));
        let _ = std::fs::write(&lexicon_path, &data);
    }

    // Load the project into memory
    match mt_engine.load_project(&req.project_id) {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "project_id": req.project_id,
                "size_bytes": main_data.len(),
                "message": "Snapshot downloaded and loaded from cloud storage"
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Downloaded but failed to load project: {}", e)
            })),
        ),
    }
}

/// List all cloud backups
async fn backup_list(State(state): State<EngineState>) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState { cloud_backup, .. } = state;

    // Check if cloud backup is configured
    let backup_manager = match cloud_backup {
        Some(ref manager) => manager,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Cloud backup is not configured"
                })),
            );
        }
    };

    match backup_manager.list_snapshots().await {
        Ok(entries) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "backups": entries,
                "count": entries.len()
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to list cloud backups: {}", e)
            })),
        ),
    }
}

/// Delete a cloud backup
async fn backup_delete(
    State(state): State<EngineState>,
    Path(project_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState { cloud_backup, .. } = state;

    // Check if cloud backup is configured
    let backup_manager = match cloud_backup {
        Some(ref manager) => manager,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Cloud backup is not configured"
                })),
            );
        }
    };

    // Validate project ID
    if !validate_project_id(&project_id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid project ID format"})),
        );
    }

    match backup_manager.delete_snapshot(&project_id).await {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "project_id": project_id,
                "message": "Cloud backup deleted"
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to delete cloud backup: {}", e)
            })),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TuningConfig;
    use crate::normalization::NormalizationConfig;
    use crate::projects::ProjectContext;
    use crate::structures::MainStats;
    use crate::taxonomy::Taxonomy;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn source_event_time_prefers_explicit_and_reads_structured_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "source_timestamp".to_string(),
            serde_json::json!("2024-01-01T00:00:00.250Z"),
        );

        assert_eq!(source_event_time(Some(42.0), Some(&metadata)), Some(42.0));
        assert_eq!(
            source_event_time(None, Some(&metadata)),
            Some(1_704_067_200.25)
        );
    }

    fn recall_result(
        match_integrity: f64,
        intersection_count: usize,
    ) -> crate::engine::RecallResult {
        crate::engine::RecallResult {
            memory_id: 1,
            content: "content".to_string(),
            score: 1.0,
            match_integrity,
            intersection_count,
            recency_score: 0.0,
            reinforcement_score: 0.0,
            salience_score: 0.0,
            created_at: 0.0,
            metadata: HashMap::new(),
            explain: None,
        }
    }

    #[test]
    fn parent_fusion_defaults_off_and_force_runs() {
        let results = vec![recall_result(0.95, 4)];

        assert!(!should_run_parent_fusion(
            &results,
            ParentFusionMode::Off,
            None,
            Some("summarize the key points"),
        ));
        assert!(should_run_parent_fusion(
            &results,
            ParentFusionMode::Force,
            None,
            Some("plain lookup"),
        ));
    }

    #[test]
    fn parent_fusion_auto_requires_synthesis_query() {
        assert!(!should_run_parent_fusion(
            &[recall_result(0.4, 1)],
            ParentFusionMode::Auto,
            None,
            Some("what is my favorite dessert"),
        ));
        assert!(should_run_parent_fusion(
            &[recall_result(0.4, 1)],
            ParentFusionMode::Auto,
            None,
            Some("summarize my language service progress in order"),
        ));
    }

    #[test]
    fn ordered_reconstruction_is_opt_in_and_intent_gated() {
        let mut intent = crate::facets::QueryIntent::default();
        assert!(!should_run_ordered_reconstruction(
            OrderedReconstructionMode::Off,
            Some(&intent)
        ));
        assert!(should_run_ordered_reconstruction(
            OrderedReconstructionMode::Force,
            None
        ));
        assert!(!should_run_ordered_reconstruction(
            OrderedReconstructionMode::Auto,
            Some(&intent)
        ));

        intent.labels.push("ordered_reconstruction".to_string());
        assert!(should_run_ordered_reconstruction(
            OrderedReconstructionMode::Auto,
            Some(&intent)
        ));

        intent.labels.clear();
        intent
            .labels
            .push("multi_evidence_collection".to_string());
        assert!(should_run_ordered_reconstruction(
            OrderedReconstructionMode::Auto,
            Some(&intent)
        ));
    }

    #[test]
    fn evidence_coverage_is_opt_in_and_intent_gated() {
        let mut intent = crate::facets::QueryIntent::default();
        assert!(!should_run_evidence_coverage(
            EvidenceCoverageMode::Off,
            Some(&intent)
        ));
        assert!(should_run_evidence_coverage(
            EvidenceCoverageMode::Force,
            None
        ));
        assert!(!should_run_evidence_coverage(
            EvidenceCoverageMode::Auto,
            Some(&intent)
        ));

        intent.labels.push("multi_evidence_summary".to_string());
        assert!(should_run_evidence_coverage(
            EvidenceCoverageMode::Auto,
            Some(&intent)
        ));

        intent.labels.clear();
        intent
            .labels
            .push("multi_evidence_collection".to_string());
        assert!(should_run_evidence_coverage(
            EvidenceCoverageMode::Auto,
            Some(&intent)
        ));

        intent.labels.clear();
        intent.labels.push("ordered_reconstruction".to_string());
        assert!(should_run_evidence_coverage(
            EvidenceCoverageMode::Auto,
            Some(&intent)
        ));
    }

    #[test]
    fn evidence_coverage_selects_diverse_session_evidence() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "evidence_coverage_test".to_string(),
        );

        let add_turn = |session: &str,
                        order: i64,
                        plan: Option<i64>,
                        content: &str,
                        cues: &[&str]|
         -> MemoryId {
            let mut metadata = HashMap::new();
            metadata.insert("source_session_id".to_string(), serde_json::json!(session));
            metadata.insert("source_turn_index".to_string(), serde_json::json!(order));
            if let Some(plan) = plan {
                metadata.insert("source_plan_idx".to_string(), serde_json::json!(plan));
            }
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                Some(metadata),
                MainStats::default(),
                false,
            )
        };

        let integration = add_turn(
            "thread-a",
            1,
            Some(1),
            "We designed language service integration.",
            &["source_role:assistant", "type:answer", "has:list", "language", "service", "integration", "architecture"],
        );
        let deployment = add_turn(
            "thread-a",
            2,
            Some(2),
            "We planned deployment and release steps.",
            &["source_role:assistant", "type:answer", "deployment", "release", "service"],
        );
        let performance = add_turn(
            "thread-a",
            3,
            Some(3),
            "We improved performance and latency.",
            &["source_role:assistant", "type:answer", "performance", "latency", "service"],
        );
        let unrelated = add_turn(
            "thread-a",
            4,
            Some(4),
            "We discussed a lunch menu.",
            &["source_role:assistant", "type:answer", "lunch", "menu"],
        );
        let distractor = add_turn(
            "thread-b",
            1,
            Some(2),
            "A different deployment discussion happened elsewhere.",
            &["source_role:assistant", "type:answer", "deployment", "service"],
        );

        let pivot = crate::engine::RecallResult {
            memory_id: deployment,
            content: "We planned deployment and release steps.".to_string(),
            score: 140.0,
            match_integrity: 0.6,
            intersection_count: 2,
            recency_score: 1.0,
            reinforcement_score: 0.0,
            salience_score: 0.0,
            created_at: 0.0,
            metadata: HashMap::new(),
            explain: None,
        };
        let pivot_score = pivot.score;

        let evidence = evidence_coverage_results(
            &ctx,
            &[
                ("language".to_string(), 1.0),
                ("service".to_string(), 0.8),
                ("integration".to_string(), 1.0),
                ("deployment".to_string(), 1.0),
                ("performance".to_string(), 1.0),
            ],
            &[pivot],
            10,
            100,
            1,
            true,
        );

        let ids: Vec<MemoryId> = evidence.iter().map(|result| result.memory_id).collect();
        assert!(ids.contains(&integration));
        assert!(ids.contains(&deployment));
        assert!(ids.contains(&performance));
        assert!(!ids.contains(&unrelated));
        assert!(!ids.contains(&distractor));
        assert!(evidence
            .iter()
            .all(|result| result.metadata.contains_key("evidence_coverage")));
        assert!(evidence.iter().any(|result| result
            .metadata
            .contains_key("evidence_coverage_source_plan")));
        assert!(evidence
            .iter()
            .all(|result| result.score < pivot_score));
    }

    #[test]
    fn slate_rerank_is_mode_and_intent_gated() {
        let mut intent = crate::facets::QueryIntent::default();
        intent.labels.push("multi_evidence_summary".to_string());

        assert!(!slate_rerank_requested(
            OrderedReconstructionMode::Off,
            EvidenceCoverageMode::Off,
            Some(&intent)
        ));
        assert!(slate_rerank_requested(
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&intent)
        ));

        let plain_intent = crate::facets::QueryIntent::default();
        assert!(!slate_rerank_requested(
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&plain_intent)
        ));
    }

    #[test]
    fn slate_rerank_promotes_coverage_candidates_below_protected_top() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "slate_rerank_test".to_string(),
        );

        let add_turn = |session: &str,
                        order: i64,
                        role: &str,
                        content: &str,
                        cues: &[&str]|
         -> MemoryId {
            let mut metadata = HashMap::new();
            metadata.insert("source_session_id".to_string(), serde_json::json!(session));
            metadata.insert("source_turn_index".to_string(), serde_json::json!(order));
            metadata.insert("source_role".to_string(), serde_json::json!(role));
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                Some(metadata),
                MainStats::default(),
                false,
            )
        };
        let make_result = |memory_id: MemoryId,
                           score: f64,
                           metadata: HashMap<String, serde_json::Value>|
         -> crate::engine::RecallResult {
            crate::engine::RecallResult {
                memory_id,
                content: format!("memory {memory_id}"),
                score,
                match_integrity: 0.2,
                intersection_count: 1,
                recency_score: 0.0,
                reinforcement_score: 0.0,
                salience_score: 0.0,
                created_at: 0.0,
                metadata,
                explain: None,
            }
        };

        let protected_a = add_turn(
            "thread-a",
            1,
            "assistant",
            "Protected top result A.",
            &["overview"],
        );
        let protected_b = add_turn(
            "thread-a",
            2,
            "assistant",
            "Protected top result B.",
            &["overview"],
        );
        let protected_c = add_turn(
            "thread-a",
            3,
            "assistant",
            "Protected top result C.",
            &["overview"],
        );
        let mut results = vec![
            make_result(protected_a, 300.0, HashMap::new()),
            make_result(protected_b, 290.0, HashMap::new()),
            make_result(protected_c, 280.0, HashMap::new()),
        ];

        for rank in 0..25 {
            let id = add_turn(
                "thread-b",
                rank,
                "assistant",
                "Generic distractor.",
                &["generic", "discussion"],
            );
            results.push(make_result(id, 270.0 - rank as f64, HashMap::new()));
        }

        let relevant_late = add_turn(
            "thread-a",
            24,
            "assistant",
            "We covered deployment and latency.",
            &["deployment", "latency", "service", "type:answer"],
        );
        let relevant_later = add_turn(
            "thread-a",
            40,
            "assistant",
            "We also covered integration architecture.",
            &["integration", "architecture", "service", "type:answer"],
        );
        let mut evidence_metadata = HashMap::new();
        evidence_metadata.insert("evidence_coverage".to_string(), serde_json::json!(true));
        results.push(make_result(relevant_late, 150.0, evidence_metadata.clone()));
        results.push(make_result(relevant_later, 149.0, evidence_metadata));

        let mut intent = crate::facets::QueryIntent::default();
        intent.labels.push("multi_evidence_summary".to_string());
        let moved = apply_slate_rerank(
            &ctx,
            &mut results,
            &[
                ("deployment".to_string(), 1.0),
                ("latency".to_string(), 1.0),
                ("integration".to_string(), 1.0),
                ("architecture".to_string(), 1.0),
                ("service".to_string(), 0.8),
            ],
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&intent),
            100,
        );
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let ids: Vec<MemoryId> = results.iter().map(|result| result.memory_id).collect();
        assert_eq!(&ids[..3], &[protected_a, protected_b, protected_c]);
        assert!(ids.iter().position(|id| *id == relevant_late).unwrap() < 20);
        assert!(ids.iter().position(|id| *id == relevant_later).unwrap() < 20);
        assert!(moved >= 2);
        assert!(results
            .iter()
            .any(|result| result.memory_id == relevant_late
                && result.metadata.contains_key("slate_rerank")));
    }

    #[test]
    fn slate_rerank_promotes_strong_summary_candidates_without_helper_metadata() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "slate_rerank_summary_signal_test".to_string(),
        );

        let add_turn = |session: &str,
                        order: i64,
                        content: &str,
                        cues: &[&str]|
         -> MemoryId {
            let mut metadata = HashMap::new();
            metadata.insert("source_session_id".to_string(), serde_json::json!(session));
            metadata.insert("source_turn_index".to_string(), serde_json::json!(order));
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                Some(metadata),
                MainStats::default(),
                false,
            )
        };
        let make_result = |memory_id: MemoryId,
                           score: f64|
         -> crate::engine::RecallResult {
            crate::engine::RecallResult {
                memory_id,
                content: format!("memory {memory_id}"),
                score,
                match_integrity: 0.2,
                intersection_count: 1,
                recency_score: 0.0,
                reinforcement_score: 0.0,
                salience_score: 0.0,
                created_at: 0.0,
                metadata: HashMap::new(),
                explain: None,
            }
        };

        let protected_a = add_turn("thread-a", 1, "Protected A.", &["overview"]);
        let protected_b = add_turn("thread-a", 2, "Protected B.", &["overview"]);
        let protected_c = add_turn("thread-a", 3, "Protected C.", &["overview"]);
        let mut results = vec![
            make_result(protected_a, 300.0),
            make_result(protected_b, 290.0),
            make_result(protected_c, 280.0),
        ];

        for rank in 0..25 {
            let id = add_turn(
                "thread-b",
                rank,
                "Generic project discussion.",
                &["generic", "project"],
            );
            results.push(make_result(id, 270.0 - rank as f64));
        }

        let relevant = add_turn(
            "thread-c",
            8,
            "City autocomplete in the weather app uses a debounced API lookup.",
            &["city", "autocomplete", "weather", "app", "lookup"],
        );
        results.push(make_result(relevant, 150.0));

        let mut intent = crate::facets::QueryIntent::default();
        intent.labels.push("multi_evidence_summary".to_string());
        let moved = apply_slate_rerank(
            &ctx,
            &mut results,
            &[
                ("city".to_string(), 1.0),
                ("autocomplete".to_string(), 1.0),
                ("weather".to_string(), 1.0),
                ("app".to_string(), 0.8),
                ("implementation".to_string(), 0.8),
            ],
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&intent),
            100,
        );
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let ids: Vec<MemoryId> = results.iter().map(|result| result.memory_id).collect();
        assert_eq!(&ids[..3], &[protected_a, protected_b, protected_c]);
        assert!(ids.iter().position(|id| *id == relevant).unwrap() < 20);
        assert!(moved >= 1);
        assert!(results
            .iter()
            .any(|result| result.memory_id == relevant
                && result.metadata.contains_key("slate_rerank")));
    }

    #[test]
    fn slate_rerank_promotes_standing_instruction_for_instruction_query() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "slate_rerank_instruction_test".to_string(),
        );

        let add_turn = |content: &str, cues: &[&str]| -> MemoryId {
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                None,
                MainStats::default(),
                false,
            )
        };
        let make_result = |memory_id: MemoryId,
                           score: f64|
         -> crate::engine::RecallResult {
            crate::engine::RecallResult {
                memory_id,
                content: format!("memory {memory_id}"),
                score,
                match_integrity: 0.1,
                intersection_count: 1,
                recency_score: 0.0,
                reinforcement_score: 0.0,
                salience_score: 0.0,
                created_at: 0.0,
                metadata: HashMap::new(),
                explain: None,
            }
        };

        let protected_a = add_turn("Protected A", &["layout"]);
        let protected_b = add_turn("Protected B", &["layout"]);
        let protected_c = add_turn("Protected C", &["layout"]);
        let mut results = vec![
            make_result(protected_a, 300.0),
            make_result(protected_b, 290.0),
            make_result(protected_c, 280.0),
        ];
        for rank in 0..45 {
            let id = add_turn("Generic layout discussion", &["layout", "project"]);
            results.push(make_result(id, 270.0 - rank as f64));
        }

        let instruction = add_turn(
            "Always include semantic HTML5 tag usage details when I ask about markup structure.",
            &[
                "type:standing_instruction",
                "instruction_trigger:markup",
                "semantic",
                "html5",
                "tag",
                "structure",
            ],
        );
        results.push(make_result(instruction, 120.0));

        let mut intent = crate::facets::QueryIntent::default();
        intent.labels.push("instruction_applicable".to_string());
        let moved = apply_slate_rerank(
            &ctx,
            &mut results,
            &[
                ("blog".to_string(), 1.0),
                ("layout".to_string(), 1.0),
                ("header".to_string(), 1.0),
                ("navigation".to_string(), 1.0),
                ("footer".to_string(), 1.0),
            ],
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&intent),
            100,
        );
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let ids: Vec<MemoryId> = results.iter().map(|result| result.memory_id).collect();
        assert_eq!(&ids[..3], &[protected_a, protected_b, protected_c]);
        assert!(ids.iter().position(|id| *id == instruction).unwrap() < 20);
        assert!(moved >= 1);
        assert!(results
            .iter()
            .any(|result| result.memory_id == instruction
                && result.metadata.contains_key("slate_rerank")));
    }

    #[test]
    fn slate_rerank_orders_selected_ordered_candidates_after_selection() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "slate_rerank_ordered_test".to_string(),
        );

        let add_turn = |session: &str, order: i64, content: &str, cues: &[&str]| -> MemoryId {
            let mut metadata = HashMap::new();
            metadata.insert("source_session_id".to_string(), serde_json::json!(session));
            metadata.insert("source_turn_index".to_string(), serde_json::json!(order));
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                Some(metadata),
                MainStats::default(),
                false,
            )
        };
        let make_result = |memory_id: MemoryId,
                           score: f64,
                           ordered: bool|
         -> crate::engine::RecallResult {
            let mut metadata = HashMap::new();
            if ordered {
                metadata.insert("ordered_reconstruction".to_string(), serde_json::json!(true));
            }
            crate::engine::RecallResult {
                memory_id,
                content: format!("memory {memory_id}"),
                score,
                match_integrity: 0.3,
                intersection_count: 1,
                recency_score: 0.0,
                reinforcement_score: 0.0,
                salience_score: 0.0,
                created_at: 0.0,
                metadata,
                explain: None,
            }
        };

        let protected_a = add_turn("thread-a", 1, "Protected A", &["bootstrap"]);
        let protected_b = add_turn("thread-a", 2, "Protected B", &["bootstrap"]);
        let protected_c = add_turn("thread-a", 3, "Protected C", &["bootstrap"]);
        let mut results = vec![
            make_result(protected_a, 300.0, false),
            make_result(protected_b, 290.0, false),
            make_result(protected_c, 280.0, false),
        ];
        for rank in 0..25 {
            let id = add_turn("thread-b", rank, "Generic project discussion", &["project"]);
            results.push(make_result(id, 270.0 - rank as f64, false));
        }

        let first = add_turn("thread-a", 5, "Bootstrap CDN setup", &["bootstrap", "cdn"]);
        let second = add_turn("thread-a", 7, "Bootstrap form classes", &["bootstrap", "form"]);
        let third = add_turn("thread-a", 11, "Bootstrap modal upgrade", &["bootstrap", "modal"]);
        results.push(make_result(third, 151.0, true));
        results.push(make_result(first, 150.0, true));
        results.push(make_result(second, 149.0, true));

        let mut intent = crate::facets::QueryIntent::default();
        intent.labels.push("ordered_reconstruction".to_string());
        let moved = apply_slate_rerank(
            &ctx,
            &mut results,
            &[
                ("bootstrap".to_string(), 1.0),
                ("cdn".to_string(), 1.0),
                ("form".to_string(), 1.0),
                ("modal".to_string(), 1.0),
            ],
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&intent),
            100,
        );
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let ids: Vec<MemoryId> = results.iter().map(|result| result.memory_id).collect();
        assert_eq!(&ids[..3], &[protected_a, protected_b, protected_c]);
        let first_pos = ids.iter().position(|id| *id == first).unwrap();
        let second_pos = ids.iter().position(|id| *id == second).unwrap();
        let third_pos = ids.iter().position(|id| *id == third).unwrap();
        assert!(first_pos < 20);
        assert!(second_pos < 20);
        assert!(third_pos < 20);
        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
        assert!(moved >= 3);
    }

    #[test]
    fn ordered_reconstruction_scans_selected_session_in_order() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "ordered_test".to_string(),
        );

        let add_turn = |session: &str, order: i64, content: &str, cues: &[&str]| -> MemoryId {
            let mut metadata = HashMap::new();
            metadata.insert("source_session_id".to_string(), serde_json::json!(session));
            metadata.insert("source_turn_index".to_string(), serde_json::json!(order));
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                Some(metadata),
                MainStats::default(),
                false,
            )
        };

        let first = add_turn(
            "thread-a",
            1,
            "We integrated the language detection service.",
            &["language", "service", "integrate"],
        );
        let second = add_turn(
            "thread-a",
            2,
            "Then we optimized translation service latency.",
            &["translation", "service", "optimize"],
        );
        let distractor = add_turn(
            "thread-b",
            1,
            "A different service discussion happened elsewhere.",
            &["translation", "service", "discussion"],
        );

        let mut pivot_metadata = HashMap::new();
        pivot_metadata.insert("source_session_id".to_string(), serde_json::json!("thread-a"));
        pivot_metadata.insert("source_turn_index".to_string(), serde_json::json!(2));
        let pivot = crate::engine::RecallResult {
            memory_id: second,
            content: "Then we optimized translation service latency.".to_string(),
            score: 120.0,
            match_integrity: 0.6,
            intersection_count: 2,
            recency_score: 1.0,
            reinforcement_score: 0.0,
            salience_score: 0.0,
            created_at: 0.0,
            metadata: pivot_metadata,
            explain: None,
        };

        let ordered = ordered_reconstruction_results(
            &ctx,
            &[
                ("language".to_string(), 1.0),
                ("translation".to_string(), 1.0),
                ("service".to_string(), 1.0),
                ("optimize".to_string(), 1.0),
            ],
            &[pivot],
            10,
            100,
            1,
            true,
        );

        let ids: Vec<MemoryId> = ordered.iter().map(|result| result.memory_id).collect();
        assert!(ids.contains(&first));
        assert!(ids.contains(&second));
        assert!(!ids.contains(&distractor));
        assert!(ordered
            .iter()
            .all(|result| result.metadata.contains_key("ordered_reconstruction")));
    }

    #[test]
    fn segment_link_requires_parent_and_chunk_idx() {
        assert_eq!(
            segment_link_from_cues(&[
                "parent:abc".to_string(),
                "chunk_idx:7".to_string(),
                "source_role:user".to_string(),
            ]),
            Some(("parent:abc".to_string(), 7))
        );
        assert_eq!(
            segment_link_from_cues(&["parent:abc".to_string()]),
            None
        );
    }

    #[test]
    fn stitched_chunk_join_removes_overlapped_sentences() {
        let joined = join_stitched_chunk_contents(&[
            "First sentence. Shared sentence.".to_string(),
            "Shared sentence. Final sentence.".to_string(),
        ]);

        assert_eq!(joined, "First sentence. Shared sentence. Final sentence.");
    }

    #[test]
    fn source_answer_projection_requires_assistant_answer_language() {
        let source_answer_intent = crate::facets::QueryIntent {
            labels: vec!["source_answer".to_string()],
            ..Default::default()
        };
        assert!(!source_answer_projection_requested(
            Some(&source_answer_intent),
            Some("What did I buy last week?")
        ));
        assert!(source_answer_projection_requested(
            Some(&source_answer_intent),
            Some("What was in the assistant answer?")
        ));

        let assistant_intent = crate::facets::QueryIntent {
            labels: vec!["source_assistant".to_string()],
            ..Default::default()
        };
        assert!(source_answer_projection_requested(
            Some(&assistant_intent),
            Some("Can you remind me?")
        ));
    }

    #[test]
    fn user_context_projection_targets_advice_without_source_intents() {
        assert!(user_context_projection_requested(
            None,
            Some("I've been having trouble with battery life. Any tips?")
        ));

        let recommendation_intent = crate::facets::QueryIntent {
            labels: vec!["recommendation".to_string()],
            ..Default::default()
        };
        assert!(user_context_projection_requested(
            Some(&recommendation_intent),
            Some("Can you recommend something for me?")
        ));
        assert!(!user_context_projection_requested(
            Some(&recommendation_intent),
            Some("Can you suggest a hotel for my upcoming trip to Miami?")
        ));

        for label in ["source_answer", "source_assistant", "source_user", "decision_selection"] {
            let intent = crate::facets::QueryIntent {
                labels: vec![label.to_string()],
                ..Default::default()
            };
            assert!(
                !user_context_projection_requested(Some(&intent), Some("Any tips?")),
                "source-specific query should not request user context projection for {label}"
            );
        }
    }

    #[test]
    fn user_context_projection_anchors_require_specific_context() {
        let phone_accessory_anchors =
            projection_anchor_cues(Some("Can you suggest some useful accessories for my phone?"));
        assert_eq!(
            phone_accessory_anchors,
            vec!["accessory".to_string(), "phone".to_string()]
        );

        let media_recommendation_anchors =
            projection_anchor_cues(Some("Can you recommend a show or movie for me to watch tonight?"));
        assert!(media_recommendation_anchors.is_empty());

        let troubleshooting_anchors = projection_anchor_cues(Some(
            "I've been having trouble with the battery life on my phone lately. Any tips?",
        ));
        assert!(troubleshooting_anchors.contains(&"battery".to_string()));
        assert!(troubleshooting_anchors.contains(&"life".to_string()));
        assert!(troubleshooting_anchors.contains(&"phone".to_string()));

        let navigation_anchors = projection_anchor_cues(Some(
            "I'm a bit anxious about getting around Tokyo. Do you have any helpful tips?",
        ));
        assert_eq!(
            navigation_anchors,
            vec!["anxious".to_string(), "tokyo".to_string()]
        );

        let relevant = "assistant: A power bank can help with phone battery life while traveling.";
        let incidental = "assistant: You could schedule a phone call during the morning.";

        assert!(projection_anchor_match_count(relevant, &troubleshooting_anchors) >= 2);
        assert!(projection_anchor_match_count(incidental, &troubleshooting_anchors) < 2);
        assert!(!projection_pivot_matches_context(
            "assistant: A camera bag can complement your Sony setup.",
            4,
            &phone_accessory_anchors,
            true
        ));
        assert!(!projection_pivot_matches_context(
            "assistant: A camera bag can complement your Sony setup.",
            3,
            &phone_accessory_anchors,
            true
        ));
        assert!(projection_pivot_matches_context(
            "assistant: A phone case is a useful accessory for your phone setup.",
            2,
            &phone_accessory_anchors,
            true
        ));
        assert!(!projection_pivot_matches_context(
            "assistant: A camera bag can complement your Sony setup.",
            4,
            &phone_accessory_anchors,
            false
        ));

        let vague_interest_intent = crate::facets::QueryIntent {
            labels: vec!["vague_interest_recommendation".to_string()],
            ..Default::default()
        };
        assert!(suppress_user_context_projection_for_intent(Some(
            &vague_interest_intent
        )));
    }

    #[test]
    fn standing_instruction_projection_is_cuepack_intent_gated() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "standing_instruction_test".to_string(),
        );

        let instruction_id = ctx.main.add_memory(
            "Always provide fallback strategies when I ask about error handling in API services."
                .to_string(),
            vec!["api".to_string(), "error_handling".to_string()],
            None,
            MainStats::default(),
            false,
        );

        assert!(standing_instruction_projection_cues(
            &ctx,
            None,
            Some("What are some ways I can manage problems that come up when my API calls fail?")
        )
        .cues
        .is_empty());

        let intent = crate::facets::QueryIntent {
            labels: vec!["instruction_applicable".to_string()],
            ..Default::default()
        };
        let projection = standing_instruction_projection_cues(
            &ctx,
            Some(&intent),
            Some("What are some ways I can manage problems that come up when my API calls fail?"),
        );

        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "type:standing_instruction"));
        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "instruction_trigger:api"));

        let projection_results = ctx.main.recall_weighted(
            projection.cues.clone(),
            10,
            false,
            None,
            1,
            false,
            true,
            None,
            None,
        );
        let mut all_results = Vec::new();
        merge_standing_instruction_projection_results(
            &ctx,
            &mut all_results,
            projection_results,
            &projection.anchors,
        );

        let projected = all_results
            .iter()
            .find(|result| result.memory_id == instruction_id)
            .expect("standing instruction should be projected");
        assert!(projected
            .metadata
            .contains_key("standing_instruction_projection"));
    }

    #[test]
    fn standing_instruction_projection_uses_morphological_anchor_variants() {
        let anchors =
            standing_instruction_projection_anchors(Some("How do I implement a login feature?"));
        assert!(anchors.contains(&"implement".to_string()));
        assert!(anchors.contains(&"implementation".to_string()));

        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "standing_instruction_morphology_test".to_string(),
        );

        ctx.main.add_memory(
            "Always format code snippets with syntax highlighting when I ask about implementation details."
                .to_string(),
            Vec::new(),
            None,
            MainStats::default(),
            false,
        );

        let intent = crate::facets::QueryIntent {
            labels: vec!["instruction_applicable".to_string()],
            ..Default::default()
        };
        let projection = standing_instruction_projection_cues(
            &ctx,
            Some(&intent),
            Some("How do I implement a login feature?"),
        );

        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "instruction_trigger:implementation"));
    }

    #[test]
    fn standing_instruction_projection_maps_chance_to_probability_anchor() {
        let anchors = standing_instruction_projection_anchors(Some(
            "How do I calculate the chance of drawing a red card from a standard deck?",
        ));
        assert!(anchors.contains(&"chance".to_string()));
        assert!(anchors.contains(&"probability".to_string()));

        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "standing_instruction_probability_test".to_string(),
        );

        ctx.main.add_memory(
            "Always provide step-by-step explanations with concrete examples when I ask about probability concepts."
                .to_string(),
            Vec::new(),
            None,
            MainStats::default(),
            false,
        );

        let intent = crate::facets::QueryIntent {
            labels: vec!["instruction_applicable".to_string()],
            ..Default::default()
        };
        let projection = standing_instruction_projection_cues(
            &ctx,
            Some(&intent),
            Some("How do I calculate the chance of drawing a red card from a standard deck?"),
        );

        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "instruction_trigger:probability"));
    }

    #[test]
    fn preference_projection_is_cuepack_intent_gated() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "preference_projection_test".to_string(),
        );

        let memory_id = ctx.main.add_memory(
            "I prefer geometric vector methods over purely trigonometric formulas for clarity, so can you explain how to use vector algebra to calculate geodesic length between two points on a sphere?".to_string(),
            vec![
                "sphere".to_string(),
                "two_point".to_string(),
                "vector".to_string(),
                "geodesic".to_string(),
            ],
            None,
            MainStats::default(),
            false,
        );

        let query = "Can you show me how to find the shortest path between two points on a sphere?";
        assert!(preference_projection_cues(&ctx, None, Some(query))
            .cues
            .is_empty());

        let intent = crate::facets::QueryIntent {
            labels: vec!["preference_applicable".to_string()],
            ..Default::default()
        };
        let projection = preference_projection_cues(&ctx, Some(&intent), Some(query));

        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "type:preference"));
        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "sphere" || cue == "two_point"));

        let projection_results = ctx.main.recall_weighted(
            projection.cues.clone(),
            10,
            false,
            None,
            1,
            false,
            true,
            None,
            None,
        );
        let mut all_results = Vec::new();
        merge_preference_projection_results(
            &ctx,
            &mut all_results,
            projection_results,
            &projection.anchors,
        );

        let projected = all_results
            .iter()
            .find(|result| result.memory_id == memory_id)
            .expect("matching preference should be projected");
        assert!(projected.metadata.contains_key("preference_projection"));
    }

    #[test]
    fn user_context_projection_merge_marks_and_updates_results() {
        let mut existing = recall_result(0.2, 1);
        existing.memory_id = 10;
        existing.score = 10.0;

        let mut projected = recall_result(0.8, 3);
        projected.memory_id = 10;
        projected.score = 50.0;
        projected
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));

        let mut all_results = vec![existing];
        merge_user_context_projection_results(&mut all_results, vec![projected]);

        assert_eq!(all_results.len(), 1);
        assert_eq!(all_results[0].score, 50.0);
        assert!(all_results[0]
            .metadata
            .contains_key("user_context_projection"));
    }

    #[test]
    fn source_prompt_projection_filters_short_scaffold_prompts() {
        let mut scaffold = recall_result(0.1, 1);
        scaffold.memory_id = 20;
        scaffold.content = "user: Write another scene".to_string();
        scaffold.score = 5000.0;
        scaffold
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));

        let mut source = recall_result(0.8, 9);
        source.memory_id = 21;
        source.content =
            "user: Write a comedy movie scene. Andy wears an untidy stained white shirt."
                .to_string();
        source.score = 600.0;
        source
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));

        let mut results = Vec::new();
        merge_source_prompt_projection_results(
            &mut results,
            vec![scaffold, source],
            Some("what was Andy wearing in the script you wrote for the comedy movie scene?"),
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_id, 21);
        assert!(results[0]
            .metadata
            .contains_key("source_prompt_projection"));
        assert!(results[0].score > 600.0);
    }

    #[test]
    fn user_context_adjacency_prefers_nearest_prior_user_turn() {
        let mut expected = recall_result(0.2, 1);
        expected.memory_id = 30;
        expected.score = 100.0;
        expected.created_at = 1.0;
        expected
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));
        expected.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-3"),
        );
        expected
            .metadata
            .insert("user_context_projection".to_string(), serde_json::json!(true));

        let mut pivot = recall_result(0.6, 2);
        pivot.memory_id = 31;
        pivot.score = 500.0;
        pivot.created_at = 2.0;
        pivot
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));
        pivot.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-3"),
        );

        let mut later_user = recall_result(0.2, 1);
        later_user.memory_id = 32;
        later_user.score = 900.0;
        later_user.created_at = 3.0;
        later_user
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));
        later_user.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-3"),
        );
        later_user
            .metadata
            .insert("user_context_projection".to_string(), serde_json::json!(true));

        let mut results = vec![expected, pivot, later_user];
        apply_user_context_adjacency_preference(&mut results, None, Some("Any tips?"));

        assert!(results[0].score > results[2].score);
        assert!(results[0]
            .metadata
            .contains_key("user_context_adjacency_boost"));
        assert!(!results[2]
            .metadata
            .contains_key("user_context_adjacency_boost"));
    }

    #[test]
    fn user_context_adjacency_considers_bounded_multiple_pivots() {
        fn with_source(
            mut result: crate::engine::RecallResult,
            role: &str,
            session: &str,
            projected: bool,
        ) -> crate::engine::RecallResult {
            result
                .metadata
                .insert("source_role".to_string(), serde_json::json!(role));
            result.metadata.insert(
                "source_session_id".to_string(),
                serde_json::json!(session),
            );
            if projected {
                result
                    .metadata
                    .insert("user_context_projection".to_string(), serde_json::json!(true));
            }
            result
        }

        let mut first_user = recall_result(0.2, 1);
        first_user.memory_id = 40;
        first_user.score = 90.0;
        first_user.created_at = 1.0;

        let mut first_pivot = recall_result(0.6, 2);
        first_pivot.memory_id = 41;
        first_pivot.score = 900.0;
        first_pivot.created_at = 2.0;

        let mut second_user = recall_result(0.2, 1);
        second_user.memory_id = 42;
        second_user.score = 80.0;
        second_user.created_at = 3.0;

        let mut second_pivot = recall_result(0.6, 2);
        second_pivot.memory_id = 43;
        second_pivot.score = 800.0;
        second_pivot.created_at = 4.0;

        let mut expected = recall_result(0.2, 1);
        expected.memory_id = 44;
        expected.score = 70.0;
        expected.created_at = 5.0;

        let mut expected_pivot = recall_result(0.6, 2);
        expected_pivot.memory_id = 45;
        expected_pivot.score = 500.0;
        expected_pivot.created_at = 6.0;

        let mut results = vec![
            with_source(first_user, "user", "conversation-7", true),
            with_source(first_pivot, "assistant", "conversation-7", false),
            with_source(second_user, "user", "conversation-7", true),
            with_source(second_pivot, "assistant", "conversation-7", false),
            with_source(expected, "user", "conversation-7", true),
            with_source(expected_pivot, "assistant", "conversation-7", false),
        ];

        apply_user_context_adjacency_preference(
            &mut results,
            None,
            Some("Any helpful tips?"),
        );

        assert!(results[4]
            .metadata
            .contains_key("user_context_adjacency_boost"));
        assert!(results[4].score > 70.0);
    }

    #[test]
    fn source_session_cue_is_derived_from_structured_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("Answer ShareGPT hA7AkP3 0"),
        );

        assert_eq!(
            source_session_cue_from_metadata(&metadata).as_deref(),
            Some("source_session:answer_sharegpt_ha7akp3_0")
        );
    }

    #[test]
    fn list_answer_detection_covers_ordinals_without_topic_words() {
        assert!(query_wants_list_answer(Some(
            "What was the 7th item you listed?"
        )));
        assert!(query_wants_list_answer(Some(
            "Remind me what was in the list you provided."
        )));
        assert!(!query_wants_list_answer(Some(
            "What did I purchase yesterday?"
        )));
    }

    #[test]
    fn source_role_preference_demotes_structured_role_mismatches() {
        let mut user_result = recall_result(1.0, 3);
        user_result.score = 100.0;
        user_result
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));

        let mut assistant_result = recall_result(1.0, 3);
        assistant_result.memory_id = 50;
        assistant_result.score = 80.0;
        assistant_result
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));

        let intent = crate::facets::QueryIntent {
            labels: vec!["source_assistant".to_string()],
            ..Default::default()
        };
        let mut results = vec![user_result, assistant_result];
        apply_source_role_preference(&mut results, Some(&intent));

        assert!(results[0].score < results[1].score);
        assert_eq!(results[1].score, 80.0);
    }

    #[test]
    fn source_answer_adjacency_prefers_immediate_assistant_reply() {
        let mut pivot = recall_result(1.0, 5);
        pivot.score = 1000.0;
        pivot.created_at = 1.0;
        pivot.metadata
            .insert("source_role".to_string(), serde_json::json!("user"));
        pivot.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-1"),
        );

        let mut immediate_answer = recall_result(1.0, 2);
        immediate_answer.memory_id = 60;
        immediate_answer.score = 300.0;
        immediate_answer.created_at = 2.0;
        immediate_answer
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));
        immediate_answer.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-1"),
        );

        let mut later_answer = recall_result(1.0, 8);
        later_answer.memory_id = 61;
        later_answer.score = 1000.0;
        later_answer.created_at = 6.0;
        later_answer
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));
        later_answer.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-1"),
        );

        let intent = crate::facets::QueryIntent {
            labels: vec!["source_answer".to_string(), "source_assistant".to_string()],
            ..Default::default()
        };
        let mut results = vec![pivot, immediate_answer, later_answer];
        apply_source_answer_adjacency_preference(&mut results, Some(&intent));

        assert!(results[1].score > results[2].score);
        assert!(results[1]
            .metadata
            .contains_key("source_answer_adjacency_boost"));
    }

    #[test]
    fn decision_adjacency_prefers_selection_after_proposal() {
        let mut proposal = recall_result(1.0, 6);
        proposal.score = 3000.0;
        proposal.created_at = 1.0;
        proposal.content =
            "assistant: Here are some potential names: Radik, Nucleus, Fissionator.".to_string();
        proposal
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));
        proposal.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-2"),
        );

        let mut selected = recall_result(1.0, 1);
        selected.memory_id = 70;
        selected.score = 300.0;
        selected.created_at = 2.0;
        selected.content = "user: Fissionator is a really cool one.".to_string();
        selected
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));
        selected.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-2"),
        );

        let mut later = recall_result(1.0, 4);
        later.memory_id = 71;
        later.score = 900.0;
        later.created_at = 5.0;
        later.content = "assistant: Fissionator could have radioactive attacks.".to_string();
        later
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));
        later.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-2"),
        );

        let intent = crate::facets::QueryIntent {
            labels: vec![
                "decision_selection".to_string(),
                "naming_decision".to_string(),
            ],
            ..Default::default()
        };
        let mut results = vec![proposal, selected, later];
        apply_decision_adjacency_preference(&mut results, Some(&intent));

        assert!(results[1].score > results[0].score);
        assert!(results[1].score > results[2].score);
        assert!(results[1]
            .metadata
            .contains_key("decision_adjacency_boost"));
    }
}
