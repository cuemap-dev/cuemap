use crate::auth::AuthConfig;
use crate::multi_tenant::{MultiTenantEngine, validate_project_id};
use crate::projects::ProjectContext;
use crate::normalization::normalize_cue;
use crate::taxonomy::validate_cues;
use crate::jobs::{Job, JobQueue};
use axum::{
    extract::{Path, State},
    http::{StatusCode, HeaderMap},
    middleware,
    response::IntoResponse,
    routing::{get, patch, post, delete},
    Json, Router,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct AddMemoryRequest {
    content: String,
    cues: Vec<String>,
    #[serde(default)]
    metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub disable_temporal_chunking: bool,
    #[serde(default)]
    pub async_ingest: bool,
}

#[derive(Debug, Serialize)]
pub struct AddMemoryResponse {
    id: String,
    status: String,
    cues: Vec<String>,
    latency_ms: f64,
}

#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    #[serde(default)]
    cues: Vec<String>,
    #[serde(default)]
    query_text: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default = "default_auto_reinforce")]
    auto_reinforce: bool,
    #[serde(default)]
    projects: Option<Vec<String>>,
    #[serde(default)]
    min_intersection: Option<usize>,
    #[serde(default)]
    pub explain: bool,
    #[serde(default)]
    pub disable_pattern_completion: bool,
    #[serde(default)]
    pub disable_salience_bias: bool,
    #[serde(default)]
    pub disable_systems_consolidation: bool,
}

#[derive(Debug, Deserialize)]
pub struct RecallGroundedRequest {
    pub query_text: String,
    #[serde(default = "default_token_budget")]
    pub token_budget: u32,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub projects: Option<Vec<String>>,
    #[serde(default = "default_true")]
    pub auto_reinforce: bool,  // Default to true - memories should learn from usage
    #[serde(default)]
    pub disable_pattern_completion: bool,
    #[serde(default)]
    pub disable_salience_bias: bool,
    #[serde(default)]
    pub disable_systems_consolidation: bool,
    #[serde(default)]
    pub min_intersection: Option<usize>,
}

fn default_true() -> bool {
    true
}

fn default_token_budget() -> u32 {
    500
}

#[derive(Debug, Serialize)]
pub struct RecallGroundedResponse {
    pub verified_context: String,
    pub proof: crate::grounding::GroundingProof,
    pub engine_latency_ms: f64,
    pub signature: String,
}

fn default_auto_reinforce() -> bool {
    true
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Deserialize)]
pub struct ReinforceRequest {
    cues: Vec<String>,
}


#[derive(Debug, Deserialize)]
pub struct AddAliasRequest {
    pub from: String,
    pub to: String,
    pub weight: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct GetAliasRequest {
    pub cue: String,
}

#[derive(Debug, Deserialize)]
pub struct MergeAliasRequest {
    pub cues: Vec<String>,
    pub to: String,
}

#[derive(Debug, Serialize)]
pub struct AliasResponse {
    pub id: String,
    pub from: String,
    pub to: String,
    pub weight: f64,
}

/// Response for /lexicon/inspect/:cue endpoint
#[derive(Debug, Serialize)]
pub struct LexiconInspectResponse {
    pub cue: String,
    pub outgoing: Vec<LexiconEntry>,  // What this token maps to
    pub incoming: Vec<LexiconEntry>,  // Other tokens that map to the same canonical
}

#[derive(Debug, Serialize)]
pub struct LexiconEntry {
    pub memory_id: String,
    pub content: String,      // The canonical cue
    pub token: String,        // The raw token (from cues)
    pub reinforcement_score: f64,
    pub created_at: f64,
    #[serde(default)]
    pub affected_memories_count: usize,  // Main memories that have this token but not the canonical
}

/// Request for POST /lexicon/wire - manually wire a token to a canonical cue
#[derive(Debug, Deserialize)]
pub struct WireLexiconRequest {
    pub token: String,
    pub canonical: String,
}

#[derive(Debug, Serialize)]
pub struct ReinforceResponse {
    status: String,
    memory_id: String,
}

#[derive(Clone)]
pub enum EngineState {
    SingleTenant { 
        project: Arc<ProjectContext>, 
        read_only: bool,
        job_queue: Arc<JobQueue> 
    },
    MultiTenant { 
        mt_engine: Arc<MultiTenantEngine>, 
        read_only: bool,
        job_queue: Arc<JobQueue>
    },
}

/// Routes for single-tenant mode
pub fn routes(project: std::sync::Arc<ProjectContext>, job_queue: Arc<JobQueue>, auth_config: AuthConfig, read_only: bool) -> Router {
    let mut router = Router::new()
        .route("/", get(root))
        .route("/memories", post(add_memory))
        .route("/recall", post(recall))
        .route("/memories/:id/reinforce", patch(reinforce_memory))
        .route("/memories/:id", get(get_memory).delete(delete_memory))
        .route("/stats", get(get_stats))
        .route("/recall/grounded", post(recall_grounded))
        .route("/aliases", post(add_alias).get(get_aliases))
        .route("/aliases/merge", post(merge_aliases))
        .route("/graph", get(get_graph))
        .route("/lexicon/inspect/:cue", get(lexicon_inspect))
        .route("/lexicon/entry/:id", delete(lexicon_delete))
        .route("/lexicon/graph", get(lexicon_graph))
        .route("/lexicon/wire", post(lexicon_wire))
        .route("/lexicon/synonyms/:cue", get(lexicon_synonyms))
        .fallback(crate::web::handler)
        .with_state(EngineState::SingleTenant { 
            project,
            read_only,
            job_queue 
        });
    
    // Add auth middleware if enabled
    if auth_config.is_enabled() {
        router = router.layer(middleware::from_fn_with_state(auth_config, crate::auth::auth_middleware));
    }
    
    router
}

/// Routes for multi-tenant mode
pub fn routes_with_mt_engine(mt_engine: Arc<MultiTenantEngine>, job_queue: Arc<JobQueue>, auth_config: AuthConfig, read_only: bool) -> Router {
    let mut router = Router::new()
        .route("/", get(root))
        .route("/memories", post(add_memory_mt))
        .route("/recall", post(recall_mt))
        .route("/memories/:id/reinforce", patch(reinforce_memory_mt))
        .route("/memories/:id", get(get_memory_mt).delete(delete_memory_mt))
        .route("/stats", get(get_stats_mt))
        .route("/projects", get(list_projects))
        .route("/recall/grounded", post(recall_grounded_mt))
        .route("/projects/:id", delete(delete_project))
        .route("/aliases", post(add_alias_mt).get(get_aliases_mt))
        .route("/aliases/merge", post(merge_aliases_mt))
        .route("/graph", get(get_graph))
        .route("/lexicon/inspect/:cue", get(lexicon_inspect_mt))
        .route("/lexicon/entry/:id", delete(lexicon_delete_mt))
        .route("/lexicon/graph", get(lexicon_graph_mt))
        .route("/lexicon/wire", post(lexicon_wire_mt))
        .route("/lexicon/synonyms/:cue", get(lexicon_synonyms_mt))
        .fallback(crate::web::handler)
        .with_state(EngineState::MultiTenant { 
            mt_engine,
            read_only,
            job_queue 
        });
    
    // Add auth middleware if enabled
    if auth_config.is_enabled() {
        router = router.layer(middleware::from_fn_with_state(auth_config, crate::auth::auth_middleware));
    }
    
    router
}

async fn root() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "CueMap Rust Engine",
        "version": "0.2.1",
        "description": "High-performance Temporal-Associative Memory Store"
    }))
}

async fn add_memory(
    State(state): State<EngineState>,
    Json(req): Json<AddMemoryRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, read_only, job_queue } = state {
        use std::time::Instant;
        let start = Instant::now();

        // Check if read-only
        if read_only {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({
                    "error": "Read-only mode: modifications are not allowed"
                })),
            );
        }
        
        // 1. Cue Preparation Strategy
        // If cues are empty, bootstrap from content
        let mut initial_cues = req.cues;
        if initial_cues.is_empty() {
            // Bootstrap from content tokens
            let tokens = crate::nl::tokenize_to_cues(&req.content);
            initial_cues.extend(tokens);
        }

        // 2. Synchronous Semantic Expansion (WordNet)
        // We do this here to make memory instantly recallable with synonyms
        // Skip if async_ingest is requested
        let wordnet_cues = if !req.async_ingest {
            project.semantic_engine.expand_wordnet(&req.content, &initial_cues, 0.65, 3)
        } else {
            Vec::new()
        };
        initial_cues.extend(wordnet_cues);

        // 3. Normalize cues
        let mut normalized_cues = Vec::new();
        for cue in initial_cues {
            let (normalized, _) = normalize_cue(&cue, &project.normalization);
            normalized_cues.push(normalized);
        }
        
        // 2. Validate cues
        let report = validate_cues(normalized_cues, &project.taxonomy);
        
        let memory_id = project.main.add_memory(req.content.clone(), report.accepted.clone(), req.metadata, req.disable_temporal_chunking);
        
        // Enqueue background jobs
        job_queue.enqueue(Job::TrainLexiconFromMemory {
            project_id: "default".to_string(), 
            memory_id: memory_id.clone()
        }).await;
        
        job_queue.enqueue(Job::ProposeCues {
            project_id: "default".to_string(),
            memory_id: memory_id.clone(),
            content: req.content,
        }).await;
        
        job_queue.enqueue(Job::UpdateGraph {
            project_id: "default".to_string(),
            memory_id: memory_id.clone(),
        }).await;
        
        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;

        (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": memory_id,
                "status": "stored",
                "cues": report.accepted,
                "rejected_cues": report.rejected,
                "latency_ms": latency_ms
            })),
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "id": "",
                "status": "error"
            })),
        )
    }
}

async fn recall(
    State(state): State<EngineState>,
    Json(req): Json<RecallRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use std::time::Instant;
    
    if let EngineState::SingleTenant { ref project, ref job_queue, .. } = state {
        let start = Instant::now();
        
        // Collect cues from request
        let mut cues_to_process = req.cues.clone();
        
        // Resolve cues from text if present
        let t_lex = Instant::now();
        let mut lexicon_memory_ids: Vec<String> = Vec::new();
        if let Some(ref text) = req.query_text {
            // 1. Lexicon Recall (High Confidence)
            let (resolved, lex_mids) = project.resolve_cues_from_text(&text);
            cues_to_process.extend(resolved);
            lexicon_memory_ids = lex_mids;

            // 2. Raw Token Fallback (For new/untrained words)
            // This ensures we don't drop words like "beer" just because they aren't in the lexicon yet
            let tokens = crate::nl::tokenize_to_cues(&text);
            for token in tokens {
                if !cues_to_process.contains(&token) {
                    cues_to_process.push(token);
                }
            }
        }
        let lex_ms = t_lex.elapsed().as_secs_f64() * 1000.0;

        // Normalize query cues
        let t_norm = Instant::now();
        let mut normalized_cues = Vec::new();
        for cue in &cues_to_process {
            let (normalized, _) = normalize_cue(cue, &project.normalization);
            normalized_cues.push(normalized);
        }
        let norm_ms = t_norm.elapsed().as_secs_f64() * 1000.0;
        
        // Expand aliases (only for original tokens)
        let t_expand = Instant::now();
        let original_tokens = if let Some(ref text) = req.query_text {
            crate::nl::tokenize_to_cues(text)
        } else {
            req.cues.clone()
        };
        let expanded_cues = project.expand_query_cues(normalized_cues, &original_tokens);
        let expand_ms = t_expand.elapsed().as_secs_f64() * 1000.0;
        
        let t_search = Instant::now();
        // Always pass auto_reinforce=false to avoid blocking, we'll do it async
        let results = project.main.recall_weighted(
            expanded_cues.clone(), 
            req.limit, 
            false, // Never block on reinforcement
            req.min_intersection,
            req.explain,
            req.disable_pattern_completion,
            req.disable_salience_bias,
            req.disable_systems_consolidation
        );
        let search_ms = t_search.elapsed().as_secs_f64() * 1000.0;
        
        let elapsed = start.elapsed();
        let engine_latency_ms = elapsed.as_secs_f64() * 1000.0;
        
        // Debug log timing breakdown
        tracing::info!(
            "Recall breakdown: lex={:.2}ms norm={:.2}ms expand={:.2}ms search={:.2}ms | cues={} expanded={}",
            lex_ms, norm_ms, expand_ms, search_ms, cues_to_process.len(), expanded_cues.len()
        );
        
        // Async reinforcement via background job (doesn't block response)
        if req.auto_reinforce && !results.is_empty() {
            let memory_ids: Vec<String> = results.iter().map(|r| r.memory_id.clone()).collect();
            let cues: Vec<String> = expanded_cues.iter().map(|(c, _)| c.clone()).collect();
            job_queue.enqueue(crate::jobs::Job::ReinforceMemories {
                project_id: "default".to_string(),
                memory_ids,
                cues,
            }).await;
        }
        
        // Reinforce Lexicon memories (async)
        if req.auto_reinforce && !lexicon_memory_ids.is_empty() {
            let tokens = if let Some(ref text) = req.query_text {
                crate::nl::tokenize_to_cues(text)
            } else {
                Vec::new()
            };
            job_queue.enqueue(crate::jobs::Job::ReinforceLexicon {
                project_id: "default".to_string(),
                memory_ids: lexicon_memory_ids,
                cues: tokens,
            }).await;
        }
        
        // Add query explanation if requested
        if req.explain {
            let explanation = serde_json::json!({
                "normalized_query": cues_to_process,
                "expanded_cues": expanded_cues
            });
            
            return (StatusCode::OK, Json(serde_json::json!({ 
                "results": results,
                "engine_latency": engine_latency_ms,
                "explain": explanation
            })));
        }
        
        (StatusCode::OK, Json(serde_json::json!({ 
            "results": results,
            "engine_latency": engine_latency_ms
        })))
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Invalid state"})),
        )
    }
}

async fn reinforce_memory(
    State(state): State<EngineState>,
    Path(memory_id): Path<String>,
    Json(req): Json<ReinforceRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, read_only, .. } = state {
        // Check if read-only
        if read_only {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({
                    "error": "Read-only mode: modifications are not allowed"
                })),
            );
        }
        
        // Normalize cues
        let mut normalized_cues = Vec::new();
        for cue in req.cues {
            let (normalized, _) = normalize_cue(&cue, &project.normalization);
            normalized_cues.push(normalized);
        }
        
        let success = project.main.reinforce_memory(&memory_id, normalized_cues);
        
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
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "status": "error",
                "memory_id": ""
            })),
        )
    }
}

async fn get_memory(
    State(state): State<EngineState>,
    Path(memory_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, .. } = state {
        match project.main.get_memory(&memory_id) {
            Some(memory) => (StatusCode::OK, Json(serde_json::json!(memory))),
            None => (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Memory not found"})),
            ),
        }
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Invalid state"})),
        )
    }
}

/// GDPR-compliant delete: permanently removes a memory from the main store
async fn delete_memory(
    State(state): State<EngineState>,
    Path(memory_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, read_only, .. } = state {
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
        }
        
        let deleted = project.main.delete_memory(&memory_id);
        if deleted {
            (StatusCode::OK, Json(serde_json::json!({
                "status": "deleted",
                "memory_id": memory_id
            })))
        } else {
            (StatusCode::NOT_FOUND, Json(serde_json::json!({
                "error": "Memory not found",
                "memory_id": memory_id
            })))
        }
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

async fn get_stats(State(state): State<EngineState>) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, .. } = state {
        let stats = project.main.get_stats();
        (StatusCode::OK, Json(serde_json::Value::Object(stats.into_iter().collect())))
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Invalid state"})),
        )
    }
}

async fn get_graph(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let limit = params.get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    match state {
        EngineState::SingleTenant { project, .. } => {
            let graph = project.main.get_graph_data(limit);
            (StatusCode::OK, Json(graph))
        },
        EngineState::MultiTenant { mt_engine, .. } => {
            let project_id = match extract_project_id(&headers) {
                Ok(id) => id,
                Err(_) => {
                    // Fallback: try to get from query param "project"
                    if let Some(p) = params.get("project") {
                        if validate_project_id(p) {
                            p.to_string()
                        } else {
                            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Invalid project ID format in query param"})));
                        }
                    } else {
                         return (
                            StatusCode::BAD_REQUEST, 
                            Json(serde_json::json!({"error": "Missing X-Project-ID header or project query param"}))
                        );
                    }
                }
            };

            let ctx = mt_engine.get_or_create_project(project_id.clone());
            let graph = ctx.main.get_graph_data(limit);
            (StatusCode::OK, Json(graph))
        }
    }
}

async fn recall_grounded(
    State(state): State<EngineState>,
    Json(req): Json<RecallGroundedRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use std::time::Instant;
    use crate::grounding::{GroundingEngine, create_grounding_proof};

    if let EngineState::SingleTenant { project, job_queue, .. } = state {
        let start = Instant::now();
        
        // 1. Standard CueMap Recall
        let (resolved, lexicon_memory_ids) = project.resolve_cues_from_text(&req.query_text);
        let mut normalized_cues = Vec::new();
        for cue in &resolved {
            let (normalized, _) = crate::normalization::normalize_cue(cue, &project.normalization);
            normalized_cues.push(normalized);
        }
        let original_tokens = crate::nl::tokenize_to_cues(&req.query_text);
        let expanded_cues = project.expand_query_cues(normalized_cues, &original_tokens);
        let results = project.main.recall_weighted(
            expanded_cues.clone(), 
            req.limit.max(20),
            false, // Never block on reinforcement, we do it async
            req.min_intersection,
            true,
            req.disable_pattern_completion,
            req.disable_salience_bias,
            req.disable_systems_consolidation
        );
        
        // 2. Apply Budgeting Logic
        let (selected, excluded, context_block) = GroundingEngine::select_memories(
            req.query_text.clone(),
            resolved.clone(),
            expanded_cues.clone(),
            results.clone(),
            req.token_budget,
        );
        
        // 3. Create Proof
        let proof = create_grounding_proof(
            uuid::Uuid::new_v4().to_string(),
            req.query_text.clone(),
            resolved,
            expanded_cues.clone(),
            req.token_budget,
            selected,
            excluded,
        );
        
        // 4. Sign Context
        let signer = crate::crypto::CryptoEngine::new();
        let signature = signer.sign(&context_block);

        let elapsed = start.elapsed();
        
        // 5. Async Reinforcement
        // Respects auto_reinforce flag
        if req.auto_reinforce && !results.is_empty() {
             let memory_ids: Vec<String> = results.iter().map(|r| r.memory_id.clone()).collect();
             let cues: Vec<String> = expanded_cues.iter().map(|(c, _)| c.clone()).collect();
             job_queue.enqueue(crate::jobs::Job::ReinforceMemories {
                 project_id: "default".to_string(),
                 memory_ids,
                 cues,
             }).await;
        }
        
        // 6. Reinforce Lexicon memories (async)
        if req.auto_reinforce && !lexicon_memory_ids.is_empty() {
            let tokens = crate::nl::tokenize_to_cues(&req.query_text);
            job_queue.enqueue(crate::jobs::Job::ReinforceLexicon {
                project_id: "default".to_string(),
                memory_ids: lexicon_memory_ids,
                cues: tokens,
            }).await;
        }

        (StatusCode::OK, Json(serde_json::json!({ 
            "verified_context": context_block,
            "proof": proof,
            "engine_latency_ms": elapsed.as_secs_f64() * 1000.0,
            "signature": signature
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

// Alias Handlers (Single Tenant)

async fn add_alias(
    State(state): State<EngineState>,
    Json(req): Json<AddAliasRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, read_only, .. } = state {
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only"})));
        }

        let alias_id = uuid::Uuid::new_v4().to_string();
        let content = serde_json::json!({
            "from": req.from,
            "to": req.to,
            "downweight": req.weight.unwrap_or(0.85),
            "status": "active",
            "reason": "manual"
        }).to_string();

        let cues = vec![
            "type:alias".to_string(),
            format!("from:{}", req.from),
            format!("to:{}", req.to),
            "status:active".to_string(),
            "reason:manual".to_string(),
        ];

        project.aliases.upsert_memory_with_id(
            alias_id.clone(),
            content,
            cues,
            None,
            false // no reinforce
        );

        (StatusCode::OK, Json(serde_json::json!({"id": alias_id, "status": "created"})))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

async fn get_aliases(
    State(state): State<EngineState>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, .. } = state {
        let cue = params.get("cue").cloned().unwrap_or_default();
        if cue.is_empty() {
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing 'cue' query param"})));
        }

        let query_cues = vec![
            "type:alias".to_string(), 
            format!("to:{}", cue),
            "status:active".to_string()
        ];
        
        let results = project.aliases.recall(query_cues, 50, false);
        let mut aliases = Vec::new();
        
        for res in results {
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&res.content) {
                let from_match = data.get("from").and_then(|v| v.as_str()).map(|v| v == cue).unwrap_or(false);
                let to_match = data.get("to").and_then(|v| v.as_str()).map(|v| v == cue).unwrap_or(false);
                
                if from_match || to_match {
                    aliases.push(data);
                }
            }
        }
        
        (StatusCode::OK, Json(serde_json::json!({"aliases": aliases})))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

/// Lexicon Surgeon: Inspect a cue in the Lexicon
/// Returns incoming (tokens that map to this cue) and outgoing (what this token maps to)
async fn lexicon_inspect(
    State(state): State<EngineState>,
    axum::extract::Path(cue): axum::extract::Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, .. } = state {
        let cue_lower = cue.to_lowercase();
        
        // Lexicon Structure (from jobs.rs):
        // - memory_id: "cue:{canonical}"
        // - content: "{canonical}" (the canonical cue)
        // - cues: ["{token}"] (the token(s) that trigger this canonical)
        
        // Helper: count main memories that have token in cues but NOT canonical
        let count_affected = |token: &str, canonical: &str| -> usize {
            let token_lower = token.to_lowercase();
            let canonical_lower = canonical.to_lowercase();
            let mut count = 0;
            for ref_multi in project.main.get_memories().iter() {
                let memory = ref_multi.value();
                let cues_lower: Vec<String> = memory.cues.iter().map(|c| c.to_lowercase()).collect();
                // Has token but not canonical = affected if unwired
                if cues_lower.contains(&token_lower) && !cues_lower.contains(&canonical_lower) {
                    count += 1;
                }
            }
            count
        };
        
        // 1. OUTGOING: What does this token trigger?
        // Search for memories where the searched cue appears in the `cues` array
        let outgoing_results = project.lexicon.recall_fast(vec![cue_lower.clone()], 100);
        let outgoing: Vec<LexiconEntry> = outgoing_results.iter().map(|r| {
            let token = r.metadata.get("cues")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str())
                .unwrap_or(&cue_lower)
                .to_string();
            
            let affected = count_affected(&token, &r.content);
            
            LexiconEntry {
                memory_id: r.memory_id.clone(),
                content: r.content.clone(),
                token,
                reinforcement_score: r.reinforcement_score,
                created_at: r.created_at,
                affected_memories_count: affected,
            }
        }).collect();
        
        // 2. INCOMING: What tokens map to this canonical cue?
        let mut incoming: Vec<LexiconEntry> = Vec::new();
        for ref_multi in project.lexicon.get_memories().iter() {
            let memory = ref_multi.value();
            if memory.content.to_lowercase() == cue_lower {
                for token in &memory.cues {
                    let affected = count_affected(token, &memory.content);
                    incoming.push(LexiconEntry {
                        memory_id: memory.id.clone(),
                        content: memory.content.clone(),
                        token: token.clone(),
                        reinforcement_score: memory.reinforcement_count as f64,
                        created_at: memory.created_at,
                        affected_memories_count: affected,
                    });
                }
            }
        }
        
        (StatusCode::OK, Json(serde_json::json!(LexiconInspectResponse {
            cue: cue_lower,
            outgoing,
            incoming,
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

/// Delete a lexicon entry (unwire a token mapping)
async fn lexicon_delete(
    State(state): State<EngineState>,
    axum::extract::Path(memory_id): axum::extract::Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, read_only, .. } = state {
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
        }
        
        let deleted = project.lexicon.delete_memory(&memory_id);
        if deleted {
            (StatusCode::OK, Json(serde_json::json!({
                "status": "deleted",
                "memory_id": memory_id
            })))
        } else {
            (StatusCode::NOT_FOUND, Json(serde_json::json!({
                "error": "Entry not found",
                "memory_id": memory_id
            })))
        }
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

/// Get full Lexicon as graph data for visualization
async fn lexicon_graph(
    State(state): State<EngineState>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, .. } = state {
        let mut nodes = Vec::new();
        let mut links = Vec::new();
        let mut token_to_canonical: HashMap<String, Vec<String>> = HashMap::new();
        
        // Collect all Lexicon entries (no limit - return everything)
        for ref_multi in project.lexicon.get_memories().iter() {
            let memory = ref_multi.value();
            
            // Each Lexicon entry: cues (tokens) -> content (canonical)
            let canonical = memory.content.clone();
            for token in &memory.cues {
                token_to_canonical.entry(token.clone())
                    .or_default()
                    .push(canonical.clone());
            }
        }
        
        // Build nodes for unique tokens and canonicals
        let mut node_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        
        for (token, canonicals) in &token_to_canonical {
            // Token node
            if !node_ids.contains(token) {
                nodes.push(serde_json::json!({
                    "id": token,
                    "label": token,
                    "group": "token"
                }));
                node_ids.insert(token.clone());
            }
            
            // Canonical nodes and links
            for canonical in canonicals {
                if !node_ids.contains(canonical) {
                    nodes.push(serde_json::json!({
                        "id": canonical,
                        "label": canonical,
                        "group": "canonical"
                    }));
                    node_ids.insert(canonical.clone());
                }
                
                // Link: token -> canonical
                if token != canonical {  // Don't link to self
                    links.push(serde_json::json!({
                        "source": token,
                        "target": canonical
                    }));
                }
            }
        }
        
        (StatusCode::OK, Json(serde_json::json!({
            "nodes": nodes,
            "links": links,
            "total_entries": nodes.len()
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

/// Manually wire a token to a canonical cue in the Lexicon
async fn lexicon_wire(
    State(state): State<EngineState>,
    Json(req): Json<WireLexiconRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, read_only, .. } = state {
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
        }
        
        let token = req.token.to_lowercase();
        let canonical = req.canonical.to_lowercase();
        let lex_id = format!("cue:{}", canonical);
        
        // Use upsert to create or reinforce the mapping
        project.lexicon.upsert_memory_with_id(
            lex_id.clone(),
            canonical.clone(),
            vec![token.clone()],
            None,
            false
        );
        
        (StatusCode::OK, Json(serde_json::json!({
            "status": "wired",
            "memory_id": lex_id,
            "token": token,
            "canonical": canonical
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

/// Get WordNet synonyms for a cue that are NOT already wired to it
async fn lexicon_synonyms(
    State(state): State<EngineState>,
    axum::extract::Path(cue): axum::extract::Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, .. } = state {
        let cue_lower = cue.to_lowercase();
        
        // 1. Get bidirectional neighbors (incoming + outgoing)
        let mut connected: std::collections::HashSet<String> = std::collections::HashSet::new();
        connected.insert(cue_lower.clone()); // Include self
        
        for ref_multi in project.lexicon.get_memories().iter() {
            let memory = ref_multi.value();
            let mem_canon = memory.content.to_lowercase();
            
            // Outgoing: If cue is a token here, the canonical is connected
            if memory.cues.iter().any(|c| c.to_lowercase() == cue_lower) {
                connected.insert(mem_canon.clone());
            }
            
            // Incoming: If cue is canonical, tokens are connected
            if mem_canon == cue_lower {
                for token in &memory.cues {
                    connected.insert(token.to_lowercase());
                }
            }
            
            // General Exclusion: If memory contains cue in any way, exclude all its participants
            // This is safer to avoid "already connected" suggestions
            if mem_canon == cue_lower || memory.cues.iter().any(|c| c.to_lowercase() == cue_lower) {
                 connected.insert(mem_canon.clone());
                 for t in &memory.cues { connected.insert(t.to_lowercase()); }
            }
        }
        
        // 2. Recursive WordNet Expansion (Depth 2)
        let mut candidates = std::collections::HashSet::new();
        
        // Layer 1
        let layer1 = project.semantic_engine.expand_wordnet(&cue_lower, &[cue_lower.clone()], 0.50, 50);
        for w1 in layer1 {
            candidates.insert(w1.clone());
            
            // Layer 2 (Expand results of Layer 1)
            let layer2 = project.semantic_engine.expand_wordnet(&w1, &[], 0.50, 20); // slightly smaller limit for layer 2
            for w2 in layer2 {
                candidates.insert(w2);
            }
        }
        
        // 3. Filter, Deduplicate, and Limit
        let mut suggestions: Vec<String> = candidates
            .into_iter()
            .filter(|s| !connected.contains(&s.to_lowercase()) && s.len() > 2) // Filter short noise
            .collect();
            
        suggestions.sort(); // Sorting helps stability
        suggestions.truncate(50); // Cap at 50 to avoid payload explosion
        
        // 4. Check existing in graph
        let mut existing_in_graph: Vec<String> = Vec::new();
        let mut new_suggestions: Vec<String> = Vec::new();
        
        for syn in &suggestions {
            let exists = project.lexicon.get_memories().iter().any(|ref_multi| {
                let memory = ref_multi.value();
                memory.content.to_lowercase() == syn.to_lowercase() ||
                memory.cues.iter().any(|c| c.to_lowercase() == syn.to_lowercase())
            });
            
            if exists {
                existing_in_graph.push(syn.clone());
            } else {
                new_suggestions.push(syn.clone());
            }
        }
        
        (StatusCode::OK, Json(serde_json::json!({
            "cue": cue_lower,
            "suggestions": suggestions,
            "existing_in_graph": existing_in_graph,
            "new_only": new_suggestions
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

async fn merge_aliases(
    State(state): State<EngineState>,
    Json(req): Json<MergeAliasRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::SingleTenant { project, read_only, .. } = state {
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only"})));
        }

        let mut created_ids = Vec::new();

        for from_cue in req.cues {
            let alias_id = uuid::Uuid::new_v4().to_string();
            let content = serde_json::json!({
                "from": from_cue,
                "to": req.to,
                "downweight": 1.0,
                "status": "active",
                "reason": "manual_merge"
            }).to_string();

            let cues = vec![
                "type:alias".to_string(),
                format!("from:{}", from_cue),
                format!("to:{}", req.to),
                "status:active".to_string(),
                "reason:manual_merge".to_string(),
            ];

            project.aliases.upsert_memory_with_id(
                alias_id.clone(),
                content,
                cues,
                None,
                false
            );
            created_ids.push(alias_id);
        }

        (StatusCode::OK, Json(serde_json::json!({
            "status": "merged", 
            "target": req.to, 
            "count": created_ids.len()
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

// Multi-tenant handlers
fn extract_project_id(headers: &HeaderMap) -> Result<String, (StatusCode, Json<serde_json::Value>)> {
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

async fn add_memory_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<AddMemoryRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    if let EngineState::MultiTenant { mt_engine, read_only, job_queue } = state {
        use std::time::Instant;
        let start = Instant::now();

        // Check if read-only
        if read_only {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({
                    "error": "Read-only mode: modifications are not allowed"
                })),
            );
        }
        
        let ctx = mt_engine.get_or_create_project(project_id.clone());
        
        // 1. Cue Preparation Strategy
        // If cues are empty, bootstrap from content
        let step1_start = Instant::now();
        let mut initial_cues = req.cues;
        if initial_cues.is_empty() {
             // Bootstrap from content tokens
             let tokens = crate::nl::tokenize_to_cues(&req.content);
             initial_cues.extend(tokens);
        }
        let t_tokenize = step1_start.elapsed().as_secs_f64() * 1000.0;
        
        // 2. Normalize cues
        let step3_start = Instant::now();
        let mut normalized_cues = Vec::new();
        for cue in initial_cues {
            let (normalized, _) = normalize_cue(&cue, &ctx.normalization);
            normalized_cues.push(normalized);
        }
        let t_normalize = step3_start.elapsed().as_secs_f64() * 1000.0;
        
        // 3. Validate cues
        let step4_start = Instant::now();
        let report = validate_cues(normalized_cues, &ctx.taxonomy);
        let _accepted_count = report.accepted.len();
        let t_validate = step4_start.elapsed().as_secs_f64() * 1000.0;
        
        let step5_start = Instant::now();
        let memory_id = ctx.main.add_memory(req.content.clone(), report.accepted.clone(), req.metadata, req.disable_temporal_chunking);
        let t_insert = step5_start.elapsed().as_secs_f64() * 1000.0;

        job_queue.enqueue(Job::TrainLexiconFromMemory {
            project_id: project_id.clone(), 
            memory_id: memory_id.clone()
        }).await;
        
        job_queue.enqueue(Job::ProposeCues {
            project_id: project_id.clone(),
            memory_id: memory_id.clone(),
            content: req.content,
        }).await;

        job_queue.enqueue(Job::UpdateGraph {
            project_id: project_id.clone(),
            memory_id: memory_id.clone(),
        }).await;
        
        tracing::info!(
            "POST /memories project={} cues={} id={} timings: tok={:.2}ms norm={:.2}ms val={:.2}ms ins={:.2}ms",
            project_id,
            report.accepted.len(),
            memory_id,
            t_tokenize, t_normalize, t_validate, t_insert
        );
        
        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;

        (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": memory_id,
                "status": "stored",
                "cues": report.accepted,
                "rejected_cues": report.rejected,
                "latency_ms": latency_ms,
                "_debug_timings": {
                    "tokenization": t_tokenize,
                    "normalization": t_normalize,
                    "validation": t_validate,
                    "insertion": t_insert
                }
            })),
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "id": "",
                "status": "error"
            })),
        )
    }
}

async fn recall_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<RecallRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use std::time::Instant;
    
    if let EngineState::MultiTenant { ref mt_engine, ref job_queue, .. } = state {
        // Cross-domain query if projects array is provided
        if let Some(projects) = req.projects {
            let start = Instant::now();
            
            // Query all projects in parallel using rayon
            let (all_results, reinforce_tasks): (Vec<serde_json::Value>, Vec<Option<(String, Vec<String>, Vec<String>)>>) = projects
                .par_iter()
                .map(|project_id| {
                    let ctx = mt_engine.get_or_create_project(project_id.clone());
                    
                    // Collect cues
                    let mut cues_to_process = req.cues.clone();
                    
                    let (original_tokens, _lexicon_mids) = if let Some(text) = &req.query_text {
                         let (resolved, lex_mids) = ctx.resolve_cues_from_text(text);
                         cues_to_process.extend(resolved);
                         (crate::nl::tokenize_to_cues(text), lex_mids)
                    } else {
                        (req.cues.clone(), Vec::new())
                    };
                    
                    // Normalize query cues
                    let mut normalized_cues = Vec::new();
                    for cue in &cues_to_process {
                        let (normalized, _) = normalize_cue(cue, &ctx.normalization);
                        normalized_cues.push(normalized);
                    }
                    
                    // Expand aliases (only for original tokens)
                    let expanded_cues = ctx.expand_query_cues(normalized_cues, &original_tokens);
                    let results = ctx.main.recall_weighted(
                        expanded_cues.clone(), 
                        req.limit, 
                        false,
                        req.min_intersection,
                        req.explain,
                        req.disable_pattern_completion,
                        req.disable_salience_bias,
                        req.disable_systems_consolidation
                    );
                    
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
                                "expanded_cues": expanded_cues
                            })
                        );
                    }
                    
                    // Collect reinforcement task
                    // Respects auto_reinforce flag
                    let task = if req.auto_reinforce && !results.is_empty() {
                         let memory_ids: Vec<String> = results.iter().map(|r| r.memory_id.clone()).collect();
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
                    job_queue.enqueue(crate::jobs::Job::ReinforceMemories {
                        project_id: pid,
                        memory_ids: mids,
                        cues,
                    }).await;
                }
            }
            
            let elapsed = start.elapsed();
            let total_results: usize = all_results.iter()
                .filter_map(|r| r.get("results").and_then(|res| res.as_array().map(|a| a.len())))
                .sum();
            
            let engine_latency_ms = elapsed.as_secs_f64() * 1000.0;
            
            tracing::info!(
                "POST /recall cross-domain projects={} cues={} results={} latency={:.2}ms",
                projects.len(),
                req.cues.len(),
                total_results,
                engine_latency_ms
            );
            
            return (StatusCode::OK, Json(serde_json::json!({ 
                "results": all_results,
                "engine_latency": engine_latency_ms
            })));
        }
        
        // Single project query using X-Project-ID header
        let project_id = match extract_project_id(&headers) {
            Ok(id) => id,
            Err(e) => return e,
        };
        
        let start = Instant::now();
        let ctx = mt_engine.get_or_create_project(project_id.clone());
        
        // Collect cues
        let mut cues_to_process = req.cues.clone();
        
        let t_lex = Instant::now();
        let mut lexicon_memory_ids: Vec<String> = Vec::new();
        if let Some(ref text) = req.query_text {
             // 1. Lexicon Recall
             let (resolved, lex_mids) = ctx.resolve_cues_from_text(text);
             cues_to_process.extend(resolved);
             lexicon_memory_ids = lex_mids;

             // 2. Raw Token Fallback
             let tokens = crate::nl::tokenize_to_cues(text);
             for token in tokens {
                 if !cues_to_process.contains(&token) {
                     cues_to_process.push(token);
                 }
             }
        }
        let lex_ms = t_lex.elapsed().as_secs_f64() * 1000.0;
        
        // Normalize query cues
        let t_norm = Instant::now();
        let mut normalized_cues = Vec::new();
        for cue in &cues_to_process {
            let (normalized, _) = normalize_cue(cue, &ctx.normalization);
            normalized_cues.push(normalized);
        }
        let norm_ms = t_norm.elapsed().as_secs_f64() * 1000.0;
        
        // Expand aliases (only for original tokens, not Lexicon synonyms)
        let t_expand = Instant::now();
        let original_tokens = if let Some(ref text) = req.query_text {
            crate::nl::tokenize_to_cues(text)
        } else {
            req.cues.clone()
        };
        let expanded_cues = ctx.expand_query_cues(normalized_cues, &original_tokens);
        let expand_ms = t_expand.elapsed().as_secs_f64() * 1000.0;
        
        let t_search = Instant::now();
        // Always pass auto_reinforce=false to avoid blocking, we'll do it async
        let results = ctx.main.recall_weighted(
            expanded_cues.clone(), 
            req.limit, 
            false, // Never block on reinforcement
            req.min_intersection,
            req.explain,
            req.disable_pattern_completion,
            req.disable_salience_bias,
            req.disable_systems_consolidation
        );
        let search_ms = t_search.elapsed().as_secs_f64() * 1000.0;
        
        let elapsed = start.elapsed();
        
        let engine_latency_ms = elapsed.as_secs_f64() * 1000.0;
        
        tracing::info!(
            "Recall breakdown: lex={:.2}ms norm={:.2}ms expand={:.2}ms search={:.2}ms | cues={} expanded={}",
            lex_ms, norm_ms, expand_ms, search_ms, cues_to_process.len(), expanded_cues.len()
        );
        
        tracing::info!(
            "POST /recall project={} cues={} results={} latency={:.2}ms",
            project_id,
            cues_to_process.len(),
            results.len(),
            engine_latency_ms
        );
        
        // Async reinforcement via background job (doesn't block response)
        if req.auto_reinforce && !results.is_empty() {
            let memory_ids: Vec<String> = results.iter().map(|r| r.memory_id.clone()).collect();
            let cues: Vec<String> = expanded_cues.iter().map(|(c, _)| c.clone()).collect();
            job_queue.enqueue(crate::jobs::Job::ReinforceMemories {
                project_id: project_id.clone(),
                memory_ids,
                cues,
            }).await;
        }
        
        // Reinforce Lexicon memories (async)
        if req.auto_reinforce && !lexicon_memory_ids.is_empty() {
            let tokens = if let Some(ref text) = req.query_text {
                crate::nl::tokenize_to_cues(text)
            } else {
                Vec::new()
            };
            job_queue.enqueue(crate::jobs::Job::ReinforceLexicon {
                project_id: project_id.clone(),
                memory_ids: lexicon_memory_ids,
                cues: tokens,
            }).await;
        }
        
        if req.explain {
            return (StatusCode::OK, Json(serde_json::json!({ 
                "results": results,
                "engine_latency": engine_latency_ms,
                "explain": {
                    "query_cues": cues_to_process,
                    "expanded_cues": expanded_cues
                }
            })));
        }
        
        (StatusCode::OK, Json(serde_json::json!({ 
            "results": results,
            "engine_latency": engine_latency_ms
        })))
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Invalid state"})),
        )
    }
}

async fn reinforce_memory_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Path(memory_id): Path<String>,
    Json(req): Json<ReinforceRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    if let EngineState::MultiTenant { mt_engine, .. } = state {
        let ctx = mt_engine.get_or_create_project(project_id);
        
        // Normalize cues
        let mut normalized_cues = Vec::new();
        for cue in req.cues {
            let (normalized, _) = normalize_cue(&cue, &ctx.normalization);
            normalized_cues.push(normalized);
        }
        
        let success = ctx.main.reinforce_memory(&memory_id, normalized_cues);
        
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
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "status": "error",
                "memory_id": ""
            })),
        )
    }
}

async fn get_memory_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Path(memory_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    if let EngineState::MultiTenant { mt_engine, .. } = state {
        let ctx = mt_engine.get_or_create_project(project_id);
        match ctx.main.get_memory(&memory_id) {
            Some(memory) => (StatusCode::OK, Json(serde_json::json!(memory))),
            None => (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Memory not found"})),
            ),
        }
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Invalid state"})),
        )
    }
}

/// GDPR-compliant delete (multi-tenant)
async fn delete_memory_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Path(memory_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    if let EngineState::MultiTenant { mt_engine, read_only, .. } = state {
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
        }
        
        let ctx = mt_engine.get_or_create_project(project_id);
        let deleted = ctx.main.delete_memory(&memory_id);
        if deleted {
            (StatusCode::OK, Json(serde_json::json!({
                "status": "deleted",
                "memory_id": memory_id
            })))
        } else {
            (StatusCode::NOT_FOUND, Json(serde_json::json!({
                "error": "Memory not found",
                "memory_id": memory_id
            })))
        }
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

async fn get_stats_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    if let EngineState::MultiTenant { mt_engine, .. } = state {
        let ctx = mt_engine.get_or_create_project(project_id);
        let stats = ctx.main.get_stats();
        (StatusCode::OK, Json(serde_json::Value::Object(stats.into_iter().collect())))
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Invalid state"})),
        )
    }
}

async fn recall_grounded_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<RecallGroundedRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use std::time::Instant;
    use crate::grounding::{GroundingEngine, create_grounding_proof};

    let project_id = if let Some(ref projects) = req.projects {
        projects.first().cloned().unwrap_or_else(|| {
             headers.get("X-Project-ID").and_then(|v| v.to_str().ok()).unwrap_or("default").to_string()
        })
    } else {
        match extract_project_id(&headers) {
            Ok(id) => id,
            Err(e) => return e,
        }
    };

    if let EngineState::MultiTenant { mt_engine, .. } = state {
        let start = Instant::now();
        let ctx = mt_engine.get_or_create_project(project_id);
        
        // 1. Standard CueMap Recall
        let (resolved, _lexicon_memory_ids) = ctx.resolve_cues_from_text(&req.query_text);
        let mut normalized_cues = Vec::new();
        for cue in &resolved {
            let (normalized, _) = crate::normalization::normalize_cue(cue, &ctx.normalization);
            normalized_cues.push(normalized);
        }
        let original_tokens = crate::nl::tokenize_to_cues(&req.query_text);
        let expanded_cues = ctx.expand_query_cues(normalized_cues, &original_tokens);
        
        let results = ctx.main.recall_weighted(
            expanded_cues.clone(), 
            req.limit.max(20),
            req.auto_reinforce, 
            req.min_intersection,
            true,
            req.disable_pattern_completion,
            req.disable_salience_bias,
            req.disable_systems_consolidation
        );
        
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
        
        (StatusCode::OK, Json(serde_json::json!({ 
            "verified_context": context_block,
            "proof": proof,
            "engine_latency_ms": elapsed.as_secs_f64() * 1000.0
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

async fn list_projects(
    State(state): State<EngineState>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::MultiTenant { mt_engine, .. } = state {
        let projects = mt_engine.list_projects();
        (StatusCode::OK, Json(serde_json::json!({ "projects": projects })))
    } else {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Not in multi-tenant mode"})),
        )
    }
}

async fn delete_project(
    State(state): State<EngineState>,
    Path(project_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let EngineState::MultiTenant { mt_engine, .. } = state {
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
    } else {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Not in multi-tenant mode"})),
        )
    }
}

// Multi-tenant Alias Handlers

async fn add_alias_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<AddAliasRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    if let EngineState::MultiTenant { mt_engine, read_only, .. } = state {
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only"})));
        }

        let ctx = mt_engine.get_or_create_project(project_id);
        
        let alias_id = uuid::Uuid::new_v4().to_string();
        let content = serde_json::json!({
            "from": req.from,
            "to": req.to,
            "downweight": req.weight.unwrap_or(0.85),
            "status": "active",
            "reason": "manual"
        }).to_string();

        let cues = vec![
            "type:alias".to_string(),
            format!("from:{}", req.from),
            format!("to:{}", req.to),
            "status:active".to_string(),
            "reason:manual".to_string(),
        ];

        ctx.aliases.upsert_memory_with_id(
            alias_id.clone(),
            content,
            cues,
            None,
            false 
        );

        (StatusCode::OK, Json(serde_json::json!({"id": alias_id, "status": "created"})))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

async fn get_aliases_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    if let EngineState::MultiTenant { mt_engine, .. } = state {
        let ctx = mt_engine.get_or_create_project(project_id);
        
        let cue = params.get("cue").cloned().unwrap_or_default();
        if cue.is_empty() {
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing 'cue' query param"})));
        }
        
        let query_cues = vec![
            "type:alias".to_string(), 
            format!("to:{}", cue),
            "status:active".to_string()
        ];
        
        let results = ctx.aliases.recall(query_cues, 50, false);
        let mut aliases = Vec::new();
        
        for res in results {
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&res.content) {
                let from_match = data.get("from").and_then(|v| v.as_str()).map(|v| v == cue).unwrap_or(false);
                let to_match = data.get("to").and_then(|v| v.as_str()).map(|v| v == cue).unwrap_or(false);
                
                if from_match || to_match {
                    aliases.push(data);
                }
            }
        }
        
        (StatusCode::OK, Json(serde_json::json!({"aliases": aliases})))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

/// Lexicon Surgeon (Multi-tenant): Inspect a cue in the Lexicon
async fn lexicon_inspect_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Path(cue): axum::extract::Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    if let EngineState::MultiTenant { mt_engine, .. } = state {
        let ctx = mt_engine.get_or_create_project(project_id);
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
        let outgoing: Vec<LexiconEntry> = outgoing_results.iter().map(|r| {
            let token = r.metadata.get("cues")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str())
                .unwrap_or(&cue_lower)
                .to_string();
            
            let affected = count_affected(&token, &r.content);
            
            LexiconEntry {
                memory_id: r.memory_id.clone(),
                content: r.content.clone(),
                token,
                reinforcement_score: r.reinforcement_score,
                created_at: r.created_at,
                affected_memories_count: affected,
            }
        }).collect();
        
        // 2. INCOMING: What tokens map to this canonical cue?
        let mut incoming: Vec<LexiconEntry> = Vec::new();
        for ref_multi in ctx.lexicon.get_memories().iter() {
            let memory = ref_multi.value();
            if memory.content.to_lowercase() == cue_lower {
                for token in &memory.cues {
                    let affected = count_affected(token, &memory.content);
                    incoming.push(LexiconEntry {
                        memory_id: memory.id.clone(),
                        content: memory.content.clone(),
                        token: token.clone(),
                        reinforcement_score: memory.reinforcement_count as f64,
                        created_at: memory.created_at,
                        affected_memories_count: affected,
                    });
                }
            }
        }
        
        (StatusCode::OK, Json(serde_json::json!(LexiconInspectResponse {
            cue: cue_lower,
            outgoing,
            incoming,
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

/// Delete a lexicon entry (multi-tenant)
async fn lexicon_delete_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Path(memory_id): axum::extract::Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    if let EngineState::MultiTenant { mt_engine, read_only, .. } = state {
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
        }
        
        let ctx = mt_engine.get_or_create_project(project_id);
        let deleted = ctx.lexicon.delete_memory(&memory_id);
        if deleted {
            (StatusCode::OK, Json(serde_json::json!({
                "status": "deleted",
                "memory_id": memory_id
            })))
        } else {
            (StatusCode::NOT_FOUND, Json(serde_json::json!({
                "error": "Entry not found",
                "memory_id": memory_id
            })))
        }
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

/// Get full Lexicon as graph data (multi-tenant)
async fn lexicon_graph_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    if let EngineState::MultiTenant { mt_engine, .. } = state {
        let ctx = mt_engine.get_or_create_project(project_id);
        let mut nodes = Vec::new();
        let mut links = Vec::new();
        let mut token_to_canonical: HashMap<String, Vec<String>> = HashMap::new();
        
        // Return all entries (no limit)
        for ref_multi in ctx.lexicon.get_memories().iter() {
            let memory = ref_multi.value();
            let canonical = memory.content.clone();
            for token in &memory.cues {
                token_to_canonical.entry(token.clone())
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
        
        (StatusCode::OK, Json(serde_json::json!({
            "nodes": nodes,
            "links": links,
            "total_entries": nodes.len()
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

/// Manually wire a token to a canonical cue (multi-tenant)
async fn lexicon_wire_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<WireLexiconRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    if let EngineState::MultiTenant { mt_engine, read_only, .. } = state {
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
        }
        
        let ctx = mt_engine.get_or_create_project(project_id);
        let token = req.token.to_lowercase();
        let canonical = req.canonical.to_lowercase();
        let lex_id = format!("cue:{}", canonical);
        
        ctx.lexicon.upsert_memory_with_id(
            lex_id.clone(),
            canonical.clone(),
            vec![token.clone()],
            None,
            false
        );
        
        (StatusCode::OK, Json(serde_json::json!({
            "status": "wired",
            "memory_id": lex_id,
            "token": token,
            "canonical": canonical
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

/// Get WordNet synonyms for a cue (multi-tenant)
async fn lexicon_synonyms_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Path(cue): axum::extract::Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    if let EngineState::MultiTenant { mt_engine, .. } = state {
        let ctx = mt_engine.get_or_create_project(project_id);
        let cue_lower = cue.to_lowercase();
        
        // 1. Get bidirectional neighbors
        let mut connected: std::collections::HashSet<String> = std::collections::HashSet::new();
        connected.insert(cue_lower.clone());
        
        for ref_multi in ctx.lexicon.get_memories().iter() {
            let memory = ref_multi.value();
            let mem_canon = memory.content.to_lowercase();
            
            if mem_canon == cue_lower || memory.cues.iter().any(|c| c.to_lowercase() == cue_lower) {
                 connected.insert(mem_canon.clone());
                 for t in &memory.cues { connected.insert(t.to_lowercase()); }
            }
        }
        
        // 2. Recursive WordNet Expansion (Depth 2)
        let mut candidates = std::collections::HashSet::new();
        let layer1 = ctx.semantic_engine.expand_wordnet(&cue_lower, &[cue_lower.clone()], 0.50, 50);
        for w1 in layer1 {
            candidates.insert(w1.clone());
            let layer2 = ctx.semantic_engine.expand_wordnet(&w1, &[], 0.50, 20);
            for w2 in layer2 {
                candidates.insert(w2);
            }
        }
        
        // 3. Filter
        let mut suggestions: Vec<String> = candidates
            .into_iter()
            .filter(|s| !connected.contains(&s.to_lowercase()) && s.len() > 2)
            .collect();
            
        suggestions.sort();
        suggestions.truncate(50);
        
        // 4. Categorize
        let mut existing_in_graph: Vec<String> = Vec::new();
        let mut new_suggestions: Vec<String> = Vec::new();
        
        for syn in &suggestions {
            let exists = ctx.lexicon.get_memories().iter().any(|ref_multi| {
                let memory = ref_multi.value();
                memory.content.to_lowercase() == syn.to_lowercase() ||
                memory.cues.iter().any(|c| c.to_lowercase() == syn.to_lowercase())
            });
            
            if exists {
                existing_in_graph.push(syn.clone());
            } else {
                new_suggestions.push(syn.clone());
            }
        }
        
        (StatusCode::OK, Json(serde_json::json!({
            "cue": cue_lower,
            "suggestions": suggestions,
            "existing_in_graph": existing_in_graph,
            "new_only": new_suggestions
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}

async fn merge_aliases_mt(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<MergeAliasRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    if let EngineState::MultiTenant { mt_engine, read_only, .. } = state {
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only"})));
        }

        let ctx = mt_engine.get_or_create_project(project_id);
        let mut created_ids = Vec::new();

        for from_cue in req.cues {
            let alias_id = uuid::Uuid::new_v4().to_string();
            let content = serde_json::json!({
                "from": from_cue,
                "to": req.to,
                "downweight": 1.0, 
                "status": "active",
                "reason": "manual_merge"
            }).to_string();

            let cues = vec![
                "type:alias".to_string(),
                format!("from:{}", from_cue),
                format!("to:{}", req.to),
                "status:active".to_string(),
                "reason:manual_merge".to_string(),
            ];

            ctx.aliases.upsert_memory_with_id(
                alias_id.clone(),
                content,
                cues,
                None,
                false
            );
            created_ids.push(alias_id);
        }

        (StatusCode::OK, Json(serde_json::json!({
            "status": "merged", 
            "target": req.to, 
            "count": created_ids.len()
        })))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Invalid state"})))
    }
}
