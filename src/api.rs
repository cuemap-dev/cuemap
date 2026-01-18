use crate::auth::AuthConfig;
use crate::multi_tenant::{MultiTenantEngine, validate_project_id};
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
    /// When true, uses O(limit) recall_intersection for ~1ms latency.
    /// No pattern completion, no scoring - just direct index lookup.
    #[serde(default)]
    pub fast_mode: bool,
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

/// Request for POST /ingest/url - ingest content from a URL
#[derive(Debug, Deserialize)]
pub struct IngestUrlRequest {
    pub url: String,
}

#[derive(Debug, Serialize)]
pub struct ReinforceResponse {
    status: String,
    memory_id: String,
}

#[derive(Clone)]
pub struct EngineState {
    pub mt_engine: Arc<MultiTenantEngine>,
    pub read_only: bool,
    pub job_queue: Arc<JobQueue>,
}

/// API Routes
pub fn routes(mt_engine: Arc<MultiTenantEngine>, job_queue: Arc<JobQueue>, auth_config: AuthConfig, read_only: bool) -> Router {
    let mut router = Router::new()
        .route("/", get(root))
        .route("/memories", post(add_memory))
        .route("/recall", post(recall))
        .route("/memories/:id/reinforce", patch(reinforce_memory))
        .route("/memories/:id", get(get_memory).delete(delete_memory))
        .route("/stats", get(get_stats))
        .route("/projects", get(list_projects))
        .route("/sandbox/create", post(create_sandbox_project))
        .route("/recall/grounded", post(recall_grounded))
        .route("/projects/:id", delete(delete_project))
        .route("/aliases", post(add_alias).get(get_aliases))
        .route("/aliases/merge", post(merge_aliases))
        .route("/graph", get(get_graph))
        .route("/lexicon/inspect/:cue", get(lexicon_inspect))
        .route("/lexicon/entry/:id", delete(lexicon_delete))
        .route("/lexicon/graph", get(lexicon_graph))
        .route("/lexicon/wire", post(lexicon_wire))
        .route("/lexicon/synonyms/:cue", get(lexicon_synonyms))
        .route("/ingest/url", post(ingest_url))
        .route("/ingest/content", post(ingest_content))
        .route("/ingest/file", post(ingest_file))
        .route("/jobs/status", get(jobs_status))
        .fallback(crate::web::handler)
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(EngineState { 
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
        "version": "0.6.0",
        "description": "High-performance Temporal-Associative Memory Store"
    }))
}

async fn get_graph(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let limit = params.get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    let EngineState { mt_engine, .. } = state;
    
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

    let ctx = match mt_engine.get_or_create_project(project_id.clone()) {
        Ok(c) => c,
        Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
    };
    
    let limit_clone = limit;
    let ctx_clone = ctx.clone();
    
    let graph = tokio::task::spawn_blocking(move || {
        ctx_clone.main.get_graph_data(limit_clone)
    }).await.unwrap();

    (StatusCode::OK, Json(graph))
}

// Handlers
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

async fn add_memory(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<AddMemoryRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    use std::time::Instant;
    let start = Instant::now();
    let EngineState { mt_engine, read_only, job_queue } = state;

    // Check if read-only
    if read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": "Read-only mode: modifications are not allowed"
            })),
        );
    }
    
    let ctx = match mt_engine.get_or_create_project(project_id.clone()) {
        Ok(c) => c,
        Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
    };
    
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

    // Buffer background jobs (will be processed after ingestion completes)
    let session = job_queue.session_manager.get_or_create(&project_id);
    session.expect_write();
    
    job_queue.buffer(&project_id, Job::ProposeCues {
        project_id: project_id.clone(),
        memory_id: memory_id.clone(),
        content: req.content.clone(),
    }).await;

    job_queue.buffer(&project_id, Job::TrainLexiconFromMemory {
        project_id: project_id.clone(), 
        memory_id: memory_id.clone()
    }).await;
    
    job_queue.buffer(&project_id, Job::UpdateGraph {
        project_id: project_id.clone(),
        memory_id: memory_id.clone(),
    }).await;
    
    session.write_complete();
    
    tracing::debug!(
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
}

async fn recall(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<RecallRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use std::time::Instant;
    
    let EngineState { ref mt_engine, ref job_queue, .. } = &state;
    {
        // Cross-domain query if projects array is provided
        if let Some(projects) = req.projects {
            let start = Instant::now();
            
            // Query all projects in parallel using rayon
            let (all_results, reinforce_tasks): (Vec<serde_json::Value>, Vec<Option<(String, Vec<String>, Vec<String>)>>) = projects
                .par_iter()
                .map(|project_id| {
                    let ctx = match mt_engine.get_or_create_project(project_id.clone()) {
                        Ok(c) => c,
                        Err(_) => return (serde_json::json!({"project_id": project_id, "error": "Capacity reached"}), None),
                    };
                    
                    // Collect cues
                    let mut cues_to_process = req.cues.clone();
                    
                    let (original_tokens, _lexicon_mids) = if let Some(text) = &req.query_text {
                         let (resolved, lex_mids) = ctx.resolve_cues_from_text(text, false);
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
                    
                    // Use fast O(1) recall when fast_mode is enabled
                    let results = if req.fast_mode {
                        ctx.main.recall_intersection(expanded_cues.clone(), req.limit)
                    } else {
                        ctx.main.recall_weighted(
                            expanded_cues.clone(), 
                            req.limit, 
                            false,
                            req.min_intersection,
                            req.explain,
                            req.disable_pattern_completion,
                            req.disable_salience_bias,
                            req.disable_systems_consolidation
                        )
                    };
                    
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
            
            tracing::debug!(
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
        let ctx = match mt_engine.get_or_create_project(project_id.clone()) {
            Ok(c) => c,
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
        };
        
        // Collect cues
        let mut cues_to_process = req.cues.clone();
        
        let t_lex = Instant::now();
        let mut lexicon_memory_ids: Vec<String> = Vec::new();
        if let Some(ref text) = req.query_text {
             // 1. Lexicon Recall
             let (resolved, lex_mids) = ctx.resolve_cues_from_text(text, false);
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
        
        tracing::debug!(
            "Recall breakdown: lex={:.2}ms norm={:.2}ms expand={:.2}ms search={:.2}ms | cues={} expanded={}",
            lex_ms, norm_ms, expand_ms, search_ms, cues_to_process.len(), expanded_cues.len()
        );
        
        tracing::debug!(
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
    }
}

async fn reinforce_memory(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Path(memory_id): Path<String>,
    Json(req): Json<ReinforceRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    let EngineState { mt_engine, .. } = state;
        let ctx = match mt_engine.get_or_create_project(project_id) {
            Ok(c) => c,
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
        };
        
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
}

async fn get_memory(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Path(memory_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    let EngineState { mt_engine, .. } = state;
    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
    };
    match ctx.main.get_memory(&memory_id) {
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
    Path(memory_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState { mt_engine, read_only, .. } = state;
    if read_only {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
    }
    
    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
    };
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
}

async fn get_stats(
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
        Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
    };
    let stats = ctx.main.get_stats();
    (StatusCode::OK, Json(serde_json::Value::Object(stats.into_iter().collect())))
}

/// Get job/ingestion progress for a project
async fn jobs_status(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    let EngineState { job_queue, .. } = state;
    if let Some(session) = job_queue.get_session(&project_id) {
        let progress = session.get_progress();
        (StatusCode::OK, Json(serde_json::json!(progress)))
    } else {
        // No active session - return idle status
        (StatusCode::OK, Json(serde_json::json!({
            "phase": "idle",
            "writes_completed": 0,
            "writes_total": 0,
            "propose_cues_completed": 0,
            "propose_cues_total": 0,
            "train_lexicon_completed": 0,
            "train_lexicon_total": 0,
            "update_graph_completed": 0,
            "update_graph_total": 0
        })))
    }
}

async fn recall_grounded(
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

    let EngineState { mt_engine, .. } = state;
        let start = Instant::now();
        let ctx = match mt_engine.get_or_create_project(project_id) {
            Ok(c) => c,
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
        };
        
        // 1. Standard CueMap Recall
        let (resolved, _lexicon_memory_ids) = ctx.resolve_cues_from_text(&req.query_text, false);
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
}

async fn list_projects(
    State(state): State<EngineState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState { mt_engine, .. } = state;
    let projects = mt_engine.list_projects();
    (StatusCode::OK, Json(serde_json::json!({ "projects": projects })))
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

    let EngineState { mt_engine, read_only, .. } = state;
    if read_only {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only"})));
    }

    let ctx = match mt_engine.get_or_create_project(project_id) {
        Ok(c) => c,
        Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
    };
    
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
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
        };
        
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
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
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
}

/// Delete a lexicon entry (multi-tenant)
async fn lexicon_delete(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Path(memory_id): axum::extract::Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    let EngineState { mt_engine, read_only, .. } = state;
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
        }
        
        let ctx = match mt_engine.get_or_create_project(project_id) {
            Ok(c) => c,
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
        };
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
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
        };
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

    let EngineState { mt_engine, read_only, .. } = state;
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
        }
        
        let ctx = match mt_engine.get_or_create_project(project_id) {
            Ok(c) => c,
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
        };
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
}

/// Get WordNet synonyms for a cue (multi-tenant)
async fn lexicon_synonyms(
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
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
        };
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

    let EngineState { mt_engine, read_only, .. } = state;
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only"})));
        }

        let ctx = match mt_engine.get_or_create_project(project_id) {
            Ok(c) => c,
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
        };
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
}

async fn create_sandbox_project(
    State(state): State<EngineState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState { mt_engine, .. } = state;
    let project_id = generate_random_id();
    match mt_engine.get_or_create_project(project_id.clone()) {
        Ok(_) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "project_id": project_id,
                "status": "created",
                "expires_in_secs": 300
            })),
        ),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": e})),
        ),
    }
}

fn generate_random_id() -> String {
    use rand::seq::SliceRandom;
    let adjectives = ["gentle", "swift", "brave", "clever", "wild", "bright", "calm", "cool", "deep", "easy"];
    let nouns = ["dolphin", "eagle", "tiger", "fox", "owl", "wolf", "bear", "lion", "hawk", "shark"];
    let mut rng = rand::thread_rng();
    let adj = adjectives.choose(&mut rng).unwrap();
    let noun = nouns.choose(&mut rng).unwrap();
    let num = rand::random::<u16>() % 1000;
    format!("{}-{}-{}", adj, noun, num)
}

/// Ingest content from a URL using the Agent's Ingester
async fn ingest_url(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<IngestUrlRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use crate::agent::ingester::Ingester;
    use crate::agent::AgentConfig;
    
    let EngineState { read_only, job_queue, .. } = state;
    if read_only {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
    }
    
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    // Create an ingester for this request
    let config = AgentConfig {
        watch_dir: String::new(), // Not used for API-driven ingestion
        throttle_ms: 0,
    };
    let mut ingester = Ingester::new(config, job_queue);
    
    // Use the Ingester's process_url method
    match ingester.process_url(&req.url, &project_id).await {
        Ok(memory_ids) => (StatusCode::OK, Json(serde_json::json!({
            "status": "ingested",
            "url": req.url,
            "chunks": memory_ids.len(),
            "memory_ids": memory_ids
        }))),
        Err(e) => (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": format!("Failed to ingest URL: {}", e)
        }))),
    }
}

/// Request for POST /ingest/content - ingest raw content
#[derive(Debug, Deserialize)]
pub struct IngestContentRequest {
    pub content: String,
    #[serde(default = "default_filename")]
    pub filename: String, // Used to determine content type (e.g. "notes.md", "data.json")
}

fn default_filename() -> String {
    "content.txt".to_string()
}

/// Ingest raw content using the Agent's Ingester
async fn ingest_content(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<IngestContentRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use crate::agent::ingester::Ingester;
    use crate::agent::AgentConfig;
    
    let EngineState { read_only, job_queue, .. } = state;
    if read_only {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
    }
    
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    // Create an ingester for this request
    let config = AgentConfig {
        watch_dir: String::new(),
        throttle_ms: 0,
    };
    let mut ingester = Ingester::new(config, job_queue);
    
    // Use the Ingester's process_content method
    match ingester.process_content(&req.content, &req.filename, &project_id).await {
        Ok(memory_ids) => (StatusCode::OK, Json(serde_json::json!({
            "status": "ingested",
            "filename": req.filename,
            "chunks": memory_ids.len(),
            "memory_ids": memory_ids
        }))),
        Err(e) => (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": format!("Failed to ingest content: {}", e)
        }))),
    }
}

/// Ingest a binary file via multipart upload (for PDFs, Office docs, etc.)
async fn ingest_file(
    State(state): State<EngineState>,
    headers: HeaderMap,
    mut multipart: axum_extra::extract::Multipart,
) -> (StatusCode, Json<serde_json::Value>) {
    use crate::agent::chunker::Chunker;
    use crate::jobs::Job;
    use sha2::{Sha256, Digest};
    use std::io::Write;
    
    let EngineState { read_only, job_queue, .. } = state;
        if read_only {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
        }
        
        let project_id = match extract_project_id(&headers) {
            Ok(id) => id,
            Err(e) => return e,
        };
        
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
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "error": "No file data received"
            })));
        }
        
        // Write to temp file
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join(&filename);
        
        let mut temp_file = match std::fs::File::create(&temp_path) {
            Ok(f) => f,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to create temp file: {}", e)
            }))),
        };
        
        if let Err(e) = temp_file.write_all(&file_bytes) {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to write temp file: {}", e)
            })));
        }
        drop(temp_file);
        
        // Chunk the file
        let chunks = Chunker::chunk_binary_file(&temp_path);
        
        // Clean up temp file
        let _ = std::fs::remove_file(&temp_path);
        
        if chunks.is_empty() {
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "error": "Failed to extract content from file (0 chunks)"
            })));
        }
        
        // Track session for progress reporting
        let session = job_queue.session_manager.get_or_create(&project_id);
        for _ in &chunks {
            session.expect_write();
        }
        
        // Enqueue jobs for each chunk
        let source = format!("file:{}", filename);
        let mut memory_ids = Vec::new();
        
        for chunk in chunks.iter() {
            let mut chunk_hasher = Sha256::new();
            chunk_hasher.update(chunk.content.as_bytes());
            let chunk_hash = format!("{:x}", chunk_hasher.finalize());
            let memory_id = format!("{}:{}", source, chunk_hash);
            
            // ExtractAndIngest does the write - enqueue immediately
            job_queue.enqueue(Job::ExtractAndIngest {
                project_id: project_id.clone(),
                memory_id: memory_id.clone(),
                content: chunk.content.clone(),
                file_path: source.clone(),
                structural_cues: chunk.structural_cues.clone(),
                category: chunk.category,
            }).await;
            
            // Buffer downstream jobs for phased processing
            job_queue.buffer(&project_id, Job::ProposeCues {
                project_id: project_id.clone(),
                memory_id: memory_id.clone(),
                content: chunk.content.clone(),
            }).await;
            
            job_queue.buffer(&project_id, Job::TrainLexiconFromMemory {
                project_id: project_id.clone(),
                memory_id: memory_id.clone(),
            }).await;
            
            job_queue.buffer(&project_id, Job::UpdateGraph {
                project_id: project_id.clone(),
                memory_id: memory_id.clone(),
            }).await;
            
            session.write_complete();
            
            memory_ids.push(memory_id);
        }
        
        (StatusCode::OK, Json(serde_json::json!({
            "status": "ingested",
            "filename": filename,
            "chunks": memory_ids.len(),
            "memory_ids": memory_ids
        })))
}

