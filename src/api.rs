use crate::auth::AuthConfig;
use crate::structures::{MainStats, LexiconStats, MemoryStats};
use crate::multi_tenant::{MultiTenantEngine, validate_project_id};
use crate::normalization::normalize_cue;
use crate::taxonomy::validate_cues;
use crate::jobs::{Job, JobQueue};
use crate::metrics::MetricsCollector;
use crate::persistence::CloudBackupManager;
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


#[derive(Debug, Deserialize, Serialize)]
pub struct AddMemoryRequest {
    pub content: String,
    pub cues: Vec<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub disable_temporal_chunking: bool,
    #[serde(default)]
    pub async_ingest: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddMemoryResponse {
    id: String,
    status: String,
    cues: Vec<String>,
    latency_ms: f64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RecallRequest {
    #[serde(default)]
    pub cues: Vec<String>,
    #[serde(default)]
    pub query_text: Option<String>,
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
    pub disable_pattern_completion: bool,
    #[serde(default)]
    pub disable_salience_bias: bool,
    #[serde(default)]
    pub disable_systems_consolidation: bool,
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
    2048
}

#[derive(Debug, Serialize, Deserialize)]
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
    pub id: String,
    pub from: String,
    pub to: String,
    pub weight: f64,
}

/// Response for /lexicon/inspect/:cue endpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct LexiconInspectResponse {
    pub cue: String,
    pub outgoing: Vec<LexiconEntry>,  // What this token maps to
    pub incoming: Vec<LexiconEntry>,  // Other tokens that map to the same canonical
}

#[derive(Debug, Serialize, Deserialize)]
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

// Context API - Query Expansion
#[derive(Debug, Deserialize, Serialize)]
pub struct ContextExpandRequest {
    pub query: String,
    #[serde(default = "default_context_limit")]
    pub limit: usize,
    #[serde(default)]
    pub min_score: Option<f64>,
}

fn default_context_limit() -> usize {
    20
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContextExpandResponse {
    pub query_cues: Vec<String>,
    pub expansions: Vec<ExpansionCandidate>,
    pub latency_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpansionCandidate {
    pub term: String,
    pub score: f64,
    pub co_occurrence_count: u64,
    pub source_cues: Vec<String>,
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
    memory_id: String,
}

#[derive(Clone)]
pub struct EngineState {
    pub mt_engine: Arc<MultiTenantEngine>,
    pub read_only: bool,
    pub job_queue: Arc<JobQueue>,
    pub metrics: Arc<MetricsCollector>,
    pub cloud_backup: Option<Arc<CloudBackupManager>>,
}

/// API Routes
pub fn routes(
    mt_engine: Arc<MultiTenantEngine>, 
    job_queue: Arc<JobQueue>, 
    metrics: Arc<MetricsCollector>, 
    auth_config: AuthConfig, 
    read_only: bool,
    cloud_backup: Option<Arc<CloudBackupManager>>,
) -> Router {
    let mut router = Router::new()
        .route("/", get(root))
        .route("/memories", post(add_memory))
        .route("/recall", post(recall))
        .route("/recall/web", post(recall_web))
        .route("/memories/:id/reinforce", patch(reinforce_memory))
        .route("/memories/:id", get(get_memory).delete(delete_memory))
        .route("/stats", get(get_stats))
        .route("/projects", get(list_projects).post(create_project))
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
        .route("/context/expand", post(context_expand))
        .route("/metrics", get(prometheus_metrics))
        // Cloud backup endpoints
        .route("/backup/upload", post(backup_upload))
        .route("/backup/download", post(backup_download))
        .route("/backup/list", get(backup_list))
        .route("/backup/:project_id", delete(backup_delete))
        .fallback(crate::web::handler)
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(EngineState { 
            mt_engine,
            read_only,
            job_queue,
            metrics,
            cloud_backup,
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
        "version": "0.6.1",
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

fn extract_project_id_optional(headers: &HeaderMap) -> Option<String> {
    headers
        .get("X-Project-ID")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .filter(|s| validate_project_id(s))
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
    let EngineState { mt_engine, read_only, job_queue, metrics, .. } = state;

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
    let memory_id = ctx.main.add_memory(
        req.content.clone(), 
        report.accepted.clone(), 
        req.metadata, 
        MainStats::default(),
        req.disable_temporal_chunking
    );
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
    
    tracing::info!(
        "POST /memories project={} cues={} id={} timings: tok={:.2}ms norm={:.2}ms val={:.2}ms ins={:.2}ms",
        project_id,
        report.accepted.len(),
        memory_id,
        t_tokenize, t_normalize, t_validate, t_insert
    );
    
    let elapsed = start.elapsed();
    let latency_ms = elapsed.as_secs_f64() * 1000.0;

    // Record metrics
    metrics.record_ingestion();

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

#[axum::debug_handler]
async fn recall(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<RecallRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use std::time::Instant;
    
    let EngineState { ref mt_engine, ref job_queue, .. } = &state;
    
    // --- Path 1: Cross-domain query ---
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
                
                // Expand aliases
                let expanded_cues = ctx.expand_query_cues(normalized_cues, &original_tokens);
                
                // Read Market Heatmap (scoped locally inside this closure, no awaits here, so it's safe)
                let heatmap = ctx.market_heatmap.read().ok();
                let heatmap_ref = heatmap.as_deref();

                let results = ctx.main.recall_weighted(
                    expanded_cues.clone(), 
                    req.limit, 
                    false,
                    req.min_intersection,
                    req.explain,
                    req.disable_pattern_completion,
                    req.disable_salience_bias,
                    req.disable_systems_consolidation,
                    heatmap_ref
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

        state.metrics.record_recall(engine_latency_ms);
        
        return (StatusCode::OK, Json(serde_json::json!({ 
            "results": all_results,
            "engine_latency": engine_latency_ms
        })));
    }
    
    // --- Path 2: Single project query ---
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
    
    // Expand aliases
    let t_expand = Instant::now();
    let original_tokens = if let Some(ref text) = req.query_text {
        crate::nl::tokenize_to_cues(text)
    } else {
        req.cues.clone()
    };
    let expanded_cues = ctx.expand_query_cues(normalized_cues, &original_tokens);
    let expand_ms = t_expand.elapsed().as_secs_f64() * 1000.0;
    
    let t_search = Instant::now();

    let results = {
        let heatmap = ctx.market_heatmap.read().ok();
        let heatmap_ref = heatmap.as_deref();

        ctx.main.recall_weighted(
            expanded_cues.clone(), 
            req.limit, 
            false, 
            req.min_intersection,
            req.explain,
            req.disable_pattern_completion,
            req.disable_salience_bias,
            req.disable_systems_consolidation,
            heatmap_ref
        )
    }; 

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
        // This await was causing the conflict because `heatmap` was seen as potentially live
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

    // Record metrics
    state.metrics.record_recall(engine_latency_ms);
    
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
    let project_id_opt = extract_project_id_optional(&headers);
    let EngineState { mt_engine, .. } = state;

    if let Some(project_id) = project_id_opt {
        let ctx = match mt_engine.get_or_create_project(project_id) {
            Ok(c) => c,
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
        };
        let stats = ctx.main.get_stats();
        (StatusCode::OK, Json(serde_json::Value::Object(stats.into_iter().collect())))
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
        
        let heatmap = ctx.market_heatmap.read().ok();
        let heatmap_ref = heatmap.as_deref();

        let results = ctx.main.recall_weighted(
            expanded_cues.clone(), 
            req.limit.max(20),
            req.auto_reinforce, 
            req.min_intersection,
            true,
            req.disable_pattern_completion,
            req.disable_salience_bias,
            req.disable_systems_consolidation,
            heatmap_ref
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
        let crypto = crate::crypto::CryptoEngine::new();
        let signature = crypto.sign(&context_block);
        
        (StatusCode::OK, Json(serde_json::json!({ 
            "verified_context": context_block,
            "proof": proof,
            "engine_latency_ms": elapsed.as_secs_f64() * 1000.0,
            "signature": signature
        })))
}

async fn list_projects(
    State(state): State<EngineState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState { mt_engine, .. } = state;
    let projects = mt_engine.list_projects();
    (StatusCode::OK, Json(serde_json::json!({ "projects": projects })))
}

async fn create_project(
    State(state): State<EngineState>,
    Json(req): Json<CreateProjectRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let EngineState { mt_engine, read_only, .. } = state;
    if read_only {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
    }

    if !validate_project_id(&req.project_id) {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Invalid project ID format"})));
    }

    // Check if exists first to return 409 Conflict logic if desired, or just idempotent OK
    // get_or_create_project is idempotent, but we might want to be explicit.
    // For now, let's just use get_or_create_project and return 200 OK or 201 Created.
    // Actually, if we want to mimic "create", 201 is good.
    
    match mt_engine.get_or_create_project(req.project_id.clone()) {
        Ok(_) => {
            (
                StatusCode::CREATED,
                Json(serde_json::json!({
                    "status": "created", 
                    "project_id": req.project_id 
                })),
            )
        },
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
        Some(MainStats::default()),
        false,
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
        
        let results = ctx.aliases.recall(query_cues, 50, false, None);
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
        let key_lex = ctx.lexicon.get_master_key();
        for ref_multi in ctx.lexicon.get_memories().iter() {
            let memory = ref_multi.value();
            let content = memory.access_content(key_lex.as_deref()).unwrap_or_default();
            if content.to_lowercase() == cue_lower {
                for token in &memory.cues {
                    let affected = count_affected(token, &content);
                    incoming.push(LexiconEntry {
                        memory_id: memory.id.clone(),
                        content: content.clone(),
                        token: token.clone(),
                        reinforcement_score: memory.stats.get_reinforcement_count() as f64,
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
        let key = ctx.lexicon.get_master_key();
        for ref_multi in ctx.lexicon.get_memories().iter() {
            let memory = ref_multi.value();
            let canonical = memory.access_content(key.as_deref()).unwrap_or_default();
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
            Some(LexiconStats::default()),
            false,
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
            let content = memory.access_content(ctx.lexicon.get_master_key().as_deref()).unwrap_or_default();
            let mem_canon = content.to_lowercase();
            
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
        
        let key = ctx.lexicon.get_master_key();
        for syn in &suggestions {
            let exists = ctx.lexicon.get_memories().iter().any(|ref_multi| {
                let memory = ref_multi.value();
                let content = memory.access_content(key.as_deref()).unwrap_or_default();
                content.to_lowercase() == syn.to_lowercase() ||
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
                Some(MainStats::default()),
                false,
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



/// Ingest content from a URL using the Agent's Ingester
/// Supports recursive crawling when depth > 0


async fn recall_web(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<RecallWebRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use crate::agent::ingester::Ingester;
    use crate::agent::AgentConfig;
    use crate::agent::search::search_ddg_lite;
    use std::time::Instant;

    let EngineState { read_only, job_queue, .. } = state;
    if req.persist && read_only {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode: cannot persist"})));
    }

    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };

    // Ensure project exists (auto-create)
    if let Err(e) = state.mt_engine.get_or_create_project(project_id.clone()) {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e})));
    }

    // Create an ingester for this request
    let config = AgentConfig {
        watch_dir: String::new(),
        throttle_ms: 0,
        state_file: None,
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
            Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("Failed to fetch URL: {}", e)}))),
        };
    } else {
        // Search Mode
        let search_results = match search_ddg_lite(&req.query, 5).await {
            Ok(res) => res,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("Search failed: {}", e)}))),
        };
        
        urls_processed = search_results.clone();
        
        // Parallel Fetch & Chunk using JoinSet for async concurrency
        let mut set = tokio::task::JoinSet::new();
        
        for url in search_results {
            let ingester_clone = ingester.clone();
            set.spawn(async move {
                (url.clone(), ingester_clone.fetch_and_chunk_url(&url).await)
            });
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
             return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "No content found from search results"})));
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

    let mut scored_chunks: Vec<ScoredChunk> = chunks.iter().map(|chunk| {
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
    }).collect();

    // Sort by score desc
    scored_chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    
    // Take top results (more for search mode to have variety)
    let limit = if req.url.is_none() { 10 } else { 5 };
    
    let results: Vec<serde_json::Value> = scored_chunks.into_iter()
        .take(limit)
        .filter(|r| r.score > 0.0)
        .map(|r| serde_json::json!({
            "content": r.content,
            "score": r.score,
            "intersection": r.intersection,
            // "url": r.url // Optional source info
        }))
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
                watch_dir: String::new(),
                throttle_ms: 0,
                state_file: None,
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

            if let Err(e) = async_ingester.process_chunks(chunks_clone, &project_id_clone, &source).await {
                tracing::error!("Async persistence failed for web recall: {}", e);
            }
        });
    }

    let total_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    
    (StatusCode::OK, Json(serde_json::json!({
        "urls": urls_processed,
        "results": results,
        "latency_ms": total_ms,
        "timings": {
            "fetch_chunk": fetch_ms,
            "search_overhead": if req.url.is_none() { true } else { false },
            "processing": processing_ms
        }
    })))
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
    
    let EngineState { read_only, job_queue, .. } = state;
    if read_only {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "Read-only mode"})));
    }
    
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    // Ensure project exists (auto-create)
    if let Err(e) = state.mt_engine.get_or_create_project(project_id.clone()) {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e})));
    }
    
    // Create an ingester for this request
    let config = AgentConfig {
        watch_dir: String::new(), // Not used for API-driven ingestion
        throttle_ms: 0,
        state_file: None,
    };
    let mut ingester = Ingester::new(config, job_queue);
    
    // Check if recursive crawling is requested
    if req.depth > 0 {
        // Recursive crawl
        match ingester.process_url_recursive(
            &req.url, 
            &project_id, 
            req.depth, 
            req.same_domain_only,
        ).await {
            Ok(result) => (StatusCode::OK, Json(serde_json::json!({
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
            }))),
            Err(e) => (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "error": format!("Failed to crawl URL: {}", e)
            }))),
        }
    } else {
        // Single page ingestion (original behavior)
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
    
    // Ensure project exists (auto-create)
    if let Err(e) = state.mt_engine.get_or_create_project(project_id.clone()) {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e})));
    }
    
    // Create an ingester for this request
    let config = AgentConfig {
        watch_dir: String::new(),
        throttle_ms: 0,
        state_file: None,
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
    
    // Ensure project exists (auto-create)
    if let Err(e) = state.mt_engine.get_or_create_project(project_id.clone()) {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e})));
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

/// Context API: Expand a natural language query using the co-occurrence graph
/// 
/// This endpoint tokenizes the query, looks up co-occurring terms in the graph,
/// and returns ranked expansion candidates. Think of it as a domain-specific WordNet.
async fn context_expand(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(req): Json<ContextExpandRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use std::time::Instant;
    let start = Instant::now();
    
    let EngineState { mt_engine, .. } = state;
    
    // Extract project ID
    let project_id = match extract_project_id(&headers) {
        Ok(id) => id,
        Err(e) => return e,
    };
    
    // Get project context
    let ctx = match mt_engine.get_or_create_project(project_id.clone()) {
        Ok(c) => c,
        Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": e}))),
    };
    
    // 1. Tokenize query into cues
    let query_cues = crate::nl::tokenize_to_cues(&req.query);
    
    if query_cues.is_empty() {
        return (StatusCode::OK, Json(serde_json::json!({
            "query_cues": [],
            "expansions": [],
            "latency_ms": start.elapsed().as_secs_f64() * 1000.0
        })));
    }
    
    // 2. Normalize cues through the normalization layer
    let normalized_cues: Vec<String> = query_cues
        .iter()
        .map(|cue| {
            let (normalized, _) = normalize_cue(cue, &ctx.normalization);
            normalized
        })
        .collect();
    
    // 3. Expand using co-occurrence graph
    let raw_expansions = ctx.main.expand_cues_from_graph(&normalized_cues, req.limit);
    
    // 4. Filter by min_score if specified
    let expansions: Vec<ExpansionCandidate> = raw_expansions
        .into_iter()
        .filter(|(_, score, _, _)| {
            if let Some(min) = req.min_score {
                *score >= min
            } else {
                true
            }
        })
        .map(|(term, score, count, sources)| ExpansionCandidate {
            term,
            score,
            co_occurrence_count: count,
            source_cues: sources,
        })
        .collect();
    
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    
    tracing::info!(
        "POST /context/expand project={} query=\"{}\" cues={} expansions={} latency={:.2}ms",
        project_id,
        req.query,
        normalized_cues.len(),
        expansions.len(),
        latency_ms
    );
    
    (StatusCode::OK, Json(serde_json::json!({
        "query_cues": normalized_cues,
        "expansions": expansions,
        "latency_ms": latency_ms
    })))
}

/// Prometheus metrics endpoint - returns plain text in Prometheus exposition format
async fn prometheus_metrics(
    State(state): State<EngineState>,
) -> impl IntoResponse {
    use std::sync::atomic::Ordering;
    
    let EngineState { mt_engine, metrics, job_queue, .. } = state;
    
    // Get global stats from multi-tenant engine
    let global_stats = mt_engine.get_global_stats();
    let total_memories = global_stats.get("total_memories")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total_cues = global_stats.get("total_cues")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total_projects = global_stats.get("total_projects")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    
    // Get metrics from collector
    let ingestion_count = metrics.ingestion_count.load(Ordering::Relaxed);
    let recall_count = metrics.recall_count.load(Ordering::Relaxed);
    tracing::debug!("Metrics: Found ingestion_count={}, recall_count={}", ingestion_count, recall_count);
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
    let EngineState { mt_engine, cloud_backup, .. } = state;
    
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
    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string());
    let main_path = format!("{}/snapshots/{}.bin", data_dir, req.project_id);
    let aliases_path = format!("{}/snapshots/{}_aliases.bin", data_dir, req.project_id);
    let lexicon_path = format!("{}/snapshots/{}_lexicon.bin", data_dir, req.project_id);
    
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
    match backup_manager.upload_project_snapshot(
        &req.project_id,
        main_data,
        aliases_data,
        lexicon_data,
    ).await {
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
    let EngineState { mt_engine, cloud_backup, .. } = state;
    
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
    let (main_data, aliases_data, lexicon_data) = match backup_manager.download_project_snapshot(&req.project_id).await {
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
    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string());
    let snapshots_dir = format!("{}/snapshots", data_dir);
    
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
    let main_path = format!("{}/{}.bin", snapshots_dir, req.project_id);
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
        let aliases_path = format!("{}/{}_aliases.bin", snapshots_dir, req.project_id);
        let _ = std::fs::write(&aliases_path, &data);
    }
    
    // Write lexicon snapshot if present
    if let Some(data) = lexicon_data {
        let lexicon_path = format!("{}/{}_lexicon.bin", snapshots_dir, req.project_id);
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
async fn backup_list(
    State(state): State<EngineState>,
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
