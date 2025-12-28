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
}

#[derive(Debug, Serialize)]
pub struct AddMemoryResponse {
    id: String,
    status: String,
}

#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    #[serde(default)]
    cues: Vec<String>,
    #[serde(default)]
    query_text: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
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
    #[serde(default)]
    pub disable_pattern_completion: bool,
    #[serde(default)]
    pub disable_salience_bias: bool,
    #[serde(default)]
    pub disable_systems_consolidation: bool,
}

fn default_token_budget() -> u32 {
    500
}

#[derive(Debug, Serialize)]
pub struct RecallGroundedResponse {
    pub verified_context: String,
    pub proof: crate::grounding::GroundingProof,
    pub engine_latency_ms: f64,
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
        .route("/memories/:id", get(get_memory))
        .route("/stats", get(get_stats))
        .route("/recall/grounded", post(recall_grounded))
        .route("/aliases", post(add_alias).get(get_aliases))
        .route("/aliases/merge", post(merge_aliases))
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
        .route("/memories/:id", get(get_memory_mt))
        .route("/stats", get(get_stats_mt))
        .route("/projects", get(list_projects))
        .route("/recall/grounded", post(recall_grounded_mt))
        .route("/projects/:id", delete(delete_project))
        .route("/aliases", post(add_alias_mt).get(get_aliases_mt))
        .route("/aliases/merge", post(merge_aliases_mt))
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
        // Check if read-only
        if read_only {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({
                    "error": "Read-only mode: modifications are not allowed"
                })),
            );
        }
        
        // 1. Normalize cues
        let mut normalized_cues = Vec::new();
        for cue in req.cues {
            let (normalized, _) = normalize_cue(&cue, &project.normalization);
            normalized_cues.push(normalized);
        }
        
        // 2. Validate cues
        let report = validate_cues(normalized_cues, &project.taxonomy);
        
        let memory_id = project.main.add_memory(req.content.clone(), report.accepted, req.metadata, req.disable_temporal_chunking);
        
        // Enqueue background jobs
        job_queue.enqueue(Job::TrainLexiconFromMemory {
            project_id: "default".to_string(), 
            memory_id: memory_id.clone()
        }).await;
        
        job_queue.enqueue(Job::LlmProposeCues {
            project_id: "default".to_string(),
            memory_id: memory_id.clone(),
            content: req.content,
        }).await;
        
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": memory_id,
                "status": "stored",
                "rejected_cues": report.rejected
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
    
    if let EngineState::SingleTenant { project, .. } = state {
        let start = Instant::now();
        
        // Collect cues from request
        let mut cues_to_process = req.cues;
        
        // Resolve cues from text if present
        if let Some(text) = req.query_text {
            let resolved = project.resolve_cues_from_text(&text);
            cues_to_process.extend(resolved);
        }

        // Normalize query cues
        let mut normalized_cues = Vec::new();
        for cue in &cues_to_process {
            let (normalized, _) = normalize_cue(cue, &project.normalization);
            normalized_cues.push(normalized);
        }
        
        // Expand aliases
        let expanded_cues = project.expand_query_cues(normalized_cues);
        let results = project.main.recall_weighted(
            expanded_cues.clone(), 
            req.limit, 
            req.auto_reinforce, 
            req.min_intersection,
            req.explain,
            req.disable_pattern_completion,
            req.disable_salience_bias,
            req.disable_systems_consolidation
        );
        
        let elapsed = start.elapsed();
        let engine_latency_ms = elapsed.as_secs_f64() * 1000.0;
        
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

async fn recall_grounded(
    State(state): State<EngineState>,
    Json(req): Json<RecallGroundedRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use std::time::Instant;
    use crate::grounding::{GroundingEngine, create_grounding_proof};

    if let EngineState::SingleTenant { project, .. } = state {
        let start = Instant::now();
        
        // 1. Standard CueMap Recall
        let resolved = project.resolve_cues_from_text(&req.query_text);
        let mut normalized_cues = Vec::new();
        for cue in &resolved {
            let (normalized, _) = crate::normalization::normalize_cue(cue, &project.normalization);
            normalized_cues.push(normalized);
        }
        let expanded_cues = project.expand_query_cues(normalized_cues);
        let results = project.main.recall_weighted(
            expanded_cues.clone(), 
            req.limit.max(20),
            false, 
            None,
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
        
        // 1. Normalize cues
        let mut normalized_cues = Vec::new();
        for cue in &req.cues {
            let (normalized, _) = normalize_cue(&cue, &ctx.normalization);
            normalized_cues.push(normalized);
        }
        
        // 2. Validate cues
        let report = validate_cues(normalized_cues, &ctx.taxonomy);
        
        let memory_id = ctx.main.add_memory(req.content.clone(), report.accepted, req.metadata, req.disable_temporal_chunking);
        
        // Enqueue background jobs
        job_queue.enqueue(Job::TrainLexiconFromMemory {
            project_id: project_id.clone(), 
            memory_id: memory_id.clone()
        }).await;
        
        job_queue.enqueue(Job::LlmProposeCues {
            project_id: project_id.clone(),
            memory_id: memory_id.clone(),
            content: req.content,
        }).await;
        
        tracing::info!(
            "POST /memories project={} cues={} id={}",
            project_id,
            req.cues.len(),
            memory_id
        );
        
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": memory_id,
                "status": "stored",
                "rejected_cues": report.rejected
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
    
    if let EngineState::MultiTenant { mt_engine, .. } = state {
        // Cross-domain query if projects array is provided
        if let Some(projects) = req.projects {
            let start = Instant::now();
            
            // Query all projects in parallel using rayon
            let all_results: Vec<serde_json::Value> = projects
                .par_iter()
                .map(|project_id| {
                    let ctx = mt_engine.get_or_create_project(project_id.clone());
                    
                    // Collect cues
                    let mut cues_to_process = req.cues.clone();
                    
                    if let Some(text) = &req.query_text {
                         let resolved = ctx.resolve_cues_from_text(text);
                         cues_to_process.extend(resolved);
                    }
                    
                    // Normalize query cues
                    let mut normalized_cues = Vec::new();
                    for cue in &cues_to_process {
                        let (normalized, _) = normalize_cue(cue, &ctx.normalization);
                        normalized_cues.push(normalized);
                    }
                    
                    // Expand aliases
                    let expanded_cues = ctx.expand_query_cues(normalized_cues);
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
                        .into_iter()
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
                    
                    response_block
                })
                .collect();
            
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
        let mut cues_to_process = req.cues;
        
        if let Some(text) = req.query_text {
             let resolved = ctx.resolve_cues_from_text(&text);
             cues_to_process.extend(resolved);
        }
        
        // Normalize query cues
        let mut normalized_cues = Vec::new();
        for cue in &cues_to_process {
            let (normalized, _) = normalize_cue(cue, &ctx.normalization);
            normalized_cues.push(normalized);
        }
        
        // Expand aliases
        let expanded_cues = ctx.expand_query_cues(normalized_cues);
        
        let results = ctx.main.recall_weighted(
            expanded_cues.clone(), 
            req.limit, 
            req.auto_reinforce, 
            req.min_intersection,
            req.explain,
            req.disable_pattern_completion,
            req.disable_salience_bias,
            req.disable_systems_consolidation
        );
        let elapsed = start.elapsed();
        
        let engine_latency_ms = elapsed.as_secs_f64() * 1000.0;
        
        tracing::info!(
            "POST /recall project={} cues={} results={} latency={:.2}ms",
            project_id,
            cues_to_process.len(),
            results.len(),
            engine_latency_ms
        );
        
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
        let resolved = ctx.resolve_cues_from_text(&req.query_text);
        let mut normalized_cues = Vec::new();
        for cue in &resolved {
            let (normalized, _) = crate::normalization::normalize_cue(cue, &ctx.normalization);
            normalized_cues.push(normalized);
        }
        let expanded_cues = ctx.expand_query_cues(normalized_cues);
        
        let results = ctx.main.recall_weighted(
            expanded_cues.clone(), 
            req.limit.max(20),
            false, 
            None,
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
