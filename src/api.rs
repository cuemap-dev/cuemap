use crate::auth::AuthConfig;
use crate::engine::CueMapEngine;
use crate::multi_tenant::{MultiTenantEngine, validate_project_id};
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
}

#[derive(Debug, Serialize)]
pub struct AddMemoryResponse {
    id: String,
    status: String,
}

#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    cues: Vec<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    auto_reinforce: bool,
    #[serde(default)]
    projects: Option<Vec<String>>,
    #[serde(default)]
    min_intersection: Option<usize>,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Deserialize)]
pub struct ReinforceRequest {
    cues: Vec<String>,
}



#[derive(Debug, Serialize)]
pub struct ReinforceResponse {
    status: String,
    memory_id: String,
}

#[derive(Clone)]
pub enum EngineState {
    SingleTenant { engine: CueMapEngine, read_only: bool },
    MultiTenant { mt_engine: Arc<MultiTenantEngine>, read_only: bool },
}

/// Routes for single-tenant mode
pub fn routes(engine: std::sync::Arc<CueMapEngine>, auth_config: AuthConfig, read_only: bool) -> Router {
    let mut router = Router::new()
        .route("/", get(root))
        .route("/memories", post(add_memory))
        .route("/recall", post(recall))
        .route("/memories/:id/reinforce", patch(reinforce_memory))
        .route("/memories/:id", get(get_memory))
        .route("/stats", get(get_stats))
        .with_state(EngineState::SingleTenant { 
            engine: CueMapEngine::clone(&engine),
            read_only 
        });
    
    // Add auth middleware if enabled
    if auth_config.is_enabled() {
        router = router.layer(middleware::from_fn_with_state(auth_config, crate::auth::auth_middleware));
    }
    
    router
}

/// Routes for multi-tenant mode
pub fn routes_with_mt_engine(mt_engine: Arc<MultiTenantEngine>, auth_config: AuthConfig, read_only: bool) -> Router {
    let mut router = Router::new()
        .route("/", get(root))
        .route("/memories", post(add_memory_mt))
        .route("/recall", post(recall_mt))
        .route("/memories/:id/reinforce", patch(reinforce_memory_mt))
        .route("/memories/:id", get(get_memory_mt))
        .route("/stats", get(get_stats_mt))
        .route("/projects", get(list_projects))
        .route("/projects/:id", delete(delete_project))
        .with_state(EngineState::MultiTenant { 
            mt_engine,
            read_only 
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
    if let EngineState::SingleTenant { engine, read_only } = state {
        // Check if read-only
        if read_only {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({
                    "error": "Read-only mode: modifications are not allowed"
                })),
            );
        }
        
        let memory_id = engine.add_memory(req.content, req.cues, req.metadata);
        
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": memory_id,
                "status": "stored"
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
    if let EngineState::SingleTenant { engine, .. } = state {
        let results = engine.recall(req.cues, req.limit, req.auto_reinforce);
        (StatusCode::OK, Json(serde_json::json!({ "results": results })))
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
    if let EngineState::SingleTenant { engine, read_only } = state {
        // Check if read-only
        if read_only {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({
                    "error": "Read-only mode: modifications are not allowed"
                })),
            );
        }
        
        let success = engine.reinforce_memory(&memory_id, req.cues);
        
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
    if let EngineState::SingleTenant { engine, .. } = state {
        match engine.get_memory(&memory_id) {
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
    if let EngineState::SingleTenant { engine, .. } = state {
        let stats = engine.get_stats();
        (StatusCode::OK, Json(serde_json::Value::Object(stats.into_iter().collect())))
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Invalid state"})),
        )
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
    
    if let EngineState::MultiTenant { mt_engine, read_only } = state {
        // Check if read-only
        if read_only {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({
                    "error": "Read-only mode: modifications are not allowed"
                })),
            );
        }
        
        let engine = mt_engine.get_or_create_project(project_id.clone());
        let memory_id = engine.add_memory(req.content.clone(), req.cues.clone(), req.metadata);
        
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
                "status": "stored"
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
                    let engine = mt_engine.get_or_create_project(project_id.clone());
                    let results = engine.recall_with_min_intersection(
                        req.cues.clone(), 
                        req.limit, 
                        false,
                        req.min_intersection
                    );
                    
                    let json_results: Vec<serde_json::Value> = results
                        .into_iter()
                        .map(|r| serde_json::json!({
                            "id": r.memory_id,
                            "content": r.content,
                            "score": r.score,
                            "intersection_count": r.intersection_count,
                            "recency_score": r.recency_score,
                            "metadata": r.metadata
                        }))
                        .collect();
                    
                    serde_json::json!({
                        "project_id": project_id,
                        "results": json_results
                    })
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
        let engine = mt_engine.get_or_create_project(project_id.clone());
        let results = engine.recall_with_min_intersection(
            req.cues.clone(), 
            req.limit, 
            req.auto_reinforce,
            req.min_intersection
        );
        let elapsed = start.elapsed();
        
        let engine_latency_ms = elapsed.as_secs_f64() * 1000.0;
        
        tracing::info!(
            "POST /recall project={} cues={} results={} latency={:.2}ms",
            project_id,
            req.cues.len(),
            results.len(),
            engine_latency_ms
        );
        
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
        let engine = mt_engine.get_or_create_project(project_id);
        let success = engine.reinforce_memory(&memory_id, req.cues);
        
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
        let engine = mt_engine.get_or_create_project(project_id);
        match engine.get_memory(&memory_id) {
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
        let engine = mt_engine.get_or_create_project(project_id);
        let stats = engine.get_stats();
        (StatusCode::OK, Json(serde_json::Value::Object(stats.into_iter().collect())))
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Invalid state"})),
        )
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
