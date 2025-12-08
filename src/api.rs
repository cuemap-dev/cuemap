use crate::engine::CueMapEngine;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, patch, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

pub fn routes(engine: std::sync::Arc<CueMapEngine>, multi_tenant: bool) -> Router {
    if multi_tenant {
        // Multi-tenant routes will be added in next iteration
        // For now, just use single-tenant
        Router::new()
            .route("/", get(root))
            .route("/memories", post(add_memory))
            .route("/recall", post(recall))
            .route("/memories/:id/reinforce", patch(reinforce_memory))
            .route("/memories/:id", get(get_memory))
            .route("/stats", get(get_stats))
            .with_state(CueMapEngine::clone(&engine))
    } else {
        Router::new()
            .route("/", get(root))
            .route("/memories", post(add_memory))
            .route("/recall", post(recall))
            .route("/memories/:id/reinforce", patch(reinforce_memory))
            .route("/memories/:id", get(get_memory))
            .route("/stats", get(get_stats))
            .with_state(CueMapEngine::clone(&engine))
    }
}

async fn root() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "CueMap Rust Engine",
        "version": "0.1.0",
        "description": "High-performance Temporal-Associative Memory Store"
    }))
}

async fn add_memory(
    State(engine): State<CueMapEngine>,
    Json(req): Json<AddMemoryRequest>,
) -> impl IntoResponse {
    let memory_id = engine.add_memory(req.content, req.cues, req.metadata);
    
    (
        StatusCode::OK,
        Json(AddMemoryResponse {
            id: memory_id,
            status: "stored".to_string(),
        }),
    )
}

async fn recall(
    State(engine): State<CueMapEngine>,
    Json(req): Json<RecallRequest>,
) -> impl IntoResponse {
    let results = engine.recall(req.cues, req.limit, req.auto_reinforce);
    
    (StatusCode::OK, Json(serde_json::json!({ "results": results })))
}

async fn reinforce_memory(
    State(engine): State<CueMapEngine>,
    Path(memory_id): Path<String>,
    Json(req): Json<ReinforceRequest>,
) -> impl IntoResponse {
    let success = engine.reinforce_memory(&memory_id, req.cues);
    
    if success {
        (
            StatusCode::OK,
            Json(ReinforceResponse {
                status: "reinforced".to_string(),
                memory_id,
            }),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(ReinforceResponse {
                status: "not_found".to_string(),
                memory_id,
            }),
        )
    }
}

async fn get_memory(
    State(engine): State<CueMapEngine>,
    Path(memory_id): Path<String>,
) -> impl IntoResponse {
    match engine.get_memory(&memory_id) {
        Some(memory) => (StatusCode::OK, Json(serde_json::json!(memory))),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Memory not found"})),
        ),
    }
}

async fn get_stats(State(engine): State<CueMapEngine>) -> impl IntoResponse {
    let stats = engine.get_stats();
    (StatusCode::OK, Json(stats))
}
