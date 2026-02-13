use serde::{Deserialize, Serialize};
use cuemap::engine::RecallResult;

pub mod runner;
pub mod evals;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalQuery {
    pub name: String,
    pub input: RecallRequest,
    pub expected: Option<GoldenTrace>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallRequest {
    pub query_text: Option<String>,
    pub cues: Vec<String>,
    pub limit: usize,
    pub auto_reinforce: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalizedRecall {
    pub ordered_ids: Vec<String>,
    pub scores: Vec<f64>,
    pub intersection_counts: Vec<usize>,
    pub explain: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenTrace {
    pub query: String,
    pub normalized_query: Vec<String>,
    pub results: Vec<TraceResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceResult {
    pub memory_id: String,
    pub rank: usize,
    pub score: f64,
    pub intersection_count: usize,
}

impl From<Vec<RecallResult>> for NormalizedRecall {
    fn from(results: Vec<RecallResult>) -> Self {
        let ordered_ids = results.iter().map(|r| r.memory_id.clone()).collect();
        let scores = results.iter().map(|r| r.score).collect();
        let intersection_counts = results.iter().map(|r| r.intersection_count).collect();
        
        // We only take the explain from the first result if it exists (simplification for trace)
        // or maybe we don't include explain in normalized recall for assertion unless specified?
        // Spec says: "explain: Option<ExplainBlock>"
        let explain = results.first().and_then(|r| r.explain.clone());

        Self {
            ordered_ids,
            scores,
            intersection_counts,
            explain,
        }
    }
}
