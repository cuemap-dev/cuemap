use crate::engine::RecallResult;
use crate::structures::MemoryId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedItem {
    pub memory_id: MemoryId,
    pub content: String,
    pub score: f64,
    pub intersection_count: usize,
    pub recency_component: f64,
    pub reinforcement_component: f64,
    pub match_integrity: f64,
    pub source: String,
    pub timestamp: String,
    pub estimated_tokens: u32,
    pub why: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcludedItem {
    pub memory_id: MemoryId,
    pub score: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingProof {
    pub trace_id: String,
    pub query_text: String,
    pub normalized_query: Vec<String>,
    pub expanded_cues: Vec<(String, f64)>,
    pub token_budget: u32,
    pub selected: Vec<SelectedItem>,
    pub excluded_top: Vec<ExcludedItem>,
}

pub struct GroundingEngine;

impl GroundingEngine {
    /// Estimates tokens based on character count (1 token ~= 4 chars)
    pub fn estimate_tokens(content: &str) -> u32 {
        ((content.len() as f64) / 4.0).ceil() as u32
    }

    pub fn select_memories(
        _query_text: String,
        _normalized_query: Vec<String>,
        _expanded_cues: Vec<(String, f64)>,
        results: Vec<RecallResult>,
        token_budget: u32,
    ) -> (Vec<SelectedItem>, Vec<ExcludedItem>, String) {
        let mut selected = Vec::new();
        let mut excluded_top = Vec::new();
        let mut current_tokens = 0;

        for result in results {
            let tokens = Self::estimate_tokens(&result.content);

            if current_tokens + tokens <= token_budget {
                let source = result
                    .metadata
                    .get("source")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let timestamp = result
                    .metadata
                    .get("timestamp")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| {
                        let secs = result.created_at as i64;
                        let nanos = ((result.created_at - secs as f64) * 1_000_000_000.0) as u32;
                        if let Some(dt) = chrono::DateTime::from_timestamp(secs, nanos) {
                            dt.format("%Y-%m-%dT%H:%M:%SZ").to_string()
                        } else {
                            "unknown".to_string()
                        }
                    });

                let why = format!(
                    "Ranked #{} with score {:.2} ({} matches, integrity {:.2})",
                    selected.len() + 1,
                    result.score,
                    result.intersection_count,
                    result.match_integrity
                );

                selected.push(SelectedItem {
                    memory_id: result.memory_id,
                    content: result.content,
                    score: result.score,
                    intersection_count: result.intersection_count,
                    recency_component: result.recency_score,
                    reinforcement_component: result.reinforcement_score,
                    match_integrity: result.match_integrity,
                    source,
                    timestamp,
                    estimated_tokens: tokens,
                    why,
                });
                current_tokens += tokens;
            } else {
                if excluded_top.len() < 5 {
                    // Only track top 5 exclusions
                    excluded_top.push(ExcludedItem {
                        memory_id: result.memory_id,
                        score: result.score,
                        reason: format!(
                            "Exceeds remaining token budget (needs {}, has {})",
                            tokens,
                            token_budget - current_tokens
                        ),
                    });
                }
            }
        }

        let context_block = Self::format_context_block(&selected);
        (selected, excluded_top, context_block)
    }

    pub fn format_context_block(selected: &[SelectedItem]) -> String {
        if selected.is_empty() {
            return "".to_string();
        }

        let mut block = String::from("[VERIFIED CONTEXT]\n");
        for (idx, item) in selected.iter().enumerate() {
            block.push_str(&format!(
                "({}) {} (source={}, id={}, score={:.2}, ts={})\n",
                idx + 1,
                item.content,
                item.source,
                item.memory_id,
                item.score,
                item.timestamp
            ));
        }
        block.push_str("[/VERIFIED CONTEXT]");
        block
    }
}

pub fn create_grounding_proof(
    trace_id: String,
    query_text: String,
    normalized_query: Vec<String>,
    expanded_cues: Vec<(String, f64)>,
    token_budget: u32,
    selected: Vec<SelectedItem>,
    excluded_top: Vec<ExcludedItem>,
) -> GroundingProof {
    GroundingProof {
        trace_id,
        query_text,
        normalized_query,
        expanded_cues,
        token_budget,
        selected,
        excluded_top,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn result(id: MemoryId, content: &str, metadata: HashMap<String, serde_json::Value>) -> RecallResult {
        RecallResult {
            memory_id: id,
            content: content.to_string(),
            score: 0.9,
            match_integrity: 0.8,
            intersection_count: 2,
            recency_score: 0.3,
            reinforcement_score: 0.4,
            salience_score: 0.5,
            created_at: 1_700_000_000.0,
            metadata,
            explain: None,
        }
    }

    #[test]
    fn estimates_tokens_from_character_count() {
        assert_eq!(GroundingEngine::estimate_tokens(""), 0);
        assert_eq!(GroundingEngine::estimate_tokens("1234"), 1);
        assert_eq!(GroundingEngine::estimate_tokens("12345"), 2);
    }

    #[test]
    fn selects_items_with_metadata_and_formats_context() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!("notes.md"));
        metadata.insert("timestamp".to_string(), json!("2024-01-01T00:00:00Z"));
        let (selected, excluded, block) = GroundingEngine::select_memories(
            "query".to_string(),
            vec!["query".to_string()],
            vec![("query".to_string(), 1.0)],
            vec![result(7, "1234", metadata)],
            1,
        );

        assert_eq!(selected.len(), 1);
        assert!(excluded.is_empty());
        assert_eq!(selected[0].memory_id, 7);
        assert_eq!(selected[0].source, "notes.md");
        assert_eq!(selected[0].timestamp, "2024-01-01T00:00:00Z");
        assert!(block.contains("[VERIFIED CONTEXT]"));
        assert!(block.contains("notes.md"));
        assert!(block.ends_with("[/VERIFIED CONTEXT]"));
    }

    #[test]
    fn excludes_top_five_items_that_exceed_budget_and_falls_back_to_timestamp() {
        let results = (1..=7)
            .map(|id| result(id, "12345", HashMap::new()))
            .collect();
        let (selected, excluded, block) = GroundingEngine::select_memories(
            "query".to_string(),
            Vec::new(),
            Vec::new(),
            results,
            2,
        );

        assert_eq!(selected.len(), 1);
        assert_eq!(excluded.len(), 5);
        assert!(excluded[0].reason.contains("Exceeds remaining token budget"));
        assert!(selected[0].timestamp.starts_with("2023-11-"));
        assert!(!block.is_empty());
        assert!(GroundingEngine::format_context_block(&[]).is_empty());
    }

    #[test]
    fn creates_a_serializable_grounding_proof() {
        let proof = create_grounding_proof(
            "trace-1".to_string(),
            "what?".to_string(),
            vec!["what".to_string()],
            vec![("what".to_string(), 0.75)],
            128,
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(proof.trace_id, "trace-1");
        assert_eq!(proof.token_budget, 128);
        assert_eq!(serde_json::to_value(&proof).unwrap()["query_text"], "what?");
    }
}
