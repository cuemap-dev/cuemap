use crate::config::*;
use crate::structures::{Memory, OrderedSet};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize)]
pub struct RecallResult {
    pub memory_id: String,
    pub content: String,
    pub score: f64,
    pub intersection_count: usize,
    pub recency_score: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Clone)]
pub struct CueMapEngine {
    memories: Arc<DashMap<String, Memory>>,
    cue_index: Arc<DashMap<String, OrderedSet>>,
}

impl CueMapEngine {
    pub fn new() -> Self {
        Self {
            memories: Arc::new(DashMap::new()),
            cue_index: Arc::new(DashMap::new()),
        }
    }
    
    pub fn from_state(
        memories: DashMap<String, Memory>,
        cue_index: DashMap<String, OrderedSet>,
    ) -> Self {
        Self {
            memories: Arc::new(memories),
            cue_index: Arc::new(cue_index),
        }
    }
    
    // Expose internal state for persistence
    pub fn get_memories(&self) -> &Arc<DashMap<String, Memory>> {
        &self.memories
    }
    
    pub fn get_cue_index(&self) -> &Arc<DashMap<String, OrderedSet>> {
        &self.cue_index
    }
    
    pub fn add_memory(
        &self,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> String {
        let memory = Memory::new(content, metadata);
        let memory_id = memory.id.clone();
        
        // Store memory
        self.memories.insert(memory_id.clone(), memory);
        
        // Index by cues
        for cue in cues {
            let cue_lower = cue.to_lowercase().trim().to_string();
            if !cue_lower.is_empty() {
                self.cue_index
                    .entry(cue_lower)
                    .or_insert_with(OrderedSet::new)
                    .add(memory_id.clone());
            }
        }
        
        memory_id
    }
    
    pub fn reinforce_memory(&self, memory_id: &str, cues: Vec<String>) -> bool {
        // Update last accessed
        if let Some(mut memory) = self.memories.get_mut(memory_id) {
            memory.touch();
        } else {
            return false;
        }
        
        // Move to front for each cue
        for cue in cues {
            let cue_lower = cue.to_lowercase().trim().to_string();
            if !cue_lower.is_empty() {
                let mut entry = self.cue_index
                    .entry(cue_lower)
                    .or_insert_with(OrderedSet::new);
                entry.move_to_front(memory_id);
            }
        }
        
        true
    }
    
    pub fn recall(
        &self,
        query_cues: Vec<String>,
        limit: usize,
        auto_reinforce: bool,
    ) -> Vec<RecallResult> {
        if query_cues.is_empty() {
            return Vec::new();
        }
        
        // Normalize cues
        let cues: Vec<String> = query_cues
            .iter()
            .map(|c| c.to_lowercase().trim().to_string())
            .filter(|c| !c.is_empty() && self.cue_index.contains_key(c))
            .collect();
        
        if cues.is_empty() {
            return Vec::new();
        }
        
        // Iterative deepening search
        let results = self.iterative_search(&cues, limit);
        
        // Auto-reinforce if enabled
        if auto_reinforce {
            for result in &results {
                self.reinforce_memory(&result.memory_id, query_cues.clone());
            }
        }
        
        results
    }
    
    fn iterative_search(&self, cues: &[String], limit: usize) -> Vec<RecallResult> {
        let tiers = [
            (0, TIER_1_DEPTH),
            (TIER_1_DEPTH, TIER_2_DEPTH),
            (TIER_2_DEPTH, MAX_SEARCH_DEPTH),
        ];
        
        for (tier_start, tier_end) in tiers {
            let candidates = self.gather_candidates(cues, tier_start, tier_end);
            
            if !candidates.is_empty() {
                let mut scored = self.score_candidates(candidates);
                
                // Filter by intersection threshold (inline for performance)
                scored.retain(|c| c.intersection_count >= 1);
                
                if !scored.is_empty() {
                    // Use unstable sort for better performance (order of equal elements doesn't matter)
                    scored.sort_unstable_by(|a, b| {
                        b.intersection_count
                            .cmp(&a.intersection_count)
                            .then_with(|| {
                                b.recency_score
                                    .partial_cmp(&a.recency_score)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                    });
                    
                    scored.truncate(limit);
                    return scored;
                }
            }
        }
        
        Vec::new()
    }
    
    fn gather_candidates(
        &self,
        cues: &[String],
        start: usize,
        end: usize,
    ) -> HashMap<String, Vec<usize>> {
        // Pre-allocate with estimated capacity
        let mut candidates: HashMap<String, Vec<usize>> = HashMap::with_capacity(end - start);
        
        for cue in cues {
            if let Some(ordered_set) = self.cue_index.get(cue) {
                let memories = ordered_set.get_recent(Some(end));
                
                // Only iterate over the range we care about
                for (idx, memory_id) in memories.iter().enumerate().skip(start).take(end - start) {
                    candidates
                        .entry(memory_id.clone())
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
            }
        }
        
        candidates
    }
    
    fn score_candidates(&self, candidates: HashMap<String, Vec<usize>>) -> Vec<RecallResult> {
        // Pre-allocate result vector
        let mut results = Vec::with_capacity(candidates.len());
        
        for (memory_id, positions) in candidates {
            if let Some(memory) = self.memories.get(&memory_id) {
                let intersection_count = positions.len();
                
                // Recency score: average of inverted positions (optimized)
                let pos_len = positions.len() as f64;
                let recency_score: f64 = positions
                    .iter()
                    .map(|&p| 1.0 / (p as f64 + 1.0))
                    .sum::<f64>()
                    / pos_len;
                
                // Combined score
                let score = (intersection_count as f64 * 10.0) + recency_score;
                
                results.push(RecallResult {
                    memory_id: memory_id.clone(),
                    content: memory.content.clone(),
                    score,
                    intersection_count,
                    recency_score,
                    metadata: memory.metadata.clone(),
                });
            }
        }
        
        results
    }
    
    pub fn get_memory(&self, memory_id: &str) -> Option<Memory> {
        self.memories.get(memory_id).map(|m| m.clone())
    }
    
    pub fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        stats.insert(
            "total_memories".to_string(),
            serde_json::json!(self.memories.len()),
        );
        stats.insert(
            "total_cues".to_string(),
            serde_json::json!(self.cue_index.len()),
        );
        
        let cues: Vec<String> = self.cue_index.iter().map(|e| e.key().clone()).collect();
        stats.insert("cues".to_string(), serde_json::json!(cues));
        
        stats
    }
}
