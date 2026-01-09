use crate::config::*;
use crate::structures::{Memory, OrderedSet};
use dashmap::DashMap;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Serialize)]
pub struct RecallResult {
    pub memory_id: String,
    pub content: String,
    pub score: f64,
    pub match_integrity: f64,
    pub intersection_count: usize,
    pub recency_score: f64,
    pub reinforcement_score: f64,
    pub salience_score: f64,
    pub created_at: f64,  // Timestamp when memory was created
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explain: Option<serde_json::Value>,
}

#[derive(Clone)]
pub struct CueMapEngine {
    memories: Arc<DashMap<String, Memory>>,
    cue_index: Arc<DashMap<String, OrderedSet>>,
    // Pattern Completion: cue co-occurrence matrix
    cue_co_occurrence: Arc<DashMap<String, DashMap<String, u64>>>,
    // Temporal Chunking: track last event per session/project (using a dummy key for now or extending API)
    last_events: Arc<DashMap<String, (String, f64, Vec<String>)>>,
    // Performance: Atomic counter to avoid DashMap::len() contention
    memory_count: Arc<AtomicUsize>,
}

impl CueMapEngine {
    pub fn new() -> Self {
        Self {
            memories: Arc::new(DashMap::new()),
            cue_index: Arc::new(DashMap::new()),
            cue_co_occurrence: Arc::new(DashMap::new()),
            last_events: Arc::new(DashMap::new()),
            memory_count: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    pub fn from_state(
        memories: DashMap<String, Memory>,
        cue_index: DashMap<String, OrderedSet>,
    ) -> Self {
        let count = memories.len();
        let engine = Self {
            memories: Arc::new(memories),
            cue_index: Arc::new(cue_index),
            cue_co_occurrence: Arc::new(DashMap::new()), 
            last_events: Arc::new(DashMap::new()),
            memory_count: Arc::new(AtomicUsize::new(count)),
        };

        // Rehydrate co-occurrence matrix from existing memories
        // This ensures the graph and pattern completion work after restart
        for r in engine.memories.iter() {
            let memory = r.value();
            engine.update_cue_co_occurrence(&memory.cues);
        }

        engine
    }
    
    // Expose internal state for persistence
    pub fn get_memories(&self) -> &Arc<DashMap<String, Memory>> {
        &self.memories
    }
    
    pub fn get_cue_index(&self) -> &Arc<DashMap<String, OrderedSet>> {
        &self.cue_index
    }
    
    pub fn update_cue_co_occurrence(&self, cues: &[String]) {
        for i in 0..cues.len() {
            let cue_a = cues[i].to_lowercase().trim().to_string();
            if cue_a.is_empty() { continue; }
            
            for j in (i + 1)..cues.len() {
                let cue_b = cues[j].to_lowercase().trim().to_string();
                if cue_b.is_empty() || cue_a == cue_b { continue; }
                
                // Update A -> B
                self.cue_co_occurrence
                    .entry(cue_a.clone())
                    .or_insert_with(DashMap::new)
                    .entry(cue_b.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                
                // Update B -> A
                self.cue_co_occurrence
                    .entry(cue_b.clone())
                    .or_insert_with(DashMap::new)
                    .entry(cue_a.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }
    }

    pub fn add_memory(
        &self,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        disable_temporal_chunking: bool,
    ) -> String {
        let mut memory = Memory::new(content, metadata);
        let memory_id = memory.id.clone();
        
        // Store cues in memory
        memory.cues = cues.clone();
        
        // 1. Salience calculation (proxies)
        // High cue density boost
        let cue_density = if !memory.content.is_empty() {
            (memory.cues.len() as f64) / (memory.content.len() as f64).sqrt()
        } else {
            0.0
        };
        memory.salience += cue_density;
        
        // Rare cue combinations boost (simulated by cue count for now)
        if memory.cues.len() > 5 {
            memory.salience += 0.5;
        }

        // 2. Temporal Chunking
        let project_id = memory.metadata.get("project_id")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();
        
        if let Some(last_event) = self.last_events.get(&project_id) {
            let (last_id, last_time, last_cues) = last_event.clone();
            let now = memory.created_at;
            
            // Time proximity (< 5 mins) and High cue overlap (> 50%)
            let time_diff = now - last_time;
            let overlap = memory.cues.iter().filter(|c| last_cues.contains(c)).count();
            let overlap_ratio = if !memory.cues.is_empty() {
                (overlap as f64) / (memory.cues.len() as f64)
            } else {
                0.0
            };
            
            if time_diff < 300.0 && overlap_ratio > 0.5 && !disable_temporal_chunking {
                let episode_cue = format!("episode:{}", last_id);
                memory.cues.push(episode_cue.clone());
            }
        }
        self.last_events.insert(project_id, (memory_id.clone(), memory.created_at, memory.cues.clone()));

        // 3. Update co-occurrence matrix (MOVED TO BACKGROUND JOB)
        // self.update_cue_co_occurrence(&memory.cues);
        
        if self.memories.insert(memory_id.clone(), memory).is_none() {
            self.memory_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Index by cues (Double Indexing)
        for cue in &cues {
            let cue_lower = cue.to_lowercase().trim().to_string();
            if cue_lower.is_empty() { continue; }

            // 1. Index full cue
            self.cue_index
                .entry(cue_lower.clone())
                .or_insert_with(OrderedSet::new)
                .add(memory_id.clone());
            
            // 2. Index value if k:v (unless flat)
            if let Some((_, value)) = cue_lower.split_once(':') {
                if !value.is_empty() {
                    self.cue_index
                        .entry(value.to_string())
                        .or_insert_with(OrderedSet::new)
                        .add(memory_id.clone());
                }
            }
        }
        
        memory_id
    }
    
    pub fn reinforce_memory(&self, memory_id: &str, cues: Vec<String>) -> bool {
        // Update last accessed
        if let Some(mut memory) = self.memories.get_mut(memory_id) {
            memory.touch();
            memory.salience += 0.1; // Manual reinforcement boost
        } else {
            return false;
        }
        
        // Update co-occurrence matrix with cues used for reinforcement
        self.update_cue_co_occurrence(&cues);

        // Move to front for each cue (Double Indexing)
        for cue in cues {
            let cue_lower = cue.to_lowercase().trim().to_string();
            if cue_lower.is_empty() { continue; }

            // 1. Move full cue
            if let Some(mut entry) = self.cue_index.get_mut(&cue_lower) {
                entry.move_to_front(memory_id);
            }
            
            // 2. Move value
            if let Some((_, value)) = cue_lower.split_once(':') {
                if !value.is_empty() {
                    if let Some(mut entry) = self.cue_index.get_mut(value) {
                        entry.move_to_front(memory_id);
                    }
                }
            }
        }
        
        true
    }

    pub fn delete_memory(&self, memory_id: &str) -> bool {
        if let Some((_, memory)) = self.memories.remove(memory_id) {
             self.memory_count.fetch_sub(1, Ordering::Relaxed);
             // Remove from cue index (Double Indexing)
             for cue in memory.cues {
                 let cue_lower = cue.to_lowercase().trim().to_string();
                 if cue_lower.is_empty() { continue; }
                 
                 // 1. Remove from full cue entry
                 if let Some(mut entry) = self.cue_index.get_mut(&cue_lower) {
                     entry.remove(memory_id);
                 }
                 
                 // 2. Remove from value entry
                 if let Some((_, value)) = cue_lower.split_once(':') {
                     if !value.is_empty() {
                         if let Some(mut entry) = self.cue_index.get_mut(value) {
                             entry.remove(memory_id);
                         }
                     }
                 }
             }
            true
        } else {
            false
        }
    }

    pub fn get_cue_frequency(&self, cue: &str) -> usize {
        let cue_lower = cue.to_lowercase();
        let cue_trimmed = cue_lower.trim();
        if let Some(set) = self.cue_index.get(cue_trimmed) {
            set.len()
        } else {
            0
        }
    }
    
    pub fn total_memories(&self) -> usize {
        self.memory_count.load(Ordering::Relaxed)
    }

    pub fn upsert_memory_with_id(
        &self,
        id: String,
        content: String,
        cues: Vec<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        reinforce: bool,
    ) -> String {
        // If exists: attach cues + optionally touch
        if self.memories.contains_key(&id) {
            self.attach_cues(&id, cues.clone());
            if reinforce {
                self.reinforce_memory(&id, cues);
            }
            return id;
        }
        
        // Insert new
        let mut memory = Memory::new(content, metadata);
        memory.id = id.clone();
        memory.cues = cues.clone();
        
        self.memories.insert(id.clone(), memory);
        
        // Index by cues (Double Indexing)
        for cue in &cues { // Iterate by reference to avoid move
            let cue_lower = cue.to_lowercase().trim().to_string();
            if cue_lower.is_empty() { continue; }
            
            // 1. Index full cue
            self.cue_index
                .entry(cue_lower.clone())
                .or_insert_with(OrderedSet::new)
                .add(id.clone());
            
            // 2. Index value
            if let Some((_, value)) = cue_lower.split_once(':') {
                if !value.is_empty() {
                    self.cue_index
                        .entry(value.to_string())
                        .or_insert_with(OrderedSet::new)
                        .add(id.clone());
                }
            }
        }
        
        // FIX: Update co-occurrence matrix for new memory
        self.update_cue_co_occurrence(&cues);
        
        id
    }

    pub fn attach_cues(&self, memory_id: &str, cues: Vec<String>) -> bool {
        // 1. Get memory and check if it exists
        if let Some(mut memory) = self.memories.get_mut(memory_id) {
            // 2. Identify new cues (deduplication)
            let mut new_cues = Vec::new();
            for cue in cues {
                if !memory.cues.contains(&cue) {
                    new_cues.push(cue);
                }
            }

            if new_cues.is_empty() {
                return false;
            }

            // 3. Update memory.cues
            memory.cues.extend(new_cues.clone());

            // 4. Update index for new cues (Double Indexing)
            for cue in new_cues {
                let cue_lower = cue.to_lowercase().trim().to_string();
                if cue_lower.is_empty() { continue; }
                
                // 1. Index full cue
                self.cue_index
                    .entry(cue_lower.clone())
                    .or_insert_with(OrderedSet::new)
                    .add(memory_id.to_string());
                
                // 2. Index value
                if let Some((_, value)) = cue_lower.split_once(':') {
                    if !value.is_empty() {
                        self.cue_index
                            .entry(value.to_string())
                            .or_insert_with(OrderedSet::new)
                            .add(memory_id.to_string());
                    }
                }
            }
            
            // FIX: Update co-occurrence with extended cue set
            // We pass ALL cues to reinforce associations between old and new cues
            let all_cues = memory.cues.clone();
            drop(memory); // Release lock before calling update (though update uses different map, safer)
            self.update_cue_co_occurrence(&all_cues);
            
            return true;
        } else {
            false
        }
    }
    
    pub fn recall(
        &self,
        query_cues: Vec<String>,
        limit: usize,
        auto_reinforce: bool,
    ) -> Vec<RecallResult> {
        self.recall_with_min_intersection(query_cues, limit, auto_reinforce, None)
    }
    
    pub fn recall_with_min_intersection(
        &self,
        query_cues: Vec<String>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
    ) -> Vec<RecallResult> {
        if query_cues.is_empty() {
            return Vec::new();
        }
        
        // Default weight of 1.0 for standard recall
        let weighted_cues: Vec<(String, f64)> = query_cues
            .into_iter()
            .map(|c| (c, 1.0))
            .collect();
            
        self.recall_weighted(weighted_cues, limit, auto_reinforce, min_intersection, false, false, false, false)
    }

    /// Fast O(1) lookup for lexicon-style queries.
    /// Returns memories that match ANY query cue, ordered by recency.
    /// No scoring, no pattern completion - just direct index lookup.
    pub fn recall_fast(&self, query_cues: Vec<String>, limit: usize) -> Vec<RecallResult> {
        if query_cues.is_empty() {
            return Vec::new();
        }
        
        // We need to collect ALL candidates first to sort them
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();
        
        for cue in query_cues {
            let cue_lower = cue.to_lowercase();
            let cue_trimmed = cue_lower.trim();
            if cue_trimmed.is_empty() { continue; }
            
            if let Some(ordered_set) = self.cue_index.get(cue_trimmed) {
                // Grab more than the limit initially (2x limit) to allow for sorting
                for memory_id in ordered_set.get_recent(Some(limit * 2)) {
                    if seen.contains(memory_id) { continue; }
                    seen.insert(memory_id.clone());
                    
                    if let Some(memory) = self.memories.get(memory_id) {
                        candidates.push(RecallResult {
                            memory_id: memory_id.clone(),
                            content: memory.content.clone(),
                            score: 1.0,
                            match_integrity: 1.0,
                            intersection_count: 1,
                            recency_score: 1.0,
                            reinforcement_score: memory.reinforcement_count as f64,
                            salience_score: memory.salience,
                            created_at: memory.created_at,
                            metadata: memory.metadata.clone(),
                            explain: None,
                        });
                    }
                }
            }
        }
        
        // Sort by Hierarchy of Signals (Cascading Sort)
        candidates.sort_by(|a, b| {
            // 1. Primary: Learned Relevance (Hebbian) - "What have I successfully recalled before?"
            b.reinforcement_score.partial_cmp(&a.reinforcement_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                
            // 2. Secondary: Intrinsic Value (Amygdala) - "Which memory has rarer/richer cues?"
            // This SOLVES the Cold Start. "Lemon Cheesecake" (rare) > "Food" (common).
            .then_with(|| {
                b.salience_score.partial_cmp(&a.salience_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            
            // 3. Tertiary: Freshness (Temporal) - "If both are unreinforced and equally salient, show the new one."
            .then_with(|| {
                b.created_at.partial_cmp(&a.created_at)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        
        candidates.into_iter().take(limit).collect()
    }

    pub fn recall_weighted(
        &self,
        query_cues: Vec<(String, f64)>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        explain: bool,
        disable_pattern_completion: bool,
        disable_salience_bias: bool,
        disable_systems_consolidation: bool,
    ) -> Vec<RecallResult> {
        if query_cues.is_empty() {
            return Vec::new();
        }
        
        // Normalize primary cues
        let mut active_cues: Vec<(String, f64)> = query_cues
            .iter()
            .map(|(c, w)| (c.to_lowercase().trim().to_string(), *w))
            .filter(|(c, _)| !c.is_empty() && self.cue_index.contains_key(c))
            .collect();
        
        if active_cues.is_empty() {
            return Vec::new();
        }

        // 1. Pattern Completion (Hippocampal CA3)
        // Find cues that strongly co-occur with the query cues
        if !disable_pattern_completion {
            let mut inferred_candidates: HashMap<String, u64> = HashMap::new();
            for (cue, _) in &active_cues {
                if let Some(co_map) = self.cue_co_occurrence.get(cue) {
                for entry in co_map.iter() {
                    let (inferred_cue, count) = entry.pair();
                    // Skip if already in query
                    if active_cues.iter().any(|(c, _)| c == inferred_cue) {
                        continue;
                    }
                    *inferred_candidates.entry(inferred_cue.clone()).or_insert(0) += *count;
                }
            }
        }

        // Take top-K inferred cues and inject them with low weight
            let mut inferred_list: Vec<(String, u64)> = inferred_candidates.into_iter().collect();
            inferred_list.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            
            let pattern_completion_weight = 0.7; // Weight for inferred cues
            for (inf_cue, _) in inferred_list.into_iter().take(5) {
                active_cues.push((inf_cue, pattern_completion_weight));
            }
        }
        
        // 2. Consolidated search using Selective Set Intersection
        let mut results = self.consolidated_search(&active_cues, limit, explain, disable_salience_bias, disable_systems_consolidation);
        
        // Filter by minimum intersection if specified (on primary cues only?)
        // For now, simple retention.
        if let Some(min_int) = min_intersection {
            results.retain(|r| r.intersection_count >= min_int);
        }
        
        // 3. Auto-reinforce if enabled (only primary cues)
        if auto_reinforce {
            let primary_cues: Vec<String> = query_cues.iter().map(|(c, _)| c.clone()).collect();
            for result in &results {
                self.reinforce_memory(&result.memory_id, primary_cues.clone());
            }
        }

        // Global sort by score
        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit);
        
        results
    }
    
    fn consolidated_search(&self, query_cues: &[(String, f64)], _limit: usize, explain: bool, disable_salience_bias: bool, disable_systems_consolidation: bool) -> Vec<RecallResult> {
        if query_cues.is_empty() {
            return Vec::new();
        }

        // 1. Gather cue data
        let mut cue_data = Vec::with_capacity(query_cues.len());
        for (cue, weight) in query_cues {
            if let Some(ordered_set) = self.cue_index.get(cue) {
                cue_data.push((cue.clone(), *weight, ordered_set));
            }
        }

        if cue_data.is_empty() {
            return Vec::new();
        }

        // 2. Perform Union-based search with O(1) Probing
        // We iterate through EVERY cue's list up to MAX_DRIVER_SCAN to ensure partial matches are found.
        let mut candidates = Vec::new();
        let mut seen_memories = HashSet::new();

        for (cue_idx, (_cue, _weight, set)) in cue_data.iter().enumerate() {
            let scan_limit = std::cmp::min(set.len(), MAX_DRIVER_SCAN);
            let items = set.get_recent(Some(scan_limit));

            for (pos_rev, memory_id) in items.iter().enumerate() {
                // If we've already processed this memory from a previous (likely more selective or relevant) cue, skip it
                if seen_memories.contains(*memory_id) {
                    continue;
                }
                seen_memories.insert((*memory_id).clone());

                let mut total_weight = 0.0;
                let mut positions_info = Vec::with_capacity(cue_data.len());

                // 3. For each NEW candidate, probe ALL query cue lists to get full intersection data
                for (other_idx, (_other_cue, other_weight, other_set)) in cue_data.iter().enumerate() {
                    // Optimization: if it's the current set we're iterating, we know it's there
                    if other_idx == cue_idx {
                        total_weight += *other_weight;
                        positions_info.push((pos_rev, other_set.len(), *other_weight));
                        continue;
                    }

                    // O(1) probe into other sets
                    if let Some(oldest_idx) = other_set.get_index_of(memory_id) {
                        total_weight += *other_weight;
                        let recency_pos = (other_set.len() - 1) - oldest_idx;
                        positions_info.push((recency_pos, other_set.len(), *other_weight));
                    }
                }

                // 4. Collect candidate
                candidates.push(((*memory_id).clone(), positions_info, total_weight));
            }
        }

        // 5. Score candidates
        self.score_consolidated_candidates(candidates, explain, disable_salience_bias, disable_systems_consolidation)
    }

    fn score_consolidated_candidates(&self, candidates: Vec<(String, Vec<(usize, usize, f64)>, f64)>, explain: bool, disable_salience_bias: bool, disable_systems_consolidation: bool) -> Vec<RecallResult> {
        const MAX_REC_WEIGHT: f64 = 20.0;
        const MAX_FREQ_WEIGHT: f64 = 5.0;
        
        let mut results = Vec::with_capacity(candidates.len());
        
        for (memory_id, positions_info, total_weight) in candidates {
            if let Some(memory) = self.memories.get(&memory_id) {
                // Skip consolidated summaries if disabled
                if disable_systems_consolidation && memory.cues.iter().any(|c| c == "type:summary") {
                    continue;
                }
                let mut total_recency = 0.0;
                let mut total_w_rec = 0.0;
                let mut total_w_freq = 0.0;
                
                let match_count = positions_info.len() as f64;

                for (pos, list_len, weight) in positions_info {
                    let pos_f64 = pos as f64;
                    let list_len_f64 = list_len as f64;
                    let sigma = list_len_f64.sqrt();
                    let ratio = pos_f64 / sigma;
                    
                    let w_rec = MAX_REC_WEIGHT / (ratio + 1.0);
                    let w_freq = 1.0 + (MAX_FREQ_WEIGHT * (1.0 - (1.0 / (ratio + 1.0))));
                    
                    let mut recency_component = 1.0 / (pos_f64 + 1.0);
                    if pos == 0 {
                        recency_component += 1.0;
                    }
                    
                    total_recency += recency_component * weight; // Weigh the recency contribution
                    total_w_rec += w_rec;
                    total_w_freq += w_freq;
                }
                
                let avg_w_rec = total_w_rec / match_count;
                let avg_w_freq = total_w_freq / match_count;
                let recency_score = total_recency / match_count;
                
                let frequency_score = if memory.reinforcement_count > 0 {
                    (memory.reinforcement_count as f64).log10()
                } else {
                    0.0
                };
                
                let salience_score = if disable_salience_bias {
                    0.0
                } else {
                    memory.salience
                };
                let intersection_score = total_weight * 100.0;
                
                // Final score includes salience
                let score = intersection_score + (recency_score * avg_w_rec) + (frequency_score * avg_w_freq) + (salience_score * 10.0);
                
                // Match integrity calculation
                // 1. Intersection strength (relative to match count)
                let intersection_strength = total_weight / match_count.max(1.0);
                // 2. Context agreement: how many of the memory's cues matched the query
                let context_agreement = if !memory.cues.is_empty() {
                    match_count / (memory.cues.len() as f64)
                } else {
                    0.0
                };
                // 3. Reinforcement boost (capped)
                let reinforcement_boost = (frequency_score / 2.0).min(1.0);
                
                let match_integrity = (intersection_strength * 0.5 + context_agreement * 0.3 + reinforcement_boost * 0.2).min(1.0);

                let explain_data = if explain {
                    Some(serde_json::json!({
                        "intersection_weighted": total_weight,
                        "intersection_score": intersection_score,
                        "recency_component": recency_score,
                        "frequency_component": frequency_score,
                        "salience_score": salience_score,
                        "match_integrity": match_integrity,
                        "weights": {
                            "recency": avg_w_rec,
                            "frequency": avg_w_freq,
                            "salience": 10.0
                        },
                        "match_count": match_count
                    }))
                } else {
                    None
                };

                results.push(RecallResult {
                    memory_id,
                    content: memory.content.clone(),
                    score,
                    match_integrity,
                    intersection_count: match_count as usize,
                    recency_score,
                    reinforcement_score: frequency_score,
                    salience_score,
                    created_at: memory.created_at,
                    metadata: memory.metadata.clone(),
                    explain: explain_data,
                });
            }
        }
        
        results
    }
    
    pub fn get_memory(&self, memory_id: &str) -> Option<Memory> {
        self.memories.get(memory_id).map(|m| m.clone())
    }
    
    pub fn consolidate_memories(&self, cue_overlap_threshold: f64) -> Vec<(String, Vec<String>)> {
        let mut to_merge = Vec::new();
        let mut seen = HashSet::new();

        // 1. Find overlapping memories
        // This is a naive O(N^2) or O(N * C) approach, but we can limit it using cues
        for entry in self.memories.iter() {
            let (id_a, mem_a) = entry.pair();
            if seen.contains(id_a) { continue; }
            
            let mut group = vec![id_a.clone()];
            
            // Use the first cue to find candidates
            if let Some(first_cue) = mem_a.cues.first() {
                if let Some(ordered_set) = self.cue_index.get(first_cue) {
                    for id_b in ordered_set.get_recent(None) {
                        if id_a == id_b || seen.contains(id_b) { continue; }
                        
                        if let Some(mem_b) = self.memories.get(id_b) {
                            // Calculate Jaccard similarity of cues
                            let cues_a: HashSet<_> = mem_a.cues.iter().collect();
                            let cues_b: HashSet<_> = mem_b.cues.iter().collect();
                            
                            let intersection = cues_a.intersection(&cues_b).count();
                            let union = cues_a.union(&cues_b).count();
                            let similarity = (intersection as f64) / (union as f64);
                            
                            if similarity >= cue_overlap_threshold {
                                group.push(id_b.clone());
                            }
                        }
                    }
                }
            }
            
            if group.len() > 1 {
                for id in &group {
                    seen.insert(id.clone());
                }
                to_merge.push(group);
            }
        }

        let mut results = Vec::new();
        // 2. Merge groups
        for group in to_merge {
            let mut combined_content = String::new();
            let mut combined_cues = HashSet::new();
            let mut total_reinforcement = 0;
            let mut max_salience: f64 = 0.0;
            
            for id in &group {
                if let Some(mem) = self.memories.get(id) {
                    if !combined_content.is_empty() { combined_content.push_str("\n---\n"); }
                    combined_content.push_str(&mem.content);
                    for cue in &mem.cues { combined_cues.insert(cue.clone()); }
                    total_reinforcement += mem.reinforcement_count;
                    max_salience = max_salience.max(mem.salience);
                }
            }
            
            // We NO LONGER delete old memories. Consolidation is additive.
            // Original memories are kept for trust and ground truth.
            
            // Add summary memory (keeping signal, reducing noise)
            let mut summary_content = format!("[Consolidated Memory]\n{}", combined_content);
            if summary_content.len() > 1000 {
                summary_content.truncate(1000);
                summary_content.push_str("... [truncated]");
            }
            
            let mut metadata = HashMap::new();
            metadata.insert("consolidated".to_string(), serde_json::json!(true));
            metadata.insert("original_count".to_string(), serde_json::json!(group.len()));
            
            let mut cues_vec: Vec<String> = combined_cues.into_iter().collect();
            cues_vec.push("type:summary".to_string());
            
            let new_id = self.add_memory(summary_content, cues_vec, Some(metadata), false);
            
            // Adjust properties
            if let Some(mut new_mem) = self.memories.get_mut(&new_id) {
                new_mem.reinforcement_count = total_reinforcement;
                new_mem.salience = max_salience * 0.8; // Lower priority than fresh memories
            }
            
            results.push((new_id, group));
        }
        
        results
    }

    pub fn get_graph_data(&self, limit: usize) -> serde_json::Value {
        let mut nodes = Vec::new();
        let mut links = Vec::new();
        let mut added_nodes = HashSet::new();

        // 1. Get recent memories
        let mut memories: Vec<_> = self.memories.iter().map(|kv| kv.value().clone()).collect();
        // Sort by last accessed desc
        memories.sort_unstable_by(|a, b| b.last_accessed.partial_cmp(&a.last_accessed).unwrap_or(std::cmp::Ordering::Equal));
        if limit > 0 {
            memories.truncate(limit);
        }

        for mem in &memories {
            if !added_nodes.contains(&mem.id) {
                // Truncate content for label
                let label: String = mem.content.chars().take(50).collect();
                let label = if mem.content.len() > 50 { format!("{}...", label) } else { label };
                
                nodes.push(serde_json::json!({
                    "id": mem.id,
                    "label": label,
                    "group": "memory",
                    "val": mem.salience.max(1.0)
                }));
                added_nodes.insert(mem.id.clone());
            }

            for cue in &mem.cues {
                let cue_id = format!("cue:{}", cue);
                if !added_nodes.contains(&cue_id) {
                    nodes.push(serde_json::json!({
                        "id": cue_id,
                        "label": cue,
                        "group": "cue",
                        "val": 1.0
                    }));
                    added_nodes.insert(cue_id.clone());
                }
                
                links.push(serde_json::json!({
                    "source": mem.id,
                    "target": cue_id,
                    "value": 2.0
                }));
            }
        }
        
        // 2. Add cue-cue edges from co-occurrence
        for node in &nodes {
             if node["group"] == "cue" {
                 let cue_label = node["label"].as_str().unwrap();
                 if let Some(co_map) = self.cue_co_occurrence.get(cue_label) {
                      for entry in co_map.iter() {
                          let (other_cue, count) = entry.pair();
                          let other_id = format!("cue:{}", other_cue);
                          
                          // Only visualize connection if both are in the graph to avoid explosion
                          if added_nodes.contains(&other_id) {
                              // Avoid double links: only add if A < B
                              if cue_label < other_cue.as_str() {
                                  links.push(serde_json::json!({
                                      "source": format!("cue:{}", cue_label),
                                      "target": other_id,
                                      "value": (*count as f64).min(5.0) // Cap strength
                                  }));
                              }
                          }
                      }
                 }
             }
        }

        serde_json::json!({
            "nodes": nodes,
            "links": links
        })
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
