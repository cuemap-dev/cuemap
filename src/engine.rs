use crate::structures::{Memory, OrderedSet, MainStats, LexiconStats, MemoryStats};
use crate::crypto::EncryptionKey;
use tracing::info;
use dashmap::DashMap;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};


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
pub struct CueMapEngine<T>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Default + Send + Sync + MemoryStats + 'static
{
    memories: Arc<DashMap<String, Memory<T>>>,
    cue_index: Arc<DashMap<String, OrderedSet>>,
    // Pattern Completion: cue co-occurrence matrix
    cue_co_occurrence: Arc<DashMap<String, DashMap<String, u64>>>,
    // Temporal Chunking: track last event per session/project (using a dummy key for now or extending API)
    last_events: Arc<DashMap<String, (String, f64, Vec<String>)>>,
    // Performance: Atomic counter to avoid DashMap::len() contention
    memory_count: Arc<AtomicUsize>,
    // Performance: Atomic counter for cues
    cue_count: Arc<AtomicUsize>,
    // Security: Master key for encryption (optional)
    master_key: Option<Arc<EncryptionKey>>,
}


impl<T> CueMapEngine<T>
where

    T: Serialize + for<'de> Deserialize<'de> + Clone + Default + Send + Sync + MemoryStats + 'static
{
    pub fn new() -> Self {
        Self {
            memories: Arc::new(DashMap::new()),
            cue_index: Arc::new(DashMap::new()),
            cue_co_occurrence: Arc::new(DashMap::new()),
            last_events: Arc::new(DashMap::new()),
            memory_count: Arc::new(AtomicUsize::new(0)),
            cue_count: Arc::new(AtomicUsize::new(0)),
            master_key: None,
        }
    }

    pub fn with_key(key: Option<EncryptionKey>) -> Self {
        let mut engine = Self::new();
        engine.master_key = key.map(Arc::new);
        engine
    }

    pub fn set_master_key(&mut self, key: Option<Arc<EncryptionKey>>) {
        self.master_key = key;
    }

    pub fn get_master_key(&self) -> Option<Arc<EncryptionKey>> {
        self.master_key.clone()
    }


    
    pub fn from_state(
        memories: DashMap<String, Memory<T>>,
        cue_index: DashMap<String, OrderedSet>,
    ) -> Self {
        let count = memories.len();
        let cue_count_val = cue_index.len();
        let engine = Self {
            memories: Arc::new(memories),
            cue_index: Arc::new(cue_index),
            cue_co_occurrence: Arc::new(DashMap::new()), 
            last_events: Arc::new(DashMap::new()),
            memory_count: Arc::new(AtomicUsize::new(count)),
            cue_count: Arc::new(AtomicUsize::new(cue_count_val)),
            master_key: None,
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
    pub fn get_memories(&self) -> &Arc<DashMap<String, Memory<T>>> {
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
        stats: T,
        disable_temporal_chunking: bool,
    ) -> String {
        // Create payload (Compressed or Encrypted)
        let payload = match Memory::<T>::create_payload(&content, self.master_key.as_deref()) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Failed to create memory payload: {}", e);
                return String::new(); // Or handle error better?
            }
        };

        let mut memory = Memory::new(payload, metadata);
        let memory_id = memory.id.clone();
        
        // Store cues in memory
        memory.cues = cues.clone();
        memory.stats = stats;
        
        // 1. Temporal Chunking
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

        // 2. Update co-occurrence matrix (MOVED TO BACKGROUND JOB)
        // self.update_cue_co_occurrence(&memory.cues);
        
        if self.memories.insert(memory_id.clone(), memory).is_none() {
            self.memory_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Index by cues (Double Indexing)
        for cue in &cues {
            let cue_lower = cue.to_lowercase().trim().to_string();
            if cue_lower.is_empty() { continue; }

            // 1. Index full cue
            // 1. Index full cue
            if !self.cue_index.contains_key(&cue_lower) {
                 self.cue_count.fetch_add(1, Ordering::Relaxed);
            }
            self.cue_index
                .entry(cue_lower.clone())
                .or_insert_with(OrderedSet::new)
                .add(memory_id.clone());

            // For now, let's just accept that `cue_count` might be slightly off if we don't track it perfectly,
            // OR we fix it by not using DashMap::len() for stats.
            
            // Re-think: Is `cue_index.len()` actually O(N)?
            // DashMap::len() locks all shards to sum them up. Yes.
            
            // Correct approach: increment only if we add a NEW key.
            // But we already have the entry lock.
            // If the set was empty (newly created), it's a new cue?
            // No, OrderedSet::new() creates empty.
            // Basic logic: if we call `or_insert_with`, and it runs the closure, it's new.
            // But `entry` returns a RefMut, we don't know if closure ran.
            
            // Alternative: check if key exists first? No, race condition.
            // Alternative 2: Trust DashMap::len() is slow, but maybe we only need it occasionally?
            // No, UI calls it every 10s.
            
            // Let's implement roughly correct counting:
            // We can't easily know if it's a new key without checking.
            // Let's use `if !self.cue_index.contains_key(...)` fast check? No races.
            
            // Wait, we are inside `add_memory` which is basically single-writer per memory...
            // But concurrent add_memories happen.
            
            // Let's just use `cue_index.len()` inside `new/from_state`, 
            // and checking `if self.cue_index.insert(...)` returns None?
            // But we use `entry().or_insert()`.
            
            // OK, let's change logic to:
            if !self.cue_index.contains_key(&cue_lower) {
                 self.cue_count.fetch_add(1, Ordering::Relaxed);
            }
             self.cue_index.entry(cue_lower.clone()).or_insert_with(OrderedSet::new).add(memory_id.clone());
             
             // 2. Index value
             if let Some((_, value)) = cue_lower.split_once(':') {
                 if !value.is_empty() {
                     let val_str = value.to_string();
                      if !self.cue_index.contains_key(&val_str) {
                         self.cue_count.fetch_add(1, Ordering::Relaxed);
                      }
                      self.cue_index.entry(val_str).or_insert_with(OrderedSet::new).add(memory_id.clone());
                 }
             }


        }
        
        memory_id
    }
    
    pub fn reinforce_memory(&self, memory_id: &str, cues: Vec<String>) -> bool {
        // Update last accessed
        if let Some(mut memory) = self.memories.get_mut(memory_id) {
            memory.touch();
            memory.stats.manual_boost(); // Manual reinforcement boost
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
                     if entry.is_empty() {
                         drop(entry); // Release RefMut to allow removal
                         if self.cue_index.remove(&cue_lower).is_some() {
                             self.cue_count.fetch_sub(1, Ordering::Relaxed);
                         }
                     }
                 }
                 
                 // 2. Remove from value entry
                 if let Some((_, value)) = cue_lower.split_once(':') {
                     if !value.is_empty() {
                         if let Some(mut entry) = self.cue_index.get_mut(value) {
                             entry.remove(memory_id);
                             if entry.is_empty() {
                                 drop(entry);
                                 if self.cue_index.remove(value).is_some() {
                                     self.cue_count.fetch_sub(1, Ordering::Relaxed);
                                 }
                             }
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
        stats: Option<T>,
        reinforce: bool,
        overwrite_cues: bool,
    ) -> String {
        // If exists: handle update
        if self.memories.contains_key(&id) {
            {
                if let Some(mut memory) = self.memories.get_mut(&id) {
                    // Update content ALWAYS
                    match Memory::<T>::create_payload(&content, self.master_key.as_deref()) {
                        Ok(p) => memory.content = p,
                        Err(e) => tracing::error!("Failed to update content: {}", e),
                    }
                    
                    if let Some(m) = metadata {
                        memory.metadata = m;
                    }
                    // We need to drop lock before attach/overwrite ops to avoid deadlocks 
                    // (though attach_cues re-acquires check, better safe)
                }
            }
            
            if overwrite_cues {
                // Remove old cues from index + Replace cues
                // We need to get old cues first
                let old_cues = if let Some(mem) = self.memories.get(&id) {
                    mem.cues.clone()
                } else {
                    Vec::new()
                };
                
                self.remove_cues_from_index(&id, &old_cues);
                
                if let Some(mut mem) = self.memories.get_mut(&id) {
                    mem.cues = Vec::new(); // Clear
                }
                // Now attach new cues (effectively replacing)
                self.attach_cues(&id, cues.clone());
            } else {
                // Merge mode
                self.attach_cues(&id, cues.clone());
            }

            if reinforce {
                self.reinforce_memory(&id, cues);
            }
            return id;
        }
        
        // Insert new
        let payload = match Memory::<T>::create_payload(&content, self.master_key.as_deref()) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Failed to create memory payload: {}", e);
                return id; 
            }
        };

        let mut memory = Memory::new(payload, metadata);
        memory.id = id.clone();
        memory.cues = cues.clone();
        if let Some(s) = stats {
            memory.stats = s;
        }
        
        if self.memories.insert(id.clone(), memory).is_none() {
            self.memory_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Index by cues (Double Indexing)
        for cue in &cues { // Iterate by reference to avoid move
            let cue_lower = cue.to_lowercase().trim().to_string();
            if cue_lower.is_empty() { continue; }
            
            // 1. Index full cue
            let cue_lower_clone = cue_lower.clone();
            if !self.cue_index.contains_key(&cue_lower_clone) {
                 self.cue_count.fetch_add(1, Ordering::Relaxed);
            }
            self.cue_index
                .entry(cue_lower_clone)
                .or_insert_with(OrderedSet::new)
                .add(id.clone());
            
            // 2. Index value
            if let Some((_, value)) = cue_lower.split_once(':') {
                if !value.is_empty() {
                    let val_str = value.to_string();
                    if !self.cue_index.contains_key(&val_str) {
                         self.cue_count.fetch_add(1, Ordering::Relaxed);
                    }
                    self.cue_index
                        .entry(val_str)
                        .or_insert_with(OrderedSet::new)
                        .add(id.clone());
                }
            }

        }
        
        self.update_cue_co_occurrence(&cues);
        
        id
    }

    pub fn attach_cues(&self, memory_id: &str, cues: Vec<String>) -> bool {
        // 1. Get memory and check if it exists
        if let Some(mut memory) = self.memories.get_mut(memory_id) {
            // 2. Identify new cues (deduplication)
            let mut new_cues = Vec::new();
            for cue in cues {
                let cue_lower = cue.to_lowercase().trim().to_string();
                if cue_lower.is_empty() { continue; }
                
                // Check against existing cues (case-insensitive check technically needed, but we store as-is)
                // Assuming existing cues were normalized or we just check strict equality
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
                
                // 1. Index full cue
                let cue_lower_clone = cue_lower.clone();
                if !self.cue_index.contains_key(&cue_lower_clone) {
                     self.cue_count.fetch_add(1, Ordering::Relaxed);
                }
                self.cue_index
                    .entry(cue_lower_clone)
                    .or_insert_with(OrderedSet::new)
                    .add(memory_id.to_string());
                
                // 2. Index value
                if let Some((_, value)) = cue_lower.split_once(':') {
                    if !value.is_empty() {
                        let val_str = value.to_string();
                         if !self.cue_index.contains_key(&val_str) {
                             self.cue_count.fetch_add(1, Ordering::Relaxed);
                         }
                        self.cue_index
                            .entry(val_str)
                            .or_insert_with(OrderedSet::new)
                            .add(memory_id.to_string());
                    }
                }

            }
            
            let all_cues = memory.cues.clone();
            drop(memory); 
            self.update_cue_co_occurrence(&all_cues);
            
            return true;
        } else {
            false
        }
    }
    
    pub fn remove_cues_from_index(&self, memory_id: &str, cues: &[String]) {
        for cue in cues {
             let cue_lower = cue.to_lowercase().trim().to_string();
             if cue_lower.is_empty() { continue; }
             
             // 1. Remove from full cue entry
             if let Some(mut entry) = self.cue_index.get_mut(&cue_lower) {
                 entry.remove(memory_id);
                 if entry.is_empty() {
                     drop(entry); 
                     if self.cue_index.remove(&cue_lower).is_some() {
                         self.cue_count.fetch_sub(1, Ordering::Relaxed);
                     }
                 }
             }
             
             // 2. Remove from value entry
             if let Some((_, value)) = cue_lower.split_once(':') {
                 if !value.is_empty() {
                     if let Some(mut entry) = self.cue_index.get_mut(value) {
                         entry.remove(memory_id);
                         if entry.is_empty() {
                             drop(entry);
                             if self.cue_index.remove(value).is_some() {
                                 self.cue_count.fetch_sub(1, Ordering::Relaxed);
                             }
                         }
                     }
                 }
             }
         }
    }
    
    pub fn recall(
        &self,
        query_cues: Vec<String>,
        limit: usize,
        auto_reinforce: bool,
        heatmap: Option<&HashMap<String, f32>>,
    ) -> Vec<RecallResult> {
        self.recall_with_min_intersection(query_cues, limit, auto_reinforce, None, heatmap)
    }
    
    pub fn recall_with_min_intersection(
        &self,
        query_cues: Vec<String>,
        limit: usize,
        auto_reinforce: bool,
        min_intersection: Option<usize>,
        heatmap: Option<&HashMap<String, f32>>,
    ) -> Vec<RecallResult> {
        if query_cues.is_empty() {
            return Vec::new();
        }
        
        // Default weight of 1.0 for standard recall
        let weighted_cues: Vec<(String, f64)> = query_cues
            .into_iter()
            .map(|c| (c, 1.0))
            .collect();
            
        self.recall_weighted(weighted_cues, limit, auto_reinforce, min_intersection, false, false, false, false, heatmap)
    }

    /// O(limit) recall using intersection-first strategy.
    /// Only scans the smallest cue list up to `limit` items, probes others in O(1).
    /// Returns results in recency order - no expensive sorting.
    /// Best for: simple keyword queries where speed > perfect ranking.
    pub fn recall_intersection(&self, query_cues: Vec<(String, f64)>, limit: usize) -> Vec<RecallResult> {
        if query_cues.is_empty() || limit == 0 {
            return Vec::new();
        }

        // 1. Normalize and collect cue sets with sizes
        let mut cue_sets: Vec<(String, f64, dashmap::mapref::one::Ref<String, OrderedSet>)> = Vec::new();
        for (cue, weight) in &query_cues {
            let cue_lower = cue.to_lowercase();
            let cue_trimmed = cue_lower.trim().to_string();
            if cue_trimmed.is_empty() { continue; }
            
            if let Some(ordered_set) = self.cue_index.get(&cue_trimmed) {
                cue_sets.push((cue_trimmed, *weight, ordered_set));
            }
        }

        if cue_sets.is_empty() {
            return Vec::new();
        }

        // 2. Sort by set size - smallest (most selective) first
        cue_sets.sort_by(|a, b| a.2.len().cmp(&b.2.len()));

        // 3. Iterate ONLY the smallest cue's list, up to limit items
        let (_driver_cue, driver_weight, driver_set) = &cue_sets[0];
        let other_sets = &cue_sets[1..];
        
        let mut results = Vec::with_capacity(limit);
        let scan_limit = driver_set.len().min(limit * 10); // Scan 10x limit to find enough intersections
        
        for memory_id in driver_set.get_recent(Some(scan_limit)) {
            // 4. O(1) probes into other cue sets
            let mut total_weight = *driver_weight;
            let mut match_count = 1;
            
            for (_other_cue, other_weight, other_set) in other_sets {
                if other_set.get_index_of(memory_id).is_some() {
                    total_weight += other_weight;
                    match_count += 1;
                }
            }

            // 5. Fetch memory and build result
            if let Some(memory) = self.memories.get(memory_id) {
                let decrypted_content = memory.access_content(self.master_key.as_deref())
                    .unwrap_or_else(|_| "<decryption failed>".to_string());
                
                results.push(RecallResult {
                    memory_id: memory_id.clone(),
                    content: decrypted_content,
                    score: total_weight * 100.0, // Simple intersection-based score
                    match_integrity: (match_count as f64) / (cue_sets.len() as f64),
                    intersection_count: match_count,
                    recency_score: 1.0,
                    reinforcement_score: memory.stats.get_reinforcement_count() as f64,
                    salience_score: memory.stats.get_salience(),
                    created_at: memory.created_at,
                    metadata: memory.metadata.clone(),
                    explain: None,
                });

                // 6. Early termination when limit reached
                if results.len() >= limit {
                    break;
                }
            }
        }

        results
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
                        let decrypted_content = memory.access_content(self.master_key.as_deref())
                             .unwrap_or_else(|_| "<decryption failed>".to_string());

                        candidates.push(RecallResult {
                            memory_id: memory_id.clone(),
                            content: decrypted_content,
                            score: 1.0,
                            match_integrity: 1.0,
                            intersection_count: 1,
                            recency_score: 1.0,
                            reinforcement_score: memory.stats.get_reinforcement_count() as f64,
                            salience_score: memory.stats.get_salience(),
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
        heatmap: Option<&HashMap<String, f32>>,
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
                    
                    // Skip metadata/structural cues in pattern completion inference
                    // We only want to infer semantic synonyms (e.g. "food" -> "diet"), 
                    // not structural context (e.g. "food" -> "domain:youtube")
                    if inferred_cue.contains(':') {
                        continue;
                    }

                    // Skip superstring inferences (Compounding)
                    // If we query "health", don't infer "surgo_health" or "gut_health".
                    // We want Lateral Expansion (synonyms), not Vertical Expansion (specialization).
                    if inferred_cue.contains(cue) {
                        continue;
                    }

                    *inferred_candidates.entry(inferred_cue.clone()).or_insert(0) += *count;
                }
                }
            }

            // Take top-K inferred cues and inject them with low weight
            let mut inferred_list: Vec<(String, u64)> = inferred_candidates.into_iter().collect();
            inferred_list.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            
            // Inferred cues are "suggestions", they must NEVER overpower explicit query terms.
            // Even with high IDF, an inferred cue should be a tie-breaker, not a driver.
            let pattern_completion_weight = 0.1; 
            for (inf_cue, _) in inferred_list.into_iter().take(5) {
                active_cues.push((inf_cue, pattern_completion_weight));
            }
        }
        
        // 2. Consolidated search using Selective Set Intersection
        let mut results = self.consolidated_search(&active_cues, limit, explain, disable_salience_bias, disable_systems_consolidation, heatmap);
        
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
    
    fn consolidated_search(&self, query_cues: &[(String, f64)], limit: usize, explain: bool, disable_salience_bias: bool, disable_systems_consolidation: bool, heatmap: Option<&HashMap<String, f32>>) -> Vec<RecallResult> {
        if query_cues.is_empty() {
            return Vec::new();
        }

        // 1. Gather cue data with set sizes for sorting
        let mut cue_data: Vec<(String, f64, dashmap::mapref::one::Ref<String, OrderedSet>)> = Vec::with_capacity(query_cues.len());
        let total_memories = self.memories.len() as f64;
        
        for (cue, weight) in query_cues {
            if let Some(ordered_set) = self.cue_index.get(cue) {
                // IDF Weighting (BM25 variant): Penalize common cues, boost rare ones
                // BM25's IDF accounts for the complement (memories WITHOUT this cue),
                // making it much more aggressive at demoting high-frequency cues.
                // e.g. at df=40% of corpus: old formula gave 0.91, BM25 gives 0.40
                let df = ordered_set.len() as f64;
                let idf = ((total_memories - df + 0.5) / (df + 0.5)).ln().max(0.1);
                let adjusted_weight = weight * idf;
                
                cue_data.push((cue.clone(), adjusted_weight, ordered_set));
            }
        }

        if cue_data.is_empty() {
            return Vec::new();
        }

        // OPTIMIZATION 1: Sort by set size (smallest first)
        // Processing rarer cues first produces fewer candidates to probe
        cue_data.sort_by(|a, b| a.2.len().cmp(&b.2.len()));

        // OPTIMIZATION 2: Adaptive scan limit based on requested limit
        // For limit=5, we don't need to scan 10k items per cue
        // Scale: limit * 100, capped at 2000 for safety
        let adaptive_scan_limit = (limit * 100).min(2000);

        // 2. Perform Union-based search with O(1) Probing
        let mut candidates = Vec::new();
        let mut seen_memories = HashSet::new();

        for (cue_idx, (_cue, _weight, set)) in cue_data.iter().enumerate() {
            let scan_limit = std::cmp::min(set.len(), adaptive_scan_limit);
            let items = set.get_recent(Some(scan_limit));

            for (pos_rev, memory_id) in items.iter().enumerate() {
                // If we've already processed this memory from a previous cue, skip it
                if seen_memories.contains(*memory_id) {
                    continue;
                }
                seen_memories.insert((*memory_id).clone());

                let mut total_weight = 0.0;
                let mut positions_info = Vec::with_capacity(cue_data.len());

                // 3. For each NEW candidate, probe ALL query cue lists to get full intersection data
                for (other_idx, (other_cue, other_weight, other_set)) in cue_data.iter().enumerate() {
                    // Optimization: if it's the current set we're iterating, we know it's there
                    if other_idx == cue_idx {
                        total_weight += *other_weight;
                        positions_info.push((pos_rev, other_set.len(), *other_weight, other_cue.clone()));
                        continue;
                    }

                    // O(1) probe into other sets
                    if let Some(oldest_idx) = other_set.get_index_of(memory_id) {
                        total_weight += *other_weight;
                        let recency_pos = (other_set.len() - 1) - oldest_idx;
                        positions_info.push((recency_pos, other_set.len(), *other_weight, other_cue.clone()));
                    }
                }

                // 4. Collect candidate
                candidates.push(((*memory_id).clone(), positions_info, total_weight));
            }
        }

        // 5. Score candidates
        self.score_consolidated_candidates(candidates, explain, disable_salience_bias, disable_systems_consolidation, heatmap)
    }

    fn score_consolidated_candidates(
        &self, 
        candidates: Vec<(String, Vec<(usize, usize, f64, String)>, f64)>, 
        explain: bool, 
        disable_salience_bias: bool, 
        disable_systems_consolidation: bool,
        heatmap: Option<&HashMap<String, f32>>
    ) -> Vec<RecallResult> {
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

                for (pos, list_len, _weight, _cue_name) in &positions_info {
                    let pos_f64 = *pos as f64;
                    let list_len_f64 = *list_len as f64;
                    let sigma = list_len_f64.sqrt();
                    let ratio = pos_f64 / sigma;
                    
                    let w_rec = MAX_REC_WEIGHT / (ratio + 1.0);
                    let w_freq = 1.0 + (MAX_FREQ_WEIGHT * (1.0 - (1.0 / (ratio + 1.0))));
                    
                    let recency_component = 1.0 / (pos_f64 + 1.0);
                    
                    total_recency += recency_component; // Independent of IDF weight
                    total_w_rec += w_rec;
                    total_w_freq += w_freq;
                }
                
                let avg_w_rec = total_w_rec / match_count;
                let avg_w_freq = total_w_freq / match_count;
                let recency_score = total_recency / match_count;
                
                let frequency_score = if memory.stats.get_reinforcement_count() > 0 {
                    (memory.stats.get_reinforcement_count() as f64).log10()
                } else {
                    0.0
                };
                
                let (salience_score, effective_salience, market_lift) = if disable_salience_bias {
                    (0.0, 0.0, 0.0)
                } else {
                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                    let eff = memory.stats.get_effective_salience(now);
                    
                    let mut lift = 0.0;
                    if let Some(map) = heatmap {
                        for cue in &memory.cues {
                            if let Some(val) = map.get(cue) {
                                lift += *val as f64;
                            }
                        }
                    }
                    
                    (eff + lift, eff, lift)
                };
                
                let intersection_score = total_weight * 100.0;
                
                // Final score includes salience
                // We use salience_score (Effective + Market) here
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

                tracing::info!(
                    "Scoring Memory {}: score={:.2} [int={:.2} rec={:.2} freq={:.2} sal={:.2} (eff={:.2}, mkt={:.2})]",
                    memory_id, score, intersection_score, recency_score * avg_w_rec, frequency_score * avg_w_freq, salience_score * 10.0, effective_salience, market_lift
                );

                let explain_data = if explain {
                    Some(serde_json::json!({
                        "intersection_weighted": total_weight,
                        "intersection_score": intersection_score,
                        "recency_component": recency_score,
                        "frequency_component": frequency_score,
                        "salience_score": salience_score,
                        "effective_salience": effective_salience,
                        "market_lift": market_lift,
                        "match_integrity": match_integrity,
                        "weights": {
                            "recency": avg_w_rec,
                            "frequency": avg_w_freq,
                            "salience": 10.0
                        },
                        "match_count": match_count,
                        "matched_cues": positions_info.iter().map(|(_, _, _, name)| name.clone()).collect::<Vec<_>>()
                    }))
                } else {
                    None
                };

                results.push(RecallResult {
                    memory_id,
                    content: memory.access_content(self.master_key.as_deref()).unwrap_or_else(|_| "<decryption failed>".to_string()),
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
    
    pub fn get_memory(&self, memory_id: &str) -> Option<Memory<T>> {
        self.memories.get(memory_id).map(|m| m.clone())
    }
    
    // Consolidate Memory function removed from generic implementation
    // It requires specific knowledge of how to merge T
    // Will be re-implemented in specialized impl blocks if needed

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

        // 2. Add memories and their directly connected cues
        for mem in &memories {
            if !added_nodes.contains(&mem.id) {
                // Truncate content for label
                let content_str = mem.access_content(self.master_key.as_deref()).unwrap_or_default();
                let label: String = content_str.chars().take(50).collect();
                let label = if content_str.len() > 50 { format!("{}...", label) } else { label };
                
                nodes.push(serde_json::json!({
                    "id": mem.id,
                    "label": label,
                    "group": "memory",
                    "val": mem.stats.get_salience().max(1.0)
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
        
        // 3. Add cue-cue edges from co-occurrence (With budget)
        // Prevent O(N^2) explosion by capping total links
        let max_total_links = 10_000;
        if links.len() < max_total_links {
            let mut co_links = Vec::new();
            
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
                                      let weight = (*count as f64).min(5.0);
                                      co_links.push((
                                          format!("cue:{}", cue_label),
                                          other_id,
                                          weight
                                      ));
                                  }
                              }
                          }
                     }
                 }
            }
            
            // Sort by weight descending (prioritize strong connections)
            co_links.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            
            // Take what fits in budget
            let remaining = max_total_links.saturating_sub(links.len());
            for (source_id, target_id, weight) in co_links.into_iter().take(remaining) {
                links.push(serde_json::json!({
                    "source": source_id,
                    "target": target_id,
                    "value": weight
                }));
            }
        }

        serde_json::json!({
            "nodes": nodes,
            "links": links
        })
    }

    /// Context API: Expand query cues using the co-occurrence graph
    /// Returns Vec of (term, score, raw_count, source_cues) for each candidate
    /// 
    /// Scoring: Aggregates co-occurrence counts across all query cues.
    /// Terms that co-occur with multiple query cues get higher scores.
    pub fn expand_cues_from_graph(&self, query_cues: &[String], limit: usize) -> Vec<(String, f64, u64, Vec<String>)> {
        if query_cues.is_empty() {
            return Vec::new();
        }

        // Normalize query cues
        let normalized_cues: Vec<String> = query_cues
            .iter()
            .map(|c| c.to_lowercase().trim().to_string())
            .filter(|c| !c.is_empty())
            .collect();

        if normalized_cues.is_empty() {
            return Vec::new();
        }

        // Fast path: Single cue query - just return top co-occurring terms directly
        if normalized_cues.len() == 1 {
            let query_cue = &normalized_cues[0];
            if let Some(co_map) = self.cue_co_occurrence.get(query_cue) {
                let mut results: Vec<(String, f64, u64, Vec<String>)> = co_map
                    .iter()
                    .filter(|entry| {
                        let candidate = entry.key();
                        // Skip metadata cues and superstrings
                        !candidate.contains(':') && !candidate.contains(query_cue)
                    })
                    .map(|entry| {
                        let (term, count) = entry.pair();
                        (term.clone(), *count as f64, *count, vec![query_cue.clone()])
                    })
                    .collect();
                
                // Sort by count descending
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(limit);
                return results;
            } else {
                return Vec::new();
            }
        }

        // Multi-cue path: Aggregate candidates across all query cues
        let mut candidates: HashMap<String, (u64, Vec<String>)> = HashMap::new();

        for query_cue in &normalized_cues {
            if let Some(co_map) = self.cue_co_occurrence.get(query_cue) {
                for entry in co_map.iter() {
                    let (candidate_cue, count) = entry.pair();
                    
                    // Skip if candidate is already in query (no point expanding to itself)
                    if normalized_cues.contains(candidate_cue) {
                        continue;
                    }
                    
                    // Skip metadata/structural cues (same filter as pattern completion)
                    if candidate_cue.contains(':') {
                        continue;
                    }
                    
                    // Skip superstring inferences (avoid vertical specialization)
                    if candidate_cue.contains(query_cue) {
                        continue;
                    }

                    candidates
                        .entry(candidate_cue.clone())
                        .and_modify(|(total, sources)| {
                            *total += *count;
                            if !sources.contains(query_cue) {
                                sources.push(query_cue.clone());
                            }
                        })
                        .or_insert((*count, vec![query_cue.clone()]));
                }
            }
        }

        // Convert to vec and sort by score (count) descending
        let mut results: Vec<(String, f64, u64, Vec<String>)> = candidates
            .into_iter()
            .map(|(term, (count, sources))| {
                // Score = raw count (can be refined with IDF later)
                let score = count as f64;
                (term, score, count, sources)
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Truncate to limit
        results.truncate(limit);
        
        results
    }

    pub fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        stats.insert(
            "total_memories".to_string(),
            serde_json::json!(self.memory_count.load(Ordering::Relaxed)),
        );
        stats.insert(
            "total_cues".to_string(),
            serde_json::json!(self.cue_count.load(Ordering::Relaxed)),
        );

        stats
    }
}

// ==================================================================================
// Specialized Implementation for "Brain" (MainStats)
// ==================================================================================

impl CueMapEngine<MainStats> {
    /// Decays dynamic salience for all memories and updates generic salience proxy
    pub fn decay_salience(&self, decay_rate: f64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for mut memory in self.memories.iter_mut() {
             let stats = &mut memory.value_mut().stats;
             let time_delta = now.saturating_sub(stats.last_boosted_at);
             
             // Simple exponential decay: N(t) = N0 * e^(-lambda * t)
             // We use hours as time unit
             let hours_passed = (time_delta as f64) / 3600.0;
             if hours_passed > 0.1 {
                 let decay_factor = (-decay_rate * hours_passed).exp();
                 stats.dynamic_salience *= decay_factor;
                 
                 // Clamp near zero
                 if stats.dynamic_salience < 0.01 {
                     stats.dynamic_salience = 0.0;
                 }
             }
        }
    }

    /// Reinforces memory by adding dynamic heat (Brain logic)
    /// Algorithm: Inverse Proportional Boosting
    /// NewScore = Current + (Amount / (1.0 + Current))
    /// This prevents "Context Poisoning" (log explosion) from high-frequency events.
    pub fn reinforce_dynamic(&self, memory_id: &str, amount: f64) {
        if let Some(mut memory) = self.memories.get_mut(memory_id) {
            memory.touch();
            let stats = &mut memory.stats;
            
            // Logarithmic Saturation
            stats.dynamic_salience += amount / (1.0 + stats.dynamic_salience);
            
            stats.reinforcement_count += 1;
            stats.last_boosted_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
    }

    /// Calculates "Effective Importance" by combining Intrinsic + Decayed Dynamic + Market Heatmap
    pub fn score_with_decay_and_market(
        &self, 
        candidate_ids: Vec<String>, 
        heatmap: &HashMap<String, f32>
    ) -> Vec<RecallResult> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        let mut results = Vec::new();

        for id in candidate_ids {
            if let Some(mem) = self.memories.get(&id) {
                let stats = &mem.stats;
                
                // 1. Effective Salience (Intrinsic + Decayed Dynamic) - centralized in MemoryStats trait
                let effective_salience = stats.get_effective_salience(now);
                
                // 2. Market Heatmap Logic
                // Logic: Sum of heatmap values for cues found in this memory
                let mut market_lift = 0.0;
                for cue in &mem.cues {
                    if let Some(score) = heatmap.get(cue) {
                        market_lift += *score as f64;
                    }
                }
                
                // 3. Aggregate Total Salience
                let total_salience = effective_salience + market_lift;
                
                // Construct result
                results.push(RecallResult {
                    memory_id: mem.id.clone(),
                    content: mem.access_content(self.master_key.as_deref()).unwrap_or_else(|_| "<decryption failed>".to_string()),
                    score: total_salience, 
                    match_integrity: 1.0, 
                    intersection_count: 0, 
                    recency_score: effective_salience,
                    reinforcement_score: stats.reinforcement_count as f64,
                    salience_score: total_salience,
                    created_at: mem.created_at,
                    metadata: mem.metadata.clone(),
                    explain: Some(serde_json::json!({
                        "intrinsic": stats.intrinsic_salience,
                        "effective_salience": effective_salience,
                        "market_lift": market_lift,
                        "total_salience": total_salience
                    })),
                });
            }
        }
        
        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        results
    }
    
    /// Prune memories with low salience (Brain Plasticity)
    pub fn prune_low_salience(&self, threshold: f64) -> usize {
        let mut to_remove = Vec::new();
        
        for entry in self.memories.iter() {
            let stats = &entry.value().stats;
            // Total effective salience
            let total_salience = stats.intrinsic_salience + stats.dynamic_salience;
            
            // Protect high reinforcement memories from pruning even if cold?
            // Maybe not, if unused for a LONG time.
            
            if total_salience < threshold && stats.reinforcement_count < 5 {
                to_remove.push(entry.key().clone());
            }
        }
        
        let count = to_remove.len();
        for id in to_remove {
            self.delete_memory(&id);
        }
        
        count
    }

    /// Consolidate memories - specialized for MainStats
    pub fn consolidate_memories(&self, cue_overlap_threshold: f64) -> Vec<(String, Vec<String>)> {
        let mut to_merge = Vec::new();
        let mut seen = HashSet::new();

        // 1. Find overlapping memories (Naive)
        for entry in self.memories.iter() {
            let (id_a, mem_a) = entry.pair();
            if seen.contains(id_a) { continue; }
            
            // Skip already consolidated memories to avoid recursion
            if mem_a.metadata.get("consolidated").and_then(|v| v.as_bool()).unwrap_or(false) {
                continue;
            }
            
            let mut group = vec![id_a.clone()];
            
            if let Some(first_cue) = mem_a.cues.first() {
                if let Some(ordered_set) = self.cue_index.get(first_cue) {
                    for id_b in ordered_set.get_recent(None) {
                        if id_a == id_b || seen.contains(id_b) { continue; }
                        
                        if let Some(mem_b) = self.memories.get(id_b) {
                             if mem_b.metadata.get("consolidated").and_then(|v| v.as_bool()).unwrap_or(false) {
                                continue;
                            }
                            
                            let cues_a: HashSet<_> = mem_a.cues.iter().collect();
                            let cues_b: HashSet<_> = mem_b.cues.iter().collect();
                            
                            let intersection = cues_a.intersection(&cues_b).count();
                            let union = cues_a.union(&cues_b).count();
                            
                            if union > 0 {
                                let similarity = (intersection as f64) / (union as f64);
                                if similarity >= cue_overlap_threshold {
                                    group.push(id_b.clone());
                                }
                            }
                        }
                    }
                }
            }
            
            if group.len() > 1 {
                for id in &group { seen.insert(id.clone()); }
                to_merge.push(group);
            }
        }

        let mut results = Vec::new();
        
        // 2. Merge
        for group in to_merge {
            let mut combined_content = String::new();
            let mut combined_cues = HashSet::new();
            
            // MainStats aggregation
            let mut total_intrinsic = 0.0;
            let mut max_dynamic: f64 = 0.0;
            let mut total_reinforcement = 0;
            
            for id in &group {
                if let Some(mem) = self.memories.get(id) {
                    if !combined_content.is_empty() { combined_content.push_str("\n---\n"); }
                    if let Ok(c) = mem.access_content(self.master_key.as_deref()) {
                        combined_content.push_str(&c);
                    }
                    for cue in &mem.cues { combined_cues.insert(cue.clone()); }
                    
                    total_intrinsic += mem.stats.intrinsic_salience;
                    max_dynamic = max_dynamic.max(mem.stats.dynamic_salience);
                    total_reinforcement += mem.stats.reinforcement_count;
                }
            }
            
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
            
            // Create stats
            let new_stats = MainStats {
                intrinsic_salience: (total_intrinsic / group.len() as f64) * 1.2, // Boost consolidated intrinsic
                dynamic_salience: max_dynamic, // Keep urgency of most urgent part
                last_boosted_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                reinforcement_count: total_reinforcement,
            };
            
            let new_id = self.add_memory(summary_content, cues_vec, Some(metadata), new_stats, false);
            results.push((new_id, group));
        }
        
        results
    }
}


// ==================================================================================
// Specialized Implementation for "Dictionary" (LexiconStats)
// ==================================================================================

impl CueMapEngine<LexiconStats> {
    
    /// Tiered Reinforcement for Dictionary (Minute/Daily Buckets)
    pub fn reinforce_tiered(&self, memory_id: &str, amount: u64) {
        if let Some(mut memory) = self.memories.get_mut(memory_id) {
             memory.touch();
             let stats = &mut memory.stats;
             stats.total_count += amount;
             stats.last_reinforced = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
             
             // Calculate buckets
             let now_mins = (stats.last_reinforced / 60) as u32;
             let now_day = (stats.last_reinforced / 86400) as u32;
             
             *stats.minute_stats.entry(now_mins).or_insert(0) += amount as u16;
             *stats.daily_stats.entry(now_day).or_insert(0) += amount as u32;
             
             // Cleanup old minute buckets (keep last 60 mins)
             let min_threshold = now_mins.saturating_sub(60);
             stats.minute_stats.retain(|&k, _| k >= min_threshold);
        }
    }
    
    /// Trending identification (Spike detection)
    /// Sums bucket counts in window to identify trending cues
    pub fn get_trending_items(&self, limit: usize) -> Vec<(String, f64)> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let current_min = (now / 60) as u32;
        // Look at last 15 mins
        let window_start = current_min.saturating_sub(15);
        
        let mut trending = Vec::new();
        
        for entry in self.memories.iter() {
            let stats = &entry.value().stats;
            
            // Calculate velocity in window
            let mut recent_velocity = 0.0;
            for (min, count) in &stats.minute_stats {
                if *min >= window_start {
                    recent_velocity += *count as f64;
                }
            }
            
            if recent_velocity >= 1.0 {
                // Normalize by baseline (total count / age?) or simply raw velocity for "Trending Now"
                trending.push((entry.key().clone(), recent_velocity));
            }
        }
        
        trending.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Log top 5 items with bucket breakdown
        if !trending.is_empty() {
             info!("Trending: Identification finished. Found {} candidates >= 1.0 velocity", trending.len());
             for i in 0..trending.len().min(5) {
                 let (cue, velocity) = &trending[i];
                 if let Some(entry) = self.memories.get(cue) {
                     let stats = &entry.stats;
                     let m_count = stats.minute_stats.len();
                     let d_count = stats.daily_stats.len();
                     let total = stats.total_count;
                     info!("Trending: [Rank {}] Cue '{}' velocity={:.2}. Stats: totals={}, min_buckets={}, day_buckets={}, last_ref={}", 
                        i+1, cue, velocity, total, m_count, d_count, stats.last_reinforced);
                 }
             }
        }

        trending.truncate(limit);
        
        trending
    }
}
