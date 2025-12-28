use indexmap::IndexSet;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub created_at: f64,
    pub last_accessed: f64,
    #[serde(default)]
    pub reinforcement_count: u64,
    #[serde(default = "default_salience")]
    pub salience: f64,
    #[serde(default)]
    pub cues: Vec<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

fn default_salience() -> f64 {
    1.0
}

impl Memory {
    pub fn new(content: String, metadata: Option<HashMap<String, serde_json::Value>>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        
        Self {
            id: Uuid::new_v4().to_string(),
            content,
            created_at: now,
            last_accessed: now,
            reinforcement_count: 0,
            salience: 1.0,
            cues: Vec::new(),
            metadata: metadata.unwrap_or_default(),
        }
    }
    
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        self.reinforcement_count += 1;
    }
}

/// Ordered set implementation using IndexSet for O(1) operations
/// Most recent items are at the back (end)
/// 
/// IndexSet provides:
/// - O(1) insertion at end
/// - O(1) removal by value (via shift_remove)
/// - O(1) lookup
/// - Maintains insertion order
/// 
/// TODO: Optimize storage by interning UUID strings to u64 integers for V2.
/// This would reduce memory overhead from ~5M string copies to ~5M u64s (8 bytes each)
/// for a 1M memory dataset with 5 cues per memory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrderedSet {
    pub items: IndexSet<String>,
}

impl OrderedSet {
    pub fn new() -> Self {
        Self {
            items: IndexSet::new(),
        }
    }
    
    /// Add item to the end (most recent position) - O(1) amortized
    /// If item exists, removes it first then re-adds at end
    pub fn add(&mut self, item: String) {
        // shift_remove is O(1) average case (hash lookup + swap with last)
        // insert is O(1) amortized
        self.items.shift_remove(&item);
        self.items.insert(item);
    }

    /// Remove item from set - O(1) amortized
    pub fn remove(&mut self, item: &str) -> bool {
        self.items.shift_remove(item)
    }
    
    /// Move item to end (most recent position) - O(1) amortized
    /// This is the critical operation for reinforcement
    pub fn move_to_front(&mut self, item: &str) {
        // O(1) removal + O(1) insertion = O(1) total
        if self.items.shift_remove(item) {
            self.items.insert(item.to_string());
        }
    }
    
    /// Get items in reverse order (most recent first) - O(min(n, limit))
    /// Returns references to avoid cloning strings (zero-copy)
    pub fn get_recent(&self, limit: Option<usize>) -> Vec<&String> {
        let iter = self.items.iter().rev();
        
        match limit {
            Some(lim) => iter.take(lim).collect(),
            None => iter.collect(),
        }
    }
    
    /// Get items as owned strings (for serialization)
    /// Only use when you need to own the strings
    pub fn get_recent_owned(&self, limit: Option<usize>) -> Vec<String> {
        let iter = self.items.iter().rev();
        
        match limit {
            Some(lim) => iter.take(lim).cloned().collect(),
            None => iter.cloned().collect(),
        }
    }

    /// Get the index of an item in the set - O(1)
    /// Note: Returns index in insertion order (oldest -> newest)
    pub fn get_index_of(&self, item: &str) -> Option<usize> {
        self.items.get_index_of(item)
    }
    
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.items.len()
    }
    
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
