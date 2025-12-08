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
    pub metadata: HashMap<String, serde_json::Value>,
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
            metadata: metadata.unwrap_or_default(),
        }
    }
    
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
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
#[derive(Debug, Clone, Default)]
pub struct OrderedSet {
    items: IndexSet<String>,
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
    
    /// Move item to end (most recent position) - O(1) amortized
    /// This is the critical operation for reinforcement
    pub fn move_to_front(&mut self, item: &str) {
        // O(1) removal + O(1) insertion = O(1) total
        if self.items.shift_remove(item) {
            self.items.insert(item.to_string());
        }
    }
    
    /// Get items in reverse order (most recent first) - O(min(n, limit))
    pub fn get_recent(&self, limit: Option<usize>) -> Vec<String> {
        let iter = self.items.iter().rev();
        
        match limit {
            Some(lim) => iter.take(lim).cloned().collect(),
            None => iter.cloned().collect(),
        }
    }
    
    pub fn len(&self) -> usize {
        self.items.len()
    }
    
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
