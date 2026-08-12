use indexmap::IndexSet;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::crypto::{self, EncryptionKey};
use crate::intent::IntentClassification;
use crate::semantic::StoredSemanticVector;
use ahash::RandomState;

// =============================================================================
// Stats Types - Specialized payloads for different memory engines
// =============================================================================

/// Main Memory stats: Intrinsic Importance + Event-Driven Dynamic Salience
/// Used by the "Brain" engine for reactive scoring with decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MainStats {
    /// User-set or source-rule importance (default 1.0)
    #[serde(default = "default_intrinsic_salience")]
    pub intrinsic_salience: f64,
    /// Event-driven "heat" - boosted by events, decays with time
    #[serde(default)]
    pub dynamic_salience: f64,
    /// Unix timestamp when dynamic_salience was last boosted (for decay calc)
    #[serde(default)]
    pub last_boosted_at: u64,
    /// Legacy: reinforcement count for backward compat
    #[serde(default)]
    pub reinforcement_count: u64,
}

fn default_intrinsic_salience() -> f64 {
    1.0
}

impl Default for MainStats {
    fn default() -> Self {
        Self {
            intrinsic_salience: 1.0,
            dynamic_salience: 0.0,
            last_boosted_at: 0,
            reinforcement_count: 0,
        }
    }
}

impl MainStats {
    /// Backward-compat: get "salience" as sum of intrinsic + dynamic
    pub fn salience(&self) -> f64 {
        self.intrinsic_salience + self.dynamic_salience
    }

    /// Calculate effective salience at a specific point in time (with decay)
    pub fn effective_salience_at(&self, now: u64) -> f64 {
        let half_life = 3600.0; // 1 Hour
        let time_delta = now.saturating_sub(self.last_boosted_at);
        let decay_factor = 2.0_f64.powf(-(time_delta as f64) / half_life);

        self.intrinsic_salience + (self.dynamic_salience * decay_factor)
    }
}

/// Lexicon stats: Tiered Time-Series Statistics
/// Used by the "Dictionary" engine for statistical tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconStats {
    /// Total count of reinforcements
    #[serde(default)]
    pub total_count: u64,
    /// High-resolution: Minute timestamp (unix_secs / 60) -> count
    #[serde(default)]
    pub minute_stats: HashMap<u32, u16>,
    /// Low-resolution: Day timestamp (unix_secs / 86400) -> count
    #[serde(default)]
    pub daily_stats: HashMap<u32, u32>,
    /// Unix timestamp of last reinforcement
    #[serde(default)]
    pub last_reinforced: u64,
}

impl Default for LexiconStats {
    fn default() -> Self {
        Self {
            total_count: 0,
            minute_stats: HashMap::new(),
            daily_stats: HashMap::new(),
            last_reinforced: 0,
        }
    }
}

// =============================================================================
// Traits
// =============================================================================

/// Trait to allow Generic Engine to read basic stats for ranking/scoring
pub trait MemoryStats {
    fn get_salience(&self) -> f64;
    /// Calculate effective salience at a specific timestamp (allows for time-decay)
    fn get_effective_salience(&self, now: u64) -> f64;
    fn get_reinforcement_count(&self) -> u64;
    fn manual_boost(&mut self);
}

impl MemoryStats for MainStats {
    fn get_salience(&self) -> f64 {
        self.intrinsic_salience + self.dynamic_salience
    }

    fn get_effective_salience(&self, now: u64) -> f64 {
        self.effective_salience_at(now)
    }

    fn get_reinforcement_count(&self) -> u64 {
        self.reinforcement_count
    }

    fn manual_boost(&mut self) {
        self.intrinsic_salience += 0.1;
        self.reinforcement_count += 1;
    }
}

impl MemoryStats for LexiconStats {
    fn get_salience(&self) -> f64 {
        // For Lexicon, total_count is the rough equivalent of salience/importance
        self.total_count as f64
    }

    fn get_effective_salience(&self, _now: u64) -> f64 {
        // Lexicon doesn't decay (yet), just returns total count
        self.total_count as f64
    }

    fn get_reinforcement_count(&self) -> u64 {
        self.total_count
    }

    fn manual_boost(&mut self) {
        self.total_count += 1;
    }
}

// =============================================================================
// Generic Memory Struct
// =============================================================================

pub type MemoryId = u32;
pub const INVALID_MEMORY_ID: MemoryId = 0;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryScoringFeatures {
    #[serde(default)]
    pub version: u8,
    #[serde(default)]
    pub scored_cue_len: usize,
    #[serde(default)]
    pub has_summary_type: bool,
    #[serde(default)]
    pub structured_family_mask: u64,
}

/// Generic Memory wrapper for all memory types.
/// The `stats` field contains type-specific payload (MainStats or LexiconStats).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory<T> {
    pub id: MemoryId,
    #[serde(default)]
    pub source_key: Option<String>,
    // Content is now just raw bytes (Compressed OR Encrypted)
    pub content: Vec<u8>,
    pub created_at: f64,
    pub last_accessed: f64,
    #[serde(default)]
    pub cues: Vec<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub disk_backed: bool,
    #[serde(default)]
    pub scoring_features: MemoryScoringFeatures,
    /// Optional externally precomputed semantic representation. It is kept
    /// alongside the memory so the in-memory ANN index can be rebuilt after a
    /// snapshot restore.
    #[serde(default)]
    pub semantic_vector: Option<StoredSemanticVector>,
    /// Optional versioned intent classification used by CueKey gating and
    /// confidence-weighted hybrid reranking.
    #[serde(default)]
    pub intent_classification: Option<IntentClassification>,
    /// Type-specific stats payload
    pub stats: T,
}

impl<T: Default> Memory<T> {
    // Note: We avoid taking String directly in `new` to force explicit conversion choice.
    // Callers should use `new_with_payload` or handle compression/encryption first.
    // Or we provide a helper that takes key.

    pub fn new(content: Vec<u8>, metadata: Option<HashMap<String, serde_json::Value>>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Self {
            id: INVALID_MEMORY_ID,
            source_key: None,
            content,
            created_at: now,
            last_accessed: now,
            cues: Vec::new(),
            metadata: metadata.unwrap_or_default(),
            disk_backed: false,
            scoring_features: MemoryScoringFeatures::default(),
            semantic_vector: None,
            intent_classification: None,
            stats: T::default(),
        }
    }

    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
    }

    pub fn access_content(&self, key: Option<&EncryptionKey>) -> Result<String, String> {
        Self::decode_content_bytes(&self.content, key)
    }

    /// Decode content from raw bytes (used when fetching from disk)
    pub fn decode_content_bytes(
        bytes: &[u8],
        key: Option<&EncryptionKey>,
    ) -> Result<String, String> {
        // 1. Try to detect if it's just compressed (not encrypted)
        if crypto::is_compressed(bytes) {
            let decompressed = crypto::decompress(bytes)
                .map_err(|e| format!("Decompression failed (plaintext): {}", e))?;
            return String::from_utf8(decompressed).map_err(|e| format!("Invalid UTF-8: {}", e));
        }

        // 2. Fallback: Assume Encrypted
        let k = key.ok_or_else(|| {
            "Memory appears encrypted (no magic bytes) but no key provided".to_string()
        })?;

        let decrypted = crypto::decrypt(bytes, k)?;
        // The decrypted payload MUST be compressed zstd data
        let decompressed = crypto::decompress(&decrypted)
            .map_err(|e| format!("Decompression failed (after decrypt): {}", e))?;

        String::from_utf8(decompressed).map_err(|e| format!("Invalid UTF-8: {}", e))
    }

    /// Create payload from string (compress and optionally encrypt)
    pub fn create_payload(text: &str, key: Option<&EncryptionKey>) -> Result<Vec<u8>, String> {
        let compressed =
            crypto::compress(text.as_bytes()).map_err(|e| format!("Compression failed: {}", e))?;

        if let Some(k) = key {
            crypto::encrypt(&compressed, k)
        } else {
            Ok(compressed)
        }
    }
}

/// Convenience: Memory<MainStats> can increment reinforcement_count on touch
impl Memory<MainStats> {
    pub fn touch_and_reinforce(&mut self) {
        self.touch();
        self.stats.reinforcement_count += 1;
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrderedSet {
    pub items: IndexSet<MemoryId, RandomState>,
}

impl OrderedSet {
    pub fn new() -> Self {
        Self {
            items: IndexSet::with_hasher(RandomState::new()),
        }
    }

    /// Add item to the end (most recent position) - O(1) amortized
    /// If item exists, removes it first then re-adds at end
    pub fn add(&mut self, item: MemoryId) {
        // shift_remove is O(1) average case (hash lookup + swap with last)
        // insert is O(1) amortized
        self.items.shift_remove(&item);
        self.items.insert(item);
    }

    /// Remove item from set - O(1) amortized
    pub fn remove(&mut self, item: MemoryId) -> bool {
        self.items.shift_remove(&item)
    }

    /// Move item to end (most recent position) - O(1) amortized
    /// This is the critical operation for reinforcement
    pub fn move_to_front(&mut self, item: MemoryId) {
        // O(1) removal + O(1) insertion = O(1) total
        if self.items.shift_remove(&item) {
            self.items.insert(item);
        }
    }

    /// Get items in reverse order (most recent first) - O(min(n, limit))
    /// Returns references to avoid cloning strings (zero-copy)
    pub fn get_recent(&self, limit: Option<usize>) -> Vec<MemoryId> {
        let iter = self.items.iter().rev();

        match limit {
            Some(lim) => iter.take(lim).copied().collect(),
            None => iter.copied().collect(),
        }
    }

    /// Get items as owned strings (for serialization)
    /// Only use when you need to own the strings
    pub fn get_recent_owned(&self, limit: Option<usize>) -> Vec<MemoryId> {
        self.get_recent(limit)
    }

    /// Get the index of an item in the set - O(1)
    /// Note: Returns index in insertion order (oldest -> newest)
    pub fn get_index_of(&self, item: MemoryId) -> Option<usize> {
        self.items.get_index_of(&item)
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
