/// Performance tuning configuration for CueMap engine

// Search configuration
pub const MAX_DRIVER_SCAN: usize = 10000;
pub const MAX_SEARCH_DEPTH: usize = 5000; // Deprecated, but keeping for compatibility/reference

// DashMap shard configuration (power of 2)
// Higher = less contention but more memory
// Default is 64, we can tune based on workload
pub const DASHMAP_SHARD_COUNT: usize = 128;

// Pre-allocation hints
#[allow(dead_code)]
pub const EXPECTED_CUES_PER_MEMORY: usize = 4;
#[allow(dead_code)]
pub const EXPECTED_MEMORIES_PER_CUE: usize = 100;

// Alias Proposal Configuration
pub const ALIAS_MIN_CUE_MEMORIES: usize = 20;
pub const ALIAS_MAX_CUE_MEMORIES: usize = 50_000;
pub const ALIAS_MAX_CANDIDATES: usize = 1500;
pub const ALIAS_SIZE_SIMILARITY_MAX_RATIO: f64 = 0.10;
pub const ALIAS_OVERLAP_THRESHOLD: f64 = 0.90;
pub const ALIAS_SAMPLE_SIZE: usize = 512;

