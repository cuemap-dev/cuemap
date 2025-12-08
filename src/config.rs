/// Performance tuning configuration for CueMap engine

// Tier depths for iterative deepening
pub const TIER_1_DEPTH: usize = 10;
pub const TIER_2_DEPTH: usize = 100;
pub const MAX_SEARCH_DEPTH: usize = 1000;

// DashMap shard configuration (power of 2)
// Higher = less contention but more memory
// Default is 64, we can tune based on workload
pub const DASHMAP_SHARD_COUNT: usize = 128;

// Pre-allocation hints
pub const EXPECTED_CUES_PER_MEMORY: usize = 4;
pub const EXPECTED_MEMORIES_PER_CUE: usize = 100;
