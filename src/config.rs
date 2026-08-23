use crate::semantic::SemanticConfig;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::PathBuf;

/// Performance tuning configuration for CueMap engine

// Search configuration (Deprecated constants, mapped to TuningConfig now)
pub const MAX_DRIVER_SCAN: usize = 10000;
pub const MAX_SEARCH_DEPTH: usize = 5000;

// DashMap shard configuration (power of 2)
pub const DASHMAP_SHARD_COUNT: usize = 128;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default)]
    pub server: ServerSettings,
    #[serde(default)]
    pub security: SecurityConfig,
    #[serde(default)]
    pub persistence: PersistenceConfig,
    #[serde(default)]
    pub jobs: JobsConfig,
    #[serde(default)]
    pub agent: AgentConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub tuning: TuningConfig,
    #[serde(default)]
    pub semantic: SemanticConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server: ServerSettings::default(),
            security: SecurityConfig::default(),
            persistence: PersistenceConfig::default(),
            jobs: JobsConfig::default(),
            agent: AgentConfig::default(),
            search: SearchConfig::default(),
            tuning: TuningConfig::default(),
            semantic: SemanticConfig::default(),
        }
    }
}

pub fn get_base_dir() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(home).join(".cuemap");
    if !path.exists() {
        let _ = fs::create_dir_all(&path);
    }
    path
}

impl ServerConfig {
    pub fn load(config_path: Option<PathBuf>, profile: Option<String>) -> Result<Self, String> {
        // 1. Start with defaults based on profile
        let profile_name = profile.unwrap_or_else(|| "default".to_string());
        let mut config = Self::default_for_profile(&profile_name);

        // 2. Load from config file matching profile (or just global config)
        let path = config_path.unwrap_or_else(|| get_base_dir().join("server_config.toml"));

        if path.exists() {
            let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
            let file_config: ServerConfig =
                toml::from_str(&content).map_err(|e| format!("Failed to parse config: {}", e))?;

            // Merge file config onto defaults
            // Note: This is a shallow merge implementation for simplicity.
            // In a robust system, we'd use a crate like `config` to merge fields deeply.
            // For now, we trust `toml` to deserialize partially if Option, but since we use structs with defaults,
            // `toml::from_str` usually replaces the whole struct if present.
            // To do proper layering without `config` crate is verbose.
            // Simplified approach: Parsing the file gives us a full config with defaults filled in by serde if missing in file.
            // So we just use the file config, but we need to ensure CLI args override it later.
            config = file_config;
        } else {
            // info!("Config file not found at {:?}, using defaults", path);
        }

        // 3. Environment variables overrides (Manual mapping for key fields)
        if let Ok(port) = env::var("CUEMAP_PORT") {
            if let Ok(p) = port.parse() {
                config.server.port = p;
            }
        }
        if let Ok(data_dir) = env::var("CUEMAP_DATA_DIR") {
            if !data_dir.trim().is_empty() {
                config.server.data_dir = data_dir;
            }
        }
        if let Ok(snapshot_interval) = env::var("CUEMAP_SNAPSHOT_INTERVAL_SECONDS") {
            if let Ok(seconds) = snapshot_interval.parse() {
                config.persistence.snapshot_interval_seconds = seconds;
            }
        }
        if let Ok(key) = env::var("CUEMAP_SECRET_KEY") {
            config.security.secret_key = Some(key);
        }
        if let Ok(key) = env::var("CUEMAP_SIGNING_PRIVATE_KEY") {
            config.security.signing_private_key = Some(key);
        }
        if let Ok(key) = env::var("CUEMAP_MASTER_KEY") {
            config.security.master_key = Some(key);
        }

        if let Ok(profile) = env::var("CUEMAP_SEMANTIC_PROFILE") {
            config.semantic.profile = match profile.trim().to_ascii_lowercase().as_str() {
                "edge" => crate::semantic::SemanticProfile::Edge,
                "balanced" => crate::semantic::SemanticProfile::Balanced,
                "quality" => crate::semantic::SemanticProfile::Quality,
                "off" => crate::semantic::SemanticProfile::Off,
                _ => config.semantic.profile,
            };
        }
        if let Ok(enabled) = env::var("CUEMAP_SEMANTIC_ENCODER_ENABLED") {
            if let Ok(enabled) = enabled.parse::<bool>() {
                config.semantic.encoder_enabled = enabled;
            }
        }
        if let Ok(dimensions) = env::var("CUEMAP_SEMANTIC_DIMENSIONS") {
            if let Ok(dimensions) = dimensions.parse::<usize>() {
                config.semantic.dimensions = dimensions;
            }
        }
        if let Ok(storage) = env::var("CUEMAP_SEMANTIC_STORAGE") {
            config.semantic.storage = match storage.trim().to_ascii_lowercase().as_str() {
                "f32" => crate::semantic::SemanticStorage::F32,
                "f16" => crate::semantic::SemanticStorage::F16,
                "int8" => crate::semantic::SemanticStorage::Int8,
                "auto" => crate::semantic::SemanticStorage::Auto,
                _ => config.semantic.storage,
            };
        }
        if let Ok(index) = env::var("CUEMAP_SEMANTIC_INDEX") {
            config.semantic.index = match index.trim().to_ascii_lowercase().as_str() {
                "exact" => crate::semantic::SemanticIndexMode::Exact,
                "ann" => crate::semantic::SemanticIndexMode::Ann,
                "auto" => crate::semantic::SemanticIndexMode::Auto,
                _ => config.semantic.index,
            };
        }
        if let Ok(model_id) = env::var("CUEMAP_SEMANTIC_MODEL_ID") {
            config.semantic.model_id = model_id;
        }
        if let Ok(model_version) = env::var("CUEMAP_SEMANTIC_MODEL_VERSION") {
            config.semantic.model_version = model_version;
        }
        if let Ok(model_path) = env::var("CUEMAP_SEMANTIC_MODEL_PATH") {
            config.semantic.model_path = model_path;
        }
        if let Ok(tokenizer_path) = env::var("CUEMAP_SEMANTIC_TOKENIZER_PATH") {
            config.semantic.tokenizer_path = tokenizer_path;
        }
        if let Ok(max_tokens) = env::var("CUEMAP_SEMANTIC_MAX_TOKENS") {
            if let Ok(max_tokens) = max_tokens.parse::<usize>() {
                config.semantic.max_tokens = max_tokens;
            }
        }
        if let Ok(threads) = env::var("CUEMAP_SEMANTIC_ENCODER_THREADS") {
            if let Ok(threads) = threads.parse::<usize>() {
                config.semantic.encoder_threads = threads;
            }
        }
        if let Ok(enabled) = env::var("CUEMAP_SEMANTIC_COREML_ENABLED") {
            if let Ok(enabled) = enabled.parse::<bool>() {
                config.semantic.coreml_enabled = enabled;
            }
        }
        if let Ok(weight) = env::var("CUEMAP_SEMANTIC_RERANK_WEIGHT") {
            if let Ok(weight) = weight.parse::<f64>() {
                config.semantic.semantic_rerank_weight = weight;
            }
        }
        if let Ok(limit) = env::var("CUEMAP_SEMANTIC_RERANK_CANDIDATE_LIMIT") {
            if let Ok(limit) = limit.parse::<usize>() {
                config.semantic.semantic_rerank_candidate_limit = limit;
            }
        }
        if let Ok(capacity) = env::var("CUEMAP_SEMANTIC_QUERY_CACHE_CAPACITY") {
            if let Ok(capacity) = capacity.parse::<usize>() {
                config.semantic.query_embedding_cache_capacity = capacity;
            }
        }
        if let Ok(enabled) = env::var("CUEMAP_SEMANTIC_INTENT_RERANK_ENABLED") {
            if let Ok(enabled) = enabled.parse::<bool>() {
                config.semantic.intent_rerank_enabled = enabled;
            }
        }
        if let Ok(weight) = env::var("CUEMAP_SEMANTIC_INTENT_RERANK_WEIGHT") {
            if let Ok(weight) = weight.parse::<f64>() {
                config.semantic.intent_rerank_weight = weight;
            }
        }
        if let Ok(penalty) = env::var("CUEMAP_SEMANTIC_INTENT_NO_RECALL_PENALTY") {
            if let Ok(penalty) = penalty.parse::<f64>() {
                config.semantic.intent_no_recall_penalty = penalty;
            }
        }
        if let Ok(max_delta) = env::var("CUEMAP_SEMANTIC_INTENT_RERANK_MAX_DELTA") {
            if let Ok(max_delta) = max_delta.parse::<f64>() {
                config.semantic.intent_rerank_max_delta = max_delta;
            }
        }

        config.semantic = config.semantic.resolved();
        Ok(config)
    }

    fn default_for_profile(profile: &str) -> Self {
        let mut config = Self::default();
        match profile {
            "read_only" => {
                config.server.read_only = true;
                config.persistence.enabled = false;
                config.jobs.background_processing = false;
                config.agent.enabled = false;
            }
            "live" => {
                config.persistence.enabled = true;
                config.jobs.background_processing = true;
            }
            "benchmark" => {
                config.persistence.enabled = false;
                config.jobs.background_processing = false;
                config.server.log_level = "warn".to_string();
            }
            _ => {} // Default
        }
        config
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerSettings {
    pub port: u16,
    pub host: String,
    pub data_dir: String,
    pub assets_dir: Option<String>,
    pub log_level: String,
    pub read_only: bool,
    #[serde(default)]
    pub store_content_on_disk: bool,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "0.0.0.0".to_string(),
            data_dir: get_base_dir().join("data").to_string_lossy().to_string(),
            assets_dir: None,
            log_level: "info".to_string(),
            read_only: false,
            store_content_on_disk: false,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub require_auth: bool,
    pub api_keys: Vec<String>,
    pub master_key: Option<String>,
    pub signing_private_key: Option<String>,
    pub secret_key: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistenceConfig {
    pub snapshot_interval_seconds: u64,
    pub enabled: bool,
    pub compress_snapshots: bool,
    #[serde(default)]
    pub cloud: CloudConfig,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            snapshot_interval_seconds: 60,
            enabled: true,
            compress_snapshots: true,
            cloud: CloudConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CloudConfig {
    pub provider: String, // "none", "s3", "gcs", "azure"
    pub bucket: String,
    pub region: String,
    pub endpoint: Option<String>,
    pub prefix: String,
    pub auto_backup: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct JobsConfig {
    pub background_processing: bool,
    pub market_heatmap_interval_seconds: u64,
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self {
            background_processing: true,
            market_heatmap_interval_seconds: 60,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentConfig {
    pub enabled: bool,
    pub watch_dir: Option<String>, // Deprecated in favor of project meta, but kept for global agent
    pub throttle_ms: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            watch_dir: None,
            throttle_ms: 100,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    pub max_scan_depth: usize,
    pub dashmap_shards: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_scan_depth: 10000,
            dashmap_shards: 128,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuningConfig {
    // Scoring
    pub max_rec_weight: f64,
    pub max_freq_weight: f64,
    pub intersection_score_multiplier: f64,
    pub salience_score_multiplier: f64,

    // Search / Scan
    pub idf_threshold_percent: f64,
    pub idf_min_count: usize,
    pub adaptive_scan_factor: usize,
    pub adaptive_scan_max: usize,

}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            // Defaults matching previous hardcoded constants
            max_rec_weight: 20.0,
            max_freq_weight: 5.0,
            intersection_score_multiplier: 100.0,
            salience_score_multiplier: 10.0,

            idf_threshold_percent: 0.1,
            idf_min_count: 20,
            adaptive_scan_factor: 100,
            adaptive_scan_max: 2000,

        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::{SemanticIndexMode, SemanticProfile, SemanticStorage};
    use std::ffi::OsString;

    #[test]
    fn profile_defaults_apply_expected_runtime_modes() {
        let read_only = ServerConfig::default_for_profile("read_only");
        assert!(read_only.server.read_only);
        assert!(!read_only.persistence.enabled);
        assert!(!read_only.jobs.background_processing);

        let live = ServerConfig::default_for_profile("live");
        assert!(live.persistence.enabled);
        assert!(live.jobs.background_processing);

        let benchmark = ServerConfig::default_for_profile("benchmark");
        assert!(!benchmark.persistence.enabled);
        assert!(!benchmark.jobs.background_processing);
        assert_eq!(benchmark.server.log_level, "warn");

        let default = ServerConfig::default_for_profile("unknown");
        assert_eq!(default.server.port, 8080);
    }

    #[test]
    fn load_reads_toml_and_applies_environment_overrides() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("server.toml");
        let mut file_config = ServerConfig::default();
        file_config.server.port = 9000;
        file_config.persistence.enabled = false;
        std::fs::write(&path, toml::to_string(&file_config).unwrap()).unwrap();

        let vars = [
            ("CUEMAP_PORT", "9123"),
            ("CUEMAP_DATA_DIR", "/tmp/cuemap-test-data"),
            ("CUEMAP_SNAPSHOT_INTERVAL_SECONDS", "7"),
            ("CUEMAP_SECRET_KEY", "secret"),
            ("CUEMAP_SIGNING_PRIVATE_KEY", "signing"),
            ("CUEMAP_MASTER_KEY", "master"),
            ("CUEMAP_SEMANTIC_PROFILE", "edge"),
            ("CUEMAP_SEMANTIC_ENCODER_ENABLED", "true"),
            ("CUEMAP_SEMANTIC_DIMENSIONS", "384"),
            ("CUEMAP_SEMANTIC_STORAGE", "int8"),
            ("CUEMAP_SEMANTIC_INDEX", "exact"),
            ("CUEMAP_SEMANTIC_MODEL_ID", "test-model"),
            ("CUEMAP_SEMANTIC_MODEL_VERSION", "v-test"),
            ("CUEMAP_SEMANTIC_MAX_TOKENS", "64"),
            ("CUEMAP_SEMANTIC_ENCODER_THREADS", "2"),
            ("CUEMAP_SEMANTIC_COREML_ENABLED", "true"),
            ("CUEMAP_SEMANTIC_RERANK_WEIGHT", "0.75"),
            ("CUEMAP_SEMANTIC_RERANK_CANDIDATE_LIMIT", "11"),
            ("CUEMAP_SEMANTIC_QUERY_CACHE_CAPACITY", "13"),
            ("CUEMAP_SEMANTIC_INTENT_RERANK_ENABLED", "true"),
            ("CUEMAP_SEMANTIC_INTENT_RERANK_WEIGHT", "0.25"),
            ("CUEMAP_SEMANTIC_INTENT_NO_RECALL_PENALTY", "0.5"),
            ("CUEMAP_SEMANTIC_INTENT_RERANK_MAX_DELTA", "0.2"),
        ];
        for (key, value) in vars {
            std::env::set_var(key, value);
        }

        let loaded = ServerConfig::load(Some(path), Some("live".to_string())).unwrap();

        for (key, _) in vars {
            std::env::remove_var(key);
        }

        assert_eq!(loaded.server.port, 9123);
        assert_eq!(loaded.server.data_dir, "/tmp/cuemap-test-data");
        assert_eq!(loaded.persistence.snapshot_interval_seconds, 7);
        assert_eq!(loaded.security.secret_key.as_deref(), Some("secret"));
        assert_eq!(loaded.security.signing_private_key.as_deref(), Some("signing"));
        assert_eq!(loaded.security.master_key.as_deref(), Some("master"));
        assert_eq!(loaded.semantic.profile, SemanticProfile::Edge);
        assert!(loaded.semantic.encoder_enabled);
        assert_eq!(loaded.semantic.dimensions, 384);
        assert_eq!(loaded.semantic.storage, SemanticStorage::Int8);
        assert_eq!(loaded.semantic.index, SemanticIndexMode::Exact);
        assert_eq!(loaded.semantic.model_id, "test-model");
        assert_eq!(loaded.semantic.model_version, "v-test");
        assert_eq!(loaded.semantic.max_tokens, 64);
        assert_eq!(loaded.semantic.encoder_threads, 2);
        assert!(loaded.semantic.coreml_enabled);
        assert_eq!(loaded.semantic.semantic_rerank_weight, 0.75);
        assert_eq!(loaded.semantic.semantic_rerank_candidate_limit, 11);
        assert_eq!(loaded.semantic.query_embedding_cache_capacity, 13);
        assert!(loaded.semantic.intent_rerank_enabled);
        assert_eq!(loaded.semantic.intent_rerank_weight, 0.25);
        assert_eq!(loaded.semantic.intent_no_recall_penalty, 0.5);
        assert_eq!(loaded.semantic.intent_rerank_max_delta, 0.2);
    }

    #[test]
    fn load_ignores_invalid_environment_values_and_missing_files() {
        let key = "CUEMAP_PORT";
        let previous: Option<OsString> = std::env::var_os(key);
        std::env::set_var(key, "not-a-port");
        let loaded = ServerConfig::load(Some(PathBuf::from("/path/that/does/not/exist")), None)
            .unwrap();
        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        assert_eq!(loaded.server.port, 8080);
    }
}
