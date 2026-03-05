use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::env;
use std::fs;


/// Performance tuning configuration for CueMap engine

// Search configuration (Deprecated constants, mapped to TuningConfig now)
pub const MAX_DRIVER_SCAN: usize = 10000;
pub const MAX_SEARCH_DEPTH: usize = 5000; 

// DashMap shard configuration (power of 2)
pub const DASHMAP_SHARD_COUNT: usize = 128;

#[derive(Clone, Debug, Default, PartialEq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CueGenStrategy {
    #[default]
    Default,  // Minimal expansion (WordNet / Synonyms only)
    Glove,    // Deep semantic expansion (GloVe + WordNet)
    Ollama   // Local Ollama with Mistral (+ WordNet)
}

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
    pub llm: LlmConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub tuning: TuningConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server: ServerSettings::default(),
            security: SecurityConfig::default(),
            persistence: PersistenceConfig::default(),
            jobs: JobsConfig::default(),
            agent: AgentConfig::default(),
            llm: LlmConfig::default(),
            search: SearchConfig::default(),
            tuning: TuningConfig::default(),
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
        let path = config_path.unwrap_or_else(|| {
            get_base_dir().join("server_config.toml")
        });

        if path.exists() {
            let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
            let file_config: ServerConfig = toml::from_str(&content).map_err(|e| format!("Failed to parse config: {}", e))?;
            
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
            if let Ok(p) = port.parse() { config.server.port = p; }
        }
        if let Ok(key) = env::var("CUEMAP_SECRET_KEY") {
            config.security.secret_key = Some(key);
        }
        if let Ok(key) = env::var("CUEMAP_MASTER_KEY") {
            config.security.master_key = Some(key);
        }
        
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
            },
            "live" => {
                config.persistence.enabled = true;
                config.jobs.background_processing = true;
                config.jobs.consolidation_enabled = true;
            },
            "benchmark" => {
                config.persistence.enabled = false;
                config.jobs.background_processing = false;
                config.server.log_level = "warn".to_string();
            },
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
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub require_auth: bool,
    pub api_keys: Vec<String>,
    pub master_key: Option<String>,
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
pub struct JobsConfig {
    pub background_processing: bool,
    pub consolidation_enabled: bool,
    pub market_heatmap_interval_seconds: u64,
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self {
            background_processing: true,
            consolidation_enabled: false,
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
pub struct LlmConfig {
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub url: String,
    pub api_key: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: "ollama".to_string(),
            model: "mistral".to_string(),
            url: "http://localhost:11434".to_string(),
            api_key: None,
        }
    }
}

// Helper to convert to existing structure if needed
impl LlmConfig {
    pub fn to_legacy(&self) -> crate::llm::LlmConfig {
        crate::llm::LlmConfig {
            provider: self.provider.clone(),
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            ollama_url: self.url.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    pub max_scan_depth: usize,
    pub dashmap_shards: usize,
    pub cuegen_strategy: CueGenStrategy,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_scan_depth: 10000,
            dashmap_shards: 128,
            cuegen_strategy: CueGenStrategy::Default,
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

    // Expansion
    pub expansion_threshold: f64,
    pub expansion_limit: usize,
    pub max_proposed_cues: usize,
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
            
            expansion_threshold: 0.65,
            expansion_limit: 3,
            max_proposed_cues: 10,
        }
    }
}
