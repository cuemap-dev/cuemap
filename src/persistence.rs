//! Persistence layer with bincode serialization, background snapshots, and cloud backup.
//!
//! # Cloud Backup
//!
//! Supports backing up snapshots to cloud storage providers:
//! - **AWS S3** (and S3-compatible like MinIO, DigitalOcean Spaces)
//! - **Google Cloud Storage (GCS)**
//! - **Azure Blob Storage**
//!
//! ## Environment Variables
//!
//! ### AWS S3
//! - `AWS_ACCESS_KEY_ID` - AWS access key
//! - `AWS_SECRET_ACCESS_KEY` - AWS secret key
//! - `AWS_REGION` - AWS region (can be overridden by --cloud-region)
//!
//! ### Google Cloud Storage
//! - `GOOGLE_SERVICE_ACCOUNT` - Path to service account JSON file
//!
//! ### Azure Blob Storage
//! - `AZURE_STORAGE_ACCOUNT_NAME` - Storage account name
//! - `AZURE_STORAGE_ACCOUNT_KEY` - Storage account key

use crate::engine::CueMapEngine;
use crate::structures::{Memory, MemoryId, MemoryStats, OrderedSet};
use ahash::RandomState;
use bytes::Bytes;
use dashmap::DashMap;
use object_store::{
    aws::AmazonS3Builder, azure::MicrosoftAzureBuilder, gcp::GoogleCloudStorageBuilder,
    local::LocalFileSystem, path::Path as ObjectPath, ObjectStore, PutPayload,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

#[derive(Debug, Serialize, Deserialize)]
struct PersistedState<T> {
    memories: HashMap<MemoryId, Memory<T>>,
    source_key_to_id: HashMap<String, MemoryId>,
    cue_index: HashMap<String, Vec<MemoryId>>, // Flattened OrderedSet
    next_memory_id: MemoryId,
    version: u32,
    saved_at: u64,
    #[serde(default)]
    cue_global_counts: Option<HashMap<String, u64>>,
}

// Version 2 snapshots used bincode. Version 3 uses zstd-compressed JSON so
// serde_json::Value metadata can be restored reliably (bincode cannot
// deserialize dynamic JSON values) without making snapshots unnecessarily
// large. Keep accepting v2 for snapshots that do not contain such metadata.
const PERSISTENCE_VERSION: u32 = 3;
const LEGACY_PERSISTENCE_VERSION: u32 = 2;

fn serialize_state<T>(state: &PersistedState<T>) -> Result<Vec<u8>, std::io::Error>
where
    T: Serialize,
{
    let json = serde_json::to_vec(state).map_err(|error| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, error)
    })?;
    zstd::stream::encode_all(std::io::Cursor::new(json), 3)
}

fn deserialize_state<T>(data: &[u8]) -> Result<PersistedState<T>, std::io::Error>
where
    T: for<'de> Deserialize<'de>,
{
    let decoded = if crate::crypto::is_compressed(data) {
        zstd::stream::decode_all(std::io::Cursor::new(data))?
    } else {
        data.to_vec()
    };

    match serde_json::from_slice(&decoded) {
        Ok(state) => Ok(state),
        Err(json_error) => bincode::deserialize(&decoded).map_err(|bincode_error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                "failed to decode snapshot as JSON ({json_error}) or legacy bincode ({bincode_error})"
                ),
            )
        }),
    }
}

fn check_snapshot_version(version: u32) -> Result<(), std::io::Error> {
    if version == PERSISTENCE_VERSION || version == LEGACY_PERSISTENCE_VERSION {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Unsupported snapshot version {} (expected {} or {}). Reingest required.",
                version, PERSISTENCE_VERSION, LEGACY_PERSISTENCE_VERSION
            ),
        ))
    }
}

pub struct PersistenceManager {
    data_dir: PathBuf,
    snapshot_interval: Duration,
}

impl PersistenceManager {
    pub fn new(data_dir: impl AsRef<Path>, snapshot_interval_secs: u64) -> Self {
        let data_dir = data_dir.as_ref().to_path_buf();

        // Create data directory if it doesn't exist
        if let Err(e) = fs::create_dir_all(&data_dir) {
            error!("Failed to create data directory {:?}: {}", data_dir, e);
        }

        Self {
            data_dir,
            snapshot_interval: Duration::from_secs(snapshot_interval_secs),
        }
    }

    /// Save engine state to a specific path (used by multi-tenant)
    /// Save engine state to a specific path (used by multi-tenant)
    pub fn save_to_path<T>(
        engine: &CueMapEngine<T>,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        T: Serialize
            + for<'de> Deserialize<'de>
            + Clone
            + Default
            + Send
            + Sync
            + MemoryStats
            + 'static,
    {
        let start = std::time::Instant::now();

        let memories = engine.get_memories();
        let source_key_to_id = engine.get_source_key_to_id();
        let cue_index = engine.get_cue_index();

        // Convert DashMaps to serializable format
        let memories_map: HashMap<MemoryId, Memory<T>> = memories
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();

        let source_key_to_id_map: HashMap<String, MemoryId> = source_key_to_id
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();

        let cue_index_map: HashMap<String, Vec<MemoryId>> = cue_index
            .iter()
            .map(|entry| {
                let cue = entry.key().clone();
                let ordered_set = entry.value();
                let memory_ids = ordered_set.get_recent_owned(None);
                (cue, memory_ids)
            })
            .collect();

        // Convert global counts
        let global_counts_map = {
            let counts_dm = &engine.cue_global_counts;
            if !counts_dm.is_empty() {
                let mut counts = HashMap::new();
                for entry in counts_dm.iter() {
                    counts.insert(entry.key().clone(), *entry.value());
                }
                Some(counts)
            } else {
                None
            }
        };

        let state = PersistedState {
            memories: memories_map,
            source_key_to_id: source_key_to_id_map,
            cue_index: cue_index_map,
            next_memory_id: engine.next_memory_id(),
            version: PERSISTENCE_VERSION,
            saved_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            cue_global_counts: global_counts_map,
        };

        // JSON is intentionally used for new snapshots, then compressed to
        // keep the on-disk representation close to the old bincode size.
        let data = serialize_state(&state)?;

        // Write to temp file first (atomic operation)
        let temp_path = path.with_extension("bin.tmp");
        fs::write(&temp_path, &data)?;

        // Rename to final location (atomic on most filesystems)
        fs::rename(&temp_path, path)?;

        let duration = start.elapsed();
        info!(
            "Saved {} memories and {} cues to {:?} in {:?} ({} bytes)",
            state.memories.len(),
            state.cue_index.len(),
            path,
            duration,
            data.len()
        );

        Ok(())
    }

    /// Load engine state from a specific path (used by multi-tenant)
    /// Load engine state from a specific path (used by multi-tenant)
    pub fn load_from_path<T>(
        path: &Path,
    ) -> Result<
        (
            DashMap<MemoryId, Memory<T>, RandomState>,
            DashMap<String, MemoryId, RandomState>,
            DashMap<String, OrderedSet, RandomState>,
            MemoryId,
            Option<DashMap<String, u64, RandomState>>,
        ),
        Box<dyn std::error::Error>,
    >
    where
        T: Serialize
            + for<'de> Deserialize<'de>
            + Clone
            + Default
            + Send
            + Sync
            + MemoryStats
            + 'static,
    {
        if !path.exists() {
            return Err(format!("Snapshot not found: {:?}", path).into());
        }

        info!("Loading state from {:?}", path);

        let data = fs::read(path)?;
        let state: PersistedState<T> = deserialize_state(&data)?;
        check_snapshot_version(state.version)?;

        info!(
            "Loaded {} memories and {} cues from snapshot (version: {}, saved: {})",
            state.memories.len(),
            state.cue_index.len(),
            state.version,
            state.saved_at
        );

        // Convert to DashMaps
        let memories = DashMap::with_hasher(RandomState::new());
        for (id, memory) in state.memories {
            memories.insert(id, memory);
        }

        let source_key_to_id = DashMap::with_hasher(RandomState::new());
        for (source_key, memory_id) in state.source_key_to_id {
            source_key_to_id.insert(source_key, memory_id);
        }

        let cue_index = DashMap::with_hasher(RandomState::new());
        for (cue, memory_ids) in state.cue_index {
            let mut ordered_set = OrderedSet::new();
            for memory_id in memory_ids {
                ordered_set.add(memory_id);
            }
            cue_index.insert(cue, ordered_set);
        }

        let loaded_global_counts = state.cue_global_counts.map(|raw_counts| {
            let dm = DashMap::with_hasher(RandomState::new());
            for (cue, count) in raw_counts {
                dm.insert(cue, count);
            }
            dm
        });

        Ok((
            memories,
            source_key_to_id,
            cue_index,
            state.next_memory_id.max(1),
            loaded_global_counts,
        ))
    }

    /// List all snapshot files in a directory (main engines only, not aliases/lexicon)
    pub fn list_snapshots_in_dir(dir: &Path) -> Vec<String> {
        let mut snapshots = Vec::new();

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                    // Only include main engine files, not aliases or lexicon
                    if filename.ends_with(".bin")
                        && !filename.ends_with(".tmp")
                        && !filename.ends_with("_aliases.bin")
                        && !filename.ends_with("_lexicon.bin")
                    {
                        let project_id = filename.replace(".bin", "");
                        snapshots.push(project_id);
                    }
                }
            }
        }

        snapshots.sort();
        snapshots
    }

    /// Delete a snapshot file
    #[allow(dead_code)]
    pub fn delete_snapshot(path: &Path) -> Result<(), String> {
        if path.exists() {
            fs::remove_file(path).map_err(|e| format!("Failed to delete snapshot: {}", e))?;
        }
        Ok(())
    }

    fn snapshot_path(&self) -> PathBuf {
        self.data_dir.join("cuemap.bin")
    }

    fn temp_snapshot_path(&self) -> PathBuf {
        self.data_dir.join("cuemap.bin.tmp")
    }

    pub fn load_state<T>(
        &self,
    ) -> Result<
        (
            DashMap<MemoryId, Memory<T>, RandomState>,
            DashMap<String, MemoryId, RandomState>,
            DashMap<String, OrderedSet, RandomState>,
            MemoryId,
        ),
        Box<dyn std::error::Error>,
    >
    where
        T: Serialize
            + for<'de> Deserialize<'de>
            + Clone
            + Default
            + Send
            + Sync
            + MemoryStats
            + 'static,
    {
        let snapshot_path = self.snapshot_path();

        if !snapshot_path.exists() {
            info!("No existing snapshot found, starting with empty state");
            return Ok((
                DashMap::with_hasher(RandomState::new()),
                DashMap::with_hasher(RandomState::new()),
                DashMap::with_hasher(RandomState::new()),
                1,
            ));
        }

        info!("Loading state from {:?}", snapshot_path);

        let data = fs::read(&snapshot_path)?;
        let state: PersistedState<T> = deserialize_state(&data)?;
        check_snapshot_version(state.version)?;

        info!(
            "Loaded {} memories and {} cues from snapshot (version: {}, saved: {})",
            state.memories.len(),
            state.cue_index.len(),
            state.version,
            state.saved_at
        );

        // Convert to DashMaps
        let memories = DashMap::with_hasher(RandomState::new());
        for (id, memory) in state.memories {
            memories.insert(id, memory);
        }

        let source_key_to_id = DashMap::with_hasher(RandomState::new());
        for (source_key, memory_id) in state.source_key_to_id {
            source_key_to_id.insert(source_key, memory_id);
        }

        let cue_index = DashMap::with_hasher(RandomState::new());
        for (cue, memory_ids) in state.cue_index {
            let mut ordered_set = OrderedSet::new();
            for memory_id in memory_ids {
                ordered_set.add(memory_id);
            }
            cue_index.insert(cue, ordered_set);
        }

        Ok((
            memories,
            source_key_to_id,
            cue_index,
            state.next_memory_id.max(1),
        ))
    }

    pub fn save_state<T>(&self, engine: &CueMapEngine<T>) -> Result<(), Box<dyn std::error::Error>>
    where
        T: Serialize
            + for<'de> Deserialize<'de>
            + Clone
            + Default
            + Send
            + Sync
            + MemoryStats
            + 'static,
    {
        let start = std::time::Instant::now();

        let memories = engine.get_memories();
        let source_key_to_id = engine.get_source_key_to_id();
        let cue_index = engine.get_cue_index();

        // Convert DashMaps to serializable format
        let memories_map: HashMap<MemoryId, Memory<T>> = memories
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();

        let source_key_to_id_map: HashMap<String, MemoryId> = source_key_to_id
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();

        let cue_index_map: HashMap<String, Vec<MemoryId>> = cue_index
            .iter()
            .map(|entry| {
                let cue = entry.key().clone();
                let ordered_set = entry.value();
                // Use owned version for serialization
                let memory_ids = ordered_set.get_recent_owned(None);
                (cue, memory_ids)
            })
            .collect();

        // Convert global counts
        let global_counts_map = {
            let counts_dm = &engine.cue_global_counts;
            if !counts_dm.is_empty() {
                let mut counts = HashMap::new();
                for entry in counts_dm.iter() {
                    counts.insert(entry.key().clone(), *entry.value());
                }
                Some(counts)
            } else {
                None
            }
        };

        let state = PersistedState {
            memories: memories_map,
            source_key_to_id: source_key_to_id_map,
            cue_index: cue_index_map,
            next_memory_id: engine.next_memory_id(),
            version: PERSISTENCE_VERSION,
            saved_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            cue_global_counts: global_counts_map,
        };

        // Keep the on-disk format compressed JSON so arbitrary metadata values
        // round-trip without sacrificing snapshot size.
        let data = serialize_state(&state)?;

        // Write to temp file first (atomic operation)
        let temp_path = self.temp_snapshot_path();
        fs::write(&temp_path, &data)?;

        // Rename to final location (atomic on most filesystems)
        fs::rename(&temp_path, &self.snapshot_path())?;

        let duration = start.elapsed();
        info!(
            "Saved {} memories and {} cues to snapshot in {:?} ({} bytes)",
            state.memories.len(),
            state.cue_index.len(),
            duration,
            data.len()
        );

        Ok(())
    }

    pub async fn start_background_snapshots<T>(
        &self,
        engine: Arc<CueMapEngine<T>>,
    ) -> tokio::task::JoinHandle<()>
    where
        T: Serialize
            + for<'de> Deserialize<'de>
            + Clone
            + Default
            + Send
            + Sync
            + MemoryStats
            + 'static,
    {
        let persistence = self.clone();

        tokio::spawn(async move {
            let mut interval = interval(persistence.snapshot_interval);

            loop {
                interval.tick().await;

                if let Err(e) = persistence.save_state(&engine) {
                    error!("Background snapshot failed: {}", e);
                } else {
                    info!("Background snapshot completed");
                }
            }
        })
    }
}

impl Clone for PersistenceManager {
    fn clone(&self) -> Self {
        Self {
            data_dir: self.data_dir.clone(),
            snapshot_interval: self.snapshot_interval,
        }
    }
}

/// Setup graceful shutdown handler
pub async fn setup_shutdown_handler<T>(
    persistence: PersistenceManager,
    engine: Arc<CueMapEngine<T>>,
) where
    T: Serialize
        + for<'de> Deserialize<'de>
        + Clone
        + Default
        + Send
        + Sync
        + MemoryStats
        + 'static,
{
    tokio::spawn(async move {
        // Wait for SIGINT (Ctrl+C) or SIGTERM
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("Failed to create SIGINT handler");
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to create SIGTERM handler");

        tokio::select! {
            _ = sigint.recv() => {
                info!("Received SIGINT, shutting down gracefully...");
            }
            _ = sigterm.recv() => {
                info!("Received SIGTERM, shutting down gracefully...");
            }
        }

        // Save final snapshot
        info!("Saving final snapshot before shutdown...");
        if let Err(e) = persistence.save_state(&engine) {
            error!("Failed to save final snapshot: {}", e);
        } else {
            info!("Final snapshot saved successfully");
        }

        // Exit
        std::process::exit(0);
    });
}

// ============================================================================
// Cloud Backup Support
// ============================================================================

/// Supported cloud storage providers
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum CloudProvider {
    /// Amazon S3 or S3-compatible storage (MinIO, DigitalOcean Spaces, etc.)
    S3 {
        bucket: String,
        region: String,
        /// Optional custom endpoint for S3-compatible services
        endpoint: Option<String>,
    },
    /// Google Cloud Storage
    GCS { bucket: String },
    /// Azure Blob Storage
    Azure { container: String, account: String },
    /// Local filesystem (for testing)
    Local { path: String },
}

/// Configuration for cloud backup
#[derive(Debug, Clone, Default)]
pub struct CloudBackupConfig {
    /// Whether cloud backup is enabled
    pub enabled: bool,
    /// Cloud provider configuration
    pub provider: Option<CloudProvider>,
    /// Object key prefix, e.g. "cuemap/snapshots/"
    pub prefix: String,
    /// Automatically backup after each local save
    pub auto_backup: bool,
}

impl CloudBackupConfig {
    /// Create a new cloud backup configuration from CLI arguments
    pub fn from_args(
        provider_type: Option<&str>,
        bucket: Option<&str>,
        region: Option<&str>,
        endpoint: Option<&str>,
        prefix: &str,
        auto_backup: bool,
    ) -> Result<Self, String> {
        let provider = match provider_type {
            Some("s3") => {
                let bucket = bucket.ok_or("--cloud-bucket is required for S3")?;
                let region = region.unwrap_or("us-east-1");
                Some(CloudProvider::S3 {
                    bucket: bucket.to_string(),
                    region: region.to_string(),
                    endpoint: endpoint.map(|s| s.to_string()),
                })
            }
            Some("gcs") => {
                let bucket = bucket.ok_or("--cloud-bucket is required for GCS")?;
                Some(CloudProvider::GCS {
                    bucket: bucket.to_string(),
                })
            }
            Some("azure") => {
                let container = bucket.ok_or("--cloud-bucket is required for Azure")?;
                let account = std::env::var("AZURE_STORAGE_ACCOUNT_NAME")
                    .map_err(|_| "AZURE_STORAGE_ACCOUNT_NAME env var is required for Azure")?;
                Some(CloudProvider::Azure {
                    container: container.to_string(),
                    account,
                })
            }
            Some("local") => {
                let path = bucket.ok_or("--cloud-bucket (path) is required for local provider")?;
                Some(CloudProvider::Local {
                    path: path.to_string(),
                })
            }
            Some(unknown) => return Err(format!("Unknown cloud provider: {}", unknown)),
            None => None,
        };

        Ok(Self {
            enabled: provider.is_some(),
            provider,
            prefix: prefix.to_string(),
            auto_backup,
        })
    }
}

/// Backup entry metadata
#[derive(Debug, Clone, Serialize)]
pub struct BackupEntry {
    pub project_id: String,
    pub size_bytes: u64,
    pub last_modified: String,
    pub path: String,
}

/// Cloud backup manager for uploading/downloading snapshots to cloud storage
pub struct CloudBackupManager {
    config: CloudBackupConfig,
    store: Arc<dyn ObjectStore>,
}

impl CloudBackupManager {
    /// Create a new cloud backup manager
    pub async fn new(
        config: CloudBackupConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let provider = config
            .provider
            .as_ref()
            .ok_or("No cloud provider configured")?;

        let store: Arc<dyn ObjectStore> = match provider {
            CloudProvider::S3 {
                bucket,
                region,
                endpoint,
            } => {
                info!(
                    "Initializing S3 cloud backup: bucket={}, region={}",
                    bucket, region
                );

                let mut builder = AmazonS3Builder::from_env()
                    .with_bucket_name(bucket)
                    .with_region(region);

                if let Some(ep) = endpoint {
                    info!("Using custom S3 endpoint: {}", ep);
                    builder = builder
                        .with_endpoint(ep)
                        .with_virtual_hosted_style_request(false);
                }

                Arc::new(builder.build()?)
            }
            CloudProvider::GCS { bucket } => {
                info!("Initializing GCS cloud backup: bucket={}", bucket);

                let builder = GoogleCloudStorageBuilder::from_env().with_bucket_name(bucket);
                Arc::new(builder.build()?)
            }
            CloudProvider::Azure { container, account } => {
                info!(
                    "Initializing Azure cloud backup: account={}, container={}",
                    account, container
                );

                let builder = MicrosoftAzureBuilder::from_env()
                    .with_account(account)
                    .with_container_name(container);
                Arc::new(builder.build()?)
            }
            CloudProvider::Local { path } => {
                info!("Initializing local cloud backup: path={}", path);

                // Create directory if it doesn't exist
                std::fs::create_dir_all(path)?;
                Arc::new(LocalFileSystem::new_with_prefix(path)?)
            }
        };

        info!("Cloud backup manager initialized successfully");

        Ok(Self { config, store })
    }

    /// Get the object path for a project snapshot
    fn get_object_path(&self, project_id: &str, suffix: &str) -> ObjectPath {
        let key = format!("{}{}{}", self.config.prefix, project_id, suffix);
        ObjectPath::from(key)
    }

    /// Upload a single snapshot file to cloud storage
    pub async fn upload_snapshot(
        &self,
        project_id: &str,
        data: Bytes,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let path = self.get_object_path(project_id, ".bin");
        let size = data.len() as u64;

        debug!("Uploading snapshot: {} ({} bytes)", path, size);

        let payload = PutPayload::from_bytes(data);
        self.store.put(&path, payload).await?;

        info!("Uploaded snapshot: {} ({} bytes)", project_id, size);
        Ok(size)
    }

    /// Upload all 3 engine files for a project (main, aliases, lexicon)
    pub async fn upload_project_snapshot(
        &self,
        project_id: &str,
        main_data: Bytes,
        aliases_data: Option<Bytes>,
        lexicon_data: Option<Bytes>,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let mut total_size = 0u64;

        // Upload main engine
        let main_path = self.get_object_path(project_id, ".bin");
        let main_size = main_data.len() as u64;
        self.store
            .put(&main_path, PutPayload::from_bytes(main_data))
            .await?;
        total_size += main_size;
        debug!("Uploaded main: {} ({} bytes)", main_path, main_size);

        // Upload aliases engine if provided
        if let Some(data) = aliases_data {
            let path = self.get_object_path(project_id, "_aliases.bin");
            let size = data.len() as u64;
            self.store.put(&path, PutPayload::from_bytes(data)).await?;
            total_size += size;
            debug!("Uploaded aliases: {} ({} bytes)", path, size);
        }

        // Upload lexicon engine if provided
        if let Some(data) = lexicon_data {
            let path = self.get_object_path(project_id, "_lexicon.bin");
            let size = data.len() as u64;
            self.store.put(&path, PutPayload::from_bytes(data)).await?;
            total_size += size;
            debug!("Uploaded lexicon: {} ({} bytes)", path, size);
        }

        info!(
            "Uploaded project snapshot: {} ({} bytes total)",
            project_id, total_size
        );
        Ok(total_size)
    }

    /// Download a project snapshot from cloud storage
    pub async fn download_snapshot(
        &self,
        project_id: &str,
    ) -> Result<Bytes, Box<dyn std::error::Error + Send + Sync>> {
        let path = self.get_object_path(project_id, ".bin");

        debug!("Downloading snapshot: {}", path);

        let result = self.store.get(&path).await?;
        let data = result.bytes().await?;

        info!("Downloaded snapshot: {} ({} bytes)", project_id, data.len());
        Ok(data)
    }

    /// Download all 3 engine files for a project (main, aliases, lexicon)
    pub async fn download_project_snapshot(
        &self,
        project_id: &str,
    ) -> Result<(Bytes, Option<Bytes>, Option<Bytes>), Box<dyn std::error::Error + Send + Sync>>
    {
        // Download main engine (required)
        let main_path = self.get_object_path(project_id, ".bin");
        let main_result = self.store.get(&main_path).await?;
        let main_data = main_result.bytes().await?;

        // Download aliases engine (optional)
        let aliases_path = self.get_object_path(project_id, "_aliases.bin");
        let aliases_data = match self.store.get(&aliases_path).await {
            Ok(result) => Some(result.bytes().await?),
            Err(object_store::Error::NotFound { .. }) => None,
            Err(e) => {
                warn!("Failed to download aliases for {}: {}", project_id, e);
                None
            }
        };

        // Download lexicon engine (optional)
        let lexicon_path = self.get_object_path(project_id, "_lexicon.bin");
        let lexicon_data = match self.store.get(&lexicon_path).await {
            Ok(result) => Some(result.bytes().await?),
            Err(object_store::Error::NotFound { .. }) => None,
            Err(e) => {
                warn!("Failed to download lexicon for {}: {}", project_id, e);
                None
            }
        };

        info!(
            "Downloaded project snapshot: {} (main: {} bytes, aliases: {:?}, lexicon: {:?})",
            project_id,
            main_data.len(),
            aliases_data.as_ref().map(|d| d.len()),
            lexicon_data.as_ref().map(|d| d.len())
        );

        Ok((main_data, aliases_data, lexicon_data))
    }

    /// List all available cloud backups
    pub async fn list_snapshots(
        &self,
    ) -> Result<Vec<BackupEntry>, Box<dyn std::error::Error + Send + Sync>> {
        use futures::TryStreamExt;

        let prefix = ObjectPath::from(self.config.prefix.clone());

        debug!("Listing snapshots with prefix: {}", prefix);

        let mut entries = Vec::new();
        let mut list_stream = self.store.list(Some(&prefix));

        while let Some(meta) = list_stream.try_next().await? {
            let path_str = meta.location.to_string();

            // Only include main engine files (not aliases/lexicon)
            if path_str.ends_with(".bin")
                && !path_str.ends_with("_aliases.bin")
                && !path_str.ends_with("_lexicon.bin")
            {
                // Extract project_id from path
                let filename = path_str
                    .strip_prefix(&self.config.prefix)
                    .unwrap_or(&path_str);
                let project_id = filename.strip_suffix(".bin").unwrap_or(filename);

                entries.push(BackupEntry {
                    project_id: project_id.to_string(),
                    size_bytes: meta.size as u64,
                    last_modified: meta.last_modified.to_rfc3339(),
                    path: path_str,
                });
            }
        }

        info!("Found {} cloud backups", entries.len());
        Ok(entries)
    }

    /// Delete a project snapshot from cloud storage
    pub async fn delete_snapshot(
        &self,
        project_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Delete all 3 engine files
        let paths = vec![
            self.get_object_path(project_id, ".bin"),
            self.get_object_path(project_id, "_aliases.bin"),
            self.get_object_path(project_id, "_lexicon.bin"),
        ];

        for path in paths {
            match self.store.delete(&path).await {
                Ok(_) => debug!("Deleted: {}", path),
                Err(object_store::Error::NotFound { .. }) => {
                    debug!("Not found (skipped): {}", path);
                }
                Err(e) => {
                    error!("Failed to delete {}: {}", path, e);
                    return Err(e.into());
                }
            }
        }

        info!("Deleted cloud backup: {}", project_id);
        Ok(())
    }

    /// Check if cloud backup is configured for auto-backup
    pub fn is_auto_backup_enabled(&self) -> bool {
        self.config.auto_backup
    }

    /// Get the cloud provider configuration
    pub fn get_config(&self) -> &CloudBackupConfig {
        &self.config
    }
}

#[cfg(test)]
#[path = "../tests/unit/persistence.rs"]
mod tests;
