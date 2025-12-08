//! Persistence layer with bincode serialization and background snapshots.

use crate::engine::CueMapEngine;
use crate::structures::{Memory, OrderedSet};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::interval;
use tracing::{error, info, warn};

#[derive(Debug, Serialize, Deserialize)]
struct PersistedState {
    memories: HashMap<String, Memory>,
    cue_index: HashMap<String, Vec<String>>, // Flattened OrderedSet
    version: u32,
    saved_at: u64,
}

const PERSISTENCE_VERSION: u32 = 1;

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
    
    fn snapshot_path(&self) -> PathBuf {
        self.data_dir.join("cuemap.bin")
    }
    
    fn temp_snapshot_path(&self) -> PathBuf {
        self.data_dir.join("cuemap.bin.tmp")
    }
    
    pub fn load_state(
        &self,
    ) -> Result<(DashMap<String, Memory>, DashMap<String, OrderedSet>), Box<dyn std::error::Error>> {
        let snapshot_path = self.snapshot_path();
        
        if !snapshot_path.exists() {
            info!("No existing snapshot found, starting with empty state");
            return Ok((DashMap::new(), DashMap::new()));
        }
        
        info!("Loading state from {:?}", snapshot_path);
        
        let data = fs::read(&snapshot_path)?;
        let state: PersistedState = bincode::deserialize(&data)?;
        
        info!(
            "Loaded {} memories and {} cues from snapshot (version: {}, saved: {})",
            state.memories.len(),
            state.cue_index.len(),
            state.version,
            state.saved_at
        );
        
        // Convert to DashMaps
        let memories = DashMap::new();
        for (id, memory) in state.memories {
            memories.insert(id, memory);
        }
        
        let cue_index = DashMap::new();
        for (cue, memory_ids) in state.cue_index {
            let mut ordered_set = OrderedSet::new();
            for memory_id in memory_ids {
                ordered_set.add(memory_id);
            }
            cue_index.insert(cue, ordered_set);
        }
        
        Ok((memories, cue_index))
    }
    
    pub fn save_state(
        &self,
        engine: &CueMapEngine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start = std::time::Instant::now();
        
        let memories = engine.get_memories();
        let cue_index = engine.get_cue_index();
        
        // Convert DashMaps to serializable format
        let memories_map: HashMap<String, Memory> = memories
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        
        let cue_index_map: HashMap<String, Vec<String>> = cue_index
            .iter()
            .map(|entry| {
                let cue = entry.key().clone();
                let ordered_set = entry.value();
                let memory_ids = ordered_set.get_recent(None); // Get all items
                (cue, memory_ids)
            })
            .collect();
        
        let state = PersistedState {
            memories: memories_map,
            cue_index: cue_index_map,
            version: PERSISTENCE_VERSION,
            saved_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        // Serialize to bincode
        let data = bincode::serialize(&state)?;
        
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
    
    pub async fn start_background_snapshots(
        &self,
        engine: Arc<CueMapEngine>,
    ) -> tokio::task::JoinHandle<()> {
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
pub async fn setup_shutdown_handler(
    persistence: PersistenceManager,
    engine: Arc<CueMapEngine>,
) {
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
