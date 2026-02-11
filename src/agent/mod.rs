pub mod chunker;
pub mod watcher;
pub mod ingester;
pub mod search;

use crate::jobs::JobQueue;
use crate::jobs::ProjectProvider;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

#[derive(Clone)]
pub struct AgentConfig {
    pub watch_dir: String,
    pub throttle_ms: u64,
    pub state_file: Option<std::path::PathBuf>,
}

pub struct Agent {
    _config: AgentConfig,
    ingester: Arc<Mutex<ingester::Ingester>>,
    _watcher: watcher::Watcher,
}

impl Agent {
    pub fn new(
        mut config: AgentConfig,
        job_queue: Arc<JobQueue>,
        _provider: Arc<dyn ProjectProvider>, // Might be needed for direct access later
    ) -> Result<Self, String> {
        // Canonicalize watch_dir to ensure absolute path matching works across the engine
        if let Ok(abs_path) = std::fs::canonicalize(&config.watch_dir) {
            config.watch_dir = abs_path.to_string_lossy().to_string();
        }
        
        info!("Initializing Self-Learning Agent watching: {}", config.watch_dir);

        let mut ingester_obj = ingester::Ingester::new(
            config.clone(),
            job_queue,
        );

        if let Some(ref state_path) = config.state_file {
            if let Err(e) = ingester_obj.load_state(state_path) {
                warn!("Failed to load agent state: {}", e);
            }
        }

        let ingester = Arc::new(Mutex::new(ingester_obj));

        let watcher = watcher::Watcher::new(config.watch_dir.clone(), ingester.clone(), config.state_file.clone())
            .map_err(|e| format!("Failed to create watcher: {}", e))?;

        Ok(Self {
            _config: config,
            ingester,
            _watcher: watcher,
        })
    }

    pub async fn start(&self) {
        info!("Agent started.");
        // Watcher runs in its own thread/task locally managed
        
        let ingester = self.ingester.clone();
        let state_file = self._config.state_file.clone();
        tokio::spawn(async move {
            let mut ingester = ingester.lock().await;
            if let Err(e) = ingester.scan_all().await {
                warn!("Initial scan failed: {}", e);
            }
            
            // Save state after initial scan
            if let Some(path) = state_file {
                if let Err(e) = ingester.save_state(&path) {
                    warn!("Failed to save agent state after initial scan: {}", e);
                }
            }
        });
    }

    pub fn get_ingester(&self) -> Arc<Mutex<ingester::Ingester>> {
        self.ingester.clone()
    }
}

