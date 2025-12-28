pub mod chunker;
pub mod watcher;
pub mod ingester;

use crate::jobs::JobQueue;
use crate::jobs::ProjectProvider;
use crate::llm::LlmConfig;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

#[derive(Clone)]
pub struct AgentConfig {
    pub watch_dir: String,
    pub throttle_ms: u64,
    pub llm: LlmConfig,
}

pub struct Agent {
    _config: AgentConfig,
    ingester: Arc<Mutex<ingester::Ingester>>,
    _watcher: watcher::Watcher,
}

impl Agent {
    pub fn new(
        config: AgentConfig,
        job_queue: Arc<JobQueue>,
        _provider: Arc<dyn ProjectProvider>, // Might be needed for direct access later
    ) -> Result<Self, String> {
        info!("Initializing Self-Learning Agent watching: {}", config.watch_dir);

        let ingester = Arc::new(Mutex::new(ingester::Ingester::new(
            config.clone(),
            job_queue,
        )));

        // Create watcher that pipes events to ingester
        let watcher = watcher::Watcher::new(config.watch_dir.clone(), ingester.clone())
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
        
        // Trigger initial scan
        let ingester = self.ingester.clone();
        tokio::spawn(async move {
            let mut ingester = ingester.lock().await;
            if let Err(e) = ingester.scan_all().await {
                warn!("Initial scan failed: {}", e);
            }
        });
    }
}
