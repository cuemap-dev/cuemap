use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info};

use crate::agent::{Agent, AgentConfig};
use crate::jobs::{JobQueue, ProjectProvider};

/// Manages dynamic per-project Agent instances
pub struct AgentManager {
    agents: RwLock<HashMap<String, Arc<Agent>>>,
    job_queue: Arc<JobQueue>,
    provider: Arc<dyn ProjectProvider>,
}

impl AgentManager {
    pub fn new(job_queue: Arc<JobQueue>, provider: Arc<dyn ProjectProvider>) -> Self {
        Self {
            agents: RwLock::new(HashMap::new()),
            job_queue,
            provider,
        }
    }

    /// Starts or updates an agent for the specified project.
    pub async fn start_agent(&self, project_id: &str, config: AgentConfig) {
        // If an agent is already running for this project, stop it first to ensure clean handoff
        self.stop_agent(project_id).await;

        info!(
            "AgentManager: Spawning new Agent for project '{}'",
            project_id
        );

        match Agent::new(config, self.job_queue.clone(), self.provider.clone()) {
            Ok(agent) => {
                let agent = Arc::new(agent);
                agent.start().await;

                let mut locked = self.agents.write().await;
                locked.insert(project_id.to_string(), agent);
                info!(
                    "AgentManager: Successfully spawned Agent for '{}'",
                    project_id
                );
            }
            Err(e) => {
                error!(
                    "AgentManager: Failed to initialize Agent for '{}': {}",
                    project_id, e
                );
            }
        }
    }

    /// Stops the agent by dropping it (which aborts the file watcher)
    pub async fn stop_agent(&self, project_id: &str) {
        let mut locked = self.agents.write().await;
        if locked.remove(project_id).is_some() {
            info!("AgentManager: Stopped Agent for '{}'", project_id);
        }
    }

    /// Retrieve the running agent if it exists
    pub async fn get_agent(&self, project_id: &str) -> Option<Arc<Agent>> {
        let locked = self.agents.read().await;
        locked.get(project_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TuningConfig;
    use crate::multi_tenant::MultiTenantEngine;

    fn config(project_id: &str, watch_dir: &std::path::Path) -> AgentConfig {
        AgentConfig {
            project_id: project_id.to_string(),
            watch_dir: watch_dir.to_string_lossy().to_string(),
            throttle_ms: 1,
            state_file: None,
            included_paths: Vec::new(),
            ignored_patterns: Vec::new(),
            ignored_extensions: Vec::new(),
        }
    }

    #[tokio::test]
    async fn manager_starts_replaces_and_stops_agents() {
        let dir = tempfile::tempdir().unwrap();
        let snapshots = dir.path().join("snapshots");
        let provider = Arc::new(MultiTenantEngine::with_snapshots_dir(
            &snapshots,
            TuningConfig::default(),
        ));
        let job_queue = Arc::new(JobQueue::new(provider.clone(), None, true));
        let manager = AgentManager::new(job_queue, provider);

        assert!(manager.get_agent("project").await.is_none());
        manager
            .start_agent("project", config("project", dir.path()))
            .await;
        assert!(manager.get_agent("project").await.is_some());

        manager
            .start_agent("project", config("project", dir.path()))
            .await;
        assert!(manager.get_agent("project").await.is_some());

        manager.stop_agent("project").await;
        assert!(manager.get_agent("project").await.is_none());
        manager.stop_agent("project").await;
    }

    #[tokio::test]
    async fn manager_does_not_register_agent_when_watcher_cannot_initialize() {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(MultiTenantEngine::with_snapshots_dir(
            dir.path().join("snapshots"),
            TuningConfig::default(),
        ));
        let job_queue = Arc::new(JobQueue::new(provider.clone(), None, true));
        let manager = AgentManager::new(job_queue, provider);
        manager
            .start_agent(
                "missing",
                config("missing", &dir.path().join("does-not-exist")),
            )
            .await;
        assert!(manager.get_agent("missing").await.is_none());
    }
}
