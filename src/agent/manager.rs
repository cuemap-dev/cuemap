use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, error};

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

        info!("AgentManager: Spawning new Agent for project '{}'", project_id);
        
        match Agent::new(config, self.job_queue.clone(), self.provider.clone()) {
            Ok(agent) => {
                let agent = Arc::new(agent);
                agent.start().await;
                
                let mut locked = self.agents.write().await;
                locked.insert(project_id.to_string(), agent);
                info!("AgentManager: Successfully spawned Agent for '{}'", project_id);
            }
            Err(e) => {
                error!("AgentManager: Failed to initialize Agent for '{}': {}", project_id, e);
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
