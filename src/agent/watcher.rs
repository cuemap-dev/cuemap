use crate::agent::ingester::Ingester;
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error};

fn dispatch_event(
    res: notify::Result<Event>,
    ingester: Arc<Mutex<Ingester>>,
    state_file: Option<std::path::PathBuf>,
    handle: tokio::runtime::Handle,
) {
    match res {
        Ok(event) => {
            if event
                .paths
                .iter()
                .any(|path| Ingester::is_ignore_config_path(path))
            {
                handle.spawn(async move {
                    let mut locked = ingester.lock().await;
                    if let Err(e) = locked.reload_filters_and_rescan().await {
                        error!("Error reloading ignore configuration: {}", e);
                    }
                    if let Some(ref sp) = state_file {
                        let _ = locked.save_state(sp);
                    }
                });
                return;
            }

            if event.kind.is_remove() {
                for path in event.paths {
                    let ingester = ingester.clone();
                    let state_file = state_file.clone();
                    handle.spawn(async move {
                        let mut locked = ingester.lock().await;
                        if let Err(e) = locked.delete_file_path(path.clone()).await {
                            error!("Error processing deletion {:?}: {}", path, e);
                        }
                        if let Some(ref sp) = state_file {
                            let _ = locked.save_state(sp);
                        }
                    });
                }
            } else {
                for path in event.paths {
                    if path.exists() || path.extension().is_some() {
                        debug!("File event {:?}: {:?}", event.kind, path);
                        let ingester = ingester.clone();
                        let state_file = state_file.clone();
                        handle.spawn(async move {
                            let mut locked = ingester.lock().await;
                            if let Err(e) = locked.process_file_path(path.clone()).await {
                                debug!("Skipping file {:?}: {}", path, e);
                            }
                            if let Some(ref sp) = state_file {
                                let _ = locked.save_state(sp);
                            }
                        });
                    }
                }
            }
        }
        Err(e) => error!("Watch error: {:?}", e),
    }
}

pub struct Watcher {
    _watcher: RecommendedWatcher,
}

impl Watcher {
    pub fn new(
        path: String,
        ingester: Arc<Mutex<Ingester>>,
        state_file: Option<std::path::PathBuf>,
    ) -> notify::Result<Self> {
        let path_obj = Path::new(&path);

        let handle = tokio::runtime::Handle::current();

        let watcher_plugin = move |res: notify::Result<Event>| {
            dispatch_event(res, ingester.clone(), state_file.clone(), handle.clone());
        };

        let mut watcher = notify::recommended_watcher(watcher_plugin)?;

        watcher.watch(path_obj, RecursiveMode::Recursive)?;

        Ok(Self { _watcher: watcher })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentConfig, ingester::Ingester};
    use crate::jobs::JobQueue;
    use crate::multi_tenant::MultiTenantEngine;
    use crate::config::TuningConfig;
    use notify::EventKind;
    use std::time::Duration;

    async fn wait_for_file_state(
        ingester: &Arc<Mutex<Ingester>>,
        path: &std::path::Path,
        expected: bool,
    ) {
        let normalized_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let key = normalized_path.to_string_lossy().to_lowercase();
        let suffix = format!("/{}", path.file_name().unwrap().to_string_lossy().to_lowercase());
        for _ in 0..80 {
            let present = ingester
                .lock()
                .await
                .get_file_hashes()
                .keys()
                .any(|candidate| candidate == &key || candidate.ends_with(&suffix));
            if present == expected {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let present = ingester
            .lock()
            .await
            .get_file_hashes()
            .keys()
            .any(|candidate| candidate == &key || candidate.ends_with(&suffix));
        assert_eq!(present, expected, "timed out waiting for {:?}", path);
    }

    #[tokio::test]
    async fn watcher_can_attach_to_an_existing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(MultiTenantEngine::with_snapshots_dir(
            dir.path().join("snapshots"),
            TuningConfig::default(),
        ));
        let queue = Arc::new(JobQueue::new(provider.clone(), None, true));
        let config = AgentConfig {
            project_id: "watcher-test".to_string(),
            watch_dir: dir.path().to_string_lossy().to_string(),
            throttle_ms: 1,
            state_file: None,
            included_paths: Vec::new(),
            ignored_patterns: Vec::new(),
            ignored_extensions: Vec::new(),
        };
        let ingester = Arc::new(Mutex::new(Ingester::new(config, queue)));
        let watcher = Watcher::new(dir.path().to_string_lossy().to_string(), ingester, None);
        assert!(watcher.is_ok());
    }

    #[tokio::test]
    async fn watcher_processes_updates_removals_and_ignore_config_changes() {
        let dir = tempfile::tempdir().unwrap();
        let state_file = dir.path().join("agent-state.json");
        let provider = Arc::new(MultiTenantEngine::with_snapshots_dir(
            dir.path().join("snapshots"),
            TuningConfig::default(),
        ));
        let queue = Arc::new(JobQueue::new(provider.clone(), None, true));
        let config = AgentConfig {
            project_id: "watcher-events".to_string(),
            watch_dir: dir.path().to_string_lossy().to_string(),
            throttle_ms: 0,
            state_file: Some(state_file.clone()),
            included_paths: Vec::new(),
            ignored_patterns: Vec::new(),
            ignored_extensions: Vec::new(),
        };
        let note = dir.path().join("notes.md");
        std::fs::write(&note, "first version").unwrap();
        let ingester = Arc::new(Mutex::new(Ingester::new(config, queue)));
        ingester
            .lock()
            .await
            .process_file_path(note.clone())
            .await
            .unwrap();
        let handle = tokio::runtime::Handle::current();
        dispatch_event(
            Ok(Event::new(EventKind::Create(notify::event::CreateKind::Any)).add_path(note.clone())),
            ingester.clone(),
            Some(state_file.clone()),
            handle.clone(),
        );
        wait_for_file_state(&ingester, &note, true).await;
        for _ in 0..20 {
            if state_file.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(state_file.exists());

        std::fs::write(&note, "third version").unwrap();
        dispatch_event(
            Ok(Event::new(EventKind::Modify(notify::event::ModifyKind::Any)).add_path(note.clone())),
            ingester.clone(),
            Some(state_file.clone()),
            handle.clone(),
        );
        wait_for_file_state(&ingester, &note, true).await;

        let ignore = dir.path().join(".cuemapignore");
        std::fs::write(&ignore, "notes.md\n").unwrap();
        dispatch_event(
            Ok(Event::new(EventKind::Modify(notify::event::ModifyKind::Any)).add_path(ignore)),
            ingester.clone(),
            Some(state_file.clone()),
            handle.clone(),
        );
        wait_for_file_state(&ingester, &note, false).await;

        std::fs::remove_file(&note).unwrap();
        dispatch_event(
            Ok(Event::new(EventKind::Remove(notify::event::RemoveKind::Any)).add_path(note.clone())),
            ingester.clone(),
            Some(state_file),
            handle,
        );
        wait_for_file_state(&ingester, &note, false).await;

        dispatch_event(Err(notify::Error::generic("synthetic watcher error")), ingester, None, tokio::runtime::Handle::current());
    }
}
