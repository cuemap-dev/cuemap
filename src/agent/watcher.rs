use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, debug};
use crate::agent::ingester::Ingester;

pub struct Watcher {
    _watcher: RecommendedWatcher,
}

impl Watcher {
    pub fn new(path: String, ingester: Arc<Mutex<Ingester>>, state_file: Option<std::path::PathBuf>) -> notify::Result<Self> {
        let path_obj = Path::new(&path);
        
        let tx_ingester = ingester.clone();
        let tx_state_file = state_file.clone();
        let handle = tokio::runtime::Handle::current();
        
        let watcher_plugin = move |res: notify::Result<Event>| {
            match res {
                Ok(event) => {
                    if event.kind.is_remove() {
                        for path in event.paths {
                            let ingester = tx_ingester.clone();
                            let state_file = tx_state_file.clone();
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
                        // Treat everything else as a potential update (Create, Modify, Rename, etc.)
                        // The Ingester's process_file_path checks if file exists and hashes it,
                        // so spurious events are cheap/safe.
                        for path in event.paths {
                             // Only process if it looks like a file we care about (simple check)
                             // detailed check is in ingester
                            if path.exists() || path.extension().is_some() {
                                debug!("File event {:?}: {:?}", event.kind, path);
                                let ingester = tx_ingester.clone();
                                let state_file = tx_state_file.clone();
                                handle.spawn(async move {
                                    let mut locked = ingester.lock().await;
                                    // this handles existence check internally
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
                },
                Err(e) => error!("Watch error: {:?}", e),
            }
        };

        let mut watcher = notify::recommended_watcher(watcher_plugin)?;

        watcher.watch(path_obj, RecursiveMode::Recursive)?;

        Ok(Self {
            _watcher: watcher,
        })
    }
}
