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
    pub fn new(path: String, ingester: Arc<Mutex<Ingester>>) -> notify::Result<Self> {
        let path_obj = Path::new(&path);
        
        let tx_ingester = ingester.clone();
        let handle = tokio::runtime::Handle::current();
        
        let watcher_plugin = move |res: notify::Result<Event>| {
            match res {
                Ok(event) => {
                    // Filter for Modify, Create, Remove
                    if event.kind.is_modify() || event.kind.is_create() {
                        for path in event.paths {
                            if path.is_file() {
                                debug!("File changed: {:?}", path);
                                let ingester = tx_ingester.clone();
                                // Spawn onto the specific runtime handle
                                handle.spawn(async move {
                                    let mut locked = ingester.lock().await;
                                    if let Err(e) = locked.process_file_path(path.clone()).await {
                                       // reduce noise
                                       debug!("Skipping file {:?}: {}", path, e);
                                    }
                                });
                            }
                        }
                    } else if event.kind.is_remove() {
                        for path in event.paths {
                            debug!("File removed: {:?}", path);
                            let ingester = tx_ingester.clone();
                            handle.spawn(async move {
                                let mut locked = ingester.lock().await;
                                if let Err(e) = locked.delete_file_path(path.clone()).await {
                                    error!("Error processing deletion {:?}: {}", path, e);
                                }
                            });
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
