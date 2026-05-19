use dashmap::DashMap;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use ahash::RandomState;
use tracing::{info, warn};
use crate::persistence::PersistenceManager;

use std::sync::OnceLock;
use std::path::PathBuf;

#[derive(Clone)]
pub struct GlobalLexicons {
    /// Maps lexicon name to its metadata and lazy graph
    pub entries: Arc<DashMap<String, Arc<LexiconEntry>, RandomState>>,
}

pub struct LexiconEntry {
    pub path: PathBuf,
    pub data: OnceLock<Option<LexiconData>>,
}

#[derive(Clone)]
pub struct CompactGraph {
    pub vocab: Vec<String>,
    pub vocab_to_id: ahash::HashMap<String, u32>,
    pub edges: Vec<Vec<(u32, u64)>>,
    pub counts: Vec<u64>,
}

impl CompactGraph {
    pub fn from_maps(
        graph: DashMap<String, HashMap<String, u64>, RandomState>,
        counts: DashMap<String, u64, RandomState>,
    ) -> Self {
        let mut vocab = Vec::new();
        let mut vocab_to_id = ahash::HashMap::default();
        
        let mut get_id = |s: &str| -> u32 {
            if let Some(&id) = vocab_to_id.get(s) {
                id
            } else {
                let id = vocab.len() as u32;
                vocab.push(s.to_string());
                vocab_to_id.insert(s.to_string(), id);
                id
            }
        };

        // By using into_iter(), we consume and drop the DashMap nodes 
        // immediately as we convert them, freeing up memory rapidly.
        let mut edges_map: ahash::HashMap<u32, Vec<(u32, u64)>> = ahash::HashMap::default();
        let mut counts_map: ahash::HashMap<u32, u64> = ahash::HashMap::default();

        for entry in counts.into_iter() {
            let id = get_id(&entry.0);
            counts_map.insert(id, entry.1);
        }

        for entry in graph.into_iter() {
            let id_a = get_id(&entry.0);
            let mut edges = Vec::with_capacity(entry.1.len());
            for inner in entry.1.into_iter() {
                let id_b = get_id(&inner.0);
                edges.push((id_b, inner.1));
            }
            edges_map.insert(id_a, edges);
        }

        let len = vocab.len();
        let mut dense_edges = vec![Vec::new(); len];
        let mut dense_counts = vec![0; len];

        for (id, edges) in edges_map.into_iter() {
            dense_edges[id as usize] = edges;
        }
        for (id, count) in counts_map.into_iter() {
            dense_counts[id as usize] = count;
        }

        Self {
            vocab,
            vocab_to_id,
            edges: dense_edges,
            counts: dense_counts,
        }
    }
}

pub struct LexiconData {
    pub compact: CompactGraph,
}

impl GlobalLexicons {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(DashMap::with_hasher(RandomState::new())),
        }
    }

    pub fn load_from_dir(dir: impl AsRef<Path>) -> Self {
        let global = Self::new();
        let dir = dir.as_ref();
        
        if !dir.exists() {
            return global;
        }

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                
                if file_name.ends_with("_lexicon.bin") { continue; }

                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("bin") {
                    let name = file_name.trim_end_matches(".bin")
                        .trim_start_matches("stackoverflow_lexicon_")
                        .to_string();

                    info!("Found external lexicon: {}", name);
                    global.entries.insert(name, Arc::new(LexiconEntry {
                        path: path.clone(),
                        data: OnceLock::new(),
                    }));
                }
            }
        }
        
        global
    }
    
    pub fn get_lexicon(&self, name: &str) -> Option<Arc<LexiconEntry>> {
        self.entries.get(name).map(|e| e.value().clone())
    }

    pub fn list_available(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.key().clone()).collect()
    }
}

impl LexiconEntry {
    pub fn get_data(&self) -> Option<&LexiconData> {
        let data_opt = self.data.get_or_init(|| {
            info!("Lazy loading external lexicon from {:?}", self.path);
            match PersistenceManager::load_from_path::<crate::structures::MainStats>(&self.path) {
                Ok((_, _, opt_graph, opt_counts)) => {
                    match (opt_graph, opt_counts) {
                        (Some(graph), Some(counts)) => {
                            Some(LexiconData { compact: CompactGraph::from_maps(graph, counts) })
                        },
                        (Some(graph), None) => {
                            // Fallback: build counts from graph if missing
                            let counts = DashMap::with_hasher(RandomState::new());
                            for r in graph.iter() {
                                let total: u64 = r.value().values().sum();
                                counts.insert(r.key().clone(), total);
                            }
                            Some(LexiconData { compact: CompactGraph::from_maps(graph, counts) })
                        }
                        _ => {
                            warn!("No graph found in {:?}", self.path);
                            None
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to lazy load lexicon {:?}: {}", self.path, e);
                    None
                }
            }
        });

        data_opt.as_ref()
    }

    pub fn get_compact(&self) -> Option<&CompactGraph> {
        self.get_data().map(|d| &d.compact)
    }
}
