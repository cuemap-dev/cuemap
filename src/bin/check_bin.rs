use cuemap::structures::{MainStats, LexiconStats};
use cuemap::persistence::PersistenceManager;
use std::path::Path;

fn main() {
    let path = Path::new("/Users/kaan/cuemap/rust_engine/snapshots/stackoverflow_lexicon_python.bin");
    
    match PersistenceManager::load_from_path::<MainStats>(path) {
        Ok((memories, cues, _cooc, graph)) => {
            println!("Loaded as MainStats! Memories: {}, Cues: {}, Graph links: {}", memories.len(), cues.len(), graph.map_or(0, |g| g.len()));
        }
        Err(e) => {
            println!("Failed to load as MainStats: {}", e);
            match PersistenceManager::load_from_path::<LexiconStats>(path) {
                Ok((memories, cues, _cooc, graph)) => {
                    println!("Loaded as LexiconStats! Memories: {}, Cues: {}, Graph links: {}", memories.len(), cues.len(), graph.map_or(0, |g| g.len()));
                }
                Err(e) => {
                    println!("Failed to load as LexiconStats: {}", e);
                }
            }
        }
    }
}
