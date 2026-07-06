use cuemap::persistence::PersistenceManager;
use cuemap::structures::{LexiconStats, MainStats};
use std::path::Path;

fn main() {
    let path =
        Path::new("/Users/kaan/cuemap/rust_engine/snapshots/stackoverflow_lexicon_python.bin");

    match PersistenceManager::load_from_path::<MainStats>(path) {
        Ok((memories, _, cues, _, _counts)) => {
            println!(
                "Loaded as MainStats! Memories: {}, Cues: {}",
                memories.len(),
                cues.len()
            );
        }
        Err(e) => {
            println!("Failed to load as MainStats: {}", e);
            match PersistenceManager::load_from_path::<LexiconStats>(path) {
                Ok((memories, _, cues, _, _counts)) => {
                    println!(
                        "Loaded as LexiconStats! Memories: {}, Cues: {}",
                        memories.len(),
                        cues.len()
                    );
                }
                Err(e) => {
                    println!("Failed to load as LexiconStats: {}", e);
                }
            }
        }
    }
}
