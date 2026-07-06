use cuemap::persistence::PersistenceManager;
use std::env;
use std::path::Path;

use cuemap::structures::MainStats;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: inspect_snapshot <path_to_snapshot>");
        return;
    }
    let path = Path::new(&args[1]);

    // We can use the public load_from_path
    match PersistenceManager::load_from_path::<MainStats>(path) {
        Ok((memories, _source_key_to_id, cue_index, _next_memory_id, _counts)) => {
            println!("Snapshot Summary for {:?}", path);
            println!("----------------------------------------");
            println!("Total Memories: {}", memories.len());
            println!("Total Cues:     {}", cue_index.len());
            println!("----------------------------------------\n");

            // Sort keys for deterministic output
            let mut keys: Vec<u32> = Vec::new();
            for entry in memories.iter() {
                keys.push(*entry.key());
            }
            keys.sort();

            for key in keys {
                let memory = memories.get(&key).unwrap();
                println!("ID: {}", key);
                println!("  Cues:    {:?}", memory.cues);
                // Preview content (first 50 chars)
                // Preview content type
                // Use the crypto helper to guess type
                let preview = if cuemap::crypto::is_compressed(&memory.content) {
                    "[Compressed Zstd]"
                } else {
                    "[Encrypted/Binary]"
                };
                println!("  Content: {}", preview);
                println!("");
            }
        }
        Err(e) => eprintln!("Error loading snapshot: {}", e),
    }
}
