use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{warn, debug};
use thesaurus::synonyms;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::collections::HashSet;
use postagger::PerceptronTagger;

static POS_TAGGER: OnceLock<Option<Arc<PerceptronTagger>>> = OnceLock::new();

#[derive(Clone)]
pub struct SemanticEngine {
    // Shared, mutex-protected LRU cache for WordNet results
    // Usage: cache[word] = list of synonyms
    wordnet_cache: Arc<Mutex<LruCache<String, Vec<String>>>>,
    // POS tagger for filtering non-nouns from expansion
    pos_tagger: Option<Arc<PerceptronTagger>>,
}

impl SemanticEngine {
    pub fn new(_data_dir: Option<&Path>) -> Self {
        debug!("SemanticEngine initialized in deterministic WordNet/POS mode; embeddings are deprecated in v0.7");

        // Initialize LRU cache with capacity 10,000
        let cache = LruCache::new(NonZeroUsize::new(10000).unwrap());

        let pos_tagger = POS_TAGGER.get_or_init(|| {
            // Embed the files directly into the binary
            let weights_data = include_bytes!("../data/tagger/weights.json");
            let classes_data = include_bytes!("../data/tagger/classes.txt");
            let tags_data = include_bytes!("../data/tagger/tags.json");

            // Write them to temporary files so postagger can read them (since its API requires file paths)
            let temp_dir = std::env::temp_dir().join("cuemap_tagger");
            if let Err(e) = std::fs::create_dir_all(&temp_dir) {
                warn!("Failed to create temp directory for POS tagger: {}", e);
                None
            } else {
                let weights_path = temp_dir.join("weights.json");
                let classes_path = temp_dir.join("classes.txt");
                let tags_path = temp_dir.join("tags.json");

                let mut success = true;
                if let Err(e) = std::fs::write(&weights_path, weights_data) { warn!("Failed to write temp weights.json: {}", e); success = false; }
                if let Err(e) = std::fs::write(&classes_path, classes_data) { warn!("Failed to write temp classes.txt: {}", e); success = false; }
                if let Err(e) = std::fs::write(&tags_path, tags_data) { warn!("Failed to write temp tags.json: {}", e); success = false; }

                if success {
                    let w_str = weights_path.to_str().unwrap_or_default();
                    let c_str = classes_path.to_str().unwrap_or_default();
                    let t_str = tags_path.to_str().unwrap_or_default();
                    
                    tracing::info!("Loading embedded POS tagger from {:?}", temp_dir);
                    Some(Arc::new(PerceptronTagger::new(w_str, c_str, t_str)))
                } else {
                    None
                }
            }
        }).clone();

        Self { 
            wordnet_cache: Arc::new(Mutex::new(cache)),
            pos_tagger,
        }
    }

    /// Expand cues using WordNet with POS filtering and deterministic synonym order.
    pub fn expand_wordnet(&self, content: &str, known_cues: &[String], _threshold: f32, limit: usize) -> Vec<String> {
        let mut new_cues = Vec::new();
        
        // 1. Identify unique input words
        let mut words_to_lookup = HashSet::new();
        
        debug!("Input known_cues: {:?}", known_cues);
        
        // POS-based filtering
        let allowed_by_pos: Option<HashSet<String>> = if let Some(tagger) = &self.pos_tagger {
            // The postagger crate has byte boundary bugs with non-ASCII UTF-8 chars
            // (e.g., Turkish 'ğ', Arabic text, emoji). Convert to ASCII-safe before tagging.
            // This is acceptable since POS tagging is only for filtering semantic expansion,
            // not for the actual content we store.
            let sanitized: String = content.chars()
                .filter(|c| c.is_ascii())
                .collect();
            
            let tags = tagger.tag(&sanitized);
            let mut allowed = HashSet::new();
            
            // Debug logs for tagging
            if !tags.is_empty() {
                let debug_tags: Vec<String> = tags.iter().take(10).map(|t| format!("{}({})", t.word, t.tag)).collect();
                debug!("POS Tags for '{}': {:?}", content.chars().take(50).collect::<String>(), debug_tags);
            }

            for tag in tags {
                let tag_str = &tag.tag;
                let word_lower = tag.word.to_lowercase();
                
                // Allow Nouns (NN*) and specific Adjectives (JJ*)
                // We rely on the fact that generic adjectives usually don't have useful synonyms or are handled downstream
                if tag_str.starts_with("NN") || tag_str.starts_with("JJ") {
                    allowed.insert(word_lower);
                }
            }
            Some(allowed)
        } else {
            None
        };

        for cue in known_cues {
            let word = if let Some((key, value)) = cue.split_once(':') {
                if key == "id" || key == "path" || key == "source" || key == "file" || key == "type" || key == "status" || key == "reason" {
                    continue;
                }
                value
            } else {
                cue.as_str()
            };
            
            let word_lower = word.to_lowercase();
            
            // 1. Check POS allowed (if available)
            if let Some(allowed) = &allowed_by_pos {
                let is_allowed = allowed.contains(&word_lower);
                // debug!("Checking cue '{}' ({}): allowed={}", word, word_lower, is_allowed);
                if !is_allowed {
                    continue;
                }
            }
            
            words_to_lookup.insert(word.to_string());
        }
        
        // Debug filtering results
        if let Some(allowed) = &allowed_by_pos {
             debug!("Allowed POS words ({}) : {:?}", allowed.len(), allowed);
        }
        debug!("Final Words to Lookup: {:?}", words_to_lookup);

        if words_to_lookup.is_empty() {
            return Vec::new();
        }

        // 2. Deterministic WordNet processing. Sort input words and preserve thesaurus order.
        let mut words: Vec<String> = words_to_lookup.into_iter().collect();
        words.sort();
        let mut results = Vec::new();
        for word in words {
            let cached = {
                let cache = self.wordnet_cache.lock().unwrap();
                cache.peek(&word).cloned()
            };
            
            let raw_syns = if let Some(syns) = cached {
                syns
            } else {
                let syns = synonyms(&word);
                {
                    let mut cache = self.wordnet_cache.lock().unwrap();
                    cache.put(word.clone(), syns.clone());
                }
                syns
            };
            
            if !raw_syns.is_empty() {
                debug!("Synonyms for '{}': {:?}", word, raw_syns);
            }

            results.extend(
                raw_syns.into_iter()
                    .filter(|syn| syn.len() > 2 && syn != &word && !syn.contains(' '))
                    .take(limit)
            );
        }
            
        new_cues.extend(results);
        new_cues
    }


    /// Expand cues using GloVe embeddings (if available)
    pub fn expand_glove(&self, _content: &str, _known_cues: &[String]) -> Vec<String> {
        Vec::new()
    }

    /// Expand cues based on the global context of the content
    /// Finds neighbors to the mean context vector
    pub fn expand_global_context(&self, _content: &str) -> Vec<String> {
        Vec::new()
    }
}
