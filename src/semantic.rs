use std::path::Path;
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::BufReader;
use tracing::{warn, debug};
use finalfusion::prelude::*;
use finalfusion::vocab::Vocab;
use finalfusion::storage::Storage;
use finalfusion::io::{ReadEmbeddings, WriteEmbeddings};
use thesaurus::synonyms;
use ndarray::{ArrayView1, Array1};
use std::cmp::Ordering;
use lru::LruCache;
use std::num::NonZeroUsize;
use rayon::prelude::*;
use std::collections::HashSet;
use postagger::PerceptronTagger;

#[derive(Clone)]
pub struct SemanticEngine {
    // Shared reference to embeddings to avoid reloading per project
    embeddings: Option<Arc<Embeddings<VocabWrap, StorageWrap>>>,
    // Shared, mutex-protected LRU cache for WordNet results
    // Usage: cache[word] = list of synonyms
    wordnet_cache: Arc<Mutex<LruCache<String, Vec<String>>>>,
    // POS tagger for filtering non-nouns from expansion
    pos_tagger: Option<Arc<PerceptronTagger>>,
}

impl SemanticEngine {
    pub fn new(data_dir: Option<&Path>) -> Self {
        let embeddings = if let Some(dir) = data_dir {
            // Try to load GloVe embeddings if provided
            // Priority 1: Finalfusion format (Fast)
            let fifu_path = dir.join("glove.50d.fifu");
            // Priority 2: Standard Text format (Slow load, but common)
            let txt_path = dir.join("glove.6B.50d.txt");

            if fifu_path.exists() {
                debug!("Loading semantic memory from {:?}", fifu_path);
                match File::open(&fifu_path) {
                    Ok(f) => {
                        let mut reader = BufReader::new(f);
                        match Embeddings::read_embeddings(&mut reader) {
                            Ok(emb) => {
                                debug!("Loaded {} word vectors (Binary)", emb.len());
                                Some(Arc::new(emb))
                            },
                            Err(e) => {
                                warn!("Failed to parse embeddings: {}", e);
                                None
                            }
                        }
                    },
                    Err(e) => {
                        warn!("Failed to open embeddings file: {}", e);
                        None
                    }
                }
            } else if txt_path.exists() {
                debug!("Found text embeddings at {:?}. Checking format...", txt_path);
                match File::open(&txt_path) {
                    Ok(mut f) => {
                        // Check for header
                        let mut buffer = [0u8; 20];
                        use std::io::{Read, Seek, SeekFrom};


                        let has_header = if let Ok(_) = f.read_exact(&mut buffer) {
                            // Reset position
                            let _ = f.seek(SeekFrom::Start(0));
                            // Check if starts with number (Word2Vec) or word (GloVe)
                            buffer[0].is_ascii_digit()
                        } else {
                            false 
                        };

                        if has_header {
                            // Word2Vec format with header - use standard reader
                            let mut reader = BufReader::new(f);
                            match Embeddings::read_embeddings(&mut reader) {
                                Ok(emb) => {
                                    debug!("Loaded {} word vectors (Word2Vec)", emb.len());
                                    Some(Arc::new(emb))
                                },
                                Err(e) => {
                                    warn!("Failed to parse text embeddings: {}", e);
                                    None
                                }
                            }
                        } else {
                            // GloVe format (no header) - use ReadText trait
                            use finalfusion::compat::text::ReadText;
                            debug!("Detected headerless GloVe format. Using ReadText...");
                            
                            let mut reader = BufReader::new(f);
                            match Embeddings::read_text(&mut reader) {
                                Ok(emb) => {
                                    debug!("Loaded {} word vectors (GloVe)", emb.len());
                                    // Convert to wrapped types for storage
                                    let emb_wrapped: Embeddings<VocabWrap, StorageWrap> = emb.into();
                                    // Optimization: Save as .fifu for next time
                                    let fifu_file = File::create(&fifu_path);
                                    match fifu_file {
                                        Ok(mut out) => {
                                            if let Err(e) = emb_wrapped.write_embeddings(&mut out) {
                                                warn!("Failed to save optimized binary: {}", e);
                                            } else {
                                                debug!("Saved optimized embeddings to {:?}", fifu_path);
                                            }
                                        },
                                        Err(e) => warn!("Could not create binary file: {}", e)
                                    }
                                    Some(Arc::new(emb_wrapped))
                                },
                                Err(e) => {
                                    warn!("Failed to parse GloVe embeddings: {}", e);
                                    None
                                }
                            }
                        }
                    },
                    Err(e) => {
                        warn!("Failed to open text embeddings file: {}", e);
                        None
                    }
                }
            } else {
                debug!("No bundled embeddings found. Looked for glove.50d.fifu or glove.6B.50d.txt");
                None
            }
        } else {
            None
        };

        // Initialize LRU cache with capacity 10,000
        let cache = LruCache::new(NonZeroUsize::new(10000).unwrap());

        // Initialize POS tagger
        let pos_tagger = if let Some(dir) = data_dir {
            let weights_path = dir.join("tagger/weights.json");
            let classes_path = dir.join("tagger/classes.txt");
            let tags_path = dir.join("tagger/tags.json");
            
            if weights_path.exists() && classes_path.exists() {
                // Paths must be strings for the library
                let w_str = weights_path.to_str().unwrap_or_default();
                let c_str = classes_path.to_str().unwrap_or_default();
                let t_str = tags_path.to_str().unwrap_or_default();
                
                debug!("Loading POS tagger from {:?}", dir.join("tagger"));
                Some(Arc::new(PerceptronTagger::new(w_str, c_str, t_str)))
            } else {
                warn!("POS tagger files not found in {:?}", dir.join("tagger"));
                None
            }
        } else {
            None
        };

        Self { 
            embeddings,
            wordnet_cache: Arc::new(Mutex::new(cache)),
            pos_tagger,
        }
    }

    /// Expand cues using WordNet with Context-Aware Semantic Ranking
    /// If embeddings are available, we score synonyms by similarity to the content's context vector.
    /// This acts as Word Sense Disambiguation (WSD).
    pub fn expand_wordnet(&self, content: &str, known_cues: &[String], threshold: f32, limit: usize) -> Vec<String> {
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

        // 2. Check Cache (skip context check for cache hits to save time? 
        // No, context sensitive WSD implies the same word expands differently in different contexts.
        // But our cache is `word -> synonyms`. It's context-free.
        // TRADEOFF: We cache the "most common" expansion or we disable caching for WSD?
        // IF we really want WSD, `word` -> `synonyms` cache is invalid because "Coke" -> "Soda" in one, "Coal" in another.
        // FOR NOW: We will DISABLE the simple global cache for WSD to ensure quality, 
        // OR we use the context vector to filter the cached superset.
        // Let's rely on the fast embedding lookup and skip the cache for the expansion logic itself,
        // or just cache the raw synonyms from `thesaurus` and do the ranking live.
        
        // Let's cache the RAW `thesaurus` lookup failure/success, but ranking must be dynamic.
        // Actually, `thesaurus::synonyms` gives the same list every time. We can cache that.
        // Then we filter/rank.
        
        let context_vec = self.get_context_vector(content);

        // 3. Parallel Processing
        let results: Vec<String> = words_to_lookup
            .into_par_iter()
            .flat_map(|word| {
                // Check cache first
                let cached = {
                    let cache = self.wordnet_cache.lock().unwrap();
                    cache.peek(&word).cloned()
                };
                
                let raw_syns = if let Some(syns) = cached {
                    syns
                } else {
                    // Cache miss - get from thesaurus
                    let syns = synonyms(&word);
                    // Update cache
                    {
                        let mut cache = self.wordnet_cache.lock().unwrap();
                        cache.put(word.clone(), syns.clone());
                    }
                    syns
                };
                
                if !raw_syns.is_empty() {
                     debug!("Synonyms for '{}': {:?}", word, raw_syns);
                }
                
                if raw_syns.is_empty() {
                    return Vec::new();
                }

                if let Some(ref ctx) = context_vec {
                     // WSD MODE: Rank by similarity to context
                     if let Some(emb_store) = &self.embeddings {
                         let mut ranked: Vec<(String, f32)> = Vec::new();
                         for syn in raw_syns {
                             if syn.len() <= 2 || syn == word { continue; }
                             
                             if let Some(chem) = emb_store.embedding(&syn) {
                                 let sim = chem.view().dot(ctx); // Dot product as similarity score
                                 if sim > threshold {
                                    debug!("Found good match for '{}': {}", syn, sim);
                                     ranked.push((syn, sim));
                                 }
                             } else {
                                 // If unknown to GloVe, maybe keep it but penalize? 
                                 // Or discard to be safe? Discarding reduces noise.
                                 // Let's give it a low base score so known-good matches win.
                                 // update: we are not using this for now.
                                 //ranked.push((syn, -1.0));
                             }
                         }
                         // Sort descending
                         ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                         return ranked.into_iter()
                             .filter(|(syn, _)| !syn.contains(' '))  // Reject multi-word phrases
                             .take(limit)  // Top N synonyms only
                             .map(|(w, _)| w)
                             .collect();
                     }
                 }

                // Fallback / Naive Mode (No embeddings) DISABLED FOR NOW: 5/1/26
                // We need to clone word to use it in filter if we are in fallback
                //raw_syns.into_iter()
                //    .filter(|s| s.len() > 2 && s != &word && !s.contains(' '))  // Reject multi-word
                //    .take(3)  // Top 3 synonyms only
                //    .collect()
                return raw_syns;
            })
            .collect();
            
        new_cues.extend(results);
        new_cues
    }


    /// Expand cues using GloVe embeddings (if available)
    pub fn expand_glove(&self, _content: &str, known_cues: &[String]) -> Vec<String> {
        let embeddings = match &self.embeddings {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut new_cues = Vec::new();

        for cue in known_cues {
            let word = if let Some((key, value)) = cue.split_once(':') {
                if key == "id" || key == "path" || key == "source" || key == "file" || key == "type" || key == "status" || key == "reason" {
                    continue;
                }
                value
            } else {
                cue.as_str()
            };
            
            if let Some(res) = embeddings.embedding(word) {
                    // Find 5 nearest neighbors
                    let neighbors = self.search(embeddings, res.view(), 5);
                    for neighbor in neighbors {
                        if neighbor != word && neighbor.len() > 2 {
                            // Emit flat cue
                            new_cues.push(neighbor);
                        }
                    }
            }
        }

        new_cues
    }

    fn search(&self, embeddings: &Embeddings<VocabWrap, StorageWrap>, target: ArrayView1<f32>, k: usize) -> Vec<String> {
        let vocab = embeddings.vocab();
        let storage = embeddings.storage();
        
        let mut similarities = Vec::with_capacity(vocab.words_len());
        
        // Compute cosine similarity (assuming vectors are roughly normalized or just dot product for ranking)
        // For GloVe, we should normalize.
        let target_norm = target.dot(&target).sqrt();
        if target_norm < 1e-6 {
            return Vec::new();
        }

        for (i, word) in vocab.words().iter().enumerate() {
            let vec = storage.embedding(i);
            let dot = target.dot(&vec);
            let norm = vec.dot(&vec).sqrt();
            if norm > 1e-6 {
                let sim = dot / (target_norm * norm);
                similarities.push((word, sim));
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        similarities.into_iter()
            .filter(|(_, sim)| *sim >= 0.60)
            .filter(|(w, _)| !crate::nl::get_stopwords().contains(w.as_str()))
            .take(k)
            .map(|(w, _)| w.to_string())
            .collect()
    }

    /// Compute the context vector (mean of all token embeddings in content)
    pub fn get_context_vector(&self, content: &str) -> Option<Array1<f32>> {
        let embeddings = self.embeddings.as_ref()?;
        let tokens = crate::nl::tokenize_to_cues(content); // Returns flat tokens now
        
        let mut sum_vec: Option<Array1<f32>> = None;
        let mut count = 0;
        
        for token in tokens {
            if let Some(emb) = embeddings.embedding(&token) {
                 if let Some(ref mut sum) = sum_vec {
                     *sum = &*sum + &emb.view();
                 } else {
                     sum_vec = Some(emb.to_owned());
                 }
                 count += 1;
            }
        }
        
        if count > 0 {
            sum_vec.map(|v| v / (count as f32))
        } else {
            None
        }
    }

    /// Expand cues based on the global context of the content
    /// Finds neighbors to the mean context vector
    pub fn expand_global_context(&self, content: &str) -> Vec<String> {
        let embeddings = match &self.embeddings {
            Some(e) => e,
            None => return Vec::new(),
        };
        
        if let Some(context_vec) = self.get_context_vector(content) {
            // Find neighbors to the context vector
            // We use a prefix "related:" to distinguish, or flat if user prefers?
            // User said: "NO MORE cues in the format of CONTEXT:CUE"
            // So we emit flat cues.
            let neighbors = self.search(embeddings, context_vec.view(), 5);
            
            // Filter out tokens that are already effectively in the content to avoid redundancy?
            // Or just emit them. The dedup logic downstream handles duplicates.
            neighbors
        } else {
            Vec::new()
        }
    }
    /// Check similarity between a word and a vector
    pub fn check_similarity(&self, word: &str, target: ArrayView1<f32>) -> Option<f32> {
        let embeddings = self.embeddings.as_ref()?;
        
        let word_clean = if let Some((_, val)) = word.split_once(':') {
            val
        } else {
            word
        };

        if let Some(vec) = embeddings.embedding(word_clean) {
            let target_norm = target.dot(&target).sqrt();
            let vec_norm = vec.dot(&vec).sqrt();
            
            if target_norm < 1e-6 || vec_norm < 1e-6 {
                return Some(0.0);
            }
            
            let dot = target.dot(&vec.view());
            Some(dot / (target_norm * vec_norm))
        } else {
            None
        }
    }
}
