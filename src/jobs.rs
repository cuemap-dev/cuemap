use crate::multi_tenant::MultiTenantEngine;
use crate::projects::ProjectContext;
use crate::llm::{LlmConfig, propose_cues};
use crate::normalization::normalize_cue;
use crate::taxonomy::validate_cues;
use crate::config::*;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};
use std::collections::HashSet;
use rayon::prelude::*;
use smallvec::SmallVec;
use uuid::Uuid;

#[derive(Debug)]
pub enum Job {
    ProposeCues { project_id: String, memory_id: String, content: String },
    TrainLexiconFromMemory { project_id: String, memory_id: String },
    ProposeAliases { project_id: String },
    ExtractAndIngest { project_id: String, memory_id: String, content: String, file_path: String },
    VerifyFile { project_id: String, file_path: String, valid_memory_ids: Vec<String> },
    UpdateGraph { project_id: String, memory_id: String },
    ReinforceMemories { project_id: String, memory_ids: Vec<String>, cues: Vec<String> },
    ReinforceLexicon { project_id: String, memory_ids: Vec<String>, cues: Vec<String> },
}

pub struct JobQueue {
    sender: mpsc::Sender<Job>,
}

// Abstraction to access projects regardless of mode
pub trait ProjectProvider: Send + Sync + 'static {
    fn get_project(&self, project_id: &str) -> Option<Arc<ProjectContext>>;
}

impl ProjectProvider for MultiTenantEngine {
    fn get_project(&self, project_id: &str) -> Option<Arc<ProjectContext>> {
        self.get_project(&project_id.to_string())
    }
}

// Wrapper for single tenant
pub struct SingleTenantProvider {
    pub project: Arc<ProjectContext>,
}

impl ProjectProvider for SingleTenantProvider {
    fn get_project(&self, _project_id: &str) -> Option<Arc<ProjectContext>> {
        Some(self.project.clone())
    }
}

impl JobQueue {
    pub fn new(provider: Arc<dyn ProjectProvider>) -> Self {
        let (tx, mut rx) = mpsc::channel(1000);
        
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                process_job(job, &provider).await;
            }
        });
        
        Self { sender: tx }
    }
    
    pub async fn enqueue(&self, job: Job) {
        if let Err(e) = self.sender.send(job).await {
            warn!("Failed to enqueue job: {}", e);
        }
    }
}

struct CueCandidate {
    cue: String,
    len: usize,
    sample: HashSet<String>, // Hashed set for fast lookups in stage 1
}

// --- Helper Functions ---

/// Split cue into significant tokens
fn cue_tokens(cue: &str) -> SmallVec<[String; 8]> {
    let mut tokens = SmallVec::new();
    let parts = cue.split(|c| c == ':' || c == '-' || c == '_');
    
    for part in parts {
        let lower = part.to_lowercase();
        if lower.len() >= 3 {
            tokens.push(lower);
        }
    }
    tokens
}

/// Check if two cues share at least one significant token
fn lexical_gate(a: &str, b: &str) -> bool {
    // 1. Check if one contains the other (simple rewrite)
    if a.contains(b) || b.contains(a) {
        return true;
    }
    
    // 2. Token overlap
    let tokens_a = cue_tokens(a);
    if tokens_a.is_empty() { return false; }
    
    let tokens_b = cue_tokens(b);
    if tokens_b.is_empty() { return false; }
    
    for ta in &tokens_a {
        for tb in &tokens_b {
            if ta == tb {
                return true;
            }
        }
    }
    
    false
}

/// Check if cue is in canonical key:value format
fn is_canonical_format(cue: &str) -> bool {
    match cue.split_once(':') {
        Some((k, v)) => !k.is_empty() && !v.is_empty(),
        None => false,
    }
}

/// Deterministically choose (canonical, alias)
fn choose_canonical(a: &str, b: &str) -> (String, String) {
    let a_canon = is_canonical_format(a);
    let b_canon = is_canonical_format(b);
    
    if a_canon && !b_canon {
        (a.to_string(), b.to_string())
    } else if !a_canon && b_canon {
        (b.to_string(), a.to_string())
    } else {
        // Tie-breaker: lexicographical
        if a < b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        }
    }
}

/// Check if a cue is suitable for lexicon training (excluding high-cardinality cues)
pub fn is_lexicon_trainable(cue: &str) -> bool {
    let lower = cue.to_lowercase();
    !lower.starts_with("path:") && 
    !lower.starts_with("id:") && 
    !lower.starts_with("memory_id:") && 
    !lower.starts_with("file:") && 
    !lower.starts_with("alias_id:") &&
    !lower.starts_with("source:")
}

async fn process_job(job: Job, provider: &Arc<dyn ProjectProvider>) {
    match job {
        Job::TrainLexiconFromMemory { project_id, memory_id } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                // Fetch memory from main engine
                if let Some(memory) = ctx.main.get_memory(&memory_id) {
                    // Tokenize content
                    let tokens = crate::nl::tokenize_to_cues(&memory.content);
                    
                    if tokens.is_empty() {
                        return;
                    }
                    
                    let mut identity_count = 0;
                    let mut synonym_count = 0;
                    let mut sample_synonyms: Vec<String> = Vec::new();
                    
                    // REFACTOR: Avoid global N^2 association (pollution).
                    // Instead of associating ALL tokens with ALL cues, we:
                    // 1. Associate each token with ITSELF (Identity).
                    // 2. Associate each token with its DIRECT synonyms (WordNet).
                    // This ensures "favorite" predicts "dearie", but "favorite" does NOT predict "ample" (from "stuffed").
                    
                    for token in &tokens {
                        if !is_lexicon_trainable(&token) {
                            continue;
                        }

                        // 1. Train Identity: Token -> Token
                        let lex_id = format!("cue:{}", token);
                        ctx.lexicon.upsert_memory_with_id(
                            lex_id.clone(),
                            token.clone(),
                            vec![token.clone()], // Triggered by itself
                            None,
                            false
                        );
                        identity_count += 1;

                        // 2. Train Synonyms: Token -> Synonym
                        // We use the SemanticEngine to find what this token expands to.
                        // These are valid cues that should be triggered by this token.
                        let expanded = ctx.semantic_engine.expand_wordnet(&token, &[token.clone()], 0.65, 3);
                        
                        for synonym in expanded {
                            if !is_lexicon_trainable(&synonym) {
                                continue;
                            }
                            // Upsert: Synonym triggered by Token
                            let syn_id = format!("cue:{}", synonym);
                            ctx.lexicon.upsert_memory_with_id(
                                syn_id,
                                synonym.clone(),
                                vec![token.clone()],
                                None,
                                false
                            );
                            synonym_count += 1;
                            if sample_synonyms.len() < 5 {
                                sample_synonyms.push(format!("{}->{}", token, synonym));
                            }
                        }
                    }
                    
                    // Log summary
                    if identity_count > 0 || synonym_count > 0 {
                        let sample_str = if !sample_synonyms.is_empty() {
                            format!(" (e.g. {})", sample_synonyms.join(", "))
                        } else {
                            String::new()
                        };
                        info!("Job: Lexicon trained {} identity + {} synonym mappings for memory {}{}", 
                            identity_count, synonym_count, memory_id, sample_str);
                    }
                }
            }
        }
        Job::ProposeCues { project_id, memory_id, content } => {
             if let Some(ctx) = provider.get_project(&project_id) {
                 info!("Job: Proposing cues for memory {} in project {} (strategy: {:?})", memory_id, project_id, ctx.cuegen_strategy);
                 
                 // 1. Resolve known cues (Lexicon recall)
                 let (mut known_cues, _) = ctx.resolve_cues_from_text(&content);
                 
                 // 2. Bootstrap if needed (for static strategies to have something to expand)
                 // If we have very few cues, add raw tokens to allow expansion
                 if known_cues.len() < 3 {
                     let tokens = crate::nl::tokenize_to_cues(&content);
                     // Take top key phrases/tokens
                     for token in tokens.into_iter().take(10) {
                         if !known_cues.contains(&token) {
                             known_cues.push(token);
                         }
                     }
                 }
                 
                 // Track cues by source for detailed logging
                 let mut wordnet_cues: Vec<String> = Vec::new();
                 let mut glove_cues: Vec<String> = Vec::new();
                 let mut context_cues: Vec<String> = Vec::new();
                 let mut llm_cues: Vec<String> = Vec::new();
                 
                 // IDF Filtering: Identify expansion candidates (rare cues only)
                 let total = ctx.total_memories();
                 // Threshold: 10% of corpus, minimum 20 memories.
                 let threshold = (total as f64 * 0.1).max(20.0) as usize;

                 // PERF/QUALITY: Use raw tokens for expansion to avoid Lexicon Pollution loop.
                 // We only expand what is explicitly in the content.
                 // Filter by IDF to skip common words (e.g. "the").
                 let tokens = crate::nl::tokenize_to_cues(&content);
                 let expansion_candidates: Vec<String> = tokens.iter()
                     .filter(|c| ctx.get_cue_frequency(c) <= threshold)
                     .cloned()
                     .collect();
                 
                 
                 // 3. Static Semantic Expansion (Always on - WordNet)
                 let wn_result = ctx.semantic_engine.expand_wordnet(&content, &expansion_candidates, 0.65, 3);
                 wordnet_cues.extend(wn_result);
                 
                // 4. Strategy Specific Expansion
                match ctx.cuegen_strategy {
                    CueGenStrategy::Default => {
                        // Minimal strategy: Only WordNet (handled below always-on)
                        // No extra expansion.
                    },
                    CueGenStrategy::Glove => {
                        // GloVe Expansion (Nearest Neighbors of Cues)
                        let glove_result = ctx.semantic_engine.expand_glove(&content, &expansion_candidates);
                        glove_cues.extend(glove_result);
                        
                        // Global Context Expansion (Nearest Neighbors of Context Vector)
                        let context_result = ctx.semantic_engine.expand_global_context(&content);
                        context_cues.extend(context_result);
                    },
                     CueGenStrategy::Ollama | CueGenStrategy::Openai | CueGenStrategy::Google => {
                         // LLM Expansion
                         if let Some(config) = LlmConfig::from_strategy(&ctx.cuegen_strategy) {
                             match propose_cues(&content, &config, &known_cues).await {
                                 Ok(result) => llm_cues.extend(result),
                                 Err(e) => error!("Job: LLM failed: {}", e),
                             }
                         }
                     }
                 }
                 
                 // Log source breakdown before normalization
                 let log_sample = |name: &str, cues: &[String]| {
                     if !cues.is_empty() {
                         let sample: Vec<_> = cues.iter().take(5).collect();
                         let suffix = if cues.len() > 5 { format!(" (+{} more)", cues.len() - 5) } else { String::new() };
                         info!("  └─ {}: {:?}{}", name, sample, suffix);
                     }
                 };
                 
                 log_sample("WordNet", &wordnet_cues);
                 log_sample("GloVe", &glove_cues);
                 log_sample("Context", &context_cues);
                 log_sample("LLM", &llm_cues);
                 

                 
                 // Merge all proposed cues with deduplication and filtering
                 let mut seen = HashSet::new();
                 let mut proposed_cues = Vec::new();
                 
                 let filter_and_add = |cues: Vec<String>, seen: &mut HashSet<String>, out: &mut Vec<String>| {
                     for cue in cues {
                         let lower = cue.to_lowercase();

                         // Skip very short cues
                         if lower.len() < 3 {
                             continue;
                         }
                         // Skip duplicates
                         if seen.contains(&lower) {
                             continue;
                         }
                         seen.insert(lower);
                         out.push(cue);
                     }
                 };
                 
                 filter_and_add(wordnet_cues, &mut seen, &mut proposed_cues);
                 filter_and_add(glove_cues, &mut seen, &mut proposed_cues);
                 filter_and_add(context_cues, &mut seen, &mut proposed_cues);
                 filter_and_add(llm_cues, &mut seen, &mut proposed_cues);
                 
                 // Cap total proposed cues to prevent explosion
                 const MAX_PROPOSED_CUES: usize = 10;
                 if proposed_cues.len() > MAX_PROPOSED_CUES {
                     proposed_cues.truncate(MAX_PROPOSED_CUES);
                     debug!("Job: Truncated proposed cues to {}", MAX_PROPOSED_CUES);
                 }
                 
                 // 5. Merge, Normalize & Validate
                 let mut normalized_cues = Vec::new();
                 for cue in proposed_cues {
                     let (normalized, _) = normalize_cue(&cue, &ctx.normalization);
                     normalized_cues.push(normalized);
                 }
                 
                 let report = validate_cues(normalized_cues, &ctx.taxonomy);
                 
                 // 6. Attach accepted cues
                 if !report.accepted.is_empty() {
                     ctx.main.attach_cues(&memory_id, report.accepted.clone());
                     let sample: Vec<_> = report.accepted.iter().take(8).collect();
                     let suffix = if report.accepted.len() > 8 { format!(" (+{} more)", report.accepted.len() - 8) } else { String::new() };
                     info!("Job: Attached {} cues to memory {}: {:?}{}", report.accepted.len(), memory_id, sample, suffix);
                     
                     // 7. Retrain lexicon with new cues
                     let tokens = crate::nl::tokenize_to_cues(&content);
                     if !tokens.is_empty() {
                         for canonical_cue in report.accepted {
                             if !is_lexicon_trainable(&canonical_cue) {
                                 continue;
                             }
                             
                              let lex_id = format!("cue:{}", canonical_cue);
                              
                              // Filter out identity mappings
                              let filtered_tokens: Vec<String> = tokens.iter()
                                  .filter(|t| t.as_str() != canonical_cue.as_str() && !canonical_cue.contains(t.as_str()))
                                  .cloned()
                                  .collect();
                                  
                              if filtered_tokens.is_empty() {
                                  continue;
                              }

                              ctx.lexicon.upsert_memory_with_id(
                                  lex_id, 
                                  canonical_cue, 
                                  filtered_tokens, 
                                  None,
                                  false
                              );
                         }
                     }
                 }
             }
        }
        Job::ProposeAliases { project_id } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                let cue_index = ctx.main.get_cue_index();
                
                // 1. Filter and Select Mid-Frequency Cues
                let mut stats: Vec<(String, usize)> = cue_index
                    .iter()
                    .map(|entry| (entry.key().clone(), entry.value().len()))
                    .filter(|(k, cnt)| k.len() >= 3 && *cnt >= ALIAS_MIN_CUE_MEMORIES && *cnt <= ALIAS_MAX_CUE_MEMORIES)
                    .collect();
                
                stats.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                let drop_count = (stats.len() as f64 * 0.01) as usize;
                let stats = stats.into_iter().skip(drop_count).take(ALIAS_MAX_CANDIDATES).collect::<Vec<_>>();
                
                if stats.is_empty() {
                    return;
                }
                
                // 2. Build Candidates
                let candidates: Vec<CueCandidate> = stats
                    .into_iter()
                    .filter_map(|(key, len)| {
                        if let Some(entry) = cue_index.get(&key) {
                            let sample_vec = entry.get_recent_owned(Some(ALIAS_SAMPLE_SIZE));
                            let sample_set: HashSet<String> = sample_vec.into_iter().collect();
                            Some(CueCandidate {
                                cue: key,
                                len,
                                sample: sample_set,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                
                info!("Job: Analyzing {} candidates for aliases in project {}", candidates.len(), project_id);
                
                // 3. Parallel Comparison
                let proposals: Vec<(String, String, f64, String)> = candidates
                    .par_iter()
                    .enumerate()
                    .fold(Vec::new, |mut acc, (i, cand_a)| {
                        for cand_b in candidates.iter().skip(i + 1) {
                            let diff = (cand_a.len as isize - cand_b.len as isize).abs();
                            let max_len = std::cmp::max(cand_a.len, cand_b.len);
                            if (diff as f64 / max_len as f64) > ALIAS_SIZE_SIMILARITY_MAX_RATIO {
                                continue;
                            }
                            
                            if !lexical_gate(&cand_a.cue, &cand_b.cue) {
                                continue;
                            }
                            
                            let intersection = cand_a.sample.intersection(&cand_b.sample).count();
                            let min_sample_len = std::cmp::min(cand_a.sample.len(), cand_b.sample.len());
                            if min_sample_len == 0 { continue; }
                            
                            let sample_score = intersection as f64 / min_sample_len as f64;
                            if sample_score < (ALIAS_OVERLAP_THRESHOLD - 0.15) {
                                continue;
                            }
                            
                            if let Some(entry_a) = cue_index.get(&cand_a.cue) {
                                if let Some(entry_b) = cue_index.get(&cand_b.cue) {
                                    let (smaller, larger) = if entry_a.len() < entry_b.len() {
                                        (&entry_a.items, &entry_b.items)
                                    } else {
                                        (&entry_b.items, &entry_a.items)
                                    };
                                    
                                    let exact_intersection = smaller.iter().filter(|id| larger.contains(*id)).count();
                                    let min_len = smaller.len();
                                    if min_len == 0 { continue; }
                                    
                                    let exact_score = exact_intersection as f64 / min_len as f64;
                                    
                                    if exact_score >= ALIAS_OVERLAP_THRESHOLD {
                                        let (canon, alias) = choose_canonical(&cand_a.cue, &cand_b.cue);
                                        let alias_id_str = format!("{}->{}", alias, canon);
                                        let alias_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, alias_id_str.as_bytes());
                                        acc.push((alias, canon, exact_score, alias_uuid.to_string()));
                                    }
                                }
                            }
                        }
                        acc
                    })
                    .reduce(Vec::new, |mut a, b| { a.extend(b); a });
                
                // 4. Register Proposals
                for (from, to, score, alias_id) in proposals {
                    let id_cue = format!("alias_id:{}", alias_id);
                    if !ctx.aliases.get_cue_index().contains_key(&id_cue) {
                        let content = serde_json::json!({
                            "from": from,
                            "to": to,
                            "downweight": score,
                            "status": "proposed",
                            "reason": "overlap_analysis"
                        }).to_string();
                        
                        let cues = vec![
                            "type:alias".to_string(),
                            format!("from:{}", from),
                            format!("to:{}", to),
                            "status:proposed".to_string(),
                            "reason:overlap_analysis".to_string(),
                            id_cue
                        ];
                        
                        ctx.aliases.upsert_memory_with_id(alias_id.clone(), content, cues, None, false);
                        info!("Job: Proposed alias {} -> {} (score: {:.2})", from, to, score);
                    }
                }
            }
        }
        Job::ExtractAndIngest { project_id, memory_id, content, file_path } => {
             if let Some(config) = LlmConfig::from_env() {
                 debug!("Agent: Starting extraction for {}", memory_id);
                 match crate::llm::extract_facts(&content, &config).await {
                     Ok((extracted_content, cues)) => {
                         if let Some(ctx) = provider.get_project(&project_id) {
                              let mut final_cues = cues;
                              final_cues.push(format!("path:{}", file_path));
                              final_cues.push("source:agent".to_string());
                              
                              ctx.main.upsert_memory_with_id(
                                  memory_id.clone(),
                                  extracted_content.clone(),
                                  final_cues.clone(),
                                  None,
                                  false
                              );
                              
                              let tokens = crate::nl::tokenize_to_cues(&extracted_content);
                              for canonical_cue in &final_cues {
                                   if !is_lexicon_trainable(canonical_cue) {
                                       continue;
                                   }
                                   
                                   let lex_id = format!("cue:{}", canonical_cue);
                                   ctx.lexicon.upsert_memory_with_id(
                                       lex_id,
                                       canonical_cue.clone(),
                                       tokens.clone(),
                                       None,
                                       false
                                   );
                              }
                              
                              info!("Agent: Ingested memory {} ({} cues)", memory_id, final_cues.len());
                         }
                     }
                     Err(e) => error!("Agent: Extraction failed for {}: {}", memory_id, e),
                 }
             }
        }
        Job::VerifyFile { project_id, file_path, valid_memory_ids } => {
             if let Some(ctx) = provider.get_project(&project_id) {
                  // Strategy:
                  // 1. Look up all memories associated with "path:{file_path}"
                  // 2. Filter for those that are NOT in valid_memory_ids
                  // 3. Delete them
                  
                  let path_cue = format!("path:{}", file_path);
                  if let Some(ordered_set) = ctx.main.get_cue_index().get(&path_cue) {
                      // Get all memory IDs associated with this file
                      let current_memories = ordered_set.get_recent_owned(None);
                      let valid_set: HashSet<String> = valid_memory_ids.into_iter().collect();
                      
                      let mut deleted_count = 0;
                      for mem_id in current_memories {
                          // Only delete if it's an agent-managed memory (check prefix "file:")
                          // and not in the valid set.
                          if mem_id.starts_with("file:") && !valid_set.contains(&mem_id) {
                               if ctx.main.delete_memory(&mem_id) {
                                   deleted_count += 1;
                               }
                          }
                      }
                      
                      if deleted_count > 0 {
                          info!("Agent: Verified {}. Pruned {} stale memories.", file_path, deleted_count);
                      } else {
                          debug!("Agent: Verified {}. No stale memories found.", file_path);
                      }
                  }
             }
        }
        Job::UpdateGraph { project_id, memory_id } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                if let Some(memory) = ctx.main.get_memories().get(&memory_id) {
                    let cues = memory.cues.clone();
                    // Async update of the co-occurrence matrix
                    ctx.main.update_cue_co_occurrence(&cues);
                    debug!("Job: Updated graph connectivity for {} cues (memory: {})", cues.len(), memory_id);
                }
            }
        }
        Job::ReinforceMemories { project_id, memory_ids, cues } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                for memory_id in &memory_ids {
                    ctx.main.reinforce_memory(memory_id, cues.clone());
                }
                info!("Job: Reinforced {} memories with {} cues", memory_ids.len(), cues.len());
            }
        }
        Job::ReinforceLexicon { project_id, memory_ids, cues } => {
            if let Some(ctx) = provider.get_project(&project_id) {
                for memory_id in &memory_ids {
                    ctx.lexicon.reinforce_memory(memory_id, cues.clone());
                }
                info!("Job: Reinforced {} lexicon entries with {} cues", memory_ids.len(), cues.len());
            }
        }
    }
}

