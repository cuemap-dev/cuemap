use cuemap_rust::engine::CueMapEngine;
use super::{GoldenTrace, NormalizedRecall};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub trait Eval {
    // Setup returns the engine state (memories) for this eval
    fn setup(&self) -> CueMapEngine;
    fn run(&self, engine: &CueMapEngine) -> EvalResult;
}

#[derive(Debug, Serialize, Deserialize)]
pub enum EvalResult {
    Pass,
    Fail(String),
    Error(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalReport {
    pub eval_name: String,
    pub timestamp: u64,
    pub result: EvalResult,
    pub snapshot_path: Option<String>,
}

pub struct EvalRunner {
    pub base_snapshot_path: Option<String>,
    pub reports_dir: PathBuf,
}

impl EvalRunner {
    pub fn new(base_snapshot_path: Option<String>, reports_dir: PathBuf) -> Self {
        if !reports_dir.exists() {
            let _ = fs::create_dir_all(&reports_dir);
        }
        Self {
            base_snapshot_path,
            reports_dir,
        }
    }

    pub fn save_report(&self, eval_name: &str, result: EvalResult) -> Result<PathBuf, String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let report = EvalReport {
            eval_name: eval_name.to_string(),
            timestamp,
            result,
            snapshot_path: self.base_snapshot_path.clone(),
        };
        
        let filename = format!("{}_{}.json", eval_name, timestamp);
        let path = self.reports_dir.join(filename);
        
        let data = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        fs::write(&path, data).map_err(|e| e.to_string())?;
        
        Ok(path)
    }

    pub fn assert_trace(
        &self,
        actual: &NormalizedRecall,
        golden: &GoldenTrace,
        epsilon: f64,
    ) -> Result<(), String> {
        // 1. Check ordering of memory_id (Top N)
        // Golden trace results count defines N
        let top_n = golden.results.len();

        for (i, golden_result) in golden.results.iter().enumerate() {
            if i >= actual.ordered_ids.len() {
                return Err(format!(
                    "Actual results length {} is less than golden length {}",
                    actual.ordered_ids.len(),
                    top_n
                ));
            }

            let actual_id = &actual.ordered_ids[i];
            
            // Allow exact match or if memory_id is "file:test.py:calculate_sum", maybe partial?
            // Spec says "Same ordering of memory_id"
            if actual_id != &golden_result.memory_id {
                 return Err(format!(
                    "Rank {}: Expected ID {}, got {}",
                    i + 1,
                    golden_result.memory_id,
                    actual_id
                ));
            }

            // 2. Check scores with epsilon
            let actual_score = actual.scores[i];
            if (actual_score - golden_result.score).abs() > epsilon {
                 return Err(format!(
                    "Rank {}: Score mismatch. Expected {}, got {} (diff {})",
                    i + 1,
                    golden_result.score,
                    actual_score,
                    (actual_score - golden_result.score).abs()
                ));
            }

            // 3. Check intersection counts (exact)
            let actual_intersection = actual.intersection_counts[i];
            if actual_intersection != golden_result.intersection_count {
                 return Err(format!(
                    "Rank {}: Intersection count mismatch. Expected {}, got {}",
                    i + 1,
                    golden_result.intersection_count,
                    actual_intersection
                ));
            }
        }
        
        Ok(())
    }
}
