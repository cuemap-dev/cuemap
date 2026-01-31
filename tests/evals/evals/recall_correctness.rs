use cuemap_rust::engine::CueMapEngine;
use super::super::runner::{Eval, EvalResult};

pub struct ParaphraseInvarianceEval {
    pub query1: Vec<String>,
    pub query2: Vec<String>,
    pub epsilon: f64,
}

impl Eval for ParaphraseInvarianceEval {
    fn setup(&self) -> CueMapEngine {
        // In a real scenario, this would load from a snapshot file.
        // For now, we return a new engine or populated one if we have helpers.
        // We'll assume the runner handles snapshot loading or we pass it in.
        // But the trait says setup returns Engine.
        CueMapEngine::new()
    }

    fn run(&self, engine: &CueMapEngine) -> EvalResult {
        let res1 = engine.recall(self.query1.clone(), 5, false, None);
        let res2 = engine.recall(self.query2.clone(), 5, false, None);

        if res1.is_empty() || res2.is_empty() {
             return EvalResult::Fail("One or both queries returned empty results".to_string());
        }

        let top1 = &res1[0];
        let top2 = &res2[0];

        if top1.memory_id != top2.memory_id {
            return EvalResult::Fail(format!(
                "Top result mismatch: {} vs {}",
                top1.memory_id, top2.memory_id
            ));
        }

        let score_diff = (top1.score - top2.score).abs();
        if score_diff > self.epsilon {
            return EvalResult::Fail(format!(
                "Score difference too high: {} > {}",
                score_diff, self.epsilon
            ));
        }

        EvalResult::Pass
    }
}

pub struct SpecificitySensitivityEval {
    pub general_query: Vec<String>,
    pub specific_query: Vec<String>,
    pub general_target_id: String,
    pub specific_target_id: String,
}

impl Eval for SpecificitySensitivityEval {
    fn setup(&self) -> CueMapEngine {
        CueMapEngine::new()
    }

    fn run(&self, engine: &CueMapEngine) -> EvalResult {
        // Q2: What does calculate_sum do?
        let specific_res = engine.recall(self.specific_query.clone(), 10, false, None);
        
        let score_specific_target = specific_res.iter().find(|r| r.memory_id == self.specific_target_id).map(|r| r.score).unwrap_or(0.0);
        let score_general_target = specific_res.iter().find(|r| r.memory_id == self.general_target_id).map(|r| r.score).unwrap_or(0.0);

        // Assert specific target scores higher on specific query
        if score_specific_target <= score_general_target {
             return EvalResult::Fail(format!(
                "Specificity check failed: Specific target {} score ({}) <= General target {} score ({})",
                self.specific_target_id, score_specific_target, self.general_target_id, score_general_target
            ));
        }

        // Q1: What does calculator do?
        let general_res = engine.recall(self.general_query.clone(), 10, false, None);
        let score_general_on_general = general_res.iter().find(|r| r.memory_id == self.general_target_id).map(|r| r.score).unwrap_or(0.0);
        
        if score_general_on_general <= 0.0 {
             return EvalResult::Fail("General target should have positive score on general query".to_string());
        }

        EvalResult::Pass
    }
}

pub struct NegativeKnowledgeEval {
    pub query: Vec<String>,
    pub match_integrity_threshold: f64,
}

impl Eval for NegativeKnowledgeEval {
    fn setup(&self) -> CueMapEngine {
        CueMapEngine::new()
    }

    fn run(&self, engine: &CueMapEngine) -> EvalResult {
        let results = engine.recall(self.query.clone(), 5, false, None);

        if let Some(top) = results.first() {
            if top.score >= self.match_integrity_threshold {
                 return EvalResult::Fail(format!(
                    "Negative knowledge failed: Top score {} >= threshold {}",
                    top.score, self.match_integrity_threshold
                ));
            }
            
            if top.intersection_count > 1 {
                 return EvalResult::Fail(format!(
                    "Negative knowledge failed: Intersection count {} > 1",
                    top.intersection_count
                ));
            }
        }

        EvalResult::Pass
    }
}
