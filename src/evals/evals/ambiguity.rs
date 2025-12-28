use crate::engine::CueMapEngine;
use crate::evals::runner::{Eval, EvalResult};

pub struct AmbiguityRecognitionEval {
    pub query: Vec<String>,
    pub ambiguity_epsilon: f64,
    pub min_candidates: usize,
}

impl Eval for AmbiguityRecognitionEval {
    fn setup(&self) -> CueMapEngine {
        CueMapEngine::new()
    }

    fn run(&self, engine: &CueMapEngine) -> EvalResult {
        let results = engine.recall(self.query.clone(), 5, false);

        if results.len() < self.min_candidates {
            return EvalResult::Fail(format!(
                "Not enough candidates found for ambiguity check: {} < {}",
                results.len(), self.min_candidates
            ));
        }

        let first = &results[0];
        let second = &results[1];

        let score_diff = (first.score - second.score).abs();

        if score_diff >= self.ambiguity_epsilon {
            return EvalResult::Fail(format!(
                "Results are too distinct for ambiguity: |{} - {}| = {} >= {}",
                first.score, second.score, score_diff, self.ambiguity_epsilon
            ));
        }

        EvalResult::Pass
    }
}

pub struct UnanswerableQuestionEval {
    pub query: Vec<String>,
    pub low_integrity_threshold: f64,
}

impl Eval for UnanswerableQuestionEval {
    fn setup(&self) -> CueMapEngine {
        CueMapEngine::new()
    }

    fn run(&self, engine: &CueMapEngine) -> EvalResult {
        let results = engine.recall(self.query.clone(), 5, false);

        if results.is_empty() {
            return EvalResult::Pass;
        }

        // checking the match_integrity field in RecallResult
        let top_result = &results[0];
        if top_result.match_integrity >= self.low_integrity_threshold {
            return EvalResult::Fail(format!(
                "Unanswerable query returned high match integrity result: {} >= {}",
                top_result.match_integrity, self.low_integrity_threshold
            ));
        }

        // Also check if score is suspiciously high (optional, but good for robustness)
        // For now, match_integrity is the primary indicator defined in spec
        
        EvalResult::Pass
    }
}
