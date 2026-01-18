use cuemap_rust::engine::CueMapEngine;
use super::super::runner::{Eval, EvalResult};

pub struct ReinforcementEffectEval {
    pub query: Vec<String>,
    pub iterations: usize,
}

impl Eval for ReinforcementEffectEval {
    fn setup(&self) -> CueMapEngine {
        CueMapEngine::new()
    }

    fn run(&self, engine: &CueMapEngine) -> EvalResult {
        // Baseline
        let init_res = engine.recall(self.query.clone(), 1, false);
        if init_res.is_empty() {
             return EvalResult::Fail("No initial results".to_string());
        }
        let init_score = init_res[0].reinforcement_score;
        let mem_id = init_res[0].memory_id.clone();

        // Reinforce
        for _ in 0..self.iterations {
            engine.reinforce_memory(&mem_id, self.query.clone());
        }

        // Check Effect
        let final_res = engine.recall(self.query.clone(), 1, false);
        if final_res.is_empty() {
            return EvalResult::Fail("Lost memory after reinforcement".to_string());
        }
        let final_score = final_res[0].reinforcement_score;

        if final_score <= init_score {
            return EvalResult::Fail(format!(
                "Reinforcement did not increase score: {} -> {}",
                init_score, final_score
            ));
        }

        EvalResult::Pass
    }
}
