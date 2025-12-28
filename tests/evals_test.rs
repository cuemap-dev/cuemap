use cuemap_rust::evals::runner::{Eval, EvalResult};
use cuemap_rust::evals::evals::recall_correctness::{ParaphraseInvarianceEval, SpecificitySensitivityEval, NegativeKnowledgeEval};
use cuemap_rust::evals::evals::determinism::DeterministicReplayEval;
use cuemap_rust::evals::evals::dynamics::ReinforcementEffectEval;
use cuemap_rust::evals::evals::ambiguity::{AmbiguityRecognitionEval, UnanswerableQuestionEval};

#[test]
fn test_paraphrase_invariance() {
    let eval = ParaphraseInvarianceEval {
        query1: vec!["calculator".to_string()],
        query2: vec!["calculate".to_string()], // Simple variation
        epsilon: 0.1,
    };
    
    // Setup engine with mock data
    let engine = eval.setup();
    engine.add_memory("Calculator function".to_string(), vec!["calculator".to_string(), "calculate".to_string()], None, false);
    engine.add_memory("Other function".to_string(), vec!["other".to_string()], None, false);
    
    let result = eval.run(&engine);
    match result {
        EvalResult::Pass => assert!(true),
        EvalResult::Fail(msg) => panic!("Eval failed: {}", msg),
        EvalResult::Error(e) => panic!("Eval error: {}", e),
    }
}

#[test]
fn test_specificity_sensitivity() {
    let eval = SpecificitySensitivityEval {
        general_query: vec!["calculator".to_string()],
        specific_query: vec!["calculate_sum".to_string()],
        general_target_id: "calc".to_string(),
        specific_target_id: "sum".to_string(),
    };
    
    let engine = eval.setup();
    // General memory
    engine.upsert_memory_with_id("calc".to_string(), "Calculator".to_string(), vec!["calculator".to_string()], None, false);
    // Specific memory
    engine.upsert_memory_with_id("sum".to_string(), "Calculate Sum".to_string(), vec!["calculate_sum".to_string(), "calculator".to_string()], None, false);
    
    let result = eval.run(&engine);
    match result {
        EvalResult::Pass => assert!(true),
        EvalResult::Fail(msg) => panic!("Eval failed: {}", msg),
        EvalResult::Error(e) => panic!("Eval error: {}", e),
    }
}

#[test]
fn test_negative_knowledge() {
    let eval = NegativeKnowledgeEval {
        query: vec!["banana".to_string()],
        match_integrity_threshold: 0.5,
    };
    
    let engine = eval.setup();
    engine.add_memory("Calculator".to_string(), vec!["calculator".to_string()], None, false);
    
    let result = eval.run(&engine);
    match result {
        EvalResult::Pass => assert!(true),
        EvalResult::Fail(msg) => panic!("Eval failed: {}", msg),
        EvalResult::Error(e) => panic!("Eval error: {}", e),
    }
}

#[test]
fn test_determinism() {
    let eval = DeterministicReplayEval {
        query: vec!["calculator".to_string()],
        interactions: 5,
    };
    
    let engine = eval.setup();
    engine.add_memory("Calculator".to_string(), vec!["calculator".to_string()], None, false);
    
    let result = eval.run(&engine);
    match result {
        EvalResult::Pass => assert!(true),
        EvalResult::Fail(msg) => panic!("Eval failed: {}", msg),
        EvalResult::Error(e) => panic!("Eval error: {}", e),
    }
}

#[test]
fn test_reinforcement() {
    let eval = ReinforcementEffectEval {
        query: vec!["calculator".to_string()],
        iterations: 5,
    };
    
    let engine = eval.setup();
    engine.add_memory("Calculator".to_string(), vec!["calculator".to_string()], None, false);
    
    let result = eval.run(&engine);
    match result {
        EvalResult::Pass => assert!(true),
        EvalResult::Fail(msg) => panic!("Eval failed: {}", msg),
        EvalResult::Error(e) => panic!("Eval error: {}", e),
    }
}

#[test]
fn test_ambiguity_recognition() {
    let eval = AmbiguityRecognitionEval {
        query: vec!["topic_a".to_string(), "topic_b".to_string()],
        ambiguity_epsilon: 1.0, 
        min_candidates: 2,
    };
    
    let engine = eval.setup();
    // Use disjoint cues to neutralize recency bias
    engine.upsert_memory_with_id("A".to_string(), "Content A".to_string(), vec!["topic_a".to_string()], None, false);
    engine.upsert_memory_with_id("B".to_string(), "Content B".to_string(), vec!["topic_b".to_string()], None, false);
    
    let result = eval.run(&engine);
    match result {
        EvalResult::Pass => assert!(true),
        EvalResult::Fail(msg) => panic!("Eval failed: {}", msg),
        EvalResult::Error(e) => panic!("Eval error: {}", e),
    }
}

#[test]
fn test_unanswerable_question() {
    let eval = UnanswerableQuestionEval {
        query: vec!["why_is_sky_blue".to_string()],
        low_integrity_threshold: 0.3,
    };
    
    let engine = eval.setup();
    engine.add_memory("Calculator".to_string(), vec!["calculator".to_string()], None, false);
    
    let result = eval.run(&engine);
    match result {
        EvalResult::Pass => assert!(true),
        EvalResult::Fail(msg) => panic!("Eval failed: {}", msg),
        EvalResult::Error(e) => panic!("Eval error: {}", e),
    }
}
