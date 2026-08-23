    use super::*;
    use crate::config::TuningConfig;
    use crate::normalization::NormalizationConfig;
    use crate::persistence::{CloudBackupConfig, CloudBackupManager};
    use crate::projects::ProjectContext;
    use crate::structures::MainStats;
    use crate::taxonomy::Taxonomy;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tower::ServiceExt;

    #[test]
    fn source_event_time_prefers_explicit_and_reads_structured_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "source_timestamp".to_string(),
            serde_json::json!("2024-01-01T00:00:00.250Z"),
        );

        assert_eq!(source_event_time(Some(42.0), Some(&metadata)), Some(42.0));
        assert_eq!(
            source_event_time(None, Some(&metadata)),
            Some(1_704_067_200.25)
        );
    }

    #[test]
    fn api_helper_paths_cover_query_plan_cuebridge_and_parent_fusion() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "api-helper-paths".to_string(),
        );

        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), serde_json::json!("assistant"));
        metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("helper-session"),
        );
        metadata.insert(
            "source_timestamp".to_string(),
            serde_json::json!("2024-01-01T00:00:00Z"),
        );

        let first = ctx.main.add_memory(
            "First sentence. Shared detail.".to_string(),
            vec![
                "parent:helper-document".to_string(),
                "chunk_idx:0".to_string(),
                "source_role:assistant".to_string(),
                "has:list".to_string(),
                "has:number".to_string(),
            ],
            Some(metadata.clone()),
            MainStats::default(),
            true,
        );
        let second = ctx.main.add_memory(
            "Shared detail. Second sentence.".to_string(),
            vec![
                "parent:helper-document".to_string(),
                "chunk_idx:1".to_string(),
                "source_role:assistant".to_string(),
                "has:list".to_string(),
                "has:number".to_string(),
            ],
            Some(metadata),
            MainStats::default(),
            true,
        );

        let mut expanded = vec![("has:list".to_string(), 1.0), ("today".to_string(), 1.0)];
        let plan = apply_query_plan(
            &ctx,
            Some("Can you list the deployment steps today in order?"),
            Some("2024-01-02"),
            &mut expanded,
        )
        .expect("query plan should be compiled");
        assert!(plan.labels.iter().any(|label| label == "ordered_reconstruction"));
        assert!(plan.labels.iter().any(|label| label == "multi_evidence_collection"));
        assert!(expanded.iter().any(|(cue, _)| cue == "has:list"));

        let expansion = crate::cuebridge::CueBridgeGapExpansion {
            artifact: "helper-pack".to_string(),
            artifact_hash: "hash".to_string(),
            entry_id: "entry".to_string(),
            cue: "deployment".to_string(),
            weight: 1.2,
            confidence: 0.9,
            score: 0.8,
        };
        let mut existing = recall_result(0.5, 2);
        existing.memory_id = first;
        existing.score = 1.0;
        existing.explain = Some(serde_json::json!({}));
        let mut improved = recall_result(0.9, 3);
        improved.memory_id = first;
        improved.score = 2.0;
        let mut added = recall_result(0.8, 2);
        added.memory_id = second;
        merge_cuebridge_gap_results(
            &mut vec![existing],
            vec![improved, added],
            &[expansion],
            true,
        );

        let mut candidates = Vec::new();
        for (memory_id, score, chunk_idx) in [(first, 2.0, 0), (second, 1.5, 1)] {
            let mut result = recall_result(0.8, 3);
            result.memory_id = memory_id;
            result.score = score;
            result.metadata.insert("chunk_idx".to_string(), serde_json::json!(chunk_idx));
            candidates.push(result);
        }
        let fused = build_parent_fusion_results(&ctx, candidates, 2, true);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].metadata["parent_fusion"], true);
        assert!(fused[0].explain.is_some());

        let mut merged = vec![fused[0].clone()];
        let mut weaker = fused[0].clone();
        weaker.score -= 10.0;
        merge_parent_fusion_results(&mut merged, vec![weaker]);
        let mut stronger = fused[0].clone();
        stronger.score += 10.0;
        merge_parent_fusion_results(&mut merged, vec![stronger]);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn projection_helpers_cover_source_instruction_preference_and_decision_paths() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "api-projection-paths".to_string(),
        );

        let mut assistant_meta = HashMap::new();
        assistant_meta.insert("source_role".to_string(), serde_json::json!("assistant"));
        assistant_meta.insert(
            "source_session_id".to_string(),
            serde_json::json!("projection-session"),
        );
        assistant_meta.insert("source_turn_index".to_string(), serde_json::json!(2));
        let assistant_id = ctx.main.add_memory(
            "Dessert migration answer with probability details.".to_string(),
            vec![
                "source_role:assistant".to_string(),
                "type:answer".to_string(),
                "has:list".to_string(),
                "dessert".to_string(),
                "migration".to_string(),
                "probability".to_string(),
            ],
            Some(assistant_meta.clone()),
            MainStats::default(),
            true,
        );

        let mut user_meta = HashMap::new();
        user_meta.insert("source_role".to_string(), serde_json::json!("user"));
        user_meta.insert(
            "source_session_id".to_string(),
            serde_json::json!("projection-session"),
        );
        user_meta.insert("source_turn_index".to_string(), serde_json::json!(1));
        let user_id = ctx.main.add_memory(
            "User dessert migration preference.".to_string(),
            vec![
                "source_role:user".to_string(),
                "dessert".to_string(),
                "migration".to_string(),
            ],
            Some(user_meta),
            MainStats::default(),
            true,
        );

        let standing_id = ctx.main.add_memory(
            "Always use the migration probability threshold.".to_string(),
            vec![
                "type:standing_instruction".to_string(),
                "instruction:conditional".to_string(),
                "instruction:always".to_string(),
                "instruction_trigger:probability".to_string(),
                "probability".to_string(),
                "migration".to_string(),
            ],
            Some(assistant_meta.clone()),
            MainStats::default(),
            true,
        );
        let preference_id = ctx.main.add_memory(
            "I prefer dessert migration options.".to_string(),
            vec![
                "type:preference".to_string(),
                "preference:explicit".to_string(),
                "preference_value:dessert".to_string(),
                "preference_topic:dessert".to_string(),
                "preference_contrast:migration".to_string(),
                "dessert".to_string(),
                "migration".to_string(),
                "source_role:user".to_string(),
            ],
            Some(assistant_meta.clone()),
            MainStats::default(),
            true,
        );
        let decision_id = ctx.main.add_memory(
            "The naming decision selected migration.".to_string(),
            vec![
                "type:decision".to_string(),
                "type:selection".to_string(),
                "type:naming".to_string(),
                "migration".to_string(),
            ],
            Some(assistant_meta),
            MainStats::default(),
            true,
        );

        let plan = crate::facets::StructuralQueryPlan {
            labels: vec![
                "__semantic_facets_removed__".to_string(),
                "source_answer".to_string(),
                "source_assistant".to_string(),
                "personal_recommendation_context".to_string(),
                "naming_decision".to_string(),
            ],
            ..Default::default()
        };
        let mut assistant_result = recall_result(0.9, 3);
        assistant_result.memory_id = assistant_id;
        assistant_result.content = "Dessert migration answer with probability details.".to_string();
        assistant_result.created_at = 2.0;
        assistant_result.metadata = [
            ("source_role".to_string(), serde_json::json!("assistant")),
            ("source_session_id".to_string(), serde_json::json!("projection-session")),
        ]
        .into_iter()
        .collect();
        let mut user_result = recall_result(0.6, 5);
        user_result.memory_id = user_id;
        user_result.content = "User dessert migration preference.".to_string();
        user_result.created_at = 1.0;
        user_result.metadata = [
            ("source_role".to_string(), serde_json::json!("user")),
            ("source_session_id".to_string(), serde_json::json!("projection-session")),
            ("user_context_projection".to_string(), serde_json::json!(true)),
        ]
        .into_iter()
        .collect();
        let all_results = vec![assistant_result.clone(), user_result.clone()];

        assert!(source_answer_projection_requested(Some(&plan), Some("what did the assistant answer")));
        assert!(!source_answer_projection_cues(&ctx, Some(&plan), Some("list the answer"), &all_results).is_empty());
        assert!(!source_prompt_projection_cues(
            &ctx,
            Some(&plan),
            Some("assistant answer about dessert migration"),
            &all_results,
        )
        .is_empty());
        assert!(user_context_projection_requested(
            Some(&crate::facets::StructuralQueryPlan {
                labels: vec!["__semantic_facets_removed__".to_string()],
                ..Default::default()
            }),
            Some("what advice about dessert migration should I use"),
        ));
        assert!(!user_context_projection_cues(
            &ctx,
            Some(&crate::facets::StructuralQueryPlan {
                labels: vec!["__semantic_facets_removed__".to_string(), "personal_recommendation_context".to_string()],
                ..Default::default()
            }),
            Some("what advice about dessert migration should I use"),
            &all_results,
        )
        .is_empty());

        let standing = standing_instruction_projection_cues(
            &ctx,
            Some(&plan),
            Some("what probability migration should we use"),
        );
        assert!(!standing.cues.is_empty());
        assert!(!standing.anchors.is_empty());
        let preference = preference_projection_cues(
            &ctx,
            Some(&plan),
            Some("which dessert migration do I prefer"),
        );
        assert!(!preference.cues.is_empty());
        assert!(!preference.anchors.is_empty());
        assert!(!decision_projection_cues(&ctx, Some(&plan), &all_results).is_empty());

        let mut projected = all_results.clone();
        let mut standing_result = recall_result(0.4, 3);
        standing_result.memory_id = standing_id;
        standing_result.intersection_count = 5;
        let mut preference_result = recall_result(0.4, 3);
        preference_result.memory_id = preference_id;
        preference_result.intersection_count = 5;
        let mut decision_result = recall_result(0.4, 3);
        decision_result.memory_id = decision_id;
        decision_result.intersection_count = 5;
        merge_source_answer_projection_results(&mut projected, vec![assistant_result.clone()]);
        merge_source_prompt_projection_results(
            &mut projected,
            vec![user_result.clone()],
            Some("assistant answer about dessert migration"),
        );
        merge_user_context_projection_results(&mut projected, vec![user_result.clone()]);
        merge_standing_instruction_projection_results(
            &ctx,
            &mut projected,
            vec![standing_result],
            &standing.anchors,
        );
        merge_preference_projection_results(
            &ctx,
            &mut projected,
            vec![preference_result],
            &preference.anchors,
        );
        merge_decision_projection_results(&mut projected, vec![decision_result]);
        apply_source_role_preference(&mut projected, Some(&plan));
        apply_source_answer_adjacency_preference(&mut projected, Some(&plan));
        apply_user_context_adjacency_preference(
            &mut projected,
            Some(&crate::facets::StructuralQueryPlan {
                labels: vec!["__semantic_facets_removed__".to_string()],
                ..Default::default()
            }),
            Some("what advice about dessert migration should I use"),
        );
        assert!(projected.iter().any(|result| result.metadata.contains_key("decision_projection")));
    }

    fn recall_result(
        match_integrity: f64,
        intersection_count: usize,
    ) -> crate::engine::RecallResult {
        crate::engine::RecallResult {
            memory_id: 1,
            content: "content".to_string(),
            score: 1.0,
            match_integrity,
            intersection_count,
            recency_score: 0.0,
            reinforcement_score: 0.0,
            salience_score: 0.0,
            created_at: 0.0,
            metadata: HashMap::new(),
            explain: None,
        }
    }

    #[test]
    fn parent_fusion_defaults_off_and_force_runs() {
        let results = vec![recall_result(0.95, 4)];

        assert!(!should_run_parent_fusion(
            &results,
            ParentFusionMode::Off,
            None,
            Some("summarize the key points"),
        ));
        assert!(should_run_parent_fusion(
            &results,
            ParentFusionMode::Force,
            None,
            Some("plain lookup"),
        ));
    }

    #[test]
    fn parent_fusion_auto_requires_synthesis_query() {
        assert!(!should_run_parent_fusion(
            &[recall_result(0.4, 1)],
            ParentFusionMode::Auto,
            None,
            Some("what is my favorite dessert"),
        ));
        assert!(should_run_parent_fusion(
            &[recall_result(0.4, 1)],
            ParentFusionMode::Auto,
            None,
            Some("summarize my language service progress in order"),
        ));
    }

    #[test]
    fn ordered_reconstruction_is_opt_in_and_intent_gated() {
        let mut intent = crate::facets::StructuralQueryPlan::default();
        assert!(!should_run_ordered_reconstruction(
            OrderedReconstructionMode::Off,
            Some(&intent)
        ));
        assert!(should_run_ordered_reconstruction(
            OrderedReconstructionMode::Force,
            None
        ));
        assert!(!should_run_ordered_reconstruction(
            OrderedReconstructionMode::Auto,
            Some(&intent)
        ));

        intent.labels.push("ordered_reconstruction".to_string());
        assert!(should_run_ordered_reconstruction(
            OrderedReconstructionMode::Auto,
            Some(&intent)
        ));

        intent.labels.clear();
        intent
            .labels
            .push("multi_evidence_collection".to_string());
        assert!(should_run_ordered_reconstruction(
            OrderedReconstructionMode::Auto,
            Some(&intent)
        ));
    }

    #[test]
    fn evidence_coverage_is_opt_in_and_intent_gated() {
        let mut intent = crate::facets::StructuralQueryPlan::default();
        assert!(!should_run_evidence_coverage(
            EvidenceCoverageMode::Off,
            Some(&intent)
        ));
        assert!(should_run_evidence_coverage(
            EvidenceCoverageMode::Force,
            None
        ));
        assert!(!should_run_evidence_coverage(
            EvidenceCoverageMode::Auto,
            Some(&intent)
        ));

        intent.labels.push("multi_evidence_summary".to_string());
        assert!(should_run_evidence_coverage(
            EvidenceCoverageMode::Auto,
            Some(&intent)
        ));

        intent.labels.clear();
        intent
            .labels
            .push("multi_evidence_collection".to_string());
        assert!(should_run_evidence_coverage(
            EvidenceCoverageMode::Auto,
            Some(&intent)
        ));

        intent.labels.clear();
        intent.labels.push("ordered_reconstruction".to_string());
        assert!(should_run_evidence_coverage(
            EvidenceCoverageMode::Auto,
            Some(&intent)
        ));
    }

    #[test]
    fn evidence_coverage_selects_diverse_session_evidence() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "evidence_coverage_test".to_string(),
        );

        let add_turn = |session: &str,
                        order: i64,
                        plan: Option<i64>,
                        content: &str,
                        cues: &[&str]|
         -> MemoryId {
            let mut metadata = HashMap::new();
            metadata.insert("source_session_id".to_string(), serde_json::json!(session));
            metadata.insert("source_turn_index".to_string(), serde_json::json!(order));
            if let Some(plan) = plan {
                metadata.insert("source_plan_idx".to_string(), serde_json::json!(plan));
            }
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                Some(metadata),
                MainStats::default(),
                false,
            )
        };

        let integration = add_turn(
            "thread-a",
            1,
            Some(1),
            "We designed language service integration.",
            &["source_role:assistant", "type:answer", "has:list", "language", "service", "integration", "architecture"],
        );
        let deployment = add_turn(
            "thread-a",
            2,
            Some(2),
            "We planned deployment and release steps.",
            &["source_role:assistant", "type:answer", "deployment", "release", "service"],
        );
        let performance = add_turn(
            "thread-a",
            3,
            Some(3),
            "We improved performance and latency.",
            &["source_role:assistant", "type:answer", "performance", "latency", "service"],
        );
        let unrelated = add_turn(
            "thread-a",
            4,
            Some(4),
            "We discussed a lunch menu.",
            &["source_role:assistant", "type:answer", "lunch", "menu"],
        );
        let distractor = add_turn(
            "thread-b",
            1,
            Some(2),
            "A different deployment discussion happened elsewhere.",
            &["source_role:assistant", "type:answer", "deployment", "service"],
        );

        let pivot = crate::engine::RecallResult {
            memory_id: deployment,
            content: "We planned deployment and release steps.".to_string(),
            score: 140.0,
            match_integrity: 0.6,
            intersection_count: 2,
            recency_score: 1.0,
            reinforcement_score: 0.0,
            salience_score: 0.0,
            created_at: 0.0,
            metadata: HashMap::new(),
            explain: None,
        };
        let pivot_score = pivot.score;

        let evidence = evidence_coverage_results(
            &ctx,
            &[
                ("language".to_string(), 1.0),
                ("service".to_string(), 0.8),
                ("integration".to_string(), 1.0),
                ("deployment".to_string(), 1.0),
                ("performance".to_string(), 1.0),
            ],
            &[pivot],
            10,
            100,
            1,
            true,
        );

        let ids: Vec<MemoryId> = evidence.iter().map(|result| result.memory_id).collect();
        assert!(ids.contains(&integration));
        assert!(ids.contains(&deployment));
        assert!(ids.contains(&performance));
        assert!(!ids.contains(&unrelated));
        assert!(!ids.contains(&distractor));
        assert!(evidence
            .iter()
            .all(|result| result.metadata.contains_key("evidence_coverage")));
        assert!(evidence.iter().any(|result| result
            .metadata
            .contains_key("evidence_coverage_source_plan")));
        assert!(evidence
            .iter()
            .all(|result| result.score < pivot_score));
    }

    #[test]
    fn slate_rerank_is_mode_and_intent_gated() {
        let mut intent = crate::facets::StructuralQueryPlan::default();
        intent.labels.push("multi_evidence_summary".to_string());

        assert!(!slate_rerank_requested(
            OrderedReconstructionMode::Off,
            EvidenceCoverageMode::Off,
            Some(&intent)
        ));
        assert!(slate_rerank_requested(
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&intent)
        ));

        let plain_intent = crate::facets::StructuralQueryPlan::default();
        assert!(!slate_rerank_requested(
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&plain_intent)
        ));
    }

    #[test]
    fn slate_rerank_promotes_coverage_candidates_below_protected_top() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "slate_rerank_test".to_string(),
        );

        let add_turn = |session: &str,
                        order: i64,
                        role: &str,
                        content: &str,
                        cues: &[&str]|
         -> MemoryId {
            let mut metadata = HashMap::new();
            metadata.insert("source_session_id".to_string(), serde_json::json!(session));
            metadata.insert("source_turn_index".to_string(), serde_json::json!(order));
            metadata.insert("source_role".to_string(), serde_json::json!(role));
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                Some(metadata),
                MainStats::default(),
                false,
            )
        };
        let make_result = |memory_id: MemoryId,
                           score: f64,
                           metadata: HashMap<String, serde_json::Value>|
         -> crate::engine::RecallResult {
            crate::engine::RecallResult {
                memory_id,
                content: format!("memory {memory_id}"),
                score,
                match_integrity: 0.2,
                intersection_count: 1,
                recency_score: 0.0,
                reinforcement_score: 0.0,
                salience_score: 0.0,
                created_at: 0.0,
                metadata,
                explain: None,
            }
        };

        let protected_a = add_turn(
            "thread-a",
            1,
            "assistant",
            "Protected top result A.",
            &["overview"],
        );
        let protected_b = add_turn(
            "thread-a",
            2,
            "assistant",
            "Protected top result B.",
            &["overview"],
        );
        let protected_c = add_turn(
            "thread-a",
            3,
            "assistant",
            "Protected top result C.",
            &["overview"],
        );
        let mut results = vec![
            make_result(protected_a, 300.0, HashMap::new()),
            make_result(protected_b, 290.0, HashMap::new()),
            make_result(protected_c, 280.0, HashMap::new()),
        ];

        for rank in 0..25 {
            let id = add_turn(
                "thread-b",
                rank,
                "assistant",
                "Generic distractor.",
                &["generic", "discussion"],
            );
            results.push(make_result(id, 270.0 - rank as f64, HashMap::new()));
        }

        let relevant_late = add_turn(
            "thread-a",
            24,
            "assistant",
            "We covered deployment and latency.",
            &["deployment", "latency", "service", "type:answer"],
        );
        let relevant_later = add_turn(
            "thread-a",
            40,
            "assistant",
            "We also covered integration architecture.",
            &["integration", "architecture", "service", "type:answer"],
        );
        let mut evidence_metadata = HashMap::new();
        evidence_metadata.insert("evidence_coverage".to_string(), serde_json::json!(true));
        results.push(make_result(relevant_late, 150.0, evidence_metadata.clone()));
        results.push(make_result(relevant_later, 149.0, evidence_metadata));

        let mut intent = crate::facets::StructuralQueryPlan::default();
        intent.labels.push("multi_evidence_summary".to_string());
        let moved = apply_slate_rerank(
            &ctx,
            &mut results,
            &[
                ("deployment".to_string(), 1.0),
                ("latency".to_string(), 1.0),
                ("integration".to_string(), 1.0),
                ("architecture".to_string(), 1.0),
                ("service".to_string(), 0.8),
            ],
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&intent),
            100,
        );
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let ids: Vec<MemoryId> = results.iter().map(|result| result.memory_id).collect();
        assert_eq!(&ids[..3], &[protected_a, protected_b, protected_c]);
        assert!(ids.iter().position(|id| *id == relevant_late).unwrap() < 20);
        assert!(ids.iter().position(|id| *id == relevant_later).unwrap() < 20);
        assert!(moved >= 2);
        assert!(results
            .iter()
            .any(|result| result.memory_id == relevant_late
                && result.metadata.contains_key("slate_rerank")));
    }

    #[test]
    fn slate_rerank_promotes_strong_summary_candidates_without_helper_metadata() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "slate_rerank_summary_signal_test".to_string(),
        );

        let add_turn = |session: &str,
                        order: i64,
                        content: &str,
                        cues: &[&str]|
         -> MemoryId {
            let mut metadata = HashMap::new();
            metadata.insert("source_session_id".to_string(), serde_json::json!(session));
            metadata.insert("source_turn_index".to_string(), serde_json::json!(order));
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                Some(metadata),
                MainStats::default(),
                false,
            )
        };
        let make_result = |memory_id: MemoryId,
                           score: f64|
         -> crate::engine::RecallResult {
            crate::engine::RecallResult {
                memory_id,
                content: format!("memory {memory_id}"),
                score,
                match_integrity: 0.2,
                intersection_count: 1,
                recency_score: 0.0,
                reinforcement_score: 0.0,
                salience_score: 0.0,
                created_at: 0.0,
                metadata: HashMap::new(),
                explain: None,
            }
        };

        let protected_a = add_turn("thread-a", 1, "Protected A.", &["overview"]);
        let protected_b = add_turn("thread-a", 2, "Protected B.", &["overview"]);
        let protected_c = add_turn("thread-a", 3, "Protected C.", &["overview"]);
        let mut results = vec![
            make_result(protected_a, 300.0),
            make_result(protected_b, 290.0),
            make_result(protected_c, 280.0),
        ];

        for rank in 0..25 {
            let id = add_turn(
                "thread-b",
                rank,
                "Generic project discussion.",
                &["generic", "project"],
            );
            results.push(make_result(id, 270.0 - rank as f64));
        }

        let relevant = add_turn(
            "thread-c",
            8,
            "City autocomplete in the weather app uses a debounced API lookup.",
            &["city", "autocomplete", "weather", "app", "lookup"],
        );
        results.push(make_result(relevant, 150.0));

        let mut intent = crate::facets::StructuralQueryPlan::default();
        intent.labels.push("multi_evidence_summary".to_string());
        let moved = apply_slate_rerank(
            &ctx,
            &mut results,
            &[
                ("city".to_string(), 1.0),
                ("autocomplete".to_string(), 1.0),
                ("weather".to_string(), 1.0),
                ("app".to_string(), 0.8),
                ("implementation".to_string(), 0.8),
            ],
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&intent),
            100,
        );
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let ids: Vec<MemoryId> = results.iter().map(|result| result.memory_id).collect();
        assert_eq!(&ids[..3], &[protected_a, protected_b, protected_c]);
        assert!(ids.iter().position(|id| *id == relevant).unwrap() < 20);
        assert!(moved >= 1);
        assert!(results
            .iter()
            .any(|result| result.memory_id == relevant
                && result.metadata.contains_key("slate_rerank")));
    }

    #[test]
    fn slate_rerank_promotes_standing_instruction_for_instruction_query() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "slate_rerank_instruction_test".to_string(),
        );

        let add_turn = |content: &str, cues: &[&str]| -> MemoryId {
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                None,
                MainStats::default(),
                false,
            )
        };
        let make_result = |memory_id: MemoryId,
                           score: f64|
         -> crate::engine::RecallResult {
            crate::engine::RecallResult {
                memory_id,
                content: format!("memory {memory_id}"),
                score,
                match_integrity: 0.1,
                intersection_count: 1,
                recency_score: 0.0,
                reinforcement_score: 0.0,
                salience_score: 0.0,
                created_at: 0.0,
                metadata: HashMap::new(),
                explain: None,
            }
        };

        let protected_a = add_turn("Protected A", &["layout"]);
        let protected_b = add_turn("Protected B", &["layout"]);
        let protected_c = add_turn("Protected C", &["layout"]);
        let mut results = vec![
            make_result(protected_a, 300.0),
            make_result(protected_b, 290.0),
            make_result(protected_c, 280.0),
        ];
        for rank in 0..45 {
            let id = add_turn("Generic layout discussion", &["layout", "project"]);
            results.push(make_result(id, 270.0 - rank as f64));
        }

        let instruction = add_turn(
            "Always include semantic HTML5 tag usage details when I ask about markup structure.",
            &[
                "type:standing_instruction",
                "instruction_trigger:markup",
                "semantic",
                "html5",
                "tag",
                "structure",
            ],
        );
        results.push(make_result(instruction, 120.0));

        let mut intent = crate::facets::StructuralQueryPlan::default();
        intent.labels.push("instruction_applicable".to_string());
        let moved = apply_slate_rerank(
            &ctx,
            &mut results,
            &[
                ("blog".to_string(), 1.0),
                ("layout".to_string(), 1.0),
                ("header".to_string(), 1.0),
                ("navigation".to_string(), 1.0),
                ("footer".to_string(), 1.0),
            ],
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&intent),
            100,
        );
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let ids: Vec<MemoryId> = results.iter().map(|result| result.memory_id).collect();
        assert_eq!(&ids[..3], &[protected_a, protected_b, protected_c]);
        assert!(ids.iter().position(|id| *id == instruction).unwrap() < 20);
        assert!(moved >= 1);
        assert!(results
            .iter()
            .any(|result| result.memory_id == instruction
                && result.metadata.contains_key("slate_rerank")));
    }

    #[test]
    fn slate_rerank_orders_selected_ordered_candidates_after_selection() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "slate_rerank_ordered_test".to_string(),
        );

        let add_turn = |session: &str, order: i64, content: &str, cues: &[&str]| -> MemoryId {
            let mut metadata = HashMap::new();
            metadata.insert("source_session_id".to_string(), serde_json::json!(session));
            metadata.insert("source_turn_index".to_string(), serde_json::json!(order));
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                Some(metadata),
                MainStats::default(),
                false,
            )
        };
        let make_result = |memory_id: MemoryId,
                           score: f64,
                           ordered: bool|
         -> crate::engine::RecallResult {
            let mut metadata = HashMap::new();
            if ordered {
                metadata.insert("ordered_reconstruction".to_string(), serde_json::json!(true));
            }
            crate::engine::RecallResult {
                memory_id,
                content: format!("memory {memory_id}"),
                score,
                match_integrity: 0.3,
                intersection_count: 1,
                recency_score: 0.0,
                reinforcement_score: 0.0,
                salience_score: 0.0,
                created_at: 0.0,
                metadata,
                explain: None,
            }
        };

        let protected_a = add_turn("thread-a", 1, "Protected A", &["bootstrap"]);
        let protected_b = add_turn("thread-a", 2, "Protected B", &["bootstrap"]);
        let protected_c = add_turn("thread-a", 3, "Protected C", &["bootstrap"]);
        let mut results = vec![
            make_result(protected_a, 300.0, false),
            make_result(protected_b, 290.0, false),
            make_result(protected_c, 280.0, false),
        ];
        for rank in 0..25 {
            let id = add_turn("thread-b", rank, "Generic project discussion", &["project"]);
            results.push(make_result(id, 270.0 - rank as f64, false));
        }

        let first = add_turn("thread-a", 5, "Bootstrap CDN setup", &["bootstrap", "cdn"]);
        let second = add_turn("thread-a", 7, "Bootstrap form classes", &["bootstrap", "form"]);
        let third = add_turn("thread-a", 11, "Bootstrap modal upgrade", &["bootstrap", "modal"]);
        results.push(make_result(third, 151.0, true));
        results.push(make_result(first, 150.0, true));
        results.push(make_result(second, 149.0, true));

        let mut intent = crate::facets::StructuralQueryPlan::default();
        intent.labels.push("ordered_reconstruction".to_string());
        let moved = apply_slate_rerank(
            &ctx,
            &mut results,
            &[
                ("bootstrap".to_string(), 1.0),
                ("cdn".to_string(), 1.0),
                ("form".to_string(), 1.0),
                ("modal".to_string(), 1.0),
            ],
            OrderedReconstructionMode::Auto,
            EvidenceCoverageMode::Off,
            Some(&intent),
            100,
        );
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let ids: Vec<MemoryId> = results.iter().map(|result| result.memory_id).collect();
        assert_eq!(&ids[..3], &[protected_a, protected_b, protected_c]);
        let first_pos = ids.iter().position(|id| *id == first).unwrap();
        let second_pos = ids.iter().position(|id| *id == second).unwrap();
        let third_pos = ids.iter().position(|id| *id == third).unwrap();
        assert!(first_pos < 20);
        assert!(second_pos < 20);
        assert!(third_pos < 20);
        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
        assert!(moved >= 3);
    }

    #[test]
    fn ordered_reconstruction_scans_selected_session_in_order() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "ordered_test".to_string(),
        );

        let add_turn = |session: &str, order: i64, content: &str, cues: &[&str]| -> MemoryId {
            let mut metadata = HashMap::new();
            metadata.insert("source_session_id".to_string(), serde_json::json!(session));
            metadata.insert("source_turn_index".to_string(), serde_json::json!(order));
            ctx.main.add_memory(
                content.to_string(),
                cues.iter().map(|cue| cue.to_string()).collect(),
                Some(metadata),
                MainStats::default(),
                false,
            )
        };

        let first = add_turn(
            "thread-a",
            1,
            "We integrated the language detection service.",
            &["language", "service", "integrate"],
        );
        let second = add_turn(
            "thread-a",
            2,
            "Then we optimized translation service latency.",
            &["translation", "service", "optimize"],
        );
        let distractor = add_turn(
            "thread-b",
            1,
            "A different service discussion happened elsewhere.",
            &["translation", "service", "discussion"],
        );

        let mut pivot_metadata = HashMap::new();
        pivot_metadata.insert("source_session_id".to_string(), serde_json::json!("thread-a"));
        pivot_metadata.insert("source_turn_index".to_string(), serde_json::json!(2));
        let pivot = crate::engine::RecallResult {
            memory_id: second,
            content: "Then we optimized translation service latency.".to_string(),
            score: 120.0,
            match_integrity: 0.6,
            intersection_count: 2,
            recency_score: 1.0,
            reinforcement_score: 0.0,
            salience_score: 0.0,
            created_at: 0.0,
            metadata: pivot_metadata,
            explain: None,
        };

        let ordered = ordered_reconstruction_results(
            &ctx,
            &[
                ("language".to_string(), 1.0),
                ("translation".to_string(), 1.0),
                ("service".to_string(), 1.0),
                ("optimize".to_string(), 1.0),
            ],
            &[pivot],
            10,
            100,
            1,
            true,
        );

        let ids: Vec<MemoryId> = ordered.iter().map(|result| result.memory_id).collect();
        assert!(ids.contains(&first));
        assert!(ids.contains(&second));
        assert!(!ids.contains(&distractor));
        assert!(ordered
            .iter()
            .all(|result| result.metadata.contains_key("ordered_reconstruction")));
    }

    #[test]
    fn segment_link_requires_parent_and_chunk_idx() {
        assert_eq!(
            segment_link_from_cues(&[
                "parent:abc".to_string(),
                "chunk_idx:7".to_string(),
                "source_role:user".to_string(),
            ]),
            Some(("parent:abc".to_string(), 7))
        );
        assert_eq!(
            segment_link_from_cues(&["parent:abc".to_string()]),
            None
        );
    }

    #[test]
    fn stitched_chunk_join_removes_overlapped_sentences() {
        let joined = join_stitched_chunk_contents(&[
            "First sentence. Shared sentence.".to_string(),
            "Shared sentence. Final sentence.".to_string(),
        ]);

        assert_eq!(joined, "First sentence. Shared sentence. Final sentence.");
    }

    #[cfg(any())]
    #[test]
    fn source_answer_projection_requires_assistant_answer_language() {
        let source_answer_intent = crate::facets::StructuralQueryPlan {
            labels: vec!["source_answer".to_string()],
            ..Default::default()
        };
        assert!(!source_answer_projection_requested(
            Some(&source_answer_intent),
            Some("What did I buy last week?")
        ));
        assert!(source_answer_projection_requested(
            Some(&source_answer_intent),
            Some("What was in the assistant answer?")
        ));

        let assistant_intent = crate::facets::StructuralQueryPlan {
            labels: vec!["source_assistant".to_string()],
            ..Default::default()
        };
        assert!(source_answer_projection_requested(
            Some(&assistant_intent),
            Some("Can you remind me?")
        ));
    }

    #[cfg(any())]
    #[test]
    fn user_context_projection_targets_advice_without_source_intents() {
        assert!(user_context_projection_requested(
            None,
            Some("I've been having trouble with battery life. Any tips?")
        ));

        let recommendation_intent = crate::facets::StructuralQueryPlan {
            labels: vec!["recommendation".to_string()],
            ..Default::default()
        };
        assert!(user_context_projection_requested(
            Some(&recommendation_intent),
            Some("Can you recommend something for me?")
        ));
        assert!(!user_context_projection_requested(
            Some(&recommendation_intent),
            Some("Can you suggest a hotel for my upcoming trip to Miami?")
        ));

        for label in ["source_answer", "source_assistant", "source_user", "decision_selection"] {
            let intent = crate::facets::StructuralQueryPlan {
                labels: vec![label.to_string()],
                ..Default::default()
            };
            assert!(
                !user_context_projection_requested(Some(&intent), Some("Any tips?")),
                "source-specific query should not request user context projection for {label}"
            );
        }
    }

    #[cfg(any())]
    #[test]
    fn user_context_projection_anchors_require_specific_context() {
        let phone_accessory_anchors =
            projection_anchor_cues(Some("Can you suggest some useful accessories for my phone?"));
        assert_eq!(
            phone_accessory_anchors,
            vec!["accessory".to_string(), "phone".to_string()]
        );

        let media_recommendation_anchors =
            projection_anchor_cues(Some("Can you recommend a show or movie for me to watch tonight?"));
        assert!(media_recommendation_anchors.is_empty());

        let troubleshooting_anchors = projection_anchor_cues(Some(
            "I've been having trouble with the battery life on my phone lately. Any tips?",
        ));
        assert!(troubleshooting_anchors.contains(&"battery".to_string()));
        assert!(troubleshooting_anchors.contains(&"life".to_string()));
        assert!(troubleshooting_anchors.contains(&"phone".to_string()));

        let navigation_anchors = projection_anchor_cues(Some(
            "I'm a bit anxious about getting around Tokyo. Do you have any helpful tips?",
        ));
        assert_eq!(
            navigation_anchors,
            vec!["anxious".to_string(), "tokyo".to_string()]
        );

        let relevant = "assistant: A power bank can help with phone battery life while traveling.";
        let incidental = "assistant: You could schedule a phone call during the morning.";

        assert!(projection_anchor_match_count(relevant, &troubleshooting_anchors) >= 2);
        assert!(projection_anchor_match_count(incidental, &troubleshooting_anchors) < 2);
        assert!(!projection_pivot_matches_context(
            "assistant: A camera bag can complement your Sony setup.",
            4,
            &phone_accessory_anchors,
            true
        ));
        assert!(!projection_pivot_matches_context(
            "assistant: A camera bag can complement your Sony setup.",
            3,
            &phone_accessory_anchors,
            true
        ));
        assert!(projection_pivot_matches_context(
            "assistant: A phone case is a useful accessory for your phone setup.",
            2,
            &phone_accessory_anchors,
            true
        ));
        assert!(!projection_pivot_matches_context(
            "assistant: A camera bag can complement your Sony setup.",
            4,
            &phone_accessory_anchors,
            false
        ));

        let vague_interest_intent = crate::facets::StructuralQueryPlan {
            labels: vec!["vague_interest_recommendation".to_string()],
            ..Default::default()
        };
        assert!(suppress_user_context_projection_for_intent(Some(
            &vague_interest_intent
        )));
    }

    #[cfg(any())]
    #[test]
    fn standing_instruction_projection_is_intent_gated() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "standing_instruction_test".to_string(),
        );

        let instruction_id = ctx.main.add_memory(
            "Always provide fallback strategies when I ask about error handling in API services."
                .to_string(),
            vec!["api".to_string(), "error_handling".to_string()],
            None,
            MainStats::default(),
            false,
        );

        assert!(standing_instruction_projection_cues(
            &ctx,
            None,
            Some("What are some ways I can manage problems that come up when my API calls fail?")
        )
        .cues
        .is_empty());

        let intent = crate::facets::StructuralQueryPlan {
            labels: vec!["instruction_applicable".to_string()],
            ..Default::default()
        };
        let projection = standing_instruction_projection_cues(
            &ctx,
            Some(&intent),
            Some("What are some ways I can manage problems that come up when my API calls fail?"),
        );

        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "type:standing_instruction"));
        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "instruction_trigger:api"));

        let projection_results = ctx.main.recall_weighted(
            projection.cues.clone(),
            10,
            false,
            None,
            1,
            false,
            true,
            None,
            None,
        );
        let mut all_results = Vec::new();
        merge_standing_instruction_projection_results(
            &ctx,
            &mut all_results,
            projection_results,
            &projection.anchors,
        );

        let projected = all_results
            .iter()
            .find(|result| result.memory_id == instruction_id)
            .expect("standing instruction should be projected");
        assert!(projected
            .metadata
            .contains_key("standing_instruction_projection"));
    }

    #[cfg(any())]
    #[test]
    fn standing_instruction_projection_uses_morphological_anchor_variants() {
        let anchors =
            standing_instruction_projection_anchors(Some("How do I implement a login feature?"));
        assert!(anchors.contains(&"implement".to_string()));
        assert!(anchors.contains(&"implementation".to_string()));

        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "standing_instruction_morphology_test".to_string(),
        );

        ctx.main.add_memory(
            "Always format code snippets with syntax highlighting when I ask about implementation details."
                .to_string(),
            Vec::new(),
            None,
            MainStats::default(),
            false,
        );

        let intent = crate::facets::StructuralQueryPlan {
            labels: vec!["instruction_applicable".to_string()],
            ..Default::default()
        };
        let projection = standing_instruction_projection_cues(
            &ctx,
            Some(&intent),
            Some("How do I implement a login feature?"),
        );

        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "instruction_trigger:implementation"));
    }

    #[cfg(any())]
    #[test]
    fn standing_instruction_projection_maps_chance_to_probability_anchor() {
        let anchors = standing_instruction_projection_anchors(Some(
            "How do I calculate the chance of drawing a red card from a standard deck?",
        ));
        assert!(anchors.contains(&"chance".to_string()));
        assert!(anchors.contains(&"probability".to_string()));

        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "standing_instruction_probability_test".to_string(),
        );

        ctx.main.add_memory(
            "Always provide step-by-step explanations with concrete examples when I ask about probability concepts."
                .to_string(),
            Vec::new(),
            None,
            MainStats::default(),
            false,
        );

        let intent = crate::facets::StructuralQueryPlan {
            labels: vec!["instruction_applicable".to_string()],
            ..Default::default()
        };
        let projection = standing_instruction_projection_cues(
            &ctx,
            Some(&intent),
            Some("How do I calculate the chance of drawing a red card from a standard deck?"),
        );

        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "instruction_trigger:probability"));
    }

    #[cfg(any())]
    #[test]
    fn preference_projection_is_intent_gated() {
        let ctx = ProjectContext::new(
            NormalizationConfig::default(),
            Taxonomy::default(),
            Arc::new(TuningConfig::default()),
            crate::config::ServerConfig::default(),
            "preference_projection_test".to_string(),
        );

        let memory_id = ctx.main.add_memory(
            "I prefer geometric vector methods over purely trigonometric formulas for clarity, so can you explain how to use vector algebra to calculate geodesic length between two points on a sphere?".to_string(),
            vec![
                "sphere".to_string(),
                "two_point".to_string(),
                "vector".to_string(),
                "geodesic".to_string(),
            ],
            None,
            MainStats::default(),
            false,
        );

        let query = "Can you show me how to find the shortest path between two points on a sphere?";
        assert!(preference_projection_cues(&ctx, None, Some(query))
            .cues
            .is_empty());

        let intent = crate::facets::StructuralQueryPlan {
            labels: vec!["preference_applicable".to_string()],
            ..Default::default()
        };
        let projection = preference_projection_cues(&ctx, Some(&intent), Some(query));

        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "type:preference"));
        assert!(projection
            .cues
            .iter()
            .any(|(cue, _)| cue == "sphere" || cue == "two_point"));

        let projection_results = ctx.main.recall_weighted(
            projection.cues.clone(),
            10,
            false,
            None,
            1,
            false,
            true,
            None,
            None,
        );
        let mut all_results = Vec::new();
        merge_preference_projection_results(
            &ctx,
            &mut all_results,
            projection_results,
            &projection.anchors,
        );

        let projected = all_results
            .iter()
            .find(|result| result.memory_id == memory_id)
            .expect("matching preference should be projected");
        assert!(projected.metadata.contains_key("preference_projection"));
    }

    #[cfg(any())]
    #[test]
    fn user_context_projection_merge_marks_and_updates_results() {
        let mut existing = recall_result(0.2, 1);
        existing.memory_id = 10;
        existing.score = 10.0;

        let mut projected = recall_result(0.8, 3);
        projected.memory_id = 10;
        projected.score = 50.0;
        projected
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));

        let mut all_results = vec![existing];
        merge_user_context_projection_results(&mut all_results, vec![projected]);

        assert_eq!(all_results.len(), 1);
        assert_eq!(all_results[0].score, 50.0);
        assert!(all_results[0]
            .metadata
            .contains_key("user_context_projection"));
    }

    #[cfg(any())]
    #[test]
    fn source_prompt_projection_filters_short_scaffold_prompts() {
        let mut scaffold = recall_result(0.1, 1);
        scaffold.memory_id = 20;
        scaffold.content = "user: Write another scene".to_string();
        scaffold.score = 5000.0;
        scaffold
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));

        let mut source = recall_result(0.8, 9);
        source.memory_id = 21;
        source.content =
            "user: Write a comedy movie scene. Andy wears an untidy stained white shirt."
                .to_string();
        source.score = 600.0;
        source
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));

        let mut results = Vec::new();
        merge_source_prompt_projection_results(
            &mut results,
            vec![scaffold, source],
            Some("what was Andy wearing in the script you wrote for the comedy movie scene?"),
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_id, 21);
        assert!(results[0]
            .metadata
            .contains_key("source_prompt_projection"));
        assert!(results[0].score > 600.0);
    }

    #[cfg(any())]
    #[test]
    fn user_context_adjacency_prefers_nearest_prior_user_turn() {
        let mut expected = recall_result(0.2, 1);
        expected.memory_id = 30;
        expected.score = 100.0;
        expected.created_at = 1.0;
        expected
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));
        expected.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-3"),
        );
        expected
            .metadata
            .insert("user_context_projection".to_string(), serde_json::json!(true));

        let mut pivot = recall_result(0.6, 2);
        pivot.memory_id = 31;
        pivot.score = 500.0;
        pivot.created_at = 2.0;
        pivot
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));
        pivot.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-3"),
        );

        let mut later_user = recall_result(0.2, 1);
        later_user.memory_id = 32;
        later_user.score = 900.0;
        later_user.created_at = 3.0;
        later_user
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));
        later_user.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-3"),
        );
        later_user
            .metadata
            .insert("user_context_projection".to_string(), serde_json::json!(true));

        let mut results = vec![expected, pivot, later_user];
        apply_user_context_adjacency_preference(&mut results, None, Some("Any tips?"));

        assert!(results[0].score > results[2].score);
        assert!(results[0]
            .metadata
            .contains_key("user_context_adjacency_boost"));
        assert!(!results[2]
            .metadata
            .contains_key("user_context_adjacency_boost"));
    }

    #[cfg(any())]
    #[test]
    fn user_context_adjacency_considers_bounded_multiple_pivots() {
        fn with_source(
            mut result: crate::engine::RecallResult,
            role: &str,
            session: &str,
            projected: bool,
        ) -> crate::engine::RecallResult {
            result
                .metadata
                .insert("source_role".to_string(), serde_json::json!(role));
            result.metadata.insert(
                "source_session_id".to_string(),
                serde_json::json!(session),
            );
            if projected {
                result
                    .metadata
                    .insert("user_context_projection".to_string(), serde_json::json!(true));
            }
            result
        }

        let mut first_user = recall_result(0.2, 1);
        first_user.memory_id = 40;
        first_user.score = 90.0;
        first_user.created_at = 1.0;

        let mut first_pivot = recall_result(0.6, 2);
        first_pivot.memory_id = 41;
        first_pivot.score = 900.0;
        first_pivot.created_at = 2.0;

        let mut second_user = recall_result(0.2, 1);
        second_user.memory_id = 42;
        second_user.score = 80.0;
        second_user.created_at = 3.0;

        let mut second_pivot = recall_result(0.6, 2);
        second_pivot.memory_id = 43;
        second_pivot.score = 800.0;
        second_pivot.created_at = 4.0;

        let mut expected = recall_result(0.2, 1);
        expected.memory_id = 44;
        expected.score = 70.0;
        expected.created_at = 5.0;

        let mut expected_pivot = recall_result(0.6, 2);
        expected_pivot.memory_id = 45;
        expected_pivot.score = 500.0;
        expected_pivot.created_at = 6.0;

        let mut results = vec![
            with_source(first_user, "user", "conversation-7", true),
            with_source(first_pivot, "assistant", "conversation-7", false),
            with_source(second_user, "user", "conversation-7", true),
            with_source(second_pivot, "assistant", "conversation-7", false),
            with_source(expected, "user", "conversation-7", true),
            with_source(expected_pivot, "assistant", "conversation-7", false),
        ];

        apply_user_context_adjacency_preference(
            &mut results,
            None,
            Some("Any helpful tips?"),
        );

        assert!(results[4]
            .metadata
            .contains_key("user_context_adjacency_boost"));
        assert!(results[4].score > 70.0);
    }

    #[test]
    fn source_session_cue_is_derived_from_structured_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("Answer ShareGPT hA7AkP3 0"),
        );

        assert_eq!(
            source_session_cue_from_metadata(&metadata).as_deref(),
            Some("source_session:answer_sharegpt_ha7akp3_0")
        );
    }

    #[test]
    fn list_answer_detection_covers_ordinals_without_topic_words() {
        assert!(query_wants_list_answer(Some(
            "What was the 7th item you listed?"
        )));
        assert!(query_wants_list_answer(Some(
            "Remind me what was in the list you provided."
        )));
        assert!(!query_wants_list_answer(Some(
            "What did I purchase yesterday?"
        )));
    }

    #[test]
    fn source_role_preference_demotes_structured_role_mismatches() {
        let mut user_result = recall_result(1.0, 3);
        user_result.score = 100.0;
        user_result
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));

        let mut assistant_result = recall_result(1.0, 3);
        assistant_result.memory_id = 50;
        assistant_result.score = 80.0;
        assistant_result
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));

        let intent = crate::facets::StructuralQueryPlan {
            labels: vec!["source_assistant".to_string()],
            ..Default::default()
        };
        let mut results = vec![user_result, assistant_result];
        apply_source_role_preference(&mut results, Some(&intent));

        assert!(results[0].score < results[1].score);
        assert_eq!(results[1].score, 80.0);
    }

    #[cfg(any())]
    #[test]
    fn source_answer_adjacency_prefers_immediate_assistant_reply() {
        let mut pivot = recall_result(1.0, 5);
        pivot.score = 1000.0;
        pivot.created_at = 1.0;
        pivot.metadata
            .insert("source_role".to_string(), serde_json::json!("user"));
        pivot.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-1"),
        );

        let mut immediate_answer = recall_result(1.0, 2);
        immediate_answer.memory_id = 60;
        immediate_answer.score = 300.0;
        immediate_answer.created_at = 2.0;
        immediate_answer
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));
        immediate_answer.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-1"),
        );

        let mut later_answer = recall_result(1.0, 8);
        later_answer.memory_id = 61;
        later_answer.score = 1000.0;
        later_answer.created_at = 6.0;
        later_answer
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));
        later_answer.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-1"),
        );

        let intent = crate::facets::StructuralQueryPlan {
            labels: vec!["source_answer".to_string(), "source_assistant".to_string()],
            ..Default::default()
        };
        let mut results = vec![pivot, immediate_answer, later_answer];
        apply_source_answer_adjacency_preference(&mut results, Some(&intent));

        assert!(results[1].score > results[2].score);
        assert!(results[1]
            .metadata
            .contains_key("source_answer_adjacency_boost"));
    }

    #[cfg(any())]
    #[test]
    fn decision_adjacency_prefers_selection_after_proposal() {
        let mut proposal = recall_result(1.0, 6);
        proposal.score = 3000.0;
        proposal.created_at = 1.0;
        proposal.content =
            "assistant: Here are some potential names: Radik, Nucleus, Fissionator.".to_string();
        proposal
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));
        proposal.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-2"),
        );

        let mut selected = recall_result(1.0, 1);
        selected.memory_id = 70;
        selected.score = 300.0;
        selected.created_at = 2.0;
        selected.content = "user: Fissionator is a really cool one.".to_string();
        selected
            .metadata
            .insert("source_role".to_string(), serde_json::json!("user"));
        selected.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-2"),
        );

        let mut later = recall_result(1.0, 4);
        later.memory_id = 71;
        later.score = 900.0;
        later.created_at = 5.0;
        later.content = "assistant: Fissionator could have radioactive attacks.".to_string();
        later
            .metadata
            .insert("source_role".to_string(), serde_json::json!("assistant"));
        later.metadata.insert(
            "source_session_id".to_string(),
            serde_json::json!("conversation-2"),
        );

        let intent = crate::facets::StructuralQueryPlan {
            labels: vec![
                "decision_selection".to_string(),
                "naming_decision".to_string(),
            ],
            ..Default::default()
        };
        let mut results = vec![proposal, selected, later];
        apply_decision_adjacency_preference(&mut results, Some(&intent));

        assert!(results[1].score > results[0].score);
        assert!(results[1].score > results[2].score);
        assert!(results[1]
            .metadata
            .contains_key("decision_adjacency_boost"));
    }

    #[test]
    fn project_headers_and_source_metadata_are_normalized_safely() {
        let mut headers = axum::http::HeaderMap::new();
        assert!(extract_project_id(&headers).is_err());
        headers.insert("X-Project-ID", "valid_project".parse().unwrap());
        assert_eq!(extract_project_id(&headers).unwrap(), "valid_project");
        assert_eq!(extract_project_id_optional(&headers).as_deref(), Some("valid_project"));
        headers.insert("X-Project-ID", "bad/project".parse().unwrap());
        assert!(extract_project_id(&headers).is_err());
        assert!(extract_project_id_optional(&headers).is_none());

        assert_eq!(normalize_source_value("  Assistant Role! "), Some("assistant_role".to_string()));
        assert!(normalize_source_value("-").is_none());
        let mut metadata = HashMap::new();
        metadata.insert("role".to_string(), serde_json::json!("assistant"));
        metadata.insert("thread_id".to_string(), serde_json::json!("thread-42"));
        assert_eq!(metadata_string(&metadata, &["missing", "role"]), Some("assistant"));
        assert_eq!(source_role_from_metadata(&metadata).as_deref(), Some("assistant"));
        assert_eq!(source_session_cue_from_metadata(&metadata).as_deref(), Some("source_session:thread_42"));
    }

    #[test]
    fn path_and_segment_defaults_are_safe_and_deterministic() {
        assert_eq!(normalize_included_paths(Some(vec![
            "src\\lib".to_string(),
            "./README.md".to_string(),
            "src/lib".to_string(),
        ])).unwrap(), vec!["README.md", "src/lib"]);
        assert!(normalize_included_paths(Some(vec!["../outside".to_string()])).is_err());
        assert_eq!(normalize_included_paths(Some(vec![".".to_string()])).unwrap(), Vec::<String>::new());
        assert_eq!(normalize_ignored_extensions(Some(vec![
            ".RS".to_string(),
            "rs".to_string(),
            " ".to_string(),
        ])), vec!["rs"]);
        assert_eq!(default_depth(), 1);
        assert_eq!(default_project_export_limit(), 1000);
        assert_eq!(default_parent_fusion_limit(), 80);
        assert_eq!(default_parent_fusion_min_chunks(), 2);
        assert_eq!(default_ordered_reconstruction_limit(), 80);
        assert_eq!(default_ordered_session_scan_limit(), 4096);
        assert_eq!(default_ordered_max_sessions(), 3);
        assert_eq!(default_evidence_coverage_limit(), 100);
        assert_eq!(default_evidence_coverage_session_scan_limit(), 4096);
        assert_eq!(default_evidence_coverage_max_sessions(), 3);
        assert_eq!(default_cuebridge_gap_limit(), 6);
        assert_eq!(default_filename(), "content.txt");
    }

    #[test]
    fn source_event_time_rejects_invalid_numeric_values() {
        let mut metadata = HashMap::new();
        metadata.insert("source_timestamp".to_string(), serde_json::json!(12.5));
        assert_eq!(source_event_time(None, Some(&metadata)), Some(12.5));
        metadata.insert("source_timestamp".to_string(), serde_json::json!(-1.0));
        assert_eq!(source_event_time(None, Some(&metadata)), None);
        metadata.insert("source_timestamp".to_string(), serde_json::json!("not-a-time"));
        assert_eq!(source_event_time(None, Some(&metadata)), None);
    }

    fn test_router() -> axum::Router {
        test_router_with_read_only(false)
    }

    fn test_router_with_read_only(read_only: bool) -> axum::Router {
        let snapshots = std::env::temp_dir().join(format!("cuemap-api-test-{}", uuid::Uuid::new_v4()));
        test_router_with_snapshots(snapshots, read_only)
    }

    fn test_router_with_snapshots(
        snapshots: std::path::PathBuf,
        read_only: bool,
    ) -> axum::Router {
        let mt_engine = Arc::new(MultiTenantEngine::with_snapshots_dir(
            &snapshots,
            TuningConfig::default(),
        ));
        let metrics = Arc::new(MetricsCollector::new());
        let provider: Arc<dyn crate::jobs::ProjectProvider> = mt_engine.clone();
        let job_queue = Arc::new(JobQueue::new(provider, Some(metrics.clone()), true));
        let agent_manager = Arc::new(crate::agent::manager::AgentManager::new(
            job_queue.clone(),
            mt_engine.clone(),
        ));
        routes(
            mt_engine,
            job_queue,
            metrics,
            AuthConfig::from_config(&crate::config::SecurityConfig::default()),
            read_only,
            snapshots.to_string_lossy().to_string(),
            None,
            None,
            agent_manager,
        )
    }

    async fn test_router_with_local_backup() -> axum::Router {
        let root = std::env::temp_dir().join(format!("cuemap-api-backup-{}", uuid::Uuid::new_v4()));
        let data_dir = root.join("data");
        let snapshots = data_dir.join("snapshots");
        let cloud_dir = root.join("cloud");
        std::fs::create_dir_all(&snapshots).unwrap();

        let mt_engine = Arc::new(MultiTenantEngine::with_snapshots_dir(
            &snapshots,
            TuningConfig::default(),
        ));
        let metrics = Arc::new(MetricsCollector::new());
        let provider: Arc<dyn crate::jobs::ProjectProvider> = mt_engine.clone();
        let job_queue = Arc::new(JobQueue::new(provider, Some(metrics.clone()), true));
        let agent_manager = Arc::new(crate::agent::manager::AgentManager::new(
            job_queue.clone(),
            mt_engine.clone(),
        ));
        let config = CloudBackupConfig::from_args(
            Some("local"),
            Some(cloud_dir.to_string_lossy().as_ref()),
            None,
            None,
            "cuemap/",
            false,
        )
        .unwrap();
        let backup = Arc::new(CloudBackupManager::new(config).await.unwrap());
        routes(
            mt_engine,
            job_queue,
            metrics,
            AuthConfig::from_config(&crate::config::SecurityConfig::default()),
            false,
            data_dir.to_string_lossy().to_string(),
            Some(backup),
            None,
            agent_manager,
        )
    }

    async fn json_body(response: axum::response::Response) -> serde_json::Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn local_http_url(body: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                let mut request = [0_u8; 4096];
                let _ = stream.read(&mut request).await;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{address}/fixture")
    }

    #[tokio::test]
    async fn routes_cover_root_stats_and_memory_lifecycle() {
        let router = test_router();
        let root = router
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(root.status(), StatusCode::OK);
        assert!(json_body(root).await["capabilities"].as_array().unwrap().len() >= 4);

        let missing_project = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_project.status(), StatusCode::BAD_REQUEST);

        let stored = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-test")
                    .body(Body::from(
                        r#"{"content":"hello world","cues":["greeting"],"metadata":{"source":"test"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stored.status(), StatusCode::OK);
        let stored_json = json_body(stored).await;
        let id = stored_json["id"].as_u64().unwrap();

        let fetched = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/memories/{id}"))
                    .header("X-Project-ID", "api-test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(fetched.status(), StatusCode::OK);
        assert_eq!(json_body(fetched).await["id"], id);

        let reinforced_with_cues = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri(format!("/memories/{id}/reinforce"))
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-test")
                    .body(Body::from(r#"{"cues":["greeting"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(reinforced_with_cues.status(), StatusCode::OK);

        let recalled = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-test")
                    .body(Body::from(r#"{"cues":["greeting"],"limit":5}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(recalled.status(), StatusCode::OK);
        assert!(!json_body(recalled).await["results"].as_array().unwrap().is_empty());

        let deleted = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/memories/{id}"))
                    .header("X-Project-ID", "api-test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(deleted.status(), StatusCode::OK);

        let stats = router
            .oneshot(
                Request::builder()
                    .uri("/stats")
                    .header("X-Project-ID", "api-test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stats.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn routes_cover_projects_aliases_lexicon_and_directory_preview() {
        let router = test_router();
        let project = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"project_id":"api-project"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(project.status(), StatusCode::CREATED);

        let invalid_project = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"project_id":"x"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_project.status(), StatusCode::BAD_REQUEST);

        let listed = router
            .clone()
            .oneshot(Request::builder().uri("/projects").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(listed.status(), StatusCode::OK);
        assert!(json_body(listed).await.as_array().unwrap().iter().any(|p| p["project_id"] == "api-project"));

        let alias = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/aliases")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-project")
                    .body(Body::from(r#"{"from":"rust","to":"rust_language"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(alias.status(), StatusCode::OK);

        let aliases = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/aliases?cue=rust")
                    .header("X-Project-ID", "api-project")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(aliases.status(), StatusCode::OK);

        let merged = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/aliases/merge")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-project")
                    .body(Body::from(r#"{"cues":["rust","rs"],"to":"rust_language"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(merged.status(), StatusCode::OK);

        let wired = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/lexicon/wire")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-project")
                    .body(Body::from(r#"{"token":"rs","canonical":"rust"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(wired.status(), StatusCode::OK);
        let lexicon_id = json_body(wired).await["memory_id"].as_u64().unwrap();

        for uri in ["/lexicon/inspect/rust", "/lexicon/graph"] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(uri)
                        .header("X-Project-ID", "api-project")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let deleted_lexicon = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/lexicon/entry/{lexicon_id}"))
                    .header("X-Project-ID", "api-project")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(deleted_lexicon.status(), StatusCode::OK);

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("notes.md"), "hello\nworld").unwrap();
        let preview = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ingest/directory/preview")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"watch_dir": dir.path(), "included_paths":["notes.md"]}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(preview.status(), StatusCode::OK);

        let watch = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/api-project/watch-dir")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "watch_dir": dir.path(),
                            "included_paths": ["notes.md"],
                            "ignored_extensions": [".log"]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(watch.status(), StatusCode::OK);
        let watch_info = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/projects/api-project/watch-dir")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(watch_info.status(), StatusCode::OK);
        assert_eq!(json_body(watch_info).await["initialized"], true);

        let deleted_project = router
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/projects/api-project")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(deleted_project.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn routes_cover_project_guards_reload_and_directory_validation() {
        let router = test_router();

        let missing_classify_project = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/intent/classify")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"text":"classify this"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_classify_project.status(), StatusCode::BAD_REQUEST);

        let created = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/reload-project")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::METHOD_NOT_ALLOWED);

        let project = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"project_id":"reload-project"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(project.status(), StatusCode::CREATED);

        let reloaded = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/reload-project/artifacts")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(reloaded.status(), StatusCode::OK);

        for uri in [
            "/projects/bad!id/artifacts",
            "/projects/bad!id/export",
        ] {
            let response = router
                .clone()
                .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri}");
        }

        let uninitialized_watch = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/projects/bad!id/watch-dir")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(uninitialized_watch.status(), StatusCode::OK);
        assert_eq!(json_body(uninitialized_watch).await["initialized"], false);

        let missing_watch = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/projects/not-created/watch-dir")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_watch.status(), StatusCode::OK);
        assert_eq!(json_body(missing_watch).await["initialized"], false);

        let missing_delete = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/projects/not-created")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_delete.status(), StatusCode::NOT_FOUND);

        let dir = tempfile::tempdir().unwrap();
        for uri in ["/ingest/directory/preview", "/projects/reload-project/watch-dir"] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(uri)
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "watch_dir": dir.path(),
                                "included_paths": ["../outside"]
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri}");
        }
    }

    #[tokio::test]
    async fn routes_cover_batch_ingestion_debug_grounding_export_and_metrics() {
        let router = test_router();
        let batch = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories/batch")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "batch-project")
                    .body(Body::from(
                        r#"{"memories":[{"content":"first item","cues":["batch"]},{"content":"second item","cues":["batch"]}],"minimal_response":true,"trace_timing":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(batch.status(), StatusCode::OK);
        let batch_json = json_body(batch).await;
        let ids = batch_json["ids"].as_array().unwrap();
        assert_eq!(ids.len(), 2);

        let reinforce = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri(format!("/memories/{}/reinforce", ids[0].as_u64().unwrap()))
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "batch-project")
                    .body(Body::from(r#"{"cues":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(reinforce.status(), StatusCode::OK);

        let jobs = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/jobs/status")
                    .header("X-Project-ID", "batch-project")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(jobs.status(), StatusCode::OK);
        assert!(json_body(jobs).await.get("intent_ready").is_some());

        let debug = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/debug/analyze-text")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "batch-project")
                    .body(Body::from(r#"{"text":"First sentence. Second sentence.","filename":"notes.md","segmenter":"sentence_window","segment_window_size":2}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(debug.status(), StatusCode::OK);
        assert!(json_body(debug).await["chunks"].as_array().unwrap().len() >= 1);

        let ingested = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ingest/content")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "batch-project")
                    .body(Body::from(r#"{"content":"A longer note with enough content to create a chunk.","filename":"notes.md","source_key":"notes.md"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ingested.status(), StatusCode::OK);

        let grounded = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall/grounded")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "batch-project")
                    .body(Body::from(r#"{"query_text":"batch","token_budget":64,"limit":5}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(grounded.status(), StatusCode::OK);
        assert_eq!(json_body(grounded).await["signature_alg"], "none");

        let export = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/projects/batch-project/export?limit=1&include_metadata=false")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(export.status(), StatusCode::OK);
        let export_json = json_body(export).await;
        assert_eq!(export_json["count"], 1);
        assert_eq!(export_json["include_metadata"], false);

        let metrics = router
            .clone()
            .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(metrics.status(), StatusCode::OK);
        let metrics_body = to_bytes(metrics.into_body(), usize::MAX).await.unwrap();
        assert!(String::from_utf8(metrics_body.to_vec()).unwrap().contains("cuemap_total_memories"));

        let intent = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/intent/classify")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "batch-project")
                    .body(Body::from(r#"{"text":"What happened yesterday?","target":"query"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        #[cfg(feature = "semantic-encoder")]
        assert_eq!(intent.status(), StatusCode::OK);
        #[cfg(not(feature = "semantic-encoder"))]
        assert_eq!(intent.status(), StatusCode::SERVICE_UNAVAILABLE);

        let global_stats = router
            .clone()
            .oneshot(Request::builder().uri("/stats").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(global_stats.status(), StatusCode::OK);

        let artifacts = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/projects/batch-project/artifacts")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(artifacts.status(), StatusCode::OK);

        let backup_list = router
            .clone()
            .oneshot(Request::builder().uri("/backup/list").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(backup_list.status(), StatusCode::SERVICE_UNAVAILABLE);

        let boundary = "coverage-boundary";
        let multipart = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"filename\"\r\n\r\nupload-test.md\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"upload-test.md\"\r\nContent-Type: text/plain\r\n\r\nUploaded content for multipart ingestion.\r\n--{boundary}--\r\n"
        );
        let uploaded = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ingest/file")
                    .header("content-type", format!("multipart/form-data; boundary={boundary}"))
                    .header("X-Project-ID", "batch-project")
                    .body(Body::from(multipart))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(uploaded.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn routes_cover_read_only_api_paths() {
        let router = test_router_with_read_only(true);
        let cases = [
            ("POST", "/memories", r#"{"content":"blocked"}"#),
            ("POST", "/ingest/content", r#"{"content":"blocked"}"#),
            ("POST", "/ingest/url", r#"{"url":"not-a-url"}"#),
            ("POST", "/aliases", r#"{"from":"a","to":"b"}"#),
            ("POST", "/lexicon/wire", r#"{"token":"a","canonical":"b"}"#),
            (
                "POST",
                "/projects/readonly/watch-dir",
                r#"{"watch_dir":"/does/not/matter"}"#,
            ),
        ];
        for (method, uri, body) in cases {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("content-type", "application/json")
                        .header("X-Project-ID", "read-only")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN, "{method} {uri}");
        }

        let delete = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/memories/1")
                    .header("X-Project-ID", "read-only")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(delete.status(), StatusCode::FORBIDDEN);

        let recall_web = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall/web")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "read-only")
                    .body(Body::from(r#"{"url":"not-a-url","query":"blocked","persist":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(recall_web.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn routes_cover_memory_project_and_ingestion_errors() {
        let router = test_router();

        let invalid_event_time = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-errors")
                    .body(Body::from(r#"{"content":"bad timestamp","event_time":-1}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_event_time.status(), StatusCode::BAD_REQUEST);

        let missing_header = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content":"missing project"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_header.status(), StatusCode::BAD_REQUEST);

        let stored = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-errors")
                    .body(Body::from(
                        r#"{"content":"source keyed","source_key":"source-1","minimal_response":true,"trace_timing":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stored.status(), StatusCode::OK);
        let stored_body = json_body(stored).await;
        assert!(stored_body.get("timing").is_some());
        assert!(stored_body.get("cues").is_none());

        for (uri, method) in [
            ("/memories/999999", "GET"),
            ("/memories/999999", "DELETE"),
        ] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("X-Project-ID", "api-errors")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{method} {uri}");
        }

        let reinforce_missing = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/memories/999999/reinforce")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-errors")
                    .body(Body::from(r#"{"cues":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(reinforce_missing.status(), StatusCode::NOT_FOUND);

        let batch_empty = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories/batch")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-errors")
                    .body(Body::from(r#"{"memories":[],"trace_timing":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let batch_empty_body = json_body(batch_empty).await;
        assert_eq!(batch_empty_body["count"], 0);
        assert_eq!(batch_empty_body["timings"].as_array().unwrap().len(), 0);

        let batch_failure = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories/batch")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-errors")
                    .body(Body::from(
                        r#"{"memories":[{"content":"ok"},{"content":"bad","event_time":-1}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(batch_failure.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(batch_failure).await["failed_index"], 1);

        let aliases_missing_cue = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/aliases")
                    .header("X-Project-ID", "api-errors")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(aliases_missing_cue.status(), StatusCode::BAD_REQUEST);

        let lexicon_missing = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/lexicon/entry/999999")
                    .header("X-Project-ID", "api-errors")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(lexicon_missing.status(), StatusCode::NOT_FOUND);

        let embedding_mismatch = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ingest/content")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-errors")
                    .body(Body::from(r#"{"content":"one sentence.","embeddings":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(embedding_mismatch.status(), StatusCode::BAD_REQUEST);

        let empty_content = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ingest/content")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-errors")
                    .body(Body::from(r#"{"content":""}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(empty_content.status(), StatusCode::BAD_REQUEST);

        let boundary = "api-errors-boundary";
        let missing_file = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ingest/file")
                    .header("content-type", format!("multipart/form-data; boundary={boundary}"))
                    .header("X-Project-ID", "api-errors")
                    .body(Body::from(format!(
                        "--{boundary}\r\nContent-Disposition: form-data; name=\"note\"\r\n\r\nmissing file\r\n--{boundary}--\r\n"
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_file.status(), StatusCode::BAD_REQUEST);

        let invalid_url = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ingest/url")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-errors")
                    .body(Body::from(r#"{"url":"not-a-url"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_url.status(), StatusCode::BAD_REQUEST);

        let invalid_web_url = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall/web")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-errors")
                    .body(Body::from(r#"{"url":"not-a-url","query":"test"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_web_url.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn routes_cover_remaining_guards_and_unconfigured_backup_paths() {
        let router = test_router();

        for (method, uri, body) in [
            ("POST", "/backup/upload", r#"{"project_id":"guard-project"}"#),
            ("POST", "/backup/download", r#"{"project_id":"guard-project"}"#),
        ] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE, "{method} {uri}");
        }
        let backup_delete = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/backup/guard-project")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(backup_delete.status(), StatusCode::SERVICE_UNAVAILABLE);

        let classify_invalid = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/intent/classify")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "bad!project")
                    .body(Body::from(r#"{"text":"classify this"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(classify_invalid.status(), StatusCode::BAD_REQUEST);
        let classify_ok = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/intent/classify")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "guard-project")
                    .body(Body::from(r#"{"text":"What happened yesterday?","target":"query"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        #[cfg(feature = "semantic-encoder")]
        assert_eq!(classify_ok.status(), StatusCode::OK);
        #[cfg(not(feature = "semantic-encoder"))]
        assert_eq!(classify_ok.status(), StatusCode::SERVICE_UNAVAILABLE);

        for (method, uri) in [
            ("GET", "/memories/1"),
            ("DELETE", "/memories/1"),
            ("PATCH", "/memories/1/reinforce"),
        ] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("content-type", "application/json")
                        .body(Body::from(if method == "PATCH" { r#"{"cues":[]}"# } else { "" }))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{method} {uri}");
        }
        let global_stats = router
            .clone()
            .oneshot(Request::builder().uri("/stats").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(global_stats.status(), StatusCode::OK);

        let global_jobs = router
            .clone()
            .oneshot(Request::builder().uri("/jobs/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(global_jobs.status(), StatusCode::OK);
        let global_grounded = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall/grounded")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query_text":"guard project","projects":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(global_grounded.status(), StatusCode::OK);

        let read_only = test_router_with_read_only(true);
        let read_only_create = read_only
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"project_id":"blocked-project"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(read_only_create.status(), StatusCode::FORBIDDEN);
        let read_only_delete = read_only
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/memories/1")
                    .header("X-Project-ID", "blocked-project")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(read_only_delete.status(), StatusCode::FORBIDDEN);
        for (uri, body) in [
            ("/aliases/merge", r#"{"cues":["a"],"to":"b"}"#),
            ("/lexicon/entry/1", ""),
        ] {
            let response = read_only
                .clone()
                .oneshot(
                    Request::builder()
                        .method(if uri == "/lexicon/entry/1" { "DELETE" } else { "POST" })
                        .uri(uri)
                        .header("content-type", "application/json")
                        .header("X-Project-ID", "blocked-project")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN, "{uri}");
        }

        for (uri, method, body) in [
            ("/projects/not-created/artifacts", "GET", ""),
            ("/projects/not-created/artifacts", "POST", ""),
            ("/projects/not-created/export", "GET", ""),
            ("/projects/not-created/watch-dir", "GET", ""),
        ] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert!(
                matches!(response.status(), StatusCode::OK | StatusCode::NOT_FOUND | StatusCode::SERVICE_UNAVAILABLE),
                "{method} {uri} returned {}",
                response.status()
            );
        }

        let missing_preview = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ingest/directory/preview")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"watch_dir":"/definitely/missing/cuemap-dir"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_preview.status(), StatusCode::BAD_REQUEST);

        let invalid_reload = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/bad!id/artifacts")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_reload.status(), StatusCode::BAD_REQUEST);

        let logical = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ingest/content")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "guard-project")
                    .body(Body::from(
                        r#"{"content":"First block.\n\nSecond block.","filename":"notes.md","segmenter":"logical_block","segment_window_size":2,"segment_overlap":1,"segment_min_chunk_chars":1,"segment_max_chunk_chars":100}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(logical.status(), StatusCode::OK);

        let debug_logical = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/debug/analyze-text")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "guard-project")
                    .body(Body::from(
                        r#"{"text":"First block.\n\nSecond block.","segmenter":"logical_block","segment_window_size":2,"segment_overlap":1,"segment_min_chunk_chars":1,"segment_max_chunk_chars":100}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(debug_logical.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn routes_cover_cuebridge_gap_expansion_in_recall() {
        let snapshots = std::env::temp_dir().join(format!("cuemap-api-gap-{}", uuid::Uuid::new_v4()));
        let artifact_dir = snapshots.join("artifacts").join("api-gap");
        std::fs::create_dir_all(&artifact_dir).unwrap();
        std::fs::write(
            artifact_dir.join("gap.json"),
            r#"{
                "artifact_type":"gap_pack",
                "name":"api-gap-pack",
                "entries":[{
                    "id":"deployment-release",
                    "query_signature":{"required_any":["deployment"]},
                    "expansions":[{"cue":"release","weight":1.0}],
                    "confidence":0.9,
                    "max_fanout":2
                }]
            }"#,
        )
        .unwrap();
        let router = test_router_with_snapshots(snapshots, false);

        let stored = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-gap")
                    .body(Body::from(r#"{"content":"Deployment release is ready.","cues":["deployment","release"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stored.status(), StatusCode::OK);

        let recalled = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-gap")
                    .body(Body::from(r#"{"query_text":"deployment","cues":["deployment"],"disable_cuebridge_artifacts":false,"explain":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(recalled.status(), StatusCode::OK);
        let body = json_body(recalled).await;
        assert!(body["results"].is_array());
    }

    #[tokio::test]
    async fn routes_cover_recall_modes_and_project_exports() {
        let router = test_router();
        let stored = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-recall")
                    .body(Body::from(
                        r#"{"content":"The deployment decision moved the service to a regional cluster.","cues":["deployment","decision","service","regional","cluster"],"metadata":{"source_session_id":"session-api","source_turn_index":1,"source_role":"assistant","type":"answer"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stored.status(), StatusCode::OK);

        let embedding = vec![0.01_f32; 384];
        let semantic_stored = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-recall")
                    .body(Body::from(
                        serde_json::json!({
                            "content": "Semantic deployment vector",
                            "cues": ["semantic", "deployment"],
                            "embedding": embedding,
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(semantic_stored.status(), StatusCode::OK);

        let semantic_recall = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-recall")
                    .body(Body::from(
                        serde_json::json!({
                            "semantic_mode": "semantic",
                            "query_embedding": vec![0.01_f32; 384],
                            "limit": 3,
                            "explain": true,
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(semantic_recall.status(), StatusCode::OK);
        assert!(json_body(semantic_recall).await["results"].is_array());

        let recall = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-recall")
                    .body(Body::from(
                        r#"{"query_text":"summarize the deployment decision","cues":["deployment"],"semantic_mode":"lexical","limit":5,"depth":2,"min_intersection":1,"explain":true,"trace_timing":true,"auto_reinforce":true,"ordered_reconstruction":"force","evidence_coverage":"force","parent_fusion":"force","disable_cuebridge_artifacts":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(recall.status(), StatusCode::OK);
        let recall_body = json_body(recall).await;
        assert!(recall_body.get("explain").is_some());
        assert!(recall_body.get("timing").is_some());

        let cross_project = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"projects":["api-recall","api-recall-other"],"query_text":"deployment","cues":["deployment"],"semantic_mode":"lexical","limit":3,"explain":true,"auto_reinforce":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cross_project.status(), StatusCode::OK);
        let cross_body = json_body(cross_project).await;
        assert_eq!(cross_body["results"].as_array().unwrap().len(), 2);

        let artifacts = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/projects/api-recall/artifacts")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(artifacts.status(), StatusCode::OK);

        let export_without_fields = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/projects/api-recall/export?limit=1&include_content=false&include_cues=false&include_metadata=false")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(export_without_fields.status(), StatusCode::OK);
        let export_body = json_body(export_without_fields).await;
        assert_eq!(export_body["include_content"], false);
        assert_eq!(export_body["include_cues"], false);
        assert_eq!(export_body["include_metadata"], false);
        assert!(export_body["memories"][0].get("content").is_none());

        let export_after_cursor = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/projects/api-recall/export?cursor=999999&limit=1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(export_after_cursor.status(), StatusCode::OK);
        assert_eq!(json_body(export_after_cursor).await["count"], 0);

        let get_watch_meta = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/projects/api-recall/watch-dir")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(get_watch_meta.status(), StatusCode::OK);

        let invalid_watch = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/api-recall/watch-dir")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"watch_dir":"/does/not/exist"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_watch.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn routes_cover_local_cloud_backup_lifecycle() {
        let router = test_router_with_local_backup().await;
        let stored = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "backup-project")
                    .body(Body::from(r#"{"content":"backup content","cues":["backup"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stored.status(), StatusCode::OK);

        let invalid_upload = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/backup/upload")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"project_id":"x"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_upload.status(), StatusCode::BAD_REQUEST);

        let uploaded = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/backup/upload")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"project_id":"backup-project"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(uploaded.status(), StatusCode::OK);
        assert_eq!(json_body(uploaded).await["success"], true);

        let listed = router
            .clone()
            .oneshot(Request::builder().uri("/backup/list").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(listed.status(), StatusCode::OK);
        assert!(json_body(listed).await["count"].as_u64().unwrap() >= 1);

        let downloaded = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/backup/download")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"project_id":"backup-project"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(downloaded.status(), StatusCode::OK);
        assert_eq!(json_body(downloaded).await["success"], true);

        let deleted = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/backup/backup-project")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(deleted.status(), StatusCode::OK);

        let missing_download = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/backup/download")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"project_id":"backup-project"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_download.status(), StatusCode::NOT_FOUND);

        let invalid_delete = router
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/backup/x")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_delete.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn routes_cover_local_url_ingestion_and_web_recall() {
        let router = test_router();
        let url = local_http_url(
            "<html><head><title>Deployment Notes</title></head><body><h1>Deployment</h1><p>The regional service uses a rolling release.</p></body></html>",
        )
        .await;

        let ingested = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ingest/url")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-url")
                    .body(Body::from(serde_json::json!({"url": url, "depth": 0}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ingested.status(), StatusCode::OK);
        let ingested_body = json_body(ingested).await;
        assert_eq!(ingested_body["status"], "ingested");
        assert!(ingested_body["chunks"].as_u64().unwrap() >= 1);

        let recall_url = local_http_url(
            "<html><body><h1>Deployment</h1><p>The regional service uses a rolling release.</p></body></html>",
        )
        .await;
        let web_recall = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall/web")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-url")
                    .body(Body::from(
                        serde_json::json!({"url": recall_url, "query": "deployment regional"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(web_recall.status(), StatusCode::OK);
        let web_body = json_body(web_recall).await;
        assert_eq!(web_body["urls"].as_array().unwrap().len(), 1);
        assert!(web_body["results"].is_array());

        let persisted_url = local_http_url(
            "<html><body><h1>Persisted Deployment</h1><p>The rolling release is persisted asynchronously.</p></body></html>",
        )
        .await;
        let persisted_web_recall = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall/web")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-url")
                    .body(Body::from(
                        serde_json::json!({"url": persisted_url, "query": "deployment", "persist": true}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(persisted_web_recall.status(), StatusCode::OK);

        let crawl_url = local_http_url(
            "<html><body><h1>Crawl Deployment</h1><p>A single crawlable deployment page.</p></body></html>",
        )
        .await;
        let crawled = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ingest/url")
                    .header("content-type", "application/json")
                    .header("X-Project-ID", "api-url")
                    .body(Body::from(
                        serde_json::json!({"url": crawl_url, "depth": 1, "same_domain_only": true}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(crawled.status(), StatusCode::OK);
    }
