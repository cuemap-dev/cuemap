    use super::*;

    fn config() -> SemanticConfig {
        SemanticConfig {
            enabled: true,
            dimensions: 3,
            ann_tables: 2,
            ann_bits: 4,
            ann_probes: 2,
            candidate_limit: 8,
            exact_fallback_max: 32,
            ..SemanticConfig::default()
        }
    }

    #[test]
    fn profiles_resolve_to_device_appropriate_defaults() {
        let mut config = SemanticConfig::default();
        config.profile = SemanticProfile::Edge;
        let resolved = config.resolved();
        assert!(resolved.enabled);
        assert_eq!(resolved.dimensions, 384);
        assert_eq!(resolved.storage, SemanticStorage::Int8);
        assert_eq!(resolved.max_memory_mb, 32);
        assert_eq!(resolved.model_id, "all-MiniLM-L3-v2");
        assert_eq!(resolved.model_version, "bundled-q4-minilm-l3");
        assert_eq!(resolved.max_tokens, 128);
    }

    #[test]
    fn quality_profile_uses_compact_hybrid_defaults() {
        let resolved = SemanticConfig::default().resolved();

        assert_eq!(resolved.profile, SemanticProfile::Quality);
        assert_eq!(resolved.dimensions, 384);
        assert_eq!(resolved.storage, SemanticStorage::Int8);
        assert_eq!(resolved.model_id, "all-MiniLM-L3-v2");
        assert_eq!(resolved.model_version, "bundled-qint8-minilm-l3");
        assert_eq!(resolved.max_tokens, 128);
        assert_eq!(resolved.index, SemanticIndexMode::Auto);
        assert_eq!(resolved.semantic_rerank_weight, 0.60);
        assert_eq!(resolved.semantic_rerank_candidate_limit, 200);
        assert_eq!(resolved.query_embedding_cache_capacity, 256);
        assert_eq!(resolved.intent_rerank_weight, 0.65);
        assert_eq!(resolved.intent_rerank_max_delta, 64.0);
    }

    #[test]
    fn intent_delta_cap_is_non_negative_and_configurable() {
        let mut config = SemanticConfig::default();
        config.intent_rerank_max_delta = -10.0;
        assert_eq!(config.resolved().intent_rerank_max_delta, 0.0);

        config.intent_rerank_max_delta = 7.5;
        assert_eq!(config.resolved().intent_rerank_max_delta, 7.5);
    }

    #[test]
    fn compact_storage_has_expected_size_and_similarity() {
        let vector = [1.0, 0.0, 0.0, 0.25, 0.1, -0.2, 0.05, 0.3];
        let f32_vector = StoredSemanticVector::from_f32(&vector, SemanticStorage::F32).unwrap();
        let f16_vector = StoredSemanticVector::from_f32(&vector, SemanticStorage::F16).unwrap();
        let int8_vector = StoredSemanticVector::from_f32(&vector, SemanticStorage::Int8).unwrap();
        assert!(f16_vector.estimated_bytes() < f32_vector.estimated_bytes());
        assert!(int8_vector.estimated_bytes() < f16_vector.estimated_bytes());
        assert!(int8_vector.cosine_similarity(&vector).unwrap() > 0.99);
    }

    #[test]
    fn index_returns_nearest_vector() {
        let mut index = SemanticIndex::new(config());
        let first = StoredSemanticVector::from_f32(&[1.0, 0.0, 0.0], SemanticStorage::F32).unwrap();
        let second = StoredSemanticVector::from_f32(&[0.0, 1.0, 0.0], SemanticStorage::F32).unwrap();
        let third = StoredSemanticVector::from_f32(&[0.9, 0.1, 0.0], SemanticStorage::F32).unwrap();
        index.insert(1, &first).unwrap();
        index.insert(2, &second).unwrap();
        index.insert(3, &third).unwrap();

        let results = index.query_candidate_ids(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&1));
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let mut index = SemanticIndex::new(config());
        let first = StoredSemanticVector::from_f32(&[1.0, 0.0, 0.0], SemanticStorage::F32).unwrap();
        let second = StoredSemanticVector::from_f32(&[1.0, 0.0], SemanticStorage::F32).unwrap();
        index.insert(1, &first).unwrap();
        assert!(index.insert(2, &second).is_err());
        assert!(index.query_candidate_ids(&[1.0, 0.0], 1).is_err());
    }

    #[test]
    fn reranker_is_data_driven() {
        let model = LinearReranker {
            bias: 0.25,
            weights: vec![2.0, -1.0],
        };
        assert!((model.score(&[0.5, 0.25]) - 1.0).abs() < f32::EPSILON);
    }
