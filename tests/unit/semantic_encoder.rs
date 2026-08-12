    use super::*;

    #[test]
    fn bundled_minilm_encoder_produces_normalized_vectors() {
        let config = SemanticConfig::default().resolved();
        assert_eq!(config.max_tokens, 128);
        let encoder = OnnxSemanticEncoder::from_config(&config)
            .expect("bundled MiniLM assets should load");
        let vector = encoder
            .encode("A short local semantic encoder smoke test")
            .expect("bundled MiniLM should encode text");

        assert_eq!(vector.len(), 384);
        let norm = vector
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "unexpected vector norm: {norm}");
    }

    #[test]
    fn bundled_edge_minilm_encoder_produces_normalized_vectors() {
        let mut config = SemanticConfig::default();
        config.profile = crate::semantic::SemanticProfile::Edge;
        let config = config.resolved();
        assert_eq!(config.max_tokens, 128);
        let encoder = OnnxSemanticEncoder::from_config(&config)
            .expect("bundled edge MiniLM assets should load");
        let vector = encoder
            .encode("A short edge semantic encoder smoke test")
            .expect("bundled edge MiniLM should encode text");

        assert_eq!(vector.len(), 384);
        let norm = vector
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "unexpected edge vector norm: {norm}");
    }
