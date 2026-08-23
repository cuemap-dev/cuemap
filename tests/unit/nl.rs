    use super::tokenize_to_cues;

    #[test]
    fn temporal_connector_breaks_phrase_without_removing_token() {
        let cues = tokenize_to_cues(
            "Maya switched from coffee to mint tea after the April deploy.",
        );

        assert!(cues.contains(&"after".to_string()));
        assert!(cues.contains(&"mint_tea".to_string()));
        assert!(cues.contains(&"april_deploy".to_string()));
        assert!(!cues.contains(&"mint_tea_after".to_string()));
    }
