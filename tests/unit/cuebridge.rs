    use super::*;

    #[test]
    fn gap_pack_rejects_bare_entry_without_gates() {
        let entry = RuntimeGapEntry {
            artifact: "test".to_string(),
            artifact_hash: "hash".to_string(),
            id: "gap".to_string(),
            signature: RuntimeQuerySignature::default(),
            expansions: vec![RawExpansion {
                cue: "target".to_string(),
                weight: 1.0,
            }],
            negative_gates: Vec::new(),
            confidence: 1.0,
            max_fanout: 1,
        };
        assert!(!entry.matches(&HashSet::new(), &HashSet::new(), &HashSet::new()));
    }
