use cuemap_rust::normalization::*;

#[test]
fn test_basic_normalization() {
    let config = NormalizationConfig::default();
    let (normalized, _) = normalize_cue("  TestCue  ", &config);
    assert_eq!(normalized, "testcue");
}

#[test]
fn test_rewrite_rules() {
    let config = NormalizationConfig {
        lowercase: true,
        trim: true,
        rewrite_rules: vec![
            RewriteRule {
                name: "service_prefix".to_string(),
                pattern: r"^([a-z0-9_]+)-service$".to_string(),
                replace: "service:$1".to_string(),
            },
        ],
    };

    let (normalized, trace) = normalize_cue("Payments-Service", &config);
    assert_eq!(normalized, "service:payments");
    assert_eq!(trace.applied_rules, vec!["service_prefix"]);
}

#[test]
fn test_rewrite_chaining() {
    let config = NormalizationConfig {
        lowercase: true,
        trim: true,
        rewrite_rules: vec![
            RewriteRule {
                name: "replace_dash".to_string(),
                pattern: r"-".to_string(),
                replace: "_".to_string(),
            },
            RewriteRule {
                name: "prefix_tag".to_string(),
                pattern: r"^([a-z_]+)$".to_string(),
                replace: "tag:$1".to_string(),
            },
        ],
    };

    // Input: "My-Value"
    // 1. Lowercase -> "my-value"
    // 2. replace_dash -> "my_value"
    // 3. prefix_tag -> "tag:my_value"
    let (normalized, _trace) = normalize_cue("My-Value", &config);
    assert_eq!(normalized, "tag:my_value");
}

#[test]
fn test_prefix_deduplication() {
    let config = NormalizationConfig::default();
    let (normalized, trace) = normalize_cue("lang:python:python", &config);
    assert_eq!(normalized, "lang:python");
    assert!(trace.applied_rules.contains(&"dedupe_prefix".to_string()));

    let (normalized2, _) = normalize_cue("topic:payments:payments", &config);
    assert_eq!(normalized2, "topic:payments");
}
