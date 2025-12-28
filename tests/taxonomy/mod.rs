use cuemap_rust::taxonomy::*;
use std::collections::HashMap;

#[test]
fn test_format_validation() {
    let taxonomy = Taxonomy::default();
    let cues = vec!["valid:cue".to_string(), "invalid".to_string(), "incomplete:".to_string()];
    let report = validate_cues(cues, &taxonomy);

    assert_eq!(report.accepted, vec!["valid:cue"]);
    assert_eq!(report.rejected.len(), 2);
    assert_eq!(report.rejected[0].code, "bad_format");
    assert_eq!(report.rejected[1].code, "bad_format");
}

#[test]
fn test_key_validation() {
    let taxonomy = Taxonomy {
        allowed_keys: vec!["status".to_string(), "user".to_string()],
        ..Default::default()
    };
    let cues = vec!["status:active".to_string(), "unknown:value".to_string()];
    let report = validate_cues(cues, &taxonomy);

    assert_eq!(report.accepted, vec!["status:active"]);
    assert_eq!(report.rejected.len(), 1);
    assert_eq!(report.rejected[0].code, "unknown_key");
}

#[test]
fn test_value_validation() {
    let mut allowed_values = HashMap::new();
    allowed_values.insert("status".to_string(), vec!["active".to_string(), "pending".to_string()]);

    let mut allowed_value_prefixes = HashMap::new();
    allowed_value_prefixes.insert("user".to_string(), vec!["id_".to_string()]);

    let taxonomy = Taxonomy {
        allowed_keys: vec!["status".to_string(), "user".to_string()],
        allowed_values,
        allowed_value_prefixes,
    };

    let cues = vec![
        "status:active".to_string(),   // Valid exact match
        "status:unknown".to_string(),  // Invalid value
        "user:id_123".to_string(),     // Valid prefix
        "user:admin".to_string(),      // Invalid prefix
    ];
    let report = validate_cues(cues, &taxonomy);

    assert_eq!(report.accepted, vec!["status:active", "user:id_123"]);
    assert_eq!(report.rejected.len(), 2);
    assert_eq!(report.rejected[0].code, "unknown_value"); // status:unknown
    assert_eq!(report.rejected[1].code, "unknown_value"); // user:admin
}
