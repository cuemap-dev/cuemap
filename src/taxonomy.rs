use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Taxonomy {
    #[serde(default)]
    pub allowed_keys: Vec<String>,
    #[serde(default)]
    pub allowed_values: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub allowed_value_prefixes: HashMap<String, Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationReport {
    pub accepted: Vec<String>,
    pub rejected: Vec<RejectedCue>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RejectedCue {
    pub cue: String,
    pub code: String,   // "bad_format" | "unknown_key" | "unknown_value"
    pub detail: String,
}

pub fn validate_cues(cues: Vec<String>, taxonomy: &Taxonomy) -> ValidationReport {
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();

    for cue in cues {
        // 1. Check format k:v
        let parts: Vec<&str> = cue.splitn(2, ':').collect();
        if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
            rejected.push(RejectedCue {
                cue: cue.clone(),
                code: "bad_format".to_string(),
                detail: "Cue must be in 'key:value' format".to_string(),
            });
            continue;
        }

        let key = parts[0];
        let value = parts[1];

        // 2. Check allowed keys (if restricted)
        // If allowed_keys is empty, we assume NO restriction on keys (open taxonomy)
        // UNLESS the user explicitly wants a closed taxonomy by default.
        // The plan implies explicit allowlist. If allowed_keys is NOT empty, we enforce it.
        // If it IS empty, we might still want to enforce it if the user intends a strict schema.
        // However, usually empty list means "nothing allowed" in a strict system.
        // Let's assume strict: if allowed_keys is populated, key must be in it.
        // If allowed_keys is empty, we'll assume everything is allowed (open) OR nothing is allowed.
        // Given the context of "taxonomy validator", usually it's permissive by default unless configured.
        // But the prompt says "Taxonomy... allowed_keys".
        // Let's implement: If allowed_keys is NOT empty, key MUST be present.
        if !taxonomy.allowed_keys.is_empty() && !taxonomy.allowed_keys.contains(&key.to_string()) {
            rejected.push(RejectedCue {
                cue: cue.clone(),
                code: "unknown_key".to_string(),
                detail: format!("Key '{}' is not in allowed_keys", key),
            });
            continue;
        }

        // 3. Check allowed values
        let mut value_allowed = true; // Default to true if no constraints exist for this key

        let has_value_constraints = taxonomy.allowed_values.contains_key(key);
        let has_prefix_constraints = taxonomy.allowed_value_prefixes.contains_key(key);

        if has_value_constraints || has_prefix_constraints {
            value_allowed = false; // Constraints exist, so we must satisfy at least one

            // Check exact values
            if let Some(allowed_vals) = taxonomy.allowed_values.get(key) {
                if allowed_vals.contains(&value.to_string()) {
                    value_allowed = true;
                }
            }

            // Check prefixes
            if !value_allowed {
                if let Some(allowed_prefixes) = taxonomy.allowed_value_prefixes.get(key) {
                    for prefix in allowed_prefixes {
                        if value.starts_with(prefix) {
                            value_allowed = true;
                            break;
                        }
                    }
                }
            }
        }

        if value_allowed {
            accepted.push(cue);
        } else {
            rejected.push(RejectedCue {
                cue: cue.clone(),
                code: "unknown_value".to_string(),
                detail: format!("Value '{}' is not allowed for key '{}'", value, key),
            });
        }
    }

    ValidationReport { accepted, rejected }
}

