use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RewriteRule {
    pub name: String,
    pub pattern: String,
    pub replace: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormalizationConfig {
    pub lowercase: bool,
    pub trim: bool,
    #[serde(default)]
    pub rewrite_rules: Vec<RewriteRule>,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            trim: true,
            rewrite_rules: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct NormalizeTrace {
    pub raw: String,
    pub normalized: String,
    pub applied_rules: Vec<String>,
}

/// Normalizes a cue string based on the provided configuration.
/// Applies trimming, lowercasing, and rewrite rules sequentially.
pub fn normalize_cue(raw: &str, config: &NormalizationConfig) -> (String, NormalizeTrace) {
    let mut current = raw.to_string();
    let mut applied_rules = Vec::new();

    // 1. Trim
    if config.trim {
        current = current.trim().to_string();
    }

    // 2. Lowercase
    if config.lowercase {
        current = current.to_lowercase();
    }

    // 3. Rewrite Rules
    for rule in &config.rewrite_rules {
        if let Ok(re) = Regex::new(&rule.pattern) {
            if re.is_match(&current) {
                let new_val = re.replace_all(&current, &rule.replace).to_string();
                if new_val != current {
                    current = new_val;
                    applied_rules.push(rule.name.clone());
                }
            }
        }
    }

    // 4. Fix Duplicated Prefixes (e.g., "key:value:value" -> "key:value")
    // This handles cases where rewrite rules or LLM output might accidentally double-prefix
    let parts: Vec<&str> = current.split(':').collect();
    if parts.len() >= 3 && parts[1] == parts[2] && !parts[1].is_empty() {
        // Reconstruct without the duplicated part
        // We keep parts[0] (key) and then parts[2..] (value onwards)
        let mut new_parts = vec![parts[0]];
        new_parts.extend_from_slice(&parts[2..]);
        current = new_parts.join(":");
        applied_rules.push("dedupe_prefix".to_string());
    }

    (
        current.clone(),
        NormalizeTrace {
            raw: raw.to_string(),
            normalized: current,
            applied_rules,
        },
    )
}

