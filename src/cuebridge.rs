use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

const GAP_PACK_TYPE: &str = "gap_pack";
const ALIAS_PACK_TYPE: &str = "alias_pack";
const MAX_GAP_EXPANSIONS: usize = 6;

#[derive(Debug, Clone, Default, Serialize)]
pub struct CueBridgeArtifacts {
    pub artifact_infos: Vec<CueBridgeArtifactInfo>,
    pub load_errors: Vec<String>,
    #[serde(skip)]
    gap_entries: Vec<RuntimeGapEntry>,
    #[serde(skip)]
    alias_entries: Vec<RuntimeAliasEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CueBridgeArtifactInfo {
    pub name: String,
    pub artifact_type: String,
    pub path: String,
    pub sha256: String,
    pub entry_count: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CueBridgeArtifactSummary {
    pub artifact_count: usize,
    pub gap_entry_count: usize,
    pub alias_entry_count: usize,
    pub artifacts: Vec<CueBridgeArtifactInfo>,
    pub load_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CueBridgeAliasExpansion {
    pub artifact: String,
    pub artifact_hash: String,
    pub entry_id: String,
    pub from: String,
    pub to: String,
    pub weight: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CueBridgeGapExpansion {
    pub artifact: String,
    pub artifact_hash: String,
    pub entry_id: String,
    pub cue: String,
    pub weight: f64,
    pub confidence: f64,
    pub score: f64,
}

#[derive(Debug, Deserialize)]
struct ArtifactKind {
    artifact_type: String,
}

#[derive(Debug, Deserialize)]
struct RawGapPack {
    name: String,
    #[serde(default)]
    entries: Vec<RawGapEntry>,
}

#[derive(Debug, Deserialize)]
struct RawGapEntry {
    id: String,
    query_signature: RawQuerySignature,
    #[serde(default)]
    expansions: Vec<RawExpansion>,
    #[serde(default)]
    negative_gates: Vec<String>,
    confidence: f64,
    #[serde(default)]
    max_fanout: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawQuerySignature {
    #[serde(default)]
    required_any: Vec<String>,
    #[serde(default)]
    required_all: Vec<String>,
    #[serde(default)]
    intent_any: Vec<String>,
    #[serde(default)]
    source_any: Vec<String>,
    #[serde(default)]
    ordered_tokens_any: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawExpansion {
    cue: String,
    weight: f64,
}

#[derive(Debug, Deserialize)]
struct RawAliasPack {
    name: String,
    #[serde(default)]
    entries: Vec<RawAliasEntry>,
}

#[derive(Debug, Deserialize)]
struct RawAliasEntry {
    id: String,
    from: String,
    to: String,
    #[serde(default)]
    weight: Option<f64>,
    confidence: f64,
    #[serde(default)]
    gates: Vec<String>,
}

#[derive(Debug, Clone)]
struct RuntimeGapEntry {
    artifact: String,
    artifact_hash: String,
    id: String,
    signature: RuntimeQuerySignature,
    expansions: Vec<RawExpansion>,
    negative_gates: Vec<String>,
    confidence: f64,
    max_fanout: usize,
}

#[derive(Debug, Clone, Default)]
struct RuntimeQuerySignature {
    required_any: Vec<String>,
    required_all: Vec<String>,
    intent_any: Vec<String>,
    source_any: Vec<String>,
    ordered_tokens_any: Vec<String>,
}

#[derive(Debug, Clone)]
struct RuntimeAliasEntry {
    artifact: String,
    artifact_hash: String,
    id: String,
    from: String,
    to: String,
    weight: f64,
    confidence: f64,
    gates: Vec<String>,
}

impl CueBridgeArtifacts {
    pub fn load_for_project(data_dir: &str, project_id: &str) -> Self {
        let artifact_dir = Path::new(data_dir).join("artifacts").join(project_id);
        Self::load_dir(&artifact_dir)
    }

    pub fn load_dir(dir: &Path) -> Self {
        let mut artifacts = Self::default();
        if !dir.exists() {
            return artifacts;
        }

        let mut files = Vec::new();
        collect_json_files(dir, &mut files);
        files.sort();

        for path in files {
            if let Err(err) = artifacts.load_file(&path) {
                artifacts
                    .load_errors
                    .push(format!("{}: {}", path.display(), err));
            }
        }

        artifacts.sort();
        artifacts
    }

    pub fn summary(&self) -> CueBridgeArtifactSummary {
        CueBridgeArtifactSummary {
            artifact_count: self.artifact_infos.len(),
            gap_entry_count: self.gap_entries.len(),
            alias_entry_count: self.alias_entries.len(),
            artifacts: self.artifact_infos.clone(),
            load_errors: self.load_errors.clone(),
        }
    }

    pub fn has_runtime_entries(&self) -> bool {
        !self.gap_entries.is_empty() || !self.alias_entries.is_empty()
    }

    pub fn alias_expansions<F>(
        &self,
        cue: &str,
        query_cues: &[String],
        available: F,
    ) -> Vec<CueBridgeAliasExpansion>
    where
        F: Fn(&str) -> bool,
    {
        let cue = normalize_artifact_cue(cue);
        if cue.is_empty() {
            return Vec::new();
        }
        let query_set = normalized_set(query_cues.iter());
        let mut out = Vec::new();

        for entry in &self.alias_entries {
            if entry.from != cue {
                continue;
            }
            if !entry.gates.is_empty() && !entry.gates.iter().all(|gate| query_set.contains(gate)) {
                continue;
            }
            if !available(&entry.to) {
                continue;
            }
            out.push(CueBridgeAliasExpansion {
                artifact: entry.artifact.clone(),
                artifact_hash: entry.artifact_hash.clone(),
                entry_id: entry.id.clone(),
                from: entry.from.clone(),
                to: entry.to.clone(),
                weight: entry.weight,
                confidence: entry.confidence,
            });
        }

        out.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.artifact.cmp(&b.artifact))
                .then_with(|| a.entry_id.cmp(&b.entry_id))
                .then_with(|| a.to.cmp(&b.to))
        });
        out
    }

    pub fn gap_expansions<F>(
        &self,
        query_cues: &[(String, f64)],
        query_intent: Option<&crate::facets::QueryIntent>,
        ordered_tokens: &[String],
        available: F,
        max_expansions: usize,
    ) -> Vec<CueBridgeGapExpansion>
    where
        F: Fn(&str) -> bool,
    {
        if max_expansions == 0 || self.gap_entries.is_empty() {
            return Vec::new();
        }

        let query_cue_set = normalized_set(query_cues.iter().map(|(cue, _)| cue));
        let token_set = normalized_set(ordered_tokens.iter());
        let intent_set = query_intent
            .map(|intent| normalized_set(intent.labels.iter()))
            .unwrap_or_default();
        let mut out = Vec::new();

        for entry in &self.gap_entries {
            if !entry.matches(&query_cue_set, &intent_set, &token_set) {
                continue;
            }

            let mut emitted_for_entry = 0usize;
            for expansion in &entry.expansions {
                let cue = normalize_artifact_cue(&expansion.cue);
                if cue.is_empty() || query_cue_set.contains(&cue) || !available(&cue) {
                    continue;
                }
                let weight = sanitize_weight(expansion.weight, 1.0);
                out.push(CueBridgeGapExpansion {
                    artifact: entry.artifact.clone(),
                    artifact_hash: entry.artifact_hash.clone(),
                    entry_id: entry.id.clone(),
                    cue,
                    weight,
                    confidence: entry.confidence,
                    score: weight * entry.confidence,
                });
                emitted_for_entry += 1;
                if emitted_for_entry >= entry.max_fanout {
                    break;
                }
            }
        }

        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.artifact.cmp(&b.artifact))
                .then_with(|| a.entry_id.cmp(&b.entry_id))
                .then_with(|| a.cue.cmp(&b.cue))
        });

        let mut seen = HashSet::new();
        out.into_iter()
            .filter(|entry| seen.insert(entry.cue.clone()))
            .take(max_expansions.min(MAX_GAP_EXPANSIONS))
            .collect()
    }

    fn load_file(&mut self, path: &Path) -> Result<(), String> {
        let bytes = fs::read(path).map_err(|err| format!("read failed: {}", err))?;
        let hash = sha256_bytes(&bytes);
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|err| format!("invalid json: {}", err))?;
        let kind: ArtifactKind =
            serde_json::from_value(value.clone()).map_err(|err| format!("missing artifact_type: {}", err))?;

        match kind.artifact_type.as_str() {
            GAP_PACK_TYPE => self.load_gap_pack(path, hash, value),
            ALIAS_PACK_TYPE => self.load_alias_pack(path, hash, value),
            other => Err(format!("unsupported artifact_type '{}'", other)),
        }
    }

    fn load_gap_pack(
        &mut self,
        path: &Path,
        hash: String,
        value: serde_json::Value,
    ) -> Result<(), String> {
        let raw: RawGapPack =
            serde_json::from_value(value).map_err(|err| format!("invalid GapPack: {}", err))?;
        let mut entries = Vec::new();
        for entry in raw.entries {
            let id = entry.id.trim().to_string();
            if id.is_empty() || entry.expansions.is_empty() {
                continue;
            }
            let confidence = sanitize_confidence(entry.confidence);
            if confidence <= 0.0 {
                continue;
            }
            entries.push(RuntimeGapEntry {
                artifact: raw.name.clone(),
                artifact_hash: hash.clone(),
                id,
                signature: RuntimeQuerySignature::from_raw(entry.query_signature),
                expansions: entry.expansions,
                negative_gates: normalize_vec(entry.negative_gates),
                confidence,
                max_fanout: entry.max_fanout.unwrap_or(MAX_GAP_EXPANSIONS).clamp(1, MAX_GAP_EXPANSIONS),
            });
        }
        let entry_count = entries.len();
        self.gap_entries.extend(entries);
        self.artifact_infos.push(CueBridgeArtifactInfo {
            name: raw.name,
            artifact_type: GAP_PACK_TYPE.to_string(),
            path: path.display().to_string(),
            sha256: hash,
            entry_count,
        });
        Ok(())
    }

    fn load_alias_pack(
        &mut self,
        path: &Path,
        hash: String,
        value: serde_json::Value,
    ) -> Result<(), String> {
        let raw: RawAliasPack =
            serde_json::from_value(value).map_err(|err| format!("invalid AliasPack: {}", err))?;
        let mut entries = Vec::new();
        for entry in raw.entries {
            let id = entry.id.trim().to_string();
            let from = normalize_artifact_cue(&entry.from);
            let to = normalize_artifact_cue(&entry.to);
            if id.is_empty() || from.is_empty() || to.is_empty() || from == to {
                continue;
            }
            let confidence = sanitize_confidence(entry.confidence);
            if confidence <= 0.0 {
                continue;
            }
            entries.push(RuntimeAliasEntry {
                artifact: raw.name.clone(),
                artifact_hash: hash.clone(),
                id,
                from,
                to,
                weight: sanitize_weight(entry.weight.unwrap_or(0.85), 0.85),
                confidence,
                gates: normalize_vec(entry.gates),
            });
        }
        let entry_count = entries.len();
        self.alias_entries.extend(entries);
        self.artifact_infos.push(CueBridgeArtifactInfo {
            name: raw.name,
            artifact_type: ALIAS_PACK_TYPE.to_string(),
            path: path.display().to_string(),
            sha256: hash,
            entry_count,
        });
        Ok(())
    }

    fn sort(&mut self) {
        self.artifact_infos.sort_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| a.artifact_type.cmp(&b.artifact_type))
                .then_with(|| a.path.cmp(&b.path))
        });
        self.gap_entries.sort_by(|a, b| {
            a.artifact
                .cmp(&b.artifact)
                .then_with(|| a.id.cmp(&b.id))
        });
        self.alias_entries.sort_by(|a, b| {
            a.artifact
                .cmp(&b.artifact)
                .then_with(|| a.id.cmp(&b.id))
                .then_with(|| a.from.cmp(&b.from))
                .then_with(|| a.to.cmp(&b.to))
        });
    }
}

impl RuntimeQuerySignature {
    fn from_raw(raw: RawQuerySignature) -> Self {
        Self {
            required_any: normalize_vec(raw.required_any),
            required_all: normalize_vec(raw.required_all),
            intent_any: normalize_vec(raw.intent_any),
            source_any: normalize_vec(raw.source_any),
            ordered_tokens_any: normalize_vec(raw.ordered_tokens_any),
        }
    }

    fn has_any_gate(&self) -> bool {
        !self.required_any.is_empty()
            || !self.required_all.is_empty()
            || !self.intent_any.is_empty()
            || !self.source_any.is_empty()
            || !self.ordered_tokens_any.is_empty()
    }
}

impl RuntimeGapEntry {
    fn matches(
        &self,
        query_cues: &HashSet<String>,
        intent_labels: &HashSet<String>,
        ordered_tokens: &HashSet<String>,
    ) -> bool {
        if !self.signature.has_any_gate() {
            return false;
        }
        if self
            .negative_gates
            .iter()
            .any(|gate| query_cues.contains(gate) || intent_labels.contains(gate) || ordered_tokens.contains(gate))
        {
            return false;
        }
        if !self.signature.required_all.iter().all(|cue| query_cues.contains(cue)) {
            return false;
        }
        if !self.signature.required_any.is_empty()
            && !self
                .signature
                .required_any
                .iter()
                .any(|cue| query_cues.contains(cue))
        {
            return false;
        }
        if !self.signature.intent_any.is_empty()
            && !self
                .signature
                .intent_any
                .iter()
                .any(|label| intent_labels.contains(label))
        {
            return false;
        }
        if !self.signature.source_any.is_empty()
            && !self
                .signature
                .source_any
                .iter()
                .any(|cue| query_cues.contains(cue))
        {
            return false;
        }
        if !self.signature.ordered_tokens_any.is_empty()
            && !self
                .signature
                .ordered_tokens_any
                .iter()
                .any(|token| ordered_tokens.contains(token))
        {
            return false;
        }
        true
    }
}

fn collect_json_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_json_files(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            out.push(path);
        }
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[allow(dead_code)]
fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|err| err.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let read = file.read(&mut buffer).map_err(|err| err.to_string())?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn normalize_artifact_cue(value: &str) -> String {
    value.trim().to_lowercase()
}

fn normalize_vec(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let normalized = normalize_artifact_cue(&value);
        if !normalized.is_empty() && seen.insert(normalized.clone()) {
            out.push(normalized);
        }
    }
    out
}

fn normalized_set<'a, I>(values: I) -> HashSet<String>
where
    I: Iterator<Item = &'a String>,
{
    values
        .map(|value| normalize_artifact_cue(value))
        .filter(|value| !value.is_empty())
        .collect()
}

fn sanitize_confidence(value: f64) -> f64 {
    if !value.is_finite() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn sanitize_weight(value: f64, default: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 {
        default
    } else {
        value.min(10.0)
    }
}

#[cfg(test)]
mod tests {
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
}
