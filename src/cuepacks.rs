use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const BUNDLED_MEMORY_GENERAL: &str = include_str!("../cuepacks/memory-general.toml");
const OFF_SENTINELS: &[&str] = &["off", "none", "disabled", "core-only"];
const DEFAULT_SENTINELS: &[&str] = &["default", "defaults", "bundled"];

#[derive(Debug, Clone, Default, Serialize)]
pub struct CuePackRegistry {
    packs: Vec<CompiledCuePack>,
    load_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CuePackInfo {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub enabled_by_default: bool,
    pub source: String,
    pub memory_rules: usize,
    pub query_rules: usize,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct CuePackFacetOutput {
    pub facets: Vec<String>,
    pub matched_rules: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct CuePackQueryOutput {
    pub labels: Vec<String>,
    pub weighted_cues: Vec<(String, f64)>,
    pub cue_weight_adjustments: Vec<(String, f64)>,
    pub suppress_generic: bool,
    pub matched_rules: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawCuePack {
    name: String,
    version: Option<String>,
    description: Option<String>,
    #[serde(default = "default_true")]
    enabled_by_default: bool,
    #[serde(default)]
    memory_rules: Vec<RawRule>,
    #[serde(default)]
    query_rules: Vec<RawRule>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawRule {
    id: String,
    #[serde(default)]
    contains_any: Vec<String>,
    #[serde(default)]
    contains_all: Vec<String>,
    #[serde(default)]
    regex_any: Vec<String>,
    #[serde(default)]
    emits: Vec<String>,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    weighted_cues: Vec<RawWeightedCue>,
    #[serde(default)]
    cue_weight_adjustments: Vec<RawWeightedCue>,
    #[serde(default)]
    suppress_generic: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct RawWeightedCue {
    cue: String,
    weight: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CompiledCuePack {
    name: String,
    version: String,
    description: Option<String>,
    enabled_by_default: bool,
    source: String,
    memory_rules: Vec<CompiledRule>,
    query_rules: Vec<CompiledRule>,
}

#[derive(Debug, Clone, Serialize)]
struct CompiledRule {
    id: String,
    contains_any: Vec<String>,
    contains_all: Vec<String>,
    #[serde(skip)]
    regex_any: Vec<Regex>,
    emits: Vec<String>,
    labels: Vec<String>,
    weighted_cues: Vec<(String, f64)>,
    cue_weight_adjustments: Vec<(String, f64)>,
    suppress_generic: bool,
}

fn default_true() -> bool {
    true
}

pub fn default_registry() -> &'static CuePackRegistry {
    static REGISTRY: OnceLock<CuePackRegistry> = OnceLock::new();
    REGISTRY.get_or_init(CuePackRegistry::bundled)
}

impl CuePackRegistry {
    pub fn bundled() -> Self {
        let mut registry = Self::default();
        match Self::compile_pack(BUNDLED_MEMORY_GENERAL, "bundled:memory-general") {
            Ok(pack) => registry.packs.push(pack),
            Err(err) => registry.load_errors.push(err),
        }
        registry.sort();
        registry
    }

    pub fn load(default_packs_enabled: bool, dirs: &[PathBuf]) -> Self {
        let mut registry = if default_packs_enabled {
            Self::bundled()
        } else {
            Self::default()
        };

        for dir in dirs {
            registry.load_dir(dir);
        }
        registry.sort();
        registry
    }

    pub fn load_from_default_locations(default_packs_enabled: bool) -> Self {
        let base_dir = crate::config::get_base_dir();
        Self::load(default_packs_enabled, &[base_dir.join("cuepacks")])
    }

    pub fn load_errors(&self) -> &[String] {
        &self.load_errors
    }

    pub fn infos(&self) -> Vec<CuePackInfo> {
        self.packs
            .iter()
            .map(|pack| CuePackInfo {
                name: pack.name.clone(),
                version: pack.version.clone(),
                description: pack.description.clone(),
                enabled_by_default: pack.enabled_by_default,
                source: pack.source.clone(),
                memory_rules: pack.memory_rules.len(),
                query_rules: pack.query_rules.len(),
            })
            .collect()
    }

    pub fn validate_file(path: &Path) -> Result<CuePackInfo, String> {
        let content = fs::read_to_string(path)
            .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
        let pack = Self::compile_pack(&content, &path.display().to_string())?;
        Ok(CuePackInfo {
            name: pack.name,
            version: pack.version,
            description: pack.description,
            enabled_by_default: pack.enabled_by_default,
            source: pack.source,
            memory_rules: pack.memory_rules.len(),
            query_rules: pack.query_rules.len(),
        })
    }

    pub fn extract_memory_facets(
        &self,
        content: &str,
        selection: Option<&[String]>,
    ) -> CuePackFacetOutput {
        let normalized = padded_normalized(content);
        let mut out = CuePackFacetOutput::default();
        let mut seen = HashSet::new();

        for pack in self.active_packs(selection) {
            for rule in &pack.memory_rules {
                if rule.matches(content, &normalized) {
                    out.matched_rules.push(format!("{}:{}", pack.name, rule.id));
                    for facet in &rule.emits {
                        if seen.insert(facet.clone()) {
                            out.facets.push(facet.clone());
                        }
                    }
                }
            }
        }

        out
    }

    pub fn compile_query_intent<F>(
        &self,
        query: &str,
        selection: Option<&[String]>,
        available: F,
    ) -> CuePackQueryOutput
    where
        F: Fn(&str) -> bool,
    {
        let normalized = padded_normalized(query);
        let mut out = CuePackQueryOutput::default();
        let mut labels = HashSet::new();
        let mut cues = HashSet::new();
        let mut adjustments: HashMap<String, f64> = HashMap::new();

        for pack in self.active_packs(selection) {
            for rule in &pack.query_rules {
                if !rule.matches(query, &normalized) {
                    continue;
                }

                let mut emitted = false;
                for label in &rule.labels {
                    if labels.insert(label.clone()) {
                        out.labels.push(label.clone());
                        emitted = true;
                    }
                }
                for (cue, weight) in &rule.weighted_cues {
                    if available(cue) && cues.insert(cue.clone()) {
                        out.weighted_cues.push((cue.clone(), *weight));
                        emitted = true;
                    }
                }
                for (cue, weight) in &rule.cue_weight_adjustments {
                    if !available(cue) {
                        continue;
                    }
                    adjustments
                        .entry(cue.clone())
                        .and_modify(|existing| {
                            if *existing > *weight {
                                *existing = *weight;
                            }
                        })
                        .or_insert(*weight);
                    emitted = true;
                }
                if rule.suppress_generic {
                    out.suppress_generic = true;
                    emitted = true;
                }
                if emitted {
                    out.matched_rules.push(format!("{}:{}", pack.name, rule.id));
                }
            }
        }

        out.cue_weight_adjustments = adjustments.into_iter().collect();
        out.cue_weight_adjustments.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    fn load_dir(&mut self, dir: &Path) {
        if !dir.exists() {
            return;
        }

        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(err) => {
                self.load_errors
                    .push(format!("failed to read {}: {}", dir.display(), err));
                return;
            }
        };

        let mut paths = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("toml"))
            .collect::<Vec<_>>();
        paths.sort();

        for path in paths {
            match fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {}: {}", path.display(), err))
                .and_then(|content| Self::compile_pack(&content, &path.display().to_string()))
            {
                Ok(pack) => self.replace_or_push(pack),
                Err(err) => self.load_errors.push(err),
            }
        }
    }

    fn replace_or_push(&mut self, pack: CompiledCuePack) {
        if let Some(existing) = self
            .packs
            .iter_mut()
            .find(|existing| existing.name == pack.name)
        {
            *existing = pack;
        } else {
            self.packs.push(pack);
        }
    }

    fn active_packs(&self, selection: Option<&[String]>) -> Vec<&CompiledCuePack> {
        let Some(selection) = selection else {
            return self
                .packs
                .iter()
                .filter(|pack| pack.enabled_by_default)
                .collect();
        };

        if selection
            .iter()
            .any(|name| OFF_SENTINELS.contains(&name.to_lowercase().as_str()))
        {
            return Vec::new();
        }

        let mut include_defaults = false;
        let requested = selection
            .iter()
            .filter_map(|name| {
                let lower = name.to_lowercase();
                if DEFAULT_SENTINELS.contains(&lower.as_str()) {
                    include_defaults = true;
                    None
                } else {
                    Some(lower)
                }
            })
            .collect::<HashSet<_>>();

        self.packs
            .iter()
            .filter(|pack| {
                (include_defaults && pack.enabled_by_default)
                    || requested.contains(&pack.name.to_lowercase())
            })
            .collect()
    }

    fn compile_pack(content: &str, source: &str) -> Result<CompiledCuePack, String> {
        let raw: RawCuePack =
            toml::from_str(content).map_err(|err| format!("invalid cuepack {source}: {err}"))?;
        if raw.name.trim().is_empty() {
            return Err(format!("invalid cuepack {source}: missing name"));
        }

        Ok(CompiledCuePack {
            name: raw.name.trim().to_string(),
            version: raw.version.unwrap_or_else(|| "0.1.0".to_string()),
            description: raw.description,
            enabled_by_default: raw.enabled_by_default,
            source: source.to_string(),
            memory_rules: compile_rules(raw.memory_rules, source)?,
            query_rules: compile_rules(raw.query_rules, source)?,
        })
    }

    fn sort(&mut self) {
        self.packs.sort_by(|a, b| a.name.cmp(&b.name));
        self.load_errors.sort();
    }
}

impl CompiledRule {
    fn matches(&self, original: &str, normalized: &str) -> bool {
        if !self.contains_all.is_empty()
            && !self
                .contains_all
                .iter()
                .all(|term| normalized.contains(term))
        {
            return false;
        }

        if !self.contains_any.is_empty()
            && !self
                .contains_any
                .iter()
                .any(|term| normalized.contains(term))
        {
            return false;
        }

        if !self.regex_any.is_empty() && !self.regex_any.iter().any(|re| re.is_match(original)) {
            return false;
        }

        true
    }
}

fn compile_rules(rules: Vec<RawRule>, source: &str) -> Result<Vec<CompiledRule>, String> {
    let mut compiled = Vec::with_capacity(rules.len());
    let mut ids = HashSet::new();

    for raw in rules {
        if raw.id.trim().is_empty() {
            return Err(format!("invalid cuepack {source}: rule with empty id"));
        }
        if !ids.insert(raw.id.clone()) {
            return Err(format!("invalid cuepack {source}: duplicate rule {}", raw.id));
        }

        let mut regex_any = Vec::with_capacity(raw.regex_any.len());
        for pattern in raw.regex_any {
            regex_any.push(
                Regex::new(&pattern)
                    .map_err(|err| format!("invalid cuepack {source} rule {}: {err}", raw.id))?,
            );
        }

        compiled.push(CompiledRule {
            id: raw.id,
            contains_any: raw
                .contains_any
                .into_iter()
                .map(|term| padded_rule_term(&term))
                .collect(),
            contains_all: raw
                .contains_all
                .into_iter()
                .map(|term| padded_rule_term(&term))
                .collect(),
            regex_any,
            emits: raw.emits,
            labels: raw.labels,
            weighted_cues: raw
                .weighted_cues
                .into_iter()
                .map(|item| (item.cue, item.weight))
                .collect(),
            cue_weight_adjustments: raw
                .cue_weight_adjustments
                .into_iter()
                .map(|item| (item.cue, item.weight))
                .collect(),
            suppress_generic: raw.suppress_generic,
        });
    }

    compiled.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(compiled)
}

fn padded_normalized(input: &str) -> String {
    format!(" {} ", crate::nl::normalize_text(input))
}

fn padded_rule_term(input: &str) -> String {
    let normalized = crate::nl::normalize_text(input);
    if normalized.starts_with(' ') || normalized.ends_with(' ') {
        normalized
    } else {
        format!(" {normalized} ")
    }
}
