//! Semantic retrieval primitives.
//!
//! This module intentionally contains no language ontology and no text
//! classification rules. It operates on vectors supplied by an embedding
//! provider. The default build bundles a local MiniLM-L3 encoder, while
//! callers can disable automatic encoding or compile without the encoder.

use half::f16;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const DEFAULT_RANDOM_SEED: u64 = 0x4355_454d_4150_7637;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SemanticProfile {
    Off,
    Edge,
    Balanced,
    Quality,
}

impl Default for SemanticProfile {
    fn default() -> Self {
        Self::Quality
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SemanticStorage {
    Auto,
    F32,
    F16,
    Int8,
}

impl Default for SemanticStorage {
    fn default() -> Self {
        Self::Auto
    }
}

impl SemanticStorage {
    pub fn byte_width(self) -> usize {
        match self {
            Self::Auto | Self::F32 => 4,
            Self::F16 => 2,
            Self::Int8 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SemanticIndexMode {
    Auto,
    Exact,
    Ann,
}

/// Selects which query signal is allowed to drive recall.
///
/// `Hybrid` keeps lexical candidate discovery bounded by the requested limit
/// and uses the local embedding only to rerank those candidates. `Semantic`
/// accepts query text so a configured local encoder can embed it, then uses
/// semantic candidate discovery without lexical query cues.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SemanticRecallMode {
    Lexical,
    Semantic,
    Hybrid,
}

impl Default for SemanticRecallMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

/// Local text encoder interface. Implementations must be deterministic for a
/// fixed model asset and must not perform network I/O.
pub trait SemanticEncoder: Send + Sync {
    fn dimensions(&self) -> usize;
    fn encode(&self, text: &str) -> Result<Vec<f32>, String>;
}

impl Default for SemanticIndexMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SemanticConfig {
    /// Selects a coherent device-oriented set of semantic defaults.
    pub profile: SemanticProfile,
    /// Enables vector indexing and vector candidate discovery.
    pub enabled: bool,
    /// Zero means infer the dimension from the first vector.
    pub dimensions: usize,
    /// Optional identifier for the local embedding provider that produced the
    /// vectors. CueMap does not load or call the provider.
    pub model_id: String,
    /// Optional provider/model compatibility marker.
    pub model_version: String,
    /// Enables local text-to-vector inference inside the Rust process.
    pub encoder_enabled: bool,
    /// Local ONNX model path. Empty selects the bundled MiniLM asset when the
    /// semantic encoder feature is present. No runtime download is attempted.
    pub model_path: String,
    /// Local Hugging Face tokenizer JSON path. Empty selects the bundled
    /// MiniLM tokenizer when the semantic encoder feature is present.
    pub tokenizer_path: String,
    /// Maximum tokenizer sequence length. Both bundled L3 variants use 128
    /// word pieces by default.
    pub max_tokens: usize,
    /// ONNX Runtime intra-op threads. Zero uses the runtime default.
    pub encoder_threads: usize,
    /// Enables the CoreML execution provider on Apple targets when the
    /// bundled ONNX Runtime was built with CoreML support. Non-Apple targets
    /// ignore this setting.
    pub coreml_enabled: bool,
    /// Representation used for vectors persisted with memories.
    pub storage: SemanticStorage,
    /// Selects exact candidate discovery, ANN candidate discovery, or an
    /// automatic choice based on index size.
    pub index: SemanticIndexMode,
    /// Approximate memory budget for compact vectors and ANN bookkeeping.
    /// Zero means no budget is imposed.
    pub max_memory_mb: usize,
    /// Number of independent random-projection hash tables.
    pub ann_tables: usize,
    /// Number of hyperplanes per table. Must be <= 63.
    pub ann_bits: usize,
    /// Number of one-bit neighboring buckets to probe in addition to the
    /// exact bucket for each table.
    pub ann_probes: usize,
    /// Maximum number of vector candidates passed into lexical/rerank merge.
    pub candidate_limit: usize,
    /// For small indexes, exact cosine scan avoids poor recall while the
    /// index is still warming up. Larger indexes use the ANN buckets.
    pub exact_fallback_max: usize,
    /// Multiplier applied to cosine similarity before merge/reranking.
    pub semantic_score_multiplier: f64,
    /// Weight of normalized semantic similarity in bounded hybrid reranking.
    /// Zero preserves lexical ordering; one uses semantic ordering among the
    /// existing lexical candidates. This does not affect semantic candidate
    /// discovery mode.
    pub semantic_rerank_weight: f64,
    /// Maximum lexical candidate window passed into bounded hybrid semantic
    /// reranking. The final request limit is applied after this window is
    /// reranked. Zero uses the default window.
    pub semantic_rerank_candidate_limit: usize,
    /// Number of query text embeddings retained in the bounded in-memory
    /// cache. Zero disables query embedding caching.
    pub query_embedding_cache_capacity: usize,
    /// Enables confidence-weighted intent compatibility during hybrid
    /// reranking.
    pub intent_rerank_enabled: bool,
    /// Maximum fraction of the lexical score range contributed by a matching
    /// intent. Unrelated intent pairs contribute no positive compatibility.
    pub intent_rerank_weight: f64,
    /// Maximum fraction of the lexical score range used to penalize a
    /// confidently non-memory candidate.
    pub intent_no_recall_penalty: f64,
    /// Absolute cap on the score delta contributed by intent reranking. This
    /// keeps even a strong exact-intent match from overwhelming lexical and
    /// semantic evidence.
    pub intent_rerank_max_delta: f64,
    /// Enables the optional linear reranker.
    pub reranker_enabled: bool,
    /// Bias for the tiny binary-shippable linear reranker.
    pub reranker_bias: f32,
    /// Feature weights for the tiny linear reranker. Empty means neutral.
    pub reranker_weights: Vec<f32>,
    /// Scales the reranker contribution before adding it to the base score.
    pub reranker_scale: f64,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        let encoder_enabled = cfg!(feature = "semantic-encoder");
        Self {
            profile: SemanticProfile::Quality,
            enabled: encoder_enabled,
            dimensions: 0,
            model_id: "all-MiniLM-L3-v2".to_string(),
            model_version: "bundled-qint8-minilm-l3".to_string(),
            encoder_enabled,
            model_path: String::new(),
            tokenizer_path: String::new(),
            max_tokens: 128,
            encoder_threads: 0,
            coreml_enabled: true,
            storage: SemanticStorage::Auto,
            index: SemanticIndexMode::Auto,
            max_memory_mb: 0,
            ann_tables: 0,
            ann_bits: 0,
            ann_probes: 0,
            candidate_limit: 0,
            exact_fallback_max: 0,
            semantic_score_multiplier: 100.0,
            semantic_rerank_weight: 0.60,
            semantic_rerank_candidate_limit: 200,
            query_embedding_cache_capacity: 256,
            intent_rerank_enabled: true,
            intent_rerank_weight: 0.65,
            intent_no_recall_penalty: 0.20,
            intent_rerank_max_delta: 64.0,
            reranker_enabled: false,
            reranker_bias: 0.0,
            reranker_weights: Vec::new(),
            reranker_scale: 1.0,
        }
    }
}

impl SemanticConfig {
    /// Resolves a profile and fills zero/auto fields with stable defaults.
    /// Explicit non-zero values remain authoritative, which lets a caller
    /// tune dimensions, ANN fanout, and memory limits independently.
    pub fn resolved(&self) -> Self {
        let (profile_dimensions, profile_storage, profile_max_memory_mb, profile_tables,
            profile_bits, profile_probes, profile_candidates, profile_exact) = match self.profile {
            SemanticProfile::Off => (0, SemanticStorage::F32, 0, 4, 12, 2, 256, 4096),
            // Both bundled MiniLM variants emit 384-dimensional vectors. Edge
            // saves memory with the q4 L3 model and compact storage rather
            // than projecting the model output to an incompatible dimension.
            SemanticProfile::Edge => (384, SemanticStorage::Int8, 32, 4, 10, 2, 128, 4096),
            SemanticProfile::Balanced => {
                (384, SemanticStorage::Int8, 64, 4, 12, 2, 256, 4096)
            }
            SemanticProfile::Quality => {
                (384, SemanticStorage::Int8, 256, 6, 14, 3, 512, 4096)
            }
        };

        let mut resolved = self.clone();
        if resolved.profile == SemanticProfile::Edge
            && resolved.model_path.trim().is_empty()
            && resolved.model_id == "all-MiniLM-L3-v2"
            && resolved.model_version == "bundled-qint8-minilm-l3"
        {
            resolved.model_version = "bundled-q4-minilm-l3".to_string();
        }
        if resolved.profile == SemanticProfile::Edge
            && (resolved.max_tokens == 0 || resolved.max_tokens == 256)
        {
            resolved.max_tokens = 128;
        }
        if self.profile != SemanticProfile::Off || self.encoder_enabled {
            resolved.enabled = true;
        }
        if resolved.dimensions == 0 {
            resolved.dimensions = profile_dimensions;
        }
        if resolved.storage == SemanticStorage::Auto {
            resolved.storage = profile_storage;
        }
        if resolved.max_memory_mb == 0 {
            resolved.max_memory_mb = profile_max_memory_mb;
        }
        if resolved.ann_tables == 0 {
            resolved.ann_tables = profile_tables;
        }
        if resolved.ann_bits == 0 {
            resolved.ann_bits = profile_bits;
        }
        if resolved.ann_probes == 0 {
            resolved.ann_probes = profile_probes;
        }
        if resolved.candidate_limit == 0 {
            resolved.candidate_limit = profile_candidates;
        }
        if resolved.exact_fallback_max == 0 {
            resolved.exact_fallback_max = profile_exact;
        }
        if resolved.max_tokens == 0 {
            resolved.max_tokens = if resolved.profile == SemanticProfile::Edge {
                128
            } else {
                256
            };
        }
        resolved.ann_tables = resolved.ann_tables.max(1);
        resolved.ann_bits = resolved.ann_bits.clamp(1, 63);
        resolved.candidate_limit = resolved.candidate_limit.max(1);
        resolved.semantic_score_multiplier = if resolved.semantic_score_multiplier.is_finite() {
            resolved.semantic_score_multiplier
        } else {
            100.0
        };
        resolved.semantic_rerank_weight = if resolved.semantic_rerank_weight.is_finite() {
            resolved.semantic_rerank_weight.clamp(0.0, 1.0)
        } else {
            0.60
        };
        if resolved.semantic_rerank_candidate_limit == 0 {
            resolved.semantic_rerank_candidate_limit = 200;
        }
        resolved.intent_rerank_weight = if resolved.intent_rerank_weight.is_finite() {
            resolved.intent_rerank_weight.clamp(0.0, 1.0)
        } else {
            0.65
        };
        resolved.intent_no_recall_penalty = if resolved.intent_no_recall_penalty.is_finite() {
            resolved.intent_no_recall_penalty.clamp(0.0, 1.0)
        } else {
            0.20
        };
        resolved.intent_rerank_max_delta = if resolved.intent_rerank_max_delta.is_finite() {
            resolved.intent_rerank_max_delta.max(0.0)
        } else {
            64.0
        };
        resolved
    }

    pub fn estimated_vector_bytes(&self) -> usize {
        self.resolved().storage.byte_width()
    }

    /// Conservative resident-memory estimate for compact vectors, signatures,
    /// bucket IDs, and projection planes. It is intentionally a budget guard,
    /// not a byte-perfect allocator report.
    pub fn estimated_memory_bytes(&self, memory_count: usize) -> usize {
        let config = self.resolved();
        estimate_memory_bytes(&config, config.dimensions, memory_count)
    }

    pub fn estimated_memory_bytes_for_dimensions(
        &self,
        dimensions: usize,
        memory_count: usize,
    ) -> usize {
        let config = self.resolved();
        estimate_memory_bytes(&config, dimensions, memory_count)
    }

    pub fn within_memory_budget_for_dimensions(
        &self,
        dimensions: usize,
        memory_count: usize,
    ) -> bool {
        let config = self.resolved();
        config.max_memory_mb == 0
            || estimate_memory_bytes(&config, dimensions, memory_count)
                <= config.max_memory_mb.saturating_mul(1024 * 1024)
    }

    pub fn within_memory_budget(&self, memory_count: usize) -> bool {
        let config = self.resolved();
        config.max_memory_mb == 0
            || estimate_memory_bytes(&config, config.dimensions, memory_count)
                <= config.max_memory_mb.saturating_mul(1024 * 1024)
    }
}

pub fn load_configured_encoder(
    config: &SemanticConfig,
) -> Result<Option<Arc<dyn SemanticEncoder>>, String> {
    let config = config.resolved();
    if !config.encoder_enabled {
        return Ok(None);
    }

    #[cfg(feature = "semantic-encoder")]
    {
        let encoder = crate::semantic_encoder::OnnxSemanticEncoder::from_config(&config)?;
        return Ok(Some(Arc::new(encoder)));
    }

    #[cfg(not(feature = "semantic-encoder"))]
    {
        Err("semantic encoder is configured but this binary was built without the 'semantic-encoder' feature".to_string())
    }
}

fn estimate_memory_bytes(
    config: &SemanticConfig,
    dimensions: usize,
    memory_count: usize,
) -> usize {
    if memory_count == 0 || dimensions == 0 || !config.enabled {
        return 0;
    }
    let vector_bytes = dimensions.saturating_mul(config.storage.byte_width());
    let per_memory_index_bytes = config
        .ann_tables
        .saturating_mul(std::mem::size_of::<u64>() + std::mem::size_of::<u32>())
        .saturating_add(32);
    let plane_bytes = config
        .ann_tables
        .saturating_mul(config.ann_bits)
        .saturating_mul(dimensions)
        .saturating_mul(std::mem::size_of::<f32>());
    memory_count
        .saturating_mul(vector_bytes.saturating_add(per_memory_index_bytes))
        .saturating_add(plane_bytes)
}

/// Compact persisted representation of a normalized embedding. The public
/// ingest API still accepts `Vec<f32>`; this is the representation retained by
/// memories and used to rebuild the ANN buckets after a snapshot restore.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StoredSemanticVector {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Int8 { values: Vec<i8>, scale: f32 },
}

impl StoredSemanticVector {
    pub fn from_f32(vector: &[f32], storage: SemanticStorage) -> Result<Self, String> {
        let normalized = normalize_vector(vector)?;
        let storage = if storage == SemanticStorage::Auto {
            SemanticStorage::F32
        } else {
            storage
        };
        Ok(match storage {
            SemanticStorage::F32 => Self::F32(normalized),
            SemanticStorage::F16 => Self::F16(
                normalized
                    .into_iter()
                    .map(|value| f16::from_f32(value).to_bits())
                    .collect(),
            ),
            SemanticStorage::Int8 => {
                let scale = 1.0 / 127.0;
                Self::Int8 {
                    values: normalized
                        .into_iter()
                        .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8)
                        .collect(),
                    scale,
                }
            }
            SemanticStorage::Auto => unreachable!("auto storage is resolved above"),
        })
    }

    pub fn dimensions(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(values) => values.len(),
            Self::Int8 { values, .. } => values.len(),
        }
    }

    pub fn storage(&self) -> SemanticStorage {
        match self {
            Self::F32(_) => SemanticStorage::F32,
            Self::F16(_) => SemanticStorage::F16,
            Self::Int8 { .. } => SemanticStorage::Int8,
        }
    }

    pub fn estimated_bytes(&self) -> usize {
        match self {
            Self::F32(values) => values.len() * std::mem::size_of::<f32>(),
            Self::F16(values) => values.len() * std::mem::size_of::<u16>(),
            Self::Int8 { values, .. } => values.len() * std::mem::size_of::<i8>() + size_of_f32(),
        }
    }

    pub fn normalized_values(&self) -> Vec<f32> {
        match self {
            Self::F32(values) => values.clone(),
            Self::F16(values) => values
                .iter()
                .map(|value| f16::from_bits(*value).to_f32())
                .collect(),
            Self::Int8 { values, scale } => values
                .iter()
                .map(|value| *value as f32 * *scale)
                .collect(),
        }
    }

    pub fn normalized_query(query: &[f32]) -> Result<Vec<f32>, String> {
        normalize_vector(query)
    }

    pub fn cosine_similarity(&self, query: &[f32]) -> Result<f32, String> {
        let query = Self::normalized_query(query)?;
        self.cosine_similarity_normalized(&query)
    }

    pub fn cosine_similarity_normalized(&self, query: &[f32]) -> Result<f32, String> {
        if query.len() != self.dimensions() {
            return Err(format!(
                "semantic query dimension mismatch: expected {}, received {}",
                self.dimensions(),
                query.len()
            ));
        }
        Ok(match self {
            Self::F32(values) => dot(values, query),
            Self::F16(values) => values
                .iter()
                .zip(query)
                .map(|(value, query_value)| f16::from_bits(*value).to_f32() * query_value)
                .sum(),
            Self::Int8 { values, scale } => values
                .iter()
                .zip(query)
                .map(|(value, query_value)| *value as f32 * *scale * query_value)
                .sum(),
        })
    }
}

fn size_of_f32() -> usize {
    std::mem::size_of::<f32>()
}

#[derive(Clone, Debug)]
struct ProjectionTable {
    hyperplanes: Vec<Vec<f32>>,
    buckets: HashMap<u64, Vec<u32>>,
}

/// In-memory approximate nearest-neighbor index based on random-hyperplane
/// locality-sensitive hashing. It stores only IDs, signatures, and buckets;
/// the compact vector remains owned by the corresponding memory.
#[derive(Clone, Debug)]
pub struct SemanticIndex {
    config: SemanticConfig,
    dimensions: Option<usize>,
    tables: Vec<ProjectionTable>,
    indexed_ids: HashSet<u32>,
    signatures: HashMap<u32, Vec<u64>>,
}

impl SemanticIndex {
    pub fn new(config: SemanticConfig) -> Self {
        Self {
            config: config.resolved(),
            dimensions: None,
            tables: Vec::new(),
            indexed_ids: HashSet::new(),
            signatures: HashMap::new(),
        }
    }

    pub fn config(&self) -> &SemanticConfig {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.indexed_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indexed_ids.is_empty()
    }

    pub fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }

    pub fn clear(&mut self) {
        self.dimensions = None;
        self.tables.clear();
        self.indexed_ids.clear();
        self.signatures.clear();
    }

    pub fn rebuild<I>(&mut self, vectors: I)
    where
        I: IntoIterator<Item = (u32, StoredSemanticVector)>,
    {
        self.clear();
        for (memory_id, vector) in vectors {
            if let Err(error) = self.insert(memory_id, &vector) {
                tracing::debug!(memory_id, error = %error, "Skipping semantic vector during index rebuild");
            }
        }
    }

    pub fn insert(
        &mut self,
        memory_id: u32,
        vector: &StoredSemanticVector,
    ) -> Result<(), String> {
        if !self.config.enabled {
            return Ok(());
        }
        let normalized = vector.normalized_values();
        let dimensions = normalized.len();
        if dimensions == 0 {
            return Err("semantic vectors cannot be empty".to_string());
        }
        if let Some(expected) = self.dimensions {
            if expected != dimensions {
                return Err(format!(
                    "semantic vector dimension mismatch: expected {}, received {}",
                    expected, dimensions
                ));
            }
        } else {
            let configured = self.config.dimensions;
            if configured != 0 && configured != dimensions {
                return Err(format!(
                    "semantic vector dimension mismatch: configured {}, received {}",
                    configured, dimensions
                ));
            }
        }

        let projected_count = if self.indexed_ids.contains(&memory_id) {
            self.indexed_ids.len()
        } else {
            self.indexed_ids.len().saturating_add(1)
        };
        if !self
            .config
            .within_memory_budget_for_dimensions(dimensions, projected_count)
        {
            return Err("semantic index memory budget exceeded".to_string());
        }

        if self.dimensions.is_none() {
            self.initialize_tables(dimensions);
        }

        self.remove(memory_id);
        let signatures = self
            .tables
            .iter_mut()
            .map(|table| {
                let signature = signature(&table.hyperplanes, &normalized);
                table.buckets.entry(signature).or_default().push(memory_id);
                signature
            })
            .collect::<Vec<_>>();
        self.indexed_ids.insert(memory_id);
        self.signatures.insert(memory_id, signatures);
        Ok(())
    }

    pub fn remove(&mut self, memory_id: u32) -> bool {
        let existed = self.indexed_ids.remove(&memory_id);
        if !existed {
            return false;
        }
        if let Some(signatures) = self.signatures.remove(&memory_id) {
            for (table, table_signature) in self.tables.iter_mut().zip(signatures) {
                if let Some(ids) = table.buckets.get_mut(&table_signature) {
                    ids.retain(|id| *id != memory_id);
                    if ids.is_empty() {
                        table.buckets.remove(&table_signature);
                    }
                }
            }
        }
        true
    }

    /// Returns IDs that should be scored against the compact vectors stored by
    /// the engine. Exact mode is still bounded by `limit`; ANN mode is bounded
    /// after bucket union so query work remains predictable.
    pub fn query_candidate_ids(&self, query: &[f32], limit: usize) -> Result<Vec<u32>, String> {
        if !self.config.enabled || limit == 0 || self.indexed_ids.is_empty() {
            return Ok(Vec::new());
        }
        let query = normalize_vector(query)?;
        if self.dimensions != Some(query.len()) {
            return Err(format!(
                "semantic query dimension mismatch: expected {:?}, received {}",
                self.dimensions,
                query.len()
            ));
        }

        let exact = self.config.index == SemanticIndexMode::Exact
            || (self.config.index == SemanticIndexMode::Auto
                && self.indexed_ids.len() <= self.config.exact_fallback_max);
        let mut candidate_ids = HashSet::new();
        if exact {
            candidate_ids.extend(self.indexed_ids.iter().copied());
        } else {
            for table in &self.tables {
                let exact_signature = signature(&table.hyperplanes, &query);
                for bucket_key in probe_keys(
                    exact_signature,
                    self.config.ann_bits,
                    self.config.ann_probes,
                ) {
                    if let Some(ids) = table.buckets.get(&bucket_key) {
                        candidate_ids.extend(ids.iter().copied());
                    }
                }
            }
            // Empty/very sparse ANN buckets should not turn semantic recall
            // into a silent hard miss. Bound the emergency scan and leave the
            // normal large-index path approximate.
            if candidate_ids.is_empty() {
                candidate_ids.extend(self.indexed_ids.iter().copied().take(limit));
            }
        }

        let mut candidate_ids = candidate_ids.into_iter().collect::<Vec<_>>();
        candidate_ids.sort_unstable();
        candidate_ids.truncate(limit.min(self.config.candidate_limit.max(1)));
        Ok(candidate_ids)
    }

    fn initialize_tables(&mut self, dimensions: usize) {
        self.dimensions = Some(dimensions);
        let table_count = self.config.ann_tables.max(1);
        let bit_count = self.config.ann_bits.clamp(1, 63);
        let mut rng = SplitMix64::new(DEFAULT_RANDOM_SEED ^ dimensions as u64);
        self.tables = (0..table_count)
            .map(|_| {
                let hyperplanes = (0..bit_count)
                    .map(|_| {
                        let mut plane = (0..dimensions)
                            .map(|_| rng.next_f32() * 2.0 - 1.0)
                            .collect::<Vec<_>>();
                        let norm = plane.iter().map(|value| value * value).sum::<f32>().sqrt();
                        if norm > f32::EPSILON {
                            for value in &mut plane {
                                *value /= norm;
                            }
                        }
                        plane
                    })
                    .collect();
                ProjectionTable {
                    hyperplanes,
                    buckets: HashMap::new(),
                }
            })
            .collect();
    }
}

fn normalize_vector(vector: &[f32]) -> Result<Vec<f32>, String> {
    if vector.is_empty() {
        return Err("semantic vectors cannot be empty".to_string());
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err("semantic vectors must contain only finite values".to_string());
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return Err("semantic vectors cannot have zero magnitude".to_string());
    }
    Ok(vector.iter().map(|value| value / norm).collect())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn signature(hyperplanes: &[Vec<f32>], vector: &[f32]) -> u64 {
    hyperplanes
        .iter()
        .enumerate()
        .fold(0u64, |value, (bit, plane)| {
            if dot(plane, vector) >= 0.0 {
                value | (1u64 << bit)
            } else {
                value
            }
        })
}

fn probe_keys(signature: u64, bits: usize, probes: usize) -> Vec<u64> {
    let bits = bits.min(63);
    let mut keys = Vec::with_capacity(probes.saturating_add(1).min(bits + 1));
    keys.push(signature);
    for bit in 0..bits {
        if keys.len() >= probes.saturating_add(1) {
            break;
        }
        keys.push(signature ^ (1u64 << bit));
    }
    keys
}

#[derive(Clone, Debug)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}

/// A tiny linear reranker model. Its weights are deliberately data, not
/// ontology rules, so a trained model can be shipped as a handful of floats.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearReranker {
    pub bias: f32,
    pub weights: Vec<f32>,
}

impl LinearReranker {
    pub fn from_config(config: &SemanticConfig) -> Self {
        Self {
            bias: config.reranker_bias,
            weights: config.reranker_weights.clone(),
        }
    }

    pub fn score(&self, features: &[f32]) -> f32 {
        self.bias
            + self
                .weights
                .iter()
                .zip(features)
                .map(|(weight, feature)| weight * feature)
                .sum::<f32>()
    }
}

#[cfg(test)]
#[path = "../tests/unit/semantic.rs"]
mod tests;
