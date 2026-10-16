//! Frozen-MiniLM intent classification shared by CueKey, ingestion jobs, and
//! the hybrid recall reranker.
//!
//! A tiny offline-trained linear head maps sentence embeddings to intent
//! scores. Runtime classification contains no semantic word or phrase rules.
//! Scores are ranking logits rather than calibrated probabilities; the stored
//! margin and confidence weight make uncertain classifications matter less.

use crate::semantic::SemanticEncoder;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;

pub const INTENT_TAXONOMY_VERSION: &str = "cuekey-intents-v4";
pub const INTENT_LABELS: [&str; 8] = [
    "preference",
    "decision",
    "standing_instruction",
    "personal_fact",
    "event_or_plan",
    "summary_or_timeline",
    "chitchat",
    "action_or_command",
];

const RECALL_LABELS: [&str; 6] = [
    "preference",
    "decision",
    "standing_instruction",
    "personal_fact",
    "event_or_plan",
    "summary_or_timeline",
];

const PROBE_MAGIC: &[u8; 8] = b"CMPINTP1";
const QINT8_PROBE: &[u8] =
    include_bytes!("../assets/all-MiniLM-L3-v2/intent_probe_qint8.head");
const Q4_PROBE: &[u8] = include_bytes!("../assets/all-MiniLM-L3-v2/intent_probe_q4.head");

/// Memory-type heads trained on WildChat user sentences (scheme v2 labels,
/// `evals/intent/wildchat/`). They classify memories sentence by sentence and are used
/// only when `SemanticConfig::memory_type_head` is on; queries always use the probes above.
const MEMORY_QINT8_PROBE: &[u8] =
    include_bytes!("../assets/all-MiniLM-L3-v2/memory_type_probe_qint8.head");
const MEMORY_Q4_PROBE: &[u8] =
    include_bytes!("../assets/all-MiniLM-L3-v2/memory_type_probe_q4.head");
/// Appended to the model version of classifications made by the memory-type head.
pub const MEMORY_TYPE_HEAD_VERSION: &str = "memtype-wildchat-v1";
/// The memory heads store logits times this scale, so a softmax at this temperature
/// gives their trained probabilities (the same temperature `intent_compatibility` uses).
const MEMORY_SOFTMAX_TEMPERATURE: f32 = 0.08;
/// A sentence decides a memory's type when one memory label reaches this probability.
const MEMORY_TYPE_MIN_PROBABILITY: f32 = 0.5;
/// Most sentences classified per memory.
pub const MEMORY_MAX_SENTENCES: usize = 24;

#[derive(Debug)]
struct IntentProbe {
    dimensions: usize,
    score_scale: f32,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl IntentProbe {
    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        const HEADER_BYTES: usize = 20;
        if bytes.len() < HEADER_BYTES || &bytes[..PROBE_MAGIC.len()] != PROBE_MAGIC {
            return Err("intent probe has an invalid header".to_string());
        }
        let dimensions = read_u32(bytes, 8)? as usize;
        let class_count = read_u32(bytes, 12)? as usize;
        let score_scale = read_f32(bytes, 16)?;
        if dimensions == 0 || class_count != INTENT_LABELS.len() {
            return Err("intent probe dimensions or class count are invalid".to_string());
        }
        if !score_scale.is_finite() || score_scale <= 0.0 {
            return Err("intent probe score scale is invalid".to_string());
        }
        let weight_count = dimensions
            .checked_mul(class_count)
            .ok_or_else(|| "intent probe dimensions overflow".to_string())?;
        let value_count = weight_count
            .checked_add(class_count)
            .ok_or_else(|| "intent probe value count overflow".to_string())?;
        let expected_bytes = value_count
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|size| size.checked_add(HEADER_BYTES))
            .ok_or_else(|| "intent probe byte count overflow".to_string())?;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "intent probe size mismatch: expected {expected_bytes}, received {}",
                bytes.len()
            ));
        }
        let mut values = Vec::with_capacity(value_count);
        for offset in (HEADER_BYTES..bytes.len()).step_by(std::mem::size_of::<f32>()) {
            let value = read_f32(bytes, offset)?;
            if !value.is_finite() {
                return Err("intent probe contains a non-finite value".to_string());
            }
            values.push(value);
        }
        let biases = values.split_off(weight_count);
        Ok(Self {
            dimensions,
            score_scale,
            weights: values,
            biases,
        })
    }

    fn scores(&self, embedding: &[f32]) -> Result<Vec<f32>, String> {
        if embedding.len() != self.dimensions {
            return Err(format!(
                "intent probe dimension mismatch: expected {}, received {}",
                self.dimensions,
                embedding.len()
            ));
        }
        let mut scores = Vec::with_capacity(INTENT_LABELS.len());
        for (class, weights) in self.weights.chunks_exact(self.dimensions).enumerate() {
            let score = weights
                .iter()
                .zip(embedding)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
            scores.push((score + self.biases[class]) * self.score_scale);
        }
        Ok(scores)
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "intent probe ended unexpectedly".to_string())?;
    Ok(u32::from_le_bytes(value.try_into().map_err(|_| {
        "intent probe contains an invalid integer".to_string()
    })?))
}

fn read_f32(bytes: &[u8], offset: usize) -> Result<f32, String> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "intent probe ended unexpectedly".to_string())?;
    Ok(f32::from_le_bytes(value.try_into().map_err(|_| {
        "intent probe contains an invalid float".to_string()
    })?))
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum IntentTarget {
    Query,
    Memory,
}

impl Default for IntentTarget {
    fn default() -> Self {
        Self::Query
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentClassification {
    pub primary_intent: String,
    pub scores: BTreeMap<String, f32>,
    pub top_intents: Vec<String>,
    pub top_score: f32,
    pub margin: f32,
    pub confidence_weight: f32,
    /// Independent query-side gate for whether CueKey should ask CueMap for
    /// recall. This is deliberately separate from `primary_intent`: a
    /// historical question can be semantically ambiguous without becoming a
    /// no-recall command or social response.
    #[serde(default)]
    pub recall_eligible: bool,
    pub recall_action: String,
    /// Whether this text is useful as durable memory evidence. For memory
    /// classifications this is the signal used by the hybrid reranker to
    /// distinguish ordinary evidence from social/imperative noise.
    pub memory_eligible: bool,
    pub model_version: String,
    pub taxonomy_version: String,
}

impl IntentClassification {
    pub fn is_recall_intent(&self) -> bool {
        // The action string keeps old persisted classifications readable while
        // the explicit field is populated by the current taxonomy.
        self.recall_eligible || self.recall_action == "recall"
    }

    pub fn score(&self, label: &str) -> f32 {
        self.scores.get(label).copied().unwrap_or(0.0)
    }

    /// Convert the ranking scores into a soft distribution for reranking.
    /// This is intentionally only a relative softmax, not a calibrated
    /// probability estimate.
    pub fn soft_distribution(&self, temperature: f32) -> BTreeMap<String, f32> {
        let temperature = temperature.max(0.001);
        let max_score = self
            .scores
            .values()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut values = BTreeMap::new();
        let mut total = 0.0f32;
        for label in INTENT_LABELS {
            let value = ((self.score(label) - max_score) / temperature).exp();
            values.insert(label.to_string(), value);
            total += value;
        }
        if total > 0.0 && total.is_finite() {
            for value in values.values_mut() {
                *value /= total;
            }
        }
        values
    }
}

pub struct IntentClassifier {
    encoder: Arc<dyn SemanticEncoder>,
    probe: IntentProbe,
    memory_probe: Option<IntentProbe>,
    /// Encoder for memory typing when it differs from `encoder` (a local typing model).
    memory_encoder: Option<Arc<dyn SemanticEncoder>>,
    /// Identifies the memory head and typing encoder in `memory_model_version`.
    memory_version: String,
    model_version: String,
}

impl IntentClassifier {
    pub fn new(
        encoder: Arc<dyn SemanticEncoder>,
        model_version: impl Into<String>,
    ) -> Result<Self, String> {
        let model_version = model_version.into();
        let bytes = match model_version.as_str() {
            "bundled-qint8-minilm-l3" => QINT8_PROBE,
            "bundled-q4-minilm-l3" => Q4_PROBE,
            _ => {
                return Err(format!(
                    "no trained intent probe is available for semantic model {model_version}"
                ))
            }
        };
        let probe = IntentProbe::from_bytes(bytes)?;
        if encoder.dimensions() != probe.dimensions {
            return Err(format!(
                "intent probe dimension mismatch: model expects {}, encoder provides {}",
                probe.dimensions,
                encoder.dimensions()
            ));
        }
        Ok(Self {
            encoder,
            probe,
            memory_probe: None,
            memory_encoder: None,
            memory_version: MEMORY_TYPE_HEAD_VERSION.to_string(),
            model_version,
        })
    }

    /// Adds a memory-type head from `head_bytes` (CMPINTP1), applied to the output of
    /// `encoder` (a separate typing model) or, when `None`, of the main encoder.
    /// `version` names this head/encoder pair in `memory_model_version`, so memories typed
    /// by another pair are reclassified.
    pub fn with_custom_memory_head(
        mut self,
        encoder: Option<Arc<dyn SemanticEncoder>>,
        head_bytes: &[u8],
        version: String,
    ) -> Result<Self, String> {
        let probe = IntentProbe::from_bytes(head_bytes)?;
        let dimensions = encoder.as_ref().map_or(self.encoder.dimensions(), |encoder| encoder.dimensions());
        if probe.dimensions != dimensions {
            return Err(format!(
                "memory-type head expects {} dimensions, the typing encoder provides {dimensions}",
                probe.dimensions
            ));
        }
        self.memory_probe = Some(probe);
        self.memory_encoder = encoder;
        self.memory_version = version;
        Ok(self)
    }

    /// Adds the memory-type head for this encoder (see [`Self::classify_memory_sentences`]).
    pub fn with_memory_head(mut self) -> Result<Self, String> {
        let bytes = match self.model_version.as_str() {
            "bundled-qint8-minilm-l3" => MEMORY_QINT8_PROBE,
            "bundled-q4-minilm-l3" => MEMORY_Q4_PROBE,
            other => {
                return Err(format!("no memory-type head is available for semantic model {other}"))
            }
        };
        let probe = IntentProbe::from_bytes(bytes)?;
        if probe.dimensions != self.probe.dimensions {
            return Err("memory-type head dimensions differ from the intent probe".to_string());
        }
        self.memory_probe = Some(probe);
        self.memory_encoder = None;
        self.memory_version = MEMORY_TYPE_HEAD_VERSION.to_string();
        Ok(self)
    }

    pub fn memory_head_enabled(&self) -> bool {
        self.memory_probe.is_some()
    }

    /// Model version stored on classifications made by the memory-type head.
    pub fn memory_model_version(&self) -> String {
        format!("{}+{}", self.model_version, self.memory_version)
    }

    /// Classifies a memory from its sentences with the memory-type head.
    ///
    /// The head was trained on single user sentences. When some sentence gives a memory
    /// label (preference … summary_or_timeline) at least `MEMORY_TYPE_MIN_PROBABILITY`,
    /// the memory takes that sentence's scores, so one standing instruction inside a long
    /// request still types the memory. Otherwise the memory gets the mean of the
    /// sentences' scores. `text` is used only for the query-side structural signals.
    pub fn classify_memory_sentences(
        &self,
        text: &str,
        sentences: &[String],
    ) -> Result<IntentClassification, String> {
        let probe = self
            .memory_probe
            .as_ref()
            .ok_or_else(|| "memory-type head is not enabled".to_string())?;
        if sentences.is_empty() {
            return Err("memory has no sentences to classify".to_string());
        }
        let mut best: Option<(f32, Vec<f32>)> = None;
        let mut mean = vec![0.0f32; INTENT_LABELS.len()];
        let mut count = 0usize;
        let encoder = self.memory_encoder.as_ref().unwrap_or(&self.encoder);
        for sentence in sentences.iter().take(MEMORY_MAX_SENTENCES) {
            let mut vector = encoder.encode(sentence)?;
            normalize(&mut vector)?;
            let scores = probe.scores(&vector)?;
            let probabilities = softmax(&scores, MEMORY_SOFTMAX_TEMPERATURE);
            let memory_probability = RECALL_LABELS
                .iter()
                .filter_map(|label| INTENT_LABELS.iter().position(|candidate| candidate == label))
                .map(|index| probabilities[index])
                .fold(0.0f32, f32::max);
            if memory_probability >= MEMORY_TYPE_MIN_PROBABILITY
                && best.as_ref().map_or(true, |(probability, _)| memory_probability > *probability)
            {
                best = Some((memory_probability, scores.clone()));
            }
            for (total, score) in mean.iter_mut().zip(&scores) {
                *total += score;
            }
            count += 1;
        }
        let scores = match best {
            Some((_, scores)) => scores,
            None => mean.into_iter().map(|total| total / count as f32).collect(),
        };
        Ok(self.build_classification(text, scores, IntentTarget::Memory, self.memory_model_version()))
    }

    pub fn classify(&self, text: &str, target: IntentTarget) -> Result<IntentClassification, String> {
        let vector = self.encoder.encode(text)?;
        self.classify_vector(text, &vector, target)
    }

    pub fn classify_with_embedding(
        &self,
        text: &str,
        embedding: &[f32],
        target: IntentTarget,
    ) -> Result<IntentClassification, String> {
        if embedding.len() != self.encoder.dimensions() {
            return Err(format!(
                "intent embedding dimension mismatch: expected {}, received {}",
                self.encoder.dimensions(),
                embedding.len()
            ));
        }
        self.classify_vector(text, embedding, target)
    }

    fn classify_vector(
        &self,
        text: &str,
        vector: &[f32],
        target: IntentTarget,
    ) -> Result<IntentClassification, String> {
        let mut normalized_vector = vector.to_vec();
        normalize(&mut normalized_vector)?;
        let scores = self.probe.scores(&normalized_vector)?;
        Ok(self.build_classification(text, scores, target, self.model_version.clone()))
    }

    fn build_classification(
        &self,
        text: &str,
        scores: Vec<f32>,
        target: IntentTarget,
        model_version: String,
    ) -> IntentClassification {
        let signals = structural_signals(text);

        let mut order = (0..INTENT_LABELS.len()).collect::<Vec<_>>();
        order.sort_unstable_by(|left, right| {
            scores[*right]
                .partial_cmp(&scores[*left])
                .unwrap_or(Ordering::Equal)
        });
        let top = order[0];
        let second = order[1];
        let top_score = scores[top];
        let margin = top_score - scores[second];
        let margin_weight = ((margin - 0.02) / 0.13).clamp(0.0, 1.0);
        let score_weight = ((top_score + 0.05) / 0.35).clamp(0.0, 1.0);
        let confidence_weight = (margin_weight * 0.75 + score_weight * 0.25).clamp(0.0, 1.0);
        let primary_intent = INTENT_LABELS[top].to_string();
        let recall_eligible = recall_eligible(&primary_intent, &signals, margin, target);
        let memory_eligible = memory_eligible(&primary_intent);
        let recall_action = if recall_eligible {
            "recall"
        } else {
            "no_recall"
        };
        let mut score_map = BTreeMap::new();
        for (index, label) in INTENT_LABELS.iter().enumerate() {
            score_map.insert((*label).to_string(), scores[index]);
        }

        IntentClassification {
            primary_intent,
            scores: score_map,
            top_intents: order
                .iter()
                .take(3)
                .map(|index| INTENT_LABELS[*index].to_string())
                .collect(),
            top_score,
            margin,
            confidence_weight,
            recall_eligible,
            recall_action: recall_action.to_string(),
            memory_eligible,
            model_version,
            taxonomy_version: INTENT_TAXONOMY_VERSION.to_string(),
        }
    }

}

/// Soft compatibility between a query intent distribution and a memory
/// intent distribution. Exact intent matches are intentionally strongest.
/// Only a small set of semantically adjacent types receive a secondary
/// match; unrelated types remain neutral so intent reranking rewards useful
/// matches instead of inflating the entire candidate slate. The query's
/// primary intent appearing in the memory's top three intents is also an
/// explicit directional signal because this is the useful retrieval-side
/// question: does this memory plausibly belong to the query's intent?
pub fn intent_compatibility(
    query: &IntentClassification,
    memory: &IntentClassification,
) -> f64 {
    let query_distribution = query.soft_distribution(0.08);
    let memory_distribution = memory.soft_distribution(0.08);
    let mut total = 0.0f32;
    for query_label in INTENT_LABELS {
        let query_value = query_distribution.get(query_label).copied().unwrap_or(0.0);
        for memory_label in INTENT_LABELS {
            let memory_value = memory_distribution.get(memory_label).copied().unwrap_or(0.0);
            total += query_value * memory_value * compatibility_weight(query_label, memory_label);
        }
    }
    let soft_compatibility = total.clamp(0.0, 1.0);
    let top3_compatibility = directional_top3_compatibility(query, memory);
    f64::from(soft_compatibility.max(top3_compatibility).clamp(0.0, 1.0))
}

fn directional_top3_compatibility(
    query: &IntentClassification,
    memory: &IntentClassification,
) -> f32 {
    match memory
        .top_intents
        .iter()
        .position(|intent| intent == &query.primary_intent)
    {
        Some(0) => 1.0,
        Some(1) => 0.70,
        Some(2) => 0.45,
        _ => 0.0,
    }
}

fn compatibility_weight(query: &str, memory: &str) -> f32 {
    if query == memory {
        return 1.0;
    }
    match (query, memory) {
        ("summary_or_timeline", "event_or_plan")
        | ("event_or_plan", "summary_or_timeline") => 0.45,
        ("summary_or_timeline", "decision")
        | ("decision", "summary_or_timeline") => 0.35,
        ("summary_or_timeline", "preference")
        | ("preference", "summary_or_timeline")
        | ("summary_or_timeline", "personal_fact")
        | ("personal_fact", "summary_or_timeline")
        | ("summary_or_timeline", "standing_instruction")
        | ("standing_instruction", "summary_or_timeline") => 0.25,
        ("event_or_plan", "decision")
        | ("decision", "event_or_plan")
        | ("decision", "standing_instruction")
        | ("standing_instruction", "decision") => 0.35,
        ("preference", "personal_fact") | ("personal_fact", "preference") => 0.40,
        _ => 0.0,
    }
}

#[derive(Debug, Clone, Copy)]
struct StructuralSignals {
    question_like: bool,
    continuation: bool,
}

const SHORT_MESSAGE_MAX_TOKENS: usize = 12;
const STRUCTURAL_RECALL_FALLBACK_MARGIN: f32 = 0.05;

fn structural_signals(text: &str) -> StructuralSignals {
    let lower = text.trim().to_ascii_lowercase();
    let tokens = lower
        .split(|character: char| !character.is_ascii_alphabetic())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let first = tokens.first().copied().unwrap_or_default();
    let question_like = text.contains('?')
        || matches!(
            first,
            "what"
                | "when"
                | "where"
                | "which"
                | "who"
                | "whom"
                | "whose"
                | "why"
                | "how"
                | "am"
                | "are"
                | "is"
                | "was"
                | "were"
                | "do"
                | "does"
                | "did"
                | "have"
                | "has"
                | "had"
                | "can"
                | "could"
                | "will"
                | "would"
                | "should"
                | "might"
                | "may"
                | "must"
        );
    let last = tokens.last().copied().unwrap_or_default();
    let short_message = tokens.len() <= SHORT_MESSAGE_MAX_TOKENS;
    let continuation = short_message
        && (text.contains("...")
            || text.contains('…')
            || matches!(
                last,
                "at" | "in" | "on" | "to" | "with" | "for" | "the" | "my" | "our"
            ));

    StructuralSignals {
        question_like,
        continuation,
    }
}

fn recall_eligible(
    primary_intent: &str,
    signals: &StructuralSignals,
    margin: f32,
    target: IntentTarget,
) -> bool {
    if RECALL_LABELS.contains(&primary_intent) {
        return true;
    }
    target == IntentTarget::Query
        && margin < STRUCTURAL_RECALL_FALLBACK_MARGIN
        && (signals.question_like || signals.continuation)
}

fn memory_eligible(primary_intent: &str) -> bool {
    RECALL_LABELS.contains(&primary_intent)
}

/// Model version of classifications made by the bundled memory-type head on
/// `base_model_version` (a custom head or typing encoder has its own version; see
/// [`IntentClassifier::memory_model_version`]).
pub fn memory_model_version(base_model_version: &str) -> String {
    format!("{base_model_version}+{MEMORY_TYPE_HEAD_VERSION}")
}

fn softmax(scores: &[f32], temperature: f32) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|score| ((score - max) / temperature).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.into_iter().map(|value| value / total).collect()
}

fn normalize(vector: &mut [f32]) -> Result<(), String> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err("intent vector has zero or invalid magnitude".to_string());
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AxisEncoder;

    impl SemanticEncoder for AxisEncoder {
        fn dimensions(&self) -> usize {
            INTENT_LABELS.len()
        }

        fn encode(&self, _text: &str) -> Result<Vec<f32>, String> {
            Err("test encoder requires an explicit embedding".to_string())
        }
    }

    fn intent_axis(index: usize) -> Vec<f32> {
        let mut vector = vec![0.0; INTENT_LABELS.len()];
        vector[index] = 1.0;
        vector
    }

    fn test_classifier() -> IntentClassifier {
        let mut weights = vec![0.0; INTENT_LABELS.len() * INTENT_LABELS.len()];
        for index in 0..INTENT_LABELS.len() {
            weights[index * INTENT_LABELS.len() + index] = 1.0;
        }
        IntentClassifier {
            encoder: Arc::new(AxisEncoder),
            probe: IntentProbe {
                dimensions: INTENT_LABELS.len(),
                score_scale: 1.0,
                weights,
                biases: vec![0.0; INTENT_LABELS.len()],
            },
            memory_probe: None,
            memory_encoder: None,
            memory_version: MEMORY_TYPE_HEAD_VERSION.to_string(),
            model_version: "test-model".to_string(),
        }
    }

    /// Encodes "rule …" sentences on the standing_instruction axis and everything else
    /// on the action_or_command axis.
    struct SentenceEncoder;

    impl SemanticEncoder for SentenceEncoder {
        fn dimensions(&self) -> usize {
            INTENT_LABELS.len()
        }

        fn encode(&self, text: &str) -> Result<Vec<f32>, String> {
            let label = if text.starts_with("rule") { "standing_instruction" } else { "action_or_command" };
            Ok(intent_axis(INTENT_LABELS.iter().position(|l| *l == label).unwrap()))
        }
    }

    fn memory_test_classifier() -> IntentClassifier {
        let identity = || {
            let mut weights = vec![0.0; INTENT_LABELS.len() * INTENT_LABELS.len()];
            for index in 0..INTENT_LABELS.len() {
                weights[index * INTENT_LABELS.len() + index] = 1.0;
            }
            IntentProbe { dimensions: INTENT_LABELS.len(), score_scale: 1.0, weights, biases: vec![0.0; INTENT_LABELS.len()] }
        };
        IntentClassifier {
            encoder: Arc::new(SentenceEncoder),
            probe: identity(),
            memory_probe: Some(identity()),
            memory_encoder: None,
            memory_version: MEMORY_TYPE_HEAD_VERSION.to_string(),
            model_version: "test-model".to_string(),
        }
    }

    #[test]
    fn a_custom_head_can_use_its_own_typing_encoder() {
        let identity_bytes = {
            let mut bytes = PROBE_MAGIC.to_vec();
            bytes.extend((INTENT_LABELS.len() as u32).to_le_bytes());
            bytes.extend((INTENT_LABELS.len() as u32).to_le_bytes());
            bytes.extend(1.0f32.to_le_bytes());
            for row in 0..INTENT_LABELS.len() {
                for column in 0..INTENT_LABELS.len() {
                    bytes.extend((if row == column { 1.0f32 } else { 0.0 }).to_le_bytes());
                }
            }
            for _ in 0..INTENT_LABELS.len() {
                bytes.extend(0.0f32.to_le_bytes());
            }
            bytes
        };
        // The main encoder only produces action vectors; the typing encoder sees rules.
        let classifier = test_classifier()
            .with_custom_memory_head(Some(Arc::new(SentenceEncoder)), &identity_bytes, "custom-abc".to_string())
            .unwrap();
        assert_eq!(classifier.memory_model_version(), "test-model+custom-abc");
        let typed = classifier.classify_memory_sentences("rule one", &["rule one".to_string()]).unwrap();
        assert_eq!(typed.primary_intent, "standing_instruction");
        assert_eq!(typed.model_version, "test-model+custom-abc");
        // A head whose width does not match the typing encoder is rejected.
        assert!(test_classifier().with_custom_memory_head(None, MEMORY_QINT8_PROBE, "x".to_string()).is_err());
    }

    #[test]
    fn bundled_memory_heads_are_well_formed() {
        for bytes in [MEMORY_QINT8_PROBE, MEMORY_Q4_PROBE] {
            let probe = IntentProbe::from_bytes(bytes).unwrap();
            assert_eq!(probe.dimensions, 384);
            assert_eq!(probe.biases.len(), INTENT_LABELS.len());
            assert!((probe.score_scale - MEMORY_SOFTMAX_TEMPERATURE).abs() < 1e-6);
        }
        let error = IntentClassifier::new(Arc::new(AxisEncoder), "unknown-model");
        assert!(error.is_err());
    }

    #[test]
    fn one_memory_sentence_types_the_whole_memory() {
        let classifier = memory_test_classifier();
        assert!(classifier.memory_head_enabled());
        let sentences = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let typed = classifier
            .classify_memory_sentences("ask one. rule two.", &sentences(&["ask one", "rule two", "ask three"]))
            .unwrap();
        assert_eq!(typed.primary_intent, "standing_instruction");
        assert!(typed.memory_eligible);
        assert_eq!(typed.model_version, "test-model+memtype-wildchat-v1");
        let plain = classifier
            .classify_memory_sentences("ask one. ask two.", &sentences(&["ask one", "ask two"]))
            .unwrap();
        assert_eq!(plain.primary_intent, "action_or_command");
        assert!(!plain.memory_eligible);
        assert!(classifier.classify_memory_sentences("", &[]).is_err());
        assert!(test_classifier().classify_memory_sentences("x", &sentences(&["rule x"])).is_err());
    }

    #[test]
    fn bundled_probes_are_well_formed() {
        for bytes in [QINT8_PROBE, Q4_PROBE] {
            let probe = IntentProbe::from_bytes(bytes).unwrap();
            assert_eq!(probe.dimensions, 384);
            assert_eq!(probe.biases.len(), INTENT_LABELS.len());
            assert_eq!(
                probe.weights.len(),
                probe.dimensions * INTENT_LABELS.len()
            );
        }
    }

    #[test]
    fn classifier_requires_a_probe_for_the_exact_encoder_version() {
        let error = IntentClassifier::new(Arc::new(AxisEncoder), "unknown-model")
            .err()
            .unwrap();
        assert!(error.contains("no trained intent probe"));
    }

    #[cfg(feature = "semantic-encoder")]
    #[test]
    fn bundled_encoder_and_probe_classify_unseen_sentences() {
        let config = crate::semantic::SemanticConfig::default().resolved();
        let encoder = crate::semantic::load_configured_encoder(&config)
            .unwrap()
            .unwrap();
        let classifier = IntentClassifier::new(encoder, config.model_version).unwrap();

        for (text, expected) in [
            ("Could you lint this crate now?", "action_or_command"),
            ("That explanation was lovely, cheers.", "chitchat"),
            ("Which option did the team ultimately approve?", "decision"),
        ] {
            let classification = classifier.classify(text, IntentTarget::Query).unwrap();
            assert_eq!(classification.primary_intent, expected);
        }
    }

    #[test]
    fn surface_vocabulary_cannot_override_the_embedding_category() {
        let classifier = test_classifier();

        for (text, semantic_intent) in [
            ("Run tests and commit the changes", 0),
            ("Thanks, got it and understood", 1),
            ("We decided and settled on this", 4),
        ] {
            let classification = classifier
                .classify_with_embedding(text, &intent_axis(semantic_intent), IntentTarget::Query)
                .unwrap();
            assert_eq!(classification.primary_intent, INTENT_LABELS[semantic_intent]);
            assert_eq!(classification.top_score, 1.0);
            assert_eq!(classification.margin, 1.0);
        }
    }

    #[test]
    fn structural_shape_only_admits_uncertain_queries() {
        let classifier = test_classifier();
        let mut uncertain_no_recall = vec![0.0; INTENT_LABELS.len()];
        uncertain_no_recall[6] = 0.71;
        uncertain_no_recall[7] = 0.70;

        let question = classifier
            .classify_with_embedding(
                "Could this be relevant?",
                &uncertain_no_recall,
                IntentTarget::Query,
            )
            .unwrap();
        assert_eq!(question.primary_intent, "chitchat");
        assert!(question.recall_eligible);
        assert!(!question.memory_eligible);

        let statement = classifier
            .classify_with_embedding(
                "This may be relevant.",
                &uncertain_no_recall,
                IntentTarget::Query,
            )
            .unwrap();
        assert!(!statement.recall_eligible);

        let memory = classifier
            .classify_with_embedding(
                "Could this be relevant?",
                &uncertain_no_recall,
                IntentTarget::Memory,
            )
            .unwrap();
        assert!(!memory.recall_eligible);
        assert!(!memory.memory_eligible);

        let confident_question = classifier
            .classify_with_embedding(
                "How are you?",
                &intent_axis(6),
                IntentTarget::Query,
            )
            .unwrap();
        assert!(!confident_question.recall_eligible);
    }

    #[test]
    fn continuation_shape_is_short_and_token_bounded() {
        assert!(structural_signals("an unfinished thought...").continuation);
        assert!(structural_signals("the answer depends on").continuation);
        assert!(!structural_signals("aspirin").continuation);

        let long_fragment = format!("{}...", vec!["word"; SHORT_MESSAGE_MAX_TOKENS + 1].join(" "));
        assert!(!structural_signals(&long_fragment).continuation);
    }

    #[test]
    fn embedding_path_rejects_incompatible_dimensions() {
        let classifier = test_classifier();
        let error = classifier
            .classify_with_embedding("query", &[1.0], IntentTarget::Query)
            .unwrap_err();
        assert!(error.contains("dimension mismatch"));
    }

    fn hard_intent_classification(primary_intent: &str) -> IntentClassification {
        let mut scores = BTreeMap::new();
        for label in INTENT_LABELS {
            scores.insert(label.to_string(), if label == primary_intent { 1.0 } else { 0.0 });
        }
        IntentClassification {
            primary_intent: primary_intent.to_string(),
            scores,
            top_intents: vec![primary_intent.to_string()],
            top_score: 1.0,
            margin: 1.0,
            confidence_weight: 1.0,
            recall_eligible: true,
            recall_action: "recall".to_string(),
            memory_eligible: true,
            model_version: "test-model".to_string(),
            taxonomy_version: INTENT_TAXONOMY_VERSION.to_string(),
        }
    }

    #[test]
    fn intent_compatibility_prioritizes_exact_matches() {
        let query = hard_intent_classification("preference");
        let exact = hard_intent_classification("preference");
        let related = hard_intent_classification("personal_fact");
        let unrelated = hard_intent_classification("event_or_plan");

        let exact_score = intent_compatibility(&query, &exact);
        let related_score = intent_compatibility(&query, &related);
        let unrelated_score = intent_compatibility(&query, &unrelated);

        assert!(exact_score > 0.99);
        assert!(related_score > 0.30 && related_score < 0.50);
        assert!(unrelated_score < 0.05);
        assert!(exact_score > related_score);
        assert!(related_score > unrelated_score);
    }

    #[test]
    fn intent_compatibility_uses_directional_memory_top_three() {
        let query = hard_intent_classification("preference");
        let mut second_place = hard_intent_classification("event_or_plan");
        second_place.top_intents = vec![
            "event_or_plan".to_string(),
            "preference".to_string(),
            "decision".to_string(),
        ];
        let mut third_place = hard_intent_classification("event_or_plan");
        third_place.top_intents = vec![
            "event_or_plan".to_string(),
            "decision".to_string(),
            "preference".to_string(),
        ];
        let unrelated = hard_intent_classification("event_or_plan");

        let second_score = intent_compatibility(&query, &second_place);
        let third_score = intent_compatibility(&query, &third_place);
        let unrelated_score = intent_compatibility(&query, &unrelated);

        assert!((second_score - 0.70).abs() < 0.001);
        assert!((third_score - 0.45).abs() < 0.001);
        assert!(second_score > third_score);
        assert!(third_score > unrelated_score);
    }
}
