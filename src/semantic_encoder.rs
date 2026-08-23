//! Local ONNX encoder for the bundled qint8 MiniLM-L3 default and q4
//! MiniLM-L3 edge asset.
//!
//! This module is compiled with the default `semantic-encoder` feature. It
//! never downloads models: empty paths select the assets embedded in the
//! binary, while non-empty paths provide an explicit local override.

use crate::semantic::{SemanticConfig, SemanticEncoder};
use ort::{session::Session, value::Tensor};
use std::sync::Mutex;
use tokenizers::{
    tokenizer::TruncationDirection,
    Tokenizer, TruncationParams, TruncationStrategy,
};

const DEFAULT_DIMENSIONS: usize = 384;
const EMBEDDED_L3_MODEL: &[u8] =
    include_bytes!("../assets/all-MiniLM-L3-v2/model_qint8_arm64.onnx");
const EMBEDDED_L3_Q4_MODEL: &[u8] =
    include_bytes!("../assets/all-MiniLM-L3-v2/model_int4.onnx");
const EMBEDDED_L3_TOKENIZER: &[u8] =
    include_bytes!("../assets/all-MiniLM-L3-v2/tokenizer.json");

pub struct OnnxSemanticEncoder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dimensions: usize,
    max_tokens: usize,
}

impl OnnxSemanticEncoder {
    pub fn from_config(config: &SemanticConfig) -> Result<Self, String> {
        let use_edge_model = config.model_path.trim().is_empty()
            && config.model_version == "bundled-q4-minilm-l3";
        let mut tokenizer = if config.tokenizer_path.trim().is_empty() {
            Tokenizer::from_bytes(EMBEDDED_L3_TOKENIZER)
        } else {
            Tokenizer::from_file(&config.tokenizer_path)
        }
        .map_err(|error| format!("failed to load semantic tokenizer: {error}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_tokens.max(1),
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|error| format!("failed to configure semantic tokenizer truncation: {error}"))?;
        let mut builder = Session::builder()
            .map_err(|error| format!("failed to create ONNX session builder: {error}"))?;
        if config.encoder_threads > 0 {
            builder = builder
                .with_intra_threads(config.encoder_threads)
                .map_err(|error| format!("failed to configure ONNX threads: {error}"))?;
        }

        #[cfg(any(target_os = "ios", target_os = "macos"))]
        if config.coreml_enabled {
            builder = builder
                .with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default()
                        .with_compute_units(
                            ort::execution_providers::coreml::CoreMLComputeUnits::CPUAndNeuralEngine,
                        )
                        .build(),
                ])
                .map_err(|error| format!("failed to configure CoreML execution provider: {error}"))?;
        }

        let session = if config.model_path.trim().is_empty() {
            builder
                .commit_from_memory(if use_edge_model {
                    EMBEDDED_L3_Q4_MODEL
                } else {
                    EMBEDDED_L3_MODEL
                })
                .map_err(|error| format!("failed to load bundled semantic ONNX model: {error}"))?
        } else {
            builder
                .commit_from_file(&config.model_path)
                .map_err(|error| format!("failed to load semantic ONNX model: {error}"))?
        };

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dimensions: if config.dimensions == 0 {
                DEFAULT_DIMENSIONS
            } else {
                config.dimensions
            },
            max_tokens: config.max_tokens.max(1),
        })
    }
}

impl SemanticEncoder for OnnxSemanticEncoder {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn encode(&self, text: &str) -> Result<Vec<f32>, String> {
        let mut encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| format!("semantic tokenization failed: {error}"))?;
        if encoding.len() > self.max_tokens {
            encoding.truncate(self.max_tokens, 0, TruncationDirection::Right);
        }
        if encoding.is_empty() {
            return Err("semantic tokenizer produced an empty sequence".to_string());
        }

        let sequence_length = encoding.len();
        let input_ids = encoding
            .get_ids()
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>();
        let attention_mask = encoding
            .get_attention_mask()
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>();
        let token_type_ids = encoding
            .get_type_ids()
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>();

        let input_ids = Tensor::from_array(([1usize, sequence_length], input_ids))
            .map_err(|error| format!("failed to construct input_ids tensor: {error}"))?;
        let attention_mask = Tensor::from_array(([1usize, sequence_length], attention_mask))
            .map_err(|error| format!("failed to construct attention_mask tensor: {error}"))?;
        let token_type_ids = Tensor::from_array(([1usize, sequence_length], token_type_ids))
            .map_err(|error| format!("failed to construct token_type_ids tensor: {error}"))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| "semantic ONNX session lock was poisoned".to_string())?;
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            })
            .map_err(|error| format!("semantic ONNX inference failed: {error}"))?;
        let (shape, hidden) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("semantic ONNX output was not f32: {error}"))?;

        if shape.len() != 3
            || shape[0] != 1
            || shape[1] != sequence_length as i64
            || shape[2] != self.dimensions as i64
        {
            return Err(format!(
                "semantic ONNX output shape is incompatible: expected [1, {}, {}], received {:?}",
                sequence_length, self.dimensions, shape
            ));
        }

        let expected_values = sequence_length.saturating_mul(self.dimensions);
        if hidden.len() < expected_values {
            return Err(format!(
                "semantic ONNX output is too small: expected at least {}, received {}",
                expected_values,
                hidden.len()
            ));
        }

        // Sentence-Transformers' mean pooling: average token embeddings using
        // the attention mask, then normalize for cosine retrieval.
        let mut pooled = vec![0.0f32; self.dimensions];
        let mut token_count = 0.0f32;
        for token_index in 0..sequence_length {
            let mask = encoding.get_attention_mask()[token_index];
            if mask == 0 {
                continue;
            }
            token_count += 1.0;
            let offset = token_index * self.dimensions;
            for dimension in 0..self.dimensions {
                pooled[dimension] += hidden[offset + dimension];
            }
        }
        if token_count <= 0.0 {
            return Err("semantic attention mask contains no active tokens".to_string());
        }
        for value in &mut pooled {
            *value /= token_count;
        }
        let norm = pooled
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if norm <= f32::EPSILON || !norm.is_finite() {
            return Err("semantic encoder produced a zero or invalid vector".to_string());
        }
        for value in &mut pooled {
            *value /= norm;
        }
        Ok(pooled)
    }
}

#[cfg(test)]
#[path = "../tests/unit/semantic_encoder.rs"]
mod tests;
