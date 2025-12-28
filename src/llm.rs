use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::sync::OnceLock;
use std::process::{Command, Stdio};
use std::time::Duration;
use tracing::{info, error};

pub mod setup {
     use super::*;

     pub fn check_and_install_ollama() -> bool {
         // 1. Check if installed
         let status = Command::new("which")
             .arg("ollama")
             .output();
             
         if status.is_ok() && status.unwrap().status.success() {
             return true;
         }
         
         info!("Ollama not found. Attempting to install via brew...");
         
         // 2. Install via brew
         let install = Command::new("brew")
             .arg("install")
             .arg("ollama")
             .stdout(Stdio::inherit())
             .stderr(Stdio::inherit())
             .status();
             
         match install {
             Ok(s) if s.success() => {
                 info!("Ollama installed successfully.");
                 true
             },
             _ => {
                 error!("Failed to install Ollama via brew. Please install manually.");
                 false
             }
         }
     }
     
     pub async fn ensure_ollama_running(config: &LlmConfig) -> bool {
         if config.provider != "ollama" {
             return true;
         }
         
         if !check_and_install_ollama() {
             return false;
         }

         let client = get_client();
         let base_url = config.ollama_url.trim_end_matches("/");
         let health_url = format!("{}", base_url); // Checking root often returns 200 OK "Ollama is running"

         // 3. Check if running
         let is_running = client.get(&health_url).send().await.is_ok();
         
         if !is_running {
             info!("Ollama is not running. Starting server...");
             // Spawn in background
             let _ = Command::new("ollama")
                 .arg("serve")
                 .stdout(Stdio::null())
                 .stderr(Stdio::null())
                 .spawn();
                 
             // Wait for it to come up
             info!("Waiting for Ollama to start...");
             for _ in 0..10 {
                 tokio::time::sleep(Duration::from_secs(1)).await;
                 if client.get(&health_url).send().await.is_ok() {
                     info!("Ollama started.");
                     break;
                 }
             }
         }
         
         // 4. Check/Pull Model
         // Check if model exists
         let tags_url = format!("{}/api/tags", base_url);
         if let Ok(resp) = client.get(&tags_url).send().await {
             if let Ok(body) = resp.json::<serde_json::Value>().await {
                 let model_exists = body["models"]
                     .as_array()
                     .map(|arr| arr.iter().any(|m| m["name"].as_str().unwrap_or("").contains(&config.model)))
                     .unwrap_or(false);
                     
                 if !model_exists {
                     info!("Model '{}' not found. Pulling... (this make take a while)", config.model);
                     // Using Command to pull so we can inherit stdout and show progress
                     let pull = Command::new("ollama")
                         .arg("pull")
                         .arg(&config.model)
                         .stdout(Stdio::inherit())
                         .stderr(Stdio::inherit())
                         .status();
                         
                     if pull.is_ok() && pull.unwrap().success() {
                         info!("Model pulled successfully.");
                     } else {
                         error!("Failed to pull model '{}'.", config.model);
                         return false;
                     }
                 }
             }
         }
         
         true
     }
}

static CLIENT: OnceLock<Client> = OnceLock::new();

fn get_client() -> &'static Client {
    CLIENT.get_or_init(Client::new)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String, // "ollama" | "openai" | "google"
    pub model: String,
    pub api_key: Option<String>, // Optional for local providers like Ollama
    pub ollama_url: String, // Ollama endpoint (default: http://localhost:11434)
}

impl LlmConfig {
    pub fn from_env() -> Option<Self> {
        // Default to Ollama (local, no API key required)
        let provider = env::var("LLM_PROVIDER").unwrap_or_else(|_| "ollama".to_string());
        
        let (model, api_key, ollama_url) = if provider == "ollama" {
            let model = env::var("LLM_MODEL").unwrap_or_else(|_| "mistral".to_string());
            let url = env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
            (model, None, url)
        } else {
            let model = env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-3.5-turbo".to_string());
            let api_key = env::var("LLM_API_KEY").ok();
            (model, api_key, "http://localhost:11434".to_string())
        };
        
        Some(Self {
            provider,
            model,
            api_key,
            ollama_url,
        })
    }
}

pub async fn propose_cues(content: &str, config: &LlmConfig, known_cues: &[String]) -> Result<Vec<String>, String> {
    match config.provider.as_str() {
        "ollama" => propose_cues_ollama(content, config, known_cues).await,
        "openai" => propose_cues_openai(content, config, known_cues).await,
        "google" => propose_cues_google(content, config, known_cues).await,
        _ => Err(format!("Unsupported provider: {}", config.provider)),
    }
}

pub async fn extract_facts(content: &str, config: &LlmConfig) -> Result<(String, Vec<String>), String> {
    // Only implemented for Ollama for this milestone
    match config.provider.as_str() {
        "ollama" => extract_facts_ollama(content, config).await,
        _ => Err(format!("Unsupported provider for extraction: {}", config.provider)),
    }
}

async fn extract_facts_ollama(content: &str, config: &LlmConfig) -> Result<(String, Vec<String>), String> {
    let system_prompt = r#"You are a Knowledge Extraction Agent. 
Convert the raw file chunk into a structured memory for an agentic database.

OUTPUT FORMAT (JSON):
{
  "summary": "A concise explanation of what this code/text does.",
  "cues": ["key:value", "key:value"]
}

CUE RULES:
- Format: Strictly "lowercase_key:lowercase_value"
- NO DUPLICATED PREFIXES (e.g., do NOT output "lang:python:python", use "lang:python" instead)
- NO spaces in cues
- Use standard keys: type, lang, topic, name, subject, status
- Extract at least 3 factual cues.

Keep summary factual and dense."#;

    let url = format!("{}/api/generate", config.ollama_url);
    
    let response = get_client()
        .post(&url)
        .json(&json!({
            "model": config.model,
            "system": system_prompt,
            "prompt": content,
            "stream": false,
            "format": "json" // Force JSON mode in newer Ollama
        }))
        .send()
        .await
        .map_err(|e| format!("Ollama connection error: {}. Is Ollama running?", e))?;

    if !response.status().is_success() {
        return Err(format!("Ollama API error: {}", response.status()));
    }

    let body: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
    
    let response_text = body["response"]
        .as_str()
        .ok_or("Invalid Ollama response format")?;

    Ok(parse_extraction_response(response_text, content))
}

pub fn parse_extraction_response(response_text: &str, content: &str) -> (String, Vec<String>) {
    // Parse JSON
    let mut summary = String::new();
    let mut cues = Vec::new();

    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response_text)
        .or_else(|_| {
            // Fallback for messy markdown
             let clean = response_text
                .trim()
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_end_matches("```")
                .trim();
             serde_json::from_str(clean)
        }) {
            if let Some(s) = parsed.get("summary").and_then(|v| v.as_str()) {
                summary = s.to_string();
            }
            if let Some(cues_val) = parsed.get("cues") {
                if let Some(arr) = cues_val.as_array() {
                    for v in arr {
                        if let Some(s) = v.as_str() {
                             // Case 1: ["key:value"]
                             cues.push(s.to_string());
                        } else if let Some(obj) = v.as_object() {
                             // Case 2: [{"key": "k", "value": "v"}]
                             if let (Some(k), Some(v)) = (obj.get("key").and_then(|x| x.as_str()), obj.get("value").and_then(|x| x.as_str())) {
                                 cues.push(format!("{}:{}", k, v));
                             }
                        }
                    }
                } else if let Some(obj) = cues_val.as_object() {
                    // Case 3: {"key": "value", "key2": "value2"}
                    for (k, v) in obj {
                         if let Some(val_str) = v.as_str() {
                             cues.push(format!("{}:{}", k, val_str));
                         }
                    }
                }
            }
    }
    
    // Regex Fallback if cues are empty
    if cues.is_empty() {
        if let Ok(re) = Regex::new(r#"["']([a-z0-9_-]+:[a-z0-9_-]+)["']"#) {
            for cap in re.captures_iter(response_text) {
                let cue = cap[1].to_string();
                if !cues.contains(&cue) {
                    cues.push(cue);
                }
            }
        }
    }
    
    if summary.is_empty() {
        summary = content.chars().take(200).collect(); // Fallback prompt
    }
    
    (summary, cues)
}

async fn propose_cues_ollama(content: &str, config: &LlmConfig, known_cues: &[String]) -> Result<Vec<String>, String> {
    let context_hint = if !known_cues.is_empty() {
        format!(
            "I have already identified these potential cues based on keywords: {:?}. Use them as a starting point.\n   CRITICAL: The system is deterministic. Your goal is SEMANTIC EXPANSION (synonyms, hypernyms) to aid recall.\n   Do NOT hallucinate unrelated concepts or go 'crazy'. Keep suggestions grounded in the content.",
            known_cues
        )
    } else {
        String::new()
    };
    
    let system_prompt = format!(r#"You are a semantic tagging engine. Extract rich, queryable cues to enable powerful recall.
{} 

OUTPUT FORMAT (CRITICAL): {{"cues": ["key:value", "key:value", ...]}}

EXTRACT MULTIPLE CUE TYPES:
1. Topic/Domain: What is this about? (topic:payments, topic:food, topic:health)
2. Intent/Action: What action or goal? (intent:planning, intent:shopping, intent:fixing)
3. Subject/Entity: Key nouns and subjects (subject:recipe, subject:service, subject:checkout)
4. Attributes: Properties and qualities (diet:vegetarian, status:broken, priority:high)
5. Context: Related concepts (context:mealprep, context:debugging, context:purchase)

EXAMPLES:
Input: "I'm planning vegetarian meals for next week"
Output: {{"cues": ["intent:planning", "topic:meals", "diet:vegetarian", "subject:meal", "context:mealprep", "timeframe:weekly"]}}

Input: "Checkout is failing with payment errors"
Output: {{"cues": ["subject:checkout", "status:broken", "topic:payments", "type:error", "service:frontend", "intent:debugging"]}}

Input: "Looking for chicken breast alternatives"
Output: {{"cues": ["intent:shopping", "product:chicken", "context:alternatives", "category:meat", "topic:food"]}}

RULES:
- Each cue: Strictly "lowercase_key:lowercase_value" format
- NO DUPLICATED PREFIXES (e.g., do NOT output "topic:payments:payments", use "topic:payments")
- NO spaces or special characters in cues
- Extract 5-8 diverse cues per memory
- Include semantic neighbors (e.g., "meal" → also add "food", "recipe")
- Return ONLY valid JSON"#, context_hint);

    let url = format!("{}/api/generate", config.ollama_url);
    
    let response = get_client()
        .post(&url)
        .json(&json!({
            "model": config.model,
            "system": system_prompt,
            "prompt": content,
            "stream": false
        }))
        .send()
        .await
        .map_err(|e| format!("Ollama connection error: {}. Is Ollama running?", e))?;

    if !response.status().is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Ollama API error: {}", text));
    }

    let body: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
    
    // Extract response text from Ollama format
    let response_text = body["response"]
        .as_str()
        .ok_or("Invalid Ollama response format")?;
    
    parse_proposal_response(response_text)
}

pub fn parse_proposal_response(response_text: &str) -> Result<Vec<String>, String> {
    // PARSING STRATEGY: Try JSON first, fall back to Regex
    let mut extracted_cues = Vec::new();
    let mut json_success = false;

    // 1. JSON Parsing
    let clean_text = response_text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    
    let json_start = clean_text.find('{').unwrap_or(0);
    // Find last '}'
    let json_end = clean_text.rfind('}').map(|i| i + 1).unwrap_or(clean_text.len());
    let potential_json = &clean_text[json_start..json_end];

    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(potential_json) {
        if let Some(cues_array) = parsed.get("cues").and_then(|v| v.as_array()) {
            for cue_val in cues_array {
                if let Some(s) = cue_val.as_str() {
                    // Basic validation: "k:v" format
                    if s.contains(':') && !s.contains(' ') {
                        extracted_cues.push(s.to_string());
                    }
                }
            }
            if !extracted_cues.is_empty() {
                json_success = true;
            }
        }
    }

    // 2. Regex Fallback (if JSON failed or returned nothing)
    if !json_success {
        // Matches "key:value" or 'key:value' inside quotes
        let re = Regex::new(r#"["']([a-z0-9_-]+:[a-z0-9_-]+)["']"#).map_err(|e| e.to_string())?;
        for cap in re.captures_iter(response_text) {
            let cue = cap[1].to_string();
            // Avoid duplicates
            if !extracted_cues.contains(&cue) {
                extracted_cues.push(cue);
            }
        }
        
        if !extracted_cues.is_empty() {
             // We can log here if needed, but logging might not be visible easily
             // println!("Recovered {} cues using regex", extracted_cues.len());
        }
    }

    // Final fallback: just return what we have, or error if empty
    if extracted_cues.is_empty() {
        return Err(format!("Failed to extract cues from LLM response (JSON and Regex failed). Response was: {}", response_text));
    }
        
    Ok(extracted_cues)
}

async fn propose_cues_openai(content: &str, config: &LlmConfig, known_cues: &[String]) -> Result<Vec<String>, String> {
    let api_key = config.api_key.as_ref().ok_or("OpenAI requires LLM_API_KEY")?;
    
    let context_hint = if !known_cues.is_empty() {
        format!("Known cues (use as baseline): {:?}. EXPAND SEMANTICALLY but stay grounded.", known_cues)
    } else {
        String::new()
    };
    
    let system_prompt = format!(r#"You are a tagging engine for a deterministic memory system. Analyze the content and extract canonical cues.
{}
Output strictly JSON in this format: {{"cues": ["key:value", "service:name", ...]}}.
Rules:
- Use k:v format
- Keys must be broad categories (service, topic, lang, tool, error, status)
- Values must be single tokens or short canonical identifiers
- Precision is more important than completeness
- Only extract cues directly implied by the text
- No conversational text"#, context_hint);

    let response = get_client()
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&json!({
            "model": config.model,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": content }
            ],
            "response_format": { "type": "json_object" }
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("OpenAI API error: {}", text));
    }

    let body: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
    
    let cues_json = body["choices"][0]["message"]["content"]
        .as_str()
        .ok_or("Invalid response format")?;
        
    let parsed: serde_json::Value = serde_json::from_str(cues_json).map_err(|e| e.to_string())?;
    
    let cues = parsed["cues"]
        .as_array()
        .ok_or("Missing 'cues' array")?
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
        
    Ok(cues)
}

async fn propose_cues_google(content: &str, config: &LlmConfig, known_cues: &[String]) -> Result<Vec<String>, String> {
    let api_key = config.api_key.as_ref().ok_or("Google requires LLM_API_KEY")?;
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        config.model, api_key
    );

    let prompt = format!(
        "Extract canonical cues (k:v format) from this content. Return JSON {{ \"cues\": [...] }}. Content: {}. Known cues: {:?} (Expand semantically but stay grounded)",
        content, known_cues
    );

    let response = get_client()
        .post(&url)
        .json(&json!({
            "contents": [{
                "parts": [{ "text": prompt }]
            }]
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Google API error: {}", text));
    }

    let body: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
    
    // Parse Gemini response structure (simplified)
    let text = body["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or("Invalid Gemini response")?;
    
    // Gemini often includes markdown code blocks ```json ... ```
    let clean_text = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```");
        
    let parsed: serde_json::Value = serde_json::from_str(clean_text).map_err(|e| e.to_string())?;
    
    let cues = parsed["cues"]
        .as_array()
        .ok_or("Missing 'cues' array")?
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
        
    Ok(cues)
}
