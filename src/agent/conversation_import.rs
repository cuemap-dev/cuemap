//! Conversation-aware segmentation for `/ingest/content` (`"segmenter": "conversation"`).
//!
//! Chat histories are cut at message boundaries, so every memory holds one message
//! (or one part of a long message) and carries the metadata that conversation
//! features read: conversation id, turn index, role, part index and count, message
//! id and time (see `crate::conversation`). Recognised inputs:
//!
//! - Claude.ai exports: conversations with `chat_messages` (`sender`, `text` or text
//!   `content` blocks). When a conversation names `current_leaf_message_uuid` and its
//!   messages carry `parent_message_uuid`, only the branch ending at that leaf is kept.
//! - ChatGPT exports: conversations with a `mapping` tree. The branch ending at
//!   `current_node` is kept, as the ChatGPT interface shows it. System messages,
//!   hidden messages, tool calls and tool output are skipped.
//! - JSON message lists: `[{"role", "content"}]` or `{"messages": [...]}`, with string
//!   content or `[{"type": "text", "text"}]` blocks.
//! - Plain-text transcripts whose messages start with a label: `User:` / `Assistant:`
//!   (also `Human:`, `AI:`, `System:`), or speaker names (`Alice:`). When any role label
//!   is present, only role labels start messages. Otherwise the transcript must open
//!   with a label, at least one speaker must speak twice, and there must be at most
//!   two distinct speakers per three messages, so that "Note:" or "Example:" lines in
//!   ordinary prose are not read as a conversation.
//!
//! System messages, tool calls and tool output are skipped in every format. Each
//! conversation in an export keeps its own id, and turn indices restart at 0 over the
//! kept messages.

use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Default size above which one message is split into parts.
pub const DEFAULT_MAX_MESSAGE_CHARS: usize = 2000;

/// One message of a conversation, in display order.
#[derive(Debug, Clone, PartialEq)]
pub struct ConversationMessage {
    pub conversation_id: String,
    pub title: Option<String>,
    pub turn: usize,
    /// "user", "assistant", "system", "tool", or a speaker's name as written.
    pub role: String,
    pub text: String,
    pub message_id: Option<String>,
    /// RFC 3339 or the export's own timestamp string.
    pub timestamp: Option<String>,
}

/// A memory to ingest: one message, or one part of a long message.
#[derive(Debug, Clone, PartialEq)]
pub struct ConversationChunk {
    pub content: String,
    pub metadata: HashMap<String, Value>,
    pub structural_cues: Vec<String>,
    /// Stable per-chunk key, appended to the request's source key.
    pub key: String,
}

/// Parses `content` into messages. `default_conversation` names conversations that
/// carry no id of their own (message lists and transcripts).
pub fn parse_conversations(
    content: &str,
    default_conversation: &str,
) -> Result<Vec<ConversationMessage>, String> {
    let trimmed = content.trim_start();
    let messages = if trimmed.starts_with('[') || trimmed.starts_with('{') {
        let value: Value = serde_json::from_str(content)
            .map_err(|error| format!("conversation JSON could not be parsed: {error}"))?;
        parse_json(&value, default_conversation)
    } else {
        parse_transcript(content, default_conversation)
    };
    if messages.is_empty() {
        return Err(
            "no conversation messages found: expected a Claude or ChatGPT export, a JSON message \
             list, or a transcript whose messages start with a role or speaker label"
                .to_string(),
        );
    }
    Ok(messages)
}

/// Parses `content` and turns each message into one or more chunks of at most
/// `max_message_chars` characters (plus the role prefix).
pub fn segment_conversations(
    content: &str,
    default_conversation: &str,
    max_message_chars: usize,
) -> Result<Vec<ConversationChunk>, String> {
    let messages = parse_conversations(content, default_conversation)?;
    let mut chunks = Vec::new();
    for message in messages {
        let parts = split_message(&message.text, max_message_chars.max(1));
        let count = parts.len();
        for (part_index, part) in parts.into_iter().enumerate() {
            let mut metadata = HashMap::new();
            metadata.insert("source_session_id".to_string(), json!(message.conversation_id));
            metadata.insert("source_turn_index".to_string(), json!(message.turn));
            metadata.insert("source_role".to_string(), json!(message.role));
            metadata.insert("source_type".to_string(), json!("chat_message"));
            if count > 1 {
                metadata.insert("source_part_index".to_string(), json!(part_index));
                metadata.insert("source_part_count".to_string(), json!(count));
            }
            if let Some(id) = &message.message_id {
                metadata.insert("source_message_id".to_string(), json!(id));
            }
            if let Some(timestamp) = &message.timestamp {
                metadata.insert("source_date".to_string(), json!(timestamp));
            }
            if let Some(title) = &message.title {
                metadata.insert("source_conversation_title".to_string(), json!(title));
            }
            let mut structural_cues = vec!["source_type:chat_message".to_string()];
            if let Some(role) = cue_value(&message.role) {
                structural_cues.push(format!("source_role:{role}"));
            }
            let message_key = message.message_id.clone().unwrap_or_else(|| message.turn.to_string());
            chunks.push(ConversationChunk {
                content: format!("{}: {}", message.role, part),
                metadata,
                structural_cues,
                key: format!("{}/{}/{}", message.conversation_id, message_key, part_index),
            });
        }
    }
    Ok(chunks)
}

fn cue_value(value: &str) -> Option<String> {
    let normalized = value
        .trim()
        .to_lowercase()
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    (!normalized.is_empty()).then_some(normalized)
}

fn is_skipped_role(role: &str) -> bool {
    role == "system" || role == "tool"
}

/// Canonical role for known labels; other names are kept as written (speakers).
fn canonical_role(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(match trimmed.to_lowercase().as_str() {
        "user" | "human" => "user".to_string(),
        "assistant" | "ai" | "model" | "bot" => "assistant".to_string(),
        "system" | "developer" => "system".to_string(),
        "tool" | "function" => "tool".to_string(),
        _ => trimmed.to_string(),
    })
}

fn string_field(object: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match object.get(*key) {
        Some(Value::String(text)) if !text.trim().is_empty() => Some(text.trim().to_string()),
        Some(Value::Number(number)) => Some(number.to_string()),
        _ => None,
    })
}

fn epoch_to_rfc3339(value: &Value) -> Option<String> {
    let seconds = value.as_f64()?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    chrono::DateTime::from_timestamp(seconds.trunc() as i64, 0).map(|time| time.to_rfc3339())
}

/// Text from a string or from `[{"type": "text", "text": ...}]` blocks.
fn text_content(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.trim().to_string(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| match block {
                Value::String(text) => Some(text.as_str()),
                Value::Object(object)
                    if object.get("type").and_then(Value::as_str).map_or(true, |kind| kind == "text") =>
                {
                    object.get("text").and_then(Value::as_str)
                }
                _ => None,
            })
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn parse_json(value: &Value, default_conversation: &str) -> Vec<ConversationMessage> {
    let items: Vec<&Value> = match value {
        Value::Array(items) => items.iter().collect(),
        other => vec![other],
    };
    let is_conversation = |item: &&Value| {
        item.get("chat_messages").is_some_and(Value::is_array)
            || item.get("mapping").is_some_and(Value::is_object)
    };
    if !items.is_empty() && items.iter().all(is_conversation) {
        let mut messages = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let Some(object) = item.as_object() else { continue };
            let fallback = if items.len() == 1 {
                default_conversation.to_string()
            } else {
                format!("{default_conversation}:{index}")
            };
            if object.contains_key("chat_messages") {
                messages.extend(parse_claude_conversation(object, &fallback));
            } else {
                messages.extend(parse_chatgpt_conversation(object, &fallback));
            }
        }
        return messages;
    }
    // A message list, bare or under "messages".
    let (list, conversation_id, title) = match value {
        Value::Array(list) => (list.as_slice(), default_conversation.to_string(), None),
        Value::Object(object) => match object.get("messages").and_then(Value::as_array) {
            Some(list) => (
                list.as_slice(),
                string_field(object, &["conversation_id", "id", "uuid"])
                    .unwrap_or_else(|| default_conversation.to_string()),
                string_field(object, &["title", "name"]),
            ),
            None => return Vec::new(),
        },
        _ => return Vec::new(),
    };
    let mut messages = Vec::new();
    for item in list {
        let Some(object) = item.as_object() else { continue };
        let Some(role) = string_field(object, &["role", "speaker", "sender", "author"])
            .and_then(|role| canonical_role(&role))
        else {
            continue;
        };
        let text = text_content(object.get("content").or_else(|| object.get("text")));
        if text.is_empty() || is_skipped_role(&role) {
            continue;
        }
        messages.push(ConversationMessage {
            conversation_id: conversation_id.clone(),
            title: title.clone(),
            turn: messages.len(),
            role,
            text,
            message_id: string_field(object, &["id", "uuid", "message_id"]),
            timestamp: string_field(object, &["timestamp", "created_at", "date"])
                .or_else(|| object.get("create_time").and_then(epoch_to_rfc3339)),
        });
    }
    messages
}

fn parse_claude_conversation(conversation: &Map<String, Value>, fallback: &str) -> Vec<ConversationMessage> {
    let conversation_id = string_field(conversation, &["uuid", "id"]).unwrap_or_else(|| fallback.to_string());
    let title = string_field(conversation, &["name", "title"]);
    let all: Vec<&Map<String, Value>> = conversation
        .get("chat_messages")
        .and_then(Value::as_array)
        .map(|list| list.iter().filter_map(Value::as_object).collect())
        .unwrap_or_default();

    // Keep the branch ending at the current leaf when the export records one.
    let leaf = string_field(conversation, &["current_leaf_message_uuid"]);
    let by_id: HashMap<String, &Map<String, Value>> = all
        .iter()
        .filter_map(|message| string_field(message, &["uuid"]).map(|id| (id, *message)))
        .collect();
    let ordered: Vec<&Map<String, Value>> = match leaf.as_ref().and_then(|id| by_id.get(id)) {
        Some(&leaf_message) => {
            let mut chain = vec![leaf_message];
            let mut seen = HashSet::new();
            let mut current = leaf_message;
            while let Some(parent) = string_field(current, &["parent_message_uuid"]) {
                if !seen.insert(parent.clone()) {
                    break;
                }
                match by_id.get(&parent) {
                    Some(&message) => {
                        chain.push(message);
                        current = message;
                    }
                    None => break,
                }
            }
            chain.reverse();
            chain
        }
        None => all,
    };

    let mut messages = Vec::new();
    for message in ordered {
        let Some(role) = string_field(message, &["sender", "role"]).and_then(|role| canonical_role(&role)) else {
            continue;
        };
        if is_skipped_role(&role) {
            continue;
        }
        let mut text = text_content(message.get("text"));
        if text.is_empty() {
            text = text_content(message.get("content"));
        }
        if text.is_empty() {
            continue;
        }
        messages.push(ConversationMessage {
            conversation_id: conversation_id.clone(),
            title: title.clone(),
            turn: messages.len(),
            role,
            text,
            message_id: string_field(message, &["uuid", "id"]),
            timestamp: string_field(message, &["created_at", "timestamp"]),
        });
    }
    messages
}

fn parse_chatgpt_conversation(conversation: &Map<String, Value>, fallback: &str) -> Vec<ConversationMessage> {
    let conversation_id =
        string_field(conversation, &["conversation_id", "id"]).unwrap_or_else(|| fallback.to_string());
    let title = string_field(conversation, &["title"]);
    let Some(mapping) = conversation.get("mapping").and_then(Value::as_object) else {
        return Vec::new();
    };
    // The displayed branch: from `current_node` up to the root. Without a current node,
    // the leaf whose message is newest stands in for it.
    let leaf = string_field(conversation, &["current_node"])
        .filter(|id| mapping.contains_key(id))
        .or_else(|| {
            mapping
                .iter()
                .filter(|(_, node)| node.get("children").and_then(Value::as_array).map_or(true, Vec::is_empty))
                .max_by(|(a_id, a), (b_id, b)| {
                    let time = |node: &Value| {
                        node.pointer("/message/create_time").and_then(Value::as_f64).unwrap_or(0.0)
                    };
                    time(a).total_cmp(&time(b)).then_with(|| b_id.cmp(a_id))
                })
                .map(|(id, _)| id.clone())
        });
    let mut chain = Vec::new();
    let mut seen = HashSet::new();
    let mut current = leaf;
    while let Some(id) = current {
        if !seen.insert(id.clone()) {
            break;
        }
        let Some(node) = mapping.get(&id) else { break };
        chain.push(node);
        current = node.get("parent").and_then(Value::as_str).map(str::to_string);
    }
    chain.reverse();

    let mut messages = Vec::new();
    for node in chain {
        let Some(message) = node.get("message").and_then(Value::as_object) else { continue };
        let role = message
            .get("author")
            .and_then(|author| author.get("role"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if role != "user" && role != "assistant" {
            continue; // system prompts and tool output
        }
        let hidden = message
            .get("metadata")
            .and_then(|metadata| metadata.get("is_visually_hidden_from_conversation"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let recipient = message.get("recipient").and_then(Value::as_str).unwrap_or("all");
        if hidden || recipient != "all" {
            continue; // hidden context and calls to tools
        }
        let content = message.get("content");
        let kind = content.and_then(|content| content.get("content_type")).and_then(Value::as_str).unwrap_or("text");
        if kind != "text" && kind != "multimodal_text" {
            continue;
        }
        let text = text_content(content.and_then(|content| content.get("parts")));
        if text.is_empty() {
            continue;
        }
        messages.push(ConversationMessage {
            conversation_id: conversation_id.clone(),
            title: title.clone(),
            turn: messages.len(),
            role: role.to_string(),
            text,
            message_id: string_field(message, &["id"]),
            timestamp: message.get("create_time").and_then(epoch_to_rfc3339),
        });
    }
    messages
}

fn label_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(?P<label>[\p{L}][\p{L}\p{N} ._'\-]{0,39}):(?:[ \t]+(?P<rest>.*))?$").unwrap())
}

fn is_role_label(label: &str) -> bool {
    matches!(
        label.trim().to_lowercase().as_str(),
        "user" | "human" | "assistant" | "ai" | "system"
    )
}

fn parse_transcript(content: &str, conversation_id: &str) -> Vec<ConversationMessage> {
    let lines: Vec<&str> = content.lines().collect();
    let labels: Vec<Option<(String, String)>> = lines
        .iter()
        .map(|line| {
            label_re().captures(line.trim_end()).map(|caps| {
                (caps["label"].trim().to_string(), caps.name("rest").map_or("", |m| m.as_str()).to_string())
            })
        })
        .collect();
    let role_mode = labels.iter().flatten().any(|(label, _)| is_role_label(label));
    let starts_message = |label: &str| -> bool {
        if role_mode {
            is_role_label(label)
        } else {
            true
        }
    };
    let starts: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(|(index, label)| label.as_ref().filter(|(label, _)| starts_message(label)).map(|_| index))
        .collect();
    if starts.len() < 2 {
        return Vec::new();
    }
    // The transcript must open with a message.
    let first_text_line = lines.iter().position(|line| !line.trim().is_empty());
    if first_text_line != starts.first().copied() {
        return Vec::new();
    }
    if !role_mode {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for &index in &starts {
            if let Some((label, _)) = labels[index].as_ref() {
                *counts.entry(label.as_str()).or_default() += 1;
            }
        }
        let repeated = counts.values().any(|&count| count >= 2);
        if !repeated || counts.len() * 3 > starts.len() * 2 {
            return Vec::new();
        }
    }

    let mut messages = Vec::new();
    for (position, &start) in starts.iter().enumerate() {
        let end = starts.get(position + 1).copied().unwrap_or(lines.len());
        let Some((label, first)) = labels[start].as_ref() else { continue };
        let mut body = vec![first.as_str()];
        body.extend(lines[start + 1..end].iter().copied());
        let text = body.join("\n").trim().to_string();
        let Some(role) = canonical_role(label) else { continue };
        if text.is_empty() || is_skipped_role(&role) {
            continue;
        }
        messages.push(ConversationMessage {
            conversation_id: conversation_id.to_string(),
            title: None,
            turn: messages.len(),
            role,
            text,
            message_id: None,
            timestamp: None,
        });
    }
    messages
}

/// Splits a message into parts of at most `max_chars` characters, preferring
/// paragraph, then line, then sentence, then word boundaries.
pub fn split_message(text: &str, max_chars: usize) -> Vec<String> {
    fn split_on(text: &str, max_chars: usize, separators: &[&str]) -> Vec<String> {
        if text.chars().count() <= max_chars {
            return vec![text.to_string()];
        }
        let Some((&separator, rest)) = separators.split_first() else {
            let chars: Vec<char> = text.chars().collect();
            return chars.chunks(max_chars).map(|chunk| chunk.iter().collect()).collect();
        };
        let pieces: Vec<&str> = text.split_inclusive(separator).collect();
        if pieces.len() < 2 {
            return split_on(text, max_chars, rest);
        }
        let mut parts = Vec::new();
        let mut current = String::new();
        for piece in pieces {
            if current.chars().count() + piece.chars().count() > max_chars && !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            if piece.chars().count() > max_chars {
                parts.extend(split_on(piece, max_chars, rest));
            } else {
                current.push_str(piece);
            }
        }
        if !current.is_empty() {
            parts.push(current);
        }
        parts
    }
    split_on(text, max_chars, &["\n\n", "\n", ". ", " "])
        .into_iter()
        .map(|part| part.trim().to_string())
        .filter(|part| !part.is_empty())
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/conversation_import.rs"]
mod tests;
