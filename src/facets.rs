use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const MAX_FACETS: usize = 64;
const MAX_ENTITIES: usize = 16;

fn money_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)([$€£]\s*\d|\b\d+(?:[.,]\d+)?\s*(?:usd|eur|gbp|dollars?|euros?|pounds?)\b)").unwrap())
}

fn number_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b\d+(?:[.,]\d+)?\b").unwrap())
}

fn date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b").unwrap())
}

fn duration_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(?:for\s+)?\d+(?:[.,]\d+)?\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b").unwrap())
}

fn prefix_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*([A-Za-z][A-Za-z0-9_-]{1,32})\s*:\s+").unwrap())
}

fn quoted_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#""([^"\n]{2,80})"|'([^'\n]{2,80})'"#).unwrap())
}

fn proper_noun_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,3}\b").unwrap())
}

fn is_entity_noise(raw: &str) -> bool {
    let normalized = raw.trim().to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "i" | "the" | "a" | "an" | "this" | "that" | "here" | "there"
            | "what" | "when" | "where" | "which" | "who" | "whom" | "whose" | "why" | "how"
            | "do" | "does" | "did" | "can" | "could" | "should" | "would" | "will"
            | "please" | "tell" | "user" | "assistant" | "system" | "human" | "bot" | "agent"
    )
}

fn product_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[A-Za-z]{1,12}[- ]?[A-Z]?\d[A-Za-z0-9-]{1,12}(?:\s+[A-Z]{1,4})?\b").unwrap())
}

fn list_item_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\d+[\.)]\s+").unwrap())
}

fn normalize_value(value: &str) -> Option<String> {
    let normalized = value
        .trim()
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_");

    if normalized.len() < 2 || normalized.len() > 64 {
        None
    } else {
        Some(normalized)
    }
}

fn push_unique(out: &mut Vec<String>, seen: &mut HashSet<String>, cue: impl Into<String>) {
    if out.len() >= MAX_FACETS {
        return;
    }
    let cue = cue.into();
    if cue.is_empty() {
        return;
    }
    let key = cue.to_lowercase();
    if seen.insert(key) {
        out.push(cue);
    }
}

fn metadata_string<'a>(metadata: &'a HashMap<String, Value>, keys: &[&str]) -> Option<&'a str> {
    for key in keys {
        if let Some(value) = metadata.get(*key).and_then(|v| v.as_str()) {
            if !value.trim().is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn add_source_facets(content: &str, metadata: Option<&HashMap<String, Value>>, existing_cues: &[String], out: &mut Vec<String>, seen: &mut HashSet<String>) {
    if let Some(metadata) = metadata {
        if let Some(role) = metadata_string(metadata, &["source_role", "role", "speaker", "author_role"]) {
            if let Some(value) = normalize_value(role) {
                push_unique(out, seen, format!("source_role:{}", value));
            }
        }
        if let Some(channel) = metadata_string(metadata, &["source_channel", "channel", "conversation", "room"]) {
            if let Some(value) = normalize_value(channel) {
                push_unique(out, seen, format!("source_channel:{}", value));
            }
        }
        if let Some(source_type) = metadata_string(metadata, &["source_type", "source", "kind"]) {
            if let Some(value) = normalize_value(source_type) {
                push_unique(out, seen, format!("source_type:{}", value));
            }
        }
    }

    for cue in existing_cues {
        if let Some((key, value)) = cue.split_once(':') {
            let target = match key {
                "role" | "speaker" => Some("source_role"),
                "channel" => Some("source_channel"),
                "source" | "category" => Some("source_type"),
                _ => None,
            };
            if let Some(target) = target {
                if let Some(value) = normalize_value(value) {
                    push_unique(out, seen, format!("{}:{}", target, value));
                }
            }
        }
    }

    if let Some(cap) = prefix_re().captures(content) {
        if let Some(value) = cap.get(1).and_then(|m| normalize_value(m.as_str())) {
            push_unique(out, seen, format!("source_role:{}", value));
        }
    }
}

fn add_evidence_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    if number_re().is_match(content) {
        push_unique(out, seen, "has:number");
    }
    if money_re().is_match(content) {
        push_unique(out, seen, "has:money");
    }
    if date_re().is_match(content) {
        push_unique(out, seen, "has:date");
    }
    if duration_re().is_match(content) {
        push_unique(out, seen, "has:duration");
    }

    let list_markers = content
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("- ")
                || trimmed.starts_with("* ")
                || list_item_re().is_match(trimmed)
        })
        .take(3)
        .count();
    if list_markers >= 2 {
        push_unique(out, seen, "has:list");
    }
}

fn has_any(lower: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| lower.contains(needle))
}

fn add_type_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = content.to_lowercase();

    if has_any(&lower, &["favorite", "prefer", "preference", " i like ", " i love ", " enjoy ", " fan of ", "would rather"]) {
        push_unique(out, seen, "type:preference");
    }
    if has_any(&lower, &["don't like", "do not like", "dislike", "hate", "avoid", "can't stand", "not a fan"]) {
        push_unique(out, seen, "type:dislike");
    }
    if has_any(&lower, &[" i own ", " i have ", " i've got ", " bought ", " purchased ", " currently have "]) {
        push_unique(out, seen, "type:ownership");
    }
    if has_any(&lower, &["recommend", "suggest", "suggestion", "you should", "try ", "option", "would be good"]) {
        push_unique(out, seen, "type:recommendation");
    }
    if has_any(&lower, &["recipe", "ingredient", "preheat", "tablespoon", "teaspoon", "bake", "simmer", "saute", "cook for"]) {
        push_unique(out, seen, "type:recipe");
    }
    if has_any(&lower, &["answer is", "the answer", "correct answer", "here's", "here is", "you can ", "you could "]) {
        push_unique(out, seen, "type:answer");
    }
    if has_any(&lower, &["usually", "always", "every morning", "every night", "daily", "weekly", "routine", "habit", "wind down"]) {
        push_unique(out, seen, "type:routine");
    }
}

fn add_temporal_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = content.to_lowercase();
    if has_any(&lower, &["currently", "current ", "right now", "now ", "latest", "newest", "recently updated"]) {
        push_unique(out, seen, "temporal:current");
    }
    if has_any(&lower, &["recently", "lately", "the other day", "past few", "last few"]) {
        push_unique(out, seen, "temporal:recent");
    }
    if has_any(&lower, &["last week", "past week", "previous week"]) {
        push_unique(out, seen, "temporal:last_week");
    }
    if has_any(&lower, &["yesterday", "today", "tomorrow", "last ", "next ", "ago", "past "]) {
        push_unique(out, seen, "temporal:relative");
    }
    for marker in [
        ("yesterday", "temporal:yesterday"),
        ("today", "temporal:today"),
        ("tomorrow", "temporal:tomorrow"),
        ("last month", "temporal:last_month"),
        ("last year", "temporal:last_year"),
    ] {
        if lower.contains(marker.0) {
            push_unique(out, seen, marker.1);
        }
    }
}

fn add_entity_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let mut entities = Vec::new();
    for cap in quoted_re().captures_iter(content) {
        if let Some(raw) = cap.get(1).or_else(|| cap.get(2)).map(|m| m.as_str()) {
            entities.push(raw.to_string());
        }
    }
    for mat in product_re().find_iter(content) {
        entities.push(mat.as_str().to_string());
    }
    for mat in proper_noun_re().find_iter(content) {
        let raw = mat.as_str();
        if is_entity_noise(raw) {
            continue;
        }
        entities.push(raw.to_string());
    }

    let mut entity_seen = HashSet::new();
    for entity in entities {
        if entity_seen.len() >= MAX_ENTITIES {
            break;
        }
        if let Some(value) = normalize_value(&entity) {
            if entity_seen.insert(value.clone()) {
                push_unique(out, seen, format!("entity:{}", value));
            }
        }
    }
}

pub fn extract_memory_facets(content: &str, metadata: Option<&HashMap<String, Value>>, existing_cues: &[String]) -> Vec<String> {
    let mut facets = Vec::new();
    let mut seen = HashSet::new();

    add_source_facets(content, metadata, existing_cues, &mut facets, &mut seen);
    add_evidence_facets(content, &mut facets, &mut seen);
    add_type_facets(content, &mut facets, &mut seen);
    add_temporal_facets(content, &mut facets, &mut seen);
    add_entity_facets(content, &mut facets, &mut seen);

    facets
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryIntent {
    pub labels: Vec<String>,
    pub weighted_cues: Vec<(String, f64)>,
    pub suppress_generic: bool,
}

fn push_weighted_if_available<F>(
    out: &mut Vec<(String, f64)>,
    seen: &mut HashSet<String>,
    available: &F,
    cue: &str,
    weight: f64,
) where
    F: Fn(&str) -> bool,
{
    if available(cue) && seen.insert(cue.to_string()) {
        out.push((cue.to_string(), weight));
    }
}

fn add_label(out: &mut Vec<String>, label: &str) {
    if !out.iter().any(|existing| existing == label) {
        out.push(label.to_string());
    }
}

pub fn compile_query_intent<F>(query: &str, available: F) -> QueryIntent
where
    F: Fn(&str) -> bool,
{
    let lower = query.to_lowercase();
    let mut intent = QueryIntent::default();
    let mut seen = HashSet::new();

    let is_count = has_any(&lower, &["how many", "number of", "count ", "total number", "different "]);
    let is_money = has_any(&lower, &["how much", "cost", "costs", "spent", "paid", "price", "money", "dollars", "usd", "$"]);
    let is_duration = has_any(&lower, &["how long", "duration", "for how many", "years", "months", "weeks", "days"]);
    let is_current = has_any(&lower, &["current", "currently", "right now", "latest", "newest", "recent"]);
    let is_preference = has_any(&lower, &["favorite", "prefer", "preference", "do i like", "i like", "recommend for me", "would i like"]);
    let is_source_answer = has_any(&lower, &["what did", "what was said", "you said", "you told", "you suggested", "you recommended", "provided", "answer"]);
    let is_temporal = has_any(&lower, &["when", "last ", "past ", "ago", "yesterday", "today", "tomorrow", "week", "month", "year"]);
    let is_recipe = has_any(&lower, &["recipe", "ingredient", "cook", "bake"]);
    let is_recommendation = has_any(&lower, &["recommend", "suggest", "should i", "what should"]);

    if is_count {
        add_label(&mut intent.labels, "count");
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "has:number", 3.0);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "has:list", 1.6);
        intent.suppress_generic = true;
    }
    if is_money {
        add_label(&mut intent.labels, "money");
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "has:money", 3.5);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "has:number", 1.5);
        intent.suppress_generic = true;
    }
    if is_duration {
        add_label(&mut intent.labels, "duration");
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "has:duration", 3.2);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "has:number", 1.2);
        intent.suppress_generic = true;
    }
    if is_current {
        add_label(&mut intent.labels, "latest_current");
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "temporal:current", 3.0);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "temporal:recent", 1.8);
    }
    if is_preference {
        add_label(&mut intent.labels, "preference");
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "type:preference", 3.0);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "type:dislike", 2.4);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "type:ownership", 1.2);
    }
    if is_source_answer {
        add_label(&mut intent.labels, "source_answer");
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "type:answer", 2.8);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "type:recommendation", 2.0);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "has:list", 1.4);
    }
    if is_temporal {
        add_label(&mut intent.labels, "temporal_window");
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "has:date", 2.3);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "temporal:relative", 2.0);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "temporal:last_week", 2.0);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "temporal:recent", 1.5);
    }
    if is_recipe {
        add_label(&mut intent.labels, "recipe");
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "type:recipe", 2.8);
    }
    if is_recommendation {
        add_label(&mut intent.labels, "recommendation");
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "type:recommendation", 2.6);
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, "type:preference", 1.8);
    }

    let mut entity_facets = Vec::new();
    let mut entity_seen = HashSet::new();
    add_entity_facets(query, &mut entity_facets, &mut entity_seen);
    for cue in entity_facets {
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, &cue, 2.2);
    }

    intent
}

pub fn is_weak_query_cue(cue: &str) -> bool {
    matches!(
        cue,
        "many" | "number" | "count" | "total" | "different" | "time" | "times" | "current"
            | "currently" | "latest" | "newest" | "recent" | "recently" | "past" | "last"
            | "ago" | "week" | "month" | "year" | "long" | "much" | "cost" | "price"
    )
}
