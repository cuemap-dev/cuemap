//! Deterministic structural extraction and bounded query-shape planning.
//!
//! This module deliberately does not attempt to classify language into an
//! ontology. Content extraction emits cues for observable structure: numbers,
//! quantities, identifiers, dates, times, durations, lists, document/code
//! markers, source metadata, surface entities, emoji, discourse markers, and
//! explicit temporal relations. Query planning adds only bounded English
//! query-shape heuristics (grammatical perspective, answer shape,
//! collection/summary/order shape, and reference-time resolution). Semantic
//! retrieval belongs in a vector/learned layer rather than in domain-specific
//! regex families here.

use chrono::{Datelike, Duration, NaiveDate};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const MAX_FACETS: usize = 64;
const MAX_ENTITIES: usize = 16;
// Keep this explicit so source-role weighting can be benchmark-ablated without
// confusing it with a semantic classifier. The current value preserves the
// established v0.7.2 retrieval behavior.
const QUERY_PERSPECTIVE_SOURCE_ROLE_WEIGHT: f64 = 2.0;

fn money_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)([$€£]\s*\d|\b\d+(?:[.,]\d+)?\s*(?:usd|eur|gbp|dollars?|euros?|pounds?)\b)",
        )
        .unwrap()
    })
}

fn url_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)\b(?:https?://(?:localhost(?::\d{1,5})?|(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?|(?:[a-z0-9-]+\.)+[a-z]{2,})(?:[\x2f:\x3f\x23][^\s<>()]*)?|localhost:\d{1,5}\b)",
        )
        .unwrap()
    })
}

fn email_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+\b")
            .unwrap()
    })
}

fn inline_code_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"`[^`\n]{1,120}`"#).unwrap())
}

fn file_name_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#"(?i)(?:^|[\s("'`])(?:\.[a-z][a-z0-9_-]*|[a-z][a-z0-9_.-]*\.[a-z][a-z0-9_.-]*)(?:$|[\s)"'`,;:.])"#,
        )
        .unwrap()
    })
}

fn file_path_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#"(?ix)(?:^|[\s("'`])(?:[a-z]:[\\/](?:[a-z0-9_.-]+[\\/])*[a-z0-9_.-]+|\.{1,2}[\\/](?:[a-z0-9_.-]+[\\/])*[a-z0-9_.-]+|/(?:[a-z0-9_.-]+[\\/])*[a-z0-9_.-]+|(?:[a-z0-9_.-]+[\\/]){1,}[a-z0-9_.-]+)(?:$|[\s)"'`,;:.])"#,
        )
        .unwrap()
    })
}

fn has_code_fence(content: &str) -> bool {
    content
        .lines()
        .any(|line| line.trim_start().starts_with("```"))
}

fn number_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b\d+(?:[.,]\d+)?\b").unwrap())
}

fn measurement_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)\b(?P<value>\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?P<unit>microseconds?|milliseconds?|nanoseconds?|seconds?|minutes?|hours?|days?|bytes?|kilobytes?|megabytes?|gigabytes?|terabytes?|kilograms?|grams?|milligrams?|pounds?|degrees?\s*[cf]|°\s*[cf]|celsius|fahrenheit|kelvin|us|μs|µs|ms|ns|sec|s|min|hr|h|d|b|kb|mb|gb|tb|kg|g|mg|lbs?)\b",
        )
        .unwrap()
    })
}

fn percentage_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?P<value>\d+(?:[.,]\d+)?)\s*(?:%|percent)(?:\b|[\s.,;:!?]|$)")
            .unwrap()
    })
}

fn between_range_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)\bbetween\s+(?P<min>\d+(?:[.,]\d+)?)\s+and\s+(?P<max>\d+(?:[.,]\d+)?)(?:\s*(?P<unit>milliseconds?|seconds?|minutes?|hours?|days?|kilograms?|grams?|pounds?|degrees?\s*[cf]|°\s*[cf]|celsius|fahrenheit|kelvin|ms|s|min|hr|h|d|kg|g|lbs?))?\b",
        )
        .unwrap()
    })
}

fn from_range_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)\bfrom\s+(?P<min>\d+(?:[.,]\d+)?)\s+to\s+(?P<max>\d+(?:[.,]\d+)?)(?:\s*(?P<unit>milliseconds?|seconds?|minutes?|hours?|days?|kilograms?|grams?|pounds?|degrees?\s*[cf]|°\s*[cf]|celsius|fahrenheit|kelvin|ms|s|min|hr|h|d|kg|g|lbs?))?\b",
        )
        .unwrap()
    })
}

fn uuid_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b").unwrap())
}

fn semver_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\bv?\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?\b").unwrap())
}

fn issue_reference_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)(?:\b(?:pr|gh)\s*)?#(?P<issue>\d+)\b|\b(?:gh|pr)-(?P<hyphen_issue>\d+)\b").unwrap())
}

fn ip_address_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b").unwrap())
}

fn port_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\bport\s+(?P<port>\d{1,5})\b|\b(?:localhost|(?:\d{1,3}\.){3}\d{1,3}|(?:[a-z0-9-]+\.)+[a-z]{2,}):(?P<host_port>\d{1,5})\b").unwrap())
}

fn domain_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b").unwrap())
}

fn environment_variable_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b").unwrap())
}

fn user_mention_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"@[A-Za-z][A-Za-z0-9_.-]{1,63}\b").unwrap())
}

fn hashtag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"#[A-Za-z][A-Za-z0-9_-]{1,63}\b").unwrap())
}

fn commit_hash_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[0-9a-fA-F]{7,40}\b").unwrap())
}

fn negation_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:not|never|neither|no|without|cannot|can't|don't|doesn't|didn't|won't|wouldn't|shouldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|mustn't)\b",
        )
        .unwrap()
    })
}

fn json_object_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"(?s)\{\s*["']?[A-Za-z0-9_.-]+["']?\s*:"#).unwrap())
}

fn key_value_line_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*[A-Za-z][A-Za-z0-9_.-]*\s*:\s*\S+").unwrap())
}

fn xml_element_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<[A-Za-z][A-Za-z0-9_.:-]*(?:\s[^>]*)?>.*?</[A-Za-z][A-Za-z0-9_.:-]*>").unwrap())
}

fn markdown_table_separator_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$").unwrap())
}

fn stack_trace_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?im)^\s*(?:traceback\s*\(|at\s+[A-Za-z0-9_.$/-]+(?:\([^\n]*\)|:\d+)|caused by:)").unwrap())
}

fn diff_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:diff --git |\+\+\+ |--- |@@\s+-\d+)").unwrap())
}

fn heading_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s{0,3}(?P<hashes>#{1,6})\s+\S+").unwrap())
}

fn checklist_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\[[ xX]\]\s+\S+").unwrap())
}

fn block_quote_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*>\s+\S+").unwrap())
}

fn markdown_link_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\[[^\]\n]{1,120}\]\([^\)\n]{1,240}\)").unwrap())
}

fn fenced_language_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*```(?P<language>[A-Za-z][A-Za-z0-9_+-]*)\b").unwrap())
}

fn date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b").unwrap()
    })
}

fn short_numeric_date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})(?:/\d{2,4})?\b").unwrap())
}

fn weekday_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(?:mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|saturdays?|sundays?)\b").unwrap())
}

fn clock_time_12_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:0?[1-9]|1[0-2])(?::[0-5]\d)?\s*(?:am|pm|a\.m\.|p\.m\.)\b")
            .unwrap()
    })
}

fn clock_time_capture_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?P<hour>0?[1-9]|1[0-2])(?::(?P<minute>[0-5]\d))?\s*(?P<meridiem>am|pm|a\.m\.|p\.m\.)\b").unwrap()
    })
}

fn clock_time_24_capture_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b(?P<hour>[01]\d|2[0-3]):(?P<minute>[0-5]\d)\b").unwrap()
    })
}

fn has_clock_time(content: &str) -> bool {
    clock_time_12_re().is_match(content) || clock_time_24_capture_re().is_match(content)
}

fn duration_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)\b(?:for\s+)?(?:\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
        )
        .unwrap()
    })
}

fn cadence_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?ix)\b(?:(?:once|twice)|(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+times?|\d+(?:[.,]\d+)?\s+hours?)\s+(?:a|an|per|each|every)\s+(?P<unit>day|week|month|year)s?\b").unwrap()
    })
}

fn every_unit_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\bevery\s+(?P<unit>day|week|month|year)\b").unwrap())
}

fn metadata_date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})\b").unwrap())
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

fn product_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[A-Za-z]{1,12}[- ]?[A-Z]?\d[A-Za-z0-9-]{1,12}(?:\s+[A-Z]{1,4})?\b").unwrap())
}

fn list_item_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\d+[\.)]\s+").unwrap())
}

fn inline_list_item_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?:^|\s)\d{1,2}[\.)]\s+\S").unwrap())
}

fn temporal_event_relation_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?is)\b(?P<relation>after|before)\s+(?:the\s+)?(?P<anchor>[A-Za-z0-9][A-Za-z0-9'_-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'_-]*){0,3})").unwrap()
    })
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

fn normalized_numeric_value(value: &str) -> Option<String> {
    let lowercase = value.to_ascii_lowercase();
    let normalized = match lowercase.as_str() {
        "one" => "1",
        "two" => "2",
        "three" => "3",
        "four" => "4",
        "five" => "5",
        "six" => "6",
        "seven" => "7",
        "eight" => "8",
        "nine" => "9",
        "ten" => "10",
        "eleven" => "11",
        "twelve" => "12",
        value => value,
    };
    normalize_value(normalized).or_else(|| {
        (normalized.len() == 1 && normalized.chars().all(|character| character.is_ascii_digit()))
            .then(|| normalized.to_string())
    })
}

fn canonical_quantity_unit(unit: &str) -> Option<&'static str> {
    match unit.trim().to_ascii_lowercase().replace('°', "").as_str() {
        "microsecond" | "microseconds" | "us" | "μs" | "µs" => Some("us"),
        "millisecond" | "milliseconds" | "ms" => Some("ms"),
        "nanosecond" | "nanoseconds" | "ns" => Some("ns"),
        "second" | "seconds" | "sec" | "s" => Some("s"),
        "minute" | "minutes" | "min" => Some("min"),
        "hour" | "hours" | "hr" | "h" => Some("h"),
        "day" | "days" | "d" => Some("d"),
        "byte" | "bytes" | "b" => Some("b"),
        "kilobyte" | "kilobytes" | "kb" => Some("kb"),
        "megabyte" | "megabytes" | "mb" => Some("mb"),
        "gigabyte" | "gigabytes" | "gb" => Some("gb"),
        "terabyte" | "terabytes" | "tb" => Some("tb"),
        "kilogram" | "kilograms" | "kg" => Some("kg"),
        "gram" | "grams" | "g" => Some("g"),
        "milligram" | "milligrams" | "mg" => Some("mg"),
        "pound" | "pounds" | "lb" | "lbs" => Some("lb"),
        "degree c" | "degrees c" | "celsius" | "c" => Some("celsius"),
        "degree f" | "degrees f" | "fahrenheit" | "f" => Some("fahrenheit"),
        "kelvin" | "k" => Some("kelvin"),
        _ => None,
    }
}

fn normalized_identifier(value: &str) -> Option<String> {
    normalize_value(value.trim_matches(|character: char| {
        !character.is_ascii_alphanumeric()
    }))
}

fn push_unique(out: &mut Vec<String>, seen: &mut HashSet<String>, cue: impl Into<String>) {
    if out.len() >= MAX_FACETS {
        return;
    }
    let cue = cue.into();
    if seen.insert(cue.to_lowercase()) {
        out.push(cue);
    }
}

fn metadata_string<'a>(metadata: &'a HashMap<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|key| {
        metadata.get(*key).and_then(Value::as_str).filter(|value| !value.trim().is_empty())
    })
}

fn parse_date_text(raw: &str) -> Option<NaiveDate> {
    let cap = metadata_date_re().captures(raw)?;
    let year = cap.name("year")?.as_str().parse::<i32>().ok()?;
    let month = cap.name("month")?.as_str().parse::<u32>().ok()?;
    let day = cap.name("day")?.as_str().parse::<u32>().ok()?;
    NaiveDate::from_ymd_opt(year, month, day)
}

fn parse_date_value(value: &Value) -> Option<NaiveDate> {
    if let Some(raw) = value.as_str() {
        return parse_date_text(raw);
    }
    let seconds = value.as_f64()?;
    let days = (seconds / 86_400.0).floor() as i64;
    NaiveDate::from_ymd_opt(1970, 1, 1)?.checked_add_signed(Duration::days(days))
}

fn metadata_date(metadata: &HashMap<String, Value>) -> Option<NaiveDate> {
    [
        "source_date",
        "source_timestamp",
        "timestamp",
        "created_at",
        "datetime",
        "date",
    ]
    .iter()
    .find_map(|key| metadata.get(*key).and_then(parse_date_value))
}

fn source_date_facet(date: NaiveDate) -> String {
    format!("source_date:{:04}_{:02}_{:02}", date.year(), date.month(), date.day())
}

fn source_week_facet(date: NaiveDate) -> String {
    let week = date.iso_week();
    format!("source_week:{:04}_w{:02}", week.year(), week.week())
}

fn source_month_facet(date: NaiveDate) -> String {
    format!("source_month:{:04}_{:02}", date.year(), date.month())
}

fn source_year_facet(date: NaiveDate) -> String {
    format!("source_year:{:04}", date.year())
}

fn month_name_number(token: &str) -> Option<u32> {
    match token.to_ascii_lowercase().as_str() {
        "jan" | "january" => Some(1),
        "feb" | "february" => Some(2),
        "mar" | "march" => Some(3),
        "apr" | "april" => Some(4),
        "may" => Some(5),
        "jun" | "june" => Some(6),
        "jul" | "july" => Some(7),
        "aug" | "august" => Some(8),
        "sep" | "sept" | "september" => Some(9),
        "oct" | "october" => Some(10),
        "nov" | "november" => Some(11),
        "dec" | "december" => Some(12),
        _ => None,
    }
}

fn surface_tokens(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

fn query_tokens(value: &str) -> Vec<String> {
    surface_tokens(value)
        .into_iter()
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn is_numeric_date_token(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    let digits = lower
        .strip_suffix("st")
        .or_else(|| lower.strip_suffix("nd"))
        .or_else(|| lower.strip_suffix("rd"))
        .or_else(|| lower.strip_suffix("th"))
        .unwrap_or(&lower);
    !digits.is_empty() && digits.chars().all(|ch| ch.is_ascii_digit())
}

fn is_temporal_context_word(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "in"
            | "on"
            | "by"
            | "since"
            | "during"
            | "from"
            | "until"
            | "before"
            | "after"
            | "around"
            | "near"
            | "last"
            | "next"
            | "early"
            | "late"
            | "mid"
    )
}

fn is_pronoun_or_auxiliary(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "i"
            | "you"
            | "he"
            | "she"
            | "we"
            | "they"
            | "it"
            | "me"
            | "him"
            | "her"
            | "us"
            | "them"
            | "this"
            | "that"
            | "these"
            | "those"
            | "am"
            | "are"
            | "is"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "do"
            | "does"
            | "did"
            | "have"
            | "has"
            | "had"
            | "can"
            | "could"
            | "may"
            | "might"
            | "must"
            | "shall"
            | "should"
            | "will"
            | "would"
    )
}

fn known_pos_tag(token: &str) -> Option<&'static str> {
    crate::nl::get_known_pos_tag(token)
        .or_else(|| crate::nl::get_known_pos_tag(&token.to_ascii_lowercase()))
}

fn is_noun_or_adjective(token: Option<&String>) -> bool {
    token
        .and_then(|token| known_pos_tag(token))
        .is_some_and(|tag| matches!(tag, "NN" | "NNS" | "NNP" | "NNPS" | "JJ" | "JJR" | "JJS"))
}

fn is_strong_preceding_temporal_context(token: &str) -> bool {
    is_temporal_context_word(token)
}

fn is_ambiguous_month(month: u32) -> bool {
    matches!(month, 3 | 4 | 5 | 6 | 8)
}

fn classify_capitalized_ambiguous_month(
    previous: Option<&String>,
    next: Option<&String>,
) -> bool {
    if previous.is_some_and(|token| is_strong_preceding_temporal_context(token))
        || previous.is_some_and(|token| is_numeric_date_token(token))
        || next.is_some_and(|token| is_numeric_date_token(token))
    {
        return true;
    }

    // A capitalized month at the end of a sentence is normally a date. When
    // followed by a known verb/auxiliary/preposition, it is more likely to be
    // a name, command, or adjective ("March toward...", "June joined...").
    if next.is_none() {
        return true;
    }
    if next.is_some_and(|token| is_pronoun_or_auxiliary(token)) {
        return false;
    }
    if next
        .and_then(|token| known_pos_tag(token))
        .is_some_and(|tag| matches!(tag, "VB" | "VBD" | "VBG" | "VBN" | "VBP" | "VBZ" | "MD" | "IN" | "TO" | "RB"))
    {
        return false;
    }

    is_noun_or_adjective(next)
}

fn is_temporal_month_at(tokens: &[String], index: usize) -> bool {
    let Some(month) = month_name_number(&tokens[index]) else {
        return false;
    };
    let previous = index.checked_sub(1).and_then(|index| tokens.get(index));
    let next = tokens.get(index + 1);
    let raw = &tokens[index];
    let capitalized = raw.chars().next().is_some_and(char::is_uppercase);

    // Lowercase ambiguous month names are only accepted after explicit
    // temporal syntax. This prevents "It may in fact work" and "Version 5
    // may fail" from becoming calendar evidence.
    if !capitalized {
        if month == 5 {
            return previous.is_some_and(|token| is_strong_preceding_temporal_context(token));
        }
        if is_ambiguous_month(month) {
            return previous.is_some_and(|token| is_strong_preceding_temporal_context(token))
                || next.is_some_and(|token| is_numeric_date_token(token));
        }
        return true;
    }

    if month == 5 && known_pos_tag(raw) == Some("MD") {
        return false;
    }
    if next.is_some_and(|token| is_pronoun_or_auxiliary(token)) {
        return false;
    }

    if !is_ambiguous_month(month) {
        return true;
    }

    classify_capitalized_ambiguous_month(previous, next)
}

fn has_temporal_month(content: &str) -> bool {
    let tokens = surface_tokens(content);
    tokens
        .iter()
        .enumerate()
        .any(|(index, _)| is_temporal_month_at(&tokens, index))
}

fn is_entity_noise(raw: &str) -> bool {
    let normalized = raw.trim().to_ascii_lowercase();
    crate::nl::get_stopwords().contains(normalized.as_str())
        || matches!(
            normalized.as_str(),
            "i" | "the" | "a" | "an" | "this" | "that" | "here" | "there" | "what"
                | "when" | "where" | "which" | "who" | "whom" | "whose" | "why" | "how"
                | "do" | "does" | "did" | "can" | "could" | "should" | "would" | "will"
                | "please" | "tell" | "user" | "assistant" | "system" | "human" | "bot"
                | "agent"
        )
}

fn is_temporal_month_entity(content: &str, raw: &str) -> bool {
    if raw.split_whitespace().count() != 1 || month_name_number(raw.trim()).is_none() {
        return false;
    }
    let tokens = surface_tokens(content);
    tokens.iter().enumerate().any(|(index, token)| {
        token.eq_ignore_ascii_case(raw.trim()) && is_temporal_month_at(&tokens, index)
    })
}

fn add_metadata_temporal_facets(
    metadata: Option<&HashMap<String, Value>>,
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    let Some(date) = metadata.and_then(metadata_date) else {
        return;
    };
    push_unique(out, seen, "source_time:dated");
    push_unique(out, seen, source_date_facet(date));
    push_unique(out, seen, source_week_facet(date));
    push_unique(out, seen, source_month_facet(date));
    push_unique(out, seen, source_year_facet(date));
}

fn add_source_facets(
    content: &str,
    metadata: Option<&HashMap<String, Value>>,
    existing_cues: &[String],
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    if let Some(metadata) = metadata {
        for (keys, target) in [
            (&["source_role", "role", "speaker", "author_role"][..], "source_role"),
            (&["source_channel", "channel", "conversation", "room"][..], "source_channel"),
            (&["source_type", "source", "kind"][..], "source_type"),
            (&["source_session_id", "session_id", "conversation_id", "thread_id"][..], "source_session"),
        ] {
            if let Some(value) = metadata_string(metadata, keys).and_then(normalize_value) {
                push_unique(out, seen, format!("{target}:{value}"));
            }
        }
    }

    for cue in existing_cues {
        let Some((key, value)) = cue.split_once(':') else {
            continue;
        };
        let target = match key {
            "role" | "speaker" => Some("source_role"),
            "channel" => Some("source_channel"),
            "source" | "category" => Some("source_type"),
            "session" | "conversation" | "thread" => Some("source_session"),
            _ => None,
        };
        if let Some(target) = target {
            if let Some(value) = normalize_value(value) {
                push_unique(out, seen, format!("{target}:{value}"));
            }
        }
    }

    if let Some(value) = prefix_re().captures(content).and_then(|cap| cap.get(1)).and_then(|m| normalize_value(m.as_str())) {
        push_unique(out, seen, format!("source_role:{value}"));
    }
}

fn clock_hour_24(hour: u32, meridiem: &str) -> u32 {
    let meridiem = meridiem.to_ascii_lowercase();
    if meridiem.starts_with('p') {
        if hour == 12 { 12 } else { hour + 12 }
    } else if hour == 12 {
        0
    } else {
        hour
    }
}

fn time_of_day_for_hour(hour: u32) -> &'static str {
    match hour {
        5..=11 => "morning",
        12..=16 => "afternoon",
        17..=21 => "evening",
        _ => "night",
    }
}

fn add_time_of_day_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = format!(" {} ", content.to_ascii_lowercase());
    for (term, facet) in [
        (" morning ", "morning"),
        (" afternoon ", "afternoon"),
        (" evening ", "evening"),
        (" tonight ", "evening"),
        (" night ", "night"),
        (" bedtime ", "night"),
    ] {
        if lower.contains(term) {
            push_unique(out, seen, format!("time_of_day:{facet}"));
        }
    }
    for cap in clock_time_capture_re().captures_iter(content) {
        let Some(hour) = cap.name("hour").and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        let Some(meridiem) = cap.name("meridiem").map(|m| m.as_str()) else {
            continue;
        };
        push_unique(out, seen, format!("time_of_day:{}", time_of_day_for_hour(clock_hour_24(hour, meridiem))));
    }
    for cap in clock_time_24_capture_re().captures_iter(content) {
        let Some(hour) = cap.name("hour").and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        push_unique(out, seen, format!("time_of_day:{}", time_of_day_for_hour(hour)));
    }
}

fn add_cadence_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let mut matched = false;
    for unit in cadence_re().captures_iter(content).chain(every_unit_re().captures_iter(content)) {
        let Some(unit) = unit.name("unit").and_then(|m| normalize_value(m.as_str())) else {
            continue;
        };
        matched = true;
        push_unique(out, seen, "has:frequency");
        push_unique(out, seen, "schedule:frequency");
        push_unique(out, seen, format!("frequency_unit:{unit}"));
        if unit == "week" {
            push_unique(out, seen, "schedule:weekly");
        }
    }
    if matched && has_clock_time(content) {
        push_unique(out, seen, "has:time");
    }
}

fn add_quantity_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    for caps in measurement_re().captures_iter(content) {
        let Some(value) = caps
            .name("value")
            .and_then(|value| normalized_numeric_value(value.as_str()))
        else {
            continue;
        };
        let Some(unit) = caps
            .name("unit")
            .and_then(|unit| canonical_quantity_unit(unit.as_str()))
        else {
            continue;
        };
        push_unique(out, seen, "has:measurement");
        push_unique(out, seen, format!("quantity_unit:{unit}"));
        push_unique(out, seen, format!("measurement:{value}_{unit}"));
    }

    for caps in percentage_re().captures_iter(content) {
        let Some(value) = caps
            .name("value")
            .and_then(|value| normalized_numeric_value(value.as_str()))
        else {
            continue;
        };
        push_unique(out, seen, "has:percentage");
        push_unique(out, seen, format!("percentage:{value}"));
    }

    for caps in between_range_re()
        .captures_iter(content)
        .chain(from_range_re().captures_iter(content))
    {
        let Some(min) = caps
            .name("min")
            .and_then(|value| normalized_numeric_value(value.as_str()))
        else {
            continue;
        };
        let Some(max) = caps
            .name("max")
            .and_then(|value| normalized_numeric_value(value.as_str()))
        else {
            continue;
        };
        push_unique(out, seen, "has:numeric_range");
        push_unique(out, seen, "has:comparator");
        push_unique(out, seen, "comparison:between");
        push_unique(out, seen, format!("range_min:{min}"));
        push_unique(out, seen, format!("range_max:{max}"));
        if let Some(unit) = caps
            .name("unit")
            .and_then(|unit| canonical_quantity_unit(unit.as_str()))
        {
            push_unique(out, seen, format!("quantity_unit:{unit}"));
            push_unique(out, seen, format!("range:{min}_{max}_{unit}"));
        }
    }

    let lower = content.to_ascii_lowercase();
    let less_than = query_has_any(
        &lower,
        &["under", "below", "less than", "at most", "no more than"],
    );
    let greater_than = query_has_any(
        &lower,
        &["over", "above", "more than", "at least", "no less than"],
    );
    let approximately = query_has_any(
        &lower,
        &["approximately", "approx", "about", "around", "roughly"],
    );
    if less_than || greater_than || approximately {
        push_unique(out, seen, "has:comparator");
    }
    if less_than {
        push_unique(out, seen, "comparison:less_than");
    }
    if greater_than {
        push_unique(out, seen, "comparison:greater_than");
    }
    if approximately {
        push_unique(out, seen, "comparison:approximately");
    }
}

fn valid_ip_address(raw: &str) -> bool {
    let octets = raw.split('.').collect::<Vec<_>>();
    octets.len() == 4
        && octets.iter().all(|octet| {
            octet.parse::<u16>().is_ok_and(|value| value <= 255)
        })
}

fn add_identifier_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    for matched in uuid_re().find_iter(content) {
        let Some(value) = normalized_identifier(matched.as_str()) else {
            continue;
        };
        push_unique(out, seen, "has:uuid");
        push_unique(out, seen, format!("uuid:{value}"));
    }

    for matched in semver_re().find_iter(content) {
        let raw = matched
            .as_str()
            .strip_prefix('v')
            .or_else(|| matched.as_str().strip_prefix('V'))
            .unwrap_or(matched.as_str());
        let Some(value) = normalized_identifier(raw) else {
            continue;
        };
        push_unique(out, seen, "has:semver");
        push_unique(out, seen, format!("version:{value}"));
    }

    for caps in issue_reference_re().captures_iter(content) {
        let Some(value) = caps
            .name("issue")
            .or_else(|| caps.name("hyphen_issue"))
            .and_then(|value| normalized_identifier(value.as_str()))
        else {
            continue;
        };
        push_unique(out, seen, "has:issue_reference");
        push_unique(out, seen, format!("issue:{value}"));
    }

    for matched in ip_address_re().find_iter(content) {
        if !valid_ip_address(matched.as_str()) {
            continue;
        }
        let Some(value) = normalized_identifier(matched.as_str()) else {
            continue;
        };
        push_unique(out, seen, "has:ip_address");
        push_unique(out, seen, format!("ip:{value}"));
    }

    for caps in port_re().captures_iter(content) {
        let Some(raw) = caps
            .name("port")
            .or_else(|| caps.name("host_port"))
            .map(|value| value.as_str())
        else {
            continue;
        };
        let Some(value) = raw.parse::<u32>().ok().filter(|value| *value <= 65_535) else {
            continue;
        };
        push_unique(out, seen, "has:port");
        push_unique(out, seen, format!("port:{value}"));
    }

    for matched in domain_re().find_iter(content) {
        let Some(value) = normalized_identifier(matched.as_str()) else {
            continue;
        };
        // The ingestion agent already owns `domain:*`; use a distinct
        // namespace for content-derived domain structure.
        push_unique(out, seen, "has:domain");
        push_unique(out, seen, format!("domain_name:{value}"));
    }

    for matched in environment_variable_re().find_iter(content) {
        let Some(value) = normalized_identifier(matched.as_str()) else {
            continue;
        };
        push_unique(out, seen, "has:environment_variable");
        push_unique(out, seen, format!("env:{value}"));
    }

    for matched in user_mention_re().find_iter(content) {
        let Some(value) = normalized_identifier(matched.as_str().trim_start_matches('@')) else {
            continue;
        };
        push_unique(out, seen, "has:user_mention");
        push_unique(out, seen, format!("mention:{value}"));
    }

    for matched in hashtag_re().find_iter(content) {
        let Some(value) = normalized_identifier(matched.as_str().trim_start_matches('#')) else {
            continue;
        };
        push_unique(out, seen, "has:hashtag");
        push_unique(out, seen, format!("hashtag:{value}"));
    }

    for matched in commit_hash_re().find_iter(content) {
        let raw = matched.as_str();
        if !raw.chars().any(|character| matches!(character, 'a'..='f' | 'A'..='F')) {
            continue;
        }
        let Some(value) = normalized_identifier(raw) else {
            continue;
        };
        push_unique(out, seen, "has:commit_hash");
        push_unique(out, seen, format!("commit:{value}"));
    }
}

fn clean_file_reference(raw: &str) -> String {
    let mut value = raw
        .trim()
        .trim_matches(|character: char| matches!(character, '"' | '\'' | '`' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':' | '!' | '?'))
        .to_string();
    while value.ends_with('.') && !value.starts_with('.') {
        value.pop();
    }
    value
}

fn add_one_file_reference(reference: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let reference = clean_file_reference(reference);
    if reference.is_empty() || reference.contains("://") {
        return;
    }
    let normalized_separators = reference.replace('\\', "/");
    let segments = normalized_separators
        .split('/')
        .filter(|segment| !segment.is_empty() && *segment != "." && *segment != "..")
        .filter(|segment| !segment.ends_with(':'))
        .collect::<Vec<_>>();
    let Some(basename) = segments.last().copied() else {
        return;
    };

    if segments.len() > 1 {
        push_unique(out, seen, "has:directory_path");
        for segment in &segments[..segments.len() - 1] {
            if let Some(value) = normalize_value(segment) {
                push_unique(out, seen, format!("path_segment:{value}"));
            }
        }
    }

    if let Some(value) = normalize_value(basename) {
        push_unique(out, seen, format!("file_name:{value}"));
    }
    if !basename.starts_with('.') {
        if let Some(extension) = basename.rsplit_once('.').map(|(_, extension)| extension) {
            if let Some(value) = normalize_value(extension) {
                push_unique(out, seen, format!("file_extension:{value}"));
            }
        }
    }
}

fn add_file_structure_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    for matched in file_path_re().find_iter(content) {
        add_one_file_reference(matched.as_str(), out, seen);
    }
    for matched in file_name_re().find_iter(content) {
        add_one_file_reference(matched.as_str(), out, seen);
    }
}

fn looks_like_csv(content: &str) -> bool {
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter(|line| line.split(',').count() >= 2)
        .take(3)
        .count()
        >= 2
}

fn add_document_structure_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    if json_object_re().is_match(content) {
        push_unique(out, seen, "has:json");
    }
    let key_value_lines = key_value_line_re().find_iter(content).count();
    if key_value_lines >= 2 {
        push_unique(out, seen, "has:key_value_pairs");
        push_unique(out, seen, "has:yaml");
    }
    if xml_element_re().is_match(content) {
        push_unique(out, seen, "has:xml");
    }
    if looks_like_csv(content) {
        push_unique(out, seen, "has:csv");
    }
    let table_rows = content.lines().filter(|line| line.contains('|')).count();
    if markdown_table_separator_re().is_match(content) && table_rows >= 2 {
        push_unique(out, seen, "has:markdown_table");
    }
    if stack_trace_re().is_match(content) {
        push_unique(out, seen, "has:stack_trace");
    }
    if diff_re().is_match(content) {
        push_unique(out, seen, "has:diff");
    }
    for caps in heading_re().captures_iter(content) {
        let Some(level) = caps.name("hashes").map(|hashes| hashes.as_str().len()) else {
            continue;
        };
        push_unique(out, seen, "has:heading");
        push_unique(out, seen, format!("heading_level:{level}"));
    }
    if checklist_re().is_match(content) {
        push_unique(out, seen, "has:checklist");
    }
    if block_quote_re().is_match(content) {
        push_unique(out, seen, "has:block_quote");
    }
    if markdown_link_re().is_match(content) {
        push_unique(out, seen, "has:markdown_link");
    }
    for caps in fenced_language_re().captures_iter(content) {
        let Some(language) = caps.name("language").map(|language| language.as_str().to_ascii_lowercase().replace(|character: char| !character.is_ascii_alphanumeric(), "_")) else {
            continue;
        };
        if !language.is_empty() {
            push_unique(out, seen, "has:code");
            push_unique(out, seen, format!("code_language:{language}"));
        }
    }
}

fn is_emoji(character: char) -> bool {
    let code = character as u32;
    (0x1f300..=0x1faff).contains(&code) || (0x2600..=0x27bf).contains(&code)
}

fn add_emoji_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let mut emoji = false;
    for character in content.chars() {
        emoji |= is_emoji(character);
    }
    if emoji {
        push_unique(out, seen, "has:emoji");
    }
}

fn add_discourse_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = content.to_ascii_lowercase();
    let has_negation = negation_re().is_match(&lower);
    if has_negation {
        push_unique(out, seen, "has:negation");
    }
    if query_has_any(
        &lower,
        &["but", "however", "although", "whereas", "instead of", "rather than"],
    ) {
        push_unique(out, seen, "has:contrast");
    }
    if query_has_any(&lower, &["actually", "correction", "to clarify", "i mean", "rather than"])
        || (has_negation && query_has_any(&lower, &["but"]))
    {
        push_unique(out, seen, "has:correction");
    }
    if query_has_any(
        &lower,
        &[
            "used to",
            "no longer",
            "no more",
            "changed my mind",
            "instead of",
            "replaced by",
            "superseded by",
        ],
    ) {
        push_unique(out, seen, "has:supersession");
    }
}

fn add_evidence_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    if number_re().is_match(content) { push_unique(out, seen, "has:number"); }
    if money_re().is_match(content) { push_unique(out, seen, "has:money"); }
    if url_re().is_match(content) { push_unique(out, seen, "has:url"); }
    if email_re().is_match(content) { push_unique(out, seen, "has:email"); }
    if quoted_re().is_match(content) { push_unique(out, seen, "has:quote"); }
    if inline_code_re().is_match(content) || has_code_fence(content) {
        push_unique(out, seen, "has:code");
    }
    let has_file_path = file_path_re().is_match(content);
    if has_file_path || file_name_re().is_match(content) {
        push_unique(out, seen, "has:file_name");
    }
    if has_file_path { push_unique(out, seen, "has:file_path"); }
    if date_re().is_match(content)
        || short_numeric_date_re().is_match(content)
        || has_temporal_month(content)
    {
        push_unique(out, seen, "has:date");
    }
    let tokens = surface_tokens(content);
    for (index, token) in tokens.iter().enumerate() {
        if let Some(month) = month_name_number(token) {
            if is_temporal_month_at(&tokens, index) {
                push_unique(out, seen, format!("content_month:{month:02}"));
            }
        }
    }
    if duration_re().is_match(content) { push_unique(out, seen, "has:duration"); }
    if weekday_re().is_match(content) {
        push_unique(out, seen, "has:weekday");
        push_unique(out, seen, "schedule:weekly");
    }
    if has_clock_time(content) { push_unique(out, seen, "has:time"); }
    add_time_of_day_facets(content, out, seen);
    add_cadence_facets(content, out, seen);
    add_quantity_facets(content, out, seen);
    add_identifier_facets(content, out, seen);
    add_file_structure_facets(content, out, seen);
    add_document_structure_facets(content, out, seen);
    add_emoji_facets(content, out, seen);
    add_discourse_facets(content, out, seen);

    let list_markers = content
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("- ") || trimmed.starts_with("* ") || list_item_re().is_match(trimmed)
        })
        .take(3)
        .count();
    let inline_markers = inline_list_item_re().find_iter(content).take(3).count();
    if list_markers >= 2 || inline_markers >= 3 {
        push_unique(out, seen, "has:list");
    }
}

fn add_surface_entities(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let mut candidates = Vec::new();
    for cap in quoted_re().captures_iter(content) {
        if let Some(raw) = cap.get(1).or_else(|| cap.get(2)).map(|m| m.as_str()) {
            candidates.push(raw.to_string());
        }
    }
    candidates.extend(product_re().find_iter(content).map(|m| m.as_str().to_string()));
    candidates.extend(proper_noun_re().find_iter(content).filter_map(|m| {
        let raw = m.as_str();
        (!is_entity_noise(raw) && !is_temporal_month_entity(content, raw)).then(|| raw.to_string())
    }));

    let mut entity_seen = HashSet::new();
    for candidate in candidates {
        if entity_seen.len() >= MAX_ENTITIES { break; }
        if let Some(value) = normalize_value(&candidate) {
            if entity_seen.insert(value.clone()) {
                push_unique(out, seen, format!("entity:{value}"));
            }
        }
    }
}

fn temporal_event_relations(content: &str) -> Vec<(String, String)> {
    let mut relations = Vec::new();
    let mut seen = HashSet::new();
    for caps in temporal_event_relation_re().captures_iter(content) {
        let Some(relation) = caps.name("relation").map(|m| m.as_str().to_ascii_lowercase()) else {
            continue;
        };
        let Some(anchor) = caps.name("anchor") else { continue; };
        let cues = crate::nl::tokenize_to_cues(anchor.as_str());
        let Some(anchor) = cues.iter().filter(|cue| cue.contains('_')).max_by_key(|cue| (cue.split('_').count(), cue.len())).or_else(|| cues.first()) else {
            continue;
        };
        let pair = (relation, anchor.clone());
        if seen.insert(pair.clone()) { relations.push(pair); }
    }
    relations
}

fn add_temporal_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = content.to_ascii_lowercase();
    for (needle, facet) in [
        ("currently", "temporal:current"),
        ("right now", "temporal:current"),
        ("latest", "temporal:current"),
        ("recently", "temporal:recent"),
        ("lately", "temporal:recent"),
        ("last week", "temporal:last_week"),
        ("past week", "temporal:last_week"),
        ("previous week", "temporal:last_week"),
        ("yesterday", "temporal:yesterday"),
        ("today", "temporal:today"),
        ("tomorrow", "temporal:tomorrow"),
        ("last month", "temporal:last_month"),
        ("last year", "temporal:last_year"),
    ] {
        if lower.contains(needle) { push_unique(out, seen, facet); }
    }
    if lower.contains("last ") || lower.contains("next ") || lower.contains("ago") || lower.contains("past ") {
        push_unique(out, seen, "temporal:relative");
    }
    for (relation, anchor) in temporal_event_relations(content) {
        push_unique(out, seen, format!("temporal_relation:{relation}"));
        push_unique(out, seen, format!("temporal_anchor:{anchor}"));
    }
}

pub fn extract_memory_facets_core(
    content: &str,
    metadata: Option<&HashMap<String, Value>>,
    existing_cues: &[String],
) -> Vec<String> {
    let mut facets = Vec::new();
    let mut seen = HashSet::new();
    add_source_facets(content, metadata, existing_cues, &mut facets, &mut seen);
    add_metadata_temporal_facets(metadata, &mut facets, &mut seen);
    add_evidence_facets(content, &mut facets, &mut seen);
    add_temporal_facets(content, &mut facets, &mut seen);
    add_surface_entities(content, &mut facets, &mut seen);
    facets
}

pub fn extract_memory_facets(
    content: &str,
    metadata: Option<&HashMap<String, Value>>,
    existing_cues: &[String],
) -> Vec<String> {
    extract_memory_facets_core(content, metadata, existing_cues)
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StructuralQueryPlan {
    /// Query-shape labels are structural only; semantic intent labels are not emitted.
    /// Perspective and query-shape heuristics are currently English-specific.
    pub labels: Vec<String>,
    pub weighted_cues: Vec<(String, f64)>,
    #[serde(default)]
    pub cue_weight_adjustments: Vec<(String, f64)>,
    pub suppress_generic: bool,
}

fn push_weighted_if_available<F>(
    weighted: &mut Vec<(String, f64)>,
    seen: &mut HashSet<String>,
    available: &F,
    cue: impl Into<String>,
    weight: f64,
) where
    F: Fn(&str) -> bool,
{
    let cue = cue.into();
    if available(&cue) && seen.insert(cue.clone()) {
        weighted.push((cue, weight));
    }
}

fn push_adjustment(out: &mut Vec<(String, f64)>, cue: &str, multiplier: f64) {
    if let Some((_, existing)) = out.iter_mut().find(|(existing, _)| existing == cue) {
        *existing *= multiplier;
    } else {
        out.push((cue.to_string(), multiplier));
    }
}

fn add_label(out: &mut Vec<String>, label: &str) {
    if !out.iter().any(|existing| existing == label) {
        out.push(label.to_string());
    }
}

fn query_has_any(query: &str, needles: &[&str]) -> bool {
    let tokens = query_tokens(query);
    needles.iter().any(|needle| {
        let needle_tokens = query_tokens(needle);
        !needle_tokens.is_empty()
            && tokens
                .windows(needle_tokens.len())
                .any(|window| window == needle_tokens.as_slice())
    })
}

fn add_answer_shape_label(query: &str, plan: &mut StructuralQueryPlan) {
    let tokens = query_tokens(query);
    if tokens.is_empty() {
        return;
    }

    let shape = if query_has_any(query, &["who", "whose"]) {
        Some("person")
    } else if query_has_any(query, &["where"]) {
        Some("location")
    } else if query_has_any(query, &["when", "what time"]) {
        Some("time")
    } else if query_has_any(query, &["how many", "number of", "count of"]) {
        Some("count")
    } else if query_has_any(query, &["how much", "amount of", "cost of", "price of"]) {
        Some("amount")
    } else if query_has_any(query, &["why"]) {
        Some("reason")
    } else if query_has_any(query, &["how long"]) {
        Some("duration")
    } else if query_has_any(query, &["which", "what options", "what choices"]) {
        Some("selection")
    } else if query_has_any(query, &["what kind", "what type", "what category"]) {
        Some("category")
    } else if is_question_auxiliary(&tokens[0])
        && tokens.get(1).is_none_or(|token| token != "you")
        && !tokens.iter().any(|token| is_question_word(token))
    {
        Some("boolean")
    } else {
        None
    };

    if let Some(shape) = shape {
        add_label(&mut plan.labels, &format!("answer_shape_{shape}"));
    }
}

fn query_perspective_tokens(query: &str) -> Vec<String> {
    let normalized = query
        .to_ascii_lowercase()
        .replace('’', "'")
        // Expand only grammatical contractions. Predicate/content words are
        // intentionally not inspected or classified.
        .replace("n't", " not")
        .replace("'re", " are")
        .replace("'ve", " have")
        .replace("'m", " am")
        .replace("'ll", " will")
        .replace("'d", " would")
        .replace("'s", " is");
    query_tokens(&normalized)
}

fn is_question_auxiliary(token: &str) -> bool {
    matches!(
        token,
        "am"
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
            | "may"
            | "might"
            | "must"
            | "shall"
            | "should"
            | "will"
            | "would"
    )
}

fn is_question_word(token: &str) -> bool {
    matches!(token, "what" | "which" | "when" | "where" | "why" | "how" | "who")
}

fn is_embedded_question_marker(token: &str) -> bool {
    is_question_word(token) || matches!(token, "if" | "whether")
}

fn perspective_after_auxiliary(tokens: &[String], auxiliary_index: usize) -> Option<&'static str> {
    let subject_index = if tokens.get(auxiliary_index + 1).is_some_and(|token| token == "not") {
        auxiliary_index + 2
    } else {
        auxiliary_index + 1
    };
    tokens
        .get(subject_index)
        .and_then(|token| query_perspective_for_subject(token))
}

fn query_perspective_for_subject(token: &str) -> Option<&'static str> {
    match token {
        "i" | "we" | "me" | "us" | "my" | "our" | "mine" | "ours" => {
            Some("first_person")
        }
        "you" | "your" | "yours" => Some("second_person"),
        "he" | "she" | "they" | "it" | "him" | "her" | "them" | "his" | "their"
        | "its" | "theirs" => Some("third_person"),
        _ => None,
    }
}

fn is_request_wrapper(tokens: &[String]) -> bool {
    match tokens.first().map(String::as_str) {
        Some("please") => true,
        Some("can" | "could" | "will" | "would") => {
            tokens.get(1).is_some_and(|token| token == "you")
        }
        Some("do") => {
            tokens.get(1).is_some_and(|token| token == "you")
                && tokens.get(2).is_some_and(|token| token == "remember")
        }
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbeddedPerspective {
    None,
    One(&'static str),
    Conflicting,
}

fn embedded_query_perspective(tokens: &[String]) -> EmbeddedPerspective {
    let mut found = None;
    for index in 1..tokens.len() {
        let Some(perspective) = (|| {
            if !is_embedded_question_marker(&tokens[index]) {
                return None;
            }

            if let Some(perspective) = tokens
                .get(index + 1)
                .and_then(|token| query_perspective_for_subject(token))
            {
                return Some(perspective);
            }

            if tokens
                .get(index + 1)
                .is_some_and(|token| is_question_auxiliary(token))
            {
                return perspective_after_auxiliary(tokens, index + 1);
            }

            None
        })() else {
            continue;
        };

        match found {
            None => found = Some(perspective),
            Some(existing) if existing == perspective => {}
            Some(_) => return EmbeddedPerspective::Conflicting,
        }
    }

    match found {
        Some(perspective) => EmbeddedPerspective::One(perspective),
        None => EmbeddedPerspective::None,
    }
}

fn outer_query_perspective(tokens: &[String]) -> Option<&'static str> {
    if tokens.len() >= 2 && is_question_auxiliary(&tokens[0]) {
        if let Some(perspective) = perspective_after_auxiliary(&tokens, 0) {
            return Some(perspective);
        }
    }

    if tokens.len() >= 3
        && is_question_word(&tokens[0])
        && is_question_auxiliary(&tokens[1])
    {
        if let Some(perspective) = perspective_after_auxiliary(&tokens, 1) {
            return Some(perspective);
        }
    }

    None
}

/// Detects who the query is grammatically asking about without inspecting the
/// predicate or topic. Embedded perspective overrides the outer clause only
/// for recognized request wrappers; otherwise disagreement is left unweighted.
fn query_perspective(query: &str) -> Option<&'static str> {
    let tokens = query_perspective_tokens(query);
    let outer = outer_query_perspective(&tokens);
    let embedded = embedded_query_perspective(&tokens);

    if is_request_wrapper(&tokens) {
        return match embedded {
            EmbeddedPerspective::One(perspective) => Some(perspective),
            EmbeddedPerspective::None | EmbeddedPerspective::Conflicting => None,
        };
    }

    match embedded {
        EmbeddedPerspective::None => outer,
        EmbeddedPerspective::One(inner) => match outer {
            None => Some(inner),
            Some(outer) if outer == inner => Some(outer),
            Some(_) => None,
        },
        EmbeddedPerspective::Conflicting => None,
    }
}

pub fn compile_query_plan<F>(query: &str, available: F) -> StructuralQueryPlan
where
    F: Fn(&str) -> bool,
{
    compile_query_plan_with_reference_time(query, None, available)
}

pub fn compile_query_plan_with_reference_time<F>(
    query: &str,
    reference_time: Option<&str>,
    available: F,
) -> StructuralQueryPlan
where
    F: Fn(&str) -> bool,
{
    let lower = query.to_ascii_lowercase();
    let mut plan = StructuralQueryPlan::default();
    let mut seen = HashSet::new();
    let structural = extract_memory_facets_core(query, None, &[]);

    for facet in structural {
        let label = facet.replace(':', "_");
        add_label(&mut plan.labels, &label);
        push_weighted_if_available(
            &mut plan.weighted_cues,
            &mut seen,
            &available,
            facet,
            2.0,
        );
    }

    let explicit_source_role = plan.labels.iter().any(|label| {
        label == "source_role_user" || label == "source_role_assistant"
    });
    if let Some(perspective) = query_perspective(query) {
        add_label(
            &mut plan.labels,
            &format!("query_perspective_{perspective}"),
        );

        if !explicit_source_role {
            // Grammatical perspective and stored-message authorship are
            // different dimensions. In this deployment, first-person
            // questions usually ask about user-authored memories and
            // second-person questions usually ask about assistant-authored
            // memories, so retain those as soft retrieval preferences.
            let source_role = match perspective {
                "first_person" => Some(("source_user", "user")),
                "second_person" => Some(("source_assistant", "assistant")),
                "third_person" => None,
                _ => None,
            };
            if let Some((label, role)) = source_role {
                add_label(&mut plan.labels, label);
                push_weighted_if_available(
                    &mut plan.weighted_cues,
                    &mut seen,
                    &available,
                    format!("source_role:{role}"),
                    QUERY_PERSPECTIVE_SOURCE_ROLE_WEIGHT,
                );
            }
        }
    }

    if query_has_any(&lower, &["timeline", "sequence", "in order", "chronological", "what order"]) {
        add_label(&mut plan.labels, "ordered_reconstruction");
    }
    if query_has_any(&lower, &["summarize", "summary", "overview", "recap", "main points", "key points"]) {
        add_label(&mut plan.labels, "multi_evidence_summary");
    }
    if query_has_any(&lower, &["list", "all", "several", "multiple", "different options", "examples"]) {
        add_label(&mut plan.labels, "multi_evidence_collection");
        push_weighted_if_available(&mut plan.weighted_cues, &mut seen, &available, "has:list", 1.2);
    }
    add_answer_shape_label(query, &mut plan);

    if let Some(reference_time) = reference_time.and_then(parse_date_text) {
        if lower.contains("today") {
            push_weighted_if_available(&mut plan.weighted_cues, &mut seen, &available, source_date_facet(reference_time), 2.0);
            push_adjustment(&mut plan.cue_weight_adjustments, "today", 0.35);
            add_label(&mut plan.labels, "temporal_resolved_date");
        } else if lower.contains("yesterday") {
            if let Some(date) = reference_time.checked_sub_signed(Duration::days(1)) {
                push_weighted_if_available(&mut plan.weighted_cues, &mut seen, &available, source_date_facet(date), 2.0);
                push_adjustment(&mut plan.cue_weight_adjustments, "yesterday", 0.35);
                add_label(&mut plan.labels, "temporal_resolved_date");
            }
        }
    }

    plan.suppress_generic = false;
    plan
}

pub fn is_weak_query_cue(cue: &str) -> bool {
    matches!(
        cue,
        "many" | "number" | "count" | "total" | "different" | "time" | "times" | "current"
            | "currently" | "latest" | "newest" | "recent" | "recently" | "past" | "last"
            | "ago" | "week" | "month" | "year" | "long" | "much" | "cost" | "price"
    )
}

#[cfg(test)]
#[path = "../tests/unit/facets.rs"]
mod tests;
