use chrono::{Datelike, Duration, NaiveDate};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const MAX_FACETS: usize = 64;
const MAX_ENTITIES: usize = 16;

fn money_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)([$€£]\s*\d|\b\d+(?:[.,]\d+)?\s*(?:usd|eur|gbp|dollars?|euros?|pounds?)\b)",
        )
        .unwrap()
    })
}

fn number_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b\d+(?:[.,]\d+)?\b").unwrap())
}

fn date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b").unwrap())
}

fn short_numeric_date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})(?:/\d{2,4})?\b").unwrap())
}

fn weekday_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(?:mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|saturdays?|sundays?)\b").unwrap())
}

fn clock_time_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(?:[01]?\d|2[0-3])(?::[0-5]\d)?\s*(?:am|pm|a\.m\.|p\.m\.)\b").unwrap())
}

fn clock_time_capture_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?P<hour>[01]?\d|2[0-3])(?::(?P<minute>[0-5]\d))?\s*(?P<meridiem>am|pm|a\.m\.|p\.m\.)\b",
        )
        .unwrap()
    })
}

fn duration_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:for\s+)?
            (?:
                \d+(?:[.,]\d+)?
                |
                one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve
            )
            \s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b
            ",
        )
        .unwrap()
    })
}

fn cadence_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)\b(?:(?:once|twice)|(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+times?|\d+(?:[.,]\d+)?\s+hours?)\s+(?:a|an|per|each|every)\s+(?P<unit>day|week|month|year)s?\b",
        )
        .unwrap()
    })
}

fn every_unit_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\bevery\s+(?P<unit>day|week|month|year)\b").unwrap())
}

fn age_year_old_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?P<age>\d{1,3})\s*[- ]\s*years?\s*[- ]\s*old\b").unwrap()
    })
}

fn current_age_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:(?:i|we)\s*(?:am|are|'m|'re|’m|’re)\s+(?:(?:a|an)\s+)?|as\s+(?:a|an)\s+|currently\s+)(?P<age>\d{1,3})\s*[- ]\s*years?\s*[- ]\s*old\b",
        )
        .unwrap()
    })
}

fn event_age_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:at\s+(?:the\s+)?age\s+of|at\s+age|by\s+age|when\s+(?:i|we|he|she|they|you)\s+(?:was|were))\s+(?P<age>\d{1,3})\b",
        )
        .unwrap()
    })
}

fn metadata_date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})\b")
            .unwrap()
    })
}

fn first_person_completed_action_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:i|we)(?:\s+|'ve\s+|’ve\s+|'d\s+|’d\s+)
            (?:
                (?:(?:just|recently|finally|already)\s+)[a-z]{3,}(?:ed|ied) |
                (?:(?:just|recently|finally|already|also)\s+)?
                (?:went | gone | visited | attended | joined | participated |
                   completed | finished | started | planted | bought | purchased |
                   booked | watched | read | wrote | made | built | created |
                   cooked | baked | ran | walked | hiked | traveled | travelled |
                   played | tried) |
                (?:(?:just|recently|finally|already)\s+)?got\s+back\s+from |
                (?:(?:have|had|'ve|’ve)\s+)?been\s+to
            )\b",
        )
        .unwrap()
    })
}

fn first_person_did_named_event_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:i|we)\s+
            (?:(?:just|recently|finally|already|also)\s+)?
            did\s+
            (?:(?:the|a|an)\s+)?
            .{0,90}
            \b(?:event|walk|run|ride|drive|gala|fundraiser|fund-raiser|workshop|tour|class)\b
            ",
        )
        .unwrap()
    })
}

fn first_person_acquired_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:i|we)\s+
            (?:(?:just|recently|finally|already|also|actually)\s+)?
            (?:got|bought|purchased|ordered|picked\s+up)\s+
            (?:(?:him|her|them|someone|somebody)\s+)?
            (?:(?:a|an|the|my|our|new|some|\d+)\b)
            ",
        )
        .unwrap()
    })
}

fn first_person_acquisition_source_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:i|we)\b
            .{0,120}?
            \b(?:got|bought|purchased|ordered|picked\s+up)\b
            .{0,120}?
            \b(?:from|at|via|through)\s+
            (?:a|an|the|my|our|new|some)?\s*
            [A-Za-z0-9][A-Za-z0-9'&.-]{1,}
            ",
        )
        .unwrap()
    })
}

fn ownership_source_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:(?:my|our|the)\s+)?
            (?:new|current|latest|recent)\s+
            [A-Za-z0-9][A-Za-z0-9'&.-]{2,}(?:\s+[A-Za-z0-9][A-Za-z0-9'&.-]{2,}){0,3}
            \s+(?:is|are|was|were|came)\s+from\s+
            [A-Za-z0-9][A-Za-z0-9'&.-]{1,}
            ",
        )
        .unwrap()
    })
}

fn first_person_competition_event_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:i|we)\s+
            (?:
                (?:just\s+|recently\s+)?(?:completed|finished|ran|raced|joined|entered|played|participated\s+in|participate\s+in) |
                (?:will|am\s+going\s+to|are\s+going\s+to|'m\s+going\s+to|'re\s+going\s+to|’m\s+going\s+to|’re\s+going\s+to|plan\s+to|planning\s+to)\s+(?:participate\s+in|play|run|race|join|enter) |
                (?:am|are|'m|'re|’m|’re)\s+(?:participating\s+in|playing|running|racing)
            )
            .{0,80}
            \b(?:tournament|triathlon|marathon|half\s+marathon|race|run|5k|10k|match|game|competition|bike\s+ride)\b",
        )
        .unwrap()
    })
}

fn first_person_with_companion_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:i|we)\b
            .{0,140}?
            \b(?:went|go|going|attended|saw|seen|visited|joined|participated|did|completed|watched|traveled|travelled|been)\b
            .{0,140}?
            \bwith\s+
            (?:(?:my|our|a|an|the|some|a\s+group\s+of)\s+)?
            [A-Za-z0-9][A-Za-z0-9'&.-]{1,}
            ",
        )
        .unwrap()
    })
}

fn companion_query_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \bwho\s+(?:(?:did|do|will)\s+)?(?:i|we)\b.{0,90}\bwith\b
            |
            \bwho\s+(?:was|were)\s+with\s+(?:me|us)\b
            |
            \bwith\s+whom\b
            ",
        )
        .unwrap()
    })
}

fn first_person_completed_clean_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:i|we)\b
            (?:
                .{0,50}?\b(?:cleaned|washed|polished|conditioned)\b
                |
                .{0,70}?\b(?:got\s+around\s+to|finished|finally\s+finished|remember\s+I|remember\s+we)\s+
                    (?:cleaning|washing|polishing|conditioning)\b
            )
            .{0,100}?
            \b(?:my|our|the|a|an|this|that|these|those)\b
            ",
        )
        .unwrap()
    })
}

fn completed_clean_query_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:what|which|when|where|did|have|had)\b
            .{0,80}?
            \b(?:i|we)\b
            .{0,60}?
            \b(?:clean|cleaned|cleaning|wash|washed|washing|polish|polished|polishing|condition|conditioned|conditioning)\b
            ",
        )
        .unwrap()
    })
}

fn co_residence_with_self_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:live|lived|living|stay|stayed|staying)\s+with\s+(?:me|us)\b
            |
            \b[A-Za-z][A-Za-z'-]{0,24}(?:(?:\s+(?:has|have|had))|(?:['’](?:ve|d)))\s+
                been\s+(?:living|staying)\s+with\s+(?:me|us)\b
            ",
        )
        .unwrap()
    })
}

fn co_residence_query_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:live|lived|living|stay|stayed|staying)\s+with\s+(?:me|us)\b
            |
            \bbeen\s+(?:living|staying)\s+with\s+(?:me|us)\b
            ",
        )
        .unwrap()
    })
}

fn first_person_project_work_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:
                (?:i|we)(?:\s+)?
                (?:
                    (?:am|are|'m|'re|’m|’re|was|were|have\s+been|has\s+been|'ve\s+been|’ve\s+been)\s+
                    (?:working\s+on|leading|managing|running|building|developing|researching|presenting)
                    |
                    (?:led|managed|ran|built|developed|presented|researched|participated\s+in)
                )
                .{0,80}
                \b(?:project|research|campaign|initiative|case\s+competition|poster|feature)\b
                |
                (?:my|our)\s+(?:current\s+|latest\s+|new\s+|solo\s+|research\s+)?(?:project|research|campaign|initiative)
            )\b",
        )
        .unwrap()
    })
}

fn decision_selection_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:
                (?:i|we)\s+(?:choose|chose|pick|picked|select|selected|decided|settled\s+on|went\s+with) |
                (?:let's|lets)\s+(?:go\s+with|call\s+it|name\s+it) |
                [a-z0-9][a-z0-9'_-]*(?:\s+[a-z0-9][a-z0-9'_-]*){0,5}\s+
                    is\s+(?:a|an|the)?\s*(?:[a-z0-9'_-]+\s+){0,5}(?:one|choice|name|pick)
            )\b",
        )
        .unwrap()
    })
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

fn entity_class_relation_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\b(?P<class>(?:[A-Z][A-Za-z'-]{2,}|[a-z][A-Za-z'-]{2,})(?:\s+(?:[A-Z][A-Za-z'-]{2,}|[a-z][A-Za-z'-]{2,})){0,3})\s+(?i:like|such\s+as)\s+(?P<name>[A-Z][A-Za-z'-]{1,40})\b",
        )
        .unwrap()
    })
}

fn preferred_attribute_relation_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\b(?P<name>[A-Z][A-Za-z0-9'&.-]{1,40}(?:\s+[A-Z][A-Za-z0-9'&.-]{1,40}){0,3})\s+(?i:has\s+been|have\s+been|is|are|was|were|became|remains)\s+(?:(?:my|our|the)\s+)?(?i:favou?rite|preferred)\s+(?P<class>[A-Za-z][A-Za-z'-]{2,}(?:\s+[A-Za-z][A-Za-z'-]{2,}){0,2})\b",
        )
        .unwrap()
    })
}

fn titled_person_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b(?P<title>Dr|Prof)\.?\s+(?P<name>[A-Z][A-Za-z'-]{1,40})\b").unwrap()
    })
}

fn role_before_title_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\b(?P<role>[A-Z]{2,8}|[A-Za-z][A-Za-z'-]{2,})(?:\s+(?P<role2>[A-Za-z][A-Za-z'-]{2,}|[A-Z]{2,8})){0,3}\s+(?i:Dr|Prof)\.?\s+[A-Z][A-Za-z'-]{1,40}\b",
        )
        .unwrap()
    })
}

fn possessed_role_before_title_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\b(?i:my|our|the|a|an)\s+(?P<role>[A-Za-z][A-Za-z'-]{2,}(?:\s+[A-Za-z][A-Za-z'-]{2,}){0,3})\s+(?i:Dr|Prof)\.?\s+[A-Z][A-Za-z'-]{1,40}\b",
        )
        .unwrap()
    })
}

fn role_before_name_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\b(?P<role>[A-Za-z][A-Za-z'-]{2,}(?:\s+[A-Za-z][A-Za-z'-]{2,}){0,2})\s+[A-Z][A-Za-z'-]{1,40}\b",
        )
        .unwrap()
    })
}

fn sibling_count_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:(?:i|we)\s+(?:have|had|have\s+got|'ve\s+got|’ve\s+got|got)\s+(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:older\s+|younger\s+|little\s+|big\s+|half\s+|step\s+)?(?:brothers?|sisters?|siblings?)|(?:come\s+from|grew\s+up\s+in|part\s+of)\s+(?:a\s+)?family\s+with\s+(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:older\s+|younger\s+|little\s+|big\s+|half\s+|step\s+)?(?:brothers?|sisters?|siblings?))\b",
        )
        .unwrap()
    })
}

fn self_sibling_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:(?:my|our)\s+(?:older\s+|younger\s+|little\s+|big\s+|half\s+|step\s+)?(?:brothers?|sisters?|siblings?)|(?:i|we)\s+(?:have|had|have\s+got|'ve\s+got|’ve\s+got|got)\s+.{0,40}\b(?:brothers?|sisters?|siblings?)|(?:come\s+from|grew\s+up\s+in|part\s+of)\s+(?:a\s+)?family\s+with\s+.{0,40}\b(?:brothers?|sisters?|siblings?))\b",
        )
        .unwrap()
    })
}

fn possessive_object_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:my|our)\s+(?P<object>[A-Za-z0-9][A-Za-z0-9'-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'-]*){0,6})",
        )
        .unwrap()
    })
}

fn owned_object_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:i|we)\s+(?:currently\s+)?
            (?:
                (?:have|got)\s+(?:a|an|the|this|that|these|those|my|our|some|\d+)\s+
                (?P<object_det>[A-Za-z0-9][A-Za-z0-9'-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'-]*){0,4})
                |
                (?:own|use|keep|maintain|set\s+up)\s+
                (?P<object_direct>[A-Za-z0-9][A-Za-z0-9'-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'-]*){0,4})
            )",
        )
        .unwrap()
    })
}

fn owned_object_capture_text<'a>(cap: &'a regex::Captures<'a>) -> Option<&'a str> {
    cap.name("object_det")
        .or_else(|| cap.name("object_direct"))
        .map(|m| m.as_str())
}

fn first_person_possession_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:
                (?:i|we)(?:\s+have|\s+had|'ve|’ve)\s+had\s+(?:my|our)\s+ |
                (?:i|we)(?:'ve|’ve|\s+have)?\s+been\s+(?:playing|using|wearing|driving|riding|keeping|maintaining)\s+(?:my|our)\s+ |
                (?:i|we)(?:'m|’m|\s+am|\s+was|\s+are|\s+were)?\s*(?:thinking\s+of|planning\s+to|trying\s+to|looking\s+to)?\s*(?:sell|selling|sold)\s+(?:my|our)\s+ |
                which\s+(?:i|we)(?:'ve|’ve|\s+have|\s+had)?\s+had\b
            )",
        )
        .unwrap()
    })
}

fn homegrown_source_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \bhome[-\s]?grown\b |
            \b(?:harvest(?:ed|ing)?|grew|grown|growing|picked|planted)\b.{0,80}
                \b(?:garden|yard|backyard|planters?|raised\s+beds?|greenhouse|farm|balcony)\b |
            \b(?:garden|yard|backyard|planters?|raised\s+beds?|greenhouse|farm|balcony)\b.{0,80}
                \b(?:harvest(?:ed|ing)?|grew|grown|growing|picked|planted)\b
            ",
        )
        .unwrap()
    })
}

fn ingredient_context_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)\b(?:ingredients?|recipes?|cooking|cook(?:ing|ed)?|baking|baked|meals?|dinner|lunch|breakfast|dish(?:es)?|served?)\b",
        )
        .unwrap()
    })
}

fn is_entity_noise(raw: &str) -> bool {
    let normalized = raw.trim().to_ascii_lowercase();
    if crate::nl::get_stopwords().contains(normalized.as_str()) {
        return true;
    }
    matches!(
        normalized.as_str(),
        "i" | "the"
            | "a"
            | "an"
            | "this"
            | "that"
            | "here"
            | "there"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "whom"
            | "whose"
            | "why"
            | "how"
            | "do"
            | "does"
            | "did"
            | "can"
            | "could"
            | "should"
            | "would"
            | "will"
            | "please"
            | "tell"
            | "user"
            | "assistant"
            | "system"
            | "human"
            | "bot"
            | "agent"
    )
}

fn product_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b[A-Za-z]{1,12}[- ]?[A-Z]?\d[A-Za-z0-9-]{1,12}(?:\s+[A-Z]{1,4})?\b").unwrap()
    })
}

fn list_item_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\d+[\.)]\s+").unwrap())
}

fn inline_list_item_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?:^|\s)\d{1,2}[\.)]\s+\S").unwrap())
}

fn quantity_unit_object_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?P<value>\d+(?:[.,]\d+)?)\s*[- ]\s*(?P<unit>[A-Za-z][A-Za-z]{1,20})\s+(?P<object>[A-Za-z][A-Za-z'-]{2,}(?:\s+[A-Za-z][A-Za-z'-]{2,}){0,3})",
        )
        .unwrap()
    })
}

fn quantity_object_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?P<value>\d+(?:[.,]\d+)?)\s+(?P<object>[A-Za-z][A-Za-z'-]{2,}(?:\s+[A-Za-z][A-Za-z'-]{2,}){0,3})",
        )
        .unwrap()
    })
}

fn contained_singular_object_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:has|have|contains|contain|includes|include)\s+(?:my|our|a|an|one)\s+(?P<object>[A-Za-z][A-Za-z'-]{2,}(?:\s+[A-Za-z][A-Za-z'-]{2,}){0,3})",
        )
        .unwrap()
    })
}

fn completed_count_object_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(?:completed|finished|passed|took|taken)\s+
            (?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+
            (?P<object>[A-Za-z][A-Za-z'-]{2,}(?:\s+[A-Za-z][A-Za-z'-]{2,}){0,3})
            ",
        )
        .unwrap()
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

fn role_phrase_stopwords() -> &'static HashSet<&'static str> {
    static STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOPWORDS.get_or_init(|| {
        HashSet::from([
            "a", "an", "and", "as", "at", "by", "for", "from", "had", "has", "have", "i", "in",
            "is", "it", "me", "my", "of", "on", "or", "our", "saw", "the", "to", "was", "with",
        ])
    })
}

fn normalize_role_phrase(value: &str) -> Option<String> {
    let stopwords = role_phrase_stopwords();
    let mut parts = Vec::new();

    for raw in value.split_whitespace() {
        let part = raw
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_ascii_lowercase();
        if part.len() < 2 || stopwords.contains(part.as_str()) {
            continue;
        }
        parts.push(part);
    }

    if parts.is_empty() || parts.len() > 4 {
        return None;
    }

    let phrase = parts.join("_");
    if phrase.len() < 3 || phrase.len() > 64 {
        None
    } else {
        Some(phrase)
    }
}

fn quantity_object_stopwords() -> &'static HashSet<&'static str> {
    static STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOPWORDS.get_or_init(|| {
        HashSet::from([
            "a", "an", "and", "ago", "am", "at", "by", "day", "days", "dollar", "dollars", "for",
            "from", "episode", "episodes", "gallon", "gallons", "hour", "hours", "in", "inch",
            "inches", "meter", "meters", "minute", "minutes", "month", "months", "named", "of",
            "on", "or", "percent", "pm", "season", "seasons", "second", "seconds", "the", "to",
            "week", "weeks", "with", "year", "years",
        ])
    })
}

fn inventory_object_stopwords() -> &'static HashSet<&'static str> {
    static STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOPWORDS.get_or_init(|| {
        let mut stopwords = quantity_object_stopwords().clone();
        stopwords.extend([
            "few", "issue", "issues", "level", "levels", "lot", "lots", "many", "more", "new",
            "old", "problem", "problems", "some",
        ]);
        stopwords
    })
}

fn normalize_quantity_token(raw: &str) -> Option<String> {
    let normalized = normalize_value(raw)?;
    if quantity_object_stopwords().contains(normalized.as_str()) {
        return None;
    }
    let stemmed = crate::nl::stem_word(&normalized);
    if stemmed.len() < 2 || quantity_object_stopwords().contains(stemmed.as_str()) {
        None
    } else {
        Some(stemmed)
    }
}

fn normalize_quantity_unit(raw: &str) -> Option<String> {
    let normalized = normalize_value(raw)?;
    let stemmed = crate::nl::stem_word(&normalized);
    if stemmed.len() < 2 {
        None
    } else {
        Some(stemmed)
    }
}

fn quantity_object_tokens(phrase: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut seen = HashSet::new();

    for raw in phrase.split_whitespace().take(4) {
        let normalized = normalize_value(raw);
        if normalized
            .as_deref()
            .map(|token| quantity_object_stopwords().contains(token))
            .unwrap_or(false)
        {
            break;
        }

        let Some(token) = normalize_quantity_token(raw) else {
            break;
        };
        if seen.insert(token.clone()) {
            tokens.push(token);
        }
    }

    tokens
}

fn inventory_object_tokens(phrase: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut seen = HashSet::new();
    let stopwords = inventory_object_stopwords();

    for raw in phrase.split_whitespace().take(5) {
        let Some(token) = normalize_quantity_token(raw) else {
            continue;
        };
        if stopwords.contains(token.as_str()) {
            continue;
        }
        if seen.insert(token.clone()) {
            tokens.push(token);
        }
    }

    tokens
}

fn numeric_value_looks_like_year(value: &str) -> bool {
    let integer = value.split(['.', ',']).next().unwrap_or(value);
    integer.len() == 4
        && integer
            .parse::<u16>()
            .map(|year| (1900..=2100).contains(&year))
            .unwrap_or(false)
}

fn numeric_value_looks_like_age(value: &str) -> bool {
    value
        .split(['.', ','])
        .next()
        .unwrap_or(value)
        .parse::<u16>()
        .map(|age| (1..=130).contains(&age))
        .unwrap_or(false)
}

fn has_valid_age_match(regex: &Regex, content: &str) -> bool {
    regex.captures_iter(content).any(|cap| {
        cap.name("age")
            .map(|m| numeric_value_looks_like_age(m.as_str()))
            .unwrap_or(false)
    })
}

fn add_numeric_object_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    for cap in quantity_unit_object_re().captures_iter(content) {
        if cap
            .name("value")
            .map(|m| numeric_value_looks_like_year(m.as_str()))
            .unwrap_or(false)
        {
            continue;
        }

        let Some(unit) = cap
            .name("unit")
            .and_then(|m| normalize_quantity_unit(m.as_str()))
        else {
            continue;
        };
        let object_tokens = cap
            .name("object")
            .map(|m| quantity_object_tokens(m.as_str()))
            .unwrap_or_default();
        if object_tokens.is_empty() {
            continue;
        }

        push_unique(out, seen, format!("quantity_unit:{}", unit));
        for object in &object_tokens {
            push_unique(out, seen, format!("quantity_object:{}", object));
        }
        if let Some(head) = object_tokens.last() {
            push_unique(out, seen, format!("quantity_unit_object:{}_{}", unit, head));
        }
    }

    for cap in quantity_object_re().captures_iter(content) {
        if cap
            .name("value")
            .map(|m| numeric_value_looks_like_year(m.as_str()))
            .unwrap_or(false)
        {
            continue;
        }

        let object_tokens = cap
            .name("object")
            .map(|m| quantity_object_tokens(m.as_str()))
            .unwrap_or_default();
        if !object_tokens.is_empty() {
            push_unique(out, seen, "quantity_count:object");
            let preceding = &content[..cap.get(0).map(|m| m.start()).unwrap_or(0)];
            let preceding_window = preceding
                .chars()
                .rev()
                .take(80)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>()
                .to_lowercase();
            if has_any(
                &format!(" {} ", preceding_window),
                &[
                    " has ",
                    " have ",
                    " contains ",
                    " contain ",
                    " includes ",
                    " include ",
                ],
            ) {
                push_unique(out, seen, "inventory_count:contained");
            }
        }
        for object in object_tokens {
            push_unique(out, seen, format!("quantity_object:{}", object));
        }
    }

    for cap in contained_singular_object_re().captures_iter(content) {
        let object_tokens = cap
            .name("object")
            .map(|m| quantity_object_tokens(m.as_str()))
            .unwrap_or_default();
        if object_tokens.is_empty() {
            continue;
        }
        push_unique(out, seen, "quantity_count:object");
        push_unique(out, seen, "inventory_count:contained");
        for object in object_tokens {
            push_unique(out, seen, format!("quantity_object:{}", object));
        }
    }

    for cap in completed_count_object_re().captures_iter(content) {
        if cap
            .name("value")
            .and_then(|m| small_number_value(&m.as_str().to_ascii_lowercase()))
            .is_none()
        {
            continue;
        }
        let object_tokens = cap
            .name("object")
            .map(|m| quantity_object_tokens(m.as_str()))
            .unwrap_or_default();
        if object_tokens.is_empty() {
            continue;
        }
        push_unique(out, seen, "quantity_count:object");
        push_unique(out, seen, "completion_count:object");
        for object in object_tokens {
            push_unique(out, seen, format!("quantity_object:{}", object));
        }
    }
}

fn add_inventory_object_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = format!(" {} ", content.to_lowercase());
    let has_ownership_signal = first_person_acquired_re().is_match(&lower)
        || first_person_possession_re().is_match(&lower)
        || owned_object_re().is_match(content);

    for cap in possessive_object_re().captures_iter(content) {
        let object_tokens = cap
            .name("object")
            .map(|m| inventory_object_tokens(m.as_str()))
            .unwrap_or_default();
        for object in object_tokens {
            push_unique(out, seen, format!("inventory_object:{}", object));
        }
    }

    for cap in owned_object_re().captures_iter(content) {
        let object_tokens = owned_object_capture_text(&cap)
            .map(inventory_object_tokens)
            .unwrap_or_default();
        for object in object_tokens {
            push_unique(out, seen, format!("inventory_object:{}", object));
        }
    }

    if has_ownership_signal {
        for cap in quantity_unit_object_re().captures_iter(content) {
            let object_tokens = cap
                .name("object")
                .map(|m| inventory_object_tokens(m.as_str()))
                .unwrap_or_default();
            for object in object_tokens {
                push_unique(out, seen, format!("inventory_object:{}", object));
            }
        }

        for cap in quantity_object_re().captures_iter(content) {
            let object_tokens = cap
                .name("object")
                .map(|m| inventory_object_tokens(m.as_str()))
                .unwrap_or_default();
            for object in object_tokens {
                push_unique(out, seen, format!("inventory_object:{}", object));
            }
        }
    }
}

fn add_age_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let has_year_old = has_valid_age_match(age_year_old_re(), content);
    let has_current_age = has_valid_age_match(current_age_re(), content);
    let has_event_age = has_valid_age_match(event_age_re(), content);

    if has_year_old || has_current_age || has_event_age {
        push_unique(out, seen, "has:age");
    }
    if has_current_age {
        push_unique(out, seen, "age:current");
    }
    if has_event_age {
        push_unique(out, seen, "age:event");
    }
    if has_year_old && !has_current_age {
        push_unique(out, seen, "age:mentioned");
    }
}

fn add_education_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = format!(" {} ", content.to_lowercase());
    let undergraduate = has_any(
        &lower,
        &[
            " undergraduate",
            " undergrad",
            " bachelor's",
            " bachelors",
            " bachelor ",
        ],
    );
    let degree = has_any(
        &lower,
        &[
            " degree",
            " bachelor's",
            " bachelors",
            " bachelor ",
            " undergraduate",
            " undergrad",
            " master's",
            " masters",
            " master ",
            " mba",
            " ph.d",
            " phd",
            " doctorate",
            " diploma",
            " associate degree",
        ],
    );
    let institution = has_any(&lower, &[" college", " university", " school"]);
    let completion = has_any(
        &lower,
        &[
            " graduated",
            " graduation",
            " completed",
            " finished",
            " earned",
            " received",
        ],
    );
    let graduation = has_any(
        &lower,
        &[
            " graduated from",
            " graduation from",
            " finished college",
            " finished university",
            " completed college",
            " completed university",
        ],
    ) || (degree && completion);

    if degree {
        push_unique(out, seen, "education:degree");
    }
    if undergraduate {
        push_unique(out, seen, "education:undergraduate");
    }
    if institution {
        push_unique(out, seen, "education:college");
    }
    if graduation {
        push_unique(out, seen, "education:graduation");
    }
}

fn family_relation_for_token(token: &str) -> Option<(&'static str, Option<&'static str>)> {
    match token {
        "brother" | "brothers" => Some(("sibling", Some("brother"))),
        "sister" | "sisters" => Some(("sibling", Some("sister"))),
        "sibling" | "siblings" => Some(("sibling", None)),
        "mother" | "mom" | "mum" | "father" | "dad" | "parent" | "parents" => {
            Some(("parent", None))
        }
        "son" | "sons" | "daughter" | "daughters" | "child" | "children" | "kid"
        | "kids" => Some(("child", None)),
        "husband" | "wife" | "spouse" => Some(("spouse", None)),
        "cousin" | "cousins" => Some(("cousin", None)),
        "aunt" | "aunts" | "uncle" | "uncles" => Some(("aunt_uncle", None)),
        "niece" | "nieces" | "nephew" | "nephews" => Some(("niece_nephew", None)),
        "grandmother" | "grandma" | "grandfather" | "grandpa" | "grandparent"
        | "grandparents" => Some(("grandparent", None)),
        _ => None,
    }
}

fn add_family_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = content.to_lowercase();
    let tokens = query_tokens(&lower);
    let mut saw_sibling = false;

    for token in tokens {
        let Some((relation, kind)) = family_relation_for_token(token.as_str()) else {
            continue;
        };
        push_unique(out, seen, format!("family_relation:{}", relation));
        if relation == "sibling" {
            saw_sibling = true;
        }
        if let Some(kind) = kind {
            if relation == "sibling" {
                push_unique(out, seen, format!("sibling_kind:{}", kind));
            }
        }
    }

    if saw_sibling && self_sibling_re().is_match(content) {
        push_unique(out, seen, "family_scope:self");
    }
    if saw_sibling && sibling_count_re().is_match(content) {
        push_unique(out, seen, "family_count:sibling");
        push_unique(out, seen, "family_scope:self");
    }
}

fn family_relation_query_facets(query: &str) -> Vec<String> {
    let mut facets = Vec::new();
    let mut seen = HashSet::new();
    for token in query_tokens(query) {
        let Some((relation, kind)) = family_relation_for_token(token.as_str()) else {
            continue;
        };
        let relation_facet = format!("family_relation:{}", relation);
        if seen.insert(relation_facet.clone()) {
            facets.push(relation_facet);
        }
        if relation == "sibling" {
            if let Some(kind) = kind {
                let kind_facet = format!("sibling_kind:{}", kind);
                if seen.insert(kind_facet.clone()) {
                    facets.push(kind_facet);
                }
            }
        }
    }
    facets
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

fn standing_always_when_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?is)\balways\s+(?P<action>.{3,220}?)\s+when\s+(?:i|we|the user|users?|someone|people)\s+(?:ask|asks|asked|am asking|are asking)\s+about\s+(?P<trigger>.{3,220})(?:[.!?]|$)",
        )
        .unwrap()
    })
}

fn standing_when_always_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?is)\bwhen\s+(?:i|we|the user|users?|someone|people)\s+(?:ask|asks|asked|am asking|are asking)\s+about\s+(?P<trigger>.{3,220}?)\s*,?\s+(?:always|please|make sure to|remember to)\s+(?P<action>.{3,220})(?:[.!?]|$)",
        )
        .unwrap()
    })
}

fn standing_make_sure_when_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?is)\b(?:make sure to|remember to|please)\s+(?P<action>.{3,220}?)\s+when\s+(?:discussing|covering|answering|responding to|talking about|i\s+ask\s+about|we\s+ask\s+about)\s+(?P<trigger>.{3,220})(?:[.!?]|$)",
        )
        .unwrap()
    })
}

fn standing_instruction_clause_trim(clause: &str) -> String {
    clause
        .split("->->")
        .next()
        .unwrap_or(clause)
        .split("```")
        .next()
        .unwrap_or(clause)
        .lines()
        .next()
        .unwrap_or(clause)
        .trim_matches(|ch: char| ch.is_whitespace() || ch == '"' || ch == '\'' || ch == ',' || ch == ';' || ch == ':')
        .trim()
        .to_string()
}

fn standing_instruction_cue_is_generic(cue: &str) -> bool {
    if cue.starts_with("type:")
        || cue.starts_with("has:")
        || cue.starts_with("temporal:")
        || cue.starts_with("source_")
    {
        return true;
    }

    if cue.contains('_') {
        let parts = cue.split('_').collect::<Vec<_>>();
        return parts.iter().all(|part| standing_instruction_cue_is_generic(part));
    }

    crate::nl::get_stopwords().contains(cue)
        || matches!(
            cue,
            "always"
                | "ask"
                | "asked"
                | "asking"
                | "about"
                | "when"
                | "provide"
                | "include"
                | "specify"
                | "confirm"
                | "explain"
                | "remember"
                | "make"
                | "sure"
                | "please"
                | "tell"
                | "show"
                | "help"
                | "advice"
                | "thing"
                | "things"
                | "way"
                | "ways"
        )
}

fn add_standing_instruction_clause_cues(
    prefix: &str,
    clause: &str,
    limit: usize,
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    let clause = standing_instruction_clause_trim(clause);
    if clause.len() < 3 {
        return;
    }

    let mut emitted = HashSet::new();
    for cue in crate::nl::tokenize_to_cues(&clause) {
        let cue = cue.trim().to_lowercase();
        if cue.len() <= 2
            || standing_instruction_cue_is_generic(&cue)
            || !emitted.insert(cue.clone())
        {
            continue;
        }
        push_unique(out, seen, format!("{prefix}:{cue}"));
        if emitted.len() >= limit {
            break;
        }
    }
}

fn add_standing_instruction_dynamic_facets(
    content: &str,
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    let patterns = [
        standing_always_when_re(),
        standing_when_always_re(),
        standing_make_sure_when_re(),
    ];

    for re in patterns {
        let Some(caps) = re.captures(content) else {
            continue;
        };
        if let Some(trigger) = caps.name("trigger") {
            add_standing_instruction_clause_cues(
                "instruction_trigger",
                trigger.as_str(),
                10,
                out,
                seen,
            );
        }
        if let Some(action) = caps.name("action") {
            add_standing_instruction_clause_cues(
                "instruction_action",
                action.as_str(),
                8,
                out,
                seen,
            );
        }
        break;
    }
}

fn preference_over_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?is)\b(?:i|we)\s+(?:really\s+|usually\s+|generally\s+|strongly\s+)?(?:prefer|like|love|enjoy)\s+(?P<value>.{3,180}?)\s+\b(?:over|rather than|instead of)\b\s+(?P<contrast>.{3,180})(?:[.!?]|$)",
        )
        .unwrap()
    })
}

fn preference_for_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?is)\b(?:i|we)\s+(?:really\s+|usually\s+|generally\s+|strongly\s+)?(?:prefer|like|love|enjoy)\s+(?P<value>.{3,180}?)\s+\b(?:for|when|while|with|in|during)\b\s+(?P<topic>.{3,180})(?:[.!?]|$)",
        )
        .unwrap()
    })
}

fn preference_simple_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?is)\b(?:i|we)\s+(?:really\s+|usually\s+|generally\s+|strongly\s+)?(?:prefer|like|love|enjoy)\s+(?P<value>.{3,180})(?:[.!?]|$)",
        )
        .unwrap()
    })
}

fn preference_rather_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?is)\b(?:i|we)\s+would\s+rather\s+(?P<value>.{3,180})(?:[.!?]|$)",
        )
        .unwrap()
    })
}

fn preference_negative_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?is)\b(?:i|we)\s+(?:do\s+not|don't|dont|dislike|avoid|can't\s+stand|cannot\s+stand)\s+(?P<value>.{3,180})(?:[.!?]|$)",
        )
        .unwrap()
    })
}

fn preference_clause_trim(clause: &str) -> String {
    clause
        .split("->->")
        .next()
        .unwrap_or(clause)
        .split("```")
        .next()
        .unwrap_or(clause)
        .lines()
        .next()
        .unwrap_or(clause)
        .split(|ch| ch == ',' || ch == ';')
        .next()
        .unwrap_or(clause)
        .trim_matches(|ch: char| {
            ch.is_whitespace() || ch == '"' || ch == '\'' || ch == ',' || ch == ';' || ch == ':'
        })
        .trim()
        .to_string()
}

fn preference_cue_is_generic(cue: &str) -> bool {
    if cue.starts_with("type:")
        || cue.starts_with("has:")
        || cue.starts_with("temporal:")
        || cue.starts_with("source_")
    {
        return true;
    }

    if cue.contains('_') {
        let parts = cue.split('_').collect::<Vec<_>>();
        return parts.iter().all(|part| preference_cue_is_generic(part));
    }

    crate::nl::get_stopwords().contains(cue)
        || matches!(
            cue,
            "prefer"
                | "preference"
                | "like"
                | "love"
                | "enjoy"
                | "want"
                | "need"
                | "rather"
                | "over"
                | "instead"
                | "for"
                | "when"
                | "while"
                | "with"
                | "about"
                | "because"
                | "can"
                | "could"
                | "would"
                | "should"
                | "show"
                | "tell"
                | "help"
                | "explain"
                | "recommend"
                | "suggest"
                | "use"
                | "using"
                | "get"
                | "make"
                | "thing"
                | "things"
                | "way"
                | "ways"
        )
}

fn add_preference_clause_cues(
    prefix: &str,
    clause: &str,
    limit: usize,
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    let clause = preference_clause_trim(clause);
    if clause.len() < 3 {
        return;
    }

    let mut emitted = HashSet::new();
    for cue in crate::nl::tokenize_to_cues(&clause) {
        let cue = cue.trim().to_lowercase();
        if cue.len() <= 2 || preference_cue_is_generic(&cue) || !emitted.insert(cue.clone()) {
            continue;
        }
        push_unique(out, seen, format!("{prefix}:{cue}"));
        if emitted.len() >= limit {
            break;
        }
    }
}

fn add_preference_dynamic_facets(
    content: &str,
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    let mut matched = false;

    if let Some(caps) = preference_over_re().captures(content) {
        if let Some(value) = caps.name("value") {
            add_preference_clause_cues("preference_value", value.as_str(), 10, out, seen);
        }
        if let Some(contrast) = caps.name("contrast") {
            add_preference_clause_cues("preference_contrast", contrast.as_str(), 8, out, seen);
        }
        matched = true;
    }

    if let Some(caps) = preference_for_re().captures(content) {
        if let Some(value) = caps.name("value") {
            add_preference_clause_cues("preference_value", value.as_str(), 10, out, seen);
        }
        if let Some(topic) = caps.name("topic") {
            add_preference_clause_cues("preference_topic", topic.as_str(), 10, out, seen);
        }
        matched = true;
    }

    if let Some(caps) = preference_rather_re().captures(content) {
        if let Some(value) = caps.name("value") {
            add_preference_clause_cues("preference_value", value.as_str(), 10, out, seen);
        }
        matched = true;
    }

    if let Some(caps) = preference_negative_re().captures(content) {
        if let Some(value) = caps.name("value") {
            add_preference_clause_cues("preference_contrast", value.as_str(), 10, out, seen);
        }
        matched = true;
    }

    if !matched {
        if let Some(caps) = preference_simple_re().captures(content) {
            if let Some(value) = caps.name("value") {
                add_preference_clause_cues("preference_value", value.as_str(), 10, out, seen);
            }
        }
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
    if let Some(seconds) = value.as_f64() {
        let days = (seconds / 86_400.0).floor() as i64;
        return NaiveDate::from_ymd_opt(1970, 1, 1)?.checked_add_signed(Duration::days(days));
    }
    None
}

fn metadata_date(metadata: &HashMap<String, Value>) -> Option<NaiveDate> {
    for key in [
        "source_date",
        "source_timestamp",
        "timestamp",
        "created_at",
        "datetime",
        "date",
    ] {
        if let Some(date) = metadata.get(key).and_then(parse_date_value) {
            return Some(date);
        }
    }
    None
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

fn content_month_facet(month: u32) -> Option<String> {
    if (1..=12).contains(&month) {
        Some(format!("content_month:{:02}", month))
    } else {
        None
    }
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

fn explicit_query_months(lower: &str) -> Vec<u32> {
    let mut months = Vec::new();
    let mut seen = HashSet::new();
    for token in query_tokens(lower) {
        if let Some(month) = month_name_number(&token) {
            if seen.insert(month) {
                months.push(month);
            }
        }
    }
    months
}

fn previous_calendar_month_date(reference_date: NaiveDate) -> Option<NaiveDate> {
    let (year, month) = if reference_date.month() == 1 {
        (reference_date.year() - 1, 12)
    } else {
        (reference_date.year(), reference_date.month() - 1)
    };
    NaiveDate::from_ymd_opt(year, month, 1)
}

fn most_recent_weekend_dates(reference_date: NaiveDate) -> Option<(NaiveDate, NaiveDate)> {
    let days_since_sunday = reference_date.weekday().num_days_from_sunday() as i64;
    let sunday = reference_date.checked_sub_signed(Duration::days(days_since_sunday))?;
    let saturday = sunday.checked_sub_signed(Duration::days(1))?;
    Some((saturday, sunday))
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
        if let Some(role) =
            metadata_string(metadata, &["source_role", "role", "speaker", "author_role"])
        {
            if let Some(value) = normalize_value(role) {
                push_unique(out, seen, format!("source_role:{}", value));
            }
        }
        if let Some(channel) = metadata_string(
            metadata,
            &["source_channel", "channel", "conversation", "room"],
        ) {
            if let Some(value) = normalize_value(channel) {
                push_unique(out, seen, format!("source_channel:{}", value));
            }
        }
        if let Some(source_type) = metadata_string(metadata, &["source_type", "source", "kind"]) {
            if let Some(value) = normalize_value(source_type) {
                push_unique(out, seen, format!("source_type:{}", value));
            }
        }
        if let Some(session) = metadata_string(
            metadata,
            &[
                "source_session_id",
                "session_id",
                "conversation_id",
                "thread_id",
            ],
        ) {
            if let Some(value) = normalize_value(session) {
                push_unique(out, seen, format!("source_session:{}", value));
            }
        }
    }

    for cue in existing_cues {
        if let Some((key, value)) = cue.split_once(':') {
            let target = match key {
                "role" | "speaker" => Some("source_role"),
                "channel" => Some("source_channel"),
                "source" | "category" => Some("source_type"),
                "session" | "conversation" | "thread" => Some("source_session"),
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
    let has_short_numeric_date = short_numeric_date_re().is_match(content);
    if date_re().is_match(content) || has_short_numeric_date {
        push_unique(out, seen, "has:date");
    }
    for token in query_tokens(content) {
        if let Some(month) = month_name_number(&token) {
            if let Some(facet) = content_month_facet(month) {
                push_unique(out, seen, facet);
            }
        }
    }
    for cap in short_numeric_date_re().captures_iter(content) {
        let Some(month) = cap
            .name("month")
            .and_then(|m| m.as_str().parse::<u32>().ok())
        else {
            continue;
        };
        let Some(day) = cap
            .name("day")
            .and_then(|m| m.as_str().parse::<u32>().ok())
        else {
            continue;
        };
        if day <= 31 {
            if let Some(facet) = content_month_facet(month) {
                push_unique(out, seen, facet);
            }
        }
    }
    if duration_re().is_match(content) {
        push_unique(out, seen, "has:duration");
    }
    let has_weekday = weekday_re().is_match(content);
    if has_weekday {
        push_unique(out, seen, "has:weekday");
        push_unique(out, seen, "schedule:weekly");
    }
    if clock_time_re().is_match(content) {
        push_unique(out, seen, "has:time");
    }
    add_time_of_day_facets(content, out, seen);
    add_cadence_facets(content, out, seen);

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
    let inline_markers = inline_list_item_re().find_iter(content).take(3).count();
    if list_markers >= 2 || inline_markers >= 3 {
        push_unique(out, seen, "has:list");
    }
}

fn clock_hour_24(hour: u32, meridiem: &str) -> u32 {
    let meridiem = meridiem.to_ascii_lowercase();
    if meridiem.starts_with('p') {
        if hour == 12 {
            12
        } else {
            hour + 12
        }
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

fn time_of_day_terms(lower: &str) -> Vec<&'static str> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for (needle, facet) in [
        (" morning", "morning"),
        (" afternoon", "afternoon"),
        (" evening", "evening"),
        (" tonight", "evening"),
        (" night", "night"),
        (" bedtime", "night"),
        (" later part of the day", "evening"),
        (" later part of day", "evening"),
    ] {
        if lower.contains(needle) && seen.insert(facet) {
            out.push(facet);
        }
    }
    out
}

fn add_time_of_day_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = format!(" {} ", content.to_lowercase());
    for facet in time_of_day_terms(&lower) {
        push_unique(out, seen, format!("time_of_day:{}", facet));
    }
    for cap in clock_time_capture_re().captures_iter(content) {
        let Some(hour) = cap
            .name("hour")
            .and_then(|m| m.as_str().parse::<u32>().ok())
        else {
            continue;
        };
        let Some(meridiem) = cap.name("meridiem").map(|m| m.as_str()) else {
            continue;
        };
        let hour = clock_hour_24(hour, meridiem);
        push_unique(
            out,
            seen,
            format!("time_of_day:{}", time_of_day_for_hour(hour)),
        );
    }
}

fn add_cadence_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let mut matched = false;
    for cap in cadence_re().captures_iter(content) {
        let Some(unit) = cap.name("unit").and_then(|m| normalize_value(m.as_str())) else {
            continue;
        };
        matched = true;
        push_unique(out, seen, "has:frequency");
        push_unique(out, seen, "schedule:frequency");
        push_unique(out, seen, format!("frequency_unit:{}", unit));
        if unit == "week" {
            push_unique(out, seen, "schedule:weekly");
        }
    }
    for cap in every_unit_re().captures_iter(content) {
        let Some(unit) = cap.name("unit").and_then(|m| normalize_value(m.as_str())) else {
            continue;
        };
        matched = true;
        push_unique(out, seen, "has:frequency");
        push_unique(out, seen, "schedule:frequency");
        push_unique(out, seen, format!("frequency_unit:{}", unit));
        if unit == "week" {
            push_unique(out, seen, "schedule:weekly");
        }
    }
    if matched && clock_time_re().is_match(content) {
        push_unique(out, seen, "has:time");
    }
}

fn has_any(lower: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| lower.contains(needle))
}

fn purchase_consideration_signal(lower: &str) -> bool {
    let first_person_planning = has_any(
        lower,
        &[
            " i am considering ",
            " i'm considering ",
            " im considering ",
            " i was considering ",
            " we are considering ",
            " we're considering ",
            " we were considering ",
            " i am thinking about ",
            " i'm thinking about ",
            " im thinking about ",
            " we are thinking about ",
            " we're thinking about ",
            " i am looking to ",
            " i'm looking to ",
            " im looking to ",
            " we are looking to ",
            " we're looking to ",
            " i plan to ",
            " i'm planning to ",
            " im planning to ",
            " we plan to ",
            " we're planning to ",
            " i want to ",
            " i'd like to ",
            " we want to ",
            " we'd like to ",
        ],
    );
    let acquisition_or_change = has_any(
        lower,
        &[
            " buy ",
            " buying ",
            " purchase ",
            " purchasing ",
            " get ",
            " getting ",
            " order ",
            " ordering ",
            " upgrade ",
            " upgrading ",
            " replace ",
            " replacing ",
            " switch ",
            " switching ",
        ],
    );

    (first_person_planning && acquisition_or_change)
        || has_any(
            lower,
            &[
                " i am in the market for ",
                " i'm in the market for ",
                " im in the market for ",
                " we are in the market for ",
                " we're in the market for ",
            ],
        )
}

fn iteration_signal(lower: &str) -> bool {
    has_any(
        lower,
        &[
            " another ",
            " a different ",
            " different version ",
            " revised ",
            " revision ",
            " updated version ",
            " alternative version ",
            " second ",
            " next version ",
            " new version ",
            " here's a more ",
            " here is a more ",
        ],
    )
}

fn inspiration_source_signal(lower: &str) -> bool {
    has_any(
        lower,
        &[
            " get inspiration from ",
            " gets inspiration from ",
            " getting inspiration from ",
            " got inspiration from ",
            " find inspiration from ",
            " finds inspiration from ",
            " finding inspiration from ",
            " found inspiration from ",
            " draw inspiration from ",
            " draws inspiration from ",
            " drawing inspiration from ",
            " drew inspiration from ",
            " take inspiration from ",
            " takes inspiration from ",
            " taking inspiration from ",
            " took inspiration from ",
            " inspired by ",
        ],
    )
}

fn charity_event_signal(lower: &str) -> bool {
    lower.contains(" charity ")
        && has_any(
            lower,
            &[
                " event ",
                " walk ",
                " run ",
                " ride ",
                " drive ",
                " gala ",
                " fundraiser ",
                " fund-raiser ",
                " raise money ",
                " raised money ",
            ],
        )
}

fn wake_time_signal(lower: &str) -> bool {
    has_any(
        lower,
        &[
            " wake up at ",
            " wake up around ",
            " wake up by ",
            " wake up before ",
            " wake up after ",
            " waking up at ",
            " waking up around ",
            " waking up by ",
            " waking up before ",
            " waking up after ",
            " wake-up time ",
            " wake-up times ",
            " wakeup time ",
            " wakeup times ",
        ],
    )
}

fn bed_time_signal(lower: &str) -> bool {
    has_any(
        lower,
        &[
            " go to bed at ",
            " go to bed around ",
            " go to bed by ",
            " go to bed before ",
            " go to bed after ",
            " went to bed at ",
            " went to bed around ",
            " went to bed by ",
            " went to bed before ",
            " went to bed after ",
            " get to bed at ",
            " get to bed around ",
            " get to bed by ",
            " get to bed before ",
            " get to bed after ",
            " get to bed until ",
            " got to bed at ",
            " got to bed around ",
            " got to bed by ",
            " got to bed before ",
            " got to bed after ",
            " got to bed until ",
            " bedtime at ",
            " bedtime around ",
            " bedtime was ",
        ],
    )
}

fn add_type_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = format!(" {} ", content.to_lowercase());
    let first_person_acquired = first_person_acquired_re().is_match(&lower);
    let acquisition_source =
        first_person_acquisition_source_re().is_match(&lower) || ownership_source_re().is_match(&lower);
    let decision_selection = has_decision_selection_language(content);
    let homegrown_source = homegrown_source_re().is_match(&lower);
    let ingredient_context = ingredient_context_re().is_match(&lower);
    let inspiration_source = inspiration_source_signal(&lower);
    let charity_event = charity_event_signal(&lower);
    let wake_time = wake_time_signal(&lower);
    let bed_time = bed_time_signal(&lower);

    if has_any(
        &lower,
        &[
            "favorite",
            "favourite",
            "prefer",
            "preference",
            " i like ",
            " i love ",
            " enjoy ",
            " fan of ",
            "would rather",
        ],
    ) {
        push_unique(out, seen, "type:preference");
    }
    if has_any(
        &lower,
        &[
            "don't like",
            "do not like",
            "dislike",
            "hate",
            "avoid",
            "can't stand",
            "not a fan",
        ],
    ) {
        push_unique(out, seen, "type:dislike");
    }
    if owned_object_re().is_match(content)
        || first_person_acquired
        || first_person_possession_re().is_match(&lower)
    {
        push_unique(out, seen, "type:ownership");
    }
    if first_person_acquired {
        push_unique(out, seen, "type:activity");
        push_unique(out, seen, "type:event");
        push_unique(out, seen, "purchase:acquired");
    }
    if acquisition_source {
        push_unique(out, seen, "type:ownership");
        push_unique(out, seen, "type:activity");
        push_unique(out, seen, "type:event");
        push_unique(out, seen, "purchase:source");
    }
    if purchase_consideration_signal(&lower) {
        push_unique(out, seen, "type:purchase_consideration");
    }
    if iteration_signal(&lower) {
        push_unique(out, seen, "type:iteration");
    }
    if inspiration_source {
        push_unique(out, seen, "type:inspiration_source");
        push_unique(out, seen, "type:interest");
    }
    if first_person_competition_event_re().is_match(&lower) {
        push_unique(out, seen, "type:activity");
        push_unique(out, seen, "type:event");
        push_unique(out, seen, "type:competition_event");
        push_unique(out, seen, "activity_domain:sport");
    }
    if first_person_with_companion_re().is_match(&lower) {
        push_unique(out, seen, "type:activity");
        push_unique(out, seen, "companion:with");
    }
    if first_person_completed_clean_re().is_match(&lower) {
        push_unique(out, seen, "type:activity");
        push_unique(out, seen, "completed_action:clean");
    }
    if co_residence_with_self_re().is_match(&lower) {
        push_unique(out, seen, "co_residence:with_self");
    }
    if first_person_project_work_re().is_match(&lower) {
        push_unique(out, seen, "type:activity");
        push_unique(out, seen, "type:project_work");
    }
    if first_person_completed_action_re().is_match(&lower)
        || first_person_did_named_event_re().is_match(&lower)
    {
        push_unique(out, seen, "type:activity");
        push_unique(out, seen, "type:event");
    }
    if charity_event {
        push_unique(out, seen, "type:event");
        push_unique(out, seen, "event_domain:charity");
    }
    if wake_time {
        push_unique(out, seen, "type:routine");
        push_unique(out, seen, "routine:wake_time");
    }
    if bed_time {
        push_unique(out, seen, "type:routine");
        push_unique(out, seen, "routine:bed_time");
    }
    if decision_selection {
        push_unique(out, seen, "type:decision");
        push_unique(out, seen, "type:selection");
        if has_any(
            &lower,
            &[" name ", " names ", " named ", " call it ", " called "],
        ) {
            push_unique(out, seen, "type:naming");
        }
    }
    if has_any(
        &lower,
        &[
            " milestone ",
            " first client",
            " first customer",
            " first sale",
            " first contract",
            " signed a contract",
            " landed my first",
            " landed our first",
            " launched my ",
            " launched our ",
            " launched the ",
            " opened my ",
            " opened our ",
        ],
    ) {
        push_unique(out, seen, "type:milestone");
    }
    if has_any(
        &lower,
        &[
            " actually ",
            " changed to ",
            " changed from ",
            " switched to ",
            " switched from ",
            " no longer ",
            " instead ",
            " updated ",
            " just wrapped up ",
            " wrapped up ",
        ],
    ) {
        push_unique(out, seen, "type:update");
    }
    if has_any(
        &lower,
        &[
            "recommend",
            "suggest",
            "suggestion",
            "you should",
            "try ",
            "option",
            "would be good",
        ],
    ) {
        push_unique(out, seen, "type:recommendation");
    }
    if has_any(
        &lower,
        &[
            " working in the field",
            " work in the field",
            " works in the field",
            " my field",
            " our field",
            " field of research",
            " research area",
            " research interests",
            " i specialize in",
            " i specialize on",
            " we specialize in",
            " we specialize on",
            " i'm specializing in",
            " im specializing in",
            " we're specializing in",
            " were specializing in",
        ],
    ) {
        push_unique(out, seen, "type:expertise");
        push_unique(out, seen, "type:interest");
    }
    if has_any(
        &lower,
        &[
            "recipe",
            "ingredient",
            "preheat",
            "tablespoon",
            "teaspoon",
            "bake",
            "simmer",
            "saute",
            "cook for",
        ],
    ) {
        push_unique(out, seen, "type:recipe");
    }
    if ingredient_context {
        push_unique(out, seen, "type:ingredient");
    }
    if homegrown_source {
        push_unique(out, seen, "type:homegrown");
    }
    if has_any(
        &lower,
        &[
            "answer is",
            "the answer",
            "correct answer",
            "here's",
            "here is",
            "you can ",
            "you could ",
        ],
    ) {
        push_unique(out, seen, "type:answer");
    }
    if has_any(
        &lower,
        &[
            "usually",
            "always",
            "every morning",
            "every night",
            "daily",
            "weekly",
            "typical week",
            "per week",
            "a week",
            "each week",
            "routine",
            "habit",
            "wind down",
        ],
    ) {
        push_unique(out, seen, "type:routine");
    }
}

pub fn has_decision_selection_language(content: &str) -> bool {
    let lower = format!(" {} ", content.to_lowercase());
    has_any(
        &lower,
        &[
            " decided ",
            " decide ",
            " chose ",
            " choose ",
            " picked ",
            " pick ",
            " selected ",
            " select ",
            " settled on ",
            " went with ",
            " go with ",
            " call it ",
            " name it ",
        ],
    ) || decision_selection_re().is_match(&lower)
}

fn navigation_route_signal(lower: &str) -> bool {
    has_any(
        lower,
        &[
            " getting around ",
            " get around ",
            " how to get to ",
            " how do i get to ",
            " how can i get to ",
            " best way to get ",
            " way to get there ",
            " way to get to ",
            " route to ",
            " route from ",
            " directions to ",
            " direction to ",
            " navigate ",
            " navigation ",
            " meeting point ",
            " travel to ",
            " travel from ",
            " transfer to ",
            " transfer at ",
        ],
    )
}

fn navigation_transit_signal(lower: &str) -> bool {
    has_any(
        lower,
        &[
            " public transport",
            " public transportation",
            " public transit",
            " mass transit",
            " take the train",
            " take a train",
            " by train",
            " train from ",
            " train to ",
            " train station",
            " subway",
            " metro",
            " take the bus",
            " take a bus",
            " by bus",
            " bus from ",
            " bus to ",
            " bus station",
            " tram",
            " ferry",
            " taxi",
            " rideshare",
            " ride share",
            " airport shuttle",
        ],
    )
}

fn navigation_station_signal(lower: &str) -> bool {
    has_any(
        lower,
        &[
            " station",
            " airport",
            " terminal",
            " platform",
            " ticket gate",
            " departure gate",
            " arrival gate",
            " gate when entering",
            " gate when exiting",
        ],
    )
}

fn navigation_fare_signal(lower: &str) -> bool {
    has_any(
        lower,
        &[
            " fare",
            " ticket",
            " tickets",
            " travel time",
            " ride time",
            " journey time",
            " transfer time",
            " approximate cost",
            " cost using",
        ],
    )
}

fn navigation_pass_signal(lower: &str) -> bool {
    has_any(
        lower,
        &[
            " transit card",
            " transport card",
            " transportation card",
            " prepaid card",
            " rail pass",
            " train pass",
            " travel pass",
            " metrocard",
            " metro card",
        ],
    )
}

fn navigation_app_signal(lower: &str) -> bool {
    has_any(
        lower,
        &[
            " transit app",
            " transport app",
            " transportation app",
            " travel app",
            " trip app",
            " route app",
            " maps app",
            " itinerary app",
            " tripit app",
            " google maps",
            " apple maps",
            " citymapper",
            " moovit",
            " downloaded the app",
            " downloaded an app",
        ],
    )
}

fn add_temporal_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = content.to_lowercase();
    if has_any(
        &lower,
        &[
            "currently",
            "current ",
            "right now",
            "now ",
            "latest",
            "newest",
            "recently updated",
        ],
    ) {
        push_unique(out, seen, "temporal:current");
    }
    if has_any(
        &lower,
        &[
            "recently",
            "lately",
            "the other day",
            "past few",
            "last few",
        ],
    ) {
        push_unique(out, seen, "temporal:recent");
    }
    if has_any(&lower, &["last week", "past week", "previous week"]) {
        push_unique(out, seen, "temporal:last_week");
    }
    if has_any(
        &lower,
        &[
            "yesterday",
            "today",
            "tomorrow",
            "last ",
            "next ",
            "ago",
            "past ",
        ],
    ) {
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

fn add_entity_attribute_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    for cap in entity_class_relation_re().captures_iter(content) {
        let Some(class_text) = cap.name("class").map(|m| m.as_str()) else {
            continue;
        };
        if normalize_role_phrase(class_text).is_none() {
            continue;
        }
        push_unique(out, seen, "type:entity_attribute");
        push_unique(out, seen, "attribute:class_relation");
        return;
    }
    for cap in preferred_attribute_relation_re().captures_iter(content) {
        let Some(class_text) = cap.name("class").map(|m| m.as_str()) else {
            continue;
        };
        if normalize_role_phrase(class_text).is_none() {
            continue;
        }
        push_unique(out, seen, "type:entity_attribute");
        push_unique(out, seen, "attribute:class_relation");
        return;
    }
}

fn add_person_role_facets(content: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    for cap in titled_person_re().captures_iter(content) {
        let Some(title) = cap.name("title").and_then(|m| normalize_value(m.as_str())) else {
            continue;
        };
        push_unique(out, seen, format!("person_title:{}", title));
        push_unique(out, seen, "person_ref:named");
    }

    for cap in possessed_role_before_title_re().captures_iter(content) {
        if let Some(role) = cap
            .name("role")
            .and_then(|m| normalize_role_phrase(m.as_str()))
        {
            push_unique(out, seen, format!("person_role_phrase:{}", role));
            push_unique(out, seen, "person_ref:named");
        }
    }

    for cap in role_before_title_re().captures_iter(content) {
        let role_text = [cap.name("role"), cap.name("role2")]
            .into_iter()
            .flatten()
            .map(|m| m.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        if let Some(role) = normalize_role_phrase(&role_text) {
            push_unique(out, seen, format!("person_role_phrase:{}", role));
            push_unique(out, seen, "person_ref:named");
        }
    }

    for cap in role_before_name_re().captures_iter(content) {
        let Some(role) = cap
            .name("role")
            .and_then(|m| normalize_role_phrase(m.as_str()))
        else {
            continue;
        };
        if role.contains('_') {
            push_unique(out, seen, format!("person_role_phrase:{}", role));
            push_unique(out, seen, "person_ref:named");
        }
    }
}

const TRANSPORT_MODES: &[(&str, &[&str])] = &[
    ("bus", &["bus", "buses"]),
    ("train", &["train", "trains"]),
    ("plane", &["plane", "planes", "flight", "flights"]),
    ("car", &["car", "cars"]),
    ("taxi", &["taxi", "taxis", "cab", "cabs"]),
    ("subway", &["subway", "subways", "metro", "metros"]),
    ("tram", &["tram", "trams"]),
    ("ferry", &["ferry", "ferries"]),
    ("bike", &["bike", "bikes", "bicycle", "bicycles"]),
    ("walk", &["walk", "walking"]),
];

const RELIGIOUS_CONTEXT_TERMS: &[&str] = &[
    "abbey",
    "ashram",
    "baptist",
    "bible",
    "buddhist",
    "cathedral",
    "catholic",
    "chapel",
    "christian",
    "church",
    "convent",
    "episcopal",
    "gurdwara",
    "hindu",
    "islamic",
    "jewish",
    "lutheran",
    "methodist",
    "monastery",
    "mosque",
    "muslim",
    "orthodox",
    "parish",
    "presbyterian",
    "rabbi",
    "shrine",
    "synagogue",
    "temple",
];

const RELIGIOUS_ACTIVITY_TERMS: &[&str] = &[
    "bible study",
    "communion",
    "eucharist",
    "liturgy",
    "mass",
    "maundy",
    "prayer",
    "sabbath service",
    "sermon",
    "service",
    "sunday school",
    "worship",
];

fn padded_contains_word(lower: &str, word: &str) -> bool {
    lower.contains(&format!(" {word} "))
}

fn has_padded_term(lower: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| padded_contains_word(lower, term))
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
    add_numeric_object_facets(content, &mut facets, &mut seen);
    add_inventory_object_facets(content, &mut facets, &mut seen);
    add_age_facets(content, &mut facets, &mut seen);
    add_education_facets(content, &mut facets, &mut seen);
    add_family_facets(content, &mut facets, &mut seen);
    add_type_facets(content, &mut facets, &mut seen);
    add_temporal_facets(content, &mut facets, &mut seen);
    add_entity_facets(content, &mut facets, &mut seen);
    add_entity_attribute_facets(content, &mut facets, &mut seen);
    add_person_role_facets(content, &mut facets, &mut seen);

    facets
}

pub fn extract_memory_facets(
    content: &str,
    metadata: Option<&HashMap<String, Value>>,
    existing_cues: &[String],
) -> Vec<String> {
    extract_memory_facets_with_cuepacks(
        content,
        metadata,
        existing_cues,
        crate::cuepacks::default_registry(),
        None,
    )
}

pub fn extract_memory_facets_with_cuepacks(
    content: &str,
    metadata: Option<&HashMap<String, Value>>,
    existing_cues: &[String],
    cuepacks: &crate::cuepacks::CuePackRegistry,
    cuepack_selection: Option<&[String]>,
) -> Vec<String> {
    let mut facets = extract_memory_facets_core(content, metadata, existing_cues);
    let mut seen: HashSet<String> = facets.iter().map(|facet| facet.to_lowercase()).collect();
    let pack_output = cuepacks.extract_memory_facets(content, cuepack_selection);
    let emits_standing_instruction = pack_output
        .facets
        .iter()
        .any(|facet| facet == "type:standing_instruction");
    let emits_explicit_preference = pack_output
        .facets
        .iter()
        .any(|facet| facet == "preference:explicit");
    for facet in pack_output.facets {
        if seen.insert(facet.to_lowercase()) {
            facets.push(facet);
        }
    }
    if emits_standing_instruction {
        add_standing_instruction_dynamic_facets(content, &mut facets, &mut seen);
    }
    if emits_explicit_preference {
        add_preference_dynamic_facets(content, &mut facets, &mut seen);
    }
    facets
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryIntent {
    pub labels: Vec<String>,
    pub weighted_cues: Vec<(String, f64)>,
    #[serde(default)]
    pub cue_weight_adjustments: Vec<(String, f64)>,
    #[serde(default)]
    pub cuepack_rules: Vec<String>,
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
    if !available(cue) {
        return;
    }
    if seen.insert(cue.to_string()) {
        out.push((cue.to_string(), weight));
    } else if let Some((_, existing_weight)) = out.iter_mut().find(|(existing, _)| existing == cue)
    {
        if *existing_weight < weight {
            *existing_weight = weight;
        }
    }
}

fn add_label(out: &mut Vec<String>, label: &str) {
    if !out.iter().any(|existing| existing == label) {
        out.push(label.to_string());
    }
}

fn push_adjustment(out: &mut Vec<(String, f64)>, cue: &str, multiplier: f64) {
    if let Some((_, existing)) = out.iter_mut().find(|(existing, _)| existing == cue) {
        if *existing > multiplier {
            *existing = multiplier;
        }
    } else {
        out.push((cue.to_string(), multiplier));
    }
}

fn normalized_phrase_initialism(value: &str) -> Option<String> {
    let parts = value
        .split('_')
        .filter(|part| {
            part.len() >= 2
                && part.chars().all(|ch| ch.is_ascii_alphabetic())
                && !crate::nl::get_stopwords().contains(*part)
        })
        .collect::<Vec<_>>();

    if parts.len() < 2 || parts.len() > 6 {
        return None;
    }

    let initials = parts
        .iter()
        .filter_map(|part| part.chars().next())
        .collect::<String>();
    if initials.len() < 2 || initials.len() > 8 || initials == value {
        None
    } else {
        Some(initials)
    }
}

fn recommendation_topic_is_generic(cue: &str) -> bool {
    if cue.contains('_') {
        return cue.split('_').all(recommendation_topic_is_generic);
    }

    matches!(
        cue,
        "can"
            | "could"
            | "would"
            | "should"
            | "please"
            | "recommend"
            | "recommendation"
            | "suggest"
            | "suggestion"
            | "idea"
            | "ideas"
            | "tip"
            | "tips"
            | "advice"
            | "help"
            | "helpful"
            | "think"
            | "try"
            | "recipe"
            | "recipes"
            | "useful"
            | "good"
            | "best"
            | "some"
            | "any"
            | "make"
            | "making"
            | "choose"
            | "choosing"
            | "chosen"
            | "choice"
            | "decide"
            | "deciding"
            | "decision"
            | "select"
            | "selecting"
            | "selection"
            | "not"
            | "sure"
            | "one"
            | "look"
            | "looking"
            | "thing"
            | "something"
            | "show"
            | "shows"
            | "movie"
            | "movies"
            | "watch"
            | "tonight"
            | "new"
            | "current"
            | "recent"
            | "lately"
            | "upcoming"
            | "weekend"
            | "today"
            | "tomorrow"
            | "later"
            | "soon"
            | "trip"
            | "travel"
            | "accessory"
            | "setup"
    )
}

fn recommendation_topic_cues(query: &str) -> Vec<(String, f64)> {
    let mut seen = HashSet::new();
    let mut cues = Vec::new();
    let tokenized = crate::nl::tokenize_to_cues(query);
    let mut head_topic = None;
    let lower = query.to_lowercase();

    for marker in [
        "making a ",
        "making an ",
        "making some ",
        "make a ",
        "make an ",
        "make some ",
        "what to ",
        "how to ",
    ] {
        let Some((_, tail)) = lower.split_once(marker) else {
            continue;
        };
        if let Some(action) = crate::nl::tokenize_to_cues(tail)
            .into_iter()
            .map(|cue| cue.trim().to_lowercase())
            .find(|cue| cue.len() > 2 && !cue.contains('_') && !recommendation_topic_is_generic(cue))
        {
            head_topic = Some(action);
            break;
        }
    }

    if head_topic.is_none() {
        for idx in 0..tokenized.len() {
            let cue = tokenized[idx].trim().to_lowercase();
            if cue.len() <= 2 || cue.contains('_') || recommendation_topic_is_generic(&cue) {
                continue;
            }

            let right = tokenized
                .iter()
                .skip(idx + 1)
                .take(3)
                .map(|part| part.trim().to_lowercase())
                .collect::<Vec<_>>();
            if right
                .iter()
                .any(|part| matches!(part.as_str(), "recipe" | "recipes" | "idea" | "ideas" | "recommendation" | "recommendations" | "suggestion" | "suggestions"))
            {
                head_topic = Some(cue);
            }
        }
    }

    for cue in tokenized {
        let cue = cue.trim().to_lowercase();
        if cue.len() <= 2 || recommendation_topic_is_generic(&cue) || !seen.insert(cue.clone()) {
            continue;
        }
        let weight = if Some(cue.as_str()) == head_topic.as_deref() {
            4.0
        } else if head_topic.is_some() && !cue.contains('_') {
            1.4
        } else {
            2.4
        };
        cues.push((cue, weight));
        if cues.len() >= 8 {
            break;
        }
    }
    cues
}

fn query_transport_modes(query: &str) -> Vec<&'static str> {
    let lower = format!(" {} ", crate::nl::normalize_text(query));
    let mut modes = Vec::new();
    for (mode, variants) in TRANSPORT_MODES {
        if variants
            .iter()
            .any(|variant| padded_contains_word(&lower, variant))
        {
            modes.push(*mode);
        }
    }
    modes
}

fn add_person_query_intent<F>(
    lower: &str,
    is_count: bool,
    intent: &mut QueryIntent,
    seen: &mut HashSet<String>,
    available: &F,
) where
    F: Fn(&str) -> bool,
{
    let doctor_event_context = has_any(
        lower,
        &[
            "doctor appointment",
            "doctor's appointment",
            "doctor appointments",
            "doctor visit",
            "doctor's visit",
        ],
    );
    let title_queries = [
        (
            has_any(lower, &["doctor", "doctors", "dr. ", " dr "])
                && (is_count || !doctor_event_context),
            "dr",
            "person_title:dr",
        ),
        (
            has_any(lower, &["professor", "professors", "prof. ", " prof "]),
            "prof",
            "person_title:prof",
        ),
    ];
    let mut saw_person_title_query = false;

    for (matches_query, lexical_title, title_facet) in title_queries {
        if !matches_query {
            continue;
        }
        saw_person_title_query = true;
        add_label(&mut intent.labels, "person_role");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            lexical_title,
            2.2,
        );
        push_weighted_if_available(&mut intent.weighted_cues, seen, available, title_facet, 2.4);
    }

    if is_count && saw_person_title_query {
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            "person_ref:named",
            1.6,
        );
    }
}

fn has_person_title_query(lower: &str) -> bool {
    has_any(
        lower,
        &["doctor", "doctors", "dr. ", " dr ", "professor", "professors", "prof. ", " prof "],
    )
}

fn query_tokens(lower: &str) -> Vec<String> {
    lower
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|token| token.to_string())
        .collect()
}

fn query_has_phrase(tokens: &[String], phrase: &[&str]) -> bool {
    !phrase.is_empty()
        && tokens
            .windows(phrase.len())
            .any(|window| window.iter().map(String::as_str).eq(phrase.iter().copied()))
}

fn count_object_stopwords() -> &'static HashSet<&'static str> {
    static STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOPWORDS.get_or_init(|| {
        HashSet::from([
            "a",
            "an",
            "and",
            "current",
            "currently",
            "different",
            "existing",
            "my",
            "new",
            "old",
            "one",
            "own",
            "kind",
            "kinds",
            "the",
            "total",
            "type",
            "types",
        ])
    })
}

fn count_object_boundary(token: &str) -> bool {
    matches!(
        token,
        "am" | "are"
            | "can"
            | "could"
            | "did"
            | "do"
            | "does"
            | "had"
            | "has"
            | "have"
            | "include"
            | "including"
            | "is"
            | "should"
            | "was"
            | "were"
            | "will"
            | "would"
    )
}

fn collect_count_object(tokens: &[String], start: usize) -> Option<Vec<String>> {
    let stopwords = count_object_stopwords();
    let mut objects = Vec::new();
    let mut idx = start;

    while idx < tokens.len() && objects.len() < 4 {
        let token = tokens[idx].as_str();
        if count_object_boundary(token) {
            break;
        }
        if !stopwords.contains(token) {
            if let Some(object) = normalize_quantity_token(token) {
                objects.push(object);
            }
        }
        idx += 1;
    }

    if objects.is_empty() {
        None
    } else {
        Some(objects)
    }
}

fn count_query_objects(lower: &str) -> Vec<String> {
    let tokens = query_tokens(lower);
    let mut objects = Vec::new();
    let mut seen = HashSet::new();

    for idx in 0..tokens.len() {
        let start = if tokens[idx] == "many" && idx > 0 && tokens[idx - 1] == "how" {
            Some(idx + 1)
        } else if tokens[idx] == "of"
            && idx > 0
            && matches!(tokens[idx - 1].as_str(), "number" | "count" | "total")
        {
            Some(idx + 1)
        } else {
            None
        };

        if let Some(start) = start {
            if let Some(candidates) = collect_count_object(&tokens, start) {
                for candidate in candidates {
                    if seen.insert(candidate.clone()) {
                        objects.push(candidate);
                    }
                }
            }
        }
    }

    objects
}

fn count_scope_preposition(token: &str) -> bool {
    matches!(
        token,
        "in" | "inside" | "within" | "from" | "for" | "across" | "among" | "on"
    )
}

fn count_scope_boundary(token: &str) -> bool {
    matches!(
        token,
        "and" | "but"
            | "in"
            | "or"
            | "from"
            | "for"
            | "because"
            | "when"
            | "where"
            | "which"
            | "who"
            | "what"
            | "that"
            | "then"
            | "than"
            | "with"
            | "without"
            | "before"
            | "after"
            | "while"
            | "during"
    )
}

fn count_scope_determiner(token: &str) -> bool {
    matches!(
        token,
        "my" | "our" | "the" | "this" | "that" | "these" | "those" | "a" | "an"
    )
}

fn count_query_scope_tokens(lower: &str) -> Vec<String> {
    let tokens = query_tokens(lower);
    let mut scoped = Vec::new();
    let mut seen = HashSet::new();

    for idx in 0..tokens.len().saturating_sub(1) {
        if !count_scope_preposition(tokens[idx].as_str()) {
            continue;
        }

        let mut cursor = idx + 1;
        if cursor < tokens.len() && count_scope_determiner(tokens[cursor].as_str()) {
            cursor += 1;
        }

        let mut phrase_tokens = Vec::new();
        while cursor < tokens.len() && phrase_tokens.len() < 4 {
            let token = tokens[cursor].as_str();
            if count_scope_boundary(token) || count_object_boundary(token) {
                break;
            }
            if !count_object_stopwords().contains(token) {
                if let Some(normalized) = normalize_quantity_token(token) {
                    phrase_tokens.push(normalized);
                }
            }
            cursor += 1;
        }

        if phrase_tokens.is_empty() {
            continue;
        }

        for token in &phrase_tokens {
            if seen.insert(token.clone()) {
                scoped.push(token.clone());
            }
        }
        if phrase_tokens.len() >= 2 {
            let phrase = phrase_tokens.join("_");
            if seen.insert(phrase.clone()) {
                scoped.push(phrase);
            }
        }
    }

    scoped
}

fn purchase_query_object_tokens(lower: &str) -> Vec<String> {
    let tokens = query_tokens(lower);
    let mut objects = Vec::new();
    let mut seen = HashSet::new();

    for idx in 0..tokens.len() {
        if !matches!(
            tokens[idx].as_str(),
            "acquire"
                | "acquired"
                | "buy"
                | "bought"
                | "get"
                | "got"
                | "order"
                | "ordered"
                | "purchase"
                | "purchased"
                | "receive"
                | "received"
        ) {
            continue;
        }

        let mut cursor = idx + 1;
        while cursor < tokens.len()
            && matches!(tokens[cursor].as_str(), "my" | "our" | "the" | "a" | "an" | "some")
        {
            cursor += 1;
        }

        let mut phrase = Vec::new();
        while cursor < tokens.len() && phrase.len() < 4 {
            let token = tokens[cursor].as_str();
            if matches!(token, "from" | "at" | "via" | "through" | "for" | "with")
                || count_object_boundary(token)
                || count_scope_boundary(token)
            {
                break;
            }
            if !matches!(token, "new" | "current" | "latest" | "same") {
                if let Some(normalized) = normalize_quantity_token(token) {
                    phrase.push(normalized);
                }
            }
            cursor += 1;
        }

        for token in &phrase {
            if seen.insert(token.clone()) {
                objects.push(token.clone());
            }
        }
        if phrase.len() >= 2 {
            let phrase_cue = phrase.join("_");
            if seen.insert(phrase_cue.clone()) {
                objects.push(phrase_cue);
            }
        }
    }

    objects
}

fn small_number_value(token: &str) -> Option<i64> {
    match token {
        "a" | "an" | "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        _ => token.parse::<i64>().ok(),
    }
}

fn temporal_quantity_unit_to_days(quantity: i64, unit: &str) -> Option<i64> {
    match unit {
        "day" | "days" => Some(quantity),
        "week" | "weeks" => Some(quantity * 7),
        "month" | "months" => Some(quantity * 30),
        "year" | "years" => Some(quantity * 365),
        _ => None,
    }
}

fn relative_query_target_date(lower: &str, reference_date: NaiveDate) -> Option<NaiveDate> {
    let tokens = query_tokens(lower);

    if tokens.iter().any(|token| token == "yesterday") {
        return reference_date.checked_sub_signed(Duration::days(1));
    }
    if tokens.iter().any(|token| token == "today") {
        return Some(reference_date);
    }
    if query_has_phrase(&tokens, &["last", "week"])
        || query_has_phrase(&tokens, &["previous", "week"])
    {
        return reference_date.checked_sub_signed(Duration::days(7));
    }

    for idx in 0..tokens.len().saturating_sub(2) {
        let Some(quantity) = small_number_value(tokens[idx].as_str()) else {
            continue;
        };
        if tokens[idx + 2] != "ago" {
            continue;
        }
        let Some(days) = temporal_quantity_unit_to_days(quantity, tokens[idx + 1].as_str()) else {
            continue;
        };
        return reference_date.checked_sub_signed(Duration::days(days));
    }

    None
}

fn add_relative_temporal_query_cues<F>(
    lower: &str,
    reference_time: Option<&str>,
    intent: &mut QueryIntent,
    seen: &mut HashSet<String>,
    available: &F,
) -> bool
where
    F: Fn(&str) -> bool,
{
    let Some(reference_date) = reference_time.and_then(parse_date_text) else {
        return false;
    };

    if lower.contains("this year") || lower.contains("current year") {
        add_label(&mut intent.labels, "temporal_resolved_year");
        for cue in ["this", "current", "year", "years"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_year_facet(reference_date),
            1.4,
        );
        return true;
    }

    if lower.contains("this month") || lower.contains("current month") {
        add_label(&mut intent.labels, "temporal_resolved_month");
        for cue in ["this", "current", "month", "months"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_month_facet(reference_date),
            1.5,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_year_facet(reference_date),
            0.7,
        );
        return true;
    }

    if lower.contains("last month")
        || lower.contains("previous month")
        || lower.contains("past month")
    {
        let Some(target_month) = previous_calendar_month_date(reference_date) else {
            return false;
        };
        add_label(&mut intent.labels, "temporal_resolved_month");
        for cue in ["last", "previous", "past", "month", "months"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_month_facet(target_month),
            1.9,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_year_facet(target_month),
            0.7,
        );
        if lower.contains("past month") && target_month.month() != reference_date.month() {
            push_weighted_if_available(
                &mut intent.weighted_cues,
                seen,
                available,
                &source_month_facet(reference_date),
                0.9,
            );
        }
        return true;
    }

    if lower.contains("this week") || lower.contains("current week") {
        add_label(&mut intent.labels, "temporal_resolved_week");
        for cue in ["this", "current", "week", "weeks"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_week_facet(reference_date),
            1.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_month_facet(reference_date),
            0.8,
        );
        return true;
    };
    let tokens = query_tokens(lower);
    if query_has_phrase(&tokens, &["last", "weekend"])
        || query_has_phrase(&tokens, &["past", "weekend"])
        || query_has_phrase(&tokens, &["previous", "weekend"])
    {
        let Some((saturday, sunday)) = most_recent_weekend_dates(reference_date) else {
            return false;
        };
        add_label(&mut intent.labels, "temporal_resolved_weekend");
        for cue in ["last", "past", "previous", "weekend", "weekends"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_date_facet(saturday),
            1.9,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_date_facet(sunday),
            1.9,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_week_facet(sunday),
            0.8,
        );
        return true;
    }
    if query_has_phrase(&tokens, &["last", "week"])
        || query_has_phrase(&tokens, &["previous", "week"])
    {
        let Some(target_date) = reference_date.checked_sub_signed(Duration::days(7)) else {
            return false;
        };
        add_label(&mut intent.labels, "temporal_resolved_week");
        for cue in ["last", "previous", "week", "weeks"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_week_facet(target_date),
            1.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            seen,
            available,
            &source_month_facet(target_date),
            0.8,
        );
        return true;
    }
    let Some(target_date) = relative_query_target_date(lower, reference_date) else {
        return false;
    };

    add_label(&mut intent.labels, "temporal_resolved_date");
    for cue in [
        "ago", "day", "days", "week", "weeks", "month", "months", "year", "years", "today",
        "yesterday", "last", "previous", "one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten", "eleven", "twelve",
    ] {
        push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
    }
    push_weighted_if_available(
        &mut intent.weighted_cues,
        seen,
        available,
        &source_date_facet(target_date),
        1.8,
    );
    push_weighted_if_available(
        &mut intent.weighted_cues,
        seen,
        available,
        &source_week_facet(target_date),
        0.8,
    );
    push_weighted_if_available(
        &mut intent.weighted_cues,
        seen,
        available,
        &source_month_facet(target_date),
        0.4,
    );
    push_weighted_if_available(
        &mut intent.weighted_cues,
        seen,
        available,
        "temporal:today",
        1.6,
    );
    true
}

fn is_inventory_query(lower: &str, is_count: bool, is_duration: bool) -> bool {
    if is_duration {
        return false;
    }

    has_any(
        lower,
        &[
            "current setup",
            "currently have",
            "currently own",
            "currently use",
            "do i have",
            "do i own",
            "my setup",
        ],
    ) || (is_count
        && has_any(
            lower,
            &[
                "do i have",
                "do we have",
                "i have",
                "we have",
                " own",
                " set up",
                " use",
                "including the one",
                "include the one",
                "plus the one",
            ],
        ))
        || (is_count
            && has_any(lower, &[" my ", " our "])
            && has_any(
                lower,
                &[
                    " in ",
                    " inside ",
                    " within ",
                    " across ",
                    " among ",
                    " between ",
                    " both ",
                    " all ",
                    " total ",
                ],
            ))
}

pub fn compile_query_intent<F>(query: &str, available: F) -> QueryIntent
where
    F: Fn(&str) -> bool,
{
    compile_query_intent_with_reference_time(query, None, available)
}

pub fn compile_query_intent_with_reference_time<F>(
    query: &str,
    reference_time: Option<&str>,
    available: F,
) -> QueryIntent
where
    F: Fn(&str) -> bool,
{
    let lower = query.to_lowercase();
    let lower_padded = format!(" {} ", lower);
    let normalized_padded = format!(" {} ", crate::nl::normalize_text(query));
    let mut intent = QueryIntent::default();
    let mut seen = HashSet::new();

    let asks_time_amount = has_any(
        &lower,
        &[
            "how much time",
            "how many seconds",
            "how many minutes",
            "how many hours",
            "how many days",
            "how many weeks",
            "how many months",
            "how many years",
        ],
    );
    let is_count = has_any(
        &lower,
        &[
            "how many",
            "number of",
            "count ",
            "total number",
            "different ",
        ],
    );
    let is_completion_count_query = is_count
        && has_any(
            &normalized_padded,
            &[
                " completed ",
                " complete ",
                " finished ",
                " finish ",
                " passed ",
                " took ",
                " taken ",
            ],
        );
    let is_age_difference = has_any(
        &lower,
        &[
            "years older",
            "year older",
            "years younger",
            "year younger",
            "older am i",
            "younger am i",
        ],
    ) || (lower.contains("how many years")
        && has_any(
            &lower,
            &[
                "how old",
                "what age",
                "when i",
                "when we",
                "graduated",
                "graduation",
                "college",
                "degree",
            ],
        ));
    let is_age_query = is_age_difference
        || has_any(
            &lower,
            &[
                "how old",
                "what age",
                "at what age",
                "age was i",
                "age am i",
                "when i was",
                "when we were",
            ],
        );
    let is_undergraduate_education_query = has_any(
        &lower_padded,
        &[
            " bachelor's",
            " bachelors",
            " bachelor ",
            " undergraduate",
            " undergrad",
        ],
    );
    let is_education_query = is_age_query
        || is_undergraduate_education_query
        || has_any(
            &lower_padded,
            &[
                " degree ",
                " degrees ",
                " graduated ",
                " graduation ",
                " graduate ",
                " college ",
                " university ",
                " alma mater ",
            ],
        );
    let is_money = has_any(
        &lower,
        &[
            "cost", "costs", "spent", "paid", "price", "money", "dollars", "usd", "$",
        ],
    ) || (lower.contains("how much") && !asks_time_amount);
    let relative_time_operator = has_any(
        &lower,
        &[
            " ago",
            "last ",
            "previous ",
            "yesterday",
            "today",
            "tomorrow",
        ],
    );
    let asks_activity_duration_amount = asks_time_amount
        && (has_any(
            &lower,
            &[
                "how many seconds of",
                "how many minutes of",
                "how many hours of",
                "how many days of",
                "how many weeks of",
                "how many months of",
                "how many years of",
            ],
        ) || has_any(
            &lower,
            &[
                "how many seconds did i spend",
                "how many seconds did we spend",
                "how many minutes did i spend",
                "how many minutes did we spend",
                "how many hours did i spend",
                "how many hours did we spend",
                "how many days did i spend",
                "how many days did we spend",
                "how many weeks did i spend",
                "how many weeks did we spend",
                "how many months did i spend",
                "how many months did we spend",
                "how many years did i spend",
                "how many years did we spend",
            ],
        ));
    let is_duration = !is_age_query
        && ((asks_time_amount && (!relative_time_operator || asks_activity_duration_amount))
            || has_any(
                &lower,
                &["how long", "duration", "for how many"],
            ));
    let count_objects = if is_count {
        count_query_objects(&lower)
    } else {
        Vec::new()
    };
    let count_scopes = if is_count {
        count_query_scope_tokens(&lower)
    } else {
        Vec::new()
    };
    let family_relation_query_facets = family_relation_query_facets(&lower);
    let is_co_residence_query = co_residence_query_re().is_match(&lower);
    let is_weekly_routine_query = has_any(
        &lower,
        &[
            "how often",
            "typical week",
            "usual week",
            "normal week",
            "per week",
            "days a week",
            "times a week",
            "each week",
            "weekly",
        ],
    ) && !has_any(&lower, &["last week", "past week", "previous week", "week ago"]);
    let is_weekday_schedule_query = has_any(
        &lower,
        &[
            "what day of the week",
            "which day of the week",
            "what weekday",
            "which weekday",
        ],
    );
    let is_wake_time_routine_query = has_any(&lower, &["what time", "when"])
        && has_any(
            &normalized_padded,
            &[
                " wake ",
                " wake up ",
                " waking ",
                " waking up ",
            ],
        )
        && has_any(
            &normalized_padded,
            &[
                " morning ",
                " mornings ",
                " weekday ",
                " weekdays ",
                " weekend ",
                " weekends ",
                " monday ",
                " tuesday ",
                " wednesday ",
                " thursday ",
                " friday ",
                " saturday ",
                " sunday ",
            ],
        )
        && !has_any(&lower, &["yesterday", "last night", "last week", "ago"]);
    let is_bed_time_query = has_any(&lower, &["what time", "when"])
        && has_any(
            &normalized_padded,
            &[
                " go to bed ",
                " went to bed ",
                " get to bed ",
                " got to bed ",
                " bedtime ",
            ],
        );
    let is_current = has_any(
        &lower,
        &[
            "current",
            "currently",
            "right now",
            "latest",
            "newest",
            "recent",
        ],
    );
    let is_temporal_order = has_any(
        &lower,
        &[
            "earliest to latest",
            "latest to earliest",
            "oldest to newest",
            "newest to oldest",
            "first to last",
            "last to first",
            "chronological order",
            "reverse chronological",
        ],
    ) || (lower.contains("order of")
        && has_any(
            &lower,
            &[
                "earliest",
                "latest",
                "oldest",
                "newest",
                "first",
                "last",
                "chronological",
            ],
        ));
    let transport_query_modes = query_transport_modes(query);
    let is_transport_mode_comparison = transport_query_modes.len() >= 2
        && has_any(
            &lower,
            &[
                "mode of transport",
                "mode of transportation",
                "transport did i use",
                "transportation did i use",
                "which mode",
                "which transport",
            ],
        )
        && has_any(&lower, &["recent", "recently", "latest", "last"]);
    let is_streaming_service_query = has_any(
        &normalized_padded,
        &[
            " streaming service ",
            " streaming services ",
            " streaming platform ",
            " streaming platforms ",
        ],
    ) || (has_any(&normalized_padded, &[" which service ", " what service "])
        && has_any(
            &normalized_padded,
            &[" watch ", " watching ", " show ", " shows "],
        ))
        || (has_any(
            &normalized_padded,
            &[
                " service did i start using ",
                " service did we start using ",
                " service did i use ",
                " service did we use ",
            ],
        ) && has_any(&normalized_padded, &[" stream ", " streaming "]));
    let is_music_streaming_service_query = is_streaming_service_query
        && has_any(
            &normalized_padded,
            &[
                " music ",
                " song ",
                " songs ",
                " track ",
                " tracks ",
                " album ",
                " albums ",
                " artist ",
                " artists ",
                " playlist ",
                " playlists ",
            ],
        );
    let is_current_reading_query = has_any(
        &normalized_padded,
        &[
            " what book am i currently reading ",
            " what book am i reading ",
            " what book are we currently reading ",
            " what book are we reading ",
            " which book am i currently reading ",
            " which book am i reading ",
            " current book ",
            " currently reading ",
        ],
    );
    let is_entity_attribute_query = has_any(
        &normalized_padded,
        &[
            " what breed ",
            " which breed ",
            " what type ",
            " which type ",
            " what model ",
            " which model ",
            " what brand ",
            " which brand ",
            " what color ",
            " which color ",
            " what size ",
            " which size ",
            " what name ",
            " which name ",
        ],
    ) && has_any(&normalized_padded, &[" my ", " our ", " mine ", " ours "]);
    let is_first_person_state = has_any(
        &lower,
        &[
            " am i ",
            "am i ",
            " do i ",
            "do i ",
            " does my ",
            "does my ",
            " my current ",
            " my latest ",
            " my newest ",
        ],
    );
    let is_preference = has_any(
        &lower,
        &[
            "favorite",
            "favourite",
            "prefer",
            "preference",
            "do i like",
            "i like",
            "recommend for me",
            "would i like",
        ],
    );
    let is_source_answer = has_any(
        &lower,
        &[
            "what did",
            "what was said",
            "did you say",
            "you said",
            "did you tell",
            "you told",
            "did you suggest",
            "you suggested",
            "did you recommend",
            "you recommended",
            "you created",
            "you made",
            "you wrote",
            "you generated",
            "you composed",
            "provided",
            "answer",
            "remind me of",
            "remind us of",
            "remind me what",
            "remind us what",
            "remind me which",
            "remind us which",
            "remind me who",
            "remind us who",
            "remind me where",
            "remind us where",
            "remind me the",
            "remind us the",
        ],
    );
    let is_activity_event_query = has_any(
        &lower,
        &[
            "what did i do",
            "what did we do",
            "activity did i",
            "activity did we",
            "event did i",
            "event did we",
            "what event",
            "what activity",
            "where did i attend",
            "where did we attend",
            "where did i participate",
            "where did we participate",
            "visited",
            "participated in",
            "attended",
        ],
    ) || (is_temporal_order
        && has_any(
            &lower,
            &[
                "visit",
                "visited",
                "attend",
                "attended",
                "participated",
                "went to",
            ],
        ));
    let is_charity_event_query = has_any(
        &normalized_padded,
        &[
            " charity event ",
            " charity events ",
            " charity walk ",
            " charity run ",
            " charity ride ",
            " charity drive ",
            " charity gala ",
            " fundraiser ",
            " fund raiser ",
        ],
    );
    let is_competition_event_query = has_any(
        &normalized_padded,
        &[
            " sport event ",
            " sports event ",
            " sport events ",
            " sports events ",
            " tournament ",
            " tournaments ",
            " race ",
            " races ",
            " triathlon ",
            " triathlons ",
            " marathon ",
            " marathons ",
            " 5k ",
            " 10k ",
            " match ",
            " matches ",
            " competition ",
            " competitions ",
        ],
    );
    let is_first_person_activity_query = has_any(
        &lower,
        &[
            "what did i do",
            "what did we do",
            "activity did i",
            "activity did we",
            "event did i",
            "event did we",
            "where did i",
            "where did we",
            "did i attend",
            "did we attend",
            "i attended",
            "we attended",
            "i participated",
            "we participated",
        ],
    );
    let is_companion_query = companion_query_re().is_match(&lower);
    let is_completed_clean_query = completed_clean_query_re().is_match(&lower);
    let is_religious_activity_query = has_padded_term(
        &normalized_padded,
        &[
            "religious activity",
            "religious event",
            "spiritual activity",
            "spiritual event",
            "faith activity",
            "faith event",
            "worship service",
        ],
    ) || (is_activity_event_query
        && (padded_contains_word(&normalized_padded, "religious")
            || padded_contains_word(&normalized_padded, "spiritual")
            || has_padded_term(&normalized_padded, RELIGIOUS_CONTEXT_TERMS)
            || has_padded_term(&normalized_padded, RELIGIOUS_ACTIVITY_TERMS)));
    let is_milestone_query = has_any(
        &lower,
        &[
            "milestone",
            "major achievement",
            "big achievement",
            "significant achievement",
            "important achievement",
            "major accomplishment",
            "significant accomplishment",
        ],
    );
    let is_decision_query = has_any(
        &lower,
        &[
            "what did we decide",
            "what did i decide",
            "what did you decide",
            "what did we finally decide",
            "what did i finally decide",
            "did we decide",
            "did i decide",
            "finally decided",
            "finally decide",
            "decided to",
            "what did we choose",
            "what did i choose",
            "which did we choose",
            "which did i choose",
            "what did we pick",
            "what did i pick",
            "settled on",
            "went with",
        ],
    );
    let is_naming_decision = is_decision_query
        && has_any(
            &lower,
            &[
                " name",
                " names",
                " named",
                " call it",
                " called",
                " what to call",
            ],
        );
    let is_assistant_source = has_any(
        &lower,
        &[
            "did you say",
            "you said",
            "did you tell",
            "you told",
            "you mentioned",
            "did you suggest",
            "you suggested",
            "did you recommend",
            "you recommended",
            "you created",
            "you made",
            "you wrote",
            "you generated",
            "you composed",
            "your answer",
            "you provided",
            "remind me of",
            "remind us of",
            "remind me what",
            "remind us what",
            "remind me which",
            "remind us which",
            "remind me who",
            "remind us who",
            "remind me where",
            "remind us where",
            "remind me the",
            "remind us the",
        ],
    );
    let is_user_source = has_any(
        &lower,
        &[
            "i said",
            "i told",
            "i mention",
            "i mentioned",
            "i bring up",
            "i brought up",
            "i discuss",
            "i discussed",
            "i talk about",
            "i talked about",
            "i ask about",
            "i asked about",
            "i asked",
            "my message",
            "i wrote",
        ],
    );
    let is_iteration_reference = has_any(
        &lower_padded,
        &[
            " second ",
            " third ",
            " fourth ",
            " another ",
            " revised ",
            " revision ",
            " updated version ",
            " different version ",
            " alternative ",
            " next version ",
        ],
    );
    let is_recipe =
        has_any(&lower, &["recipe", "ingredient", "cook", "bake"]) || ingredient_context_re().is_match(&lower);
    let is_homegrown = homegrown_source_re().is_match(&lower);
    let is_recommendation = has_any(
        &lower,
        &[
            "recommend",
            "suggest",
            "should i",
            "what should",
            "tip",
            "advice",
            "any ideas",
            "ideas on",
            "idea for",
            "idea about",
        ],
    );
    let is_personal_recommendation_context = is_recommendation
        && !is_source_answer
        && !is_assistant_source
        && has_any(
            &normalized_padded,
            &[
                " for me ",
                " for us ",
                " my ",
                " our ",
                " i ",
                " we ",
                " me ",
                " us ",
            ],
        );
    let has_explicit_recommendation_topic = has_any(
        &normalized_padded,
        &[
            " about ",
            " regarding ",
            " related to ",
            " around ",
            " on the topic of ",
            " in the area of ",
        ],
    );
    let is_vague_interest_recommendation = is_personal_recommendation_context
        && !has_explicit_recommendation_topic
        && has_any(
            &lower,
            &[
                "find interesting",
                "might find",
                "would find",
                "might like",
                "would like",
                "of interest",
            ],
        );
    let is_research_interest_recommendation = is_vague_interest_recommendation
        && has_any(
            &normalized_padded,
            &[
                " publication ",
                " publications ",
                " conference ",
                " conferences ",
                " paper ",
                " papers ",
                " literature ",
                " research ",
            ],
        );
    let is_media_watch_recommendation = is_recommendation
        && has_any(
            &normalized_padded,
            &[
                " movie ",
                " movies ",
                " show ",
                " shows ",
                " something to watch ",
                " watch tonight ",
                " tv show ",
                " tv shows ",
            ],
        );
    let is_inspiration_recommendation = is_personal_recommendation_context
        && has_any(
            &normalized_padded,
            &[
                " inspiration ",
                " inspired ",
                " inspiring ",
            ],
        );
    let is_purchase_query = has_any(
        &lower,
        &[
            "what did i buy",
            "what did we buy",
            "where did i buy",
            "where did we buy",
            "where did i get",
            "where did we get",
            "where did i purchase",
            "where did we purchase",
            "where did i order",
            "where did we order",
            "did i buy",
            "did we buy",
            "what did i purchase",
            "what did we purchase",
            "what did i order",
            "what did we order",
            "what did i receive",
            "what did we receive",
            "what did i acquire",
            "what did we acquire",
            "what did i get",
            "what did we get",
            "who did i get",
            "who did we get",
            "who did i receive",
            "who did we receive",
            "who did i acquire",
            "who did we acquire",
            "from whom did i get",
            "from whom did we get",
            "from whom did i receive",
            "from whom did we receive",
            "from whom did i acquire",
            "from whom did we acquire",
            "i bought",
            "i purchased",
            "i ordered",
            "i received",
            "i acquired",
        ],
    );
    let is_purchase_source_query = is_purchase_query
        && has_any(
            &lower,
            &[
                "where did i",
                "where did we",
                "where was it from",
                "where were they from",
                "from where",
                "from whom",
                "who did i",
                "who did we",
            ],
        );
    let is_purchase_consideration_query = has_any(
        &normalized_padded,
        &[
            " what to look for ",
            " what should i look for ",
            " what should we look for ",
            " tips on what to look for ",
            " shopping for ",
            " in the market for ",
            " looking to buy ",
            " looking to purchase ",
            " looking to get ",
            " planning to buy ",
            " planning to purchase ",
            " considering buying ",
            " considering purchasing ",
            " considering upgrading ",
            " new one ",
        ],
    ) || (has_any(
        &normalized_padded,
        &[" new ", " newer ", " upgrade ", " upgrading "],
    ) && has_any(
        &normalized_padded,
        &[
            " recommend ",
            " recommendation ",
            " suggest ",
            " suggestion ",
            " tips ",
            " advice ",
            " look for ",
        ],
    ));
    let is_navigation = navigation_route_signal(&lower_padded)
        || navigation_transit_signal(&lower_padded)
        || navigation_station_signal(&lower_padded)
        || navigation_fare_signal(&lower_padded)
        || navigation_pass_signal(&lower_padded)
        || navigation_app_signal(&lower_padded);
    let is_sibling_relation_query = has_any(
        &lower,
        &[
            "sibling",
            "siblings",
            "brother",
            "brothers",
            "sister",
            "sisters",
        ],
    );
    let is_family_relation_count = is_count
        && is_sibling_relation_query
        && has_any(
            &lower,
            &[
                "i have",
                "do i have",
                "number of",
                "total number",
                "how many",
                "count of",
            ],
        );
    let has_temporal_marker = has_any(
        &lower,
        &[
            "when",
            "last ",
            "past ",
            "ago",
            "yesterday",
            "today",
            "tomorrow",
            "week",
            "month",
            "year",
        ],
    );
    let is_future_scheduled_advice = has_any(
        &lower,
        &[
            "this weekend",
            "next weekend",
            "today",
            "tonight",
            "tomorrow",
            "later",
            "soon",
            "upcoming",
        ],
    ) && (is_recommendation
        || has_any(
            &lower,
            &[
                "any tips",
                "tips",
                "advice",
                "planning to",
                "thinking about",
                "going to",
                "looking for",
                "want to",
                "getting excited about",
            ],
        ));
    let asks_past_or_time_window = has_any(
        &lower,
        &[
            "what did",
            "what was",
            "what were",
            "where did",
            "who did",
            "which",
            "when",
            "how many",
            "how much",
            "last ",
            "past ",
            "ago",
            "yesterday",
        ],
    );
    let is_temporal_distance_question = is_count
        && lower.contains("ago")
        && has_any(
            &normalized_padded,
            &[
                " day ",
                " days ",
                " week ",
                " weeks ",
                " month ",
                " months ",
                " year ",
                " years ",
            ],
        );
    let is_temporal = !is_age_query
        && !is_weekly_routine_query
        && !is_weekday_schedule_query
        && has_temporal_marker
        && (!is_future_scheduled_advice || asks_past_or_time_window);
    let is_inventory = !is_family_relation_count
        && is_inventory_query(&lower, is_count, is_duration);

    if is_future_scheduled_advice {
        for cue in [
            "weekend",
            "today",
            "tonight",
            "tomorrow",
            "later",
            "soon",
            "upcoming",
            "trip",
            "travel",
        ] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
    }

    let query_time_of_day = time_of_day_terms(&lower_padded);
    if !query_time_of_day.is_empty() {
        add_label(&mut intent.labels, "time_of_day");
        for facet in query_time_of_day {
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                &format!("time_of_day:{}", facet),
                3.4,
            );
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:time",
            1.4,
        );
    }

    if is_count {
        add_label(&mut intent.labels, "count");
        for cue in ["many", "different", "number", "count", "total"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.25);
        }
        if !is_age_query && !is_duration && !has_person_title_query(&lower) {
            for object in &count_objects {
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    object,
                    8.0,
                );
            }
            for scope in &count_scopes {
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    scope,
                    3.6,
                );
            }
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:number",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:list",
            1.6,
        );
        if is_completion_count_query {
            add_label(&mut intent.labels, "completion_count");
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "completion_count:object",
                7.2,
            );
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "quantity_count:object",
                5.2,
            );
            for object in &count_objects {
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    &format!("quantity_object:{}", object),
                    4.8,
                );
            }
        }
        intent.suppress_generic = true;
    }
    let is_project_count_query = is_count
        && has_any(&normalized_padded, &[" project ", " projects "])
        && has_any(
            &lower,
            &[
                "have i led",
                "have we led",
                "am currently leading",
                "are currently leading",
                "currently leading",
                "led or",
                "leading",
                "working on",
                "worked on",
            ],
        );
    if is_project_count_query {
        add_label(&mut intent.labels, "project_work_count");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:project_work",
            4.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:activity",
            2.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            3.0,
        );
        for cue in ["lead", "led", "leading", "work", "working", "project"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 1.8);
        }
    }
    if is_weekly_routine_query {
        add_label(&mut intent.labels, "weekly_routine");
        for cue in ["typical", "usual", "normal", "week", "weekly"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:frequency",
            3.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "schedule:frequency",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "frequency_unit:week",
            2.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "schedule:weekly",
            3.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:weekday",
            2.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:time",
            1.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:routine",
            2.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:activity",
            1.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            1.4,
        );
    }
    if is_weekday_schedule_query {
        add_label(&mut intent.labels, "weekday_schedule");
        for cue in ["day", "week", "weekday"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:weekday",
            4.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "schedule:weekly",
            3.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:routine",
            1.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            1.4,
        );
    }
    if is_wake_time_routine_query {
        add_label(&mut intent.labels, "wake_time_routine");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "routine:wake_time",
            4.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:time",
            3.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:weekday",
            2.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:routine",
            2.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            2.2,
        );
    }
    if is_bed_time_query {
        add_label(&mut intent.labels, "bed_time");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "routine:bed_time",
            4.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:time",
            3.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "time_of_day:night",
            2.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            2.0,
        );
    }
    if is_money {
        add_label(&mut intent.labels, "money");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:money",
            3.5,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:number",
            1.5,
        );
        intent.suppress_generic = true;
    }
    if !family_relation_query_facets.is_empty() {
        add_label(&mut intent.labels, "family_relation");
        for facet in &family_relation_query_facets {
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                facet,
                3.6,
            );
        }
    }
    if is_co_residence_query {
        add_label(&mut intent.labels, "co_residence");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "co_residence:with_self",
            4.2,
        );
    }
    if is_duration {
        add_label(&mut intent.labels, "duration");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:duration",
            3.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:number",
            1.2,
        );
        for object in &count_objects {
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                object,
                6.0,
            );
        }
        for scope in &count_scopes {
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                scope,
                4.0,
            );
        }
        intent.suppress_generic = true;
    }
    if is_age_query {
        add_label(&mut intent.labels, "age_query");
        if is_age_difference {
            add_label(&mut intent.labels, "age_difference");
        }
        for cue in ["year", "years", "old", "older", "younger"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:age",
            3.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "age:current",
            3.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "age:event",
            3.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "education:graduation",
            3.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "education:degree",
            2.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "education:college",
            2.0,
        );
        intent.suppress_generic = true;
    }
    if is_education_query && !is_age_query {
        add_label(&mut intent.labels, "education_query");
        for cue in [
            "degree",
            "degrees",
            "bachelor",
            "bachelors",
            "undergraduate",
            "undergrad",
        ] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.55);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "education:degree",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "education:undergraduate",
            3.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "education:graduation",
            2.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "education:college",
            1.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            1.2,
        );
    }
    if is_family_relation_count {
        add_label(&mut intent.labels, "family_relation_count");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "family_count:sibling",
            4.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "family_scope:self",
            3.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "family_relation:sibling",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "sibling_kind:brother",
            2.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "sibling_kind:sister",
            2.8,
        );
        intent.suppress_generic = true;
    }
    if is_inventory {
        add_label(&mut intent.labels, "inventory");
        push_adjustment(&mut intent.cue_weight_adjustments, "currently", 0.25);
        push_adjustment(&mut intent.cue_weight_adjustments, "current", 0.25);
        if has_any(
            &lower,
            &["including the one", "include the one", "plus the one"],
        ) {
            push_adjustment(&mut intent.cue_weight_adjustments, "one", 0.25);
            push_adjustment(&mut intent.cue_weight_adjustments, "include", 0.45);
            push_adjustment(&mut intent.cue_weight_adjustments, "including", 0.45);
            push_adjustment(&mut intent.cue_weight_adjustments, "set", 0.45);
            push_adjustment(&mut intent.cue_weight_adjustments, "friend", 0.45);
            push_adjustment(&mut intent.cue_weight_adjustments, "kid", 0.45);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:ownership",
            2.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            3.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "quantity_count:object",
            8.0,
        );
        if !count_scopes.is_empty() {
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "inventory_count:contained",
                7.2,
            );
        }
        for object in count_query_objects(&lower) {
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                &format!("inventory_object:{}", object),
                3.6,
            );
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                &format!("quantity_object:{}", object),
                3.8,
            );
        }
    }
    if is_temporal_order {
        add_label(&mut intent.labels, "temporal_order");
        for cue in [
            "order",
            "early",
            "earliest",
            "latest",
            "oldest",
            "newest",
            "first",
            "last",
            "six",
        ] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_time:dated",
            2.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            2.2,
        );
    }
    if is_transport_mode_comparison {
        add_label(&mut intent.labels, "transport_mode_comparison");
        add_label(&mut intent.labels, "temporal_order");
        for cue in ["mode", "transport", "transportation", "use", "recent", "recently"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_time:dated",
            2.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            2.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:activity",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:event",
            2.2,
        );
        for mode in &transport_query_modes {
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                &format!("transport_event:{mode}"),
                4.0,
            );
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                &format!("transport_mode:{mode}"),
                2.8,
            );
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                mode,
                2.0,
            );
        }
    }
    if is_streaming_service_query {
        add_label(&mut intent.labels, "streaming_service_usage");
        if is_music_streaming_service_query {
            add_label(&mut intent.labels, "music_streaming_service_usage");
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "media:music_streaming",
                4.4,
            );
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "media:music",
                2.4,
            );
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "media:streaming",
            4.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:usage",
            3.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "media:watching",
            1.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            1.4,
        );
        for cue in ["service", "start", "started", "use", "using", "used", "recently"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.45);
        }
    }
    if is_current_reading_query {
        add_label(&mut intent.labels, "current_reading");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "reading:current",
            4.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "media:book_reading",
            4.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "media:book",
            2.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            1.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "temporal:current",
            2.0,
        );
        for cue in ["currently", "current", "read", "reading"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.55);
        }
    }
    if is_current
        && !is_inventory
        && !is_temporal_order
        && !is_transport_mode_comparison
        && !is_streaming_service_query
        && !is_current_reading_query
    {
        add_label(&mut intent.labels, "latest_current");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "temporal:current",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "temporal:recent",
            1.8,
        );
    }
    if (is_current
        && !is_inventory
        && !is_temporal_order
        && !is_transport_mode_comparison
        && !is_weekly_routine_query
        && !is_personal_recommendation_context
        && !is_streaming_service_query
        && !is_current_reading_query)
        || (is_first_person_state
            && !is_age_query
            && !is_inventory
            && !is_temporal_order
            && !is_transport_mode_comparison
            && !is_weekly_routine_query
            && !is_personal_recommendation_context
            && !is_streaming_service_query
            && !is_current_reading_query)
    {
        add_label(&mut intent.labels, "state_update");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:update",
            2.6,
        );
    }
    if lower.contains("what type of") {
        push_adjustment(&mut intent.cue_weight_adjustments, "type", 0.25);
    }
    if is_preference {
        add_label(&mut intent.labels, "preference");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:preference",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:dislike",
            2.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:ownership",
            1.2,
        );
    }
    if is_purchase_query {
        add_label(&mut intent.labels, "purchase");
        let purchase_objects = purchase_query_object_tokens(&lower);
        for object in &purchase_objects {
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                object,
                if object.contains('_') { 4.4 } else { 3.4 },
            );
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:ownership",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "purchase:acquired",
            4.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:activity",
            1.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:event",
            1.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            2.4,
        );
        if is_purchase_source_query {
            add_label(&mut intent.labels, "purchase_source");
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "purchase:source",
                3.6,
            );
        }
    }
    if is_purchase_consideration_query {
        add_label(&mut intent.labels, "purchase_consideration");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:purchase_consideration",
            3.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:ownership",
            1.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:preference",
            1.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            1.8,
        );
    }
    if is_entity_attribute_query {
        add_label(&mut intent.labels, "entity_attribute");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:entity_attribute",
            3.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "attribute:class_relation",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            if is_preference { 3.2 } else { 2.0 },
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:ownership",
            1.0,
        );
    }
    if is_source_answer {
        add_label(&mut intent.labels, "source_answer");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:answer",
            2.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:recommendation",
            2.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "has:list",
            1.4,
        );
    }
    if is_iteration_reference {
        add_label(&mut intent.labels, "iteration_reference");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:iteration",
            3.2,
        );
    }
    if is_activity_event_query || is_companion_query || is_completed_clean_query {
        add_label(&mut intent.labels, "activity_event");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:activity",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:event",
            2.2,
        );
        if is_charity_event_query {
            add_label(&mut intent.labels, "charity_event");
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "event_domain:charity",
                3.4,
            );
        }
        push_adjustment(&mut intent.cue_weight_adjustments, "activity", 0.6);
        push_adjustment(&mut intent.cue_weight_adjustments, "event", 0.6);
        if is_first_person_activity_query {
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "source_role:user",
                1.4,
            );
        }
        if is_companion_query {
            add_label(&mut intent.labels, "companion");
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "companion:with",
                5.0,
            );
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "source_role:user",
                1.2,
            );
        }
        if is_completed_clean_query {
            add_label(&mut intent.labels, "completed_action");
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "completed_action:clean",
                5.2,
            );
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "source_role:user",
                1.2,
            );
        }
    }
    if is_competition_event_query {
        add_label(&mut intent.labels, "competition_event");
        for cue in [
            "participate",
            "participated",
            "participating",
            "entered",
            "joined",
            "raced",
            "ran",
        ] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 2.4);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:competition_event",
            4.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "activity_domain:sport",
            3.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:activity",
            2.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:event",
            2.0,
        );
    }
    if is_religious_activity_query {
        add_label(&mut intent.labels, "religious_activity");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "activity_domain:religion",
            4.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "topic:religion",
            1.6,
        );
        push_adjustment(&mut intent.cue_weight_adjustments, "religious", 0.7);
        push_adjustment(&mut intent.cue_weight_adjustments, "spiritual", 0.7);
    }
    if is_milestone_query {
        add_label(&mut intent.labels, "milestone");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:milestone",
            3.2,
        );
        push_adjustment(&mut intent.cue_weight_adjustments, "significant", 0.6);
        push_adjustment(&mut intent.cue_weight_adjustments, "important", 0.6);
        push_adjustment(&mut intent.cue_weight_adjustments, "major", 0.6);
    }
    if is_decision_query {
        add_label(&mut intent.labels, "decision_selection");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:decision",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:selection",
            2.6,
        );
        if is_naming_decision {
            add_label(&mut intent.labels, "naming_decision");
            push_weighted_if_available(
                &mut intent.weighted_cues,
                &mut seen,
                &available,
                "type:naming",
                2.4,
            );
        }
    }
    if is_assistant_source {
        add_label(&mut intent.labels, "source_assistant");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:assistant",
            2.5,
        );
    }
    if is_user_source {
        add_label(&mut intent.labels, "source_user");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            2.5,
        );
    }
    if is_temporal {
        add_label(&mut intent.labels, "temporal_window");
        for month in explicit_query_months(&lower) {
            if let Some(facet) = content_month_facet(month) {
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    &facet,
                    4.0,
                );
            }
        }
        let added_specific_source_window = add_relative_temporal_query_cues(
            &lower,
            reference_time,
            &mut intent,
            &mut seen,
            &available,
        );
        if !added_specific_source_window {
            if is_temporal_distance_question {
                add_label(&mut intent.labels, "temporal_distance");
                for cue in [
                    "ago", "day", "days", "week", "weeks", "month", "months", "year",
                    "years",
                ] {
                    push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
                }
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    "has:date",
                    2.2,
                );
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    "temporal:relative",
                    1.6,
                );
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    "source_role:user",
                    1.1,
                );
            } else {
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    "has:date",
                    2.3,
                );
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    "temporal:relative",
                    2.0,
                );
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    "temporal:last_week",
                    2.0,
                );
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    "temporal:recent",
                    1.5,
                );
            }
        }
    }
    if is_recipe {
        add_label(&mut intent.labels, "recipe");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:recipe",
            2.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:ingredient",
            2.4,
        );
    }
    if is_homegrown {
        add_label(&mut intent.labels, "homegrown");
        for cue in ["ingredient", "ingredients", "recipe", "recipes", "weekend"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.35);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:homegrown",
            3.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:ingredient",
            2.2,
        );
    }
    if is_recommendation {
        add_label(&mut intent.labels, "recommendation");
        for cue in [
            "can",
            "could",
            "would",
            "should",
            "please",
            "recommend",
            "recommendation",
            "suggest",
            "suggestion",
            "think",
            "try",
            "new",
            "recipe",
            "recipes",
            "useful",
            "good",
            "best",
            "some",
            "any",
        ] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.45);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:recommendation",
            2.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:preference",
            1.8,
        );
        if is_vague_interest_recommendation {
            for cue in ["might", "find", "interest", "interesting"] {
                push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.45);
            }
        } else {
            for (cue, weight) in recommendation_topic_cues(query) {
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    &cue,
                    weight,
                );
            }
        }
    }
    if is_personal_recommendation_context {
        add_label(&mut intent.labels, "personal_recommendation_context");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            2.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:ownership",
            2.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:usage",
            1.8,
        );
    }
    if is_vague_interest_recommendation {
        add_label(&mut intent.labels, "vague_interest_recommendation");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:interest",
            2.4,
        );
    }
    if is_research_interest_recommendation {
        add_label(&mut intent.labels, "research_interest_recommendation");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:expertise",
            3.2,
        );
    }
    if is_media_watch_recommendation {
        add_label(&mut intent.labels, "media_watch_recommendation");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "media:watching",
            3.0,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "source_role:user",
            1.4,
        );
        for cue in ["show", "shows", "movie", "movies", "watch", "tonight"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.65);
        }
    }
    if is_inspiration_recommendation {
        add_label(&mut intent.labels, "inspiration_recommendation");
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:inspiration_source",
            4.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:interest",
            2.2,
        );
    }
    if is_navigation {
        add_label(&mut intent.labels, "navigation");
        for cue in ["bit", "got", "around", "helpful", "tip"] {
            push_adjustment(&mut intent.cue_weight_adjustments, cue, 0.45);
        }
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "type:navigation",
            3.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "travel:route",
            3.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "travel:transit",
            2.8,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "travel:station",
            2.4,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "travel:fare",
            2.2,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "travel:pass",
            3.6,
        );
        push_weighted_if_available(
            &mut intent.weighted_cues,
            &mut seen,
            &available,
            "travel:app",
            3.6,
        );
    }

    add_person_query_intent(&lower, is_count, &mut intent, &mut seen, &available);

    let mut entity_facets = Vec::new();
    let mut entity_seen = HashSet::new();
    add_entity_facets(query, &mut entity_facets, &mut entity_seen);
    for cue in entity_facets {
        push_weighted_if_available(&mut intent.weighted_cues, &mut seen, &available, &cue, 2.2);
        if let Some(value) = cue.strip_prefix("entity:") {
            if let Some(initials) = normalized_phrase_initialism(value) {
                let initialism_cue = format!("entity:{}", initials);
                push_weighted_if_available(
                    &mut intent.weighted_cues,
                    &mut seen,
                    &available,
                    &initialism_cue,
                    2.8,
                );
            }
        }
    }

    intent
}

pub fn compile_query_intent_with_cuepacks<F>(
    query: &str,
    reference_time: Option<&str>,
    available: F,
    cuepacks: &crate::cuepacks::CuePackRegistry,
    cuepack_selection: Option<&[String]>,
) -> QueryIntent
where
    F: Fn(&str) -> bool,
{
    let mut intent = compile_query_intent_with_reference_time(query, reference_time, &available);
    let pack_output = cuepacks.compile_query_intent(query, cuepack_selection, &available);

    for label in pack_output.labels {
        add_label(&mut intent.labels, &label);
    }

    let mut seen = intent
        .weighted_cues
        .iter()
        .map(|(cue, _)| cue.clone())
        .collect::<HashSet<_>>();
    for (cue, weight) in pack_output.weighted_cues {
        if seen.insert(cue.clone()) {
            intent.weighted_cues.push((cue, weight));
        }
    }

    for (cue, multiplier) in pack_output.cue_weight_adjustments {
        push_adjustment(&mut intent.cue_weight_adjustments, &cue, multiplier);
    }

    intent.cuepack_rules = pack_output.matched_rules;
    intent.suppress_generic |= pack_output.suppress_generic;
    intent
}

pub fn is_weak_query_cue(cue: &str) -> bool {
    matches!(
        cue,
        "many"
            | "number"
            | "count"
            | "total"
            | "different"
            | "time"
            | "times"
            | "current"
            | "currently"
            | "latest"
            | "newest"
            | "recent"
            | "recently"
            | "past"
            | "last"
            | "ago"
            | "week"
            | "month"
            | "year"
            | "long"
            | "much"
            | "cost"
            | "price"
    )
}
