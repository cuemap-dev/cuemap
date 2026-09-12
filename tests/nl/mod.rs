use cuemap::nl::*;
use std::collections::{HashMap, HashSet};

#[test]
fn test_tokenizer_basic() {
    let tokens = tokenize_to_cues("The quick brown fox");
    assert!(tokens.contains(&"quick".to_string()));
    assert!(tokens.contains(&"fox".to_string()));
    // Bigram "quick_brown" may or may not be generated depending on RAKE extraction
}

#[test]
fn test_tokenizer_edge_cases() {
    assert!(tokenize_to_cues("").is_empty());
    assert!(tokenize_to_cues("   ").is_empty());

    let special = tokenize_to_cues("!!! @@@ ###");
    // Should be empty or only contains non-alphanumeric tokens if they are allowed
    // Looking at common tokenizers, they usually filter punctuation.
    assert!(special.is_empty());
}

#[test]
fn test_normalize_text() {
    assert_eq!(normalize_text("  HELLO   WORLD  "), "hello world");
    assert_eq!(
        normalize_text("Mixed-Case_With_Dots.com"),
        "mixed case with dots com"
    );
}

#[test]
fn test_new_language_contexts_are_supported() {
    assert_eq!(Language::from("lang:c"), Language::C);
    assert_eq!(Language::from("lang:cpp"), Language::Cpp);
    assert_eq!(Language::from("lang:csharp"), Language::CSharp);
    assert_eq!(Language::from("lang:bash"), Language::Bash);
    assert_eq!(Language::from("lang:toml"), Language::Toml);

    assert!(get_language_stopwords(Language::C).contains("struct"));
    assert!(get_language_stopwords(Language::Cpp).contains("namespace"));
    assert!(get_language_stopwords(Language::CSharp).contains("foreach"));
    assert!(get_language_stopwords(Language::Bash).contains("function"));
    assert!(get_language_stopwords(Language::Toml).contains("true"));
}

#[test]
fn test_product_case_boundaries_emit_component_cues() {
    let cues = tokenize_to_cues("My iPhone 13 Pro syncs with GitHub and PowerPoint.");

    assert!(cues.contains(&"phone".to_string()));
    assert!(cues.contains(&"git".to_string()));
    assert!(cues.contains(&"hub".to_string()));
    assert!(cues.contains(&"power".to_string()));
    assert!(cues.contains(&"point".to_string()));
    assert!(!cues.contains(&"i".to_string()));
}

#[test]
fn test_url_sanitization() {
    let text = "Check https://github.com/user/repo for details";
    let sanitized = sanitize_text(text);
    assert!(sanitized.contains("github"));
    assert!(!sanitized.contains("https://"));
    assert!(!sanitized.contains("/user/repo"));
}

#[test]
fn test_rake_phrases() {
    let text = "Directly export a function expression instead of using a declaration";
    let cues = tokenize_to_cues(text);

    // Individual tokens should be present (lemmatized but not over-stemmed)
    assert!(cues.contains(&"directly".to_string()));
    assert!(cues.contains(&"export".to_string()));
    assert!(cues.contains(&"expression".to_string()));
    assert!(cues.contains(&"declaration".to_string()));
}

#[test]
fn test_code_tokens() {
    let text = "const result = await fetch(url)";
    let cues = tokenize_to_cues(text);

    // Code keywords are NOT globally filtered (they have meaning in natural language)
    assert!(cues.contains(&"result".to_string()));
    assert!(cues.contains(&"fetch".to_string()));
    assert!(cues.contains(&"url".to_string()));
}

#[test]
fn test_stemming() {
    // Test that different forms of words stem to the same root
    let text1 = "added comment";
    let text2 = "adding comments";
    let text3 = "this line adds a comment";

    let cues1 = tokenize_to_cues(text1);
    let cues2 = tokenize_to_cues(text2);
    let cues3 = tokenize_to_cues(text3);

    // All forms should produce "add" and "comment"
    assert!(cues1.contains(&"add".to_string()));
    assert!(cues1.contains(&"comment".to_string()));

    assert!(cues2.contains(&"add".to_string()));
    assert!(cues2.contains(&"comment".to_string()));

    assert!(cues3.contains(&"add".to_string()));
    assert!(cues3.contains(&"comment".to_string()));

    // Phrases should also be stemmed consistently
    assert!(cues1.contains(&"add_comment".to_string()));
    assert!(cues2.contains(&"add_comment".to_string()));
}

#[test]
fn test_common_nouns_are_not_cross_lemmatized() {
    assert_eq!(stem_word("dog"), "dog");
    assert_eq!(stem_word("dogs"), "dog");
    assert_eq!(stem_word("bit"), "bit");
    assert_eq!(stem_word("classes"), "class");

    let cues = tokenize_to_cues("What breed is my dog?");
    assert!(cues.contains(&"dog".to_string()));
    assert!(!cues.contains(&"hot".to_string()));

    let class_cues = tokenize_to_cues("How many fitness classes do I attend?");
    assert!(class_cues.contains(&"class".to_string()));
    assert!(!class_cues.contains(&"classis".to_string()));

    let bit_cues = tokenize_to_cues("I'm a bit anxious about getting around Tokyo.");
    assert!(bit_cues.contains(&"bit".to_string()));
    assert!(!bit_cues.contains(&"bite".to_string()));
}

#[test]
fn test_contraction_fragments_are_not_cues() {
    let cues = tokenize_to_cues("I've been testing because you're sure we'll need it.");

    assert!(!cues.contains(&"ve".to_string()));
    assert!(!cues.contains(&"re".to_string()));
    assert!(!cues.contains(&"ll".to_string()));
    assert!(cues.contains(&"test".to_string()));
    assert!(cues.contains(&"need".to_string()));
}

#[test]
fn test_multiword_verb_exceptions_do_not_rewrite_component_words() {
    let cases = [
        ("bomb", "bomb"),
        ("bombs", "bomb"),
        ("call", "call"),
        ("calls", "call"),
        ("clean", "clean"),
        ("cleans", "clean"),
        ("click", "click"),
        ("clicks", "click"),
        ("fries", "fry"),
        ("fry", "fry"),
        ("message", "message"),
        ("messages", "message"),
        ("polish", "polish"),
        ("polishes", "polish"),
        ("punch", "punch"),
        ("punches", "punch"),
        ("record", "record"),
        ("records", "record"),
        ("run", "run"),
        ("runs", "run"),
        ("shui", "shui"),
        ("shuis", "shui"),
        ("skate", "skate"),
        ("skates", "skate"),
        ("treat", "treat"),
    ];

    for (word, expected) in cases {
        assert_eq!(
            stem_word(word),
            expected,
            "{word} should stem to {expected}"
        );
    }
}

#[test]
fn test_common_lemmas_do_not_use_truncated_or_wrong_pos_forms() {
    let cases = [
        ("analyzes", "analyze"),
        ("arises", "arise"),
        ("arses", "arse"),
        ("bodies", "body"),
        ("bridges", "bridge"),
        ("buzzes", "buzz"),
        ("canvasses", "canvass"),
        ("churches", "church"),
        ("coaxes", "coax"),
        ("companies", "company"),
        ("compasses", "compass"),
        ("carouses", "carouse"),
        ("delves", "delve"),
        ("divvies", "divvy"),
        ("fishing", "fish"),
        ("frizzes", "frizz"),
        ("getting", "get"),
        ("glasses", "glass"),
        ("imagines", "imagine"),
        ("interleaves", "interleave"),
        ("judges", "judge"),
        ("paralyzes", "paralyze"),
        ("pasting", "paste"),
        ("phases", "phase"),
        ("phantasies", "phantasy"),
        ("pickaxes", "pickaxe"),
        ("pledges", "pledge"),
        ("premiered", "premiere"),
        ("programming", "program"),
        ("putting", "put"),
        ("raises", "raise"),
        ("reaches", "reach"),
        ("sasses", "sass"),
        ("sexes", "sex"),
        ("sises", "sise"),
        ("stories", "story"),
        ("taxes", "tax"),
        ("teaches", "teach"),
        ("tawses", "tawse"),
        ("tries", "try"),
        ("using", "use"),
        ("uses", "use"),
        ("wishes", "wish"),
        ("witnesses", "witness"),
        ("curries", "curry"),
        ("curtsies", "curtsy"),
        ("overemphasises", "overemphasise"),
    ];

    for (word, expected) in cases {
        assert_eq!(stem_word(word), expected, "{word} should stem to {expected}");
    }
}

#[test]
fn test_lemma_exception_table_has_no_identity_entries() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = std::path::Path::new(&manifest_dir).join("lemma_exceptions.json");
    let data = std::fs::read_to_string(path).unwrap();
    let exceptions: HashMap<String, String> = serde_json::from_str(&data).unwrap();
    let protective_identity_overrides = ["bit"].into_iter().collect::<HashSet<_>>();
    let identity_entries: Vec<_> = exceptions
        .iter()
        .filter_map(|(word, lemma)| {
            (word == lemma && !protective_identity_overrides.contains(word.as_str()))
                .then_some(word.as_str())
        })
        .collect();

    assert!(
        identity_entries.is_empty(),
        "lemma exceptions should not contain no-op entries: {identity_entries:?}"
    );
}

#[test]
fn test_symbol_router_intents() {
    let mut symbols = std::collections::HashSet::new();
    symbols.insert("process_data".to_string());
    symbols.insert("UserStore".to_string());
    symbols.insert("ingest".to_string());

    let router = SymbolRouter::new(symbols);

    // Test FIND_CALLS
    let (intent, extracted) = router.route("where is process_data used?");
    assert_eq!(intent, Intent::FindCalls);
    assert_eq!(extracted, vec!["process_data"]);

    // Test FIND_DEF
    let (intent, extracted) = router.route("what does the UserStore class do?");
    assert_eq!(intent, Intent::FindDef);
    assert_eq!(extracted, vec!["UserStore"]);

    // Test FIND_IMPORTS
    let (intent, extracted) = router.route("show me imports for ingest");
    assert_eq!(intent, Intent::FindImports);
    assert_eq!(extracted, vec!["ingest"]);

    // Test Generic
    let (intent, extracted) = router.route("hello process_data");
    assert_eq!(intent, Intent::Generic);
    assert_eq!(extracted, vec!["process_data"]);
}

#[test]
fn test_symbol_router_compilation() {
    let mut symbols = std::collections::HashSet::new();
    symbols.insert("auth_service".to_string());
    let router = SymbolRouter::new(symbols);

    let cues = router.compile_to_cues(Intent::FindCalls, vec!["auth_service".to_string()]);
    assert!(cues.contains(&"calls_function:auth_service".to_string()));
    assert!(cues.contains(&"calls_method:auth_service".to_string()));

    let cues_def = router.compile_to_cues(Intent::FindDef, vec!["auth_service".to_string()]);
    assert!(cues_def.contains(&"defines_function:auth_service".to_string()));
}
