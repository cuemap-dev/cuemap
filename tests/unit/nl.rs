    use super::{
        get_language_stopwords, get_stopwords, normalize_text, sanitize_text, tokenize_to_cues,
        tokenize_to_cues_with_lang,
        Language, SymbolRouter, Intent,
    };
    use std::collections::HashSet;

    #[test]
    fn temporal_connector_breaks_phrase_without_removing_token() {
        let cues = tokenize_to_cues(
            "Maya switched from coffee to mint tea after the April deploy.",
        );

        assert!(cues.contains(&"after".to_string()));
        assert!(cues.contains(&"mint_tea".to_string()));
        assert!(cues.contains(&"april_deploy".to_string()));
        assert!(!cues.contains(&"mint_tea_after".to_string()));
    }

    #[test]
    fn every_supported_language_has_keyword_filtering() {
        let languages = [
            Language::Default,
            Language::Rust,
            Language::Python,
            Language::TypeScript,
            Language::JavaScript,
            Language::Go,
            Language::Php,
            Language::Java,
            Language::Swift,
            Language::Dart,
            Language::ObjectiveC,
            Language::Kotlin,
            Language::C,
            Language::Cpp,
            Language::CSharp,
            Language::Bash,
            Language::Toml,
            Language::Css,
            Language::Html,
        ];
        for language in languages {
            let words = get_language_stopwords(language);
            assert!(!words.is_empty());
        }
        assert!(get_stopwords().contains("the"));
    }

    #[test]
    fn symbol_router_extracts_longest_symbols_and_compiles_intents() {
        let symbols = HashSet::from([
            "foo".to_string(),
            "foo_bar".to_string(),
            "bar".to_string(),
        ]);
        let router = SymbolRouter::new(symbols);
        let (intent, extracted) = router.route("where are foo_bar callers used?");
        assert_eq!(intent, Intent::FindCalls);
        assert_eq!(extracted, vec!["foo_bar"]);
        assert_eq!(
            router.compile_to_cues(Intent::FindDef, vec!["Thing".to_string()]),
            vec![
                "defines_function:Thing",
                "defines_class:Thing",
                "defines_struct:Thing",
                "defines_method:Thing",
            ]
        );
        assert_eq!(
            router.compile_to_cues(Intent::FindImports, vec!["serde".to_string()]),
            vec!["imports_module:serde"]
        );
        assert_eq!(
            router.compile_to_cues(Intent::Generic, vec!["plain".to_string()]),
            vec!["plain"]
        );
    }

    #[test]
    fn text_sanitization_and_normalization_handle_urls_camel_case_and_noise() {
        assert_eq!(sanitize_text("Read https://www.Example.com/path?q=1"), "Read Example");
        assert_eq!(normalize_text("HTTPServer v2 API"), "http server v 2 api");
        let cues = tokenize_to_cues_with_lang(
            "the HTTPServer uses abc12345 and running workers",
            Language::Rust,
        );
        assert!(cues.contains(&"server".to_string()));
        assert!(!cues.contains(&"abc12345".to_string()));
    }
