use cuemap::projects::*;
use cuemap::structures::MainStats;
use cuemap::config::ServerConfig;
use cuemap::{normalization::NormalizationConfig, taxonomy::Taxonomy};
use std::fs;
use std::sync::Arc;
use std::sync::atomic::Ordering;

fn write_artifact(dir: &std::path::Path, name: &str, content: &str) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join(name), content).unwrap();
}

fn context_with_data_dir(project_id: &str, data_dir: &std::path::Path) -> ProjectContext {
    let mut config = ServerConfig::default();
    config.server.data_dir = data_dir.to_string_lossy().to_string();
    ProjectContext::new(
        NormalizationConfig::default(),
        Taxonomy::default(),
        Arc::new(Default::default()),
        config,
        project_id.to_string(),
    )
}

#[test]
fn test_project_creation() {
    let store = ProjectStore::new();
    let ctx = store.get_or_create("proj_1");

    // Check if engines are initialized
    assert!(ctx.main.get_memories().is_empty());
    assert!(ctx.aliases.get_memories().is_empty());
    assert!(ctx.lexicon.get_memories().is_empty());
}

#[test]
fn test_project_persistence() {
    let store = ProjectStore::new();
    let ctx1 = store.get_or_create("proj_1");

    // Add a memory to ctx1
    ctx1.main.add_memory(
        "test".to_string(),
        vec!["cue".to_string()],
        None,
        MainStats::default(),
        false,
    );

    // Get the same project again
    let ctx2 = store.get_or_create("proj_1");

    // Should be the same instance (sharing data)
    assert_eq!(ctx2.main.get_memories().len(), 1);
}

#[test]
fn test_context_isolation() {
    let store = ProjectStore::new();
    let ctx1 = store.get_or_create("proj_A");
    let ctx2 = store.get_or_create("proj_B");

    ctx1.main
        .add_memory("A".to_string(), vec![], None, MainStats::default(), false);
    ctx2.main
        .add_memory("B".to_string(), vec![], None, MainStats::default(), false);

    assert_eq!(ctx1.main.get_memories().len(), 1);
    assert_eq!(ctx2.main.get_memories().len(), 1);

    // Verify content is different
    let _mems1 = ctx1.main.get_memories();
    let _mems2 = ctx2.main.get_memories();

    // Verify they are different objects in memory (Arc pointers)
    assert!(!Arc::ptr_eq(&ctx1, &ctx2));
}

#[test]
fn project_context_lifecycle_getters_and_artifact_reload_are_covered() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = ServerConfig::default();
    config.server.data_dir = tmp.path().to_string_lossy().to_string();
    let ctx = ProjectContext::new_with_encoder(
        NormalizationConfig::default(),
        Taxonomy::default(),
        Arc::new(Default::default()),
        config,
        "health".to_string(),
        None,
    );

    let before = ctx.get_last_activity();
    ctx.touch();
    assert!(ctx.get_last_activity() >= before);
    assert_eq!(ctx.total_memories(), 0);
    assert_eq!(ctx.get_cue_frequency("missing"), 0);
    assert_eq!(ctx.cuebridge_artifact_summary().artifact_count, 0);

    let artifact_dir = tmp.path().join("artifacts").join("health");
    write_artifact(
        &artifact_dir,
        "aliases.json",
        r#"{"schema_version":1,"artifact_type":"alias_pack","name":"health-aliases","entries":[{"id":"a1","from":"coffee","to":"tea","confidence":0.9}]}"#,
    );
    let summary = ctx.reload_cuebridge_artifacts(&tmp.path().to_string_lossy(), "health");
    assert_eq!(summary.artifact_count, 1);
    assert_eq!(summary.alias_entry_count, 1);
    assert_eq!(ctx.cuebridge_artifact_summary().alias_entry_count, 1);
}

#[test]
fn project_context_resolves_skip_lexicon_cache_and_language_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let ctx = context_with_data_dir("resolve", tmp.path());
    assert_eq!(ctx.resolve_cues_from_text("", true), (Vec::new(), Vec::new(), Vec::new()));

    let skipped = ctx.resolve_cues_from_text("Coffee and tea", true);
    assert!(!skipped.0.is_empty());
    assert!(skipped.1.is_empty());
    assert!(!skipped.2.is_empty());

    let lex_id = ctx.lexicon.add_memory(
        "beverage".to_string(),
        vec!["coffee".to_string()],
        None,
        cuemap::structures::LexiconStats::default(),
        true,
    );
    let resolved = ctx.resolve_cues_from_text("coffee", false);
    assert_eq!(resolved.1, vec![lex_id]);
    assert!(resolved.0.iter().any(|cue| cue == "beverage"));
    let cached = ctx.resolve_cues_from_text("coffee", false);
    assert_eq!(cached.0, resolved.0);
    assert!(cached.1.is_empty());

    let python = ctx.resolve_cues_from_text_with_lang("Coffee", false, cuemap::nl::Language::Python);
    assert!(!python.2.is_empty());
    assert!(ctx.query_cache.len() >= 2);

    let invalid = ctx.resolve_cues_from_text("bad:key", true);
    assert!(!invalid.2.is_empty());
}

#[test]
fn project_context_symbol_router_routes_and_refreshes_after_index_changes() {
    let tmp = tempfile::tempdir().unwrap();
    let ctx = context_with_data_dir("symbols", tmp.path());
    ctx.main.add_memory(
        "run implementation".to_string(),
        vec!["defines_function:run".to_string()],
        None,
        MainStats::default(),
        true,
    );

    let routed = ctx.resolve_cues_from_text("where is run used", true);
    assert!(routed.0.iter().any(|cue| cue == "calls_function:run"));
    assert!(routed.0.iter().any(|cue| cue == "calls_method:run"));
    let generic = ctx.resolve_cues_from_text("run", true);
    assert!(!generic.0.is_empty());

    ctx.main.add_memory(
        "new symbol".to_string(),
        vec!["defines_function:deploy".to_string()],
        None,
        MainStats::default(),
        true,
    );
    let refreshed = ctx.resolve_cues_from_text("what does deploy do", true);
    assert!(refreshed.0.iter().any(|cue| cue == "defines_function:deploy"));
    assert!(ctx.main.cue_index_version() > 0);
}

#[test]
fn project_context_inline_alias_expansion_filters_and_deduplicates() {
    let tmp = tempfile::tempdir().unwrap();
    let ctx = context_with_data_dir("inline-alias", tmp.path());
    for cue in ["tea", "coffee", "cocoa"] {
        ctx.main.add_memory(cue.to_string(), vec![cue.to_string()], None, MainStats::default(), true);
    }
    write_artifact(
        &tmp.path().join("artifacts").join("inline-alias"),
        "artifact-alias.json",
        r#"{"schema_version":1,"artifact_type":"alias_pack","name":"artifact-alias","entries":[{"id":"bridge-1","from":"coffee","to":"tea","weight":0.7,"confidence":0.8}]}"#,
    );
    ctx.reload_cuebridge_artifacts(&tmp.path().to_string_lossy(), "inline-alias");

    let add_alias = |content: &str, cues: Vec<String>| {
        ctx.aliases.add_memory(content.to_string(), cues, None, MainStats::default(), true);
    };
    add_alias(r#"{"from":"coffee","to":"tea"}"#, vec!["type:alias".into(), "from:coffee".into(), "status:active".into()]);
    add_alias(r#"{"from":"wrong","to":"cocoa","downweight":0.2}"#, vec!["type:alias".into(), "from:coffee".into(), "status:active".into()]);
    add_alias("not-json", vec!["type:alias".into(), "from:coffee".into(), "status:active".into()]);

    let (expanded, trace) = ctx.expand_query_cues_with_trace(vec!["coffee".into(), "tea".into(), "missing".into()], &["coffee".into()]);
    assert_eq!(expanded.iter().filter(|(cue, _)| cue == "tea").count(), 1);
    assert!(expanded.iter().any(|(cue, weight)| cue == "coffee" && *weight == 1.0));
    assert_eq!(trace.len(), 1);
    assert_eq!(trace[0].entry_id, "bridge-1");
    assert!(!expanded.iter().any(|(cue, _)| cue == "missing"));
    assert_eq!(ctx.expand_query_cues(vec!["coffee".into()], &[]), vec![("coffee".into(), 1.0)]);
}

#[test]
fn project_store_reuses_context_and_keeps_projects_isolated() {
    let store = ProjectStore::new();
    let first = store.get_or_create("same");
    let second = store.get_or_create("same");
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(store.projects.len(), 1);
    first.last_activity.store(42, Ordering::Relaxed);
    assert_eq!(second.get_last_activity(), 42);
    let other = store.get_or_create("other");
    assert_eq!(store.projects.len(), 2);
    assert!(!Arc::ptr_eq(&first, &other));
}
