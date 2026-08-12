use cuemap::agent::ingester::Ingester;
use cuemap::agent::AgentConfig;
use cuemap::config::TuningConfig;
use cuemap::jobs::{JobQueue, ProjectProvider};
use cuemap::multi_tenant::MultiTenantEngine;
use std::fs;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::time::{sleep, timeout, Duration};

struct MockProvider;
impl ProjectProvider for MockProvider {
    fn get_project(&self, _project_id: &str) -> Option<Arc<cuemap::projects::ProjectContext>> {
        None
    }
    fn save_project(&self, _project_id: &str) -> Result<(), String> {
        Ok(())
    }
    fn list_active_projects(&self) -> Vec<String> {
        Vec::new()
    }
}

#[tokio::test]
async fn test_ingester_filters_noise_and_ignore_files() {
    let dir = tempdir().unwrap();
    let watch_path = dir.path().to_path_buf();

    // Create some noisy files
    fs::write(watch_path.join("package-lock.json"), "{}").unwrap();
    fs::write(watch_path.join("Cargo.lock"), "").unwrap();
    fs::write(watch_path.join("tsconfig.json"), "{}").unwrap();
    fs::write(watch_path.join(".DS_Store"), "").unwrap();
    fs::write(watch_path.join("poetry.lock"), "").unwrap();
    fs::write(watch_path.join("go.sum"), "").unwrap();
    fs::write(watch_path.join("Gemfile.lock"), "").unwrap();
    fs::write(watch_path.join("composer.lock"), "").unwrap();

    // Create some noisy directories
    let idea = watch_path.join(".idea");
    fs::create_dir(&idea).unwrap();
    fs::write(idea.join("workspace.xml"), "").unwrap();

    let pycache = watch_path.join("__pycache__");
    fs::create_dir(&pycache).unwrap();
    fs::write(pycache.join("main.cpython-39.pyc"), "").unwrap();

    let build_dir = watch_path.join("build");
    fs::create_dir(&build_dir).unwrap();
    fs::write(build_dir.join("app.jar"), "").unwrap();

    let target_dir = watch_path.join("target");
    fs::create_dir(&target_dir).unwrap();
    fs::write(target_dir.join("rust-binary"), "").unwrap();

    // Create some valid files
    fs::write(watch_path.join("main.rs"), "fn main() {}").unwrap();
    fs::write(watch_path.join("README.md"), "# Hello").unwrap();

    // Create a custom ignore file
    fs::write(watch_path.join(".cuemapignore"), "*.tmp\nsecret.txt").unwrap();
    fs::write(watch_path.join("test.tmp"), "temp").unwrap();
    fs::write(watch_path.join("secret.txt"), "shhh").unwrap();

    // Create a subfolder with another ignore file
    let sub = watch_path.join("sub");
    fs::create_dir(&sub).unwrap();
    fs::write(sub.join(".antigravityignore"), "ignored_in_sub.txt").unwrap();
    fs::write(sub.join("ignored_in_sub.txt"), "hidden").unwrap();
    fs::write(sub.join("valid_in_sub.txt"), "visible").unwrap();

    let job_queue = Arc::new(JobQueue::new(Arc::new(MockProvider), None, true));
    let config = AgentConfig {
        project_id: "test_project".to_string(),
        watch_dir: watch_path.to_string_lossy().to_string(),
        throttle_ms: 0,
        state_file: None,
        included_paths: Vec::new(),
        ignored_patterns: vec!["custom_ignored.txt".to_string()],
        ignored_extensions: vec!["bak".to_string()],
    };

    fs::write(watch_path.join("custom_ignored.txt"), "ignore me").unwrap();
    fs::write(watch_path.join("old.bak"), "backup").unwrap();

    let mut ingester = Ingester::new(config, job_queue);

    ingester.scan_all().await.unwrap();

    let tracked = ingester.get_file_hashes();

    let is_tracked = |rel_path: &str| {
        let p = fs::canonicalize(watch_path.join(rel_path))
            .unwrap_or_else(|_| watch_path.join(rel_path));
        let p_str = p.to_string_lossy().to_lowercase();
        tracked.contains_key(&p_str)
    };

    // Verify noisy files are NOT tracked
    assert!(
        !is_tracked("package-lock.json"),
        "package-lock.json should be ignored"
    );
    assert!(!is_tracked("Cargo.lock"), "Cargo.lock should be ignored");
    assert!(
        !is_tracked("tsconfig.json"),
        "tsconfig.json should be ignored"
    );
    assert!(!is_tracked(".DS_Store"), ".DS_Store should be ignored");
    assert!(!is_tracked("poetry.lock"), "poetry.lock should be ignored");
    assert!(!is_tracked("go.sum"), "go.sum should be ignored");
    assert!(
        !is_tracked("Gemfile.lock"),
        "Gemfile.lock should be ignored"
    );
    assert!(
        !is_tracked("composer.lock"),
        "composer.lock should be ignored"
    );
    assert!(
        !is_tracked(".idea/workspace.xml"),
        ".idea/ should be ignored"
    );
    assert!(
        !is_tracked("__pycache__/main.cpython-39.pyc"),
        "__pycache__/ should be ignored"
    );
    assert!(!is_tracked("build/app.jar"), "build/ should be ignored");
    assert!(
        !is_tracked("target/rust-binary"),
        "target/ should be ignored"
    );

    // Verify custom ignore patterns
    assert!(
        !is_tracked("test.tmp"),
        "*.tmp should be ignored by .cuemapignore"
    );
    assert!(
        !is_tracked("secret.txt"),
        "secret.txt should be ignored by .cuemapignore"
    );
    assert!(
        !is_tracked("sub/ignored_in_sub.txt"),
        "sub/ignored_in_sub.txt should be ignored by .antigravityignore"
    );

    // Verify config-based ignores
    assert!(
        !is_tracked("custom_ignored.txt"),
        "custom_ignored.txt should be ignored by config"
    );
    assert!(
        !is_tracked("old.bak"),
        "*.bak should be ignored by config extensions"
    );

    // Verify valid files ARE tracked
    assert!(is_tracked("main.rs"), "main.rs should be tracked");
    assert!(is_tracked("README.md"), "README.md should be tracked");
    assert!(
        is_tracked("sub/valid_in_sub.txt"),
        "sub/valid_in_sub.txt should be tracked"
    );
}

#[tokio::test]
async fn nested_and_context_specific_ignore_files_do_not_hide_the_repository() {
    let dir = tempdir().unwrap();
    let watch_path = dir.path().to_path_buf();
    fs::write(watch_path.join("root.rs"), "fn root() {}").unwrap();
    fs::write(watch_path.join(".dockerignore"), "*\n").unwrap();

    let generated = watch_path.join("generated");
    fs::create_dir(&generated).unwrap();
    fs::write(generated.join(".gitignore"), "*\n").unwrap();
    fs::write(generated.join("ignored.rs"), "fn ignored() {}").unwrap();

    let source = watch_path.join("src");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("main.rs"), "fn main() {}").unwrap();

    let job_queue = Arc::new(JobQueue::new(Arc::new(MockProvider), None, true));
    let config = AgentConfig {
        project_id: "scoped_ignores".to_string(),
        watch_dir: watch_path.to_string_lossy().to_string(),
        throttle_ms: 0,
        state_file: None,
        included_paths: Vec::new(),
        ignored_patterns: Vec::new(),
        ignored_extensions: Vec::new(),
    };
    let mut ingester = Ingester::new(config, job_queue);

    let preview = ingester.preview_scope().unwrap();
    assert_eq!(preview.supported_files, 2);
    assert!(preview.entries.iter().any(|entry| entry.path == "root.rs"));
    assert!(preview.entries.iter().any(|entry| entry.path == "src"));
    assert!(!preview.entries.iter().any(|entry| entry.path == "generated"));

    ingester.scan_all().await.unwrap();
    let tracked = ingester.get_file_hashes();
    let root = fs::canonicalize(watch_path.join("root.rs"))
        .unwrap()
        .to_string_lossy()
        .to_lowercase();
    let main = fs::canonicalize(source.join("main.rs"))
        .unwrap()
        .to_string_lossy()
        .to_lowercase();
    let ignored = fs::canonicalize(generated.join("ignored.rs"))
        .unwrap()
        .to_string_lossy()
        .to_lowercase();
    assert!(tracked.contains_key(&root));
    assert!(tracked.contains_key(&main));
    assert!(!tracked.contains_key(&ignored));
}

#[tokio::test]
async fn selected_scope_applies_to_initial_and_new_files() {
    let dir = tempdir().unwrap();
    let watch_path = dir.path().to_path_buf();
    fs::create_dir(watch_path.join("src")).unwrap();
    fs::create_dir(watch_path.join("docs")).unwrap();
    fs::write(watch_path.join("src/main.rs"), "fn main() {}").unwrap();
    fs::write(watch_path.join("docs/guide.md"), "# Guide").unwrap();

    let job_queue = Arc::new(JobQueue::new(Arc::new(MockProvider), None, true));
    let config = AgentConfig {
        project_id: "selected_scope".to_string(),
        watch_dir: watch_path.to_string_lossy().to_string(),
        throttle_ms: 0,
        state_file: None,
        included_paths: vec!["src".to_string()],
        ignored_patterns: Vec::new(),
        ignored_extensions: Vec::new(),
    };
    let mut ingester = Ingester::new(config, job_queue);

    let preview = ingester.preview_scope().unwrap();
    assert_eq!(preview.supported_files, 1);
    assert_eq!(preview.entries.len(), 1);
    assert_eq!(preview.entries[0].path, "src");

    ingester.scan_all().await.unwrap();
    let main_path = fs::canonicalize(watch_path.join("src/main.rs"))
        .unwrap()
        .to_string_lossy()
        .to_lowercase();
    let guide_path = fs::canonicalize(watch_path.join("docs/guide.md"))
        .unwrap()
        .to_string_lossy()
        .to_lowercase();
    assert!(ingester.get_file_hashes().contains_key(&main_path));
    assert!(!ingester.get_file_hashes().contains_key(&guide_path));

    let new_source = watch_path.join("src/new.rs");
    fs::write(&new_source, "pub fn new_file() {}").unwrap();
    ingester.process_file_path(new_source.clone()).await.unwrap();
    let new_source = fs::canonicalize(new_source)
        .unwrap()
        .to_string_lossy()
        .to_lowercase();
    assert!(ingester.get_file_hashes().contains_key(&new_source));

    let new_doc = watch_path.join("docs/new.md");
    fs::write(&new_doc, "# Not selected").unwrap();
    ingester.process_file_path(new_doc.clone()).await.unwrap();
    let new_doc = fs::canonicalize(new_doc)
        .unwrap()
        .to_string_lossy()
        .to_lowercase();
    assert!(!ingester.get_file_hashes().contains_key(&new_doc));
}

#[tokio::test]
async fn changed_cuemapignore_is_reloaded_and_reconciled() {
    let dir = tempdir().unwrap();
    let watch_path = dir.path().to_path_buf();
    fs::write(watch_path.join("keep.rs"), "pub fn keep() {}").unwrap();
    fs::write(watch_path.join("remove.rs"), "pub fn remove() {}").unwrap();

    let job_queue = Arc::new(JobQueue::new(Arc::new(MockProvider), None, true));
    let config = AgentConfig {
        project_id: "ignore_reload".to_string(),
        watch_dir: watch_path.to_string_lossy().to_string(),
        throttle_ms: 0,
        state_file: None,
        included_paths: Vec::new(),
        ignored_patterns: Vec::new(),
        ignored_extensions: Vec::new(),
    };
    let mut ingester = Ingester::new(config, job_queue);
    ingester.scan_all().await.unwrap();
    assert_eq!(ingester.get_file_hashes().len(), 2);

    fs::write(watch_path.join(".cuemapignore"), "remove.rs\n").unwrap();
    ingester.reload_filters_and_rescan().await.unwrap();
    assert_eq!(ingester.get_file_hashes().len(), 1);

    fs::remove_file(watch_path.join(".cuemapignore")).unwrap();
    ingester.reload_filters_and_rescan().await.unwrap();
    assert_eq!(ingester.get_file_hashes().len(), 2);
}

#[tokio::test]
async fn replacing_saved_scope_prunes_previously_tracked_paths() {
    let dir = tempdir().unwrap();
    let watch_path = dir.path().join("repo");
    let state_path = dir.path().join("agent-state.json");
    fs::create_dir(&watch_path).unwrap();
    fs::create_dir(watch_path.join("src")).unwrap();
    fs::create_dir(watch_path.join("docs")).unwrap();
    fs::write(watch_path.join("src/main.rs"), "fn main() {}").unwrap();
    fs::write(watch_path.join("docs/guide.md"), "# Guide").unwrap();

    let first_queue = Arc::new(JobQueue::new(Arc::new(MockProvider), None, true));
    let first_config = AgentConfig {
        project_id: "scope_replacement".to_string(),
        watch_dir: watch_path.to_string_lossy().to_string(),
        throttle_ms: 0,
        state_file: Some(state_path.clone()),
        included_paths: Vec::new(),
        ignored_patterns: Vec::new(),
        ignored_extensions: Vec::new(),
    };
    let mut first_ingester = Ingester::new(first_config, first_queue);
    first_ingester.scan_all().await.unwrap();
    first_ingester.save_state(&state_path).unwrap();
    assert_eq!(first_ingester.get_file_hashes().len(), 2);

    let second_queue = Arc::new(JobQueue::new(Arc::new(MockProvider), None, true));
    let second_config = AgentConfig {
        project_id: "scope_replacement".to_string(),
        watch_dir: watch_path.to_string_lossy().to_string(),
        throttle_ms: 0,
        state_file: Some(state_path.clone()),
        included_paths: vec!["src".to_string()],
        ignored_patterns: Vec::new(),
        ignored_extensions: Vec::new(),
    };
    let mut second_ingester = Ingester::new(second_config, second_queue);
    second_ingester.load_state(&state_path).unwrap();
    second_ingester.scan_all().await.unwrap();

    assert_eq!(second_ingester.get_file_hashes().len(), 1);
    let tracked_path = second_ingester
        .get_file_hashes()
        .keys()
        .next()
        .unwrap();
    assert!(tracked_path.ends_with("/src/main.rs"));
}

#[tokio::test]
async fn repository_file_ingestion_records_explicit_source_type() {
    let dir = tempdir().unwrap();
    let snapshots = dir.path().join("snapshots");
    fs::create_dir_all(&snapshots).unwrap();
    let watch_path = dir.path().join("repo");
    fs::create_dir(&watch_path).unwrap();
    fs::write(watch_path.join("main.rs"), "fn main() {}").unwrap();

    let engine = Arc::new(MultiTenantEngine::with_snapshots_dir(
        snapshots,
        TuningConfig::default(),
    ));
    let project_id = "source_type_test".to_string();
    let context = engine.get_or_create_project(project_id.clone()).unwrap();
    let job_queue = Arc::new(JobQueue::new(engine, None, false));
    let config = AgentConfig {
        project_id: project_id.clone(),
        watch_dir: watch_path.to_string_lossy().to_string(),
        throttle_ms: 0,
        state_file: None,
        included_paths: Vec::new(),
        ignored_patterns: Vec::new(),
        ignored_extensions: Vec::new(),
    };
    let session = job_queue.session_manager.get_or_create(&project_id);
    let mut ingester = Ingester::new(config, job_queue);
    ingester.scan_all().await.unwrap();

    timeout(Duration::from_secs(30), async {
        loop {
            let progress = session.get_progress();
            if progress.writes_total > 0 && progress.writes_completed >= progress.writes_total {
                break;
            }
            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("timed out waiting for repository ingestion to complete");

    let memories = context.main.get_memories();
    let memory_entry = memories
        .iter()
        .next()
        .expect("repository file was not ingested");
    let memory = memory_entry.value();
    assert_eq!(
        memory
            .metadata
            .get("source_type")
            .and_then(|value| value.as_str()),
        Some("repository_file")
    );
    assert!(memory
        .cues
        .iter()
        .any(|cue| cue == "source_type:repository_file"));
}
