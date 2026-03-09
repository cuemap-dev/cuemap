use cuemap::agent::ingester::Ingester;
use cuemap::agent::AgentConfig;
use cuemap::jobs::{JobQueue, ProjectProvider};
use std::fs;
use std::sync::Arc;
use tempfile::tempdir;

struct MockProvider;
impl ProjectProvider for MockProvider {
    fn get_project(&self, _project_id: &str) -> Option<Arc<cuemap::projects::ProjectContext>> { None }
    fn save_project(&self, _project_id: &str) -> Result<(), String> { Ok(()) }
    fn list_active_projects(&self) -> Vec<String> { Vec::new() }
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
        ignored_patterns: vec!["custom_ignored.txt".to_string()],
        ignored_extensions: vec!["bak".to_string()],
    };
    
    fs::write(watch_path.join("custom_ignored.txt"), "ignore me").unwrap();
    fs::write(watch_path.join("old.bak"), "backup").unwrap();

    let mut ingester = Ingester::new(config, job_queue);
    
    ingester.scan_all().await.unwrap();
    
    let tracked = ingester.get_file_hashes();
    
    let is_tracked = |rel_path: &str| {
        let p = fs::canonicalize(watch_path.join(rel_path)).unwrap_or_else(|_| watch_path.join(rel_path));
        let p_str = p.to_string_lossy().to_lowercase();
        tracked.contains_key(&p_str)
    };

    // Verify noisy files are NOT tracked
    assert!(!is_tracked("package-lock.json"), "package-lock.json should be ignored");
    assert!(!is_tracked("Cargo.lock"), "Cargo.lock should be ignored");
    assert!(!is_tracked("tsconfig.json"), "tsconfig.json should be ignored");
    assert!(!is_tracked(".DS_Store"), ".DS_Store should be ignored");
    assert!(!is_tracked("poetry.lock"), "poetry.lock should be ignored");
    assert!(!is_tracked("go.sum"), "go.sum should be ignored");
    assert!(!is_tracked("Gemfile.lock"), "Gemfile.lock should be ignored");
    assert!(!is_tracked("composer.lock"), "composer.lock should be ignored");
    assert!(!is_tracked(".idea/workspace.xml"), ".idea/ should be ignored");
    assert!(!is_tracked("__pycache__/main.cpython-39.pyc"), "__pycache__/ should be ignored");
    assert!(!is_tracked("build/app.jar"), "build/ should be ignored");
    assert!(!is_tracked("target/rust-binary"), "target/ should be ignored");
    
    // Verify custom ignore patterns
    assert!(!is_tracked("test.tmp"), "*.tmp should be ignored by .cuemapignore");
    assert!(!is_tracked("secret.txt"), "secret.txt should be ignored by .cuemapignore");
    assert!(!is_tracked("sub/ignored_in_sub.txt"), "sub/ignored_in_sub.txt should be ignored by .antigravityignore");
    
    // Verify config-based ignores
    assert!(!is_tracked("custom_ignored.txt"), "custom_ignored.txt should be ignored by config");
    assert!(!is_tracked("old.bak"), "*.bak should be ignored by config extensions");

    // Verify valid files ARE tracked
    assert!(is_tracked("main.rs"), "main.rs should be tracked");
    assert!(is_tracked("README.md"), "README.md should be tracked");
    assert!(is_tracked("sub/valid_in_sub.txt"), "sub/valid_in_sub.txt should be tracked");
}
