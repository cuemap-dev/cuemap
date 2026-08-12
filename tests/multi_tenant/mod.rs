use cuemap::config::TuningConfig;
use cuemap::multi_tenant::*;
use cuemap::structures::MainStats;
use std::fs;
use std::time::Duration;
use tempfile::tempdir;
use tokio::time::sleep;

#[test]
fn test_project_id_validation() {
    assert!(validate_project_id("valid-project-123"));
    assert!(validate_project_id("Project_Alpha"));
    assert!(!validate_project_id("sh")); // too short
    assert!(!validate_project_id("very-long-project-id-that-exceeds-the-sixty-four-character-limit-defined-in-the-validation-logic"));
    assert!(!validate_project_id("project@id")); // invalid char
}

#[test]
fn test_multi_tenant_isolation() {
    let dir = tempdir().unwrap();
    let engine = MultiTenantEngine::with_snapshots_dir(
        dir.path(),
        TuningConfig::default(),
    );

    let ctx1 = engine.get_or_create_project("proj1".to_string()).unwrap();
    let ctx2 = engine.get_or_create_project("proj2".to_string()).unwrap();

    ctx1.main.add_memory(
        "Project 1 content".to_string(),
        vec!["cue1".to_string()],
        None,
        MainStats::default(),
        false,
    );
    ctx2.main.add_memory(
        "Project 2 content".to_string(),
        vec!["cue2".to_string()],
        None,
        MainStats::default(),
        false,
    );

    // Proj1 should not see cue2
    assert_eq!(
        ctx1.main
            .recall(vec!["cue2".to_string()], 10, false, None)
            .len(),
        0
    );
    // Proj2 should not see cue1
    assert_eq!(
        ctx2.main
            .recall(vec!["cue1".to_string()], 10, false, None)
            .len(),
        0
    );

    assert_eq!(ctx1.main.get_memories().len(), 1);
    assert_eq!(ctx2.main.get_memories().len(), 1);
}

#[test]
fn test_snapshot_roundtrip() {
    let dir = tempdir().unwrap();
    let snapshots_dir = dir.path().join("snapshots");
    fs::create_dir_all(&snapshots_dir).unwrap();

    let project_id = "persistence_test".to_string();

    {
        let engine = MultiTenantEngine::with_snapshots_dir(
            &snapshots_dir,
            TuningConfig::default(),
        );
        let ctx = engine.get_or_create_project(project_id.clone()).unwrap();
        ctx.main.add_memory(
            "persist me".to_string(),
            vec!["save:true".to_string()],
            None,
            MainStats::default(),
            false,
        );

        // Save
        engine
            .save_project(&project_id)
            .expect("Should save successfully");
    }

    // Restart engine
    {
        let engine = MultiTenantEngine::with_snapshots_dir(
            &snapshots_dir,
            TuningConfig::default(),
        );

        // Should be able to load
        let ctx = engine
            .load_project(&project_id)
            .expect("Should load successfully");
        let results = ctx
            .main
            .recall(vec!["save:true".to_string()], 10, false, None);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "persist me");
    }
}

#[test]
fn test_load_all_restores_every_project_snapshot() {
    let dir = tempdir().unwrap();
    let project_id = "load-all-project".to_string();
    let first = MultiTenantEngine::with_snapshots_dir(dir.path(), TuningConfig::default());
    let context = first.get_or_create_project(project_id.clone()).unwrap();
    context.main.add_memory(
        "load all content".to_string(),
        vec!["load-all".to_string()],
        None,
        MainStats::default(),
        true,
    );
    first.save_project(&project_id).unwrap();

    let second = MultiTenantEngine::with_snapshots_dir(dir.path(), TuningConfig::default());
    let results = second.load_all();
    assert!(matches!(results.get(&project_id), Some(Ok(()))));
    let restored = second.get_project(&project_id).expect("project should be restored");
    assert_eq!(restored.main.total_memories(), 1);
}

#[test]
fn test_delete_project() {
    let dir = tempdir().unwrap();
    let engine = MultiTenantEngine::with_snapshots_dir(
        dir.path(),
        TuningConfig::default(),
    );

    let project_id = "to_delete";
    let _ = engine
        .get_or_create_project(project_id.to_string())
        .unwrap();

    assert!(engine.get_project(&project_id.to_string()).is_some());
    assert!(engine.delete_project(&project_id.to_string()));
    assert!(engine.get_project(&project_id.to_string()).is_none());
}

#[test]
fn repository_ingestion_scope_is_persisted_in_project_metadata() {
    let dir = tempdir().unwrap();
    let snapshots_dir = dir.path().join("snapshots");
    let watch_dir = dir.path().join("repo");
    fs::create_dir_all(&snapshots_dir).unwrap();
    fs::create_dir(&watch_dir).unwrap();
    let engine = MultiTenantEngine::with_snapshots_dir(
        &snapshots_dir,
        TuningConfig::default(),
    );
    let project_id = "scope-persistence";
    engine
        .get_or_create_project(project_id.to_string())
        .unwrap();

    engine
        .set_project_watch_config(
            project_id,
            watch_dir.to_string_lossy().to_string(),
            vec!["src".to_string(), "README.md".to_string()],
            vec!["docs/**".to_string()],
            vec!["log".to_string()],
        )
        .unwrap();

    let metadata = engine
        .load_project_meta(&project_id.to_string())
        .unwrap();
    assert!(metadata.agent_enabled);
    assert_eq!(metadata.included_paths, vec!["src", "README.md"]);
    assert_eq!(metadata.ignored_patterns, vec!["docs/**"]);
    assert_eq!(metadata.ignored_extensions, vec!["log"]);
}

#[tokio::test]
async fn periodic_snapshots_persist_projects_created_after_scheduler_start() {
    let dir = tempdir().unwrap();
    let snapshots_dir = dir.path().join("snapshots");
    fs::create_dir_all(&snapshots_dir).unwrap();

    let engine = MultiTenantEngine::with_snapshots_dir(
        &snapshots_dir,
        TuningConfig::default(),
    );
    engine.start_periodic_snapshots(Duration::from_millis(20));

    let project_id = "periodic-new-project".to_string();
    let context = engine.get_or_create_project(project_id.clone()).unwrap();
    context.main.add_memory(
        "persisted by the periodic scheduler".to_string(),
        vec!["snapshot:periodic".to_string()],
        None,
        MainStats::default(),
        false,
    );

    for _ in 0..100 {
        let reloaded = MultiTenantEngine::with_snapshots_dir(
            &snapshots_dir,
            TuningConfig::default(),
        );
        if let Ok(reloaded_context) = reloaded.load_project(&project_id) {
            let matches = reloaded_context.main.recall(
                vec!["snapshot:periodic".to_string()],
                10,
                false,
                None,
            );
            if matches.len() == 1 {
                assert_eq!(matches[0].content, "persisted by the periodic scheduler");
                return;
            }
        }
        sleep(Duration::from_millis(20)).await;
    }

    panic!("periodic snapshot did not persist a project created after startup");
}
