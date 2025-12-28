use cuemap_rust::multi_tenant::*;
use std::fs;
use tempfile::tempdir;

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
    let engine = MultiTenantEngine::with_snapshots_dir(dir.path());
    
    let ctx1 = engine.get_or_create_project("proj1".to_string());
    let ctx2 = engine.get_or_create_project("proj2".to_string());
    
    ctx1.main.add_memory("Project 1 content".to_string(), vec!["cue1".to_string()], None);
    ctx2.main.add_memory("Project 2 content".to_string(), vec!["cue2".to_string()], None);
    
    // Proj1 should not see cue2
    assert_eq!(ctx1.main.recall(vec!["cue2".to_string()], 10, false).len(), 0);
    // Proj2 should not see cue1
    assert_eq!(ctx2.main.recall(vec!["cue1".to_string()], 10, false).len(), 0);
    
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
        let engine = MultiTenantEngine::with_snapshots_dir(&snapshots_dir);
        let ctx = engine.get_or_create_project(project_id.clone());
        ctx.main.add_memory("persist me".to_string(), vec!["save:true".to_string()], None);
        
        // Save
        engine.save_project(&project_id).expect("Should save successfully");
    }
    
    // Restart engine
    {
        let engine = MultiTenantEngine::with_snapshots_dir(&snapshots_dir);
        
        // Should be able to load
        let ctx = engine.load_project(&project_id).expect("Should load successfully");
        let results = ctx.main.recall(vec!["save:true".to_string()], 10, false);
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "persist me");
    }
}

#[test]
fn test_delete_project() {
    let dir = tempdir().unwrap();
    let engine = MultiTenantEngine::with_snapshots_dir(dir.path());
    
    let project_id = "to_delete";
    engine.get_or_create_project(project_id.to_string());
    
    assert!(engine.get_project(&project_id.to_string()).is_some());
    assert!(engine.delete_project(&project_id.to_string()));
    assert!(engine.get_project(&project_id.to_string()).is_none());
}
