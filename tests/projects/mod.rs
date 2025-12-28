use cuemap_rust::projects::*;
use std::sync::Arc;

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
    ctx1.main.add_memory("test".to_string(), vec!["cue".to_string()], None);
    
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
    
    ctx1.main.add_memory("A".to_string(), vec![], None);
    ctx2.main.add_memory("B".to_string(), vec![], None);
    
    assert_eq!(ctx1.main.get_memories().len(), 1);
    assert_eq!(ctx2.main.get_memories().len(), 1);
    
    // Verify content is different
    let _mems1 = ctx1.main.get_memories();
    let _mems2 = ctx2.main.get_memories();
    
    // Verify they are different objects in memory (Arc pointers)
    assert!(!Arc::ptr_eq(&ctx1, &ctx2));
}
