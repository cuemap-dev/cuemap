use cuemap_rust::jobs::*;

#[test]
fn test_lexicon_trainability() {
    assert!(is_lexicon_trainable("topic:programming"));
    assert!(is_lexicon_trainable("lang:python"));
    assert!(is_lexicon_trainable("type:function"));
    
    // Should be filtered out
    assert!(!is_lexicon_trainable("path:/users/kaan/test.py"));
    assert!(!is_lexicon_trainable("id:123"));
    assert!(!is_lexicon_trainable("memory_id:abc"));
    assert!(!is_lexicon_trainable("source:agent"));
    assert!(!is_lexicon_trainable("file:/tmp/foo"));
}
