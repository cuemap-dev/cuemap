
use cuemap::crypto::{self, EncryptionKey};
use cuemap::engine::CueMapEngine;
use cuemap::structures::{MainStats, Memory};
use std::sync::Arc;
use std::collections::HashMap;

#[test]
fn test_crypto_roundtrip() {
    let original = "Hello, world! This is a test.";
    let key = EncryptionKey::new(vec![42u8; 32]); // Dummy key
    
    // 1. Compress & Encrypt
    let compressed = crypto::compress(original.as_bytes()).expect("Compress failed");
    let encrypted = crypto::encrypt(&compressed, &key).expect("Encrypt failed");
    
    // 2. Decrypt & Decompress
    let decrypted = crypto::decrypt(&encrypted, &key).expect("Decrypt failed");
    let decompressed = crypto::decompress(&decrypted).expect("Decompress failed");
    
    // 3. Verify
    assert_eq!(original.as_bytes(), decompressed.as_slice());
    
    // 4. Verify Compression only
    let compressed_only = crypto::compress(original.as_bytes()).expect("Compress failed");
    let decompressed_only = crypto::decompress(&compressed_only).expect("Decompress failed");
    assert_eq!(original.as_bytes(), decompressed_only.as_slice());
}

#[test]
fn test_memory_payload_integration() {
    let content = "Secret Memory Content";
    let key = Arc::new(EncryptionKey::new(vec![1u8; 32]));
    
    // Test Encrypted Flow
    let payload = Memory::<MainStats>::create_payload(content, Some(&key)).expect("create payload failed");
    let memory = Memory::<MainStats>::new(payload, None);
    
    let accessed = memory.access_content(Some(&key)).expect("access content failed");
    assert_eq!(content, accessed);
    
    // Verify accessing without key fails
    assert!(memory.access_content(None).is_err());
    
    // Test Compressed-Only Flow
    let payload_plain = Memory::<MainStats>::create_payload(content, None).expect("create plain payload failed");
    let memory_plain = Memory::<MainStats>::new(payload_plain, None);
    
    let accessed_plain = memory_plain.access_content(None).expect("access plain failed");
    assert_eq!(content, accessed_plain);
}

#[test]
fn test_engine_integration() {
    let key = EncryptionKey::new(vec![7u8; 32]);
    let mut engine = CueMapEngine::<MainStats>::with_key(Some(key));
    
    let id = engine.add_memory(
        "Engine Secret".to_string(),
        vec!["test".to_string()],
        None,
        MainStats::default(),
        false
    );
    
    // Recall
    let results = engine.recall(vec!["test".to_string()], 1, false, None);
    assert!(!results.is_empty());
    assert_eq!(results[0].content, "Engine Secret");
    
    // Test persistence/loading behavior simulation
    // (We iterate memories and check content access)
    let memories = engine.get_memories();
    let mem = memories.get(&id).expect("Memory not found");
    // Direct access via helper
    // Note: engine has the key, but here we access memory directly.
    // memory.access_content requires key.
    
    // We can't access engine.master_key from here (it's private/not exposed via getter?)
    // But we know the key.
    // Let's create `key` again.
    let key_clone = EncryptionKey::new(vec![7u8; 32]);
    let content = mem.access_content(Some(&key_clone)).expect("Decrypt failed");
    assert_eq!(content, "Engine Secret");
}

#[test]
fn test_smart_access_migration() {
    // Scenario: Data is stored as Compressed (Legacy/No Key), but Engine now has a Key.
    // We expect access_content to fail authentication (decrypt) if we forced it, 
    // but Smart Access should detect it's just compressed and read it successfully.

    let content = "Legacy Content";
    let key = EncryptionKey::new(vec![8u8; 32]); // Key exists
    
    // 1. Create purely compressed payload (simulate legacy data)
    let payload = Memory::<MainStats>::create_payload(content, None).expect("create plain payload");
    
    // 2. Wrap in Memory
    let memory = Memory::<MainStats>::new(payload, None);
    
    // 3. Access WITH key
    // Should pass because of magic byte check
    let accessed = memory.access_content(Some(&key)).expect("Smart Access failed");
    assert_eq!(content, accessed);
    
    // 4. Verify accessing Encrypted data without key still fails
    let encrypted_payload = Memory::<MainStats>::create_payload(content, Some(&key)).expect("create encrypted");
    let memory_enc = Memory::<MainStats>::new(encrypted_payload, None);
    
    let result = memory_enc.access_content(None);
    assert!(result.is_err(), "Should fail to access encrypted data without key");
}
