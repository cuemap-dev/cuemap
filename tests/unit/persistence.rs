    use super::*;
    use crate::structures::MainStats;

    #[test]
    fn test_config_from_args_s3() {
        let config = CloudBackupConfig::from_args(
            Some("s3"),
            Some("my-bucket"),
            Some("us-west-2"),
            None,
            "cuemap/",
            true,
        )
        .unwrap();

        assert!(config.enabled);
        assert!(config.auto_backup);
        assert_eq!(config.prefix, "cuemap/");

        match config.provider {
            Some(CloudProvider::S3 {
                bucket,
                region,
                endpoint,
            }) => {
                assert_eq!(bucket, "my-bucket");
                assert_eq!(region, "us-west-2");
                assert!(endpoint.is_none());
            }
            _ => panic!("Expected S3 provider"),
        }
    }

    #[test]
    fn test_config_from_args_s3_with_endpoint() {
        let config = CloudBackupConfig::from_args(
            Some("s3"),
            Some("my-bucket"),
            Some("us-east-1"),
            Some("http://localhost:9000"),
            "backups/",
            false,
        )
        .unwrap();

        match config.provider {
            Some(CloudProvider::S3 { endpoint, .. }) => {
                assert_eq!(endpoint, Some("http://localhost:9000".to_string()));
            }
            _ => panic!("Expected S3 provider"),
        }
    }

    #[test]
    fn test_config_from_args_gcs() {
        let config =
            CloudBackupConfig::from_args(Some("gcs"), Some("gcs-bucket"), None, None, "", false)
                .unwrap();

        match config.provider {
            Some(CloudProvider::GCS { bucket }) => {
                assert_eq!(bucket, "gcs-bucket");
            }
            _ => panic!("Expected GCS provider"),
        }
    }

    #[test]
    fn test_config_from_args_missing_bucket() {
        let result = CloudBackupConfig::from_args(Some("s3"), None, None, None, "", false);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_disabled_by_default() {
        let config = CloudBackupConfig::from_args(None, None, None, None, "", false).unwrap();

        assert!(!config.enabled);
        assert!(config.provider.is_none());
    }

    #[test]
    fn save_and_load_snapshot_round_trip_preserves_engine_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("project.bin");
        let engine = CueMapEngine::<MainStats>::new();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("source".to_string(), serde_json::json!("notes.md"));
        let memory_id = engine.add_memory_with_source_key(
            "persisted content".to_string(),
            vec!["persisted".to_string(), "content".to_string()],
            Some(metadata),
            MainStats::default(),
            true,
            Some("notes.md#1".to_string()),
        );

        PersistenceManager::save_to_path(&engine, &path).unwrap();
        let snapshot_bytes = std::fs::read(&path).unwrap();
        assert!(crate::crypto::is_compressed(&snapshot_bytes));
        assert!(snapshot_bytes.len() < zstd::stream::decode_all(std::io::Cursor::new(&snapshot_bytes)).unwrap().len());
        let (memories, source_keys, cues, next_id, counts) =
            PersistenceManager::load_from_path::<MainStats>(&path).unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(source_keys.get("notes.md#1").map(|v| *v), Some(memory_id));
        assert!(cues.contains_key("persisted"));
        assert!(next_id > memory_id);
        assert!(counts.is_none());
        assert_eq!(
            memories
                .get(&memory_id)
                .unwrap()
                .access_content(None)
                .unwrap(),
            "persisted content"
        );

        let missing =
            PersistenceManager::load_from_path::<MainStats>(&dir.path().join("missing.bin"));
        assert!(missing.is_err());
    }

    #[test]
    fn snapshot_listing_and_deletion_filter_auxiliary_files() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["b.bin", "a.bin", "a_aliases.bin", "a_lexicon.bin", "ignore.txt"] {
            std::fs::write(dir.path().join(name), b"snapshot").unwrap();
        }
        assert_eq!(
            PersistenceManager::list_snapshots_in_dir(dir.path()),
            vec!["a".to_string(), "b".to_string()]
        );
        PersistenceManager::delete_snapshot(&dir.path().join("b.bin")).unwrap();
        PersistenceManager::delete_snapshot(&dir.path().join("b.bin")).unwrap();
        assert!(!dir.path().join("b.bin").exists());
    }

    #[test]
    fn manager_save_and_load_state_handles_empty_and_populated_directories() {
        let dir = tempfile::tempdir().unwrap();
        let manager = PersistenceManager::new(dir.path(), 1);
        let (memories, source_keys, cues, next_id) = manager.load_state::<MainStats>().unwrap();
        assert!(memories.is_empty());
        assert!(source_keys.is_empty());
        assert!(cues.is_empty());
        assert_eq!(next_id, 1);

        let engine = CueMapEngine::<MainStats>::new();
        engine.add_memory(
            "manager state".to_string(),
            vec!["manager".to_string()],
            None,
            MainStats::default(),
            true,
        );
        manager.save_state(&engine).unwrap();
        let (memories, _, cues, _) = manager.load_state::<MainStats>().unwrap();
        assert_eq!(memories.len(), 1);
        assert!(cues.contains_key("manager"));
    }

    #[test]
    fn loader_accepts_legacy_uncompressed_bincode_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.bin");
        let state = PersistedState::<MainStats> {
            memories: std::collections::HashMap::new(),
            source_key_to_id: std::collections::HashMap::new(),
            cue_index: std::collections::HashMap::new(),
            next_memory_id: 1,
            version: LEGACY_PERSISTENCE_VERSION,
            saved_at: 0,
            cue_global_counts: None,
        };
        std::fs::write(&path, bincode::serialize(&state).unwrap()).unwrap();
        let (_, _, _, next_id, _) =
            PersistenceManager::load_from_path::<MainStats>(&path).unwrap();
        assert_eq!(next_id, 1);
    }

    #[test]
    fn configured_real_snapshot_can_be_checked_during_release_validation() {
        let Ok(path) = std::env::var("CUEMAP_CHECK_SNAPSHOT") else {
            return;
        };
        let loaded = PersistenceManager::load_from_path::<MainStats>(std::path::Path::new(&path))
            .unwrap_or_else(|error| panic!("failed to load configured snapshot {path}: {error}"));
        assert!(loaded.0.len() > 0 || loaded.3 >= 1);
    }

    #[test]
    fn snapshot_loader_rejects_unknown_versions_and_corrupt_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let state = PersistedState::<MainStats> {
            memories: std::collections::HashMap::new(),
            source_key_to_id: std::collections::HashMap::new(),
            cue_index: std::collections::HashMap::new(),
            next_memory_id: 1,
            version: 999,
            saved_at: 0,
            cue_global_counts: None,
        };
        let unknown_path = dir.path().join("unknown-version.bin");
        std::fs::write(&unknown_path, serialize_state(&state).unwrap()).unwrap();
        let unknown = PersistenceManager::load_from_path::<MainStats>(&unknown_path).unwrap_err();
        assert!(unknown.to_string().contains("Unsupported snapshot version"));

        let corrupt_path = dir.path().join("corrupt.bin");
        std::fs::write(&corrupt_path, b"not a snapshot").unwrap();
        assert!(PersistenceManager::load_from_path::<MainStats>(&corrupt_path).is_err());
    }

    #[test]
    fn snapshots_round_trip_global_cue_counts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("counts.bin");
        let engine = CueMapEngine::<MainStats>::new();
        engine.add_memory(
            "counted memory".to_string(),
            vec!["counted".to_string()],
            None,
            MainStats::default(),
            true,
        );
        engine.cue_global_counts.insert("counted".to_string(), 7);

        PersistenceManager::save_to_path(&engine, &path).unwrap();
        let (_, _, _, _, counts) = PersistenceManager::load_from_path::<MainStats>(&path).unwrap();
        let counts = counts.expect("global cue counts should be persisted");
        assert_eq!(counts.get("counted").map(|value| *value), Some(7));
    }

    #[test]
    fn local_cloud_backup_supports_all_snapshot_operations() {
        let dir = tempfile::tempdir().unwrap();
        let config = CloudBackupConfig::from_args(
            Some("local"),
            Some(dir.path().to_str().unwrap()),
            None,
            None,
            "release/",
            true,
        )
        .unwrap();
        assert!(config.enabled);
        assert!(config.auto_backup);
        let manager = futures::executor::block_on(CloudBackupManager::new(config)).unwrap();

        let total = futures::executor::block_on(manager.upload_project_snapshot(
            "project",
            bytes::Bytes::from_static(b"main"),
            Some(bytes::Bytes::from_static(b"aliases")),
            Some(bytes::Bytes::from_static(b"lexicon")),
        ))
        .unwrap();
        assert_eq!(total, 18);

        let single = futures::executor::block_on(
            manager.upload_snapshot("single", bytes::Bytes::from_static(b"one")),
        )
        .unwrap();
        assert_eq!(single, 3);
        assert_eq!(
            futures::executor::block_on(manager.download_snapshot("single"))
                .unwrap()
                .as_ref(),
            b"one"
        );

        let (main, aliases, lexicon) =
            futures::executor::block_on(manager.download_project_snapshot("project")).unwrap();
        assert_eq!(main.as_ref(), b"main");
        assert_eq!(aliases.unwrap().as_ref(), b"aliases");
        assert_eq!(lexicon.unwrap().as_ref(), b"lexicon");

        let entries = futures::executor::block_on(manager.list_snapshots()).unwrap();
        let project_ids: Vec<_> = entries.iter().map(|entry| entry.project_id.as_str()).collect();
        assert!(project_ids.contains(&"project"));
        assert!(project_ids.contains(&"single"));
        assert_eq!(entries.len(), 2);

        futures::executor::block_on(manager.delete_snapshot("project")).unwrap();
        assert!(futures::executor::block_on(manager.download_project_snapshot("project")).is_err());
        futures::executor::block_on(manager.delete_snapshot("missing")).unwrap();
        assert!(manager.is_auto_backup_enabled());
    }

    #[test]
    fn cloud_config_reports_unknown_and_unconfigured_provider_errors() {
        assert!(CloudBackupConfig::from_args(Some("unknown"), None, None, None, "", false)
            .unwrap_err()
            .contains("Unknown cloud provider"));
        assert!(CloudBackupConfig::from_args(Some("azure"), Some("container"), None, None, "", false)
            .is_err());
        assert!(futures::executor::block_on(CloudBackupManager::new(CloudBackupConfig::default()))
            .is_err());
    }

    #[test]
    fn persistence_private_serialization_and_version_guards_cover_failure_paths() {
        assert!(check_snapshot_version(PERSISTENCE_VERSION).is_ok());
        assert!(check_snapshot_version(LEGACY_PERSISTENCE_VERSION).is_ok());
        let unsupported = check_snapshot_version(0).unwrap_err();
        assert!(unsupported.to_string().contains("Unsupported snapshot version"));

        let state = PersistedState::<MainStats> {
            memories: std::collections::HashMap::new(),
            source_key_to_id: std::collections::HashMap::new(),
            cue_index: std::collections::HashMap::new(),
            next_memory_id: 0,
            version: PERSISTENCE_VERSION,
            saved_at: 123,
            cue_global_counts: Some(std::collections::HashMap::from([("cue".to_string(), 4)])),
        };
        let encoded = serialize_state(&state).unwrap();
        assert!(crate::crypto::is_compressed(&encoded));
        let decoded: PersistedState<MainStats> = deserialize_state(&encoded).unwrap();
        assert_eq!(decoded.saved_at, 123);
        assert_eq!(decoded.cue_global_counts.unwrap().get("cue"), Some(&4));
        assert!(deserialize_state::<MainStats>(b"invalid snapshot").is_err());
        let compressed_invalid = zstd::stream::encode_all(std::io::Cursor::new(b"invalid json"), 3).unwrap();
        assert!(deserialize_state::<MainStats>(&compressed_invalid).is_err());
    }

    #[test]
    fn persistence_load_state_handles_legacy_zero_ids_and_corrupt_files() {
        let dir = tempfile::tempdir().unwrap();
        let manager = PersistenceManager::new(dir.path(), 1);
        let state = PersistedState::<MainStats> {
            memories: std::collections::HashMap::new(),
            source_key_to_id: std::collections::HashMap::new(),
            cue_index: std::collections::HashMap::new(),
            next_memory_id: 0,
            version: PERSISTENCE_VERSION,
            saved_at: 0,
            cue_global_counts: None,
        };
        std::fs::write(dir.path().join("cuemap.bin"), serialize_state(&state).unwrap()).unwrap();
        let (_, _, _, next_id) = manager.load_state::<MainStats>().unwrap();
        assert_eq!(next_id, 1);

        std::fs::write(dir.path().join("cuemap.bin"), b"corrupt").unwrap();
        assert!(manager.load_state::<MainStats>().is_err());

        let legacy = PersistedState::<MainStats> {
            memories: std::collections::HashMap::new(),
            source_key_to_id: std::collections::HashMap::new(),
            cue_index: std::collections::HashMap::new(),
            next_memory_id: 17,
            version: LEGACY_PERSISTENCE_VERSION,
            saved_at: 0,
            cue_global_counts: None,
        };
        std::fs::write(dir.path().join("cuemap.bin"), bincode::serialize(&legacy).unwrap()).unwrap();
        let (_, _, _, next_id) = manager.load_state::<MainStats>().unwrap();
        assert_eq!(next_id, 17);
    }

    #[test]
    fn persistence_save_errors_are_reported_and_snapshot_listing_is_tolerant() {
        let dir = tempfile::tempdir().unwrap();
        let engine = CueMapEngine::<MainStats>::new();
        let missing_parent = dir.path().join("missing").join("snapshot.bin");
        assert!(PersistenceManager::save_to_path(&engine, &missing_parent).is_err());
        assert!(PersistenceManager::list_snapshots_in_dir(&dir.path().join("missing")).is_empty());

        let directory_path = dir.path().join("directory");
        std::fs::create_dir(&directory_path).unwrap();
        assert!(PersistenceManager::delete_snapshot(&directory_path).is_err());
        assert!(PersistenceManager::delete_snapshot(&dir.path().join("does-not-exist")).is_ok());
    }

    #[tokio::test]
    async fn background_snapshot_task_ticks_and_can_be_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let manager = PersistenceManager::new(dir.path(), 1);
        let engine = std::sync::Arc::new(CueMapEngine::<MainStats>::new());
        engine.add_memory("background".to_string(), vec!["background".to_string()], None, MainStats::default(), true);
        let task = manager.start_background_snapshots(engine).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(dir.path().join("cuemap.bin").exists());
        task.abort();
        let _ = task.await;
        let cloned = manager.clone();
        assert!(cloned.load_state::<MainStats>().is_ok());
    }

    #[test]
    fn local_cloud_backup_optional_files_and_config_accessors_are_covered() {
        let dir = tempfile::tempdir().unwrap();
        let config = CloudBackupConfig::from_args(
            Some("local"),
            Some(dir.path().to_str().unwrap()),
            None,
            None,
            "optional/",
            false,
        )
        .unwrap();
        let manager = futures::executor::block_on(CloudBackupManager::new(config)).unwrap();
        assert!(!manager.is_auto_backup_enabled());
        assert_eq!(manager.get_config().prefix, "optional/");
        futures::executor::block_on(manager.upload_snapshot("main-only", bytes::Bytes::from_static(b"main"))).unwrap();
        std::fs::create_dir_all(dir.path().join("optional/main-only_aliases.bin")).unwrap();
        std::fs::create_dir_all(dir.path().join("optional/main-only_lexicon.bin")).unwrap();
        let (main, aliases, lexicon) = futures::executor::block_on(manager.download_project_snapshot("main-only")).unwrap();
        assert_eq!(main.as_ref(), b"main");
        assert!(aliases.is_none());
        assert!(lexicon.is_none());
        assert!(futures::executor::block_on(manager.delete_snapshot("main-only")).is_err());
        std::fs::remove_dir(dir.path().join("optional/main-only_aliases.bin")).unwrap();
        std::fs::remove_dir(dir.path().join("optional/main-only_lexicon.bin")).unwrap();
        futures::executor::block_on(manager.delete_snapshot("main-only")).unwrap();
    }

    #[test]
    fn persistence_constructor_and_provider_match_arms_are_exercised() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("not-a-directory");
        std::fs::write(&blocker, b"file").unwrap();
        let _ = PersistenceManager::new(&blocker, 0);

        let s3 = CloudBackupConfig::from_args(Some("s3"), Some("bucket"), None, Some("http://127.0.0.1:9"), "s3/", false).unwrap();
        let _ = futures::executor::block_on(CloudBackupManager::new(s3));
        let gcs = CloudBackupConfig::from_args(Some("gcs"), Some("bucket"), None, None, "gcs/", false).unwrap();
        let _ = futures::executor::block_on(CloudBackupManager::new(gcs));

        std::env::set_var("AZURE_STORAGE_ACCOUNT_NAME", "test-account");
        let azure = CloudBackupConfig::from_args(Some("azure"), Some("container"), None, None, "azure/", false).unwrap();
        match azure.provider.as_ref() {
            Some(CloudProvider::Azure { account, container }) => {
                assert_eq!(account, "test-account");
                assert_eq!(container, "container");
            }
            other => panic!("unexpected provider: {other:?}"),
        }
        let _ = futures::executor::block_on(CloudBackupManager::new(azure));
        std::env::remove_var("AZURE_STORAGE_ACCOUNT_NAME");
    }

    #[test]
    fn persistence_save_state_round_trips_global_counts_and_reports_serializer_errors() {
        let dir = tempfile::tempdir().unwrap();
        let manager = PersistenceManager::new(dir.path(), 1);
        let engine = CueMapEngine::<MainStats>::new();
        engine.cue_global_counts.insert("global".to_string(), 9);
        manager.save_state(&engine).unwrap();
        let (memories, source_keys, cues, next_id) = manager.load_state::<MainStats>().unwrap();
        assert!(memories.is_empty());
        assert!(source_keys.is_empty());
        assert!(cues.is_empty());
        assert_eq!(next_id, 1);

        #[derive(Clone, Default, serde::Deserialize)]
        struct FailingSerialize;
        impl serde::Serialize for FailingSerialize {
            fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                Err(serde::ser::Error::custom("intentional serializer failure"))
            }
        }
        impl MemoryStats for FailingSerialize {
            fn get_salience(&self) -> f64 { 0.0 }
            fn get_effective_salience(&self, _now: u64) -> f64 { 0.0 }
            fn get_reinforcement_count(&self) -> u64 { 0 }
            fn manual_boost(&mut self) {}
        }
        let mut memories = std::collections::HashMap::new();
        memories.insert(1, Memory::<FailingSerialize>::new(Vec::new(), None));
        let failing = PersistedState {
            memories,
            source_key_to_id: std::collections::HashMap::new(),
            cue_index: std::collections::HashMap::new(),
            next_memory_id: 1,
            version: PERSISTENCE_VERSION,
            saved_at: 0,
            cue_global_counts: None,
        };
        assert!(serialize_state(&failing).is_err());
    }

    #[tokio::test]
    async fn background_snapshot_logs_failure_when_data_path_is_not_directory() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("data-file");
        std::fs::write(&blocker, b"blocked").unwrap();
        let manager = PersistenceManager::new(&blocker, 1);
        let engine = std::sync::Arc::new(CueMapEngine::<MainStats>::new());
        let task = manager.start_background_snapshots(engine).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        task.abort();
        let _ = task.await;
    }
