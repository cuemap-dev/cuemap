use axum::Router;
use clap::Parser;
use cuemap::auth::AuthConfig;
use cuemap::config;
use cuemap::*;
use rand::{thread_rng, RngCore};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn, Level};
use tracing_subscriber::{self, fmt, prelude::*, Registry};

#[derive(Parser, Debug)]
#[command(name = "cuemap")]
#[command(about = "CueMap CLI - Unified tool for storage, ingestion, and recall")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn mock_http_server(responses: Vec<(u16, String)>) -> String {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            for (status, body) in responses {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut request = [0_u8; 8192];
                let _ = stream.read(&mut request).await;
                let reason = match status {
                    200 => "OK",
                    404 => "Not Found",
                    500 => "Internal Server Error",
                    _ => "Response",
                };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{address}")
    }

    fn shell_command(command: &str) -> (PathBuf, Vec<String>) {
        #[cfg(windows)]
        {
            let shell = std::env::var_os("COMSPEC")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("cmd.exe"));
            return (shell, vec!["/C".to_string(), command.to_string()]);
        }

        #[cfg(not(windows))]
        {
            (
                PathBuf::from("/bin/sh"),
                vec!["-c".to_string(), command.to_string()],
            )
        }
    }

    fn long_running_command() -> std::process::Command {
        #[cfg(windows)]
        {
            let shell = std::env::var_os("COMSPEC")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("cmd.exe"));
            let mut command = std::process::Command::new(shell);
            command.args(["/C", "ping 127.0.0.1 -n 31 > NUL"]);
            command
        }

        #[cfg(not(windows))]
        {
            let mut command = std::process::Command::new("sleep");
            command.arg("30");
            command
        }
    }

    fn add_args(url: String) -> AddArgs {
        AddArgs {
            content: "cli memory".to_string(),
            project: Some("cli-project".to_string()),
            metadata: Some(r#"{"source":"cli-test"}"#.to_string()),
            cues: vec!["cli".to_string()],
            disable_temporal_chunking: false,
            async_ingest: false,
            url,
        }
    }

    fn recall_args(url: String) -> RecallArgs {
        RecallArgs {
            query: "cli query".to_string(),
            project: Some("cli-project".to_string()),
            limit: 5,
            cues: vec!["cli".to_string()],
            semantic_mode: "lexical".to_string(),
            depth: 1,
            token_budget: 128,
            port: 8735,
            no_auto_reinforce: false,
            min_intersection: None,
            query_time: None,
            disable_salience_bias: false,
            parent_fusion: "off".to_string(),
            parent_fusion_limit: 80,
            parent_fusion_min_chunks: 2,
            ordered_reconstruction: "off".to_string(),
            ordered_reconstruction_limit: 80,
            ordered_session_scan_limit: 4096,
            ordered_max_sessions: 3,
            evidence_coverage: "off".to_string(),
            evidence_coverage_limit: 100,
            evidence_coverage_session_scan_limit: 4096,
            evidence_coverage_max_sessions: 3,
            expansion_depth: 1,
            enable_alias_expansion: false,
            disable_cuebridge_artifacts: false,
            cuebridge_gap_limit: 6,
            grounded: false,
            explain: true,
            trace_timing: true,
            url,
            web: false,
            target_url: None,
            persist: false,
        }
    }

    #[test]
    fn cli_parses_start_overrides_and_nested_project_commands() {
        let cli = Cli::try_parse_from([
            "cuemap", "start", "--port", "9090", "--data-dir", "/tmp/cuemap-data",
            "--profile", "benchmark", "--disable-bg-jobs", "--disable-snapshots", "--disk-content",
        ]).unwrap();
        match cli.command {
            Commands::Start(args) => {
                assert_eq!(args.port, Some(9090));
                assert_eq!(args.data_dir.as_deref(), Some("/tmp/cuemap-data"));
                assert_eq!(args.profile.as_deref(), Some("benchmark"));
                assert!(args.disable_bg_jobs);
                assert!(args.disable_snapshots);
                assert!(args.disk_content);
            }
            _ => panic!("expected start command"),
        }

        let cli = Cli::try_parse_from([
            "cuemap", "projects", "set-watch-dir", "project", "/tmp/repo", "--url", "http://example.test",
        ]).unwrap();
        assert!(matches!(cli.command, Commands::Projects(ProjectArgs { cmd: ProjectCmd::SetWatchDir { .. } })));
    }

    #[test]
    fn cli_rejects_invalid_ports_and_unknown_commands() {
        assert!(Cli::try_parse_from(["cuemap", "start", "--port", "70000"]).is_err());
        assert!(Cli::try_parse_from(["cuemap", "unknown"]).is_err());
    }

    #[test]
    fn startup_overrides_apply_all_cli_options() {
        let cli = Cli::try_parse_from([
            "cuemap",
            "start",
            "--port",
            "9091",
            "--data-dir",
            "/tmp/cuemap-data",
            "--assets-dir",
            "/tmp/cuemap-assets",
            "--snapshot-interval",
            "17",
            "--agent-dir",
            "/tmp/cuemap-watch",
            "--agent-throttle",
            "250",
            "--disable-bg-jobs",
            "--disable-snapshots",
            "--disk-content",
            "--cloud-backup",
            "local",
            "--cloud-bucket",
            "/tmp/cuemap-cloud",
            "--cloud-region",
            "eu-west-1",
            "--cloud-endpoint",
            "http://localhost:9000",
            "--cloud-prefix",
            "release/",
            "--cloud-auto-backup",
        ])
        .unwrap();

        let args = match cli.command {
            Commands::Start(args) => args,
            _ => panic!("expected start command"),
        };
        let config = apply_start_overrides(config::ServerConfig::default(), &args);

        assert_eq!(config.server.port, 9091);
        assert_eq!(config.server.data_dir, "/tmp/cuemap-data");
        assert_eq!(config.server.assets_dir.as_deref(), Some("/tmp/cuemap-assets"));
        assert_eq!(config.persistence.snapshot_interval_seconds, 17);
        assert_eq!(config.agent.watch_dir.as_deref(), Some("/tmp/cuemap-watch"));
        assert!(config.agent.enabled);
        assert_eq!(config.agent.throttle_ms, 250);
        assert!(!config.jobs.background_processing);
        assert!(!config.persistence.enabled);
        assert!(config.server.store_content_on_disk);
        assert_eq!(config.persistence.cloud.provider, "local");
        assert_eq!(config.persistence.cloud.bucket, "/tmp/cuemap-cloud");
        assert_eq!(config.persistence.cloud.region, "eu-west-1");
        assert_eq!(
            config.persistence.cloud.endpoint.as_deref(),
            Some("http://localhost:9000")
        );
        assert_eq!(config.persistence.cloud.prefix, "release/");
        assert!(config.persistence.cloud.auto_backup);
    }

    #[test]
    fn startup_config_loading_applies_profile_and_cli_overrides() {
        let root = tempfile::tempdir().unwrap();
        let config_path = root.path().join("missing.toml");
        let cli = Cli::try_parse_from([
            "cuemap",
            "start",
            "--config",
            config_path.to_str().unwrap(),
            "--profile",
            "benchmark",
            "--port",
            "9191",
        ])
        .unwrap();
        let args = match cli.command {
            Commands::Start(args) => args,
            _ => panic!("expected start command"),
        };

        let config = load_start_config(&args).unwrap();
        assert_eq!(config.server.port, 9191);
        assert!(!config.persistence.enabled);
        assert!(!config.jobs.background_processing);
    }

    #[test]
    fn snapshot_selection_handles_static_auxiliary_main_and_legacy_layouts() {
        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("data");
        let configured = data_dir.join("snapshots");
        let legacy = root.path().join("snapshots");
        std::fs::create_dir_all(&configured).unwrap();

        assert_eq!(
            select_snapshots_dir(data_dir.to_str().unwrap(), Some("/tmp/static")),
            "/tmp/static"
        );
        assert!(!has_main_snapshot(&configured));

        std::fs::write(configured.join("project_aliases.bin"), b"aliases").unwrap();
        std::fs::write(configured.join("project_lexicon.bin"), b"lexicon").unwrap();
        assert!(!has_main_snapshot(&configured));
        assert_eq!(
            select_snapshots_dir(data_dir.to_str().unwrap(), None),
            configured.to_string_lossy()
        );

        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("project.bin"), b"snapshot").unwrap();
        assert!(has_main_snapshot(&legacy));
        assert_eq!(
            select_snapshots_dir(data_dir.to_str().unwrap(), None),
            data_dir.join("..").join("snapshots").to_string_lossy()
        );

        std::fs::write(configured.join("project.bin"), b"new snapshot").unwrap();
        assert_eq!(
            select_snapshots_dir(data_dir.to_str().unwrap(), None),
            configured.to_string_lossy()
        );
        assert!(!has_main_snapshot(&root.path().join("missing")));
    }

    #[test]
    fn master_key_resolution_obeys_precedence_and_validation() {
        let mut security = config::SecurityConfig::default();
        let config_key = "22".repeat(32);
        security.master_key = Some(config_key);

        let env_key = "11".repeat(32);
        let resolved = resolve_master_key(&security, Some(&env_key), None, None).unwrap();
        assert_eq!(resolved.as_bytes(), &[0x11; 32]);

        // An explicitly supplied but malformed environment key must not silently
        // fall back to a lower-precedence config value.
        assert!(resolve_master_key(&security, Some("not-hex"), None, None).is_none());

        let password_key = resolve_master_key(
            &config::SecurityConfig::default(),
            None,
            Some("correct horse battery staple"),
            Some(b"test-salt"),
        )
        .unwrap();
        assert_eq!(password_key.as_bytes().len(), 32);

        let config_key = resolve_master_key(&security, None, None, None).unwrap();
        assert_eq!(config_key.as_bytes(), &[0x22; 32]);

        security.master_key = Some("short".to_string());
        assert!(resolve_master_key(&security, None, None, None).is_none());
        assert!(resolve_master_key(&config::SecurityConfig::default(), None, None, None).is_none());
    }

    #[test]
    fn context_signer_resolution_handles_ed25519_legacy_and_invalid_keys() {
        let mut security = config::SecurityConfig::default();
        assert!(resolve_context_signer(&security).is_none());

        security.secret_key = Some("legacy-secret".to_string());
        assert!(resolve_context_signer(&security).is_some());

        security.signing_private_key = Some("00".repeat(32));
        assert!(resolve_context_signer(&security).is_some());

        security.signing_private_key = Some("not-hex".to_string());
        assert!(resolve_context_signer(&security).is_none());
    }

    #[test]
    fn kdf_salt_environment_override_is_used_for_startup() {
        let previous = std::env::var("CUEMAP_KDF_SALT").ok();
        std::env::set_var("CUEMAP_KDF_SALT", "release-test-salt");
        assert_eq!(get_or_create_salt(), b"release-test-salt");

        if let Some(value) = previous {
            std::env::set_var("CUEMAP_KDF_SALT", value);
        } else {
            std::env::remove_var("CUEMAP_KDF_SALT");
        }
    }

    #[test]
    fn local_kdf_salt_loading_handles_existing_short_and_unwritable_files() {
        let root = tempfile::tempdir().unwrap();
        let existing = vec![7_u8; 32];
        std::fs::write(root.path().join("salt"), &existing).unwrap();
        assert_eq!(load_or_create_salt(root.path(), None), existing);

        std::fs::write(root.path().join("salt"), b"short").unwrap();
        let regenerated = load_or_create_salt(root.path(), None);
        assert_eq!(regenerated.len(), 32);
        assert_eq!(std::fs::read(root.path().join("salt")).unwrap(), regenerated);

        let missing_root = tempfile::tempdir().unwrap();
        let generated = load_or_create_salt(missing_root.path(), None);
        assert_eq!(generated.len(), 32);

        let file_root = root.path().join("not-a-directory");
        std::fs::write(&file_root, b"file").unwrap();
        assert_eq!(load_or_create_salt(&file_root, None).len(), 32);
        assert_eq!(load_or_create_salt(root.path(), Some("override")), b"override");
    }

    #[tokio::test]
    async fn detached_readiness_waiter_handles_offsets_success_timeout_and_missing_logs() {
        let root = tempfile::tempdir().unwrap();
        let log_path = root.path().join("server.log");
        std::fs::write(&log_path, "old line\nready: Unstable sorting for speed\n").unwrap();
        let start_pos = "old line\n".len() as u64;
        assert!(wait_for_readiness(
            &log_path,
            start_pos,
            "Unstable sorting for speed",
            Duration::from_millis(50),
        )
        .await
        .unwrap());

        let empty_path = root.path().join("empty.log");
        std::fs::write(&empty_path, "").unwrap();
        assert!(!wait_for_readiness(
            &empty_path,
            0,
            "never appears",
            Duration::from_millis(1),
        )
        .await
        .unwrap());
        assert!(wait_for_readiness(
            &root.path().join("missing.log"),
            0,
            "ready",
            Duration::from_millis(1),
        )
        .await
        .is_err());

        let spawned_log = root.path().join("spawned.log");
        let (shell, shell_args) = shell_command(if cfg!(windows) {
            "echo Unstable sorting for speed"
        } else {
            "printf 'Unstable sorting for speed\\n'"
        });
        assert!(spawn_detached_process(
            &shell,
            &shell_args,
            &spawned_log,
            "Unstable sorting for speed",
            Duration::from_secs(1),
        )
        .await
        .unwrap());

        let (shell, timeout_args) = shell_command(if cfg!(windows) {
            "exit 0"
        } else {
            "true"
        });
        assert!(!spawn_detached_process(
            &shell,
            &timeout_args,
            &root.path().join("spawn-timeout.log"),
            "never appears",
            Duration::from_millis(1),
        )
        .await
        .unwrap());
        let missing_executable = root.path().join("definitely-missing-cuemap-child");
        assert!(spawn_detached_process(
            &missing_executable,
            &[],
            &root.path().join("spawn-error.log"),
            "ready",
            Duration::from_millis(1),
        )
        .await
        .is_err());
    }

    #[tokio::test]
    async fn static_server_startup_builds_and_binds_the_cli_router() {
        let root = tempfile::tempdir().unwrap();
        let snapshots = root.path().join("snapshots");
        std::fs::create_dir_all(&snapshots).unwrap();
        let pid_path = root.path().join("server.pid");
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();

        let mut server_config = config::ServerConfig::default();
        server_config.server.port = port;
        server_config.server.data_dir = root.path().join("data").to_string_lossy().to_string();
        server_config.persistence.enabled = false;
        server_config.jobs.background_processing = false;
        server_config.semantic.enabled = false;
        server_config.semantic.encoder_enabled = false;
        server_config.semantic.profile = cuemap::semantic::SemanticProfile::Off;

        let task = tokio::spawn(run_server_with_pid_path(
            server_config,
            Some(snapshots.to_string_lossy().to_string()),
            true,
            pid_path.clone(),
        ));

        let ready = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .is_ok();
        assert!(ready, "static server did not bind its configured port");
        assert_eq!(std::fs::read_to_string(&pid_path).unwrap(), std::process::id().to_string());

        task.abort();
        let _ = task.await;

        let live_port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let live_pid_path = root.path().join("live-server.pid");
        let mut live_config = config::ServerConfig::default();
        live_config.server.port = live_port;
        live_config.server.data_dir = root.path().join("live-data").to_string_lossy().to_string();
        live_config.persistence.enabled = false;
        live_config.jobs.background_processing = false;
        live_config.semantic.enabled = false;
        live_config.semantic.encoder_enabled = false;
        live_config.semantic.profile = cuemap::semantic::SemanticProfile::Off;
        live_config.persistence.cloud.provider = "s3".to_string();

        let live_task = tokio::spawn(run_server_with_pid_path(
            live_config,
            None,
            true,
            live_pid_path,
        ));
        let live_ready = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if tokio::net::TcpStream::connect(("127.0.0.1", live_port)).await.is_ok() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .is_ok();
        assert!(live_ready, "live server did not bind its configured port");
        live_task.abort();
        let _ = live_task.await;
    }

    #[tokio::test]
    async fn stop_handler_covers_missing_success_and_failed_pid_paths() {
        let root = tempfile::tempdir().unwrap();
        let missing_path = root.path().join("missing.pid");
        handle_stop_at(missing_path).await;

        let mut child = long_running_command().spawn().unwrap();
        let pid_path = root.path().join("running.pid");
        std::fs::write(&pid_path, child.id().to_string()).unwrap();
        handle_stop_at(pid_path.clone()).await;
        assert!(!pid_path.exists());
        let _ = child.wait();

        let failed_path = root.path().join("failed.pid");
        // Keep the synthetic PID positive when converted to Unix pid_t.
        // u32::MAX becomes -1 on Linux, where kill(-1, SIGTERM) signals every
        // process the current user is permitted to terminate.
        std::fs::write(&failed_path, (i32::MAX as u32).to_string()).unwrap();
        handle_stop_at(failed_path.clone()).await;
        assert!(failed_path.exists());

        #[cfg(unix)]
        {
            let unsafe_path = root.path().join("unsafe.pid");
            std::fs::write(&unsafe_path, u32::MAX.to_string()).unwrap();
            handle_stop_at(unsafe_path.clone()).await;
            assert!(unsafe_path.exists());
        }
    }

    #[tokio::test]
    async fn cli_http_handlers_cover_success_and_failure_paths() {
        let add_url = mock_http_server(vec![(200, r#"{"id":42}"#.to_string())]).await;
        handle_add(add_args(add_url)).await;
        let add_error_url = mock_http_server(vec![(500, r#"{"error":"rejected"}"#.to_string())]).await;
        handle_add(add_args(add_error_url)).await;

        let ingest_url = mock_http_server(vec![(200, r#"{"status":"ingested"}"#.to_string())]).await;
        handle_ingest(IngestArgs {
            type_: IngestType::Content {
                content: "content from cli".to_string(),
                project: Some("cli-project".to_string()),
                filename: "note.md".to_string(),
                source_key: Some("cli-note".to_string()),
                metadata: Some(r#"{"source":"test"}"#.to_string()),
                structural_cues: vec!["cli".to_string()],
                segmenter: "sentence_window".to_string(),
                segment_window_size: Some(2),
                segment_overlap: Some(1),
                segment_min_chunk_chars: Some(16),
                segment_max_chunk_chars: Some(128),
                url: ingest_url,
            },
        })
        .await;

        let file_dir = tempfile::tempdir().unwrap();
        let file_path = file_dir.path().join("cli-note.md");
        std::fs::write(&file_path, "file content from cli").unwrap();
        let file_url = mock_http_server(vec![(200, r#"{"status":"ingested"}"#.to_string())]).await;
        handle_ingest(IngestArgs {
            type_: IngestType::File {
                path: file_path.to_string_lossy().to_string(),
                project: Some("cli-project".to_string()),
                url: file_url,
            },
        })
        .await;

        let file_error_url = mock_http_server(vec![(500, r#"{"error":"file rejected"}"#.to_string())]).await;
        handle_ingest(IngestArgs {
            type_: IngestType::File {
                path: file_path.to_string_lossy().to_string(),
                project: Some("cli-project".to_string()),
                url: file_error_url,
            },
        })
        .await;
        handle_ingest(IngestArgs {
            type_: IngestType::File {
                path: file_dir.path().join("missing.md").to_string_lossy().to_string(),
                project: Some("cli-project".to_string()),
                url: "http://127.0.0.1:1".to_string(),
            },
        })
        .await;

        let url_ingest_server =
            mock_http_server(vec![(200, r#"{"status":"started"}"#.to_string())]).await;
        handle_ingest(IngestArgs {
            type_: IngestType::Url {
                url: "https://example.test/docs".to_string(),
                project: Some("cli-project".to_string()),
                depth: 2,
                same_domain_only: true,
                server_url: url_ingest_server,
            },
        })
        .await;

        let url_ingest_error_server =
            mock_http_server(vec![(500, r#"{"error":"crawl rejected"}"#.to_string())]).await;
        handle_ingest(IngestArgs {
            type_: IngestType::Url {
                url: "https://example.test/docs".to_string(),
                project: Some("cli-project".to_string()),
                depth: 0,
                same_domain_only: false,
                server_url: url_ingest_error_server,
            },
        })
        .await;

        let recall_url = mock_http_server(vec![
            (
                200,
                r#"{"results":[{"score":1.25,"memory_id":42,"content":"cli result"}],"timing":{"total_ms":1.0}}"#.to_string(),
            ),
        ])
        .await;
        handle_recall(recall_args(recall_url)).await;

        let grounded_body = serde_json::json!({
            "verified_context": "grounded context",
            "proof": {
                "trace_id": "trace",
                "query_text": "cli query",
                "normalized_query": [],
                "expanded_cues": [],
                "token_budget": 128,
                "selected": [],
                "excluded_top": []
            },
            "engine_latency_ms": 1.0,
            "signature_alg": "none",
            "signature": "",
            "public_key": null
        })
        .to_string();
        let grounded_url = mock_http_server(vec![(200, grounded_body)]).await;
        let mut grounded = recall_args(grounded_url);
        grounded.grounded = true;
        handle_recall(grounded).await;

        let web_url = mock_http_server(vec![
            (
                200,
                r#"{"results":[{"score":0.8,"intersection":2,"content":"web result"}],"urls":["https://example.test"]}"#.to_string(),
            ),
        ])
        .await;
        let mut web = recall_args(web_url);
        web.web = true;
        web.target_url = Some("https://example.test/page".to_string());
        web.persist = true;
        handle_recall(web).await;

        let recall_error_url = mock_http_server(vec![(500, r#"{"error":"down"}"#.to_string())]).await;
        handle_recall(recall_args(recall_error_url)).await;
    }

    #[tokio::test]
    async fn cli_status_project_alias_and_memory_handlers_use_http_api() {
        let status_url = mock_http_server(vec![
            (200, r#"{"total_memories":2}"#.to_string()),
            (200, "cuemap_requests_total 2\n".to_string()),
        ])
        .await;
        handle_status(StatusArgs {
            server: false,
            jobs: false,
            project: Some("cli-project".to_string()),
            json: true,
            url: status_url,
        })
        .await;

        let jobs_url = mock_http_server(vec![(200, r#"{"phase":"done"}"#.to_string())]).await;
        handle_status(StatusArgs {
            server: false,
            jobs: true,
            project: Some("cli-project".to_string()),
            json: false,
            url: jobs_url,
        })
        .await;

        let jobs_json_url = mock_http_server(vec![(200, r#"{"phase":"queued"}"#.to_string())]).await;
        handle_status(StatusArgs {
            server: false,
            jobs: true,
            project: Some("cli-project".to_string()),
            json: true,
            url: jobs_json_url,
        })
        .await;

        let list_url = mock_http_server(vec![
            (200, r#"[{"project_id":"cli-project","total_memories":2}]"#.to_string()),
        ])
        .await;
        handle_projects(ProjectArgs {
            cmd: ProjectCmd::List { url: list_url },
        })
        .await;

        let create_url = mock_http_server(vec![(200, r#"{"project_id":"new-project"}"#.to_string())]).await;
        handle_projects(ProjectArgs {
            cmd: ProjectCmd::Create {
                name: "new-project".to_string(),
                url: create_url,
            },
        })
        .await;

        let watch_url = mock_http_server(vec![(200, r#"{"status":"updated"}"#.to_string())]).await;
        handle_projects(ProjectArgs {
            cmd: ProjectCmd::SetWatchDir {
                project: "cli-project".to_string(),
                path: "/tmp".to_string(),
                url: watch_url,
            },
        })
        .await;

        let alias_add_url = mock_http_server(vec![(200, r#"{"id":1}"#.to_string())]).await;
        handle_alias(AliasArgs {
            text: "rust".to_string(),
            project: Some("cli-project".to_string()),
            add: Some("rust-language".to_string()),
            weight: Some(0.9),
            url: alias_add_url,
        })
        .await;

        let alias_query_url = mock_http_server(vec![(200, "[]".to_string())]).await;
        handle_alias(AliasArgs {
            text: "rust".to_string(),
            project: Some("cli-project".to_string()),
            add: None,
            weight: None,
            url: alias_query_url,
        })
        .await;

        let memory_get_url = mock_http_server(vec![(200, r#"{"content":"memory","created_at":1.0,"cues":["cli"],"stats":{"reinforcement":1}}"#.to_string())]).await;
        handle_memories(MemoriesArgs {
            id: 42,
            reinforce: false,
            delete: false,
            cues: Vec::new(),
            project: Some("cli-project".to_string()),
            url: memory_get_url,
        })
        .await;

        let memory_reinforce_url = mock_http_server(vec![(200, r#"{"status":"reinforced"}"#.to_string())]).await;
        handle_memories(MemoriesArgs {
            id: 42,
            reinforce: true,
            delete: false,
            cues: vec!["cli".to_string()],
            project: Some("cli-project".to_string()),
            url: memory_reinforce_url,
        })
        .await;

        let memory_delete_url = mock_http_server(vec![(200, r#"{"status":"deleted"}"#.to_string())]).await;
        handle_memories(MemoriesArgs {
            id: 42,
            reinforce: false,
            delete: true,
            cues: Vec::new(),
            project: Some("cli-project".to_string()),
            url: memory_delete_url,
        })
        .await;

        let memory_missing_url = mock_http_server(vec![(404, r#"{"error":"missing"}"#.to_string())]).await;
        handle_memories(MemoriesArgs {
            id: 999,
            reinforce: false,
            delete: false,
            cues: Vec::new(),
            project: Some("cli-project".to_string()),
            url: memory_missing_url,
        })
        .await;

        let delete_missing_url = mock_http_server(vec![(404, "missing".to_string())]).await;
        handle_memories(MemoriesArgs {
            id: 404,
            reinforce: false,
            delete: true,
            cues: Vec::new(),
            project: Some("cli-project".to_string()),
            url: delete_missing_url,
        })
        .await;
    }

    #[tokio::test]
    async fn cli_logs_handler_covers_missing_head_tail_and_full_modes() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("server.log");
        std::fs::write(&log_path, "first\nsecond\nthird\n").unwrap();

        handle_logs(LogsArgs {
            head: Some(2),
            tail: None,
            follow: false,
            path: Some(log_path.to_string_lossy().to_string()),
        })
        .await;
        handle_logs(LogsArgs {
            head: None,
            tail: Some(2),
            follow: false,
            path: Some(log_path.to_string_lossy().to_string()),
        })
        .await;
        handle_logs(LogsArgs {
            head: None,
            tail: None,
            follow: false,
            path: Some(log_path.to_string_lossy().to_string()),
        })
        .await;

        let follow_path = log_path.clone();
        let follow_task = tokio::spawn(async move {
            handle_logs(LogsArgs {
                head: None,
                tail: None,
                follow: true,
                path: Some(follow_path.to_string_lossy().to_string()),
            })
            .await;
        });
        tokio::time::sleep(Duration::from_millis(120)).await;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&log_path)
            .unwrap()
            .write_all(b"followed\n")
            .unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        follow_task.abort();
        let _ = follow_task.await;

        handle_logs(LogsArgs {
            head: None,
            tail: None,
            follow: false,
            path: Some(dir.path().join("missing.log").to_string_lossy().to_string()),
        })
        .await;
    }

    #[tokio::test]
    async fn cli_handlers_cover_error_responses_and_alias_defaults() {
        let ingest_error_url =
            mock_http_server(vec![(500, r#"{"error":"invalid content"}"#.to_string())]).await;
        handle_ingest(IngestArgs {
            type_: IngestType::Content {
                content: "bad content".to_string(),
                project: Some("cli-project".to_string()),
                filename: "note.txt".to_string(),
                source_key: None,
                metadata: None,
                structural_cues: Vec::new(),
                segmenter: "sentence_window".to_string(),
                segment_window_size: None,
                segment_overlap: None,
                segment_min_chunk_chars: None,
                segment_max_chunk_chars: None,
                url: ingest_error_url,
            },
        })
        .await;

        let status_error_url = mock_http_server(vec![(500, "server error".to_string())]).await;
        handle_status(StatusArgs {
            server: true,
            jobs: false,
            project: None,
            json: false,
            url: status_error_url,
        })
        .await;
        let jobs_error_url = mock_http_server(vec![(500, "jobs error".to_string())]).await;
        handle_status(StatusArgs {
            server: false,
            jobs: true,
            project: None,
            json: true,
            url: jobs_error_url,
        })
        .await;

        let list_error_url = mock_http_server(vec![(500, "list error".to_string())]).await;
        handle_projects(ProjectArgs {
            cmd: ProjectCmd::List { url: list_error_url },
        })
        .await;
        let create_error_url = mock_http_server(vec![(500, "create error".to_string())]).await;
        handle_projects(ProjectArgs {
            cmd: ProjectCmd::Create {
                name: "bad-project".to_string(),
                url: create_error_url,
            },
        })
        .await;
        let watch_error_url = mock_http_server(vec![(500, "watch error".to_string())]).await;
        handle_projects(ProjectArgs {
            cmd: ProjectCmd::SetWatchDir {
                project: "cli-project".to_string(),
                path: "/missing".to_string(),
                url: watch_error_url,
            },
        })
        .await;

        let alias_add_error_url = mock_http_server(vec![(500, "alias error".to_string())]).await;
        handle_alias(AliasArgs {
            text: "rust".to_string(),
            project: Some("cli-project".to_string()),
            add: Some("rust-language".to_string()),
            weight: None,
            url: alias_add_error_url,
        })
        .await;
        let alias_query_result_url = mock_http_server(vec![
            (
                200,
                r#"[{"id":1,"from":"rust","to":"rust-language","weight":0.9}]"#.to_string(),
            ),
        ])
        .await;
        handle_alias(AliasArgs {
            text: "rust".to_string(),
            project: Some("cli-project".to_string()),
            add: None,
            weight: None,
            url: alias_query_result_url,
        })
        .await;

        let reinforce_missing_url = mock_http_server(vec![(404, "missing".to_string())]).await;
        handle_memories(MemoriesArgs {
            id: 404,
            reinforce: true,
            delete: false,
            cues: Vec::new(),
            project: Some("cli-project".to_string()),
            url: reinforce_missing_url,
        })
        .await;
        let delete_error_url = mock_http_server(vec![(500, "delete error".to_string())]).await;
        handle_memories(MemoriesArgs {
            id: 500,
            reinforce: false,
            delete: true,
            cues: Vec::new(),
            project: Some("cli-project".to_string()),
            url: delete_error_url,
        })
        .await;
    }

    #[tokio::test]
    async fn cli_lexicon_and_project_config_paths_are_covered() {
        let lexicon_url = mock_http_server(vec![
            (
                200,
                r#"{"cue":"rust","outgoing":[],"incoming":[]}"#.to_string(),
            ),
        ])
        .await;
        handle_lexicon(LexiconArgs {
            cmd: LexiconCmd::Inspect {
                cue: "rust".to_string(),
                project: Some("cli-project".to_string()),
                url: lexicon_url,
            },
        })
        .await;

        let lexicon_error_url = mock_http_server(vec![(500, "lexicon unavailable".to_string())]).await;
        handle_lexicon(LexiconArgs {
            cmd: LexiconCmd::Inspect {
                cue: "missing".to_string(),
                project: Some("cli-project".to_string()),
                url: lexicon_error_url,
            },
        })
        .await;

        let config_dir = tempfile::tempdir().unwrap();
        let config_path = config_dir.path().join("config.json");
        assert_eq!(read_default_project(&config_path), None);
        write_default_project(&config_path, "cli-project").unwrap();
        assert_eq!(read_default_project(&config_path).as_deref(), Some("cli-project"));

        std::fs::write(&config_path, "not-json").unwrap();
        assert_eq!(read_default_project(&config_path), None);
        std::fs::write(&config_path, r#"{"default_project":42}"#).unwrap();
        assert_eq!(read_default_project(&config_path), None);
        assert!(write_default_project(config_dir.path(), "bad-path").is_err());
    }

    #[tokio::test]
    async fn cli_handlers_cover_connection_failures_and_recall_modes() {
        let dead_url = "http://127.0.0.1:1".to_string();
        handle_add(add_args(dead_url.clone())).await;

        handle_ingest(IngestArgs {
            type_: IngestType::Content {
                content: "connection failure".to_string(),
                project: Some("cli-project".to_string()),
                filename: "failure.txt".to_string(),
                source_key: None,
                metadata: None,
                structural_cues: Vec::new(),
                segmenter: "sentence_window".to_string(),
                segment_window_size: None,
                segment_overlap: None,
                segment_min_chunk_chars: None,
                segment_max_chunk_chars: None,
                url: dead_url.clone(),
            },
        })
        .await;
        handle_ingest(IngestArgs {
            type_: IngestType::Url {
                url: "https://example.test".to_string(),
                project: Some("cli-project".to_string()),
                depth: 0,
                same_domain_only: true,
                server_url: dead_url.clone(),
            },
        })
        .await;

        for semantic_mode in ["semantic", "hybrid"] {
            let mut recall = recall_args(dead_url.clone());
            recall.semantic_mode = semantic_mode.to_string();
            recall.trace_timing = false;
            handle_recall(recall).await;
        }

        let mut grounded = recall_args(dead_url.clone());
        grounded.grounded = true;
        handle_recall(grounded).await;
        let grounded_error_url = mock_http_server(vec![(500, "grounded error".to_string())]).await;
        let mut grounded_error = recall_args(grounded_error_url);
        grounded_error.grounded = true;
        handle_recall(grounded_error).await;

        let web_error_url = mock_http_server(vec![(500, "web error".to_string())]).await;
        let mut web_error = recall_args(web_error_url);
        web_error.web = true;
        handle_recall(web_error).await;
        let mut web_failure = recall_args(dead_url.clone());
        web_failure.web = true;
        handle_recall(web_failure).await;

        let status_url = mock_http_server(vec![(200, r#"{"total_memories":1}"#.to_string())]).await;
        handle_status(StatusArgs {
            server: true,
            jobs: false,
            project: Some("cli-project".to_string()),
            json: false,
            url: status_url,
        })
        .await;
        handle_status(StatusArgs {
            server: true,
            jobs: false,
            project: Some("cli-project".to_string()),
            json: false,
            url: dead_url.clone(),
        })
        .await;
        handle_status(StatusArgs {
            server: false,
            jobs: true,
            project: Some("cli-project".to_string()),
            json: false,
            url: dead_url.clone(),
        })
        .await;

        handle_memories(MemoriesArgs {
            id: 1,
            reinforce: false,
            delete: true,
            cues: Vec::new(),
            project: Some("cli-project".to_string()),
            url: dead_url.clone(),
        })
        .await;
        handle_memories(MemoriesArgs {
            id: 1,
            reinforce: true,
            delete: false,
            cues: vec!["cli".to_string()],
            project: Some("cli-project".to_string()),
            url: dead_url.clone(),
        })
        .await;
        handle_memories(MemoriesArgs {
            id: 1,
            reinforce: false,
            delete: false,
            cues: Vec::new(),
            project: Some("cli-project".to_string()),
            url: dead_url.clone(),
        })
        .await;

        handle_alias(AliasArgs {
            text: "rust".to_string(),
            project: Some("cli-project".to_string()),
            add: Some("language".to_string()),
            weight: None,
            url: dead_url.clone(),
        })
        .await;
        handle_alias(AliasArgs {
            text: "rust".to_string(),
            project: Some("cli-project".to_string()),
            add: None,
            weight: None,
            url: dead_url.clone(),
        })
        .await;

        handle_projects(ProjectArgs {
            cmd: ProjectCmd::List {
                url: dead_url.clone(),
            },
        })
        .await;
        handle_projects(ProjectArgs {
            cmd: ProjectCmd::Create {
                name: "dead-project".to_string(),
                url: dead_url.clone(),
            },
        })
        .await;
        handle_projects(ProjectArgs {
            cmd: ProjectCmd::SetWatchDir {
                project: "cli-project".to_string(),
                path: "/tmp".to_string(),
                url: dead_url,
            },
        })
        .await;
    }

    #[test]
    fn cli_ingest_and_lexicon_server_url_defaults_parse() {
        let cli = Cli::try_parse_from(["cuemap", "ingest", "file", "note.md", "--project", "p"])
            .unwrap();
        match cli.command {
            Commands::Ingest(IngestArgs {
                type_: IngestType::File { url, .. },
            }) => assert_eq!(url, "http://localhost:8735"),
            _ => panic!("expected file ingest command"),
        }

        let cli = Cli::try_parse_from(["cuemap", "ingest", "url", "https://example.test"])
            .unwrap();
        match cli.command {
            Commands::Ingest(IngestArgs {
                type_: IngestType::Url { server_url, .. },
            }) => assert_eq!(server_url, "http://localhost:8735"),
            _ => panic!("expected URL ingest command"),
        }

        let cli = Cli::try_parse_from(["cuemap", "lexicon", "inspect", "rust"]).unwrap();
        match cli.command {
            Commands::Lexicon(LexiconArgs {
                cmd: LexiconCmd::Inspect { url, .. },
            }) => assert_eq!(url, "http://localhost:8735"),
            _ => panic!("expected lexicon inspect command"),
        }
    }
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Start the CueMap server
    Start(StartArgs),

    /// Add a memory via natural language
    Add(AddArgs),

    /// Ingest data from files or URLs
    Ingest(IngestArgs),

    /// Search memories (recall)
    Recall(RecallArgs),

    /// Manage lexicon entries
    Lexicon(LexiconArgs),

    /// Manage aliases
    Alias(AliasArgs),

    /// Manage individual memories (get/reinforce/delete)
    Memories(MemoriesArgs),

    /// Manage projects
    Projects(ProjectArgs),

    /// Set default project for CLI commands
    SetProject { project_id: String },
    /// Check server status and background jobs
    Status(StatusArgs),
    /// View or tail server logs
    Logs(LogsArgs),
    /// Stop the background server
    Stop(StopArgs),
}

#[derive(Parser, Debug)]
struct StopArgs {
    /// Server URL (to find the PID via local config if possible)
    #[arg(long, default_value = "http://localhost:8735")]
    url: String,
}

#[derive(Parser, Debug)]
struct LogsArgs {
    /// Show the first N lines
    #[arg(long)]
    head: Option<usize>,
    /// Show the last N lines
    #[arg(long)]
    tail: Option<usize>,
    /// Follow log output (live preview)
    #[arg(short, long)]
    follow: bool,
    /// Custom log file path
    #[arg(long)]
    path: Option<String>,
}

#[derive(Parser, Debug)]
struct StatusArgs {
    /// Show server metrics/stats (/stats)
    #[arg(long)]
    server: bool,
    /// Show background job status (/jobs/status)
    #[arg(long)]
    jobs: bool,
    /// Project ID (required for --jobs)
    #[arg(short, long)]
    project: Option<String>,
    /// Print machine-readable JSON without headings
    #[arg(long)]
    json: bool,
    /// Server URL
    #[arg(long, default_value = "http://localhost:8735")]
    url: String,
}

#[derive(Parser, Debug)]
struct StartArgs {
    /// Config file path (default: ~/.cuemap/server_config.toml)
    #[arg(long)]
    config: Option<String>,

    /// Config profile (default: "default")
    #[arg(long)]
    profile: Option<String>,

    /// Server port (overrides config)
    #[arg(short, long)]
    port: Option<u16>,

    /// Data directory for persistence (overrides config)
    #[arg(short, long)]
    data_dir: Option<String>,

    /// Assets directory (read-only models, taggers, defaults)
    #[arg(long)]
    assets_dir: Option<String>,

    /// Snapshot interval in seconds (overrides config)
    #[arg(short, long)]
    snapshot_interval: Option<u64>,

    /// Load static snapshots (read-only mode, disables persistence)
    #[arg(long)]
    load_static: Option<String>,

    /// Directory to watch for Self-Learning Agent (Legacy arg, overrides project meta)
    #[arg(long)]
    agent_dir: Option<String>,

    /// Agent throttle in milliseconds
    #[arg(long)]
    agent_throttle: Option<u64>,

    /// Disable background jobs (for benchmarking)
    #[arg(long)]
    disable_bg_jobs: bool,

    /// Disable periodic snapshots (for benchmarking)
    #[arg(long)]
    disable_snapshots: bool,

    /// Store memory contents on disk instead of RAM to reduce memory usage
    #[arg(long)]
    disk_content: bool,

    // ========== Cloud Backup Options ==========
    /// Cloud backup provider (s3, gcs, azure, local)
    #[arg(long)]
    cloud_backup: Option<String>,

    /// Cloud backup bucket/container name (or path for local)
    #[arg(long)]
    cloud_bucket: Option<String>,

    /// Cloud backup region (for S3)
    #[arg(long)]
    cloud_region: Option<String>,

    /// S3-compatible endpoint URL (for MinIO, DigitalOcean Spaces, etc.)
    #[arg(long)]
    cloud_endpoint: Option<String>,

    /// Cloud backup object key prefix
    #[arg(long)]
    cloud_prefix: Option<String>,

    /// Enable automatic cloud backup after each local save
    #[arg(long)]
    cloud_auto_backup: bool,

    /// Log file path
    #[arg(long)]
    log_file: Option<String>,

    /// Run server in the background
    #[arg(long)]
    detach: bool,

    /// Internal: Marker for the child process (do not use)
    #[arg(long, hide = true)]
    child_process: bool,
}

#[derive(Parser, Debug)]
struct AddArgs {
    /// Content to add
    content: String,
    /// Project ID (optional if set-project was used)
    #[arg(short, long)]
    project: Option<String>,
    /// Optional metadata (JSON string)
    #[arg(short, long)]
    metadata: Option<String>,
    /// Manual cues to associate
    #[arg(short, long)]
    cues: Vec<String>,
    /// Disable temporal chunking for this memory
    #[arg(long)]
    disable_temporal_chunking: bool,
    /// Process ingestion in background (return immediately)
    #[arg(long)]
    async_ingest: bool,
    /// Server URL
    #[arg(long, default_value = "http://localhost:8735")]
    url: String,
}

#[derive(Parser, Debug)]
struct IngestArgs {
    #[command(subcommand)]
    type_: IngestType,
}

#[derive(clap::Subcommand, Debug)]
enum IngestType {
    /// Ingest raw content through the chunking pipeline
    Content {
        content: String,
        #[arg(short, long)]
        project: Option<String>,
        /// Filename used for type detection, e.g. note.txt or page.md
        #[arg(long, default_value = "content.txt")]
        filename: String,
        /// Stable source key for dedupe and parent chunk linkage
        #[arg(long)]
        source_key: Option<String>,
        /// Optional metadata JSON object inherited by generated chunks
        #[arg(long)]
        metadata: Option<String>,
        /// Structural cue inherited by every generated chunk
        #[arg(long = "structural-cue")]
        structural_cues: Vec<String>,
        /// Text segmenter for raw content: sentence_window or logical_block
        #[arg(long, default_value = "sentence_window")]
        segmenter: String,
        /// Sentence window size used when splitting oversized text blocks
        #[arg(long)]
        segment_window_size: Option<usize>,
        /// Sentence overlap used when splitting oversized text blocks
        #[arg(long)]
        segment_overlap: Option<usize>,
        /// Minimum chunk size in characters
        #[arg(long)]
        segment_min_chunk_chars: Option<usize>,
        /// Maximum chunk size in characters
        #[arg(long)]
        segment_max_chunk_chars: Option<usize>,
        /// Server URL
        #[arg(long, default_value = "http://localhost:8735")]
        url: String,
    },
    /// Ingest a file
    File {
        path: String,
        #[arg(short, long)]
        project: Option<String>,
        /// Server URL
        #[arg(long, default_value = "http://localhost:8735")]
        url: String,
    },
    /// Ingest a URL
    Url {
        url: String,
        #[arg(short, long)]
        project: Option<String>,
        /// Crawl depth: 0 = single page, 1+ = follow links
        #[arg(short, long, default_value = "0")]
        depth: u8,
        /// Only follow links within the same domain
        #[arg(long, default_value = "true")]
        same_domain_only: bool,
        /// Server URL
        #[arg(long, default_value = "http://localhost:8735")]
        server_url: String,
    },
}

#[derive(Parser, Debug)]
struct RecallArgs {
    /// Query string
    query: String,
    /// Project ID
    #[arg(short, long)]
    project: Option<String>,
    /// Limit results
    #[arg(short, long, default_value = "10")]
    limit: usize,
    /// Manual cues to filter by
    #[arg(short, long)]
    cues: Vec<String>,
    /// Query signal mode: lexical, semantic, or hybrid
    #[arg(long, default_value = "hybrid", value_parser = ["lexical", "semantic", "hybrid"])]
    semantic_mode: String,
    /// Multi-hop recall depth
    #[arg(long, default_value = "1")]
    depth: usize,
    /// Token budget for grounded recall (context window)
    #[arg(long, default_value = "500")]
    token_budget: u32,

    /// Server port (overrides config)
    #[arg(long, default_value = "8735")]
    pub port: u16,
    /// Disable automatic reinforcement during recall
    #[arg(long)]
    no_auto_reinforce: bool,
    /// Minimum cue intersection count
    #[arg(long)]
    min_intersection: Option<usize>,
    /// Reference time for resolving relative temporal queries, e.g. "2023/05/05 16:42"
    #[arg(long)]
    query_time: Option<String>,
    /// Disable salience bias (recency vs frequency weighting)
    #[arg(long)]
    disable_salience_bias: bool,
    /// Segment parent fusion fallback mode: off, auto, or force
    #[arg(long, default_value = "off", value_parser = ["off", "auto", "force"])]
    parent_fusion: String,
    /// Internal recall limit for parent fusion fallback
    #[arg(long, default_value_t = 80)]
    parent_fusion_limit: usize,
    /// Minimum matching chunks from a parent required for fusion
    #[arg(long, default_value_t = 2)]
    parent_fusion_min_chunks: usize,
    /// Ordered reconstruction recall mode: off, auto, or force
    #[arg(long, default_value = "off", value_parser = ["off", "auto", "force"])]
    ordered_reconstruction: String,
    /// Internal recall limit for ordered reconstruction
    #[arg(long, default_value_t = 80)]
    ordered_reconstruction_limit: usize,
    /// Maximum ordered entries to scan per selected session
    #[arg(long, default_value_t = 4096)]
    ordered_session_scan_limit: usize,
    /// Maximum sessions to inspect during ordered reconstruction
    #[arg(long, default_value_t = 3)]
    ordered_max_sessions: usize,
    /// Evidence coverage recall mode for multi-evidence summary queries: off, auto, or force
    #[arg(long, default_value = "off", value_parser = ["off", "auto", "force"])]
    evidence_coverage: String,
    /// Internal result limit for evidence coverage
    #[arg(long, default_value_t = 100)]
    evidence_coverage_limit: usize,
    /// Maximum ordered entries to scan per selected session during evidence coverage
    #[arg(long, default_value_t = 4096)]
    evidence_coverage_session_scan_limit: usize,
    /// Maximum sessions to inspect during evidence coverage
    #[arg(long, default_value_t = 3)]
    evidence_coverage_max_sessions: usize,
    #[arg(long, default_value_t = 1)]
    pub expansion_depth: usize,

    /// Enable alias expansion (default: disabled)
    #[arg(long)]
    pub enable_alias_expansion: bool,
    /// Disable installed CueBridge artifacts for this recall request
    #[arg(long)]
    pub disable_cuebridge_artifacts: bool,
    /// Maximum GapPack expansion cues when CueBridge fires
    #[arg(long, default_value_t = 6)]
    pub cuebridge_gap_limit: usize,
    /// Enable grounded recall (RAG context)
    #[arg(short, long)]
    grounded: bool,
    /// Include explanation
    #[arg(short, long)]
    explain: bool,
    /// Include recall timing breakdown in JSON response
    #[arg(long)]
    trace_timing: bool,
    /// Server URL
    #[arg(long, default_value = "http://localhost:8735")]
    url: String,

    /// Enable web recall mode
    #[arg(short = 'w', long)]
    web: bool,

    /// Target URL for web recall (distinct from server url)
    #[arg(long)]
    target_url: Option<String>,

    /// Persist fetched web content (only for web recall)
    #[arg(long)]
    persist: bool,
}

#[derive(Parser, Debug)]
struct LexiconArgs {
    #[command(subcommand)]
    cmd: LexiconCmd,
}

#[derive(clap::Subcommand, Debug)]
enum LexiconCmd {
    /// Inspect a specific cue
    Inspect {
        cue: String,
        #[arg(short, long)]
        project: Option<String>,
        /// Server URL
        #[arg(long, default_value = "http://localhost:8735")]
        url: String,
    },
}

#[derive(Parser, Debug)]
struct MemoriesArgs {
    /// Memory ID
    id: u32,

    /// Reinforce this memory
    #[arg(long)]
    reinforce: bool,

    /// Delete this memory
    #[arg(long)]
    delete: bool,

    /// Specific cues for reinforcement (optional)
    #[arg(long)]
    cues: Vec<String>,

    /// Project ID
    #[arg(short, long)]
    project: Option<String>,

    /// Server URL
    #[arg(long, default_value = "http://localhost:8735")]
    url: String,
}

#[derive(Parser, Debug)]
struct AliasArgs {
    /// Text to get/add alias for
    text: String,
    /// Project ID
    #[arg(short, long)]
    project: Option<String>,
    /// Alias to add (if adding)
    #[arg(short, long)]
    add: Option<String>,
    /// Association weight (0.0 to 1.0)
    #[arg(short, long)]
    weight: Option<f64>,
    /// Server URL
    #[arg(long, default_value = "http://localhost:8735")]
    url: String,
}

#[derive(Parser, Debug)]
struct ProjectArgs {
    #[command(subcommand)]
    cmd: ProjectCmd,
}

#[derive(clap::Subcommand, Debug)]
enum ProjectCmd {
    /// List all projects
    List {
        #[arg(long, default_value = "http://localhost:8735")]
        url: String,
    },
    /// Create a new project
    Create {
        #[arg(short, long)]
        name: String,
        #[arg(long, default_value = "http://localhost:8735")]
        url: String,
    },
    /// Set watch directory for a project
    SetWatchDir {
        /// Project ID
        project: String,
        /// Path to watch directory
        path: String,
        #[arg(long, default_value = "http://localhost:8735")]
        url: String,
    },
}

fn apply_start_overrides(
    mut server_config: config::ServerConfig,
    args: &StartArgs,
) -> config::ServerConfig {
    if let Some(port) = args.port {
        server_config.server.port = port;
    }
    if let Some(data_dir) = &args.data_dir {
        server_config.server.data_dir = data_dir.clone();
    }
    if let Some(assets_dir) = &args.assets_dir {
        server_config.server.assets_dir = Some(assets_dir.clone());
    }
    if let Some(snapshot_interval) = args.snapshot_interval {
        server_config.persistence.snapshot_interval_seconds = snapshot_interval;
    }
    if let Some(agent_throttle) = args.agent_throttle {
        server_config.agent.throttle_ms = agent_throttle;
    }
    if let Some(agent_dir) = &args.agent_dir {
        server_config.agent.watch_dir = Some(agent_dir.clone());
        server_config.agent.enabled = true;
    }

    if args.disable_bg_jobs {
        server_config.jobs.background_processing = false;
    }
    if args.disable_snapshots {
        server_config.persistence.enabled = false;
    }
    if args.disk_content {
        server_config.server.store_content_on_disk = true;
    }

    if let Some(provider) = &args.cloud_backup {
        server_config.persistence.cloud.provider = provider.clone();
    }
    if let Some(bucket) = &args.cloud_bucket {
        server_config.persistence.cloud.bucket = bucket.clone();
    }
    if let Some(region) = &args.cloud_region {
        server_config.persistence.cloud.region = region.clone();
    }
    if let Some(endpoint) = &args.cloud_endpoint {
        server_config.persistence.cloud.endpoint = Some(endpoint.clone());
    }
    if let Some(prefix) = &args.cloud_prefix {
        server_config.persistence.cloud.prefix = prefix.clone();
    }
    if args.cloud_auto_backup {
        server_config.persistence.cloud.auto_backup = true;
    }

    server_config
}

fn load_start_config(args: &StartArgs) -> Result<config::ServerConfig, String> {
    let config_path = args.config.clone().map(std::path::PathBuf::from);
    let config = config::ServerConfig::load(config_path, args.profile.clone())?;
    Ok(apply_start_overrides(config, args))
}

fn has_main_snapshot(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries.flatten().any(|entry| {
                entry
                    .path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| {
                        name.ends_with(".bin")
                            && !name.ends_with("_aliases.bin")
                            && !name.ends_with("_lexicon.bin")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn select_snapshots_dir(data_dir: &str, load_static: Option<&str>) -> String {
    if let Some(static_dir) = load_static {
        return static_dir.to_string();
    }

    let configured = PathBuf::from(data_dir).join("snapshots");
    let legacy = PathBuf::from(data_dir).join("..").join("snapshots");
    let selected = if !has_main_snapshot(&configured) && has_main_snapshot(&legacy) {
        warn!(
            configured = %configured.display(),
            legacy = %legacy.display(),
            "No snapshots found in configured data directory; using legacy snapshot directory"
        );
        legacy
    } else {
        configured
    };
    selected.to_string_lossy().to_string()
}

fn resolve_master_key(
    security: &config::SecurityConfig,
    env_master_key: Option<&str>,
    env_password: Option<&str>,
    password_salt: Option<&[u8]>,
) -> Option<Arc<cuemap::crypto::EncryptionKey>> {
    if let Some(key_hex) = env_master_key {
        match hex::decode(key_hex) {
            Ok(bytes) if bytes.len() == 32 => {
                info!("Security: Master key loaded from CUEMAP_MASTER_KEY (Hex)");
                Some(Arc::new(cuemap::crypto::EncryptionKey::new(bytes)))
            }
            _ => {
                error!("Security: CUEMAP_MASTER_KEY must be a 32-byte hex string");
                None
            }
        }
    } else if let Some(passphrase) = env_password {
        info!("Security: Deriving master key from CUEMAP_MASTER_PASSWORD...");
        let salt = password_salt.unwrap_or_default();
        Some(Arc::new(cuemap::crypto::EncryptionKey::from_passphrase(
            passphrase, salt,
        )))
    } else if let Some(key_hex) = &security.master_key {
        match hex::decode(key_hex) {
            Ok(bytes) if bytes.len() == 32 => {
                info!("Security: Master key loaded from config file");
                Some(Arc::new(cuemap::crypto::EncryptionKey::new(bytes)))
            }
            _ => {
                error!("Security: master_key in config must be a 32-byte hex string");
                None
            }
        }
    } else {
        info!("Security: Encryption-at-rest disabled (no master key configured)");
        None
    }
}

fn resolve_context_signer(
    security: &config::SecurityConfig,
) -> Option<Arc<cuemap::crypto::ContextSigner>> {
    if let Some(seed_hex) = &security.signing_private_key {
        match crypto::ContextSigner::from_ed25519_seed_hex(seed_hex) {
            Ok(signer) => {
                info!("Immutable RAG: Ed25519 context signing enabled");
                Some(Arc::new(signer))
            }
            Err(err) => {
                error!("Immutable RAG: invalid Ed25519 signing private key: {}", err);
                None
            }
        }
    } else if let Some(secret) = &security.secret_key {
        info!("Immutable RAG: legacy HMAC-SHA256 context signing enabled");
        Some(Arc::new(crypto::ContextSigner::from_hmac_secret(
            secret.clone().into_bytes(),
        )))
    } else {
        None
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Start(args) => {
            if args.detach && !args.child_process {
                handle_start_detached(args).await;
            } else {
                // Layering Logic: Config File -> CLI Args
                let config = load_start_config(&args).expect("Failed to load configuration");

                run_server(config, args.load_static, args.child_process).await;
            }
        }
        Commands::Add(args) => handle_add(args).await,
        Commands::Ingest(args) => handle_ingest(args).await,
        Commands::Recall(args) => handle_recall(args).await,
        Commands::Lexicon(args) => handle_lexicon(args).await,
        Commands::Memories(args) => handle_memories(args).await,
        Commands::Alias(args) => handle_alias(args).await,
        Commands::Projects(args) => handle_projects(args).await,
        Commands::SetProject { project_id } => handle_set_project(project_id),
        Commands::Status(args) => handle_status(args).await,
        Commands::Logs(args) => handle_logs(args).await,
        Commands::Stop(args) => handle_stop(args).await,
    }
}

async fn run_server(config: config::ServerConfig, load_static: Option<String>, _is_child: bool) {
    let pid_path = config::get_base_dir().join("server.pid");
    run_server_with_pid_path(config, load_static, _is_child, pid_path).await;
}

async fn run_server_with_pid_path(
    config: config::ServerConfig,
    load_static: Option<String>,
    _is_child: bool,
    pid_path: PathBuf,
) {
    // Extract commonly used configs
    let server_config = &config.server;
    let auth_config_struct = &config.security;

    // Initialize tracing with custom filter to silence noisy components by default
    // TODO: Use config.server.log_level
    let filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(Level::INFO.into())
        .add_directive("nlprule=warn".parse().unwrap())
        .add_directive("cuemap::agent=warn".parse().unwrap())
        .add_directive("cuemap::jobs=warn".parse().unwrap())
        .add_directive("tiktoken_rs=warn".parse().unwrap());

    // Build layers
    let stdout_layer = fmt::layer().with_writer(std::io::stdout);

    let _ = Registry::default()
        .with(filter)
        .with(stdout_layer)
        .try_init();

    // Write PID file for the server
    let pid = std::process::id();
    if let Err(e) = std::fs::write(&pid_path, pid.to_string()) {
        warn!("Failed to write PID file: {}", e);
    }

    info!("CueMap Rust Engine - Production Mode");
    info!("Logs are written to stdout");

    // Initialize authentication
    // Adapt SecurityConfig to AuthConfig
    let auth_config = AuthConfig::from_config(auth_config_struct);

    // Check for start mode
    let is_static = load_static.is_some();

    if is_static {
        info!("Static loading mode enabled (read-only)");
        info!("Loading from: {}", load_static.as_ref().unwrap());
        info!("Persistence disabled - all changes will be lost on restart");
    } else {
        info!("Data directory: {}", server_config.data_dir);
        if !config.persistence.enabled {
            info!("Persistence: Snapshots DISABLED");
        } else {
            info!(
                "Snapshot interval: {}s",
                config.persistence.snapshot_interval_seconds
            );
        }
    }

    // Build the router with appropriate engine state
    info!("Multi-tenant mode enabled");

    let snapshots_dir = select_snapshots_dir(&server_config.data_dir, load_static.as_deref());

    let mut mt_engine = multi_tenant::MultiTenantEngine::with_config(
        config.clone(),
        PathBuf::from(&snapshots_dir),
    );

    // Master Key Discovery Hierarchy
    let env_master_key = std::env::var("CUEMAP_MASTER_KEY").ok();
    let env_password = std::env::var("CUEMAP_MASTER_PASSWORD").ok();
    let password_salt = env_password.as_ref().map(|_| get_or_create_salt());
    let master_key = resolve_master_key(
        auth_config_struct,
        env_master_key.as_deref(),
        env_password.as_deref(),
        password_salt.as_deref(),
    );

    if let Some(key) = master_key {
        mt_engine.set_master_key(Some(key));
    }

    let context_signer = resolve_context_signer(auth_config_struct);

    let mt_engine = Arc::new(mt_engine);

    // Auto-load all available snapshots
    info!("Loading snapshots from: {}", snapshots_dir);
    for (project_id, result) in mt_engine.load_all() {
        match result {
            Ok(()) => info!(project_id = %project_id, "Loaded project snapshot"),
            Err(error) => error!(project_id = %project_id, error = %error, "Failed to load project snapshot"),
        }
    }

    // Setup shutdown handler
    if !is_static {
        if config.persistence.enabled {
            setup_multi_tenant_shutdown_handler(mt_engine.clone()).await;
            mt_engine.start_periodic_snapshots(Duration::from_secs(
                config.persistence.snapshot_interval_seconds,
            ));
        } else {
            warn!("Periodic snapshots and shutdown save are DISABLED.");
        }
    }

    // Initialize metrics collector
    let metrics = Arc::new(cuemap::metrics::MetricsCollector::new());

    let provider: Arc<dyn jobs::ProjectProvider> = mt_engine.clone();
    let job_queue = Arc::new(jobs::JobQueue::new_with_config(
        provider,
        Some(metrics.clone()),
        config.jobs.clone(),
    ));

    let mt_engine = mt_engine;

    // Initialize dynamic Agent Manager
    let agent_manager = Arc::new(agent::manager::AgentManager::new(
        job_queue.clone(),
        mt_engine.clone(),
    ));

    // Auto-start agents for projects with watch directories configured
    for proj_stats in mt_engine.list_projects() {
        if let Ok(meta) = mt_engine.load_project_meta(&proj_stats.project_id) {
            if meta.agent_enabled {
                if let Some(watch_dir) = meta.watch_dir {
                    let agent_config = agent::AgentConfig {
                        project_id: meta.project_id.clone(),
                        watch_dir,
                        throttle_ms: config.agent.throttle_ms,
                        state_file: Some(
                            std::path::PathBuf::from(&server_config.data_dir)
                                .join("snapshots")
                                .join(format!("{}_agent_state.json", meta.project_id)),
                        ),
                        included_paths: meta.included_paths,
                        ignored_patterns: meta.ignored_patterns,
                        ignored_extensions: meta.ignored_extensions,
                    };
                    agent_manager
                        .start_agent(&meta.project_id, agent_config)
                        .await;
                }
            }
        }
    }

    // Cloud Backup (Simplified - using config)
    let cloud_backup: Option<Arc<persistence::CloudBackupManager>> =
        if config.persistence.cloud.provider != "none" {
            let c = &config.persistence.cloud;
            match persistence::CloudBackupConfig::from_args(
                Some(&c.provider),
                Some(&c.bucket),
                Some(&c.region),
                c.endpoint.as_deref(),
                &c.prefix,
                c.auto_backup,
            ) {
                Ok(conf) => match persistence::CloudBackupManager::new(conf).await {
                    Ok(m) => Some(Arc::new(m)),
                    Err(_) => None,
                },
                Err(_) => None,
            }
        } else {
            None
        };

    let app = Router::new()
        .merge(api::routes(
            mt_engine,
            job_queue,
            metrics,
            auth_config,
            is_static,
            server_config.data_dir.clone(),
            cloud_backup,
            context_signer,
            agent_manager.clone(),
        ))
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], server_config.port));
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Setup shutdown handler for multi-tenant mode
async fn setup_multi_tenant_shutdown_handler(mt_engine: Arc<multi_tenant::MultiTenantEngine>) {
    tokio::spawn(async move {
        // Create futures for both SIGINT (Ctrl+C) and SIGTERM (docker stop)
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install signal handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        // Wait for either signal
        tokio::select! {
            _ = ctrl_c => {
                info!("Shutdown signal received (SIGINT), saving all projects...");
            },
            _ = terminate => {
                info!("Shutdown signal received (SIGTERM), saving all projects...");
            },
        }

        // Wrap save operation in a timeout to prevent hanging forever
        let save_future = async {
            let save_results = mt_engine.save_all();
            let saved = save_results.iter().filter(|(_, r)| r.is_ok()).count();
            let failed = save_results.iter().filter(|(_, r)| r.is_err()).count();

            if saved > 0 {
                info!("✓ Saved {} project snapshots", saved);
            }
            if failed > 0 {
                warn!("✗ Failed to save {} projects", failed);
                for (project_id, result) in save_results.iter() {
                    if let Err(e) = result {
                        warn!("  - {}: {}", project_id, e);
                    }
                }
            }
        };

        // Enforce 5 second timeout
        match tokio::time::timeout(Duration::from_secs(5), save_future).await {
            Ok(_) => info!("Shutdown complete"),
            Err(_) => {
                error!("Shutdown timed out after 5s! Forcing exit.");
                error!("Possible cause: A project was locked by a long-running ingestion task.");
            }
        }

        std::process::exit(0);
    });
}

// ========== CLI Client Handlers ==========

fn read_default_project(config_path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(config_path).ok()?;
    let config = serde_json::from_str::<serde_json::Value>(&content).ok()?;
    config
        .get("default_project")
        .and_then(|value| value.as_str().map(str::to_string))
}

fn write_default_project(config_path: &Path, project_id: &str) -> Result<(), String> {
    let config = serde_json::json!({
        "default_project": project_id
    });
    let content = serde_json::to_string_pretty(&config).map_err(|err| err.to_string())?;
    std::fs::write(config_path, content).map_err(|err| err.to_string())
}

fn get_default_project() -> Option<String> {
    let config_path = config::get_base_dir().join("config.json");
    read_default_project(&config_path)
}

fn handle_set_project(project_id: String) {
    let config_path = config::get_base_dir().join("config.json");
    match write_default_project(&config_path, &project_id) {
        Ok(()) => println!("✓ Default project set to: {}", project_id),
        Err(_) => eprintln!("✗ Failed to write config file"),
    }
}

fn load_or_create_salt(base_dir: &Path, env_override: Option<&str>) -> Vec<u8> {
    // 1. Check environment variable override (escape hatch / migration)
    if let Some(salt_str) = env_override {
        info!("Security: Using KDF salt from environment (CUEMAP_KDF_SALT)");
        return salt_str.as_bytes().to_vec();
    }

    // 2. Check local config file
    let salt_path = base_dir.join("salt");
    if salt_path.exists() {
        if let Ok(salt) = std::fs::read(&salt_path) {
            if salt.len() >= 16 {
                info!("Security: Loaded installation salt from {:?}", salt_path);
                return salt;
            } else {
                warn!("Security: Existing salt file too short, regenerating...");
            }
        }
    }

    // 3. Generate new random salt
    info!("Security: Generating new secure salt for this installation...");
    let mut salt = vec![0u8; 32];
    thread_rng().fill_bytes(&mut salt);

    // 4. Save to file
    if let Err(e) = std::fs::write(&salt_path, &salt) {
        warn!("Security: Failed to persist salt to {:?}: {}", salt_path, e);
        warn!("          WARN: You will need to regenerate keys or set CUEMAP_KDF_SALT on next run if you restart!");
    } else {
        info!("Security: Saved new salt to {:?}", salt_path);
    }

    salt
}

fn get_or_create_salt() -> Vec<u8> {
    let env_override = std::env::var("CUEMAP_KDF_SALT").ok();
    load_or_create_salt(&config::get_base_dir(), env_override.as_deref())
}

async fn handle_add(args: AddArgs) {
    let project = args
        .project
        .or_else(get_default_project)
        .expect("Project ID required (use --project or set-project)");
    let client = reqwest::Client::new();
    let payload = api::AddMemoryRequest {
        content: args.content,
        source_key: None,
        event_time: None,
        metadata: args
            .metadata
            .map(|m| serde_json::from_str(&m).unwrap_or_default()),
        embedding: None,
        cues: args.cues,
        disable_temporal_chunking: args.disable_temporal_chunking,
        async_ingest: args.async_ingest,
        minimal_response: false,
        trace_timing: false,
    };

    let res = client
        .post(format!("{}/memories", args.url))
        .header("X-Project-ID", project)
        .json(&payload)
        .send()
        .await;

    match res {
        Ok(response) => {
            if response.status().is_success() {
                let body: serde_json::Value = response.json().await.unwrap();
                println!(
                    "✓ Memory added: {}",
                    body.get("id").and_then(|v| v.as_str()).unwrap_or("unknown")
                );
            } else {
                eprintln!("✗ Error: {}", response.text().await.unwrap_or_default());
            }
        }
        Err(e) => eprintln!("✗ Failed to connect to server: {}", e),
    }
}

async fn handle_ingest(args: IngestArgs) {
    let client = reqwest::Client::new();
    match args.type_ {
        IngestType::Content {
            content,
            project,
            filename,
            source_key,
            metadata,
            structural_cues,
            segmenter,
            segment_window_size,
            segment_overlap,
            segment_min_chunk_chars,
            segment_max_chunk_chars,
            url,
        } => {
            let project = project
                .or_else(get_default_project)
                .expect("Project ID required");
            let metadata_json = metadata.and_then(|raw| {
                serde_json::from_str::<serde_json::Value>(&raw)
                    .ok()
                    .and_then(|value| value.as_object().cloned())
                    .map(|map| serde_json::Value::Object(map))
            });
            let res = client
                .post(format!("{}/ingest/content", url))
                .header("X-Project-ID", project)
                .json(&serde_json::json!({
                    "content": content,
                    "filename": filename,
                    "source_key": source_key,
                    "metadata": metadata_json,
                    "structural_cues": structural_cues,
                    "segmenter": segmenter,
                    "segment_window_size": segment_window_size,
                    "segment_overlap": segment_overlap,
                    "segment_min_chunk_chars": segment_min_chunk_chars,
                    "segment_max_chunk_chars": segment_max_chunk_chars,
                }))
                .send()
                .await;
            match res {
                Ok(r) if r.status().is_success() => println!("✓ Content ingested"),
                Ok(r) => eprintln!("✗ Error: {}", r.text().await.unwrap_or_default()),
                Err(e) => eprintln!("✗ Failed: {}", e),
            }
        }
        IngestType::File { path, project, url } => {
            let project = project
                .or_else(get_default_project)
                .expect("Project ID required");
            if let Ok(content) = std::fs::read_to_string(&path) {
                let res = client
                    .post(format!("{}/ingest/content", url))
                    .header("X-Project-ID", project)
                    .json(&serde_json::json!({ "content": content, "filename": path }))
                    .send()
                    .await;
                match res {
                    Ok(r) if r.status().is_success() => println!("✓ File ingested"),
                    Ok(r) => eprintln!("✗ Error: {}", r.text().await.unwrap_or_default()),
                    Err(e) => eprintln!("✗ Failed: {}", e),
                }
            } else {
                eprintln!("✗ Read file failed: {}", path);
            }
        }
        IngestType::Url {
            url,
            project,
            depth,
            same_domain_only,
            server_url,
        } => {
            let project = project
                .or_else(get_default_project)
                .expect("Project ID required");
            let res = client
                .post(format!("{}/ingest/url", server_url))
                .header("X-Project-ID", project)
                .json(&api::IngestUrlRequest {
                    url,
                    depth,
                    same_domain_only,
                })
                .send()
                .await;
            match res {
                Ok(r) if r.status().is_success() => println!("✓ URL ingestion started"),
                Ok(r) => eprintln!("✗ Error: {}", r.text().await.unwrap_or_default()),
                Err(e) => eprintln!("✗ Failed: {}", e),
            }
        }
    }
}

async fn handle_recall(args: RecallArgs) {
    let project = args
        .project
        .or_else(get_default_project)
        .expect("Project ID required");
    let client = reqwest::Client::new();
    let parent_fusion = match args.parent_fusion.as_str() {
        "auto" => api::ParentFusionMode::Auto,
        "force" => api::ParentFusionMode::Force,
        _ => api::ParentFusionMode::Off,
    };
    let ordered_reconstruction = match args.ordered_reconstruction.as_str() {
        "auto" => api::OrderedReconstructionMode::Auto,
        "force" => api::OrderedReconstructionMode::Force,
        _ => api::OrderedReconstructionMode::Off,
    };
    let evidence_coverage = match args.evidence_coverage.as_str() {
        "auto" => api::EvidenceCoverageMode::Auto,
        "force" => api::EvidenceCoverageMode::Force,
        _ => api::EvidenceCoverageMode::Off,
    };

    if args.grounded {
        let payload = api::RecallGroundedRequest {
            query_text: args.query,
            limit: args.limit,
            token_budget: args.token_budget,
            auto_reinforce: !args.no_auto_reinforce,
            projects: None,
            disable_salience_bias: args.disable_salience_bias,
            min_intersection: args.min_intersection,
            disable_alias_expansion: !args.enable_alias_expansion,
            expansion_depth: args.expansion_depth,
        };
        let res = client
            .post(format!("{}/recall/grounded", args.url))
            .header("X-Project-ID", project)
            .json(&payload)
            .send()
            .await;

        match res {
            Ok(r) if r.status().is_success() => {
                let body: api::RecallGroundedResponse = r.json().await.unwrap();
                println!("\n--- GROUNDED RECALL ---");
                println!("{}", body.verified_context);
            }
            Ok(r) => eprintln!("✗ Error: {}", r.text().await.unwrap_or_default()),
            Err(e) => eprintln!("✗ Failed: {}", e),
        }
    } else if args.web {
        let payload = api::RecallWebRequest {
            url: args.target_url,
            query: args.query,
            persist: args.persist,
        };
        let res = client
            .post(format!("{}/recall/web", args.url))
            .header("X-Project-ID", project)
            .json(&payload)
            .send()
            .await;

        match res {
            Ok(r) if r.status().is_success() => {
                let body: serde_json::Value = r.json().await.unwrap();
                let results = body.get("results").and_then(|v| v.as_array()).unwrap();
                let urls = body.get("urls").and_then(|v| v.as_array());

                println!("\n--- WEB RECALL RESULTS ({}) ---", results.len());
                if let Some(urls) = urls {
                    println!("Sources: {:?}", urls);
                }

                for mem in results {
                    println!(
                        "- [{:.4}] [Intersection: {}] {}",
                        mem.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0),
                        mem.get("intersection")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                        mem.get("content").and_then(|v| v.as_str()).unwrap_or("")
                    );
                }
            }
            Ok(r) => eprintln!("✗ Error: {}", r.text().await.unwrap_or_default()),
            Err(e) => eprintln!("✗ Failed: {}", e),
        }
    } else {
        let payload = api::RecallRequest {
            cues: args.cues,
            query_text: Some(args.query),
            query_embedding: None,
            semantic_mode: match args.semantic_mode.as_str() {
                "lexical" => cuemap::semantic::SemanticRecallMode::Lexical,
                "semantic" => cuemap::semantic::SemanticRecallMode::Semantic,
                _ => cuemap::semantic::SemanticRecallMode::Hybrid,
            },
            query_time: args.query_time,
            limit: args.limit,
            auto_reinforce: !args.no_auto_reinforce,
            explain: args.explain,
            trace_timing: args.trace_timing,
            projects: None,
            min_intersection: args.min_intersection,
            disable_salience_bias: args.disable_salience_bias,
            disable_alias_expansion: !args.enable_alias_expansion,
            depth: args.depth,
            expansion_depth: args.expansion_depth,
            parent_fusion,
            parent_fusion_limit: args.parent_fusion_limit,
            parent_fusion_min_chunks: args.parent_fusion_min_chunks,
            ordered_reconstruction,
            ordered_reconstruction_limit: args.ordered_reconstruction_limit,
            ordered_session_scan_limit: args.ordered_session_scan_limit,
            ordered_max_sessions: args.ordered_max_sessions,
            evidence_coverage,
            evidence_coverage_limit: args.evidence_coverage_limit,
            evidence_coverage_session_scan_limit: args.evidence_coverage_session_scan_limit,
            evidence_coverage_max_sessions: args.evidence_coverage_max_sessions,
            disable_cuebridge_artifacts: args.disable_cuebridge_artifacts,
            cuebridge_gap_limit: args.cuebridge_gap_limit,
        };
        let res = client
            .post(format!("{}/recall", args.url))
            .header("X-Project-ID", project)
            .json(&payload)
            .send()
            .await;

        match res {
            Ok(r) if r.status().is_success() => {
                let response_body: serde_json::Value = r.json().await.unwrap();
                let results = response_body
                    .get("results")
                    .and_then(|v| v.as_array())
                    .unwrap();
                println!("\n--- RECALL RESULTS ({}) ---", results.len());
                for mem in results {
                    println!(
                        "- [{:.4}] [{}] {}",
                        mem.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0),
                        mem.get("memory_id")
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "unknown".to_string()),
                        mem.get("content").and_then(|v| v.as_str()).unwrap_or("")
                    );
                }
                if args.trace_timing {
                    if let Some(timing) = response_body.get("timing") {
                        println!("\n--- TIMING ---");
                        println!("{}", serde_json::to_string_pretty(timing).unwrap_or_default());
                    }
                }
            }
            Ok(r) => eprintln!("✗ Error: {}", r.text().await.unwrap_or_default()),
            Err(e) => eprintln!("✗ Failed: {}", e),
        }
    }
}

async fn handle_lexicon(args: LexiconArgs) {
    let client = reqwest::Client::new();

    match args.cmd {
        LexiconCmd::Inspect { cue, project, url } => {
            let project_id = project
                .or_else(get_default_project)
                .expect("Project ID required");
            let res = client
                .get(format!("{}/lexicon/inspect/{}", url, cue))
                .header("X-Project-ID", project_id)
                .send()
                .await;
            match res {
                Ok(r) if r.status().is_success() => {
                    let body: api::LexiconInspectResponse = r.json().await.unwrap();
                    println!("\n--- LEXICON INSPECT: {} ---", cue);
                    println!("Associated Memories: {}", body.outgoing.len());
                }
                _ => eprintln!("✗ Inspect failed"),
            }
        }
    }
}

async fn handle_memories(args: MemoriesArgs) {
    let project = args
        .project
        .or_else(get_default_project)
        .expect("Project ID required");
    let client = reqwest::Client::new();

    if args.delete {
        let res = client
            .delete(format!("{}/memories/{}", args.url, args.id))
            .header("X-Project-ID", project)
            .send()
            .await;

        match res {
            Ok(r) if r.status().is_success() => println!("✓ Memory deleted: {}", args.id),
            Ok(r) if r.status() == 404 => eprintln!("✗ Memory not found: {}", args.id),
            _ => eprintln!("✗ Failed to delete memory"),
        }
    } else if args.reinforce {
        let payload = api::ReinforceRequest { cues: args.cues };
        let res = client
            .patch(format!("{}/memories/{}/reinforce", args.url, args.id))
            .header("X-Project-ID", project)
            .json(&payload)
            .send()
            .await;

        match res {
            Ok(r) if r.status().is_success() => println!("✓ Memory reinforced: {}", args.id),
            Ok(r) if r.status() == 404 => eprintln!("✗ Memory not found: {}", args.id),
            _ => eprintln!("✗ Failed to reinforce memory"),
        }
    } else {
        // GET
        let res = client
            .get(format!("{}/memories/{}", args.url, args.id))
            .header("X-Project-ID", project)
            .send()
            .await;

        match res {
            Ok(r) if r.status().is_success() => {
                let mem: serde_json::Value = r.json().await.unwrap();
                println!("\n--- MEMORY: {} ---", args.id);
                println!(
                    "Content: {}",
                    mem.get("content").and_then(|v| v.as_str()).unwrap_or("")
                );
                println!(
                    "Created: {}",
                    mem.get("created_at")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0)
                );

                if let Some(cues) = mem.get("cues").and_then(|v| v.as_array()) {
                    println!("Cues: {:?}", cues);
                }

                if let Some(stats) = mem.get("stats") {
                    println!("Stats: {}", serde_json::to_string_pretty(stats).unwrap());
                }
            }
            Ok(r) if r.status() == 404 => eprintln!("✗ Memory not found: {}", args.id),
            _ => eprintln!("✗ Failed to get memory"),
        }
    }
}

async fn handle_alias(args: AliasArgs) {
    let project = args
        .project
        .or_else(get_default_project)
        .expect("Project ID required");
    let client = reqwest::Client::new();

    if let Some(alias) = args.add {
        let payload = api::AddAliasRequest {
            from: args.text,
            to: alias,
            weight: args.weight,
        };
        let res = client
            .post(format!("{}/aliases", args.url))
            .header("X-Project-ID", project)
            .json(&payload)
            .send()
            .await;
        match res {
            Ok(r) if r.status().is_success() => println!("✓ Alias added"),
            _ => eprintln!("✗ Failed to add alias"),
        }
    } else {
        let res = client
            .get(format!("{}/aliases?q={}", args.url, args.text))
            .header("X-Project-ID", project)
            .send()
            .await;
        match res {
            Ok(r) if r.status().is_success() => {
                let aliases: Vec<api::AliasResponse> = r.json().await.unwrap();
                if let Some(first) = aliases.first() {
                    println!("Alias for '{}': {}", args.text, first.to);
                } else {
                    println!("No alias found for '{}'", args.text);
                }
            }
            _ => eprintln!("✗ Failed to get alias"),
        }
    }
}

async fn handle_projects(args: ProjectArgs) {
    let client = reqwest::Client::new();
    match args.cmd {
        ProjectCmd::List { url } => {
            let res = client.get(format!("{}/projects", url)).send().await;
            match res {
                Ok(r) if r.status().is_success() => {
                    let projects: Vec<serde_json::Value> = r.json().await.unwrap();
                    println!("\n--- PROJECTS ---");
                    for p in projects {
                        println!(
                            "- {} (memories: {})",
                            p.get("project_id").unwrap(),
                            p.get("total_memories").unwrap()
                        );
                    }
                }
                _ => eprintln!("✗ Failed to list projects"),
            }
        }
        ProjectCmd::Create { name, url } => {
            let res = client
                .post(format!("{}/projects", url))
                .json(&serde_json::json!({ "project_id": name }))
                .send()
                .await;
            match res {
                Ok(r) if r.status().is_success() => println!("✓ Project created: {}", name),
                Ok(r) => eprintln!("✗ Error: {}", r.text().await.unwrap_or_default()),
                Err(e) => eprintln!("✗ Failed: {}", e),
            }
        }
        ProjectCmd::SetWatchDir { project, path, url } => {
            let res = client
                .post(format!("{}/projects/{}/watch-dir", url, project))
                .json(&serde_json::json!({ "watch_dir": path }))
                .send()
                .await;
            match res {
                Ok(r) if r.status().is_success() => {
                    println!("✓ Watch directory set for project '{}'", project)
                }
                Ok(r) => eprintln!("✗ Error: {}", r.text().await.unwrap_or_default()),
                Err(e) => eprintln!("✗ Failed: {}", e),
            }
        }
    }
}

async fn handle_status(args: StatusArgs) {
    let client = reqwest::Client::new();

    // Default to --server if no flags provided
    let show_server = args.server || (!args.server && !args.jobs);

    if show_server {
        let _res = client.get(format!("{}/stats", args.url));

        let project = args.project.clone().or_else(get_default_project);
        let mut req = client.get(format!("{}/stats", args.url));
        if let Some(p) = project {
            req = req.header("X-Project-ID", p);
        }

        match req.send().await {
            Ok(r) if r.status().is_success() => {
                let stats: serde_json::Value = r.json().await.unwrap();
                if args.json {
                    println!("{}", serde_json::to_string(&stats).unwrap());
                } else {
                    println!("\n--- SERVER STATS ---");
                    println!("{}", serde_json::to_string_pretty(&stats).unwrap());
                }
            }
            Ok(r) => eprintln!("✗ Server returned error: {}", r.status()),
            Err(e) => eprintln!("✗ Failed to connect to server: {}", e),
        }
    }

    if args.jobs {
        let project = args.project.clone().or_else(get_default_project);
        let mut req = client.get(format!("{}/jobs/status", args.url));

        if let Some(p) = project {
            req = req.header("X-Project-ID", p);
        }

        match req.send().await {
            Ok(r) if r.status().is_success() => {
                let status: serde_json::Value = r.json().await.unwrap();
                if args.json {
                    println!("{}", serde_json::to_string(&status).unwrap());
                } else {
                    println!("\n--- JOB STATUS ---");
                    println!("{}", serde_json::to_string_pretty(&status).unwrap());
                }
            }
            Ok(r) => eprintln!("✗ Server returned error: {}", r.status()),
            Err(e) => eprintln!("✗ Failed to connect to server: {}", e),
        }
    }

    // Always show general metrics/uptime if no specific flag or just status
    if !args.jobs && !args.server {
        let res = client.get(format!("{}/metrics", args.url)).send().await;
        match res {
            Ok(r) if r.status().is_success() => {
                println!("\n--- METRICS ---");
                println!("{}", r.text().await.unwrap());
            }
            _ => {}
        }
    }
}

async fn handle_logs(args: LogsArgs) {
    let log_path = args.path.unwrap_or_else(|| {
        let mut path = config::get_base_dir();
        path.push("server.log");
        path.to_string_lossy().to_string()
    });

    if !Path::new(&log_path).exists() {
        eprintln!("✗ Log file not found: {}", log_path);
        println!("  - Server might not be running or hasn't created a log file yet.");
        println!("  - Use 'cuemap start' to start the server.");
        return;
    }

    let file = match File::open(&log_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("✗ Failed to open log file: {}", e);
            return;
        }
    };

    let mut reader = BufReader::new(file);

    if let Some(n) = args.head {
        let mut line = String::new();
        for _ in 0..n {
            line.clear();
            if reader.read_line(&mut line).unwrap() == 0 {
                break;
            }
            print!("{}", line);
        }
        return;
    }

    if let Some(n) = args.tail {
        let mut lines = Vec::new();
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap() > 0 {
            lines.push(line.clone());
            line.clear();
        }

        let start = if lines.len() > n { lines.len() - n } else { 0 };
        for l in &lines[start..] {
            print!("{}", l);
        }

        if !args.follow {
            return;
        }
        // If follow is also set, we are already at the end of the file
    } else if !args.follow {
        // Print everything
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap() > 0 {
            print!("{}", line);
            line.clear();
        }
        return;
    }

    // Follow implementation
    println!("--- Following logs: {} (Ctrl+C to quit) ---", log_path);

    // Jump to end if not already there (e.g. if tail wasn't used)
    if args.tail.is_none() {
        reader.get_mut().seek(SeekFrom::End(0)).unwrap();
    }

    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // End of file, wait for more
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Ok(_) => {
                print!("{}", line);
            }
            Err(e) => {
                eprintln!("✗ Error reading logs: {}", e);
                break;
            }
        }
    }
}

async fn wait_for_readiness(
    log_path: &Path,
    start_pos: u64,
    sentinel: &str,
    timeout: Duration,
) -> std::io::Result<bool> {
    let mut file = File::open(log_path)?;
    let _ = file.seek(SeekFrom::Start(start_pos));
    let mut reader = BufReader::new(file);
    let start_time = std::time::Instant::now();
    let mut line = String::new();

    while start_time.elapsed() < timeout {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Ok(_) => {
                print!("{}", line);
                if line.contains(sentinel) {
                    return Ok(true);
                }
            }
            Err(_) => return Ok(false),
        }
    }

    Ok(false)
}

async fn spawn_detached_process(
    exe: &Path,
    child_args: &[String],
    log_path: &Path,
    sentinel: &str,
    timeout: Duration,
) -> std::io::Result<bool> {
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let stdout_file = log_file.try_clone()?;
    let stderr_file = log_file.try_clone()?;
    let start_pos = log_file.metadata().map(|metadata| metadata.len()).unwrap_or(0);

    let child = std::process::Command::new(exe)
        .args(child_args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(stdout_file))
        .stderr(std::process::Stdio::from(stderr_file))
        .spawn()?;

    println!("✓ Background server spawning (PID: {})...", child.id());
    println!("✓ Waiting for readiness sentinel in {}...", log_path.display());
    tokio::time::sleep(Duration::from_millis(100)).await;
    wait_for_readiness(log_path, start_pos, sentinel, timeout).await
}

async fn handle_start_detached(args: StartArgs) {
    let mut child_args: Vec<String> = std::env::args()
        .filter(|a| a != "--detach" && a != "-d")
        .collect();
    child_args.push("--child-process".to_string());

    // Fix: Make sure the first argument is just the binary name if it's the full path
    let exe = std::env::current_exe().expect("Failed to get current executable");

    // Determine log file path
    let log_path = args.log_file.clone().unwrap_or_else(|| {
        let mut path = config::get_base_dir();
        path.push("server.log");
        path.to_string_lossy().to_string()
    });

    // Readiness sentinel we are looking for: "Unstable sorting for speed"
    let sentinel = "Unstable sorting for speed";
    let timeout = Duration::from_secs(30);

    match spawn_detached_process(
        &exe,
        &child_args[1..],
        Path::new(&log_path),
        sentinel,
        timeout,
    )
    .await
    {
        Ok(true) => {
            println!("\n✓ CueMap server is now running in the background.");
            println!("  - View logs:   cuemap logs --follow");
            println!("  - Stop server: cuemap stop");
        }
        Ok(false) => {
            eprintln!(
                "\n✗ Timeout waiting for server readiness. Check logs at: {}",
                log_path
            );
        }
        Err(_) => eprintln!(
            "\n✗ Could not open log file to verify startup: {}",
            log_path
        ),
    }
}

async fn handle_stop(_args: StopArgs) {
    handle_stop_at(config::get_base_dir().join("server.pid")).await;
}

async fn handle_stop_at(pid_path: PathBuf) {
    if !pid_path.exists() {
        eprintln!("✗ No server.pid found. Server might not be running or wasn't started with this version.");
        return;
    }

    let pid_str = std::fs::read_to_string(&pid_path).expect("Failed to read PID file");
    let pid: u32 = pid_str.trim().parse().expect("Invalid PID in file");

    #[cfg(unix)]
    {
        use std::process::Command;
        if pid <= 1 || pid > i32::MAX as u32 {
            eprintln!("✗ Refusing to signal invalid server PID {}.", pid);
            return;
        }
        let res = Command::new("kill")
            .arg("-15") // SIGTERM
            .arg(pid.to_string())
            .status();

        match res {
            Ok(s) if s.success() => {
                println!("✓ Termination signal sent to server (PID: {})", pid);
                let _ = std::fs::remove_file(pid_path);
            }
            _ => eprintln!(
                "✗ Failed to kill process {}. It might have already exited.",
                pid
            ),
        }
    }

    #[cfg(windows)]
    {
        use std::process::Command;
        let res = Command::new("taskkill")
            .arg("/F")
            .arg("/T")
            .arg("/PID")
            .arg(pid.to_string())
            .status();

        match res {
            Ok(s) if s.success() => {
                println!("✓ Server process (PID: {}) terminated", pid);
                let _ = std::fs::remove_file(pid_path);
            }
            _ => eprintln!(
                "✗ Failed to kill process {}. It might have already exited.",
                pid
            ),
        }
    }
}
