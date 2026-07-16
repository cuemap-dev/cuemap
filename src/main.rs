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

    /// Manage deterministic CuePacks
    Cuepack(CuePackArgs),

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
    #[arg(long, default_value = "http://localhost:8080")]
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
    #[arg(long, default_value = "http://localhost:8080")]
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
    /// CuePacks to apply while extracting synchronous facets (comma-separated)
    #[arg(long, value_delimiter = ',')]
    cuepacks: Option<Vec<String>>,
    /// Disable bundled default CuePacks for this add request
    #[arg(long)]
    disable_default_cuepacks: bool,
    /// Disable temporal chunking for this memory
    #[arg(long)]
    disable_temporal_chunking: bool,
    /// Process ingestion in background (return immediately)
    #[arg(long)]
    async_ingest: bool,
    /// Server URL
    #[arg(long, default_value = "http://localhost:8080")]
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
        #[arg(long, default_value = "http://localhost:8080")]
        url: String,
    },
    /// Ingest a file
    File {
        path: String,
        #[arg(short, long)]
        project: Option<String>,
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
    /// Multi-hop recall depth
    #[arg(long, default_value = "1")]
    depth: usize,
    /// Token budget for grounded recall (context window)
    #[arg(long, default_value = "500")]
    token_budget: u32,

    /// Server port (overrides config)
    #[arg(long, default_value = "8080")]
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

    /// CuePacks to apply (comma-separated, e.g. "memory-general")
    #[arg(long, value_delimiter = ',')]
    pub cuepacks: Option<Vec<String>>,
    /// Disable bundled default CuePacks for this recall request
    #[arg(long)]
    pub disable_default_cuepacks: bool,

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
    #[arg(long, default_value = "http://localhost:8080")]
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
    },
}

#[derive(Parser, Debug)]
struct CuePackArgs {
    #[command(subcommand)]
    cmd: CuePackCmd,
}

#[derive(clap::Subcommand, Debug)]
enum CuePackCmd {
    /// List bundled and local CuePacks
    List,
    /// Inspect a loaded CuePack by name
    Inspect { name: String },
    /// Validate a CuePack TOML file
    Validate { path: String },
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
    #[arg(long, default_value = "http://localhost:8080")]
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
    #[arg(long, default_value = "http://localhost:8080")]
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
        #[arg(long, default_value = "http://localhost:8080")]
        url: String,
    },
    /// Create a new project
    Create {
        #[arg(short, long)]
        name: String,
        #[arg(long, default_value = "http://localhost:8080")]
        url: String,
    },
    /// Set watch directory for a project
    SetWatchDir {
        /// Project ID
        project: String,
        /// Path to watch directory
        path: String,
        #[arg(long, default_value = "http://localhost:8080")]
        url: String,
    },
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
                let config_path = args.config.clone().map(std::path::PathBuf::from);
                let mut config = config::ServerConfig::load(config_path, args.profile.clone())
                    .expect("Failed to load configuration");

                // Apply CLI Overrides
                if let Some(p) = args.port {
                    config.server.port = p;
                }
                if let Some(d) = &args.data_dir {
                    config.server.data_dir = d.clone();
                }
                if let Some(a) = &args.assets_dir {
                    config.server.assets_dir = Some(a.clone());
                }
                if let Some(s) = args.snapshot_interval {
                    config.persistence.snapshot_interval_seconds = s;
                }
                if let Some(t) = args.agent_throttle {
                    config.agent.throttle_ms = t;
                }
                if let Some(w) = &args.agent_dir {
                    config.agent.watch_dir = Some(w.clone());
                    config.agent.enabled = true;
                }

                // Boolean flags (only enable restriction/feature if flag is present, or if config says so)
                // For "disable" flags: if CLI says disable, force disable.
                if args.disable_bg_jobs {
                    config.jobs.background_processing = false;
                }
                if args.disable_snapshots {
                    config.persistence.enabled = false;
                }
                if args.disk_content {
                    config.server.store_content_on_disk = true;
                }

                // Cloud overrides
                if let Some(p) = &args.cloud_backup {
                    config.persistence.cloud.provider = p.clone();
                }
                if let Some(b) = &args.cloud_bucket {
                    config.persistence.cloud.bucket = b.clone();
                }
                if let Some(r) = &args.cloud_region {
                    config.persistence.cloud.region = r.clone();
                }
                if let Some(e) = &args.cloud_endpoint {
                    config.persistence.cloud.endpoint = Some(e.clone());
                }
                if let Some(p) = &args.cloud_prefix {
                    config.persistence.cloud.prefix = p.clone();
                }
                if args.cloud_auto_backup {
                    config.persistence.cloud.auto_backup = true;
                }

                run_server(config, args.load_static, args.child_process).await;
            }
        }
        Commands::Add(args) => handle_add(args).await,
        Commands::Ingest(args) => handle_ingest(args).await,
        Commands::Recall(args) => handle_recall(args).await,
        Commands::Lexicon(args) => handle_lexicon(args).await,
        Commands::Cuepack(args) => handle_cuepack(args),
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

    Registry::default().with(filter).with(stdout_layer).init();

    // Write PID file for the server
    let pid = std::process::id();
    let pid_path = config::get_base_dir().join("server.pid");
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

    use cuemap::crypto::EncryptionKey;

    // Build the router with appropriate engine state
    info!("Multi-tenant mode enabled");

    let snapshots_dir = if let Some(ref static_dir) = load_static {
        static_dir.clone()
    } else {
        PathBuf::from(&server_config.data_dir)
            .join("snapshots")
            .to_string_lossy()
            .to_string()
    };

    let mut mt_engine = multi_tenant::MultiTenantEngine::with_config(
        config.clone(),
        PathBuf::from(&snapshots_dir),
    );

    let mut cuepack_dirs = config
        .cuepacks
        .dirs
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    cuepack_dirs.push(config::get_base_dir().join("cuepacks"));
    let cuepack_registry = Arc::new(if config.cuepacks.enabled {
        cuemap::cuepacks::CuePackRegistry::load(config.cuepacks.default_packs_enabled, &cuepack_dirs)
    } else {
        cuemap::cuepacks::CuePackRegistry::load(false, &[])
    });
    for error in cuepack_registry.load_errors() {
        warn!("CuePack load error: {}", error);
    }
    info!("Loaded {} CuePacks", cuepack_registry.infos().len());

    // Master Key Discovery Hierarchy
    let master_key = if let Ok(key_hex) = std::env::var("CUEMAP_MASTER_KEY") {
        // 1. Env Var (Hex) - Highest priority for automation
        match hex::decode(&key_hex) {
            Ok(bytes) if bytes.len() == 32 => {
                info!("Security: Master key loaded from CUEMAP_MASTER_KEY (Hex)");
                Some(Arc::new(EncryptionKey::new(bytes)))
            }
            _ => {
                error!("Security: CUEMAP_MASTER_KEY must be a 32-byte hex string");
                None
            }
        }
    } else if let Ok(pass) = std::env::var("CUEMAP_MASTER_PASSWORD") {
        // 2. Env Var (Passphrase) - Secondary automation path
        info!("Security: Deriving master key from CUEMAP_MASTER_PASSWORD...");
        let salt = get_or_create_salt();
        Some(Arc::new(EncryptionKey::from_passphrase(&pass, &salt)))
    } else if let Some(key_hex) = &auth_config_struct.master_key {
        // 3. Config File (Hex)
        match hex::decode(key_hex) {
            Ok(bytes) if bytes.len() == 32 => {
                info!("Security: Master key loaded from config file");
                Some(Arc::new(EncryptionKey::new(bytes)))
            }
            _ => {
                error!("Security: master_key in config must be a 32-byte hex string");
                None
            }
        }
    } else {
        info!("Security: Encryption-at-rest disabled (no master key configured)");
        None
    };

    if let Some(key) = master_key {
        mt_engine.set_master_key(Some(key));
    }

    let context_signer = if let Some(seed_hex) = &auth_config_struct.signing_private_key {
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
    } else if let Some(secret) = &auth_config_struct.secret_key {
        info!("Immutable RAG: legacy HMAC-SHA256 context signing enabled");
        Some(Arc::new(crypto::ContextSigner::from_hmac_secret(
            secret.clone().into_bytes(),
        )))
    } else {
        None
    };

    let mt_engine = Arc::new(mt_engine);

    // Auto-load all available snapshots
    info!("Loading snapshots from: {}", snapshots_dir);
    let _ = mt_engine.load_all(); // Ignoring errors for brevity

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
                        ignored_patterns: Vec::new(),
                        ignored_extensions: Vec::new(),
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
            cuepack_registry,
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

fn get_default_project() -> Option<String> {
    let config_path = config::get_base_dir().join("config.json");
    if let Ok(content) = std::fs::read_to_string(config_path) {
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
            return config
                .get("default_project")
                .and_then(|v| v.as_str().map(|s| s.to_string()));
        }
    }
    None
}

fn handle_set_project(project_id: String) {
    let config_path = config::get_base_dir().join("config.json");
    let config = serde_json::json!({
        "default_project": project_id
    });
    if let Ok(content) = serde_json::to_string_pretty(&config) {
        if std::fs::write(config_path, content).is_ok() {
            println!("✓ Default project set to: {}", project_id);
        } else {
            eprintln!("✗ Failed to write config file");
        }
    }
}

fn get_or_create_salt() -> Vec<u8> {
    // 1. Check environment variable override (escape hatch / migration)
    if let Ok(salt_str) = std::env::var("CUEMAP_KDF_SALT") {
        info!("Security: Using KDF salt from environment (CUEMAP_KDF_SALT)");
        return salt_str.into_bytes();
    }

    // 2. Check local config file
    let salt_path = config::get_base_dir().join("salt");
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

async fn handle_add(args: AddArgs) {
    let project = args
        .project
        .or_else(get_default_project)
        .expect("Project ID required (use --project or set-project)");
    let client = reqwest::Client::new();
    let cuepacks = selected_cuepacks(args.cuepacks, args.disable_default_cuepacks);

    let payload = api::AddMemoryRequest {
        content: args.content,
        source_key: None,
        metadata: args
            .metadata
            .map(|m| serde_json::from_str(&m).unwrap_or_default()),
        cues: args.cues,
        cuepacks,
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
        IngestType::File { path, project } => {
            let project = project
                .or_else(get_default_project)
                .expect("Project ID required");
            if let Ok(content) = std::fs::read_to_string(&path) {
                let res = client
                    .post("http://localhost:8080/ingest/content")
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
        } => {
            let project = project
                .or_else(get_default_project)
                .expect("Project ID required");
            let res = client
                .post("http://localhost:8080/ingest/url")
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
    let cuepacks = selected_cuepacks(args.cuepacks.clone(), args.disable_default_cuepacks);
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
            cuepacks: cuepacks.clone(),
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
            cuepacks,
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
        LexiconCmd::Inspect { cue, project } => {
            let project_id = project
                .or_else(get_default_project)
                .expect("Project ID required");
            let res = client
                .get(format!("http://localhost:8080/lexicon/inspect/{}", cue))
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

fn selected_cuepacks(cuepacks: Option<Vec<String>>, disable_defaults: bool) -> Option<Vec<String>> {
    if disable_defaults {
        Some(vec!["off".to_string()])
    } else {
        cuepacks
    }
}

fn local_cuepack_registry() -> cuemap::cuepacks::CuePackRegistry {
    cuemap::cuepacks::CuePackRegistry::load_from_default_locations(true)
}

fn handle_cuepack(args: CuePackArgs) {
    match args.cmd {
        CuePackCmd::List => {
            let registry = local_cuepack_registry();
            println!("\n--- CUEPACKS ---");
            for info in registry.infos() {
                let default_marker = if info.enabled_by_default {
                    "default"
                } else {
                    "opt-in"
                };
                println!(
                    "- {} {} [{}] memory_rules={} query_rules={} source={}",
                    info.name,
                    info.version,
                    default_marker,
                    info.memory_rules,
                    info.query_rules,
                    info.source
                );
            }
            for error in registry.load_errors() {
                eprintln!("! {}", error);
            }
        }
        CuePackCmd::Inspect { name } => {
            let registry = local_cuepack_registry();
            let Some(info) = registry
                .infos()
                .into_iter()
                .find(|info| info.name == name)
            else {
                eprintln!("✗ CuePack not found: {}", name);
                return;
            };
            println!("\n--- CUEPACK: {} ---", info.name);
            println!("Version: {}", info.version);
            println!("Default: {}", info.enabled_by_default);
            println!("Source: {}", info.source);
            println!("Memory rules: {}", info.memory_rules);
            println!("Query rules: {}", info.query_rules);
            if let Some(description) = info.description {
                println!("{}", description);
            }
        }
        CuePackCmd::Validate { path } => {
            match cuemap::cuepacks::CuePackRegistry::validate_file(Path::new(&path)) {
                Ok(info) => {
                    println!("✓ CuePack is valid: {} {}", info.name, info.version);
                    println!(
                        "memory_rules={} query_rules={}",
                        info.memory_rules, info.query_rules
                    );
                }
                Err(err) => eprintln!("✗ {}", err),
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

    // Open log file for redirection
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .expect("Failed to open log file");

    // Clone file handles for stdout and stderr
    let stdout_file = log_file
        .try_clone()
        .expect("Failed to clone log file handle");
    let stderr_file = log_file
        .try_clone()
        .expect("Failed to clone log file handle");

    // Capture current size to start reading from
    let start_pos = log_file.metadata().map(|m| m.len()).unwrap_or(0);

    // Spawn the child
    let _child = std::process::Command::new(&exe)
        .args(&child_args[1..])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(stdout_file))
        .stderr(std::process::Stdio::from(stderr_file))
        .spawn()
        .expect("Failed to spawn background server");

    println!("✓ Background server spawning (PID: {})...", _child.id());
    println!("✓ Waiting for readiness sentinel in {}...", log_path);

    // Readiness sentinel we are looking for: "Unstable sorting for speed"
    let sentinel = "Unstable sorting for speed";
    let start_time = std::time::Instant::now();
    let timeout = Duration::from_secs(30);

    // Wait for logs to appear
    tokio::time::sleep(Duration::from_millis(100)).await;

    if let Ok(mut file) = File::open(&log_path) {
        let _ = file.seek(SeekFrom::Start(start_pos)); // Start tailing from the spawn time
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut found = false;

        while start_time.elapsed() < timeout {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Ok(_) => {
                    print!("{}", line);
                    if line.contains(sentinel) {
                        found = true;
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        if found {
            println!("\n✓ CueMap server is now running in the background.");
            println!("  - View logs:   cuemap logs --follow");
            println!("  - Stop server: cuemap stop");
        } else {
            eprintln!(
                "\n✗ Timeout waiting for server readiness. Check logs at: {}",
                log_path
            );
        }
    } else {
        eprintln!(
            "\n✗ Could not open log file to verify startup: {}",
            log_path
        );
    }
}

async fn handle_stop(_args: StopArgs) {
    let pid_path = config::get_base_dir().join("server.pid");
    if !pid_path.exists() {
        eprintln!("✗ No server.pid found. Server might not be running or wasn't started with this version.");
        return;
    }

    let pid_str = std::fs::read_to_string(&pid_path).expect("Failed to read PID file");
    let pid: u32 = pid_str.trim().parse().expect("Invalid PID in file");

    #[cfg(unix)]
    {
        use std::process::Command;
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
