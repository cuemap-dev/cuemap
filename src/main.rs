use cuemap::auth::AuthConfig;
use cuemap::config::CueGenStrategy;
use cuemap::semantic::SemanticEngine;
use cuemap::config;
use cuemap::*;
use axum::Router;
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use std::path::Path;
use std::time::Duration;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::fs::File;
use tower_http::cors::CorsLayer;
use tracing::{info, warn, error, Level};
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

    /// Manage aliases
    Alias(AliasArgs),

    /// Manage individual memories (get/reinforce/delete)
    Memories(MemoriesArgs),

    /// Context expansion (query expansion)
    Expand(ExpandArgs),

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
    /// Server URL
    #[arg(long, default_value = "http://localhost:8080")]
    url: String,
}

#[derive(Parser, Debug)]
struct StartArgs {
    /// Server port
    #[arg(short, long, default_value = "8080")]
    port: u16,
    
    /// Data directory for persistence
    #[arg(short, long, default_value = "./data")]
    data_dir: String,

    /// Assets directory (read-only models, taggers, defaults)
    #[arg(long)]
    assets_dir: Option<String>,
    
    /// Snapshot interval in seconds
    #[arg(short, long, default_value = "60")]
    snapshot_interval: u64,
    
    /// Load static snapshots (read-only mode, disables persistence)
    #[arg(long)]
    load_static: Option<String>,

    /// Directory to watch for Self-Learning Agent
    #[arg(long)]
    agent_dir: Option<String>,

    /// Agent throttle in milliseconds
    #[arg(long, default_value = "100")]
    agent_throttle: u64,

    /// Cue generation strategy
    #[arg(long, default_value = "default")]
    cuegen: CueGenStrategy,

    /// Disable background jobs (for benchmarking)
    #[arg(long, default_value = "false")]
    disable_bg_jobs: bool,

    /// Disable periodic snapshots (for benchmarking)
    #[arg(long, default_value = "false")]
    disable_snapshots: bool,

    /// Enable autonomous systems consolidation (daily job)
    #[arg(long, default_value = "false")]
    enable_consolidation: bool,

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
    #[arg(long, default_value = "cuemap/snapshots/")]
    cloud_prefix: String,

    /// Enable automatic cloud backup after each local save
    #[arg(long, default_value = "false")]
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
    /// Token budget for grounded recall (context window)
    #[arg(long, default_value = "500")]
    token_budget: u32,
    /// Disable automatic reinforcement during recall
    #[arg(long)]
    no_auto_reinforce: bool,
    /// Minimum cue intersection count
    #[arg(long)]
    min_intersection: Option<usize>,
    /// Disable pattern completion (self-learning refinement)
    #[arg(long)]
    disable_pattern_completion: bool,
    /// Disable salience bias (recency vs frequency weighting)
    #[arg(long)]
    disable_salience_bias: bool,
    /// Disable systems consolidation (long-term memory integration)
    #[arg(long)]
    disable_systems_consolidation: bool,
    /// Enable alias expansion (default: disabled)
    #[arg(long)]
    enable_alias_expansion: bool,
    /// Enable grounded recall (RAG context)
    #[arg(short, long)]
    grounded: bool,
    /// Include explanation
    #[arg(short, long)]
    explain: bool,
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
    /// Search lexicon
    Search {
        query: String,
        #[arg(short, long)]
        project: Option<String>,
    },
    /// Inspect a specific cue
    Inspect {
        cue: String,
        #[arg(short, long)]
        project: Option<String>,
    },
}

#[derive(Parser, Debug)]
struct MemoriesArgs {
    /// Memory ID
    id: String,
    
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
struct ExpandArgs {
    /// Text to expand
    text: String,
    /// Project ID
    #[arg(short, long)]
    project: Option<String>,
    /// Limit candidates
    #[arg(short, long, default_value = "5")]
    limit: usize,
    /// Minimum similarity score (0.0 to 1.0)
    #[arg(short, long)]
    min_score: Option<f64>,
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
}


#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Start(args) => {
            if args.detach && !args.child_process {
                handle_start_detached(args).await;
            } else {
                run_server(args).await;
            }
        },
        Commands::Add(args) => handle_add(args).await,
        Commands::Ingest(args) => handle_ingest(args).await,
        Commands::Recall(args) => handle_recall(args).await,
        Commands::Lexicon(args) => handle_lexicon(args).await,
        Commands::Memories(args) => handle_memories(args).await,
        Commands::Alias(args) => handle_alias(args).await,
        Commands::Expand(args) => handle_expand(args).await,
        Commands::Projects(args) => handle_projects(args).await,
        Commands::SetProject { project_id } => handle_set_project(project_id),
        Commands::Status(args) => handle_status(args).await,
        Commands::Logs(args) => handle_logs(args).await,
        Commands::Stop(args) => handle_stop(args).await,
    }
}

async fn run_server(args: StartArgs) {


    // Initialize tracing with custom filter to silence noisy components by default
    let filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(Level::INFO.into())
        .add_directive("nlprule=warn".parse().unwrap())
        .add_directive("cuemap::agent=warn".parse().unwrap())
        .add_directive("cuemap::jobs=warn".parse().unwrap())
        .add_directive("tiktoken_rs=warn".parse().unwrap());

    // Build layers
    let stdout_layer = fmt::layer().with_writer(std::io::stdout);

    Registry::default()
        .with(filter)
        .with(stdout_layer)
        .init();
    
    // Write PID file for the server
    let pid = std::process::id();
    let pid_path = get_config_dir().join("server.pid");
    if let Err(e) = std::fs::write(&pid_path, pid.to_string()) {
        warn!("Failed to write PID file: {}", e);
    }
    
    // We need to keep the guard alive for the duration of the program
    // but run_server is async and called at the end of main.
    // In our case, run_server only returns on shutdown, so it's fine.
    
    info!("CueMap Rust Engine - Production Mode");
    info!("Logs are written to stdout");
    
    // Initialize authentication
    let auth_config = AuthConfig::new();
    
    // Check for start mode
    let is_static = args.load_static.is_some();
    
    if is_static {
        info!("Static loading mode enabled (read-only)");
        info!("Loading from: {}", args.load_static.as_ref().unwrap());
        info!("Persistence disabled - all changes will be lost on restart");
    } else {
        info!("Data directory: {}", args.data_dir);
        if args.disable_snapshots {
            info!("Persistence: Snapshots DISABLED (benchmarking mode)");
        } else {
            info!("Snapshot interval: {}s", args.snapshot_interval);
        }
    }
    
    // Determine assets directory (defaults to data_dir if not set)
    let assets_path = args.assets_dir.clone().unwrap_or_else(|| args.data_dir.clone());
    info!("Assets directory: {}", assets_path);
    
    // Initialize Semantic Engine (if using bundled data)
    let semantic_engine = SemanticEngine::new(Some(Path::new(&assets_path)));
    let cuegen_strategy = args.cuegen;

    
use cuemap::crypto::EncryptionKey;

     // Build the router with appropriate engine state
    info!("Multi-tenant mode enabled");
    
    let snapshots_dir = if let Some(ref static_dir) = args.load_static {
        static_dir.clone()
    } else {
        format!("{}/snapshots", args.data_dir)
    };
    
    let mut mt_engine = multi_tenant::MultiTenantEngine::with_snapshots_dir(
        &snapshots_dir,
        cuegen_strategy.clone(),
        semantic_engine.clone(),
    );

    // Initialize Encryption Key
    if let Ok(key_hex) = std::env::var("CUEMAP_MASTER_KEY") {
        match hex::decode(&key_hex) {
            Ok(bytes) if bytes.len() == 32 => {
                info!("Security: Master key loaded from environment (hex)");
                mt_engine.set_master_key(Some(Arc::new(EncryptionKey::new(bytes))));
            }
            Ok(bytes) => {
                error!("Security: Invalid master key length (expected 32 bytes, got {})", bytes.len());
                std::process::exit(1);
            }
            Err(e) => {
                error!("Security: Invalid master key hex: {}", e);
                std::process::exit(1);
            }
        }
    } else if let Ok(passphrase) = std::env::var("CUEMAP_PASSPHRASE") {
        info!("Security: Deriving master key from passphrase");
        // Use a fixed salt for deterministic derivation across restarts
        // In production, this should ideally be configurable or random+stored,
        // but for this personal tool, a hardcoded salt ensures usability without extra config files.
        let salt = b"cuemap-secure-salt-v1";
        let key = EncryptionKey::from_passphrase(&passphrase, salt);
        mt_engine.set_master_key(Some(Arc::new(key)));
    } else {
        warn!("Security: No master key provided. Running in COMPRESSION-ONLY mode.");
        warn!("          To enable encryption, set CUEMAP_MASTER_KEY (hex) or CUEMAP_PASSPHRASE.");
    }
    
    let mt_engine = Arc::new(mt_engine);
    
    // Auto-load all available snapshots
    info!("Loading snapshots from: {}", snapshots_dir);
    let load_results = mt_engine.load_all();
    let loaded = load_results.iter().filter(|(_, r)| r.is_ok()).count();
    let failed = load_results.iter().filter(|(_, r)| r.is_err()).count();
    
    if loaded > 0 {
        info!("✓ Loaded {} project snapshots", loaded);
    }
    if failed > 0 {
        warn!("✗ Failed to load {} snapshots", failed);
        for (project_id, result) in load_results.iter() {
            if let Err(e) = result {
                warn!("  - {}: {}", project_id, e);
            }
        }
    }
    if loaded == 0 && failed == 0 {
        info!("No existing snapshots found, starting fresh");
    }
    
    // Setup shutdown handler for auto-save (skip if static mode)
    if !is_static {
        if !args.disable_snapshots {
            setup_multi_tenant_shutdown_handler(mt_engine.clone()).await;
            
            // Start periodic snapshots
            mt_engine.start_periodic_snapshots(Duration::from_secs(args.snapshot_interval));
        } else {
            warn!("Periodic snapshots and shutdown save are DISABLED (Multi-Tenant).");
        }
    }

    
    // Initialize metrics collector
    let metrics = Arc::new(cuemap::metrics::MetricsCollector::new());

    let provider: Arc<dyn jobs::ProjectProvider> = mt_engine.clone();
    let job_queue = Arc::new(jobs::JobQueue::new(provider, Some(metrics.clone()), args.disable_bg_jobs));

    // Start autonomous systems consolidation if enabled
    if args.enable_consolidation {
        let engine = mt_engine.clone();
        let queue = job_queue.clone(); 
        tokio::spawn(async move {
            info!("Systems Consolidation: Enabled (running daily)");
            let mut interval = tokio::time::interval(Duration::from_secs(86400)); // 24 hours
            // Skip immediate first tick
            interval.tick().await; 
            
            loop {
                interval.tick().await;
                info!("Systems Consolidation: Starting daily cycle...");
                
                let projects = engine.list_projects();
                for proj_stats in projects {
                    let job = jobs::Job::ConsolidateMemories { 
                        project_id: proj_stats.project_id.clone() 
                    };
                    // Enqueue job without blocking
                    queue.enqueue(job).await;
                }
            }
        });
    } else {
        info!("Systems Consolidation: Disabled (default)");
    }
    
    let mt_engine = mt_engine;
    
    // Initialize Self-Learning Agent if requested
    let _agent = if let Some(watch_dir) = args.agent_dir {
        let agent_config = agent::AgentConfig {
            watch_dir,
            throttle_ms: args.agent_throttle,
            state_file: Some(std::path::PathBuf::from(&args.data_dir).join("agent_state.json")),
        };
        
        match agent::Agent::new(agent_config, job_queue.clone(), mt_engine.clone()) {
            Ok(agent) => {
                info!("Self-Learning Agent: Initializing...");
                agent.start().await;
                Some(agent)
            }
            Err(e) => {
                error!("Failed to start Self-Learning Agent: {}", e);
                None
            }
        }
    } else {
        info!("Self-Learning Agent: Disabled (no --agent-dir provided)");
        None
    };
    
    // Metrics collector already initialized above
    
    // Initialize cloud backup manager if configured
    let cloud_backup: Option<Arc<persistence::CloudBackupManager>> = if args.cloud_backup.is_some() {
        match persistence::CloudBackupConfig::from_args(
            args.cloud_backup.as_deref(),
            args.cloud_bucket.as_deref(),
            args.cloud_region.as_deref(),
            args.cloud_endpoint.as_deref(),
            &args.cloud_prefix,
            args.cloud_auto_backup,
        ) {
            Ok(config) if config.enabled => {
                match persistence::CloudBackupManager::new(config).await {
                    Ok(manager) => {
                        info!("Cloud backup: Enabled");
                        Some(Arc::new(manager))
                    }
                    Err(e) => {
                        error!("Failed to initialize cloud backup: {}", e);
                        None
                    }
                }
            }
            Ok(_) => {
                info!("Cloud backup: Disabled");
                None
            }
            Err(e) => {
                error!("Invalid cloud backup configuration: {}", e);
                None
            }
        }
    } else {
        info!("Cloud backup: Not configured");
        None
    };
    
    let app = Router::new()
        .merge(api::routes(mt_engine, job_queue, metrics, auth_config, is_static, cloud_backup))
        .layer(CorsLayer::permissive());


    
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    info!("Server listening on {}", addr);
    info!("Performance optimizations enabled:");
    info!("   - IndexSet for O(1) operations");
    info!("   - DashMap with {} shards", config::DASHMAP_SHARD_COUNT);
    info!("   - Pre-allocated collections");
    info!("   - Unstable sorting for speed");
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    
    // Add graceful shutdown
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

fn get_config_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let path = std::path::PathBuf::from(home).join(".cuemap");
    if !path.exists() {
        let _ = std::fs::create_dir_all(&path);
    }
    path
}

fn get_default_project() -> Option<String> {
    let config_path = get_config_dir().join("config.json");
    if let Ok(content) = std::fs::read_to_string(config_path) {
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
            return config.get("default_project").and_then(|v| v.as_str().map(|s| s.to_string()));
        }
    }
    None
}

fn handle_set_project(project_id: String) {
    let config_path = get_config_dir().join("config.json");
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

async fn handle_add(args: AddArgs) {
    let project = args.project.or_else(get_default_project).expect("Project ID required (use --project or set-project)");
    let client = reqwest::Client::new();
    
    let payload = api::AddMemoryRequest {
        content: args.content,
        metadata: args.metadata.map(|m| serde_json::from_str(&m).unwrap_or_default()),
        cues: args.cues,
        disable_temporal_chunking: args.disable_temporal_chunking,
        async_ingest: args.async_ingest,
    };

    let res = client.post(format!("{}/memories", args.url))
        .header("X-Project-ID", project)
        .json(&payload)
        .send()
        .await;

    match res {
        Ok(response) => {
            if response.status().is_success() {
                let body: serde_json::Value = response.json().await.unwrap();
                println!("✓ Memory added: {}", body.get("id").and_then(|v| v.as_str()).unwrap_or("unknown"));
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
        IngestType::File { path, project } => {
            let project = project.or_else(get_default_project).expect("Project ID required");
            if let Ok(content) = std::fs::read_to_string(&path) {
                let res = client.post("http://localhost:8080/ingest/content")
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
        IngestType::Url { url, project, depth, same_domain_only } => {
            let project = project.or_else(get_default_project).expect("Project ID required");
            let res = client.post("http://localhost:8080/ingest/url")
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
    let project = args.project.or_else(get_default_project).expect("Project ID required");
    let client = reqwest::Client::new();
    
    if args.grounded {
        let payload = api::RecallGroundedRequest {
            query_text: args.query,
            limit: args.limit,
            token_budget: args.token_budget,
            auto_reinforce: !args.no_auto_reinforce,
            projects: None,
            disable_pattern_completion: args.disable_pattern_completion,
            disable_salience_bias: args.disable_salience_bias,
            disable_systems_consolidation: args.disable_systems_consolidation,
            min_intersection: args.min_intersection,
            disable_alias_expansion: !args.enable_alias_expansion,
        };
        let res = client.post(format!("{}/recall/grounded", args.url))
            .header("X-Project-ID", project)
            .json(&payload)
            .send()
            .await;
        
        match res {
            Ok(r) if r.status().is_success() => {
                let body: api::RecallGroundedResponse = r.json().await.unwrap();
                println!("\n--- GROUNDED RECALL ---");
                println!("{}", body.verified_context);
            },
            Ok(r) => eprintln!("✗ Error: {}", r.text().await.unwrap_or_default()),
            Err(e) => eprintln!("✗ Failed: {}", e),
        }
    } else if args.web {
        let payload = api::RecallWebRequest {
            url: args.target_url,
            query: args.query,
            persist: args.persist,
        };
        let res = client.post(format!("{}/recall/web", args.url))
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
                    println!("- [{:.4}] [Intersection: {}] {}", 
                        mem.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0), 
                        mem.get("intersection").and_then(|v| v.as_u64()).unwrap_or(0),
                        mem.get("content").and_then(|v| v.as_str()).unwrap_or("")
                    );
                }
            },
            Ok(r) => eprintln!("✗ Error: {}", r.text().await.unwrap_or_default()),
            Err(e) => eprintln!("✗ Failed: {}", e),
        }
    } else {
        let payload = api::RecallRequest {
            cues: args.cues,
            query_text: Some(args.query),
            limit: args.limit,
            auto_reinforce: !args.no_auto_reinforce,
            explain: args.explain,
            projects: None,
            min_intersection: args.min_intersection,
            disable_pattern_completion: args.disable_pattern_completion,
            disable_salience_bias: args.disable_salience_bias,
            disable_systems_consolidation: args.disable_systems_consolidation,
            disable_alias_expansion: !args.enable_alias_expansion,
        };
        let res = client.post(format!("{}/recall", args.url))
            .header("X-Project-ID", project)
            .json(&payload)
            .send()
            .await;
        
        match res {
            Ok(r) if r.status().is_success() => {
                let response_body: serde_json::Value = r.json().await.unwrap();
                let results = response_body.get("results").and_then(|v| v.as_array()).unwrap();
                println!("\n--- RECALL RESULTS ({}) ---", results.len());
                for mem in results {
                    println!("- [{:.4}] [{}] {}", 
                        mem.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0), 
                        mem.get("memory_id").and_then(|v| v.as_str()).unwrap_or("unknown"),
                        mem.get("content").and_then(|v| v.as_str()).unwrap_or("")
                    );
                }
            },
            Ok(r) => eprintln!("✗ Error: {}", r.text().await.unwrap_or_default()),
            Err(e) => eprintln!("✗ Failed: {}", e),
        }
    }
}

async fn handle_lexicon(args: LexiconArgs) {
    let client = reqwest::Client::new();
    
    match args.cmd {
        LexiconCmd::Search { query, project } => {
            let project_id = project.or_else(get_default_project).expect("Project ID required");
            let res = client.get(format!("http://localhost:8080/lexicon/search?q={}", query))
                .header("X-Project-ID", project_id)
                .send()
                .await;
            match res {
                Ok(r) if r.status().is_success() => {
                    let entries: Vec<api::LexiconEntry> = r.json().await.unwrap();
                    println!("\n--- LEXICON SEARCH RESULTS ---");
                    for entry in entries {
                        println!("- {}: score={:.4}", entry.token, entry.reinforcement_score);
                    }
                }
                _ => eprintln!("✗ Search failed"),
            }
        }
        LexiconCmd::Inspect { cue, project } => {
            let project_id = project.or_else(get_default_project).expect("Project ID required");
            let res = client.get(format!("http://localhost:8080/lexicon/inspect/{}", cue))
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
    let project = args.project.or_else(get_default_project).expect("Project ID required");
    let client = reqwest::Client::new();
    
    if args.delete {
        let res = client.delete(format!("{}/memories/{}", args.url, args.id))
            .header("X-Project-ID", project)
            .send()
            .await;
            
        match res {
            Ok(r) if r.status().is_success() => println!("✓ Memory deleted: {}", args.id),
            Ok(r) if r.status() == 404 => eprintln!("✗ Memory not found: {}", args.id),
            _ => eprintln!("✗ Failed to delete memory"),
        }
    } else if args.reinforce {
        let payload = api::ReinforceRequest {
            cues: args.cues,
        };
        let res = client.patch(format!("{}/memories/{}/reinforce", args.url, args.id))
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
        let res = client.get(format!("{}/memories/{}", args.url, args.id))
            .header("X-Project-ID", project)
            .send()
            .await;
            
        match res {
            Ok(r) if r.status().is_success() => {
                let mem: serde_json::Value = r.json().await.unwrap();
                println!("\n--- MEMORY: {} ---", args.id);
                println!("Content: {}", mem.get("content").and_then(|v| v.as_str()).unwrap_or(""));
                println!("Created: {}", mem.get("created_at").and_then(|v| v.as_f64()).unwrap_or(0.0));
                
                if let Some(cues) = mem.get("cues").and_then(|v| v.as_array()) {
                    println!("Cues: {:?}", cues);
                }
                
                if let Some(stats) = mem.get("stats") {
                    println!("Stats: {}", serde_json::to_string_pretty(stats).unwrap());
                }
            },
            Ok(r) if r.status() == 404 => eprintln!("✗ Memory not found: {}", args.id),
            _ => eprintln!("✗ Failed to get memory"),
        }
    }
}

async fn handle_alias(args: AliasArgs) {
    let project = args.project.or_else(get_default_project).expect("Project ID required");
    let client = reqwest::Client::new();
    
    if let Some(alias) = args.add {
        let payload = api::AddAliasRequest {
            from: args.text,
            to: alias,
            weight: args.weight,
        };
        let res = client.post(format!("{}/aliases", args.url))
            .header("X-Project-ID", project)
            .json(&payload)
            .send()
            .await;
        match res {
            Ok(r) if r.status().is_success() => println!("✓ Alias added"),
            _ => eprintln!("✗ Failed to add alias"),
        }
    } else {
        let res = client.get(format!("{}/aliases?q={}", args.url, args.text))
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

async fn handle_expand(args: ExpandArgs) {
    let project = args.project.or_else(get_default_project).expect("Project ID required");
    let client = reqwest::Client::new();
    
    let payload = api::ContextExpandRequest {
        query: args.text,
        limit: args.limit,
        min_score: args.min_score,
    };
    
    let res = client.post(format!("{}/context/expand", args.url))
        .header("X-Project-ID", project)
        .json(&payload)
        .send()
        .await;
    
    match res {
        Ok(r) if r.status().is_success() => {
            let body: api::ContextExpandResponse = r.json().await.unwrap();
            println!("\n--- CONTEXT EXPANSION ---");
            for cand in body.expansions {
                println!("- {}: score={:.4}", cand.term, cand.score);
            }
        }
        _ => eprintln!("✗ Expansion failed"),
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
                        println!("- {} (memories: {})", p.get("project_id").unwrap(), p.get("total_memories").unwrap());
                    }
                }
                _ => eprintln!("✗ Failed to list projects"),
            }
        }
        ProjectCmd::Create { name, url } => {
            let res = client.post(format!("{}/projects", url))
                .json(&api::CreateProjectRequest { project_id: name.clone() })
                .send()
                .await;
            match res {
                Ok(r) if r.status().is_success() => println!("✓ Project created: {}", name),
                _ => eprintln!("✗ Failed to create project"),
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
                println!("\n--- SERVER STATS ---");
                println!("{}", serde_json::to_string_pretty(&stats).unwrap());
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
                println!("\n--- JOB STATUS ---");
                println!("{}", serde_json::to_string_pretty(&status).unwrap());
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
        let mut path = get_config_dir();
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
    let mut child_args: Vec<String> = std::env::args().filter(|a| a != "--detach" && a != "-d").collect();
    child_args.push("--child-process".to_string());

    // Fix: Make sure the first argument is just the binary name if it's the full path
    let exe = std::env::current_exe().expect("Failed to get current executable");
    
    // Determine log file path
    let log_path = args.log_file.clone().unwrap_or_else(|| {
        let mut path = get_config_dir();
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
    let stdout_file = log_file.try_clone().expect("Failed to clone log file handle");
    let stderr_file = log_file.try_clone().expect("Failed to clone log file handle");

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
            eprintln!("\n✗ Timeout waiting for server readiness. Check logs at: {}", log_path);
        }
    } else {
        eprintln!("\n✗ Could not open log file to verify startup: {}", log_path);
    }
}

async fn handle_stop(_args: StopArgs) {
    let pid_path = get_config_dir().join("server.pid");
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
            _ => eprintln!("✗ Failed to kill process {}. It might have already exited.", pid),
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
            _ => eprintln!("✗ Failed to kill process {}. It might have already exited.", pid),
        }
    }
}
