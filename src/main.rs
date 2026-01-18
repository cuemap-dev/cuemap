
use cuemap_rust::auth::AuthConfig;
use cuemap_rust::config::CueGenStrategy;
use cuemap_rust::semantic::SemanticEngine;
use cuemap_rust::config;
use cuemap_rust::*;
use axum::Router;
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use std::path::Path;
use std::time::Duration;
use tower_http::cors::CorsLayer;
use tracing::{info, warn, error, Level};
use tracing_subscriber;

#[derive(Parser, Debug)]
#[command(name = "cuemap-rust")]
#[command(about = "CueMap Rust Engine - Production Memory Store")]
struct Args {
    /// Server port
    #[arg(short, long, default_value = "8080")]
    port: u16,
    
    /// Data directory for persistence
    #[arg(short, long, default_value = "./data")]
    data_dir: String,
    
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
}

#[tokio::main]
async fn main() {
    // Parse CLI arguments
    let args = Args::parse();
    
    // Initialize tracing with custom filter to silence nlprule
    let filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(Level::INFO.into())
        .add_directive("nlprule=warn".parse().unwrap());

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();
    
    info!("CueMap Rust Engine - Production Mode");
    
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
    
    // Initialize persistence (skip if static mode)
    // We still init persistence even if disable_snapshots is true, so we can load state.
    // We just won't start the background saver.

    
    // Initialize Semantic Engine (if using bundled data)
    let semantic_engine = SemanticEngine::new(Some(Path::new(&args.data_dir)));
    let cuegen_strategy = args.cuegen;

    
    // Build the router with appropriate engine state
    info!("Multi-tenant mode enabled");
    
    let snapshots_dir = if let Some(ref static_dir) = args.load_static {
        static_dir.clone()
    } else {
        format!("{}/snapshots", args.data_dir)
    };
    
    let mt_engine = Arc::new(multi_tenant::MultiTenantEngine::with_snapshots_dir(
        &snapshots_dir,
        cuegen_strategy.clone(),
        semantic_engine.clone(),
    ));
    
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
    

    
    let provider: Arc<dyn jobs::ProjectProvider> = mt_engine.clone();
    let job_queue = Arc::new(jobs::JobQueue::new(provider, args.disable_bg_jobs));
    
    let mt_engine = mt_engine;
    
    let app = Router::new()
        .merge(api::routes(mt_engine, job_queue, auth_config, is_static))
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
        match tokio::signal::ctrl_c().await {
            Ok(()) => {
                info!("Shutdown signal received, saving all projects...");
                
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
            }
            Err(err) => {
                warn!("Error setting up shutdown handler: {}", err);
            }
        }
    });
}


