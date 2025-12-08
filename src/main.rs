mod structures;
mod engine;
mod api;
mod config;
mod persistence;
mod auth;
mod multi_tenant;

use axum::Router;
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::{info, Level};
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
    
    /// Enable multi-tenancy
    #[arg(short, long, default_value = "false")]
    multi_tenant: bool,
}

#[tokio::main]
async fn main() {
    // Parse CLI arguments
    let args = Args::parse();
    
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();
    
    info!("🚀 CueMap Rust Engine - Production Mode");
    info!("📁 Data directory: {}", args.data_dir);
    info!("⏱️  Snapshot interval: {}s", args.snapshot_interval);
    
    // Initialize persistence
    let persistence = persistence::PersistenceManager::new(&args.data_dir, args.snapshot_interval);
    
    // Initialize engine
    let engine = if args.multi_tenant {
        info!("🏢 Multi-tenant mode enabled");
        // Multi-tenant engine will be created per-project
        Arc::new(engine::CueMapEngine::new())
    } else {
        info!("📦 Single-tenant mode");
        // Load existing state
        match persistence.load_state() {
            Ok((memories, cue_index)) => {
                info!("✅ Loaded {} memories, {} cues", memories.len(), cue_index.len());
                Arc::new(engine::CueMapEngine::from_state(memories, cue_index))
            }
            Err(e) => {
                info!("⚠️  Failed to load state: {}, starting fresh", e);
                Arc::new(engine::CueMapEngine::new())
            }
        }
    };
    
    // Start background snapshots
    let snapshot_handle = persistence.start_background_snapshots(engine.clone()).await;
    
    // Setup graceful shutdown
    persistence::setup_shutdown_handler(persistence.clone(), engine.clone()).await;
    
    // Build the router
    let app = Router::new()
        .merge(api::routes(engine, args.multi_tenant))
        .layer(CorsLayer::permissive());
    
    // Start the server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    info!("🌐 Server listening on {}", addr);
    info!("✨ Performance optimizations enabled:");
    info!("   - IndexSet for O(1) operations");
    info!("   - DashMap with {} shards", config::DASHMAP_SHARD_COUNT);
    info!("   - Pre-allocated collections");
    info!("   - Unstable sorting for speed");
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
