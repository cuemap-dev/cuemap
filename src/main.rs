mod structures;
mod engine;
mod api;
mod config;
mod persistence;
mod auth;
mod multi_tenant;

use auth::AuthConfig;
use axum::Router;
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use std::path::Path;
use tower_http::cors::CorsLayer;
use tracing::{info, warn, Level};
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
    
    /// Load static snapshots (read-only mode, disables persistence)
    #[arg(long)]
    load_static: Option<String>,
}

#[tokio::main]
async fn main() {
    // Parse CLI arguments
    let args = Args::parse();
    
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();
    
    info!("CueMap Rust Engine - Production Mode");
    
    // Initialize authentication
    let auth_config = AuthConfig::new();
    
    // Check for static loading mode
    let is_static = args.load_static.is_some();
    
    if is_static {
        info!("Static loading mode enabled (read-only)");
        info!("Loading from: {}", args.load_static.as_ref().unwrap());
        info!("Persistence disabled - all changes will be lost on restart");
    } else {
        info!("Data directory: {}", args.data_dir);
        info!("Snapshot interval: {}s", args.snapshot_interval);
    }
    
    // Initialize persistence (skip if static mode)
    let persistence = if !is_static {
        Some(persistence::PersistenceManager::new(&args.data_dir, args.snapshot_interval))
    } else {
        None
    };
    

    
    // Initialize engine for single-tenant mode
    let engine = if !args.multi_tenant {
        info!("Single-tenant mode");
        
        if let Some(ref static_dir) = args.load_static {
            // Load from static directory
            let snapshot_path = Path::new(static_dir).join("cuemap.bin");
            if snapshot_path.exists() {
                info!("Loading static snapshot from: {:?}", snapshot_path);
                match persistence::PersistenceManager::load_from_path(&snapshot_path) {
                    Ok((memories, cue_index)) => {
                        info!("Loaded {} memories, {} cues", memories.len(), cue_index.len());
                        Arc::new(engine::CueMapEngine::from_state(memories, cue_index))
                    }
                    Err(e) => {
                        warn!("Failed to load static snapshot: {}, starting fresh", e);
                        Arc::new(engine::CueMapEngine::new())
                    }
                }
            } else {
                warn!("No snapshot found at {:?}, starting fresh", snapshot_path);
                Arc::new(engine::CueMapEngine::new())
            }
        } else if let Some(ref pm) = persistence {
            // Load from data directory
            match pm.load_state() {
                Ok((memories, cue_index)) => {
                    info!("Loaded {} memories, {} cues", memories.len(), cue_index.len());
                    Arc::new(engine::CueMapEngine::from_state(memories, cue_index))
                }
                Err(e) => {
                    info!("Failed to load state: {}, starting fresh", e);
                    Arc::new(engine::CueMapEngine::new())
                }
            }
        } else {
            Arc::new(engine::CueMapEngine::new())
        }
    } else {
        // Not used in multi-tenant mode
        Arc::new(engine::CueMapEngine::new())
    };
    
    // Start background snapshots (skip if static mode)
    if let Some(ref pm) = persistence {
        if !args.multi_tenant {
            let _snapshot_handle = pm.start_background_snapshots(engine.clone()).await;
            persistence::setup_shutdown_handler(pm.clone(), engine.clone()).await;
        }
    }
    
    // Build the router with appropriate engine state
    let app = if args.multi_tenant {
        info!("Multi-tenant mode enabled");
        
        let snapshots_dir = if let Some(ref static_dir) = args.load_static {
            static_dir.clone()
        } else {
            format!("{}/snapshots", args.data_dir)
        };
        
        let mt_engine = Arc::new(multi_tenant::MultiTenantEngine::with_snapshots_dir(&snapshots_dir));
        
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
            setup_multi_tenant_shutdown_handler(mt_engine.clone()).await;
        }
        
        let mt_engine = mt_engine;
        
        Router::new()
            .merge(api::routes_with_mt_engine(mt_engine, auth_config, is_static))
            .layer(CorsLayer::permissive())
    } else {
        Router::new()
            .merge(api::routes(engine, auth_config, is_static))
            .layer(CorsLayer::permissive())
    };
    
    // Start the server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    info!("Server listening on {}", addr);
    info!("Performance optimizations enabled:");
    info!("   - IndexSet for O(1) operations");
    info!("   - DashMap with {} shards", config::DASHMAP_SHARD_COUNT);
    info!("   - Pre-allocated collections");
    info!("   - Unstable sorting for speed");
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Setup shutdown handler for multi-tenant mode
async fn setup_multi_tenant_shutdown_handler(mt_engine: Arc<multi_tenant::MultiTenantEngine>) {
    tokio::spawn(async move {
        match tokio::signal::ctrl_c().await {
            Ok(()) => {
                info!("Shutdown signal received, saving all projects...");
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
                
                info!("Shutdown complete");
                std::process::exit(0);
            }
            Err(err) => {
                warn!("Error setting up shutdown handler: {}", err);
            }
        }
    });
}


