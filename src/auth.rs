//! Authentication middleware for API key validation.

use axum::{
    extract::Request,
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use std::collections::HashSet;
use std::env;
use tracing::{debug, warn};

#[derive(Clone)]
pub struct AuthConfig {
    pub api_keys: HashSet<String>,
    pub require_auth: bool,
}

impl AuthConfig {
    pub fn new() -> Self {
        let mut api_keys = HashSet::new();
        
        // Load API keys from environment
        if let Ok(keys_str) = env::var("CUEMAP_API_KEYS") {
            for key in keys_str.split(',') {
                let key = key.trim();
                if !key.is_empty() {
                    api_keys.insert(key.to_string());
                }
            }
        }
        
        // Single API key support
        if let Ok(key) = env::var("CUEMAP_API_KEY") {
            let key = key.trim();
            if !key.is_empty() {
                api_keys.insert(key.to_string());
            }
        }
        
        let require_auth = !api_keys.is_empty() || env::var("CUEMAP_REQUIRE_AUTH").is_ok();
        
        if require_auth {
            println!("🔐 Authentication enabled ({} API keys configured)", api_keys.len());
        } else {
            println!("⚠️  Authentication disabled (no API keys configured)");
        }
        
        Self {
            api_keys,
            require_auth,
        }
    }
    
    pub fn validate_key(&self, key: &str) -> bool {
        if !self.require_auth {
            return true;
        }
        
        self.api_keys.contains(key)
    }
}
