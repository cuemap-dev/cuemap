//! Authentication middleware for API key validation.
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::collections::HashSet;
use std::env;
use tracing::info;
use crate::config::SecurityConfig;

#[derive(Clone)]
pub struct AuthConfig {
    api_keys: HashSet<String>,
    require_auth: bool,
}

impl AuthConfig {
    pub fn new() -> Self {
        Self::from_config(&SecurityConfig::default())
    }

    pub fn from_config(config: &SecurityConfig) -> Self {
        let mut api_keys = HashSet::new();
        
        // Load keys from config
        for key in &config.api_keys {
             if !key.is_empty() {
                 api_keys.insert(key.clone());
             }
        }
        
        // Load API keys from environment (Migration/Compat)
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
        
        let require_auth = config.require_auth || !api_keys.is_empty();
        
        if require_auth {
            info!("Authentication enabled ({} API keys configured)", api_keys.len());
        } else {
            info!("Authentication disabled");
        }
        
        Self {
            api_keys,
            require_auth,
        }
    }
    
    pub fn is_enabled(&self) -> bool {
        self.require_auth
    }
    
    fn validate_key(&self, key: &str) -> bool {
        if !self.require_auth {
            return true;
        }
        
        self.api_keys.contains(key)
    }
}

/// Middleware to validate API keys
pub async fn auth_middleware(
    State(auth_config): State<AuthConfig>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, impl IntoResponse> {
    if !auth_config.require_auth {
        return Ok(next.run(request).await);
    }
    
    let api_key = headers
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok());
    
    match api_key {
        Some(key) if auth_config.validate_key(key) => {
            Ok(next.run(request).await)
        }
        Some(_) => {
            Err((
                StatusCode::UNAUTHORIZED,
                "Invalid API key"
            ))
        }
        None => {
            Err((
                StatusCode::UNAUTHORIZED,
                "Missing X-API-Key header"
            ))
        }
    }
}
