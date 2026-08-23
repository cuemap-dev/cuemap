//! Authentication middleware for API key validation.
use crate::config::SecurityConfig;
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::collections::HashSet;
use std::env;
use tracing::info;

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
            let key = key.trim();
            if !key.is_empty() {
                api_keys.insert(key.to_string());
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
            info!(
                "Authentication enabled ({} API keys configured)",
                api_keys.len()
            );
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

    let api_key = headers.get("X-API-Key").and_then(|v| v.to_str().ok());

    match api_key {
        Some(key) if auth_config.validate_key(key) => Ok(next.run(request).await),
        Some(_) => Err((StatusCode::UNAUTHORIZED, "Invalid API key")),
        None => Err((StatusCode::UNAUTHORIZED, "Missing X-API-Key header")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        middleware,
        routing::get,
        http::{Request, StatusCode},
        Router,
    };
    use tower::ServiceExt;

    fn config(require_auth: bool, keys: &[&str]) -> SecurityConfig {
        SecurityConfig {
            require_auth,
            api_keys: keys.iter().map(|key| (*key).to_string()).collect(),
            ..SecurityConfig::default()
        }
    }

    #[test]
    fn disabled_auth_accepts_requests_and_filters_empty_keys() {
        let auth = AuthConfig::from_config(&config(false, &["", "  "]));
        assert!(!auth.is_enabled());
        assert!(auth.validate_key("anything"));
        assert!(auth.api_keys.is_empty());
    }

    #[test]
    fn configured_keys_enable_auth_and_validate_exact_values() {
        let auth = AuthConfig::from_config(&config(false, &["secret", "other"]));
        assert!(auth.is_enabled());
        assert!(auth.validate_key("secret"));
        assert!(!auth.validate_key("SECRET"));
        assert!(!auth.validate_key("missing"));
    }

    async fn test_router(auth: AuthConfig) -> Router {
        Router::new()
            .route("/", get(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(auth, auth_middleware))
    }

    #[tokio::test]
    async fn middleware_allows_disabled_auth() {
        let response = test_router(AuthConfig::from_config(&config(false, &[])))
            .await
            .oneshot(Request::new(Body::empty()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn middleware_rejects_missing_and_invalid_keys_and_allows_valid_key() {
        let router = test_router(AuthConfig::from_config(&config(true, &["secret"]))).await;

        let missing = router
            .clone()
            .oneshot(Request::new(Body::empty()))
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);

        let invalid = router
            .clone()
            .oneshot(
                Request::builder()
                    .header("X-API-Key", "wrong")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::UNAUTHORIZED);

        let valid = router
            .oneshot(
                Request::builder()
                    .header("X-API-Key", "secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(valid.status(), StatusCode::OK);
    }
}
