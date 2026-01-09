use cuemap_rust::api::EngineState;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

#[tokio::test]
async fn test_signed_memories() {
    // Setup
    let (project, job_queue) = crate::setup_test_project();
    
    // Add memory
    project.main.add_memory("Crucial financial data here".to_string(), vec!["finance".to_string()], None, false);
    
    let app = cuemap_rust::api::routes(project, job_queue, cuemap_rust::auth::AuthConfig::new(), false);

    // Call recall/grounded
    let payload = serde_json::json!({
        "query_text": "finance data",
        "token_budget": 100
    });
    
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/recall/grounded")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    
    // Check signature presence
    let signature = json.get("signature").expect("Signature missing").as_str().unwrap();
    let context = json.get("verified_context").expect("Context missing").as_str().unwrap();
    
    // Verify locally
    let signer = cuemap_rust::crypto::CryptoEngine::new();
    assert!(signer.verify(context, signature));
    
    // Verify tampering fails
    assert!(!signer.verify("Tampered context", signature));
}
