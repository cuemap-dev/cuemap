#[cfg(feature = "ui")]
use rust_embed::RustEmbed;
use axum::{
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
    body::Body,
};

#[cfg(feature = "ui")]
#[derive(RustEmbed)]
#[folder = "web_ui/dist"]
pub struct Assets;

#[cfg(not(feature = "ui"))]
pub struct Assets;

#[cfg(not(feature = "ui"))]
pub struct EmbeddedFile {
    pub data: std::borrow::Cow<'static, [u8]>,
}

#[cfg(not(feature = "ui"))]
impl Assets {
    pub fn get(_path: &str) -> Option<EmbeddedFile> {
        None
    }
    
    pub fn iter() -> impl Iterator<Item = std::borrow::Cow<'static, str>> {
        std::iter::empty()
    }
}

pub struct StaticFile<T>(pub T);

impl<T> IntoResponse for StaticFile<T>
where
    T: Into<String>,
{
    fn into_response(self) -> Response {
        let path = self.0.into();
        
        match Assets::get(path.as_str()) {
            Some(content) => {
                let mime = mime_guess::from_path(&path).first_or_octet_stream();
                Response::builder()
                    .header(header::CONTENT_TYPE, mime.as_ref())
                    .body(Body::from(content.data))
                    .unwrap()
            }
            None => {
                // FALLBACK: Try to serve from filesystem
                // 1. Check DATA_DIR (e.g. Volume Override)
                let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "data".to_string());
                let public_path = std::path::Path::new(&data_dir).join("public").join(&path);
                
                if public_path.exists() && public_path.is_file() {
                     // println!("Serving {} from filesystem (DATA_DIR)", path);
                     let mime = mime_guess::from_path(&path).first_or_octet_stream();
                     if let Ok(content) = std::fs::read(&public_path) {
                         return Response::builder()
                            .header(header::CONTENT_TYPE, mime.as_ref())
                            .body(Body::from(content))
                            .unwrap();
                     }
                }

                // 2. Check ASSETS_DIR (e.g. Docker Image Defaults)
                let assets_dir = std::env::var("ASSETS_DIR").unwrap_or_else(|_| data_dir.clone());
                let asset_public_path = std::path::Path::new(&assets_dir).join("public").join(&path);

                if asset_public_path.exists() && asset_public_path.is_file() {
                     // println!("Serving {} from filesystem (ASSETS_DIR)", path);
                     let mime = mime_guess::from_path(&path).first_or_octet_stream();
                     if let Ok(content) = std::fs::read(&asset_public_path) {
                         return Response::builder()
                            .header(header::CONTENT_TYPE, mime.as_ref())
                            .body(Body::from(content))
                            .unwrap();
                     }
                }

                // If not found on fs either, show 404 or index fallback
                
                // SPA Fallback: Serve index.html for unknown routes
                // Check embedded first
                if let Some(content) = Assets::get("index.html") {
                     // println!("Serving fallback index.html (embedded)");
                     Response::builder()
                        .header(header::CONTENT_TYPE, "text/html")
                        .body(Body::from(content.data))
                        .unwrap()
                } else {
                    // Check filesystem index.html in DATA_DIR
                    let index_path = std::path::Path::new(&data_dir).join("public").join("index.html");
                    if index_path.exists() {
                         // println!("Serving fallback index.html (DATA_DIR)");
                         if let Ok(content) = std::fs::read(&index_path) {
                            return Response::builder()
                                .header(header::CONTENT_TYPE, "text/html")
                                .body(Body::from(content))
                                .unwrap();
                         }
                    }

                    // Check filesystem index.html in ASSETS_DIR
                    let assets_dir = std::env::var("ASSETS_DIR").unwrap_or_else(|_| data_dir.clone());
                    let asset_index_path = std::path::Path::new(&assets_dir).join("public").join("index.html");
                    if asset_index_path.exists() {
                         // println!("Serving fallback index.html (ASSETS_DIR)");
                         if let Ok(content) = std::fs::read(&asset_index_path) {
                            return Response::builder()
                                .header(header::CONTENT_TYPE, "text/html")
                                .body(Body::from(content))
                                .unwrap();
                         }
                    }

                    println!("Web UI 404: Requested '{}'.", path);
                    println!("CRITICAL: index.html not found in assets, {}/public, or {}/public!", data_dir, assets_dir);
                    Response::builder()
                        .status(StatusCode::NOT_FOUND)
                        .body(Body::from("404 Not Found - index.html missing"))
                        .unwrap()
                }
            }
        }
    }
}

pub async fn handler(uri: Uri) -> impl IntoResponse {
    let url_path = uri.path();
    println!("Web Handler: Processing '{}'", url_path);
    let mut path = url_path.trim_start_matches('/').to_string();
    if path.starts_with("ui/") {
        path = path.replace("ui/", "");
    }
    if path.is_empty() || path == "ui" {
        path = "index.html".to_string();
    }
    StaticFile(path)
}
