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
                let mime = mime_guess::from_path(path).first_or_octet_stream();
                Response::builder()
                    .header(header::CONTENT_TYPE, mime.as_ref())
                    .body(Body::from(content.data))
                    .unwrap()
            }
            None => {
                println!("Web UI 404: Requested '{}'.", path);
                println!("Asset Listing:");
                for file in Assets::iter() {
                    println!(" - {}", file.as_ref());
                }

                // SPA Fallback: Serve index.html for unknown routes (if it exists)
                if let Some(content) = Assets::get("index.html") {
                     println!("Serving fallack index.html");
                     Response::builder()
                        .header(header::CONTENT_TYPE, "text/html")
                        .body(Body::from(content.data))
                        .unwrap()
                } else {
                    println!("CRITICAL: index.html not found in assets!");
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
