use scraper::{Html, Selector};
use reqwest::Client;

/// Search DuckDuckGo Lite and return top N result URLs
pub async fn search_ddg_lite(query: &str, limit: usize) -> Result<Vec<String>, String> {
    let client = Client::builder()
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .build()
        .map_err(|e| format!("Failed to build client: {}", e))?;

    let url = "https://lite.duckduckgo.com/lite/";
    
    // Mimic the exact form data from user's curl
    let params = [
        ("q", query),
        ("kl", "us-en"),
        ("df", "") // date filter empty
    ];

    let response = client.post(url)
        .header("Accept", "text/html,application/xhtml+xml2,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8")
        .header("Accept-Language", "en-US,en;q=0.9")
        .header("Origin", "https://lite.duckduckgo.com")
        .header("Referer", "https://lite.duckduckgo.com/")
        .header("Cookie", "kl=us-en")
        .form(&params)
        .send()
        .await
        .map_err(|e| format!("Search request failed: {}", e))?;

    let html = response.text().await
        .map_err(|e| format!("Failed to read search response: {}", e))?;
    
    // Debug logging
    if html.len() < 500 {
        tracing::warn!("DDG Lite returned short response : {}", html);
    }

    let document = Html::parse_document(&html);
    
    // DDG Lite structure: The anchor tag itself has class 'result-link'
    let link_selector = Selector::parse(".result-link").unwrap();
    
    let mut results = Vec::new();
    
    for element in document.select(&link_selector) {
        if results.len() >= limit {
            break;
        }
        
        if let Some(href) = element.value().attr("href") {
            // DDG Lite links need decoding or sometimes are direct
            // They look like: /l/?kh=-1&uddg=https%3A%2F%2Fexample.com%2F...
            // or sometimes direct links depending on user agent?
            // Actually usually plain links in Lite version but let's check.
            
            let clean_url = href.to_string();
            // Basic filtering of internal DDG links
            if clean_url.starts_with("http") && !clean_url.contains("duckduckgo.com") {
                 results.push(clean_url);
            }
        }
    }
    
    Ok(results)
}
