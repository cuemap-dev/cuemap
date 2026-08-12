use reqwest::Client;
use scraper::{Html, Selector};

fn parse_ddg_lite_results(html: &str, limit: usize) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }

    let document = Html::parse_document(html);
    // DDG Lite structure: the anchor tag itself has class 'result-link'.
    let link_selector = Selector::parse(".result-link").expect("static selector is valid");

    document
        .select(&link_selector)
        .filter_map(|element| element.value().attr("href"))
        .filter(|href| href.starts_with("http") && !href.contains("duckduckgo.com"))
        .take(limit)
        .map(str::to_owned)
        .collect()
}

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
        ("df", ""), // date filter empty
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

    let html = response
        .text()
        .await
        .map_err(|e| format!("Failed to read search response: {}", e))?;

    // Debug logging
    if html.len() < 500 {
        tracing::warn!("DDG Lite returned short response : {}", html);
    }

    Ok(parse_ddg_lite_results(&html, limit))
}

#[cfg(test)]
mod tests {
    use super::parse_ddg_lite_results;

    #[test]
    fn parses_external_links_filters_search_links_and_honors_limit() {
        let html = r#"
            <a class="result-link" href="https://example.com/one">one</a>
            <a class="result-link" href="https://duckduckgo.com/about">internal</a>
            <a class="result-link" href="/l/?uddg=https%3A%2F%2Fredirect.example%2Ftwo">redirect</a>
            <a class="result-link" href="https://example.com/three">three</a>
        "#;

        assert_eq!(
            parse_ddg_lite_results(html, 10),
            vec!["https://example.com/one", "https://example.com/three"]
        );
        assert_eq!(parse_ddg_lite_results(html, 1), vec!["https://example.com/one"]);
        assert!(parse_ddg_lite_results(html, 0).is_empty());
    }
}
