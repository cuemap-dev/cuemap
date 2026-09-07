pub fn retrieve(query: &str) -> Vec<String> {
    query.split_whitespace().map(str::to_lowercase).collect()
}
