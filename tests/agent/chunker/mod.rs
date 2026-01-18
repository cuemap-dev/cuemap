use cuemap_rust::agent::chunker::Chunker;
use std::path::PathBuf;

#[test]
fn test_csv_chunking() {
    let content = "id,name\n1,alice\n2,bob";
    let chunks = Chunker::chunk_csv(content);
    assert!(!chunks.is_empty());
    assert!(chunks[0].content.contains("alice"));
}

#[test]
fn test_json_chunking() {
    let content = "{\"key\": \"value\", \"list\": [1, 2]}";
    let chunks = Chunker::chunk_json(content);
    assert!(chunks.len() >= 2);
    assert!(chunks.iter().any(|c| c.context.contains("json_key:key")));
}

#[test]
fn test_yaml_chunking() {
    let content = "engine: cuemap\nversion: 0.5";
    let chunks = Chunker::chunk_yaml(content);
    assert!(!chunks.is_empty());
    assert!(chunks.iter().any(|c| c.content.contains("cuemap")));
}

#[test]
fn test_html_chunking() {
    let content = "<html><body><h1>Test</h1></body></html>";
    let chunks = Chunker::chunk_html(content);
    assert!(!chunks.is_empty());
    assert_eq!(chunks[0].context, "html:html");
}

#[test]
fn test_java_chunking() {
    let content = "public class Test { public void hello() {} }";
    let chunks = Chunker::chunk_java(content);
    assert!(!chunks.is_empty());
    assert_eq!(chunks[0].context, "class_declaration:Test");
}

#[test]
fn test_go_chunking() {
    let content = "package main\nfunc main() {}";
    let chunks = Chunker::chunk_go(content);
    assert!(!chunks.is_empty());
    assert_eq!(chunks[0].context, "function_declaration:main");
}

#[test]
fn test_php_chunking() {
    let content = "<?php function test() {} ?>";
    let chunks = Chunker::chunk_php(content);
    assert!(!chunks.is_empty());
    assert_eq!(chunks[0].context, "function_definition:test");
}

#[test]
fn test_css_chunking() {
    let content = ".selector { color: red; }";
    let chunks = Chunker::chunk_css(content);
    assert!(!chunks.is_empty());
    assert_eq!(chunks[0].context, "rule_set:.selector");
}

#[test]
fn test_detect_type() {
    use cuemap_rust::agent::chunker::ChunkerType;
    
    assert_eq!(Chunker::detect_type(&PathBuf::from("test.py")), ChunkerType::Python);
    assert_eq!(Chunker::detect_type(&PathBuf::from("test.csv")), ChunkerType::Csv);
    assert_eq!(Chunker::detect_type(&PathBuf::from("test.pdf")), ChunkerType::Pdf);
    assert_eq!(Chunker::detect_type(&PathBuf::from("test.docx")), ChunkerType::Office);
}
