use cuemap::agent::chunker::{Chunker, SegmenterConfig};
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
    use cuemap::agent::chunker::ChunkerType;

    assert_eq!(
        Chunker::detect_type(&PathBuf::from("test.py")),
        Some(ChunkerType::Python)
    );
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("test.csv")),
        Some(ChunkerType::Csv)
    );
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("test.pdf")),
        Some(ChunkerType::Pdf)
    );
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("test.docx")),
        Some(ChunkerType::Office)
    );
}

#[test]
fn logical_block_chunking_keeps_paragraph_and_list_structure() {
    let content = "\
Intro paragraph with the project context. It has a second sentence.

### Plan
1. Add the translation API integration.
2. Add cache invalidation.
3. Measure latency after rollout.

```python
print('keep code together')
```

Final paragraph after the code block.";
    let config = SegmenterConfig {
        window_size: 8,
        overlap: 0,
        min_chunk_chars: 20,
        max_chunk_chars: 4000,
    };

    let chunks = Chunker::chunk_text_logical_blocks(content, &config);

    assert!(chunks.len() <= 4, "too many chunks: {}", chunks.len());
    assert!(chunks
        .iter()
        .any(|chunk| chunk.content.contains("translation API integration")
            && chunk.content.contains("Measure latency")));
    assert!(chunks
        .iter()
        .any(|chunk| chunk.content.contains("print('keep code together')")));
    assert!(chunks
        .iter()
        .all(|chunk| chunk.structural_cues.iter().any(|cue| cue == "type:logical_block")));
}

#[test]
fn logical_block_chunking_splits_oversized_blocks_with_coarse_windows() {
    let content = "One sentence about language detection. Two sentences about translation. Three sentences about latency. Four sentences about caching. Five sentences about rollout. Six sentences about monitoring. Seven sentences about incidents. Eight sentences about dashboards. Nine sentences about ownership.";
    let config = SegmenterConfig {
        window_size: 4,
        overlap: 0,
        min_chunk_chars: 20,
        max_chunk_chars: 100,
    };

    let chunks = Chunker::chunk_text_logical_blocks(content, &config);

    assert!(chunks.len() > 1);
    assert!(chunks
        .iter()
        .any(|chunk| chunk.structural_cues.iter().any(|cue| cue == "type:logical_block_split")));
    assert!(chunks.iter().all(|chunk| chunk.content.len() <= 100));
}
