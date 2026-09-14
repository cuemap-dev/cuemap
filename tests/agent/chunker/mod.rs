use cuemap::agent::chunker::{ChunkCategory, Chunker, SegmenterConfig};
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
fn test_toml_chunking() {
    let content = r#"
title = "CueMap"

[package]
name = "cuemap"
version = "0.7.3"

[[bin]]
name = "cuemap"
path = "src/main.rs"
"#;
    let chunks = Chunker::chunk_file(std::path::Path::new("Cargo.toml"), content);

    assert!(!chunks.is_empty());
    assert!(chunks.iter().any(|chunk| {
        chunk.category == ChunkCategory::Structured
            && chunk.structural_cues.contains(&"lang:toml".to_string())
            && chunk.context == "table:package"
    }));
    assert!(chunks
        .iter()
        .any(|chunk| chunk.context == "table_array_element:bin"));
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
fn mainstream_tree_sitter_languages_emit_structural_cues() {
    let cases = [
        (
            "point.c",
            "#include <stdio.h>\nstruct Point { int x; };\nint distance(struct Point point) { printf(\"%d\", point.x); return point.x; }",
            "lang:c",
            "defines_function:distance",
        ),
        (
            "engine.cpp",
            "#include <vector>\nnamespace cuemap { class Engine { public: void run() {} }; }",
            "lang:cpp",
            "defines_namespace:cuemap",
        ),
        (
            "Engine.cs",
            "using System;\nnamespace CueMap { public class Engine { public void Run() { Console.WriteLine(\"ok\"); } } }",
            "lang:csharp",
            "defines_class:Engine",
        ),
        (
            "build.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\ngreet() { echo \"hello\"; }\ngreet",
            "lang:bash",
            "defines_function:greet",
        ),
    ];

    for (filename, content, lang, semantic_cue) in cases {
        let chunks = Chunker::chunk_file(std::path::Path::new(filename), content);
        assert!(!chunks.is_empty(), "Failed to chunk {filename}");
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.structural_cues.contains(&lang.to_string())),
            "{filename} missing language cue: {chunks:?}"
        );
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.structural_cues.contains(&semantic_cue.to_string())),
            "{filename} missing semantic cue: {chunks:?}"
        );
    }
}

#[test]
fn headers_use_project_and_content_context_for_classification() {
    use tempfile::tempdir;

    let plain_header = PathBuf::from("include/config.h");
    assert_eq!(
        Chunker::detect_type(&plain_header),
        Some(cuemap::agent::chunker::ChunkerType::C)
    );

    let cpp_header = Chunker::chunk_file(
        std::path::Path::new("include/engine.h"),
        "#pragma once\nnamespace cuemap { class Engine { public: void run(); }; }",
    );
    assert!(cpp_header
        .iter()
        .any(|chunk| chunk.structural_cues.contains(&"lang:cpp".to_string())));

    let apple_root = tempdir().unwrap();
    std::fs::create_dir(apple_root.path().join("CueMap.xcodeproj")).unwrap();
    std::fs::write(
        apple_root.path().join("CueMap.xcodeproj/project.pbxproj"),
        "// !$*UTF8*$!",
    )
    .unwrap();
    let apple_header = apple_root.path().join("Engine.h");
    assert_eq!(
        Chunker::detect_type(&apple_header),
        Some(cuemap::agent::chunker::ChunkerType::ObjectiveC)
    );
    let apple_chunks = Chunker::chunk_file(
        &apple_header,
        "#import <Foundation/Foundation.h>\n@interface Engine : NSObject\n@end",
    );
    assert!(apple_chunks
        .iter()
        .any(|chunk| chunk.structural_cues.contains(&"lang:objc".to_string())));
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
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("ViewController.SWIFT")),
        Some(ChunkerType::Swift)
    );
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("home.dart")),
        Some(ChunkerType::Dart)
    );
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("LegacyView.m")),
        Some(ChunkerType::ObjectiveC)
    );
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("MainActivity.kt")),
        Some(ChunkerType::Kotlin)
    );
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("Cargo.toml")),
        Some(ChunkerType::Toml)
    );
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("Engine.cs")),
        Some(ChunkerType::CSharp)
    );
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("engine.cpp")),
        Some(ChunkerType::Cpp)
    );
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("point.c")),
        Some(ChunkerType::C)
    );
    assert_eq!(
        Chunker::detect_type(&PathBuf::from("build.sh")),
        Some(ChunkerType::Bash)
    );
}

#[test]
fn mobile_language_chunkers_emit_structural_cues() {
    let cases = [
        (
            "View.swift",
            "import Foundation\nstruct Greeter {\n    func greet() {}\n}",
            "lang:swift",
            "type:struct",
            "name:Greeter",
            "defines_struct:Greeter",
        ),
        (
            "home.dart",
            "class HomeScreen {\n  void build() {}\n}",
            "lang:dart",
            "type:class",
            "name:HomeScreen",
            "defines_class:HomeScreen",
        ),
        (
            "LegacyView.m",
            "@interface LegacyView : NSObject\n@end",
            "lang:objc",
            "type:class_interface",
            "name:LegacyView",
            "defines_class:LegacyView",
        ),
        (
            "MainActivity.kt",
            "class MainActivity {\n    fun render() {}\n}",
            "lang:kotlin",
            "type:class",
            "name:MainActivity",
            "defines_class:MainActivity",
        ),
    ];

    for (filename, content, lang, type_cue, name, semantic_cue) in cases {
        let chunks = Chunker::chunk_file(std::path::Path::new(filename), content);
        assert!(!chunks.is_empty(), "Failed to chunk {filename}");
        assert!(
            chunks.iter().any(|chunk| chunk.structural_cues.contains(&lang.to_string())),
            "{filename} missing language cue: {chunks:?}"
        );
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.structural_cues.contains(&type_cue.to_string())),
            "{filename} missing type cue: {chunks:?}"
        );
        assert!(
            chunks.iter().any(|chunk| chunk.structural_cues.contains(&name.to_string())),
            "{filename} missing name cue: {chunks:?}"
        );
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.structural_cues.contains(&semantic_cue.to_string())),
            "{filename} missing semantic cue: {chunks:?}"
        );
    }
}

#[test]
fn rust_chunking_keeps_comments_in_source_neighborhoods() {
    let content = r#"//! Module-level explanation kept as searchable context.

use std::fmt;

/// Keeps lexical discovery bounded before local reranking.
/// This belongs with the function it explains.
fn rerank_candidates() {
    // Internal implementation notes are already inside the function chunk.
    let limit = 200;
}

// Standalone architecture note.
// The encoder never downloads models at runtime.

const DEFAULT_LIMIT: usize = 200;
"#;

    let chunks = Chunker::chunk_file(std::path::Path::new("semantic.rs"), content);
    let function = chunks
        .iter()
        .find(|chunk| chunk.context == "function_item:rerank_candidates")
        .expect("function chunk");

    assert!(function
        .content
        .starts_with("/// Keeps lexical discovery bounded"));
    assert!(function.content.contains("Internal implementation notes"));
    assert_eq!(
        chunks
            .iter()
            .filter(|chunk| chunk.content.contains("Internal implementation notes"))
            .count(),
        1,
        "comments already inside code chunks must not be duplicated"
    );

    assert!(chunks.iter().any(|chunk| {
        chunk.context == "comment:block"
            && chunk.content.contains("Module-level explanation")
            && chunk
                .structural_cues
                .iter()
                .any(|cue| cue == "type:comment_block")
    }));
    assert!(chunks.iter().any(|chunk| {
        chunk.context == "comment:block"
            && chunk.content.contains("encoder never downloads models")
    }));
    assert!(chunks
        .windows(2)
        .all(|pair| pair[0].start_line <= pair[1].start_line));
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

#[test]
fn logical_block_code_uses_treesitter_cues() {
    let content = "```python\nprint('keep code together')\n```";
    let config = SegmenterConfig {
        window_size: 1,
        overlap: 0,
        min_chunk_chars: 20,
        max_chunk_chars: 4000,
    };

    let chunks = Chunker::chunk_text_logical_blocks(content, &config);

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].category, ChunkCategory::Code);
    assert!(chunks[0]
        .structural_cues
        .iter()
        .any(|cue| cue == "lang:python"));
    assert!(chunks[0]
        .structural_cues
        .iter()
        .any(|cue| cue == "type:call"));
    assert!(chunks[0]
        .structural_cues
        .iter()
        .any(|cue| cue == "name:print"));
}

#[test]
fn chunk_file_dispatches_supported_formats_and_attaches_parent_links() {
    let cases = [
        ("module.py", "def greet(name):\n    return name"),
        ("module.rs", "fn greet() { println!(\"hi\"); }"),
        ("module.ts", "function greet(name: string) { return name; }"),
        ("module.js", "function greet(name) { return name; }"),
        ("module.go", "package main\nfunc main() {}"),
        ("page.html", "<main><h1>Title</h1><p>Body text.</p></main>"),
        ("page.css", ".card { color: red; }"),
        ("module.php", "<?php function greet() { return true; } ?>"),
        ("Module.java", "public class Module { void run() {} }"),
        ("notes.md", "# Heading\nA paragraph."),
        ("rows.csv", "name,email\nAlice,alice@example.com"),
        ("data.json", "{\"name\":\"Alice\"}"),
        ("data.yaml", "name: Alice"),
        ("data.xml", "<root id=\"r1\"><child/></root>"),
        ("notes.txt", "A plain text note."),
    ];

    for (filename, content) in cases {
        let chunks = Chunker::chunk_file(std::path::Path::new(filename), content);
        assert!(!chunks.is_empty(), "expected chunks for {filename}");
        assert!(chunks
            .iter()
            .all(|chunk| chunk.structural_cues.iter().any(|cue| cue.starts_with("parent:"))));
        assert!(chunks
            .iter()
            .all(|chunk| chunk.structural_cues.iter().any(|cue| cue.starts_with("chunk_idx:"))));
    }

    assert_eq!(
        Chunker::get_category_for_file(std::path::Path::new("unknown.bin")),
        ChunkCategory::Prose
    );
    assert_eq!(
        Chunker::get_category_for_file(std::path::Path::new("api.json")),
        ChunkCategory::Structured
    );
}

#[test]
fn structured_chunkers_cover_arrays_api_specs_and_fallbacks() {
    let array = Chunker::chunk_json("[1, {\"name\":\"Alice\"}]");
    assert_eq!(array.len(), 2);
    assert!(array.iter().all(|chunk| chunk.context.starts_with("json_index:")));

    let api_json = Chunker::chunk_json(
        r#"{"swagger":"2.0","info":{"title":"Demo API"},"paths":{"/health":{"get":{"summary":"Health","operationId":"health","tags":["ops"]},"parameters":{}}}}"#,
    );
    assert_eq!(api_json.len(), 1);
    assert!(api_json[0].structural_cues.iter().any(|cue| cue == "method:GET"));
    assert!(api_json[0].structural_cues.iter().any(|cue| cue == "tag:ops"));

    let api_yaml = Chunker::chunk_yaml(
        "swagger: 2.0\ninfo:\n  title: Demo\npaths:\n  /health:\n    get:\n      summary: Health\n      operationId: health\n      tags: [ops]\n",
    );
    assert_eq!(api_yaml.len(), 1);
    assert!(api_yaml[0].structural_cues.iter().any(|cue| cue == "method:GET"));

    let xml = Chunker::chunk_file(
        std::path::Path::new("data.xml"),
        "<root id=\"r1\"><child name=\"x\"/></root>",
    );
    assert!(xml[0].structural_cues.iter().any(|cue| cue == "id:r1"));
    assert_eq!(Chunker::chunk_json("not json")[0].context, "text:full");
    assert_eq!(Chunker::chunk_yaml("not: [valid")[0].context, "text:full");
}

#[test]
fn social_exports_and_binary_fallbacks_are_routed_deterministically() {
    let whatsapp = Chunker::chunk_file(
        std::path::Path::new("whatsapp.txt"),
        "[1/2/24, 10:00] Alice: Hello there\n[1/2/24, 10:01] Alice: image omitted\n[1/2/24, 10:02] Bob: Great news",
    );
    assert_eq!(whatsapp.len(), 2);
    assert!(whatsapp[0].structural_cues.iter().any(|cue| cue == "platform:whatsapp"));

    let instagram = Chunker::chunk_file(
        std::path::Path::new("instagram.json"),
        r#"[{"sender_name":"Alice","timestamp_ms":1700000000000,"content":"Hello","share":{"link":"https://example.com/post"}},{"sender_name":"Bob","timestamp_ms":0,"content":"Liked a message"}]"#,
    );
    assert_eq!(instagram.len(), 1);
    assert!(instagram[0].structural_cues.iter().any(|cue| cue == "has:shared_link"));

    let chrome = Chunker::chunk_file(
        std::path::Path::new("chrome_history.json"),
        r#"{"Browser History":[{"title":"CueMap","url":"https://cuemap.dev","time_usec":1700000000000000},{"title":"CueMap","url":"https://cuemap.dev","time_usec":1700000000000000}]}"#,
    );
    assert_eq!(chrome.len(), 1);
    assert!(chrome[0].structural_cues.iter().any(|cue| cue == "platform:chrome"));

    let youtube = Chunker::chunk_file(
        std::path::Path::new("youtube-watch-history.html"),
        r#"Watched <a href="https://www.youtube.com/watch?v=abc">Rust release</a> Jan 2, 2024 Searched for <a href="https://www.youtube.com/results">coverage</a> Jan 3, 2024"#,
    );
    assert_eq!(youtube.len(), 2);
    assert!(youtube.iter().all(|chunk| chunk.category == ChunkCategory::Conversation));

    assert!(Chunker::chunk_binary_file(std::path::Path::new("missing.pdf")).is_empty());
    assert!(Chunker::chunk_binary_file(std::path::Path::new("missing.docx")).is_empty());
}

#[test]
fn text_window_and_structural_cue_helpers_cover_edge_cases() {
    let config = SegmenterConfig {
        window_size: 2,
        overlap: 5,
        min_chunk_chars: 1,
        max_chunk_chars: 25,
    };
    let chunks = Chunker::chunk_text_with_config(
        "One sentence is here. Two sentence follows. Three sentence follows.",
        &config,
    );
    assert!(chunks.len() >= 2);
    assert!(chunks.iter().all(|chunk| chunk.content.len() <= 25));
    assert!(Chunker::chunk_text_with_config("", &config).is_empty());

    let mut chunks = vec![cuemap::agent::chunker::Chunk {
        content: "x".to_string(),
        start_line: 1,
        end_line: 1,
        context: "test".to_string(),
        structural_cues: vec!["parent:old".to_string(), "chunk_idx:9".to_string()],
        category: ChunkCategory::Prose,
    }];
    Chunker::attach_parent_links(&mut chunks, "seed");
    Chunker::inherit_structural_cues(
        &mut chunks,
        &["source:test".to_string(), "source:test".to_string()],
    );
    assert!(chunks[0].structural_cues.iter().any(|cue| cue == "source:test"));
    assert_eq!(
        chunks[0]
            .structural_cues
            .iter()
            .filter(|cue| *cue == "source:test")
            .count(),
        1
    );
    Chunker::inherit_structural_cues(&mut chunks, &[]);
}

#[test]
fn article_content_links_are_scoped_and_resolved() {
    let html = scraper::Html::parse_document(
        r##"<html><body><nav><a href="/nav">Nav</a></nav><main><h1>Title</h1><p>Long enough article content for extraction.</p><a href="/docs">Docs</a><a href="https://example.com/full">Full</a><a href="#skip">Skip</a></main><footer><a href="/footer">Footer</a></footer></body></html>"##,
    );
    let links = Chunker::extract_content_links(
        &html,
        &url::Url::parse("https://cuemap.dev/base/").unwrap(),
    );
    assert_eq!(
        links,
        vec![
            "https://cuemap.dev/docs".to_string(),
            "https://example.com/full".to_string(),
        ]
    );
}
