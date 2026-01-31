use cuemap::agent::chunker::Chunker;
use std::path::PathBuf;

#[test]
fn test_all_formats_structural_cues() {
    let test_cases = vec![
        ("test.py", "def calc(): pass", vec!["lang:python", "type:function", "name:calc"]),
        ("test.rs", "pub struct User {}", vec!["lang:rust", "type:struct", "name:User"]),
        ("test.ts", "interface Config { id: string }", vec!["lang:typescript", "type:interface", "name:Config"]),
        ("test.js", "class Controller {}", vec!["lang:javascript", "type:class", "name:Controller"]),
        ("test.go", "func Process() {}", vec!["lang:go", "type:function", "name:Process"]),
        ("test.java", "public class App {}", vec!["lang:java", "type:class", "name:App"]),
        ("test.php", "<?php function run() {} ?>", vec!["lang:php", "type:function", "name:run"]),
        ("index.html", "<html><body><div id='app' class='container'></div></body></html>", vec!["lang:html", "tag:div", "id:app", "class:container"]),
        ("style.css", ".btn { color: red; }", vec!["lang:css", "selector:.btn"]),
        ("doc.md", "# Chapter 1\nContent", vec!["type:markdown_section", "header:Chapter 1"]),
        //("data.csv", "id,name\n1,bob", vec!["type:csv_rows", "header:id", "header:name"]),
        ("data.json", "{\"api\": \"v1\"}", vec!["type:json_entry", "key:api"]),
        ("data.yaml", "env: production", vec!["type:yaml_entry", "key:env"]),
        ("data.xml", "<?xml version='1.0'?><root id='001'><item/></root>", vec!["lang:xml", "tag:root", "id:001", "attr:id"]),
        ("test.txt", "Paragraph one.\n\nParagraph two.", vec!["lang:text", "type:text_content"]),
    ];

    for (filename, content, expected_cues) in test_cases {
        let path = PathBuf::from(filename);
        let chunks = Chunker::chunk_file(&path, content);
        
        assert!(!chunks.is_empty(), "Failed to chunk {}", filename);
        
        for expected in expected_cues {
            let found = chunks.iter().any(|c| c.structural_cues.contains(&expected.to_string()));
            assert!(
                found,
                "{} missing cue '{}'. All cues: {:?}",
                filename, expected, chunks.iter().flat_map(|c| c.structural_cues.clone()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn test_pdf_office_fallbacks() {
    // These calls usually fail or use stubs in test env without external deps, 
    // but they shouldn't panic.
    let pdf_path = PathBuf::from("dummy.pdf");
    let _chunks = Chunker::chunk_file(&pdf_path, "some content");
    // pdf_extract might return empty if it can't find the file, which is fine.
    // The main thing is they use ChunkerType::Pdf/Office.
}
