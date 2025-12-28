use tree_sitter::Parser;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub context: String, // e.g., "function calculate_tax"
    pub structural_cues: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkerType {
    Python,
    Rust,
    TypeScript,
    JavaScript,
    Go,
    Html,
    Css,
    Php,
    Java,
    Markdown,
    Csv,
    Json,
    Yaml,
    Xml,
    Pdf,
    Office, // DOCX, XLSX, PPTX
    Text,
}

pub struct Chunker {
    // Parsers are not thread-safe so we create them on demand or thread-local 
    // but for simplicity here we re-create or use a pool later.
}

impl Chunker {
    pub fn chunk_file(path: &Path, content: &str) -> Vec<Chunk> {
        let file_type = Self::detect_type(path);
        
        match file_type {
            ChunkerType::Python => Self::chunk_python(content),
            ChunkerType::Rust => Self::chunk_rust(content),
            ChunkerType::TypeScript => Self::chunk_typescript(content),
            ChunkerType::JavaScript => Self::chunk_javascript(content),
            ChunkerType::Go => Self::chunk_go(content),
            ChunkerType::Html => Self::chunk_html(content),
            ChunkerType::Css => Self::chunk_css(content),
            ChunkerType::Php => Self::chunk_php(content),
            ChunkerType::Java => Self::chunk_java(content),
            ChunkerType::Markdown => Self::chunk_markdown(content),
            ChunkerType::Csv => Self::chunk_csv(content),
            ChunkerType::Json => Self::chunk_json(content),
            ChunkerType::Yaml => Self::chunk_yaml(content),
            ChunkerType::Xml => Self::chunk_xml(content),
            ChunkerType::Pdf => Self::chunk_pdf(path),
            ChunkerType::Office => Self::chunk_office(path),
            ChunkerType::Text => Self::chunk_text(content),
        }
    }

    fn detect_type(path: &Path) -> ChunkerType {
        match path.extension().and_then(|s| s.to_str()) {
            Some("py") => ChunkerType::Python,
            Some("rs") => ChunkerType::Rust,
            Some("ts" | "tsx") => ChunkerType::TypeScript,
            Some("js" | "jsx") => ChunkerType::JavaScript,
            Some("go") => ChunkerType::Go,
            Some("html" | "htm") => ChunkerType::Html,
            Some("css") => ChunkerType::Css,
            Some("php") => ChunkerType::Php,
            Some("java") => ChunkerType::Java,
            Some("md") => ChunkerType::Markdown,
            Some("csv") => ChunkerType::Csv,
            Some("json") => ChunkerType::Json,
            Some("yaml" | "yml") => ChunkerType::Yaml,
            Some("xml") => ChunkerType::Xml,
            Some("pdf") => ChunkerType::Pdf,
            Some("docx" | "xlsx" | "pptx") => ChunkerType::Office,
            _ => ChunkerType::Text,
        }
    }

    fn chunk_python(content: &str) -> Vec<Chunk> {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).expect("Error loading Python grammar");
        Self::chunk_treesitter_with_names(content, parser, &["function_definition", "class_definition"], "lang:python")
    }

    fn chunk_rust(content: &str) -> Vec<Chunk> {
        let mut parser = Parser::new();
        let language = tree_sitter_rust::LANGUAGE;
        parser.set_language(&language.into()).expect("Error loading Rust grammar");
        Self::chunk_treesitter_with_names(content, parser, &["function_item", "struct_item", "impl_item", "enum_item", "mod_item", "trait_item"], "lang:rust")
    }
    
    fn chunk_typescript(content: &str) -> Vec<Chunk> {
        let mut parser = Parser::new();
        let language = tree_sitter_typescript::LANGUAGE_TYPESCRIPT;
        parser.set_language(&language.into()).expect("Error loading TS grammar");
        Self::chunk_treesitter_with_names(content, parser, &["function_declaration", "class_declaration", "interface_declaration", "lexical_declaration", "method_definition", "constructor_declaration"], "lang:typescript")
    }

    fn chunk_javascript(content: &str) -> Vec<Chunk> {
        let mut parser = Parser::new();
        let language = tree_sitter_javascript::LANGUAGE;
        parser.set_language(&language.into()).expect("Error loading JS grammar");
        Self::chunk_treesitter_with_names(content, parser, &["function_declaration", "class_declaration", "method_definition"], "lang:javascript")
    }

    fn chunk_go(content: &str) -> Vec<Chunk> {
        let mut parser = Parser::new();
        let language = tree_sitter_go::LANGUAGE;
        parser.set_language(&language.into()).expect("Error loading Go grammar");
        Self::chunk_treesitter_with_names(content, parser, &["function_declaration", "method_declaration", "type_declaration"], "lang:go")
    }

    fn chunk_html(content: &str) -> Vec<Chunk> {
        let mut parser = Parser::new();
        let language = tree_sitter_html::LANGUAGE;
        parser.set_language(&language.into()).expect("Error loading HTML grammar");
        
        let mut chunks = Vec::new();
        if let Some(tree) = parser.parse(content, None) {
             Self::visit_html_nodes(tree.root_node(), content, &mut chunks);
        }
        if chunks.is_empty() && !content.trim().is_empty() { return Self::chunk_text(content); }
        chunks
    }

    fn visit_html_nodes(node: tree_sitter::Node, content: &str, chunks: &mut Vec<Chunk>) {
        if node.kind() == "element" {
            let start = node.start_position().row + 1;
            let end = node.end_position().row + 1;
            let text = node.utf8_text(content.as_bytes()).unwrap_or("").to_string();
            
            let mut cues = vec!["lang:html".to_string(), "type:element".to_string()];
            
            // Find tag name
            let mut tag_name = "anon";
            if let Some(st) = node.child_by_field_name("start_tag") {
                for i in 0..st.child_count() {
                    let c = st.child(i as u32).unwrap();
                    if c.kind() == "tag_name" {
                        tag_name = c.utf8_text(content.as_bytes()).unwrap_or("anon");
                    }
                }
            } else if node.kind() == "element" {
                // For tags without explicit start_tag field (sometimes happens depending on grammar version)
                for i in 0..node.child_count() {
                    let c = node.child(i as u32).unwrap();
                    if c.kind() == "start_tag" {
                         for j in 0..c.child_count() {
                             let gc = c.child(j as u32).unwrap();
                             if gc.kind() == "tag_name" {
                                 tag_name = gc.utf8_text(content.as_bytes()).unwrap_or("anon");
                             }
                         }
                    }
                }
            }
            cues.push(format!("tag:{}", tag_name));

            // Look for attributes (id, class)
            let mut st_node = node.child_by_field_name("start_tag");
            if st_node.is_none() {
                // Generic fallback search for start_tag
                for i in 0..node.child_count() {
                    let c = node.child(i as u32).unwrap();
                    if c.kind() == "start_tag" {
                        st_node = Some(c);
                        break;
                    }
                }
            }

            if let Some(st) = st_node {
                let mut st_cursor = st.walk();
                for st_child in st.children(&mut st_cursor) {
                    if st_child.kind() == "attribute" {
                        let attr_text = st_child.utf8_text(content.as_bytes()).unwrap_or("");
                        if let Some((name, val)) = attr_text.split_once('=') {
                            let clean_name = name.trim();
                            let clean_val = val.trim().trim_matches('"').trim_matches('\'');
                            
                            if clean_name == "id" {
                                cues.push(format!("id:{}", clean_val));
                            } else if clean_name == "class" {
                                for cls in clean_val.split_whitespace() {
                                    cues.push(format!("class:{}", cls));
                                }
                            }
                        }
                    }
                }
            }

            chunks.push(Chunk {
                content: text,
                start_line: start,
                end_line: end,
                context: format!("html:{}", tag_name),
                structural_cues: cues,
            });
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::visit_html_nodes(child, content, chunks);
        }
    }

    fn chunk_css(content: &str) -> Vec<Chunk> {
        let mut parser = Parser::new();
        let language = tree_sitter_css::LANGUAGE;
        parser.set_language(&language.into()).expect("Error loading CSS grammar");
        Self::chunk_treesitter_with_names(content, parser, &["rule_set"], "lang:css")
    }

    fn chunk_php(content: &str) -> Vec<Chunk> {
        let mut parser = Parser::new();
        // tree-sitter-php 0.23 uses LANGUAGE_PHP
        let language = tree_sitter_php::LANGUAGE_PHP;
        parser.set_language(&language.into()).expect("Error loading PHP grammar");
        Self::chunk_treesitter_with_names(content, parser, &["function_definition", "class_definition", "method_declaration"], "lang:php")
    }

    fn chunk_java(content: &str) -> Vec<Chunk> {
        let mut parser = Parser::new();
        let language = tree_sitter_java::LANGUAGE;
        parser.set_language(&language.into()).expect("Error loading Java grammar");
        Self::chunk_treesitter_with_names(content, parser, &["class_declaration", "method_declaration", "constructor_declaration"], "lang:java")
    }

    fn chunk_treesitter_with_names(content: &str, mut parser: Parser, node_kinds: &[&str], lang_tag: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        if let Some(tree) = parser.parse(content, None) {
             Self::visit_nodes(tree.root_node(), content, node_kinds, &mut chunks, lang_tag);
        }
        
        if chunks.is_empty() && !content.trim().is_empty() {
             return Self::chunk_text(content);
        }
        chunks
    }

    fn visit_nodes(node: tree_sitter::Node, content: &str, node_kinds: &[&str], chunks: &mut Vec<Chunk>, lang_tag: &str) {
        if node_kinds.contains(&node.kind()) {
             let name = node.child_by_field_name("name")
                .or_else(|| node.child_by_field_name("identifier"))
                .or_else(|| node.child_by_field_name("selectors"))
                .or_else(|| {
                    // Fallback for languages where identifiers aren't field-named (like some HTML nodes)
                    for i in 0..node.child_count() {
                        let c = node.child(i as u32).unwrap();
                        if c.kind() == "identifier" || c.kind() == "tag_name" || c.kind() == "selectors" {
                            return Some(c);
                        }
                    }
                    None
                })
                .map(|n| n.utf8_text(content.as_bytes()).unwrap_or("anon"))
                .unwrap_or("anon");

             let start = node.start_position().row + 1;
             let end = node.end_position().row + 1;
             let text = node.utf8_text(content.as_bytes()).unwrap_or("").to_string();
             
             let type_cue = node.kind()
                 .replace("_declaration", "")
                 .replace("_definition", "")
                 .replace("_item", "")
                 .replace("_rule", "")
                 .replace("_set", "");

             let name_label = if lang_tag == "lang:css" { "selector" } else { "name" };

             chunks.push(Chunk {
                 content: text,
                 start_line: start,
                 end_line: end,
                context: format!("{}:{}", node.kind(), name),
                structural_cues: vec![
                    lang_tag.to_string(),
                    format!("type:{}", type_cue),
                    format!("{}:{}", name_label, name),
                ],
            });
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::visit_nodes(child, content, node_kinds, chunks, lang_tag);
        }
    }



    fn chunk_markdown(content: &str) -> Vec<Chunk> {
        // Split by headers (#, ##, etc.)
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut current_block = Vec::new();
        let mut current_start = 1;
        let mut current_header = "root".to_string();
        
        for (i, line) in lines.iter().enumerate() {
            if line.starts_with('#') {
                if !current_block.is_empty() {
                    chunks.push(Chunk {
                        content: current_block.join("\n"),
                        start_line: current_start,
                        end_line: i,
                        context: current_header.clone(),
                        structural_cues: vec![
                            "type:markdown_section".to_string(),
                            format!("header:{}", current_header),
                        ],
                    });
                    current_block.clear();
                }
                current_start = i + 1;
                current_header = line.trim_start_matches('#').trim().to_string();
            }
            current_block.push(*line);
        }
        
        if !current_block.is_empty() {
            chunks.push(Chunk {
                content: current_block.join("\n"),
                start_line: current_start,
                end_line: lines.len(),
                context: current_header.clone(),
                structural_cues: vec![
                    "type:markdown_section".to_string(),
                    format!("header:{}", current_header),
                ],
            });
        }
        
        chunks
    }

    fn chunk_csv(content: &str) -> Vec<Chunk> {
        let mut rdr = csv::Reader::from_reader(content.as_bytes());
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut row_count = 0;
        let headers = rdr.headers().cloned().unwrap_or_default();
        
        // Pre-compute header cues
        let mut header_cues = vec!["type:csv_rows".to_string()];
        for h in &headers {
            header_cues.push(format!("header:{}", h));
        }
        
        for result in rdr.records() {
            if let Ok(record) = result {
                if row_count % 10 == 0 && row_count > 0 {
                    chunks.push(Chunk {
                        content: current_chunk.clone(),
                        start_line: row_count,
                        end_line: row_count + 10,
                        context: "csv_rows".to_string(),
                        structural_cues: header_cues.clone(),
                    });
                    current_chunk.clear();
                    current_chunk.push_str(&headers.iter().collect::<Vec<_>>().join(","));
                    current_chunk.push('\n');
                }
                current_chunk.push_str(&record.iter().collect::<Vec<_>>().join(","));
                current_chunk.push('\n');
                row_count += 1;
            }
        }
        
        if !current_chunk.is_empty() {
            chunks.push(Chunk {
                content: current_chunk,
                start_line: row_count.saturating_sub(10),
                end_line: row_count,
                context: "csv_rows".to_string(),
                structural_cues: header_cues,
            });
        }
        chunks
    }

    fn chunk_json(content: &str) -> Vec<Chunk> {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(content) {
            if let Some(obj) = value.as_object() {
                return obj.iter().map(|(key, val)| Chunk {
                    content: format!("\"{}\": {}", key, val),
                    start_line: 0,
                    end_line: 0,
                    context: format!("json_key:{}", key),
                    structural_cues: vec![
                        "type:json_entry".to_string(),
                        format!("key:{}", key),
                    ],
                }).collect();
            } else if let Some(arr) = value.as_array() {
                return arr.iter().enumerate().map(|(i, val)| Chunk {
                    content: val.to_string(),
                    start_line: 0,
                    end_line: 0,
                    context: format!("json_index:{}", i),
                    structural_cues: vec![
                        "type:json_item".to_string(),
                        format!("index:{}", i),
                    ],
                }).collect();
            }
        }
        Self::chunk_text(content)
    }

    fn chunk_yaml(content: &str) -> Vec<Chunk> {
        if let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(content) {
            if let Some(mapping) = value.as_mapping() {
                return mapping.iter().map(|(k, v)| {
                    let key_str = k.as_str().unwrap_or("unknown").to_string();
                    Chunk {
                        content: format!("{}: {}", serde_yaml::to_string(k).unwrap_or_default().trim(), serde_yaml::to_string(v).unwrap_or_default().trim()),
                        start_line: 0,
                        end_line: 0,
                        context: "yaml_block".to_string(),
                        structural_cues: vec![
                            "type:yaml_entry".to_string(),
                            format!("key:{}", key_str),
                        ],
                    }
                }).collect();
            }
        }
        Self::chunk_text(content)
    }

    fn chunk_xml(content: &str) -> Vec<Chunk> {
        if let Ok(doc) = roxmltree::Document::parse(content) {
            let mut chunks = Vec::new();
            for node in doc.root().children() {
                if node.is_element() {
                    let mut cues = vec![
                        "lang:xml".to_string(),
                        "type:xml_element".to_string(),
                        format!("tag:{}", node.tag_name().name()),
                    ];
                    for attr in node.attributes() {
                        cues.push(format!("attr:{}", attr.name()));
                        if attr.name() == "id" {
                            cues.push(format!("id:{}", attr.value()));
                        }
                    }

                    chunks.push(Chunk {
                        content: node.document().input_text()[node.range()].to_string(),
                        start_line: 0,
                        end_line: 0,
                        context: format!("xml_tag:{}", node.tag_name().name()),
                        structural_cues: cues,
                    });
                }
            }
            if !chunks.is_empty() {
                return chunks;
            }
        }
        Self::chunk_text(content)
    }

    fn chunk_pdf(path: &Path) -> Vec<Chunk> {
        if let Ok(content) = pdf_extract::extract_text(path) {
            return Self::chunk_text(&content);
        }
        Vec::new()
    }

    fn chunk_office(path: &Path) -> Vec<Chunk> {
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let mut full_text = String::new();

        match extension {
            "xlsx" => {
                use calamine::{Reader, Xlsx, open_workbook};
                if let Ok(mut excel) = open_workbook::<Xlsx<_>, _>(path) {
                    for sheet_name in excel.sheet_names().to_owned() {
                        if let Some(Ok(range)) = excel.worksheet_range(&sheet_name) {
                            for row in range.rows() {
                                for cell in row {
                                    full_text.push_str(&cell.to_string());
                                    full_text.push(' ');
                                }
                                full_text.push('\n');
                            }
                        }
                    }
                }
            },
            "docx" => {
                // docx-rs is better for structured reading
                if let Ok(_bytes) = std::fs::read(path) {
                    // Extract text (simplified placeholder for now as docx-rs is complex)
                    // In a real scenario we'd traverse the document tree
                    full_text.push_str("DOCX Content Placeholder");
                }
            },
            _ => {
                return Vec::new();
            }
        }
        
        if !full_text.trim().is_empty() {
            return Self::chunk_text(&full_text);
        }
        Vec::new()
    }

    fn chunk_text(content: &str) -> Vec<Chunk> {
        // Simple paragraph splitter
        // Split by double newline
        content.split("\n\n").enumerate().map(|(i, s)| {
             Chunk {
                 content: s.to_string(),
                 start_line: 0, // Hard to track line numbers with simple split
                 end_line: 0,
                 context: format!("para:{}", i),
                 structural_cues: vec![
                     "lang:text".to_string(),
                     "type:text_paragraph".to_string()
                 ],
             }
        }).filter(|c| !c.content.trim().is_empty()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(Chunker::detect_type(&PathBuf::from("test.py")), ChunkerType::Python);
        assert_eq!(Chunker::detect_type(&PathBuf::from("test.csv")), ChunkerType::Csv);
        assert_eq!(Chunker::detect_type(&PathBuf::from("test.pdf")), ChunkerType::Pdf);
        assert_eq!(Chunker::detect_type(&PathBuf::from("test.docx")), ChunkerType::Office);
    }
}
