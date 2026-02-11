use tree_sitter::Parser;
use std::path::Path;
use std::cell::RefCell;
use unicode_segmentation::UnicodeSegmentation;

// Thread-local parser pool to avoid re-creating parsers for each file.
// Tree-sitter parsers are expensive to initialize, especially when ingesting
// thousands of files. This can improve performance by 10-100x for large codebases.
thread_local! {
    static PARSERS: RefCell<Parsers> = RefCell::new(Parsers::new());
}

struct Parsers {
    python: Option<Parser>,
    rust: Option<Parser>,
    typescript: Option<Parser>,
    javascript: Option<Parser>,
    go: Option<Parser>,
    html: Option<Parser>,
    css: Option<Parser>,
    php: Option<Parser>,
    java: Option<Parser>,
}

impl Parsers {
    fn new() -> Self {
        Self {
            python: None,
            rust: None,
            typescript: None,
            javascript: None,
            go: None,
            html: None,
            css: None,
            php: None,
            java: None,
        }
    }

    fn get_python(&mut self) -> &mut Parser {
        self.python.get_or_insert_with(|| {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_python::LANGUAGE.into()).expect("Error loading Python grammar");
            parser
        })
    }

    fn get_rust(&mut self) -> &mut Parser {
        self.rust.get_or_insert_with(|| {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_rust::LANGUAGE.into()).expect("Error loading Rust grammar");
            parser
        })
    }

    fn get_typescript(&mut self) -> &mut Parser {
        self.typescript.get_or_insert_with(|| {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()).expect("Error loading TS grammar");
            parser
        })
    }

    fn get_javascript(&mut self) -> &mut Parser {
        self.javascript.get_or_insert_with(|| {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_javascript::LANGUAGE.into()).expect("Error loading JS grammar");
            parser
        })
    }

    fn get_go(&mut self) -> &mut Parser {
        self.go.get_or_insert_with(|| {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_go::LANGUAGE.into()).expect("Error loading Go grammar");
            parser
        })
    }

    fn get_html(&mut self) -> &mut Parser {
        self.html.get_or_insert_with(|| {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_html::LANGUAGE.into()).expect("Error loading HTML grammar");
            parser
        })
    }

    fn get_css(&mut self) -> &mut Parser {
        self.css.get_or_insert_with(|| {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_css::LANGUAGE.into()).expect("Error loading CSS grammar");
            parser
        })
    }

    fn get_php(&mut self) -> &mut Parser {
        self.php.get_or_insert_with(|| {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_php::LANGUAGE_PHP.into()).expect("Error loading PHP grammar");
            parser
        })
    }

    fn get_java(&mut self) -> &mut Parser {
        self.java.get_or_insert_with(|| {
            let mut parser = Parser::new();
            parser.set_language(&tree_sitter_java::LANGUAGE.into()).expect("Error loading Java grammar");
            parser
        })
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub context: String,
    pub structural_cues: Vec<String>,
    pub category: ChunkCategory,
}

/// Content category for semantic handling decisions.
/// This determines whether WordNet expansion is appropriate.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ChunkCategory {
    Code,           // Programming languages - no WordNet expansion
    #[default]
    Prose,          // Longform text - use sentence segmentation + WordNet
    Structured,     // CSV, JSON, YAML, XML - no WordNet
    ApiSpec,        // OpenAPI/Swagger - special handling
    Conversation,   // Chat exports - participant context
    WebContent,     // URLs - metadata extraction
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
    ApiSpec,        // ApiSpec/Swagger specs
    SocialExport   // Generic social media export (auto-detected format)
}

/// Configuration for sentence segmentation
#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    pub window_size: usize,       // sentences per chunk (default: 3)
    pub overlap: usize,           // sentence overlap (default: 1)  
    pub min_chunk_chars: usize,   // minimum chunk size (default: 50)
    pub max_chunk_chars: usize,   // maximum chunk size (default: 2000)
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            window_size: 3,
            overlap: 1,
            min_chunk_chars: 50,
            max_chunk_chars: 2000,
        }
    }
}

pub struct Chunker;

impl Chunker {
    /// Chunk a binary file from disk. Used for PDF/Office files that require file-based extraction.
    /// For text files, falls back to reading content and using chunk_file.
    pub fn chunk_binary_file(path: &Path) -> Vec<Chunk> {
        let file_type = match Self::detect_type(path) {
            Some(t) => t,
            None => return Vec::new(),
        };
        
        match file_type {
            ChunkerType::Pdf => Self::chunk_pdf(path),
            ChunkerType::Office => Self::chunk_office(path),
            _ => {
                // For non-binary types, read as text and use standard chunking
                if let Ok(content) = std::fs::read_to_string(path) {
                    Self::chunk_file(path, &content)
                } else {
                    Vec::new()
                }
            }
        }
    }
    
    pub fn chunk_file(path: &Path, content: &str) -> Vec<Chunk> {
        // PRIORITY 1: Path-based type detection (explicit extensions win)
        let file_type = match Self::detect_type(path) {
            Some(t) => t,
            None => {
                // PRIORITY 2: Content-based detection for social media exports or other formats without extensions
                if let Some(chunks) = Self::try_social_export_by_content(content, path) {
                    return chunks;
                }
                
                // If still unknown format, we return empty chunks to skip ingestion
                // as per user request to avoid blindly processing unknown formats as text.
                return Vec::new();
            }
        };
        
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
            ChunkerType::ApiSpec => Self::chunk_json(content),
            ChunkerType::SocialExport => Self::chunk_social_export(path, content),
        }
    }

    /// Try to detect social media export by content patterns FIRST
    fn try_social_export_by_content(content: &str, path: &Path) -> Option<Vec<Chunk>> {
        // Take first 500 chars safely (not bytes) to avoid Unicode boundary issues
        let content_start: String = content.chars().take(500).collect();
        let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        
        // WhatsApp: [date, time] sender: message pattern at start of file
        let whatsapp_re = regex::Regex::new(r"^\[?\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}").unwrap();
        if whatsapp_re.is_match(&content_start) {
            return Some(Self::chunk_whatsapp(content));
        }
        
        // Instagram: JSON array with sender_name and timestamp_ms
        if content.starts_with("[") && content.contains("\"sender_name\"") && content.contains("\"timestamp_ms\"") {
            return Some(Self::chunk_instagram(content));
        }
        
        // Chrome History: JSON with "Browser History" key
        if content.contains("\"Browser History\"") {
            return Some(Self::chunk_chrome_history(content));
        }
        
        // YouTube: HTML with Watched links  
        if filename.contains("watch-history") || filename.contains("search-history") 
            || (content.contains("youtube.com/watch") && content.contains("Watched")) {
            return Some(Self::chunk_youtube_history(content));
        }
        
        None // Not a recognized social export
    }

    /// Detect social export type from content and route to appropriate parser
    fn chunk_social_export(path: &Path, content: &str) -> Vec<Chunk> {
        let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let parent = path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("");
        
        // WhatsApp: .txt files with [date, time] sender: message pattern
        if (filename.to_lowercase().contains("whatsapp") || parent.to_lowercase().contains("whatsapp"))
            || (content.len() > 20 && regex::Regex::new(r"^\[?\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}").unwrap().is_match(&content[..content.len().min(100)])) {
            return Self::chunk_whatsapp(content);
        }
        
        // Instagram: JSON with sender_name and timestamp_ms
        if content.contains("\"sender_name\"") && content.contains("\"timestamp_ms\"") {
            return Self::chunk_instagram(content);
        }
        
        // Chrome History: JSON with Browser History key
        if content.contains("\"Browser History\"") {
            return Self::chunk_chrome_history(content);
        }
        
        // YouTube: HTML with watch/search history patterns  
        if filename.contains("watch-history") || filename.contains("search-history") 
            || content.contains("youtube.com/watch") {
            return Self::chunk_youtube_history(content);
        }
        
        // Fallback to JSON parsing
        Self::chunk_json(content)
    }

    pub fn detect_type(path: &Path) -> Option<ChunkerType> {
        // Check for social media export patterns in path
        let path_lower = path.to_string_lossy().to_lowercase();
        if path_lower.contains("whatsapp") 
            || path_lower.contains("instagram") 
            || path_lower.contains("youtube")
            || (path_lower.contains("chrome") && path_lower.contains("history")) {
            return Some(ChunkerType::SocialExport);
        }
        
        match path.extension().and_then(|s| s.to_str()) {
            Some("py") => Some(ChunkerType::Python),
            Some("rs") => Some(ChunkerType::Rust),
            Some("ts" | "tsx") => Some(ChunkerType::TypeScript),
            Some("js" | "jsx") => Some(ChunkerType::JavaScript),
            Some("go") => Some(ChunkerType::Go),
            Some("html" | "htm") => Some(ChunkerType::Html),
            Some("css") => Some(ChunkerType::Css),
            Some("php") => Some(ChunkerType::Php),
            Some("java") => Some(ChunkerType::Java),
            Some("md") => Some(ChunkerType::Markdown),
            Some("csv") => Some(ChunkerType::Csv),
            Some("json") => Some(ChunkerType::Json),
            Some("yaml" | "yml") => Some(ChunkerType::Yaml),
            Some("xml") => Some(ChunkerType::Xml),
            Some("pdf") => Some(ChunkerType::Pdf),
            Some("docx" | "xlsx" | "pptx") => Some(ChunkerType::Office),
            Some("txt" | "log") => Some(ChunkerType::Text),
            _ => None,
        }
    }

    fn chunk_python(content: &str) -> Vec<Chunk> {
        PARSERS.with(|parsers| {
            let mut parsers = parsers.borrow_mut();
            let parser = parsers.get_python();
            Self::chunk_treesitter_with_names(content, parser, 
                &["function_definition", "class_definition", "if_statement", "for_statement", "while_statement", "try_statement", "except_clause", "with_statement", "assignment", "call", "comment"], 
                "lang:python", ChunkCategory::Code)
        })
    }

    fn chunk_rust(content: &str) -> Vec<Chunk> {
        PARSERS.with(|parsers| {
            let mut parsers = parsers.borrow_mut();
            let parser = parsers.get_rust();
            Self::chunk_treesitter_with_names(content, parser, 
                &["function_item", "struct_item", "enum_item", "trait_item", 
                  "if_expression", "match_expression", "match_arm", "for_expression", "while_expression", "loop_expression"], 
                "lang:rust", ChunkCategory::Code)
        })
    }
    
    fn chunk_typescript(content: &str) -> Vec<Chunk> {
        PARSERS.with(|parsers| {
            let mut parsers = parsers.borrow_mut();
            let parser = parsers.get_typescript();
            Self::chunk_treesitter_with_names(content, parser, 
                &["function_declaration", "class_declaration", "interface_declaration", "lexical_declaration", "method_definition", "constructor_declaration", "if_statement", "for_statement", "while_statement", "expression_statement", "call_expression", "comment", "jsx_element", "jsx_self_closing_element"], 
                "lang:typescript", ChunkCategory::Code)
        })
    }

    fn chunk_javascript(content: &str) -> Vec<Chunk> {
        PARSERS.with(|parsers| {
            let mut parsers = parsers.borrow_mut();
            let parser = parsers.get_javascript();
            Self::chunk_treesitter_with_names(content, parser, 
                &["function_declaration", "class_declaration", "method_definition", "if_statement", "for_statement", "while_statement", "expression_statement", "call_expression", "comment", "jsx_element", "jsx_self_closing_element"], 
                "lang:javascript", ChunkCategory::Code)
        })
    }

    pub fn chunk_go(content: &str) -> Vec<Chunk> {
        PARSERS.with(|parsers| {
            let mut parsers = parsers.borrow_mut();
            let parser = parsers.get_go();
            Self::chunk_treesitter_with_names(content, parser, 
                &["function_declaration", "method_declaration", "type_declaration", "if_statement", "for_statement", "call_expression", "block"], 
                "lang:go", ChunkCategory::Code)
        })
    }

    pub fn chunk_html(content: &str) -> Vec<Chunk> {
        PARSERS.with(|parsers| {
            let mut parsers = parsers.borrow_mut();
            let parser = parsers.get_html();
            
            let mut chunks = Vec::new();
            if let Some(tree) = parser.parse(content, None) {
                 Self::visit_html_nodes(tree.root_node(), content, &mut chunks);
            }
            if chunks.is_empty() && !content.trim().is_empty() { return Self::chunk_text(content); }
            chunks
        })
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
                category: ChunkCategory::Code,
            });
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::visit_html_nodes(child, content, chunks);
        }
    }

    pub fn chunk_css(content: &str) -> Vec<Chunk> {
        PARSERS.with(|parsers| {
            let mut parsers = parsers.borrow_mut();
            let parser = parsers.get_css();
            Self::chunk_treesitter_with_names(content, parser, &["rule_set"], "lang:css", ChunkCategory::Code)
        })
    }

    pub fn chunk_php(content: &str) -> Vec<Chunk> {
        PARSERS.with(|parsers| {
            let mut parsers = parsers.borrow_mut();
            let parser = parsers.get_php();
            Self::chunk_treesitter_with_names(content, parser, 
                &["function_definition", "class_definition", "method_declaration", "if_statement", "for_statement", "foreach_statement", "while_statement", "expression_statement", "comment", "compound_statement"], 
                "lang:php", ChunkCategory::Code)
        })
    }

    pub fn chunk_java(content: &str) -> Vec<Chunk> {
        PARSERS.with(|parsers| {
            let mut parsers = parsers.borrow_mut();
            let parser = parsers.get_java();
            Self::chunk_treesitter_with_names(content, parser, 
                &["class_declaration", "method_declaration", "constructor_declaration", "if_statement", "for_statement", "while_statement", "expression_statement", "comment", "block"], 
                "lang:java", ChunkCategory::Code)
        })
    }

    fn chunk_treesitter_with_names(
        content: &str, 
        parser: &mut Parser, 
        node_kinds: &[&str], 
        lang_tag: &str, 
        category: ChunkCategory
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        tracing::info!("[CHUNKER DEBUG] Attempting to parse {} bytes...", content.len());
        let tree_result = parser.parse(content, None);
        tracing::info!("[CHUNKER DEBUG] Parse result: {}", if tree_result.is_some() { "SUCCESS" } else { "FAILED" });
        if let Some(tree) = tree_result {
            // New config for splitting logic
            let max_chars = 3000; // soft limit - large enough for most functions
            tracing::info!("[CHUNKER DEBUG] Parsed {} bytes, root: {}", content.len(), tree.root_node().kind());
            Self::visit_nodes_recursive(
                tree.root_node(), 
                content, 
                node_kinds, 
                &mut chunks, 
                lang_tag, 
                category,
                max_chars
            );
            tracing::info!("[CHUNKER DEBUG] After visit: {} chunks", chunks.len());
        }
        
        if chunks.is_empty() && !content.trim().is_empty() {
            eprintln!("[DEBUG CHUNKER] No chunks from tree-sitter, falling back to line-based chunker");
            // Use line-based chunking for code (not sentence-based which creates overlapping windows)
            let lines: Vec<&str> = content.lines().collect();
            let lines_per_chunk = 20; // ~20 lines per chunk for code
            
            for (chunk_idx, chunk_lines) in lines.chunks(lines_per_chunk).enumerate() {
                let chunk_content = chunk_lines.join("\n");
                if chunk_content.trim().is_empty() {
                    continue;
                }
                
                let chunk_start = 1 + (chunk_idx * lines_per_chunk);
                let chunk_end = chunk_start + chunk_lines.len() - 1;
                
                chunks.push(Chunk {
                    content: chunk_content,
                    start_line: chunk_start,
                    end_line: chunk_end,
                    context: format!("code (part {})", chunk_idx + 1),
                    structural_cues: vec![
                        lang_tag.to_string(),
                    ],
                    category,
                });
            }
        }
        chunks
    }

    fn visit_nodes_recursive(
        node: tree_sitter::Node, 
        content: &str, 
        node_kinds: &[&str], 
        chunks: &mut Vec<Chunk>, 
        lang_tag: &str, 
        category: ChunkCategory,
        max_chars: usize
    ) {
        let kind = node.kind();
        let start_row = node.start_position().row;
        let end_row = node.end_position().row;
        // let line_count = end_row - start_row;
        
        // 1. Is this a node we care about?
        let is_target_node = node_kinds.contains(&kind);
        
        // 2. Get the text length roughly (byte range is faster than utf8 conversion)
        let byte_len = node.end_byte() - node.start_byte();

        // 3. DECISION LOGIC:
        // If it's a target node AND it fits within our size limit, chunk it.
        // If it's too big, we usually SKIP making a chunk here and drill down, 
        // UNLESS it's a "leaf-ish" node (like a comment or huge string) that won't have children.
        
        let should_chunk_here = is_target_node && byte_len <= max_chars;
        
        // Debug: Log all target nodes we encounter
        if is_target_node {
            eprintln!("[DEBUG CHUNKER] Found target: {} lines {}-{} ({} bytes) should_chunk={}", 
                kind, start_row + 1, end_row + 1, byte_len, should_chunk_here);
        }
        
        if should_chunk_here {
            // --- EXTRACT IDENTIFIER ---
            let name = node.child_by_field_name("name")
                .or_else(|| node.child_by_field_name("identifier"))
                .or_else(|| node.child_by_field_name("selectors"))
                .or_else(|| {
                    for i in 0..node.child_count() {
                        let c = node.child(i as u32).unwrap();
                        if c.kind() == "identifier" || c.kind() == "tag_name" || c.kind() == "selectors" {
                            return Some(c);
                        }
                    }
                    None
                })
                .map(|n| n.utf8_text(content.as_bytes()).unwrap_or("anon"))
                .unwrap_or_else(|| {
                     if kind.contains("statement") { "stmt" }
                     else if kind.contains("expression") { "expr" }
                     else if kind.contains("comment") { "comment" }
                     else { "anon" }
                });

            let text = node.utf8_text(content.as_bytes()).unwrap_or("").to_string();
            
            let type_cue = kind
                 .replace("_declaration", "")
                 .replace("_definition", "")
                 .replace("_item", "")
                 .replace("_rule", "")
                 .replace("_set", "");

            let name_label = if lang_tag == "lang:css" { "selector" } else { "name" };

            eprintln!("[DEBUG CHUNKER] Creating chunk: {} '{}' lines {}-{} ({} bytes)", 
                kind, name, start_row + 1, end_row + 1, byte_len);

            chunks.push(Chunk {
                content: text,
                start_line: start_row + 1,
                end_line: end_row + 1,
                context: format!("{}:{}", kind, name),
                structural_cues: vec![
                    lang_tag.to_string(),
                    format!("type:{}", type_cue),
                    format!("{}:{}", name_label, name),
                ],
                category,
            });
            
            // If we chunked this node, we generally don't want to chunk its children 
            // separately UNLESS the node is massive. But since we checked byte_len <= max_chars,
            // we treat this as a "leaf chunk".
            return; 
        }

        // 4. If we didn't chunk it (because it wasn't a target OR it was too big),
        // we recurse into children to find smaller, manageable pieces.
        if is_target_node && byte_len > max_chars {
            let initial_chunk_count = chunks.len();
            
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                Self::visit_nodes_recursive(child, content, node_kinds, chunks, lang_tag, category, max_chars);
            }
            
            // If we drilled down but got nothing (e.g. big linear function), force line-based segmentation
            if chunks.len() == initial_chunk_count {
                let text = node.utf8_text(content.as_bytes()).unwrap_or("");
                
                // Use line-based chunking for code (not sentence-based which creates overlapping windows)
                let lines: Vec<&str> = text.lines().collect();
                let lines_per_chunk = 20; // ~20 lines per chunk for code
                
                for (chunk_idx, chunk_lines) in lines.chunks(lines_per_chunk).enumerate() {
                    let chunk_content = chunk_lines.join("\n");
                    if chunk_content.trim().is_empty() {
                        continue;
                    }
                    
                    let chunk_start = start_row + 1 + (chunk_idx * lines_per_chunk);
                    let chunk_end = chunk_start + chunk_lines.len() - 1;
                    
                    chunks.push(Chunk {
                        content: chunk_content,
                        start_line: chunk_start,
                        end_line: chunk_end,
                        context: format!("{} (part {})", kind, chunk_idx + 1),
                        structural_cues: vec![
                            lang_tag.to_string(),
                            format!("type:{}", kind),
                            format!("parent:{}", kind),
                        ],
                        category,
                    });
                }
            }
        } else {
            // Standard recursion for non-target nodes
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                Self::visit_nodes_recursive(child, content, node_kinds, chunks, lang_tag, category, max_chars);
            }
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
                        category: ChunkCategory::Prose,
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
                category: ChunkCategory::Prose,
            });
        }
        
        chunks
    }

    pub fn chunk_csv(content: &str) -> Vec<Chunk> {
        Self::chunk_csv_with_filename(content, "data.csv")
    }
    
    /// Chunk CSV with filename for cues
    pub fn chunk_csv_with_filename(content: &str, filename: &str) -> Vec<Chunk> {
        let mut rdr = csv::Reader::from_reader(content.as_bytes());
        let mut chunks = Vec::new();
        let headers: Vec<String> = rdr.headers()
            .map(|h| h.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();
        
        // Regex patterns for content we want to skip
        let email_re = regex::Regex::new(r"^[^@\s]+@[^@\s]+\.[^@\s]+$").unwrap();
        let numeric_re = regex::Regex::new(r"^[\d,.+-]+$").unwrap();
        let alphanum_id_re = regex::Regex::new(r"^[a-f0-9]{8,}$").unwrap(); // UUID/hash-like
        
        for (row_idx, result) in rdr.records().enumerate() {
            if let Ok(record) = result {
                let mut text_parts: Vec<String> = Vec::new();
                let mut column_cues: Vec<String> = Vec::new();
                
                for (col_idx, value) in record.iter().enumerate() {
                    let value = value.trim();
                    
                    // Skip if too short
                    if value.len() <= 3 {
                        continue;
                    }
                    
                    // Skip emails
                    if email_re.is_match(value) {
                        continue;
                    }
                    
                    // Skip purely numeric values
                    if numeric_re.is_match(value) {
                        continue;
                    }
                    
                    // Skip hash/UUID-like IDs (lowercase hex, 8+ chars)
                    if alphanum_id_re.is_match(&value.to_lowercase()) {
                        continue;
                    }
                    
                    // Skip if mostly digits (IDs like "12345abc")
                    let digit_ratio = value.chars().filter(|c| c.is_ascii_digit()).count() as f64 / value.len() as f64;
                    if digit_ratio > 0.5 {
                        continue;
                    }
                    
                    // This looks like meaningful text - include it
                    text_parts.push(value.to_string());
                    
                    // Add column name as cue if we have headers
                    if col_idx < headers.len() {
                        let header = headers[col_idx].to_lowercase().replace(" ", "_");
                        if !column_cues.contains(&format!("column:{}", header)) {
                            column_cues.push(format!("column:{}", header));
                        }
                    }
                }
                
                // Skip rows with no meaningful text
                if text_parts.is_empty() {
                    continue;
                }
                
                // Build content from meaningful text columns
                let row_content = text_parts.join(" | ");
                
                // Build cues
                let mut cues = vec![
                    "type:csv_row".to_string(),
                    format!("file:{}", filename.replace(".csv", "").to_lowercase()),
                    format!("row:{}", row_idx + 1),
                ];
                cues.extend(column_cues);
                
                chunks.push(Chunk {
                    content: row_content,
                    start_line: row_idx + 1,
                    end_line: row_idx + 1,
                    context: format!("{}:row_{}", filename, row_idx + 1),
                    structural_cues: cues,
                    category: ChunkCategory::Structured,
                });
            }
        }
        
        chunks
    }

    pub fn chunk_json(content: &str) -> Vec<Chunk> {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(content) {
            // Check for ApiSpec signature
            if value.get("ApiSpec").is_some() || value.get("swagger").is_some() {
                return Self::chunk_apispec_json(&value);
            }
            
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
                    category: ChunkCategory::Structured,
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
                    category: ChunkCategory::Structured,
                }).collect();
            }
        }
        Self::chunk_text(content)
    }

    pub fn chunk_yaml(content: &str) -> Vec<Chunk> {
        if let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(content) {
            // Check for ApiSpec signature
            if value.get("apispec").is_some() || value.get("swagger").is_some() {
                return Self::chunk_apispec_yaml(&value);
            }
            
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
                        category: ChunkCategory::Structured,
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
                        category: ChunkCategory::Structured,
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
        let filename = path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("document")
            .to_string();

        match extension {
            "xlsx" => Self::chunk_excel(path, &filename),
            "docx" => Self::chunk_docx(path, &filename),
            _ => Vec::new(),
        }
    }
    
    /// Chunk DOCX files using sentence-based segmentation
    fn chunk_docx(path: &Path, filename: &str) -> Vec<Chunk> {
        use docx_rs::*;
        
        // Read the DOCX file
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return Vec::new(),
        };
        
        // Parse the DOCX
        let docx = match read_docx(&bytes) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        
        // Extract all text from paragraphs
        let mut all_text = String::new();
        
        for child in docx.document.children {
            if let DocumentChild::Paragraph(para) = child {
                let mut para_text = String::new();
                for child in para.children {
                    if let ParagraphChild::Run(run) = child {
                        for child in run.children {
                            if let RunChild::Text(text) = child {
                                para_text.push_str(&text.text);
                            }
                        }
                    }
                }
                let para_text = para_text.trim();
                if !para_text.is_empty() {
                    all_text.push_str(para_text);
                    all_text.push(' ');
                }
            }
        }
        
        let all_text = all_text.trim();
        if all_text.is_empty() {
            return Vec::new();
        }
        
        // Segment using unicode sentences (same strategy as URL chunking)
        let sentences: Vec<&str> = all_text.unicode_sentences().collect();
        let mut chunks = Vec::new();
        
        const MIN_CHUNK_CHARS: usize = 80;
        const MAX_CHUNK_CHARS: usize = 500;
        const TARGET_SENTENCES: usize = 3;
        
        let mut current_chunk = String::new();
        let mut current_sentence_count = 0;
        let mut chunk_idx = 0;
        
        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            
            let would_be_length = current_chunk.len() + sentence.len() + 1;
            
            if !current_chunk.is_empty() && 
               (would_be_length > MAX_CHUNK_CHARS || current_sentence_count >= TARGET_SENTENCES) {
                if current_chunk.len() >= MIN_CHUNK_CHARS {
                    chunks.push(Chunk {
                        content: current_chunk.trim().to_string(),
                        start_line: chunk_idx,
                        end_line: chunk_idx,
                        context: format!("docx:{}:{}", filename, chunk_idx),
                        structural_cues: vec![
                            "type:docx".to_string(),
                            format!("file:{}", filename.replace(".docx", "").to_lowercase()),
                        ],
                        category: ChunkCategory::Prose,
                    });
                    chunk_idx += 1;
                }
                current_chunk.clear();
                current_sentence_count = 0;
            }
            
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
            current_sentence_count += 1;
        }
        
        // Don't forget the last chunk
        if current_chunk.len() >= MIN_CHUNK_CHARS {
            chunks.push(Chunk {
                content: current_chunk.trim().to_string(),
                start_line: chunk_idx,
                end_line: chunk_idx,
                context: format!("docx:{}:{}", filename, chunk_idx),
                structural_cues: vec![
                    "type:docx".to_string(),
                    format!("file:{}", filename.replace(".docx", "").to_lowercase()),
                ],
                category: ChunkCategory::Prose,
            });
        }
        
        // If no chunks from sentences, return full text as one chunk
        if chunks.is_empty() && !all_text.is_empty() {
            chunks.push(Chunk {
                content: all_text.to_string(),
                start_line: 0,
                end_line: 0,
                context: format!("docx:{}", filename),
                structural_cues: vec![
                    "type:docx".to_string(),
                    format!("file:{}", filename.replace(".docx", "").to_lowercase()),
                ],
                category: ChunkCategory::Prose,
            });
        }
        
        chunks
    }
    
    /// Chunk Excel files with smart filtering - same logic as CSV
    fn chunk_excel(path: &Path, filename: &str) -> Vec<Chunk> {
        use calamine::{Reader, Xlsx, open_workbook};
        
        let mut chunks = Vec::new();
        
        // Regex patterns for content we want to skip
        let email_re = regex::Regex::new(r"^[^@\s]+@[^@\s]+\.[^@\s]+$").unwrap();
        let numeric_re = regex::Regex::new(r"^[\d,.+-]+$").unwrap();
        let alphanum_id_re = regex::Regex::new(r"^[a-f0-9]{8,}$").unwrap();
        
        if let Ok(mut excel) = open_workbook::<Xlsx<_>, _>(path) {
            for sheet_name in excel.sheet_names().to_owned() {
                if let Some(Ok(range)) = excel.worksheet_range(&sheet_name) {
                    let rows: Vec<_> = range.rows().collect();
                    
                    // First row is assumed to be headers
                    let headers: Vec<String> = if !rows.is_empty() {
                        rows[0].iter().map(|c| c.to_string()).collect()
                    } else {
                        Vec::new()
                    };
                    
                    // Process data rows (skip header row)
                    for (row_idx, row) in rows.iter().enumerate().skip(1) {
                        let mut text_parts: Vec<String> = Vec::new();
                        let mut column_cues: Vec<String> = Vec::new();
                        
                        for (col_idx, cell) in row.iter().enumerate() {
                            let value = cell.to_string();
                            let value = value.trim();
                            
                            // Skip if too short
                            if value.len() <= 3 {
                                continue;
                            }
                            
                            // Skip emails
                            if email_re.is_match(value) {
                                continue;
                            }
                            
                            // Skip purely numeric values
                            if numeric_re.is_match(value) {
                                continue;
                            }
                            
                            // Skip hash/UUID-like IDs
                            if alphanum_id_re.is_match(&value.to_lowercase()) {
                                continue;
                            }
                            
                            // Skip if mostly digits
                            let digit_ratio = value.chars().filter(|c| c.is_ascii_digit()).count() as f64 / value.len() as f64;
                            if digit_ratio > 0.5 {
                                continue;
                            }
                            
                            // This looks like meaningful text
                            text_parts.push(value.to_string());
                            
                            // Add column name as cue
                            if col_idx < headers.len() {
                                let header = headers[col_idx].to_lowercase().replace(" ", "_");
                                if !header.is_empty() && !column_cues.contains(&format!("column:{}", header)) {
                                    column_cues.push(format!("column:{}", header));
                                }
                            }
                        }
                        
                        // Skip rows with no meaningful text
                        if text_parts.is_empty() {
                            continue;
                        }
                        
                        let row_content = text_parts.join(" | ");
                        
                        // Build cues
                        let mut cues = vec![
                            "type:excel_row".to_string(),
                            format!("file:{}", filename.replace(".xlsx", "").to_lowercase()),
                            format!("sheet:{}", sheet_name.to_lowercase().replace(" ", "_")),
                            format!("row:{}", row_idx + 1),
                        ];
                        cues.extend(column_cues);
                        
                        chunks.push(Chunk {
                            content: row_content,
                            start_line: row_idx + 1,
                            end_line: row_idx + 1,
                            context: format!("{}:{}:row_{}", filename, sheet_name, row_idx + 1),
                            structural_cues: cues,
                            category: ChunkCategory::Structured,
                        });
                    }
                }
            }
        }
        
        chunks
    }

    fn chunk_text(content: &str) -> Vec<Chunk> {
        Self::chunk_text_with_config(content, &SegmenterConfig::default())
    }

    /// Chunk text using sentence segmentation with configurable sliding window.
    /// This is the core method for longform text processing.
    pub fn chunk_text_with_config(content: &str, config: &SegmenterConfig) -> Vec<Chunk> {
        // Collect sentences using Unicode segmentation
        let sentences: Vec<&str> = content.unicode_sentences().collect();
        
        // If very short content or few sentences, just return as single chunk
        if sentences.len() <= config.window_size || content.len() < config.min_chunk_chars {
            if content.trim().is_empty() {
                return Vec::new();
            }
            return vec![Chunk {
                content: content.to_string(),
                start_line: 0,
                end_line: 0,
                context: "text:full".to_string(),
                structural_cues: vec![
                    "lang:text".to_string(),
                    "type:text_content".to_string(),
                ],
                category: ChunkCategory::Prose,
            }];
        }
        
        // Sliding window over sentences
        let mut chunks = Vec::new();
        let step = config.window_size.saturating_sub(config.overlap).max(1);
        
        // Map sentences back to line numbers if possible, otherwise use cumulative count
        let lines: Vec<&str> = content.lines().collect();
        
        for (chunk_idx, i) in (0..sentences.len()).step_by(step).enumerate() {
            let window_end = (i + config.window_size).min(sentences.len());
            let chunk_content: String = sentences[i..window_end].join(" ");
            
            // Skip if too small or too large
            if chunk_content.len() < config.min_chunk_chars {
                continue;
            }
            
            // Truncate if too large
            let final_content = if chunk_content.len() > config.max_chunk_chars {
                chunk_content.chars().take(config.max_chunk_chars).collect()
            } else {
                chunk_content
            };

            // Estimate line range (approximate for generic text)
            let start_line = (i * lines.len() / sentences.len()).max(1);
            let end_line = (window_end * lines.len() / sentences.len()).min(lines.len());
            
            chunks.push(Chunk {
                content: final_content,
                start_line,
                end_line,
                context: format!("window:{}", chunk_idx),
                structural_cues: vec![
                    "lang:text".to_string(),
                    "type:sentence_window".to_string(),
                    format!("window:{}", chunk_idx),
                ],
                category: ChunkCategory::Prose,
            });
        }
        
        // If no chunks created (edge case), fall back to full content
        if chunks.is_empty() && !content.trim().is_empty() {
            chunks.push(Chunk {
                content: content.to_string(),
                start_line: 1,
                end_line: lines.len(),
                context: "text:full".to_string(),
                structural_cues: vec![
                    "lang:text".to_string(),
                    "type:text_content".to_string(),
                ],
                category: ChunkCategory::Prose,
            });
        }
        
        chunks
    }

    /// Extract operations from ApiSpec JSON spec
    fn chunk_apispec_json(spec: &serde_json::Value) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        
        let api_title = spec.get("info")
            .and_then(|i| i.get("title"))
            .and_then(|t| t.as_str())
            .unwrap_or("api");
        
        if let Some(paths) = spec.get("paths").and_then(|p| p.as_object()) {
            for (path, operations) in paths {
                if let Some(ops) = operations.as_object() {
                    for (method, op_spec) in ops {
                        // Skip non-HTTP method keys like "parameters"
                        if !["get", "post", "put", "patch", "delete", "head", "options"].contains(&method.as_str()) {
                            continue;
                        }
                        
                        let summary = op_spec.get("summary")
                            .and_then(|s| s.as_str())
                            .unwrap_or("");
                        
                        let operation_id = op_spec.get("operationId")
                            .and_then(|s| s.as_str())
                            .unwrap_or("anonymous");
                        
                        let mut cues = vec![
                            "type:api_operation".to_string(),
                            format!("method:{}", method.to_uppercase()),
                            format!("path:{}", path),
                            format!("operation:{}", operation_id),
                            format!("api:{}", api_title),
                        ];
                        
                        // Extract tags
                        if let Some(tags) = op_spec.get("tags").and_then(|t| t.as_array()) {
                            for tag in tags {
                                if let Some(tag_str) = tag.as_str() {
                                    cues.push(format!("tag:{}", tag_str));
                                }
                            }
                        }
                        
                        chunks.push(Chunk {
                            content: format!("{} {} - {}", method.to_uppercase(), path, summary),
                            start_line: 0,
                            end_line: 0,
                            context: format!("api:{}:{}", method, path),
                            structural_cues: cues,
                            category: ChunkCategory::ApiSpec,
                        });
                    }
                }
            }
        }
        
        if chunks.is_empty() {
            // Fall back to regular JSON chunking if no paths found
            return Self::chunk_text(&spec.to_string());
        }
        
        chunks
    }

    /// Extract operations from ApiSpec YAML spec
    fn chunk_apispec_yaml(spec: &serde_yaml::Value) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        
        let api_title = spec.get("info")
            .and_then(|i| i.get("title"))
            .and_then(|t| t.as_str())
            .unwrap_or("api");
        
        if let Some(paths) = spec.get("paths").and_then(|p| p.as_mapping()) {
            for (path_val, operations) in paths {
                let path = path_val.as_str().unwrap_or("");
                if let Some(ops) = operations.as_mapping() {
                    for (method_val, op_spec) in ops {
                        let method = method_val.as_str().unwrap_or("");
                        
                        // Skip non-HTTP method keys
                        if !["get", "post", "put", "patch", "delete", "head", "options"].contains(&method) {
                            continue;
                        }
                        
                        let summary = op_spec.get("summary")
                            .and_then(|s| s.as_str())
                            .unwrap_or("");
                        
                        let operation_id = op_spec.get("operationId")
                            .and_then(|s| s.as_str())
                            .unwrap_or("anonymous");
                        
                        let mut cues = vec![
                            "type:api_operation".to_string(),
                            format!("method:{}", method.to_uppercase()),
                            format!("path:{}", path),
                            format!("operation:{}", operation_id),
                            format!("api:{}", api_title),
                        ];
                        
                        // Extract tags
                        if let Some(tags) = op_spec.get("tags").and_then(|t| t.as_sequence()) {
                            for tag in tags {
                                if let Some(tag_str) = tag.as_str() {
                                    cues.push(format!("tag:{}", tag_str));
                                }
                            }
                        }
                        
                        chunks.push(Chunk {
                            content: format!("{} {} - {}", method.to_uppercase(), path, summary),
                            start_line: 0,
                            end_line: 0,
                            context: format!("api:{}:{}", method, path),
                            structural_cues: cues,
                            category: ChunkCategory::ApiSpec,
                        });
                    }
                }
            }
        }
        
        if chunks.is_empty() {
            return Self::chunk_text(&serde_yaml::to_string(spec).unwrap_or_default());
        }
        
        chunks
    }

    /// Get the category for a file based on its type
    pub fn get_category_for_file(path: &Path) -> ChunkCategory {
        let file_type = Self::detect_type(path);
        match file_type {
            Some(ChunkerType::Python) | Some(ChunkerType::Rust) | Some(ChunkerType::TypeScript) |
            Some(ChunkerType::JavaScript) | Some(ChunkerType::Go) | Some(ChunkerType::Html) |
            Some(ChunkerType::Css) | Some(ChunkerType::Php) | Some(ChunkerType::Java) => ChunkCategory::Code,
            
            Some(ChunkerType::Csv) | Some(ChunkerType::Json) | Some(ChunkerType::Yaml) |
            Some(ChunkerType::Xml) => ChunkCategory::Structured,
            
            Some(ChunkerType::ApiSpec) => ChunkCategory::ApiSpec,
            Some(ChunkerType::SocialExport) => ChunkCategory::Conversation,
            
            Some(ChunkerType::Markdown) | Some(ChunkerType::Pdf) | Some(ChunkerType::Office) |
            Some(ChunkerType::Text) => ChunkCategory::Prose,
            
            None => ChunkCategory::Prose, // Default to Prose for unknown if we somehow get here
        }
    }

    /// Chunk URL content by fetching, extracting readable content, and segmenting.
    /// Uses Mozilla Readability algorithm to strip navbars, ads, and keep main article.
    pub async fn chunk_url(url: &str, parallel: bool) -> Result<Vec<Chunk>, String> {
        use scraper::{Html, Selector};
        
        // 1. Fetch the page with User-Agent (required by Wikipedia and many other sites)
        let client = reqwest::Client::builder()
            .user_agent("CueMap/0.6 (https://cuemap.dev; bot)")
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        
        let response = client.get(url).send().await
            .map_err(|e| format!("Failed to fetch URL: {}", e))?;
        
        let html_content = response.text().await
            .map_err(|e| format!("Failed to read response: {}", e))?;
        
        // 2. Parse base URL for metadata cue
        let parsed_url = url::Url::parse(url)
            .map_err(|e| format!("Invalid URL: {}", e))?;
        let host = parsed_url.host_str().unwrap_or("unknown");
        
        // 3. Extract metadata from HTML before Readability processing
        let document = Html::parse_document(&html_content);
        
        let mut metadata_cues = vec![
            format!("source:url"),
            format!("domain:{}", host),
        ];
        
        // Extract title
        let title_selector = Selector::parse("title").ok();
        let title = title_selector
            .and_then(|s| document.select(&s).next())
            .map(|e| e.text().collect::<String>().trim().to_string())
            .filter(|t| !t.is_empty());
        
        if let Some(ref t) = title {
            metadata_cues.push(format!("title:{}", t.chars().take(100).collect::<String>()));
        }
        
        // Extract meta description
        let meta_selector = Selector::parse("meta[name=\"description\"]").ok();
        let description = meta_selector
            .and_then(|s| document.select(&s).next())
            .and_then(|e| e.value().attr("content"))
            .filter(|d| !d.is_empty());
        
        if let Some(desc) = description {
            metadata_cues.push(format!("description:{}", desc.chars().take(200).collect::<String>()));
        }
        
        // Extract meta keywords
        let keywords_selector = Selector::parse("meta[name=\"keywords\"]").ok();
        if let Some(sel) = keywords_selector {
            if let Some(elem) = document.select(&sel).next() {
                if let Some(kw) = elem.value().attr("content") {
                    for keyword in kw.split(',').take(5) {
                        let kw_clean = keyword.trim();
                        if !kw_clean.is_empty() {
                            metadata_cues.push(format!("keyword:{}", kw_clean));
                        }
                    }
                }
            }
        }
        
        // Extract h1-h6 headings hierarchy
        for level in 1..=6 {
            let heading_selector = Selector::parse(&format!("h{}", level)).ok();
            if let Some(sel) = heading_selector {
                for (idx, heading) in document.select(&sel).enumerate() {
                    if idx >= 3 { break; } // Limit to first 3 per level
                    let heading_text: String = heading.text().collect();
                    let clean_heading = heading_text.trim();
                    if !clean_heading.is_empty() && clean_heading.len() <= 100 {
                        metadata_cues.push(format!("h{}:{}", level, clean_heading));
                    }
                }
            }
        }
        
        // Extract publish date if available (common meta tags)
        let date_selectors = [
            "meta[property=\"article:published_time\"]",
            "meta[name=\"date\"]",
            "meta[name=\"pubdate\"]",
            "time[datetime]",
        ];
        for selector_str in date_selectors {
            if let Some(sel) = Selector::parse(selector_str).ok() {
                if let Some(elem) = document.select(&sel).next() {
                    let date_val = elem.value().attr("content")
                        .or_else(|| elem.value().attr("datetime"));
                    if let Some(date) = date_val {
                        // Extract just the date part (YYYY-MM-DD)
                        let date_clean = date.chars().take(10).collect::<String>();
                        if date_clean.len() >= 10 {
                            metadata_cues.push(format!("date:{}", date_clean));
                            break;
                        }
                    }
                }
            }
        }
        
        // 4. Extract main content using smart element selection
        // Prioritize article and main content areas, exclude navigation/footer/ads
        let article_text = Self::extract_article_content(&document);
        
        // 5. Clean up the extracted text
        let clean_text = article_text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        
        if clean_text.trim().is_empty() {
            return Ok(vec![Chunk {
                content: format!("URL: {} (no content extracted)", url),
                start_line: 0,
                end_line: 0,
                context: title.unwrap_or_else(|| url.to_string()),
                structural_cues: metadata_cues,
                category: ChunkCategory::WebContent,
            }]);
        }
        
        // 6. Segment into sentence chunks (no overlap, semantic boundaries)
        // Strategy: Group 2-3 sentences per chunk for smaller, focused memories
        let sentences: Vec<&str> = clean_text.unicode_sentences().collect();
        // Target 2-3 sentences per chunk, min 80 chars, max 500 chars
        const MIN_CHUNK_CHARS: usize = 80;
        const MAX_CHUNK_CHARS: usize = 500;
        const TARGET_SENTENCES: usize = 3;

        let mut chunks = Vec::new();
        
        if parallel && sentences.len() > 50 {
            // Parallel Processing using Rayon
            use rayon::prelude::*;
            
            // Heuristic: Process in batches of 20 sentences to balance overhead vs parallelism
            let batch_size = 20;
            
            // First, generate chunks in parallel (without correct indices yet)
            let mut raw_chunks: Vec<Chunk> = sentences.par_chunks(batch_size)
                .flat_map(|batch| {
                    let mut batch_chunks = Vec::new();
                    let mut current_chunk = String::new();
                    let mut current_sentence_count = 0;
                    
                    for sentence in batch {
                        let sentence = sentence.trim();
                        if sentence.is_empty() { continue; }
                        
                        let would_be_length = current_chunk.len() + sentence.len() + 1;
                        if !current_chunk.is_empty() && (would_be_length > MAX_CHUNK_CHARS || current_sentence_count >= TARGET_SENTENCES) {
                            if current_chunk.len() >= MIN_CHUNK_CHARS {
                                let mut chunk_cues = metadata_cues.clone();
                                chunk_cues.push("type:web_content".to_string());
                                batch_chunks.push(Chunk {
                                    content: current_chunk.trim().to_string(),
                                    start_line: 0, // Placeholder
                                    end_line: 0,   // Placeholder
                                    context: format!("web:{}", host), // Placeholder suffix
                                    structural_cues: chunk_cues,
                                    category: ChunkCategory::WebContent,
                                });
                            }
                            current_chunk.clear();
                            current_sentence_count = 0;
                        }
                        
                        if !current_chunk.is_empty() { current_chunk.push(' '); }
                        current_chunk.push_str(sentence);
                        current_sentence_count += 1;
                    }
                    
                    if current_chunk.len() >= MIN_CHUNK_CHARS {
                        let mut chunk_cues = metadata_cues.clone();
                        chunk_cues.push("type:web_content".to_string());
                        batch_chunks.push(Chunk {
                            content: current_chunk.trim().to_string(),
                            start_line: 0,
                            end_line: 0,
                            context: format!("web:{}", host),
                            structural_cues: chunk_cues,
                            category: ChunkCategory::WebContent,
                        });
                    }
                    batch_chunks
                })
                .collect();
                
            // Fixup indices sequentially
            for (idx, chunk) in raw_chunks.iter_mut().enumerate() {
                chunk.start_line = idx;
                chunk.end_line = idx;
                chunk.context = format!("web:{}:{}", host, idx);
            }
            chunks = raw_chunks;
            
        } else {
            // Sequential Processing (Original Logic)
            let mut current_chunk = String::new();
            let mut current_sentence_count = 0;
            let mut chunk_idx = 0;
            
            for sentence in sentences {
                let sentence = sentence.trim();
                if sentence.is_empty() {
                    continue;
                }
                
                // Check if adding this sentence would exceed max
                let would_be_length = current_chunk.len() + sentence.len() + 1;
                
                if !current_chunk.is_empty() && 
                   (would_be_length > MAX_CHUNK_CHARS || current_sentence_count >= TARGET_SENTENCES) {
                    // Finalize current chunk if it meets minimum
                    if current_chunk.len() >= MIN_CHUNK_CHARS {
                        let mut chunk_cues = metadata_cues.clone();
                        chunk_cues.push("type:web_content".to_string());
                        
                        chunks.push(Chunk {
                            content: current_chunk.trim().to_string(),
                            start_line: chunk_idx,
                            end_line: chunk_idx,
                            context: format!("web:{}:{}", host, chunk_idx),
                            structural_cues: chunk_cues,
                            category: ChunkCategory::WebContent,
                        });
                        chunk_idx += 1;
                    }
                    current_chunk.clear();
                    current_sentence_count = 0;
                }
                
                // Add sentence to current chunk
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(sentence);
                current_sentence_count += 1;
            }
            
            // Don't forget the last chunk
            if current_chunk.len() >= MIN_CHUNK_CHARS {
                let mut chunk_cues = metadata_cues.clone();
                chunk_cues.push("type:web_content".to_string());
                
                chunks.push(Chunk {
                    content: current_chunk.trim().to_string(),
                    start_line: chunk_idx,
                    end_line: chunk_idx,
                    context: format!("web:{}:{}", host, chunk_idx),
                    structural_cues: chunk_cues,
                    category: ChunkCategory::WebContent,
                });
            }
        }
        
        // If no chunks, return full content as single chunk
        if chunks.is_empty() {
            chunks.push(Chunk {
                content: clean_text,
                start_line: 0,
                end_line: 0,
                context: title.unwrap_or_else(|| url.to_string()),
                structural_cues: metadata_cues,
                category: ChunkCategory::WebContent,
            });
        }
        
        Ok(chunks)
    }

    /// Extract main article content from HTML, filtering out navigation/ads/footers.
    /// Production-ready implementation inspired by Mozilla Readability.
    /// 
    /// Key features:
    /// - Scoped container selection (article > main > .content)
    /// - Block element extraction (p, h1-h6, li, blockquote, pre)
    /// - Ancestor-based noise exclusion
    /// - Double-newline joins to preserve structure for sentence segmentation
    fn extract_article_content(document: &scraper::Html) -> String {
        use scraper::{Selector, ElementRef};
        use std::collections::HashSet;

        // 1. Identify the best container (Scope)
        let content_selectors = [
            "article", "main", "[role=\"main\"]", ".post-content", 
            ".article-content", ".entry-content", "#content", ".content"
        ];
        
        let mut root_element = document.root_element(); // Default to full doc
        
        for selector_str in content_selectors {
            if let Ok(sel) = Selector::parse(selector_str) {
                if let Some(elem) = document.select(&sel).next() {
                    // Heuristic: Don't trap yourself in a tiny container (e.g. empty <main>)
                    // Only accept if it looks substantial (has at least 5 text nodes)
                    if elem.text().count() > 5 { 
                        root_element = elem;
                        break;
                    }
                }
            }
        }

        // 2. Identify Noise to Exclude (Relative to the root_element)
        let exclude_selectors = [
            "nav", "header", "footer", "aside", "script", "style", "noscript",
            ".nav", ".navigation", ".menu", ".sidebar", ".footer", 
            ".ad", ".advertisement", ".social-share", ".cookie-banner"
        ];
        
        let mut excluded_ids = HashSet::new();
        for sel_str in exclude_selectors {
            if let Ok(sel) = Selector::parse(sel_str) {
                for elem in root_element.select(&sel) {
                    excluded_ids.insert(elem.id());
                }
            }
        }

        // 3. Extract Block Elements (Preserving Structure)
        // We grab P, Headers, Lists, Quotes, and Preformatted text.
        let block_selector = Selector::parse("p, h1, h2, h3, h4, h5, h6, li, blockquote, pre").unwrap();
        let mut content_blocks = Vec::new();

        for element in root_element.select(&block_selector) {
            // A. Exclusion Check (Ancestry)
            let mut is_excluded = false;
            let mut current = Some(element);
            
            // Walk up the tree to check if we are inside an excluded node
            while let Some(curr_elem) = current {
                if excluded_ids.contains(&curr_elem.id()) {
                    is_excluded = true;
                    break;
                }
                // Stop if we hit the root container (optimization)
                if curr_elem == root_element { break; }
                current = curr_elem.parent().and_then(ElementRef::wrap);
            }
            
            if is_excluded { continue; }

            // B. Text Extraction
            let text = element.text()
                .collect::<Vec<_>>()
                .join(" ") // Join words within a paragraph with spaces
                .trim()
                .to_string();

            // C. Quality Filter
            // - Headers: Keep even if short ("Introduction")
            // - Paragraphs: Must be meaningful (> 20 chars or end in punctuation)
            let tag = element.value().name();
            let is_header = tag.starts_with('h');
            
            if !text.is_empty() {
                if is_header || text.len() > 20 || text.ends_with('.') || text.ends_with(':') {
                    content_blocks.push(text);
                }
            }
        }

        // 4. Final Join with Double Newlines
        // This preserves the "Block" structure for the sentence segmenter
        if content_blocks.is_empty() {
            // Fallback: If structured extraction failed, grab raw text from body
            return document.root_element().text().collect::<Vec<_>>().join(" ");
        }

        content_blocks.join("\n\n")
    }

    /// Extract links from the main article content area only.
    /// Uses the same content scoping as extract_article_content to avoid
    /// navigation, footer, and sidebar links.
    /// 
    /// Returns a list of absolute URLs found in the main content.
    pub fn extract_content_links(document: &scraper::Html, base_url: &url::Url) -> Vec<String> {
        use scraper::{Selector, ElementRef};
        use std::collections::HashSet;

        // 1. Identify the best container (same as extract_article_content)
        let content_selectors = [
            "article", "main", "[role=\"main\"]", ".post-content", 
            ".article-content", ".entry-content", "#content", ".content"
        ];
        
        let mut root_element = document.root_element();
        
        for selector_str in content_selectors {
            if let Ok(sel) = Selector::parse(selector_str) {
                if let Some(elem) = document.select(&sel).next() {
                    if elem.text().count() > 5 { 
                        root_element = elem;
                        break;
                    }
                }
            }
        }

        // 2. Identify Noise to Exclude
        let exclude_selectors = [
            "nav", "header", "footer", "aside", "script", "style", "noscript",
            ".nav", ".navigation", ".menu", ".sidebar", ".footer", 
            ".ad", ".advertisement", ".social-share", ".cookie-banner",
            ".toc", ".table-of-contents", ".breadcrumb", ".pagination"
        ];
        
        let mut excluded_ids = HashSet::new();
        for sel_str in exclude_selectors {
            if let Ok(sel) = Selector::parse(sel_str) {
                for elem in root_element.select(&sel) {
                    excluded_ids.insert(elem.id());
                }
            }
        }

        // 3. Extract links from content area only
        let link_selector = Selector::parse("a[href]").unwrap();
        let mut links = Vec::new();
        let mut seen = HashSet::new();

        for element in root_element.select(&link_selector) {
            // A. Exclusion Check (Ancestry)
            let mut is_excluded = false;
            let mut current = Some(element);
            
            while let Some(curr_elem) = current {
                if excluded_ids.contains(&curr_elem.id()) {
                    is_excluded = true;
                    break;
                }
                if curr_elem == root_element { break; }
                current = curr_elem.parent().and_then(ElementRef::wrap);
            }
            
            if is_excluded { continue; }

            // B. Extract href
            if let Some(href) = element.value().attr("href") {
                // Skip empty, anchor-only, javascript, and mailto links
                if href.is_empty() 
                    || href.starts_with('#') 
                    || href.starts_with("javascript:") 
                    || href.starts_with("mailto:")
                    || href.starts_with("tel:")
                {
                    continue;
                }

                // Resolve to absolute URL
                let absolute_url = if href.starts_with("http://") || href.starts_with("https://") {
                    href.to_string()
                } else {
                    match base_url.join(href) {
                        Ok(resolved) => resolved.to_string(),
                        Err(_) => continue,
                    }
                };

                // Deduplicate and add
                if !seen.contains(&absolute_url) {
                    seen.insert(absolute_url.clone());
                    links.push(absolute_url);
                }
            }
        }

        links
    }

    // ================== SOCIAL MEDIA EXPORT PARSERS ==================

    /// Parse WhatsApp chat export (.txt format)
    /// Format: [M/D/YY, HH:MM:SS] Sender: Message
    /// Creates ONE MEMORY PER MESSAGE with clean content and structured cues
    fn chunk_whatsapp(content: &str) -> Vec<Chunk> {
        let re = regex::Regex::new(
            r"(?m)^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*([^:]+):\s*(.+)$"
        ).unwrap();
        
        let mut chunks = Vec::new();
        
        for (idx, cap) in re.captures_iter(content).enumerate() {
            let date = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let _time = cap.get(2).map(|m| m.as_str()).unwrap_or("");
            let sender = cap.get(3).map(|m| m.as_str().trim()).unwrap_or("");
            let msg = cap.get(4).map(|m| m.as_str().trim()).unwrap_or("");
            
            // Skip media omitted messages
            if msg.contains("image omitted") || msg.contains("sticker omitted") 
               || msg.contains("video omitted") || msg.contains("audio omitted")
               || msg.contains("document omitted") || msg.contains("GIF omitted") {
                continue;
            }
            
            // Skip very short messages (reactions, single emojis)
            if msg.chars().count() < 3 {
                continue;
            }
            
            // Clean sender name for cue
            let sender_cue = sender.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
                .replace(" ", "_");
            
            // Normalize date to ISO-like format (replace / with -)
            let date_cue = date.replace("/", "-");
            
            // Build clean cues - only meaningful metadata, no numeric noise
            let cues = vec![
                "platform:whatsapp".to_string(),
                "type:message".to_string(),
                format!("sender:{}", sender_cue),
                format!("date:{}", date_cue),
            ];
            
            chunks.push(Chunk {
                content: msg.to_string(),  // Just the message text, clean
                start_line: idx + 1,
                end_line: idx + 1,
                context: format!("{}: {}", sender, date),  // Human-readable context
                structural_cues: cues,
                category: ChunkCategory::Conversation,
            });
        }
        
        if chunks.is_empty() {
            // Fallback: treat as text
            return Self::chunk_text(content);
        }
        
        chunks
    }

    /// Parse Instagram DM export (JSON format)
    /// Format: [{sender_name, timestamp_ms, content, share?, reactions?}]
    /// Creates ONE MEMORY PER MESSAGE with clean content
    fn chunk_instagram(content: &str) -> Vec<Chunk> {
        let parsed: Vec<serde_json::Value> = match serde_json::from_str(content) {
            Ok(v) => v,
            Err(_) => return Self::chunk_json(content), // Fallback
        };
        
        let mut chunks = Vec::new();
        
        for (idx, msg) in parsed.iter().enumerate() {
            let sender = msg.get("sender_name").and_then(|v| v.as_str()).unwrap_or("unknown");
            let text = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let ts = msg.get("timestamp_ms").and_then(|v| v.as_i64()).unwrap_or(0);
            
            // Skip empty messages and reactions
            if text.is_empty() || text == "Liked a message" || text.contains("sent an attachment") {
                continue;
            }
            
            // Convert timestamp to date string
            let date_cue = if ts > 0 {
                let secs = ts / 1000;
                chrono::DateTime::from_timestamp(secs, 0)
                    .map(|dt| dt.format("%Y-%m-%d").to_string())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            
            // Clean sender name for cue
            let sender_cue = sender.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>();
            
            // Build clean cues
            let mut cues = vec![
                "platform:instagram".to_string(),
                "type:dm".to_string(),
                format!("sender:{}", sender_cue),
            ];
            
            if !date_cue.is_empty() {
                cues.push(format!("date:{}", date_cue));
            }
            
            // Handle shared content - add as URL cue
            if let Some(share_link) = msg.get("share").and_then(|s| s.get("link")).and_then(|v| v.as_str()) {
                cues.push("has:shared_link".to_string());
                // Add domain as cue
                if let Ok(url) = url::Url::parse(share_link) {
                    if let Some(domain) = url.host_str() {
                        cues.push(format!("domain:{}", domain.replace(".", "_")));
                    }
                }
            }
            
            chunks.push(Chunk {
                content: text.to_string(),  // Just the message text
                start_line: idx + 1,
                end_line: idx + 1,
                context: format!("{}: {}", sender, date_cue),
                structural_cues: cues,
                category: ChunkCategory::Conversation,
            });
        }
        
        chunks
    }

    /// Parse Chrome History export (JSON format)
    /// Format: {"Browser History": [{title, url, time_usec}]}
    /// Creates ONE MEMORY PER PAGE VISIT with title as content
    fn chunk_chrome_history(content: &str) -> Vec<Chunk> {
        let parsed: serde_json::Value = match serde_json::from_str(content) {
            Ok(v) => v,
            Err(_) => return Self::chunk_json(content),
        };
        
        let history = match parsed.get("Browser History").and_then(|h| h.as_array()) {
            Some(arr) => arr,
            None => return Self::chunk_json(content),
        };
        
        let mut chunks = Vec::new();
        let mut seen_urls: std::collections::HashSet<String> = std::collections::HashSet::new();
        
        for (idx, entry) in history.iter().enumerate() {
            let title = entry.get("title").and_then(|v| v.as_str()).unwrap_or("Untitled");
            let url = entry.get("url").and_then(|v| v.as_str()).unwrap_or("");
            let time_usec = entry.get("time_usec").and_then(|v| v.as_i64()).unwrap_or(0);
            
            // Skip empty titles and duplicates
            if title.is_empty() || title == "Untitled" || seen_urls.contains(url) {
                continue;
            }
            seen_urls.insert(url.to_string());
            
            // Extract domain
            let domain = url::Url::parse(url)
                .ok()
                .and_then(|u| u.host_str().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown".to_string());
            
            // Convert timestamp to date
            let date_cue = if time_usec > 0 {
                let secs = time_usec / 1_000_000;
                chrono::DateTime::from_timestamp(secs, 0)
                    .map(|dt| dt.format("%Y-%m-%d").to_string())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            
            // Build clean cues
            let mut cues = vec![
                "platform:chrome".to_string(),
                "type:page_visit".to_string(),
                format!("domain:{}", domain.replace(".", "_")),
            ];
            
            if !date_cue.is_empty() {
                cues.push(format!("date:{}", date_cue));
            }
            
            chunks.push(Chunk {
                content: title.to_string(),  // Just the page title
                start_line: idx + 1,
                end_line: idx + 1,
                context: domain.clone(),
                structural_cues: cues,
                category: ChunkCategory::Conversation,
            });
        }
        
        chunks
    }

    /// Parse YouTube watch/search history export (HTML format)
    /// Creates ONE MEMORY PER VIDEO/SEARCH with title as content
    fn chunk_youtube_history(content: &str) -> Vec<Chunk> {
        // Regex to extract video info from YouTube Takeout HTML
        let video_re = regex::Regex::new(
            r#"Watched\s*<a\s+href="(https://www\.youtube\.com/watch\?v=[^"]+)">([^<]+)</a>.*?(\w+ \d+, \d{4})"#
        ).unwrap();
        
        let search_re = regex::Regex::new(
            r#"Searched for\s*<a[^>]*>([^<]+)</a>.*?(\w+ \d+, \d{4})"#
        ).unwrap();
        
        let mut chunks = Vec::new();
        let mut seen_titles: std::collections::HashSet<String> = std::collections::HashSet::new();
        
        // Process watched videos
        for (idx, cap) in video_re.captures_iter(content).enumerate() {
            let _url = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let title = cap.get(2).map(|m| m.as_str()).unwrap_or("");
            let date = cap.get(3).map(|m| m.as_str()).unwrap_or("");
            
            // Skip duplicates and empty
            if title.is_empty() || seen_titles.contains(title) {
                continue;
            }
            seen_titles.insert(title.to_string());
            
            // Normalize date
            let date_cue = date.replace(" ", "_").replace(",", "");
            
            let cues = vec![
                "platform:youtube".to_string(),
                "type:watched".to_string(),
                format!("date:{}", date_cue),
            ];
            
            chunks.push(Chunk {
                content: title.to_string(),  // Just the video title
                start_line: idx + 1,
                end_line: idx + 1,
                context: format!("YouTube: {}", date),
                structural_cues: cues,
                category: ChunkCategory::Conversation,
            });
        }
        
        // Process searches
        for (idx, cap) in search_re.captures_iter(content).enumerate() {
            let query = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let date = cap.get(2).map(|m| m.as_str()).unwrap_or("");
            
            // Skip duplicates and empty
            if query.is_empty() || seen_titles.contains(query) {
                continue;
            }
            seen_titles.insert(query.to_string());
            
            // Normalize date
            let date_cue = date.replace(" ", "_").replace(",", "");
            
            let cues = vec![
                "platform:youtube".to_string(),
                "type:search".to_string(),
                format!("date:{}", date_cue),
            ];
            
            chunks.push(Chunk {
                content: query.to_string(),  // Just the search query
                start_line: chunks.len() + idx + 1,
                end_line: chunks.len() + idx + 1,
                context: format!("YouTube Search: {}", date),
                structural_cues: cues,
                category: ChunkCategory::Conversation,
            });
        }
        
        if chunks.is_empty() {
            // Fallback to HTML parsing
            return Self::chunk_html(content);
        }
        
        chunks
    }
}
