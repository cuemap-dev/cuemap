use cuemap::multi_tenant::MultiTenantEngine;
use cuemap::structures::LexiconStats;
use cuemap::nl::{tokenize_to_cues_with_lang, Language};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::sync::Arc;
use std::time::Instant;
use cuemap::jobs::is_lexicon_trainable;
use cuemap::config::CueGenStrategy;
use cuemap::semantic::SemanticEngine;
use std::env;

fn strip_html(html: &str) -> String {
    let mut clean = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut current_tag = String::new();
    let skip_content = false;

    for c in html.chars() {
        if c == '<' {
            in_tag = true;
            current_tag.clear();
        } else if c == '>' {
            in_tag = false;
            // We could skip large pre blocks here if needed, 
            // but for now we'll just clean the tags.
        } else if in_tag {
            current_tag.push(c);
        } else if !in_tag && !skip_content {
            clean.push(c);
        }
    }
    
    // Decoding common entities manually or via quick-xml
    let unescaped = quick_xml::escape::unescape(&clean).unwrap_or(std::borrow::Cow::Borrowed(&clean));
    unescaped.into_owned()
}

fn is_so_noise(token: &str) -> bool {
    let noise = [
        "code", "pre", "gt", "lt", "amp", "quot", "using", "used", "use", 
        "get", "try", "can", "not", "one", "way", "would", "also", "know", 
        "like", "want", "need", "problem", "issue", "error", "found"
    ];
    noise.contains(&token)
}

fn tag_to_lang(tag: &str) -> Language {
    match tag {
        t if t.contains("rust") => Language::Rust,
        t if t.contains("python") => Language::Python,
        t if t.contains("go") || t == "golang" => Language::Go,
        t if t.contains("javascript") || t == "js" || t == "reactjs" => Language::JavaScript,
        t if t.contains("typescript") || t == "ts" => Language::TypeScript,
        t if t.contains("java") && !t.contains("javascript") => Language::Java,
        t if t.contains("php") => Language::Php,
        t if t.contains("css") || t.contains("scss") || t.contains("less") => Language::Css,
        _ => Language::Default,
    }
}

fn extract_tags(tags_str: &str) -> Vec<String> {
    tags_str.split(|c| c == '|' || c == '<' || c == '>')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_Posts.xml> [score_threshold=100] [max_posts=0] [target_tag='']", args[0]);
        std::process::exit(1);
    }
    
    let xml_path = &args[1];
    let score_threshold: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let max_posts: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0); // 0 means unlimited
    let target_tag: Option<String> = args.get(4).map(|s| s.to_lowercase());

    println!("Initializing Cuemap engine for SO Lexicon Ingestion...");
    let semantic_engine = SemanticEngine::new(Some(std::path::Path::new("data")));
    let mt_engine = Arc::new(MultiTenantEngine::new(CueGenStrategy::Default, semantic_engine));
    
    let project_id = if let Some(ref tag) = target_tag {
        format!("stackoverflow_lexicon_{}", tag.trim_end_matches('*'))
    } else {
        "stackoverflow_lexicon".to_string()
    };
    let ctx = mt_engine.get_or_create_project(project_id.clone()).expect("Failed to create project");

    println!("Reading {} with score threshold > {}", xml_path, score_threshold);
    let mut reader = Reader::from_file(xml_path)?;
    reader.config_mut().trim_text(true);

    let mut buf = Vec::new();
    let start_time = Instant::now();
    let mut posts_processed = 0;
    let mut skip_count = 0;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(ref e)) | Ok(Event::Start(ref e)) => {
                if e.name().into_inner() == b"row" {
                    let mut is_question = false;
                    let mut score = 0;
                    let mut body = String::new();
                    let mut tags = String::new();
                    let mut title = String::new();

                    for attr in e.attributes() {
                        if let Ok(attr) = attr {
                            match attr.key.into_inner() {
                                b"PostTypeId" => {
                                    if let Ok(v) = std::str::from_utf8(&attr.value) {
                                        if v == "1" || v == "2" {
                                            is_question = true; // both 1 and 2 are good to process
                                        }
                                    }
                                }
                                b"Score" => {
                                    if let Ok(v) = std::str::from_utf8(&attr.value) {
                                        score = v.parse().unwrap_or(0);
                                    }
                                }
                                b"Body" => {
                                    if let Ok(v) = std::str::from_utf8(&attr.value) {
                                        body = v.to_string();
                                    }
                                }
                                b"Tags" => {
                                    if let Ok(v) = std::str::from_utf8(&attr.value) {
                                        tags = v.to_string();
                                    }
                                }
                                b"Title" => {
                                    if let Ok(v) = std::str::from_utf8(&attr.value) {
                                        title = v.to_string();
                                    }
                                }
                                _ => {}
                            }
                        }
                    }

                    if is_question && score >= score_threshold {
                        let extracted_tags = extract_tags(&tags);
                        let tag_match = match &target_tag {
                            Some(tag) => {
                                if tag.ends_with('*') {
                                    let prefix = &tag[..tag.len() - 1];
                                    let hyphen_prefix = format!("{}-", prefix);
                                    extracted_tags.iter().any(|t| t == prefix || t.starts_with(&hyphen_prefix))
                                } else {
                                    extracted_tags.contains(tag)
                                }
                            },
                            None => true,
                        };

                        if !tag_match {
                            skip_count += 1;
                            continue;
                        }

                        let clean_body = strip_html(&body);
                        let full_text = format!("{} {}", title, clean_body);

                        // 3. Update co-occurrence matrix robustly
                        // First, link all tags together with a BOOST (5 tokens worth of weight)
                        for _ in 0..5 {
                            ctx.main.update_cue_co_occurrence(&extracted_tags);
                        }
                        
                        // Ensure tags themselves are in the lexicon
                        for tag in &extracted_tags {
                            ctx.lexicon.upsert_memory_with_id(
                                tag.clone(),
                                tag.clone(),
                                vec![tag.clone()],
                                None,
                                Some(LexiconStats::default()),
                                false,
                                false
                            );
                        }

                        // Then, link every body token ONLY to the post tags (bipartite)
                        for tag in &extracted_tags {
                            let lang = tag_to_lang(tag);
                            let tokens = tokenize_to_cues_with_lang(&full_text, lang);

                            for token in &tokens {
                                if !is_so_noise(token) && is_lexicon_trainable(token) {
                                    // Increment global counts for the body token
                                    *ctx.main.cue_global_counts.entry(token.clone()).or_insert(0) += 1;

                                    ctx.main.increment_co_occurrence(token, tag);
                                    
                                    // Also index the token in lexicon if not noise
                                    ctx.lexicon.upsert_memory_with_id(
                                        token.clone(),
                                        token.clone(),
                                        vec![token.clone()],
                                        None,
                                        Some(LexiconStats::default()),
                                        false,
                                        false
                                    );
                                }
                            }
                        }

                        posts_processed += 1;
                        if posts_processed % 1000 == 0 {
                            println!("Processed {} posts... (skipped {})", posts_processed, skip_count);
                        }
                        
                        if max_posts > 0 && posts_processed >= max_posts {
                            break;
                        }
                    } else {
                        skip_count += 1;
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                eprintln!("Error at position {}: {:?}", reader.buffer_position(), e);
                break;
            }
            _ => (), // There are several other events we don't care about
        }
        buf.clear();
    }

    println!("\nIngestion Complete in {:.2?}", start_time.elapsed());
    println!("Processed: {}, Skipped: {}", posts_processed, skip_count);
    
    // Save project state to disk
    println!("Saving Lexicon & Matrix to disk...");
    match mt_engine.save_project(&project_id) {
        Ok(_) => println!("Successfully saved project {}!", project_id),
        Err(e) => eprintln!("Failed to save project: {}", e),
    }

    Ok(())
}
