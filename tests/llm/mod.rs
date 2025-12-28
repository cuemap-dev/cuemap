use cuemap_rust::llm::*;

#[test]
fn test_extract_facts_parsing_robustness() {
    let content = "Some content here";
    
    // 1. Clean JSON
    let resp1 = r#"{"summary": "Test summary", "cues": ["a:b", "c:d"]}"#;
    let (s1, c1) = parse_extraction_response(resp1, content);
    assert_eq!(s1, "Test summary");
    assert_eq!(c1, vec!["a:b", "c:d"]);
    
    // 2. Markdown Wrapped JSON
    let resp2 = r#"```json
{"summary": "Markdown summary", "cues": ["x:y"]}
```"#;
    let (s2, c2) = parse_extraction_response(resp2, content);
    assert_eq!(s2, "Markdown summary");
    assert_eq!(c2, vec!["x:y"]);
    
    // 3. Mixed Format Cues
    let resp3 = r#"{
        "summary": "Mixed cues",
        "cues": [
            "direct:cue",
            {"key": "obj", "value": "cue"},
            "badcue"
        ]
    }"#;
    let (s3, c3) = parse_extraction_response(resp3, content);
    assert_eq!(s3, "Mixed cues");
    assert!(c3.contains(&"direct:cue".to_string()));
    assert!(c3.contains(&"obj:cue".to_string()));
    
    // 4. Broken JSON with Regex Fallback
    let resp4 = "Here are the cues: 'tag:value', 'another:one'. Sorry, I couldn't format as JSON.";
    let (s4, c4) = parse_extraction_response(resp4, content);
    assert!(s4.len() > 0);
    assert!(c4.contains(&"tag:value".to_string()));
    assert!(c4.contains(&"another:one".to_string()));
}

#[test]
fn test_propose_cues_parsing() {
    // 1. Array-based JSON
    let resp1 = r#"{"cues": ["topic:tax", "intent:calc"]}"#;
    let cues1 = parse_proposal_response(resp1).unwrap();
    assert_eq!(cues1.len(), 2);
    
    // 2. Junk text around JSON
    let resp2 = "Sure, here: \n {\"cues\": [\"a:b\"]} \n Hope that helps!";
    let cues2 = parse_proposal_response(resp2).unwrap();
    assert_eq!(cues2, vec!["a:b"]);
    
    // 3. Malformed JSON with regex recovery
    let resp3 = "No JSON, but cues are 'found:it' and 'recovered:true'.";
    let cues3 = parse_proposal_response(resp3).unwrap();
    assert!(cues3.contains(&"found:it".to_string()));
    assert!(cues3.contains(&"recovered:true".to_string()));
}
