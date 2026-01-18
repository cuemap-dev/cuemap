use cuemap_rust::nl::stem_word;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::collections::HashMap;

#[test]
fn test_generate_dictionary_and_verify() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let data_dir = PathBuf::from(manifest_dir).join("tests/data");
    
    let mut overrides: HashMap<String, String> = HashMap::new();
    let mut total_checks = 0;
    let mut nlprule_correct = 0;

    // 1. Process Verbs
    let verbs_path = data_dir.join("verbs.csv");
    if verbs_path.exists() {
        println!("Processing verbs.csv...");
        let file = File::open(verbs_path).unwrap();
        let reader = BufReader::new(file);
        
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 5 { continue; }
            
            let base = parts[0].to_lowercase();
            // Skip multi-word verbs
            if base.contains('-') { continue; }
            
            // Forms: 3rd-person, past, past-part, pres-part
            let forms = vec![parts[1], parts[2], parts[3], parts[4]];
            
            for form in forms {
                total_checks += 1;
                let form_lower = form.to_lowercase();
                if form_lower == base { continue; } // No stemming needed
                
                let stemmed = stem_word(&form_lower);
                if stemmed != base {
                    // Mismatch! nlprule failed to produce the base form expected by dataset
                    overrides.insert(form_lower, base.clone());
                } else {
                    nlprule_correct += 1;
                }
            }
        }
    } else {
        println!("verbs.csv not found");
    }

    // 2. Process Nouns
    let nouns_path = data_dir.join("nouns.csv");
    if nouns_path.exists() {
        println!("Processing nouns.csv...");
        let file = File::open(nouns_path).unwrap();
        let reader = BufReader::new(file);
        
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 2 { continue; }
            
            let singular = parts[0].to_lowercase();
            let plural = parts[1].to_lowercase();
            
            // Skip non-words
            if !singular.chars().all(|c| c.is_alphabetic() || c == '-') { continue; }
            if singular == plural { continue; }
            
            total_checks += 1;
            let stemmed = stem_word(&plural);
            
            if stemmed != singular {
                // Mismatch
                overrides.insert(plural, singular);
            } else {
                nlprule_correct += 1;
            }
        }
    } else {
        println!("nouns.csv not found");
    }

    println!("Total Checks: {}", total_checks);
    println!("nlprule Correct: {}", nlprule_correct);
    println!("Exceptions Found: {}", overrides.len());
    
    // We expect near 100% coverage (minus homonyms/ambiguities in dataset)
    // 572 failures in 145975 checks = 99.6% accuracy
    assert!(overrides.len() < 1000, "Too many lemmatization mismatches ({})!", overrides.len());
    
    // Also assert that we are actually using the dictionary
    // If dictionary wasn't working, we'd have ~70k failures
    assert!(nlprule_correct > 100_000, "nlprule/dictionary integration seems broken");
}
