use cuemap::nl::stem_word;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[test]
fn test_embedded_dictionary_smoke() {
    let cases = [
        ("favourites", "favorite"),
        ("night-watchmen", "night-watchman"),
        ("sustaining", "sustain"),
        ("homeworlds", "homeworld"),
    ];

    for (word, expected) in cases {
        assert_eq!(stem_word(word), expected, "failed to lemmatize {word}");
    }
}

#[test]
#[ignore = "requires the optional tests/data/verbs.csv and tests/data/nouns.csv quality corpus"]
fn test_generate_dictionary_and_verify() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let data_dir = PathBuf::from(manifest_dir).join("tests/data");

    assert!(
        data_dir.join("verbs.csv").is_file() && data_dir.join("nouns.csv").is_file(),
        "optional lemmatization corpus is missing; add tests/data/verbs.csv and tests/data/nouns.csv before running this ignored quality test"
    );

    let mut overrides: HashMap<String, String> = HashMap::new();
    let standalone_lemmas = collect_standalone_lemmas(&data_dir);
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

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 5 {
                continue;
            }

            let base = parts[0].to_lowercase();
            // Runtime stemming works on tokenized words. Multi-word and hyphenated
            // verb lemmas must not create single-token overrides such as
            // `dog -> hot` from the phrase `hot dog`.
            if base.contains('-') || base.contains(' ') {
                continue;
            }

            // Forms: 3rd-person, past, past-part, pres-part
            let forms = vec![parts[1], parts[2], parts[3], parts[4]];

            for form in forms {
                total_checks += 1;
                let form_lower = form.to_lowercase();
                if form_lower == base {
                    continue;
                } // No stemming needed

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
            if parts.len() < 2 {
                continue;
            }

            let singular = parts[0].to_lowercase();
            let plural = parts[1].to_lowercase();

            // Skip non-words
            if !singular.chars().all(|c| c.is_alphabetic() || c == '-') {
                continue;
            }
            if plural.contains(' ') {
                continue;
            }
            if singular == plural {
                continue;
            }
            if is_ambiguous_standalone_noun_form(&plural, &singular, &standalone_lemmas) {
                continue;
            }

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
    assert!(
        overrides.len() < 1000,
        "Too many lemmatization mismatches ({})!",
        overrides.len()
    );

    // Also assert that we are actually using the dictionary
    // If dictionary wasn't working, we'd have ~70k failures
    assert!(
        nlprule_correct > 100_000,
        "nlprule/dictionary integration seems broken"
    );
}

fn collect_standalone_lemmas(data_dir: &Path) -> HashSet<String> {
    let mut lemmas = HashSet::new();

    let verbs_path = data_dir.join("verbs.csv");
    if let Ok(file) = File::open(verbs_path) {
        let reader = BufReader::new(file);
        for line in reader.lines().map_while(Result::ok) {
            if let Some(base) = line.split('\t').next() {
                lemmas.insert(base.to_lowercase());
            }
        }
    }

    let nouns_path = data_dir.join("nouns.csv");
    if let Ok(file) = File::open(nouns_path) {
        let reader = BufReader::new(file);
        for line in reader.lines().map_while(Result::ok) {
            if let Some(singular) = line.split(',').next() {
                lemmas.insert(singular.to_lowercase());
            }
        }
    }

    lemmas
}

fn is_ambiguous_standalone_noun_form(
    plural: &str,
    singular: &str,
    standalone_lemmas: &HashSet<String>,
) -> bool {
    standalone_lemmas.contains(plural) && !is_regular_noun_form(plural, singular)
}

fn is_regular_noun_form(plural: &str, singular: &str) -> bool {
    if plural == singular || plural.starts_with(singular) || singular.starts_with(plural) {
        return true;
    }

    if let Some(stem) = plural.strip_suffix("ies") {
        if singular == format!("{}y", stem) {
            return true;
        }
    }

    if let Some(stem) = plural.strip_suffix("ves") {
        if singular == format!("{}f", stem) || singular == format!("{}fe", stem) {
            return true;
        }
    }

    ["es", "s"].iter().any(|suffix| {
        plural
            .strip_suffix(suffix)
            .is_some_and(|stem| !stem.is_empty() && singular.starts_with(stem))
    })
}
