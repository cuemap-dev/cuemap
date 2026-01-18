use std::path::Path;

fn main() {
    // Only build nlprule binaries if they don't exist
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let tokenizer_bin = Path::new(&out_dir).join("en_tokenizer.bin");
    let rules_bin = Path::new(&out_dir).join("en_rules.bin");
    
    if !tokenizer_bin.exists() || !rules_bin.exists() {
        // Build the English tokenizer and rules binaries
        // This will download the necessary data files and compile them
        nlprule_build::BinaryBuilder::new(&["en"], &out_dir)
            .build()
            .expect("Failed to build nlprule binaries");
    }
    
    // Tell Cargo to re-run if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}
