use std::collections::HashSet;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let output_path = Path::new(&out_dir).join("combined_kernel.cu");

    let mut processed = HashSet::new();
    let kernel_source = process_file("kernels/combined_vanity.cu", &mut processed);

    fs::write(&output_path, kernel_source).expect("Failed to write combined kernel");
}

fn process_file(file_path: &str, processed: &mut HashSet<String>) -> String {
    let canonical = fs::canonicalize(file_path)
        .unwrap_or_else(|_| panic!("Failed to find: {}", file_path));

    let canonical_str = canonical.to_string_lossy().to_string();

    if processed.contains(&canonical_str) {
        return String::new();
    }
    processed.insert(canonical_str.clone());

    let content = fs::read_to_string(&canonical)
        .unwrap_or_else(|_| panic!("Failed to read: {}", file_path));

    let base_dir = canonical
        .parent()
        .expect("No parent directory")
        .to_path_buf();

    let mut result = String::new();

    for line in content.lines() {
        let trimmed = line.trim();

        if let Some(include_path) = extract_local_include(trimmed) {
            // Resolve include path relative to kernels directory
            let full_path = if include_path.starts_with("primitives/") {
                Path::new("kernels").join(&include_path)
            } else {
                base_dir.join(&include_path)
            };

            result.push_str(&format!("// BEGIN: {}\n", include_path));
            result.push_str(&process_file(full_path.to_string_lossy().as_ref(), processed));
            result.push_str(&format!("// END: {}\n", include_path));
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}

fn extract_local_include(line: &str) -> Option<String> {
    if !line.starts_with("#include \"") {
        return None;
    }

    let rest = &line[10..]; // Skip '#include "'
    let end_quote = rest.find('"')?;
    Some(rest[..end_quote].to_string())
}
