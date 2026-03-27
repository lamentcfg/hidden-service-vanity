use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

pub fn validate_prefix(prefix: &str) -> Result<()> {
    if prefix.is_empty() {
        return Ok(());
    }
    for c in prefix.to_lowercase().chars() {
        if !matches!(c, 'a'..='z' | '2'..='7') {
            anyhow::bail!(
                "Invalid character '{}' in prefix '{}'. Base32 only allows a-z and 2-7.",
                c,
                prefix
            );
        }
    }
    Ok(())
}

/// Load prefixes from a file. Each line should contain one prefix.
/// Lines starting with # are treated as comments and ignored.
/// Empty lines and whitespace-only lines are ignored.
pub fn load_prefixes_from_file(path: &PathBuf) -> Result<Vec<String>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open prefix list file: {}", path.display()))?;

    let reader = BufReader::new(file);
    let mut prefixes = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "Failed to read line {} from file: {}",
                line_num + 1,
                path.display()
            )
        })?;

        // Remove comments (everything after #)
        let line = if let Some(hash_pos) = line.find('#') {
            &line[..hash_pos]
        } else {
            &line
        };

        // Trim whitespace and skip empty lines
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        prefixes.push(trimmed.to_string());
    }

    Ok(prefixes)
}

// Must match CUDA kernel implementation exactly
pub fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

pub fn estimate_attempts(prefix_len: usize) -> u64 {
    1u64 << (5 * prefix_len.min(12))
}

/// Find which prefix (if any) an address matches. Returns the index of the matching prefix.
pub fn find_matching_prefix(address: &str, prefixes: &[String]) -> Option<usize> {
    let address_lower = address.to_lowercase();
    for (i, prefix) in prefixes.iter().enumerate() {
        if address_lower.starts_with(&prefix.to_lowercase()) {
            return Some(i);
        }
    }
    None
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn generate_keypair(seed: [u8; 32]) -> ([u8; 32], [u8; 32]) {
    let signing_key = SigningKey::from_bytes(&seed);
    let verifying_key = signing_key.verifying_key();
    (seed, verifying_key.to_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_prefix_valid() {
        assert!(validate_prefix("abc").is_ok());
        assert!(validate_prefix("test").is_ok());
        assert!(validate_prefix("237").is_ok());
        assert!(validate_prefix("abc237xyz").is_ok());
        assert!(validate_prefix("ABC").is_ok());
        assert!(validate_prefix("TeSt237").is_ok());
    }

    #[test]
    fn test_validate_prefix_invalid() {
        assert!(validate_prefix("abc1").is_err());
        assert!(validate_prefix("test0").is_err());
        assert!(validate_prefix("abc89").is_err());
        assert!(validate_prefix("test!").is_err());
        assert!(validate_prefix("test_").is_err());
        assert!(validate_prefix("test ").is_err());
    }

    #[test]
    fn test_xorshift64() {
        let state1 = xorshift64(1);
        let state2 = xorshift64(state1);
        let state3 = xorshift64(state2);
        assert_eq!(state1, xorshift64(1));
        assert_eq!(state2, xorshift64(state1));
        assert_eq!(state3, xorshift64(state2));
        assert_ne!(state1, 1);
    }

    #[test]
    fn test_estimate_attempts() {
        assert_eq!(estimate_attempts(1), 32);
        assert_eq!(estimate_attempts(2), 1024);
        assert_eq!(estimate_attempts(3), 32768);
        assert_eq!(estimate_attempts(4), 1_048_576);
        assert_eq!(estimate_attempts(5), 33_554_432);
    }

    #[test]
    fn test_keypair_generation() {
        let seed = [0x42u8; 32];
        let (returned_seed, pubkey) = generate_keypair(seed);
        assert_eq!(returned_seed, seed);
        assert_eq!(pubkey.len(), 32);
        let (seed2, pubkey2) = generate_keypair(seed);
        assert_eq!(seed, seed2);
        assert_eq!(pubkey, pubkey2);
    }

    #[test]
    fn test_ed25519_debug() {
        // RFC 8032 Section 7.1 test vector
        let seed: [u8; 32] = [
            0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60,
            0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c, 0xc4,
            0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19,
            0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae, 0x7f, 0x0c,
        ];
        let expected_pubkey: [u8; 32] = [
            0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7,
            0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07, 0x3a,
            0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25,
            0xaf, 0x02, 0x1a, 0x68, 0xf7, 0x07, 0x51, 0x1a,
        ];

        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();
        let pubkey = verifying_key.to_bytes();

        println!("Seed: {:?}", seed);
        println!("Expected pubkey: {:?}", expected_pubkey);
        println!("Actual pubkey:   {:?}", pubkey);

        // Check if the seed bytes match what we expect
        let signing_key_bytes = signing_key.to_bytes();
        println!("Signing key bytes: {:?}", signing_key_bytes);
    }

    #[test]
    fn test_keypair_different_seeds() {
        let seed1 = [0x01u8; 32];
        let seed2 = [0x02u8; 32];
        let (_, pubkey1) = generate_keypair(seed1);
        let (_, pubkey2) = generate_keypair(seed2);
        assert_ne!(pubkey1, pubkey2);
    }

    #[test]
    fn test_load_prefixes_from_file() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_prefixes.txt");

        let content = "# This is a comment\nabc\n# another comment\ntest\n\n   spaced   \n# end\nxyz";
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        drop(file);

        let prefixes = load_prefixes_from_file(&file_path).unwrap();

        assert_eq!(prefixes, vec!["abc", "test", "spaced", "xyz"]);

        // Cleanup
        std::fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_load_prefixes_from_file_empty() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_empty_prefixes.txt");

        let content = "# Only comments\n\n# and empty lines\n";
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        drop(file);

        let prefixes = load_prefixes_from_file(&file_path).unwrap();
        assert!(prefixes.is_empty());

        // Cleanup
        std::fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_load_prefixes_from_file_inline_comments() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_inline_comments.txt");

        let content = "abc # inline comment\ntest#no space\n  foo  # with spaces  ";
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        drop(file);

        let prefixes = load_prefixes_from_file(&file_path).unwrap();

        assert_eq!(prefixes, vec!["abc", "test", "foo"]);

        // Cleanup
        std::fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_load_prefixes_from_file_nonexistent() {
        let result = load_prefixes_from_file(&PathBuf::from("/nonexistent/file.txt"));
        assert!(result.is_err());
    }
}
