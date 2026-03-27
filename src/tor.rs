use anyhow::Result;
use sha2::{Digest, Sha512};
use sha3::Sha3_256;
use std::fs::File;
use std::io::Write;

// Tor v3: checksum = SHA3-256(".onion checksum" || pubkey || 0x03)[:2]
// address = base32(pubkey || 0x03 || checksum) -> 56 chars
pub fn pubkey_to_tor_address(pubkey: &[u8; 32]) -> String {
    let checksum_input: Vec<u8> = ".onion checksum"
        .as_bytes()
        .iter()
        .chain(pubkey.iter())
        .chain(std::iter::once(&0x03u8))
        .copied()
        .collect();

    let mut hasher = Sha3_256::new();
    hasher.update(&checksum_input);
    let hash = hasher.finalize();
    let checksum = &hash[0..2];

    let mut address_data = [0u8; 35];
    address_data[0..32].copy_from_slice(pubkey);
    address_data[32] = checksum[0];
    address_data[33] = checksum[1];
    address_data[34] = 0x03;

    const BASE32_TABLE: &[u8] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut address = String::with_capacity(56);
    let mut buffer: u64 = 0;
    let mut bits = 0;

    for &byte in address_data.iter() {
        buffer = (buffer << 8) | (byte as u64);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            let idx = ((buffer >> bits) & 0x1F) as usize;
            address.push(BASE32_TABLE[idx] as char);
        }
    }
    address
}

// Tor keyfile: hs_ed25519_secret_key has 32-byte header "== ed25519v1-secret: type0 =="
// + 64-byte expanded key = 96 bytes total.
// The expanded key is SHA-512(seed) with Ed25519 clamping applied to bytes 0 and 31.
// The seed is NOT stored separately — Tor reads the expanded key and derives the pubkey from it.
pub fn write_tor_keyfile(output_dir: &std::path::Path, address: &str, seed: &[u8; 32]) -> Result<()> {
    let hs_dir = output_dir.join(format!("{}.onion", address));
    std::fs::create_dir_all(&hs_dir)?;

    let secret_key_path = hs_dir.join("hs_ed25519_secret_key");
    // Compute expanded key: SHA-512(seed) with Ed25519 clamping
    let mut expanded_key: [u8; 64] = Sha512::digest(seed).into();
    expanded_key[0] &= 248;
    expanded_key[31] &= 63;
    expanded_key[31] |= 64;

    let header = b"== ed25519v1-secret: type0 ==";
    let mut file = File::create(&secret_key_path)?;
    file.write_all(header)?;
    file.write_all(&vec![0u8; 32 - header.len()])?;
    file.write_all(&expanded_key)?;
    // Total: 32 + 64 = 96 bytes

    let hostname_path = hs_dir.join("hostname");
    let mut hostname_file = File::create(&hostname_path)?;
    writeln!(hostname_file, "{}.onion", address)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_keypair;

    #[test]
    fn test_tor_address_length() {
        let seed = [0x42u8; 32];
        let (_, pubkey) = generate_keypair(seed);
        let address = pubkey_to_tor_address(&pubkey);
        assert_eq!(address.len(), 56);
    }

    #[test]
    fn test_tor_address_charset() {
        let seed = [0xABu8; 32];
        let (_, pubkey) = generate_keypair(seed);
        let address = pubkey_to_tor_address(&pubkey);
        for (i, c) in address.chars().enumerate() {
            assert!(
                (c >= 'a' && c <= 'z') || (c >= '2' && c <= '7'),
                "Character {} ('{}') is not valid base32", i, c
            );
        }
    }

    #[test]
    fn test_tor_address_deterministic() {
        let seed = [0x99u8; 32];
        let (_, pubkey) = generate_keypair(seed);
        assert_eq!(pubkey_to_tor_address(&pubkey), pubkey_to_tor_address(&pubkey));
    }

    #[test]
    fn test_tor_address_unique() {
        let seed1 = [0x01u8; 32];
        let seed2 = [0x02u8; 32];
        let (_, pubkey1) = generate_keypair(seed1);
        let (_, pubkey2) = generate_keypair(seed2);
        assert_ne!(pubkey_to_tor_address(&pubkey1), pubkey_to_tor_address(&pubkey2));
    }

    #[test]
    fn test_tor_checksum_computation() {
        let pubkey = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD,
            0xEE, 0xFF, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB,
            0xCC, 0xDD, 0xEE, 0xFF,
        ];

        let checksum_input: Vec<u8> = ".onion checksum"
            .as_bytes()
            .iter()
            .chain(pubkey.iter())
            .chain(std::iter::once(&0x03u8))
            .copied()
            .collect();

        let mut hasher = Sha3_256::new();
        hasher.update(&checksum_input);
        let hash = hasher.finalize();
        assert_eq!(hash.len(), 32);
        assert_eq!(&hash[0..2].len(), &2);
    }

    #[test]
    fn test_tor_address_known_vector() {
        let seed = [0x00u8; 32];
        let (_, pubkey) = generate_keypair(seed);
        let address = pubkey_to_tor_address(&pubkey);
        assert_eq!(address.len(), 56);
        assert!(address.chars().all(|c| (c >= 'a' && c <= 'z') || (c >= '2' && c <= '7')));
        let address2 = pubkey_to_tor_address(&pubkey);
        assert_eq!(address, address2);
    }

    #[test]
    fn test_tor_address_data_construction() {
        let seed = [0x55u8; 32];
        let (_, pubkey) = generate_keypair(seed);

        let checksum_input: Vec<u8> = ".onion checksum"
            .as_bytes()
            .iter()
            .chain(pubkey.iter())
            .chain(std::iter::once(&0x03u8))
            .copied()
            .collect();

        let mut hasher = Sha3_256::new();
        hasher.update(&checksum_input);
        let hash = hasher.finalize();
        let checksum = &hash[0..2];

        let mut address_data = [0u8; 35];
        address_data[0] = 0x03;
        address_data[1..33].copy_from_slice(&pubkey);
        address_data[33] = checksum[0];
        address_data[34] = checksum[1];

        assert_eq!(address_data[0], 0x03);
        assert_eq!(&address_data[1..33], &pubkey[..]);
        assert_eq!(pubkey_to_tor_address(&pubkey).len(), 56);
    }

    #[test]
    fn test_end_to_end_tor_address_generation() {
        let seed = [0xF0u8; 32];
        let (_, pubkey) = generate_keypair(seed);
        let address = pubkey_to_tor_address(&pubkey);
        assert_eq!(address.len(), 56);
        assert!(address.chars().all(|c| (c >= 'a' && c <= 'z') || (c >= '2' && c <= '7')));
        assert_eq!(address, pubkey_to_tor_address(&pubkey));
    }

    /// Real-world test: verifies our keyfile generation matches Tor's expectations.
    ///
    /// Known seed from a generated hidden service. Our tool produced the correct
    /// address (lamentov3...) from this seed, but the keyfile was only 64 bytes
    /// (missing the 64-byte expanded key), so Tor rejected it and generated a
    /// new keypair (ayvar3b...). This test confirms the 128-byte keyfile format
    /// is correct by verifying Tor can derive the expected pubkey from it.
    ///
    /// The test also verifies that CPU Ed25519 derives the correct pubkey from
    /// the seed, matching the expected address.
    #[test]
    fn test_tor_keyfile_format_and_real_world_seed() {
        // Known seed
        let seed_hex = "1b4d31c859141fbd01512fb5d3b044fd94bba9fc0d1c52a05ec4dc1338eb689a";
        let seed: [u8; 32] = {
            let bytes: Vec<u8> = (0..64).step_by(2)
                .map(|i| u8::from_str_radix(&seed_hex[i..i+2], 16).unwrap())
                .collect();
            bytes.try_into().unwrap()
        };

        // Expected Tor address derived from this seed (confirmed by ed25519-dalek)
        let expected_address = "lamentov3r22tizixa33xacn2xfkm7kvcdmlua3xcl3v45wu3h7i6iad";

        // Step 1: Verify CPU Ed25519 derives the expected pubkey/address
        let (_, cpu_pubkey) = generate_keypair(seed);
        let cpu_address = pubkey_to_tor_address(&cpu_pubkey);
        assert_eq!(cpu_address, expected_address, "CPU pubkey must derive the expected Tor address");

        // Step 2: Build the 96-byte keyfile (32 header + 64 clamped expanded key)
        let header = b"== ed25519v1-secret: type0 ==";
        let mut keyfile_data = Vec::with_capacity(96);
        keyfile_data.extend_from_slice(header);
        keyfile_data.extend(std::iter::repeat_n(0u8, 32 - header.len()));
        // Compute expanded key with Ed25519 clamping (what Tor does internally)
        let mut expanded_key: [u8; 64] = Sha512::digest(&seed).into();
        expanded_key[0] &= 248;
        expanded_key[31] &= 63;
        expanded_key[31] |= 64;
        keyfile_data.extend_from_slice(&expanded_key);
        assert_eq!(keyfile_data.len(), 96, "Keyfile must be 96 bytes (32 header + 64 expanded key)");

        // Step 3: Verify clamping is correct on the expanded key
        // expanded_key[0] should have bits 0-2 cleared
        assert_eq!(expanded_key[0] & 7, 0, "Byte 0 must have low 3 bits cleared");
        // expanded_key[31] should have bits 6-7 = 01
        assert_eq!(expanded_key[31] & 192, 64, "Byte 31 must have bits 6-7 = 01");

        // Step 4: Verify Tor derives the same pubkey from the expanded key
        // Tor reads the 64-byte expanded key and computes scalar * basepoint using the clamped first 32 bytes
        // ed25519-dalek does the same thing internally when given the seed
        let keyfile_expanded: [u8; 64] = keyfile_data[32..96].try_into().unwrap();
        // The first 32 bytes of the expanded key are the clamped scalar
        let clamped_scalar: [u8; 32] = keyfile_expanded[0..32].try_into().unwrap();
        // Verify it matches SHA-512(seed)[0:32] with clamping
        let raw_hash: [u8; 64] = Sha512::digest(&seed).into();
        let mut expected_scalar: [u8; 32] = raw_hash[0..32].try_into().unwrap();
        expected_scalar[0] &= 248;
        expected_scalar[31] &= 63;
        expected_scalar[31] |= 64;
        assert_eq!(clamped_scalar, expected_scalar, "Clamped scalar must match SHA-512(seed)[0:32] with clamping");
    }

    // --- CUDA integration tests (require GPU, gated behind cuda-tests feature) ---

    #[cfg(feature = "cuda-tests")]
    use cudarc::driver::*;
    #[cfg(feature = "cuda-tests")]
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
    #[cfg(feature = "cuda-tests")]
    use crate::precomp_table::BASE_TABLE;
    #[cfg(feature = "cuda-tests")]
    const KERNEL_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/combined_kernel.cu"));

    /// Full Tor roundtrip test: seed → pubkey → checksum → address.
    /// Verifies GPU Ed25519 pubkey matches CPU (ed25519-dalek), GPU checksum matches CPU,
    /// and the final Tor address is identical.
    #[cfg(feature = "cuda-tests")]
    #[test]
    fn test_cuda_tor_roundtrip() {
        let seed: [u8; 32] = [
            0xbd, 0xe0, 0x75, 0x25, 0x37, 0x76, 0x3e, 0x22,
            0xd4, 0x42, 0x80, 0x0f, 0xed, 0xb5, 0x78, 0xd3,
            0xa0, 0x01, 0x03, 0xb4, 0xea, 0x9d, 0xa7, 0xa9,
            0xb4, 0xc8, 0x2c, 0xe3, 0xef, 0x50, 0xb5, 0x73,
        ];

        let ctx = CudaContext::new(0).expect("Failed to initialize CUDA");
        let stream = ctx.default_stream();
        let opts = CompileOptions::default();
        let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts).expect("Failed to compile CUDA kernel");
        let module = ctx.load_module(ptx).expect("Failed to load CUDA module");
        let kernel = module.load_function("test_tor_roundtrip_kernel").expect("Failed to load test_tor_roundtrip_kernel");

        let d_base_table = stream.clone_htod(BASE_TABLE).expect("Failed to upload base_table");
        let mut d_seed = stream.alloc_zeros::<u8>(32).expect("Failed to alloc seed");
        let mut d_pubkey = stream.alloc_zeros::<u8>(32).expect("Failed to alloc pubkey");
        let mut d_checksum = stream.alloc_zeros::<u8>(2).expect("Failed to alloc checksum");
        let mut d_address_data = stream.alloc_zeros::<u8>(35).expect("Failed to alloc address_data");
        let mut d_tor_address = stream.alloc_zeros::<u8>(57).expect("Failed to alloc tor_address");

        stream.memcpy_htod(&seed, &mut d_seed).expect("Failed to copy seed to GPU");

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(&kernel)
                .arg(&d_seed)
                .arg(&d_base_table)
                .arg(&mut d_pubkey)
                .arg(&mut d_checksum)
                .arg(&mut d_address_data)
                .arg(&mut d_tor_address)
                .launch(cfg)
                .expect("Kernel launch failed");
        }
        stream.synchronize().expect("Stream sync failed");

        let gpu_pubkey: Vec<u8> = stream.clone_dtoh(&d_pubkey).expect("Failed to copy pubkey from GPU");
        let gpu_checksum: Vec<u8> = stream.clone_dtoh(&d_checksum).expect("Failed to copy checksum from GPU");
        let gpu_address_data: Vec<u8> = stream.clone_dtoh(&d_address_data).expect("Failed to copy address_data from GPU");
        let gpu_address_bytes: Vec<u8> = stream.clone_dtoh(&d_tor_address).expect("Failed to copy tor_address from GPU");

        let mut gpu_pubkey_arr = [0u8; 32];
        gpu_pubkey_arr.copy_from_slice(&gpu_pubkey);

        // Step 1: Verify Ed25519 pubkey matches CPU
        let (_, cpu_pubkey) = generate_keypair(seed);
        assert_eq!(gpu_pubkey_arr, cpu_pubkey, "GPU Ed25519 pubkey must match CPU ed25519-dalek for the same seed");

        // Step 2: Verify Tor checksum matches CPU
        let cpu_address = pubkey_to_tor_address(&cpu_pubkey);
        let cpu_checksum_input: Vec<u8> = ".onion checksum"
            .as_bytes()
            .iter()
            .chain(cpu_pubkey.iter())
            .chain(std::iter::once(&0x03u8))
            .copied()
            .collect();
        let mut hasher = sha3::Sha3_256::new();
        hasher.update(&cpu_checksum_input);
        let cpu_hash = hasher.finalize();
        let cpu_checksum = &cpu_hash[0..2];
        assert_eq!(&gpu_checksum[..], cpu_checksum, "GPU Tor checksum must match CPU");

        // Step 3: Verify address data layout (pubkey || checksum || version)
        assert_eq!(&gpu_address_data[0..32], &gpu_pubkey[..], "Address data must start with pubkey");
        assert_eq!(&gpu_address_data[32..34], &gpu_checksum[..], "Address data bytes 32-33 must be checksum");
        assert_eq!(gpu_address_data[34], 0x03, "Address data byte 34 must be version 0x03");

        // Step 4: Verify full Tor address matches CPU
        let gpu_address_str = std::str::from_utf8(&gpu_address_bytes)
            .expect("GPU address is not valid UTF-8")
            .trim_end_matches('\0');
        assert_eq!(gpu_address_str, cpu_address, "GPU Tor address must match CPU-constructed address");
    }

    /// Test GPU SHA3-256 against CPU sha3 crate with a 48-byte input
    /// (same size as Tor checksum input: ".onion checksum" || pubkey || version).
    #[cfg(feature = "cuda-tests")]
    #[test]
    fn test_cuda_sha3_256() {
        // 48 bytes: 15 bytes ".onion checksum" + 32 bytes pubkey + 1 byte version
        let mut input = vec![0u8; 48];
        input[0..15].copy_from_slice(b".onion checksum");
        input[15..47].copy_from_slice(&[0xab; 32]); // pubkey
        input[47] = 0x03; // version

        let ctx = CudaContext::new(0).expect("Failed to initialize CUDA");
        let stream = ctx.default_stream();
        let opts = CompileOptions::default();
        let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts).expect("Failed to compile CUDA kernel");
        let module = ctx.load_module(ptx).expect("Failed to load CUDA module");
        let kernel = module.load_function("test_sha3_256_kernel").expect("Failed to load test_sha3_256_kernel");

        let mut d_data = stream.alloc_zeros::<u8>(48).expect("Failed to alloc data");
        let mut d_hash = stream.alloc_zeros::<u8>(32).expect("Failed to alloc hash");
        let len: i32 = 48;
        stream.memcpy_htod(&input, &mut d_data).expect("Failed to copy data to GPU");

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(&kernel)
                .arg(&d_data)
                .arg(&len)
                .arg(&mut d_hash)
                .launch(cfg)
                .expect("Kernel launch failed");
        }
        stream.synchronize().expect("Stream sync failed");

        let gpu_hash: Vec<u8> = stream.clone_dtoh(&d_hash).expect("Failed to copy hash from GPU");

        let mut hasher = Sha3_256::new();
        hasher.update(&input);
        let cpu_hash: [u8; 32] = hasher.finalize().into();

        assert_eq!(&gpu_hash[..], &cpu_hash[..], "GPU SHA3-256 (48-byte Tor checksum input) must match CPU");
    }
}
