use anyhow::Result;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;

use crate::utils::xorshift64;

#[cfg(feature = "cuda-tests")]
pub fn generate_i2p_random_data(seed: &[u8; 32]) -> [u8; 256] {
    let mut rand_state = u64::from_le_bytes(seed[0..8].try_into().unwrap())
                      ^ u64::from_le_bytes(seed[8..16].try_into().unwrap());
    let mut random_data = [0u8; 256];
    for chunk in 0..8 {
        for i in 0..4 {
            rand_state = xorshift64(rand_state);
            let idx = chunk * 32 + i * 8;
            random_data[idx..idx + 8].copy_from_slice(&rand_state.to_le_bytes());
        }
    }
    random_data
}

// I2P Destination: 256 random + 96 padding + 32 pubkey + 7 certificate = 391 bytes
// Per I2P spec: "Crypto Public Key is aligned at the start and Signing Public Key is aligned at the end"
pub fn construct_destination(random_data: &[u8; 256], pubkey: &[u8; 32]) -> Vec<u8> {
    let mut dest = Vec::with_capacity(391);

    dest.extend_from_slice(random_data);   // 0-255: Crypto Public Key (ElGamal, unused for destinations)
    dest.extend_from_slice(&[0u8; 96]);    // 256-351: Padding
    dest.extend_from_slice(pubkey);        // 352-383: Signing Public Key (Ed25519) - at the END before cert

    // Key Certificate for Ed25519 (type 7) with ElGamal (type 0)
    dest.extend_from_slice(&[0x05, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00]);

    dest
}

pub fn hash_to_address(hash: &[u8; 32]) -> String {
    const BASE32_TABLE: &[u8] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut address = String::with_capacity(52);
    let mut buffer: u64 = 0;
    let mut bits = 0;

    for &byte in hash.iter() {
        buffer = (buffer << 8) | (byte as u64);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            let idx = ((buffer >> bits) & 0x1F) as usize;
            address.push(BASE32_TABLE[idx] as char);
        }
    }

    // Handle remaining bits (256 bits % 5 = 1 bit remaining)
    // Encode the trailing bit as the final character
    if bits > 0 {
        let idx = ((buffer << (5 - bits)) & 0x1F) as usize;
        address.push(BASE32_TABLE[idx] as char);
    }

    address
}

/// Generate the full I2P .b32.i2p address from a seed and pubkey.
/// Returns (address, destination_bytes).
pub fn i2p_seed_to_address(seed: &[u8; 32], pubkey: &[u8; 32]) -> (String, Vec<u8>) {
    // Generate random data matching CUDA kernel's xorshift64 PRNG
    let mut random_data = [0u8; 256];
    {
        let mut state = u64::from_le_bytes([
            seed[0], seed[1], seed[2], seed[3], seed[4], seed[5], seed[6], seed[7],
        ]) ^ u64::from_le_bytes([
            seed[8], seed[9], seed[10], seed[11], seed[12], seed[13], seed[14],
            seed[15],
        ]);
        for chunk in 0..8 {
            for j in 0..4 {
                state = xorshift64(state);
                let idx = chunk * 32 + j * 8;
                random_data[idx..idx + 8].copy_from_slice(&state.to_le_bytes());
            }
        }
    }

    let destination = construct_destination(&random_data, pubkey);
    let mut hasher = Sha256::new();
    hasher.update(&destination);
    let hash: [u8; 32] = hasher.finalize().into();
    let address = hash_to_address(&hash);

    (address, destination)
}

// I2P keyfile: Destination(391) + PrivateKey(256, unused) + SigningPrivateKey(32) = 679 bytes
pub fn write_keyfile(
    path: &std::path::Path,
    destination: &[u8],
    _privkey: &[u8; 256],
    signing_privkey: &[u8; 32],
) -> Result<()> {
    let mut file_data = Vec::with_capacity(destination.len() + 256 + 32);
    file_data.extend_from_slice(destination);
    file_data.extend_from_slice(_privkey);
    file_data.extend_from_slice(signing_privkey);

    let mut file = File::create(path)?;
    file.write_all(&file_data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_keypair;

    #[test]
    fn test_destination_structure() {
        let mut random_data = [0u8; 256];
        for (i, byte) in random_data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        let mut pubkey = [0u8; 32];
        for (i, byte) in pubkey.iter_mut().enumerate() {
            *byte = ((i + 1) * 7 % 256) as u8;
        }

        let dest = construct_destination(&random_data, &pubkey);
        assert_eq!(dest.len(), 391);
        assert_eq!(&dest[0..256], &random_data[..]);
        // Padding (96 bytes) comes before pubkey
        for (i, &byte) in dest[256..352].iter().enumerate() {
            assert_eq!(byte, 0, "Padding byte {} should be zero", i);
        }
        // Pubkey at the end, right before certificate
        assert_eq!(&dest[352..384], &pubkey[..]);
        // Certificate
        assert_eq!(&dest[384..391], &[0x05, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00]);
    }

    #[test]
    fn test_destination_deterministic() {
        let random_data = [0xABu8; 256];
        let pubkey = [0xCDu8; 32];
        let dest1 = construct_destination(&random_data, &pubkey);
        let dest2 = construct_destination(&random_data, &pubkey);
        assert_eq!(dest1, dest2);
    }

    #[test]
    fn test_sha256_hashing() {
        let random_data = [0x42u8; 256];
        let pubkey = [0x37u8; 32];
        let dest = construct_destination(&random_data, &pubkey);

        let mut hasher = Sha256::new();
        hasher.update(&dest);
        let hash = hasher.finalize();
        assert_eq!(hash.len(), 32);

        let hash2: [u8; 32] = {
            let mut hasher = Sha256::new();
            hasher.update(&dest);
            hasher.finalize().into()
        };
        assert_eq!(hash.as_slice(), &hash2[..]);
    }

    #[test]
    fn test_sha256_empty() {
        let expected: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        let mut hasher = Sha256::new();
        hasher.update(&[]);
        let hash: [u8; 32] = hasher.finalize().into();
        assert_eq!(hash, expected);
    }

    #[test]
    fn test_base32_length() {
        let hash = [0u8; 32];
        let address = hash_to_address(&hash);
        assert_eq!(address.len(), 52);
    }

    #[test]
    fn test_base32_known_values() {
        let zero_hash = [0u8; 32];
        let address = hash_to_address(&zero_hash);
        assert!(address.starts_with("aaaaaaaaaa"));
        assert!(address.chars().all(|c| c >= 'a' && c <= 'z' || c >= '2' && c <= '7'));

        let ones_hash = [0xFFu8; 32];
        let address = hash_to_address(&ones_hash);
        // First 51 chars are '7' (all 1s), last char is 'q' (trailing 1 bit shifted left by 4)
        assert!(address.chars().take(51).all(|c| c == '7'));
        assert_eq!(address.chars().last().unwrap(), 'q');
    }

    #[test]
    fn test_base32_deterministic() {
        let hash = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x01, 0x23, 0x45, 0x67,
            0x89, 0xAB, 0xCD, 0xEF,
        ];
        assert_eq!(hash_to_address(&hash), hash_to_address(&hash));
    }

    #[test]
    fn test_base32_charset() {
        let mut hash = [0u8; 32];
        for i in 0..32 {
            hash[i] = ((i * 17 + 13) % 256) as u8;
        }
        let address = hash_to_address(&hash);
        for (i, c) in address.chars().enumerate() {
            assert!(
                (c >= 'a' && c <= 'z') || (c >= '2' && c <= '7'),
                "Character {} ('{}') is not valid base32",
                i, c
            );
        }
    }

    #[test]
    fn test_end_to_end_address_generation() {
        let random_data = [0x5Au8; 256];
        let pubkey = [0x3Cu8; 32];
        let dest = construct_destination(&random_data, &pubkey);
        assert_eq!(dest.len(), 391);

        let mut hasher = Sha256::new();
        hasher.update(&dest);
        let hash: [u8; 32] = hasher.finalize().into();
        let address = hash_to_address(&hash);
        assert_eq!(address.len(), 52);
        assert!(address.chars().all(|c| (c >= 'a' && c <= 'z') || (c >= '2' && c <= '7')));
    }

    /// Test vector from a real I2P keyfile generated by i2p.
    /// This validates that our CPU implementation produces correct keypairs and addresses.
    /// File: udcwkfiiibnjdxrvmsavgvywffc66xrzty4a44nopybwjgeq62va.b32.i2p.dat
    #[test]
    fn test_i2p_real_keyvector_keypair() {
        // The Ed25519 signing private key (seed) from the I2P keyfile
        // Located at bytes 647-678 in the 679-byte keyfile
        let seed: [u8; 32] = [
            0xbd, 0xe0, 0x75, 0x25, 0x37, 0x76, 0x3e, 0x22,
            0xd4, 0x42, 0x80, 0x0f, 0xed, 0xb5, 0x78, 0xd3,
            0xa0, 0x01, 0x03, 0xb4, 0xea, 0x9d, 0xa7, 0xa9,
            0xb4, 0xc8, 0x2c, 0xe3, 0xef, 0x50, 0xb5, 0x73,
        ];

        // The expected Ed25519 public key from the I2P keyfile
        // Located at bytes 352-383 in the destination (bytes 352-383 in the keyfile)
        let expected_pubkey: [u8; 32] = [
            0xbf, 0xc7, 0x00, 0xf3, 0x7b, 0xc5, 0x2c, 0xa0,
            0xf8, 0x45, 0x58, 0x5c, 0x08, 0xd9, 0x0b, 0x68,
            0xed, 0xb1, 0x7d, 0x75, 0x73, 0xda, 0x1b, 0xf5,
            0xf4, 0x72, 0xe7, 0xb4, 0x3a, 0x6b, 0x49, 0x50,
        ];

        let (_, pubkey) = generate_keypair(seed);
        assert_eq!(pubkey, expected_pubkey, "CPU Ed25519 keypair generation must match I2P keyfile");
    }

    /// Test that the full I2P address generation produces the correct .b32.i2p address.
    /// Uses the exact destination data from the real I2P keyfile.
    #[test]
    fn test_i2p_real_keyvector_address() {
        // The full 391-byte destination from the I2P keyfile (bytes 0-390)
        // This includes: 256 bytes random_data + 96 bytes padding + 32 bytes pubkey + 7 bytes cert
        // Note: In this keyfile, the "padding" is NOT zeros - it continues the random data pattern
        let destination: [u8; 391] = [
            // Random data (256 bytes) - 32-byte pattern repeated 8 times
            0x5b, 0x42, 0x44, 0xfe, 0x3d, 0x91, 0x5e, 0xec,
            0xa5, 0xf9, 0xad, 0x7b, 0xe8, 0x6a, 0x27, 0x2f,
            0xca, 0x37, 0xdc, 0x10, 0xd4, 0x4c, 0x40, 0xff,
            0x8b, 0x15, 0xea, 0xc7, 0x08, 0x41, 0xe4, 0x96,
            // ... repeated 8 times (256 bytes total for random data)
            0x5b, 0x42, 0x44, 0xfe, 0x3d, 0x91, 0x5e, 0xec,
            0xa5, 0xf9, 0xad, 0x7b, 0xe8, 0x6a, 0x27, 0x2f,
            0xca, 0x37, 0xdc, 0x10, 0xd4, 0x4c, 0x40, 0xff,
            0x8b, 0x15, 0xea, 0xc7, 0x08, 0x41, 0xe4, 0x96,
            0x5b, 0x42, 0x44, 0xfe, 0x3d, 0x91, 0x5e, 0xec,
            0xa5, 0xf9, 0xad, 0x7b, 0xe8, 0x6a, 0x27, 0x2f,
            0xca, 0x37, 0xdc, 0x10, 0xd4, 0x4c, 0x40, 0xff,
            0x8b, 0x15, 0xea, 0xc7, 0x08, 0x41, 0xe4, 0x96,
            0x5b, 0x42, 0x44, 0xfe, 0x3d, 0x91, 0x5e, 0xec,
            0xa5, 0xf9, 0xad, 0x7b, 0xe8, 0x6a, 0x27, 0x2f,
            0xca, 0x37, 0xdc, 0x10, 0xd4, 0x4c, 0x40, 0xff,
            0x8b, 0x15, 0xea, 0xc7, 0x08, 0x41, 0xe4, 0x96,
            0x5b, 0x42, 0x44, 0xfe, 0x3d, 0x91, 0x5e, 0xec,
            0xa5, 0xf9, 0xad, 0x7b, 0xe8, 0x6a, 0x27, 0x2f,
            0xca, 0x37, 0xdc, 0x10, 0xd4, 0x4c, 0x40, 0xff,
            0x8b, 0x15, 0xea, 0xc7, 0x08, 0x41, 0xe4, 0x96,
            0x5b, 0x42, 0x44, 0xfe, 0x3d, 0x91, 0x5e, 0xec,
            0xa5, 0xf9, 0xad, 0x7b, 0xe8, 0x6a, 0x27, 0x2f,
            0xca, 0x37, 0xdc, 0x10, 0xd4, 0x4c, 0x40, 0xff,
            0x8b, 0x15, 0xea, 0xc7, 0x08, 0x41, 0xe4, 0x96,
            0x5b, 0x42, 0x44, 0xfe, 0x3d, 0x91, 0x5e, 0xec,
            0xa5, 0xf9, 0xad, 0x7b, 0xe8, 0x6a, 0x27, 0x2f,
            0xca, 0x37, 0xdc, 0x10, 0xd4, 0x4c, 0x40, 0xff,
            0x8b, 0x15, 0xea, 0xc7, 0x08, 0x41, 0xe4, 0x96,
            0x5b, 0x42, 0x44, 0xfe, 0x3d, 0x91, 0x5e, 0xec,
            0xa5, 0xf9, 0xad, 0x7b, 0xe8, 0x6a, 0x27, 0x2f,
            0xca, 0x37, 0xdc, 0x10, 0xd4, 0x4c, 0x40, 0xff,
            0x8b, 0x15, 0xea, 0xc7, 0x08, 0x41, 0xe4, 0x96,
            // Padding (96 bytes) - same pattern repeated 3 more times
            0x5b, 0x42, 0x44, 0xfe, 0x3d, 0x91, 0x5e, 0xec,
            0xa5, 0xf9, 0xad, 0x7b, 0xe8, 0x6a, 0x27, 0x2f,
            0xca, 0x37, 0xdc, 0x10, 0xd4, 0x4c, 0x40, 0xff,
            0x8b, 0x15, 0xea, 0xc7, 0x08, 0x41, 0xe4, 0x96,
            0x5b, 0x42, 0x44, 0xfe, 0x3d, 0x91, 0x5e, 0xec,
            0xa5, 0xf9, 0xad, 0x7b, 0xe8, 0x6a, 0x27, 0x2f,
            0xca, 0x37, 0xdc, 0x10, 0xd4, 0x4c, 0x40, 0xff,
            0x8b, 0x15, 0xea, 0xc7, 0x08, 0x41, 0xe4, 0x96,
            0x5b, 0x42, 0x44, 0xfe, 0x3d, 0x91, 0x5e, 0xec,
            0xa5, 0xf9, 0xad, 0x7b, 0xe8, 0x6a, 0x27, 0x2f,
            0xca, 0x37, 0xdc, 0x10, 0xd4, 0x4c, 0x40, 0xff,
            0x8b, 0x15, 0xea, 0xc7, 0x08, 0x41, 0xe4, 0x96,
            // Ed25519 public key (32 bytes)
            0xbf, 0xc7, 0x00, 0xf3, 0x7b, 0xc5, 0x2c, 0xa0,
            0xf8, 0x45, 0x58, 0x5c, 0x08, 0xd9, 0x0b, 0x68,
            0xed, 0xb1, 0x7d, 0x75, 0x73, 0xda, 0x1b, 0xf5,
            0xf4, 0x72, 0xe7, 0xb4, 0x3a, 0x6b, 0x49, 0x50,
            // Certificate (7 bytes) - Ed25519 with ElGamal
            0x05, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00,
        ];

        // Expected .b32.i2p address (without the .b32.i2p suffix)
        let expected_address = "udcwkfiiibnjdxrvmsavgvywffc66xrzty4a44nopybwjgeq62va";

        // Hash the destination with SHA-256
        let mut hasher = Sha256::new();
        hasher.update(&destination);
        let hash: [u8; 32] = hasher.finalize().into();

        // Convert hash to base32 address
        let address = hash_to_address(&hash);

        assert_eq!(address, expected_address, "Address must match the I2P keyfile's .b32.i2p address");
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

    #[cfg(feature = "cuda-tests")]
    #[test]
    fn test_cuda_ed25519_known_vector() {
        let seed: [u8; 32] = [
            0xbd, 0xe0, 0x75, 0x25, 0x37, 0x76, 0x3e, 0x22,
            0xd4, 0x42, 0x80, 0x0f, 0xed, 0xb5, 0x78, 0xd3,
            0xa0, 0x01, 0x03, 0xb4, 0xea, 0x9d, 0xa7, 0xa9,
            0xb4, 0xc8, 0x2c, 0xe3, 0xef, 0x50, 0xb5, 0x73,
        ];
        let expected_pubkey: [u8; 32] = [
            0xbf, 0xc7, 0x00, 0xf3, 0x7b, 0xc5, 0x2c, 0xa0,
            0xf8, 0x45, 0x58, 0x5c, 0x08, 0xd9, 0x0b, 0x68,
            0xed, 0xb1, 0x7d, 0x75, 0x73, 0xda, 0x1b, 0xf5,
            0xf4, 0x72, 0xe7, 0xb4, 0x3a, 0x6b, 0x49, 0x50,
        ];

        let ctx = CudaContext::new(0).expect("Failed to initialize CUDA");
        let stream = ctx.default_stream();
        let opts = CompileOptions::default();
        let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts).expect("Failed to compile CUDA kernel");
        let module = ctx.load_module(ptx).expect("Failed to load CUDA module");
        let kernel = module.load_function("test_ed25519_pubkey_kernel").expect("Failed to load test_ed25519_pubkey_kernel");

        let d_base_table = stream.clone_htod(BASE_TABLE).expect("Failed to upload base_table");
        let mut d_seed = stream.alloc_zeros::<u8>(32).expect("Failed to alloc seed");
        let mut d_pubkey_out = stream.alloc_zeros::<u8>(32).expect("Failed to alloc pubkey_out");

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
                .arg(&mut d_pubkey_out)
                .launch(cfg)
                .expect("Kernel launch failed");
        }

        stream.synchronize().expect("Stream sync failed");

        let gpu_pubkey: Vec<u8> = stream.clone_dtoh(&d_pubkey_out).expect("Failed to copy pubkey from GPU");
        let mut gpu_pubkey_arr = [0u8; 32];
        gpu_pubkey_arr.copy_from_slice(&gpu_pubkey);

        assert_eq!(gpu_pubkey_arr, expected_pubkey, "GPU Ed25519 pubkey must match I2P keyfile known vector");
    }

    #[cfg(feature = "cuda-tests")]
    #[test]
    fn test_cuda_i2p_roundtrip() {
        let base_seed = [0x42u8; 32];
        let iteration: u64 = 0;

        let ctx = CudaContext::new(0).expect("Failed to initialize CUDA");
        let stream = ctx.default_stream();
        let opts = CompileOptions::default();
        let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts).expect("Failed to compile CUDA kernel");
        let module = ctx.load_module(ptx).expect("Failed to load CUDA module");
        let kernel = module.load_function("test_i2p_roundtrip_kernel").expect("Failed to load test_i2p_roundtrip_kernel");

        let d_base_table = stream.clone_htod(BASE_TABLE).expect("Failed to upload base_table");
        let mut d_base_seed = stream.alloc_zeros::<u8>(32).expect("Failed to alloc base_seed");
        let mut d_out_seed = stream.alloc_zeros::<u8>(32).expect("Failed to alloc out_seed");
        let mut d_out_pubkey = stream.alloc_zeros::<u8>(32).expect("Failed to alloc out_pubkey");
        let mut d_out_random = stream.alloc_zeros::<u8>(256).expect("Failed to alloc out_random");
        let mut d_out_address = stream.alloc_zeros::<u8>(52).expect("Failed to alloc out_address");
        let mut d_out_hash = stream.alloc_zeros::<u8>(32).expect("Failed to alloc out_hash");

        stream.memcpy_htod(&base_seed, &mut d_base_seed).expect("Failed to copy base_seed to GPU");

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(&kernel)
                .arg(&d_base_seed)
                .arg(&d_base_table)
                .arg(&iteration)
                .arg(&mut d_out_seed)
                .arg(&mut d_out_pubkey)
                .arg(&mut d_out_random)
                .arg(&mut d_out_address)
                .arg(&mut d_out_hash)
                .launch(cfg)
                .expect("Kernel launch failed");
        }

        stream.synchronize().expect("Stream sync failed");

        let gpu_seed: Vec<u8> = stream.clone_dtoh(&d_out_seed).expect("Failed to copy seed from GPU");
        let gpu_pubkey: Vec<u8> = stream.clone_dtoh(&d_out_pubkey).expect("Failed to copy pubkey from GPU");
        let gpu_random: Vec<u8> = stream.clone_dtoh(&d_out_random).expect("Failed to copy random_data from GPU");
        let gpu_address: Vec<u8> = stream.clone_dtoh(&d_out_address).expect("Failed to copy address from GPU");
        let gpu_hash: Vec<u8> = stream.clone_dtoh(&d_out_hash).expect("Failed to copy hash from GPU");

        let mut gpu_seed_arr = [0u8; 32];
        gpu_seed_arr.copy_from_slice(&gpu_seed);
        let mut gpu_pubkey_arr = [0u8; 32];
        gpu_pubkey_arr.copy_from_slice(&gpu_pubkey);
        let mut gpu_random_arr = [0u8; 256];
        gpu_random_arr.copy_from_slice(&gpu_random);
        let gpu_address_str = std::str::from_utf8(&gpu_address).expect("GPU address is not valid UTF-8");

        // Step 1: Verify Ed25519 pubkey matches CPU
        let (_, cpu_pubkey) = generate_keypair(gpu_seed_arr);
        assert_eq!(gpu_pubkey_arr, cpu_pubkey, "GPU pubkey must match CPU ed25519_dalek pubkey for the same seed");

        // Step 2: Verify random data matches CPU
        let cpu_random_data = generate_i2p_random_data(&gpu_seed_arr);
        assert_eq!(gpu_random_arr, cpu_random_data, "GPU random_data must match CPU random_data for the same seed");

        // Step 3: Verify full I2P address matches CPU
        let destination = construct_destination(&cpu_random_data, &cpu_pubkey);
        let mut hasher = Sha256::new();
        hasher.update(&destination);
        let hash: [u8; 32] = hasher.finalize().into();
        let cpu_address = hash_to_address(&hash);

        let mut gpu_hash_arr = [0u8; 32];
        gpu_hash_arr.copy_from_slice(&gpu_hash);
        assert_eq!(gpu_hash_arr, hash, "GPU SHA-256 hash must match CPU SHA-256 hash");
        assert_eq!(gpu_address_str, cpu_address, "GPU I2P address must match CPU-constructed address");
    }

    #[cfg(feature = "cuda-tests")]
    #[test]
    fn test_cuda_sha256() {
        let mut input = vec![0x42u8; 391];
        input[0] = 0xDE;
        input[63] = 0xAD;
        input[64] = 0xBE;
        input[383] = 0xEF;
        input[384] = 0xCA;
        input[390] = 0xFE;

        let ctx = CudaContext::new(0).expect("Failed to initialize CUDA");
        let stream = ctx.default_stream();
        let opts = CompileOptions::default();
        let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts).expect("Failed to compile CUDA kernel");
        let module = ctx.load_module(ptx).expect("Failed to load CUDA module");
        let kernel = module.load_function("test_sha256_kernel").expect("Failed to load test_sha256_kernel");

        let mut d_data = stream.alloc_zeros::<u8>(391).expect("Failed to alloc data");
        let mut d_hash = stream.alloc_zeros::<u8>(32).expect("Failed to alloc hash");
        stream.memcpy_htod(&input, &mut d_data).expect("Failed to copy data to GPU");

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(&kernel)
                .arg(&d_data)
                .arg(&mut d_hash)
                .launch(cfg)
                .expect("Kernel launch failed");
        }
        stream.synchronize().expect("Stream sync failed");

        let gpu_hash: Vec<u8> = stream.clone_dtoh(&d_hash).expect("Failed to copy hash from GPU");

        let mut hasher = Sha256::new();
        hasher.update(&input);
        let cpu_hash: [u8; 32] = hasher.finalize().into();

        assert_eq!(&gpu_hash[..], &cpu_hash[..], "GPU SHA-256 (391-byte) must match CPU");
    }

    #[cfg(feature = "cuda-tests")]
    #[test]
    fn test_cuda_sha512() {
        use sha2::{Digest, Sha512};

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
        let kernel = module.load_function("test_sha512_kernel").expect("Failed to load test_sha512_kernel");

        let mut d_seed = stream.alloc_zeros::<u8>(32).expect("Failed to alloc seed");
        let mut d_hash_out = stream.alloc_zeros::<u8>(64).expect("Failed to alloc hash_out");

        stream.memcpy_htod(&seed, &mut d_seed).expect("Failed to copy seed to GPU");

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(&kernel)
                .arg(&d_seed)
                .arg(&mut d_hash_out)
                .launch(cfg)
                .expect("Kernel launch failed");
        }

        stream.synchronize().expect("Stream sync failed");

        let gpu_hash: Vec<u8> = stream.clone_dtoh(&d_hash_out).expect("Failed to copy hash from GPU");

        // Compute SHA-512 on CPU
        let mut hasher = Sha512::new();
        hasher.update(seed);
        let cpu_hash: [u8; 64] = hasher.finalize().into();

        assert_eq!(&gpu_hash[..], &cpu_hash[..], "GPU SHA-512 must match CPU SHA-512");
    }
}
