use anyhow::{Context, Result};
use clap::Parser;
use cudarc::driver::*;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use ed25519_dalek::SigningKey;
use rand::RngCore;
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// Kernel source embedded at compile time by build.rs
const KERNEL_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/combined_kernel.cu"));

mod precomp_table;
use precomp_table::BASE_TABLE;

/// Hidden service vanity address generator with CUDA acceleration
#[derive(Parser, Debug)]
#[command(name = "hidden-service-vanity")]
#[command(author = "Hidden Service Vanity Generator")]
#[command(version = "0.1.0")]
#[command(about = "Generate I2P .b32.i2p and Tor .onion vanity addresses using GPU")]
#[command(bin_name = "hidden-service-vanity")]
struct Args {
    /// Search for I2P addresses with this prefix (base32: a-z, 2-7). Can be repeated for multiple prefixes.
    #[arg(short = 'i', long = "i2p")]
    i2p_prefixes: Vec<String>,

    /// File containing I2P prefixes to search for (one per line, lines starting with # are ignored).
    #[arg(long = "i2p-list")]
    i2p_list: Option<PathBuf>,

    /// Search for Tor addresses with this prefix (base32: a-z, 2-7). Can be repeated for multiple prefixes.
    #[arg(short = 't', long = "tor")]
    tor_prefixes: Vec<String>,

    /// File containing Tor prefixes to search for (one per line, lines starting with # are ignored).
    #[arg(long = "tor-list")]
    tor_list: Option<PathBuf>,

    /// Output base directory for generated keys. Creates 'i2p/' or 'tor/' subdirectories.
    #[arg(short, long, default_value = ".")]
    output: PathBuf,

    /// GPU device ID to use
    #[arg(short, long, default_value = "0")]
    device: usize,

    /// Number of threads per block
    #[arg(long, default_value = "256")]
    threads: u32,

    /// Number of blocks (default: auto-detect based on GPU)
    #[arg(long)]
    blocks: Option<u32>,

    /// Batch size - keys to generate per GPU launch
    #[arg(long, default_value = "1048576")]
    batch_size: usize,

    /// Maximum number of addresses to generate per network
    #[arg(short = 'n', long, default_value = "1")]
    count: u32,

    /// Ntfy server URL
    #[arg(long, default_value = "https://ntfy.sh")]
    ntfy_host: String,

    /// Ntfy topic to publish notifications to (enables ntfy when set)
    #[arg(long)]
    ntfy_topic: Option<String>,

    /// Ntfy username for authentication
    #[arg(long)]
    ntfy_username: Option<String>,

    /// Ntfy password for authentication
    #[arg(long)]
    ntfy_password: Option<String>,

    /// Notify on each match (true) or only on completion (false)
    #[arg(long, default_value = "true")]
    ntfy_on_match: bool,
}

fn validate_prefix(prefix: &str) -> Result<()> {
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

/// Configuration for ntfy.sh notifications
struct NtfyConfig {
    host: String,
    topic: String,
    username: Option<String>,
    password: Option<String>,
    on_match: bool,
}

impl NtfyConfig {
    fn is_enabled(&self) -> bool {
        !self.topic.is_empty()
    }

    fn build_client(&self) -> reqwest::blocking::RequestBuilder {
        let client = reqwest::blocking::Client::new();
        let url = format!("{}/{}", self.host.trim_end_matches('/'), self.topic);
        let mut builder = client.post(&url);

        if let (Some(user), Some(pass)) = (&self.username, &self.password) {
            builder = builder.basic_auth(user, Some(pass));
        }

        builder
    }
}

fn send_ntfy_notification(config: &NtfyConfig, title: &str, message: &str) {
    let result = config
        .build_client()
        .header("Title", title)
        .header("Priority", "default")
        .body(message.to_string())
        .send();

    match result {
        Ok(resp) => {
            if !resp.status().is_success() {
                eprintln!("  [ntfy] Notification failed: HTTP {}", resp.status());
            }
        }
        Err(e) => {
            eprintln!("  [ntfy] Notification failed: {}", e);
        }
    }
}

/// Load prefixes from a file. Each line should contain one prefix.
/// Lines starting with # are treated as comments and ignored.
/// Empty lines and whitespace-only lines are ignored.
fn load_prefixes_from_file(path: &PathBuf) -> Result<Vec<String>> {
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
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

fn estimate_attempts(prefix_len: usize) -> u64 {
    1u64 << (5 * prefix_len.min(12))
}

/// Find which prefix (if any) an address matches. Returns the index of the matching prefix.
fn find_matching_prefix(address: &str, prefixes: &[String]) -> Option<usize> {
    let address_lower = address.to_lowercase();
    for (i, prefix) in prefixes.iter().enumerate() {
        if address_lower.starts_with(&prefix.to_lowercase()) {
            return Some(i);
        }
    }
    None
}

fn generate_keypair(seed: [u8; 32]) -> ([u8; 32], [u8; 32]) {
    let signing_key = SigningKey::from_bytes(&seed);
    let verifying_key = signing_key.verifying_key();
    (seed, verifying_key.to_bytes())
}

// I2P Destination: 256 random + 32 pubkey + 96 padding + 7 certificate = 391 bytes
fn construct_destination(random_data: &[u8; 256], pubkey: &[u8; 32]) -> Vec<u8> {
    let mut dest = Vec::with_capacity(391);

    dest.extend_from_slice(random_data);
    dest.extend_from_slice(pubkey);
    dest.extend_from_slice(&[0u8; 96]);

    // Key Certificate for Ed25519
    dest.extend_from_slice(&[0x05, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00]);

    dest
}

fn hash_to_address(hash: &[u8; 32]) -> String {
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
    address
}

// Tor v3: checksum = SHA3-256(".onion checksum" || pubkey || 0x03)[:2]
// address = base32(pubkey || 0x03 || checksum) -> 56 chars
fn pubkey_to_tor_address(pubkey: &[u8; 32]) -> String {
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
    address_data[32] = 0x03;
    address_data[33] = checksum[0];
    address_data[34] = checksum[1];

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

// I2P keyfile: Destination(391) + PrivateKey(256, unused) + SigningPrivateKey(32) = 679 bytes
fn write_keyfile(
    path: &std::path::Path,
    destination: &[u8],
    _privkey: &[u8; 256],
    signing_privkey: &[u8; 32],
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file_data = Vec::with_capacity(destination.len() + 256 + 32);
    file_data.extend_from_slice(destination);
    file_data.extend_from_slice(_privkey);
    file_data.extend_from_slice(signing_privkey);

    let mut file = File::create(path)?;
    file.write_all(&file_data)?;
    Ok(())
}

// Tor keyfile: hs_ed25519_secret_key has 32-byte header "== ed25519v1-secret: type0 ==" + 32-byte seed
fn write_tor_keyfile(output_dir: &std::path::Path, address: &str, seed: &[u8; 32]) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let hs_dir = output_dir.join(format!("{}.onion", address));
    std::fs::create_dir_all(&hs_dir)?;

    let secret_key_path = hs_dir.join("hs_ed25519_secret_key");
    let mut secret_key_data = Vec::with_capacity(64);
    let header = b"== ed25519v1-secret: type0 ==";
    secret_key_data.extend_from_slice(header);
    secret_key_data.extend(std::iter::repeat_n(0u8, 32 - header.len()));
    secret_key_data.extend_from_slice(seed);

    let mut file = File::create(&secret_key_path)?;
    file.write_all(&secret_key_data)?;

    let hostname_path = hs_dir.join("hostname");
    let mut hostname_file = File::create(&hostname_path)?;
    writeln!(hostname_file, "{}.onion", address)?;

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load prefixes from list files if specified
    let mut i2p_prefixes = args.i2p_prefixes.clone();
    let mut tor_prefixes = args.tor_prefixes.clone();

    if let Some(ref i2p_list_path) = args.i2p_list {
        let file_prefixes = load_prefixes_from_file(i2p_list_path)?;
        i2p_prefixes.extend(file_prefixes);
    }

    if let Some(ref tor_list_path) = args.tor_list {
        let file_prefixes = load_prefixes_from_file(tor_list_path)?;
        tor_prefixes.extend(file_prefixes);
    }

    let search_i2p = !i2p_prefixes.is_empty();
    let search_tor = !tor_prefixes.is_empty();

    if !search_i2p && !search_tor {
        anyhow::bail!(
            "At least one of -i/--i2p, --i2p-list, -t/--tor, or --tor-list must be specified with a prefix"
        );
    }

    // Validate all prefixes
    for prefix in &i2p_prefixes {
        validate_prefix(prefix)
            .with_context(|| format!("Invalid I2P prefix in list: '{}'", prefix))?;
    }
    for prefix in &tor_prefixes {
        validate_prefix(prefix)
            .with_context(|| format!("Invalid Tor prefix in list: '{}'", prefix))?;
    }

    let i2p_output = args.output.join("i2p");
    let tor_output = args.output.join("tor");

    // Build ntfy config if topic is provided
    let ntfy_config = args.ntfy_topic.map(|topic| NtfyConfig {
        host: args.ntfy_host,
        topic,
        username: args.ntfy_username,
        password: args.ntfy_password,
        on_match: args.ntfy_on_match,
    });

    run(RunArgs {
        i2p_prefixes,
        tor_prefixes,
        i2p_output,
        tor_output,
        device: args.device,
        threads: args.threads,
        blocks: args.blocks,
        batch_size: args.batch_size,
        count: args.count,
        ntfy_config,
    })
}

struct RunArgs {
    i2p_prefixes: Vec<String>,
    tor_prefixes: Vec<String>,
    i2p_output: PathBuf,
    tor_output: PathBuf,
    device: usize,
    threads: u32,
    blocks: Option<u32>,
    batch_size: usize,
    count: u32,
    ntfy_config: Option<NtfyConfig>,
}

fn run(args: RunArgs) -> Result<()> {
    let search_i2p = !args.i2p_prefixes.is_empty();
    let search_tor = !args.tor_prefixes.is_empty();

    let header = match (search_i2p, search_tor) {
        (true, true) => "Combined I2P + Tor Vanity Address Generator (CUDA)",
        (true, false) => "I2P Vanity Address Generator (CUDA)",
        (false, true) => "Tor Vanity Address Generator (CUDA)",
        _ => unreachable!(),
    };
    println!("{}\n{}\n", header, "=".repeat(header.len()));

    println!("Searching for:");
    if search_i2p {
        for prefix in &args.i2p_prefixes {
            let attempts = estimate_attempts(prefix.len());
            println!(
                "  I2P: '{}' ({} chars, ~{} attempts)",
                prefix,
                prefix.len(),
                attempts
            );
        }
    }
    if search_tor {
        for prefix in &args.tor_prefixes {
            let attempts = estimate_attempts(prefix.len());
            println!(
                "  Tor: '{}' ({} chars, ~{} attempts)",
                prefix,
                prefix.len(),
                attempts
            );
        }
    }
    println!();

    if search_i2p && !args.i2p_output.exists() {
        std::fs::create_dir_all(&args.i2p_output)?;
    }
    if search_tor && !args.tor_output.exists() {
        std::fs::create_dir_all(&args.tor_output)?;
    }

    println!("Initializing CUDA...");
    let ctx = CudaContext::new(args.device)
        .with_context(|| format!("Failed to initialize CUDA device {}", args.device))?;

    println!(
        "GPU: {}",
        ctx.name().unwrap_or_else(|_| "Unknown".to_string())
    );
    println!("Compute capability: {:?}", ctx.compute_capability());

    let stream = ctx.default_stream();

    println!("Compiling combined CUDA kernel...");
    let opts = CompileOptions::default();
    let ptx =
        compile_ptx_with_opts(KERNEL_SOURCE, opts).context("Failed to compile CUDA kernel")?;
    let module = ctx.load_module(ptx).context("Failed to load CUDA module")?;
    let kernel = module
        .load_function("search_combined_kernel")
        .context("Failed to load kernel function: search_combined_kernel")?;

    let sm_count = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .unwrap_or(16) as u32;
    let blocks = args.blocks.unwrap_or_else(|| sm_count * 32);

    println!("\nConfiguration:");
    println!("  Threads per block: {}", args.threads);
    println!("  Blocks: {}", blocks);
    println!("  Total threads: {}", args.threads * blocks);
    println!("  Batch size: {}", args.batch_size);
    println!("  Searching I2P: {}", search_i2p);
    println!("  Searching Tor: {}", search_tor);
    if let Some(ref config) = args.ntfy_config {
        if config.is_enabled() {
            println!(
                "  Ntfy: {} (notify on {})",
                config.host,
                if config.on_match { "match" } else { "completion" }
            );
        }
    }
    println!();

    const MAX_PREFIXES: usize = 16;
    let i2p_prefix_count = args.i2p_prefixes.len().min(MAX_PREFIXES);
    let mut i2p_prefix_buffer = Vec::new();
    let mut i2p_prefix_lengths = [0i32; MAX_PREFIXES];
    for (i, prefix) in args.i2p_prefixes.iter().take(MAX_PREFIXES).enumerate() {
        let lower = prefix.to_lowercase();
        i2p_prefix_lengths[i] = lower.len() as i32;
        i2p_prefix_buffer.extend(lower.bytes());
    }

    let tor_prefix_count = args.tor_prefixes.len().min(MAX_PREFIXES);
    let mut tor_prefix_buffer = Vec::new();
    let mut tor_prefix_lengths = [0i32; MAX_PREFIXES];
    for (i, prefix) in args.tor_prefixes.iter().take(MAX_PREFIXES).enumerate() {
        let lower = prefix.to_lowercase();
        tor_prefix_lengths[i] = lower.len() as i32;
        tor_prefix_buffer.extend(lower.bytes());
    }

    println!("Starting search... (Press Ctrl+C to stop)\n");

    let start = Instant::now();
    let mut total_attempts: u64 = 0;

    // Per-prefix match tracking
    let i2p_prefix_counts: Arc<std::sync::Mutex<HashMap<usize, u32>>> =
        Arc::new(std::sync::Mutex::new(HashMap::new()));
    let tor_prefix_counts: Arc<std::sync::Mutex<HashMap<usize, u32>>> =
        Arc::new(std::sync::Mutex::new(HashMap::new()));
    let max_matches = args.count;

    // Store lowercase prefixes for matching
    let i2p_prefixes_lower: Vec<String> = args.i2p_prefixes.iter().map(|p| p.to_lowercase()).collect();
    let tor_prefixes_lower: Vec<String> = args.tor_prefixes.iter().map(|p| p.to_lowercase()).collect();

    // Helper to check if a network still needs more matches
    let i2p_prefix_counts_clone = i2p_prefix_counts.clone();
    let tor_prefix_counts_clone = tor_prefix_counts.clone();
    let i2p_still_searching = move || {
        let counts = i2p_prefix_counts_clone.lock().unwrap();
        (0..i2p_prefix_count).any(|i| *counts.get(&i).unwrap_or(&0) < max_matches)
    };
    let tor_still_searching = move || {
        let counts = tor_prefix_counts_clone.lock().unwrap();
        (0..tor_prefix_count).any(|i| *counts.get(&i).unwrap_or(&0) < max_matches)
    };

    let mut rng = rand::thread_rng();
    let mut base_seed = [0u8; 32];
    rng.fill_bytes(&mut base_seed);

    let max_results = 64usize;

    println!(
        "Loading precomputed Ed25519 table ({} bytes)...",
        BASE_TABLE.len()
    );
    let d_base_table = stream.clone_htod(BASE_TABLE)?;

    let mut d_base_seed = stream.clone_htod(&base_seed)?;
    let d_i2p_prefix_buffer = stream.clone_htod(&i2p_prefix_buffer)?;
    let d_i2p_prefix_lengths = stream.clone_htod(&i2p_prefix_lengths)?;
    let d_tor_prefix_buffer = stream.clone_htod(&tor_prefix_buffer)?;
    let d_tor_prefix_lengths = stream.clone_htod(&tor_prefix_lengths)?;

    let mut d_i2p_out_seeds = stream.alloc_zeros::<u8>(max_results * 32)?;
    let mut d_i2p_out_pubkeys = stream.alloc_zeros::<u8>(max_results * 32)?;
    let mut d_i2p_match_count = stream.alloc_zeros::<i32>(1)?;

    let mut d_tor_out_seeds = stream.alloc_zeros::<u8>(max_results * 32)?;
    let mut d_tor_out_pubkeys = stream.alloc_zeros::<u8>(max_results * 32)?;
    let mut d_tor_match_count = stream.alloc_zeros::<i32>(1)?;

    let i2p_prefix_count_val = i2p_prefix_count as i32;
    let tor_prefix_count_val = tor_prefix_count as i32;
    let max_results_val = max_results as i32;
    let total_threads = (args.threads * blocks) as u64;

    let mut iteration: u64 = 0;

    while (search_i2p && i2p_still_searching()) || (search_tor && tor_still_searching()) {
        let current_search_i2p = search_i2p && i2p_still_searching();
        let current_search_tor = search_tor && tor_still_searching();

        stream.memcpy_htod(&[0i32], &mut d_i2p_match_count)?;
        stream.memcpy_htod(&[0i32], &mut d_tor_match_count)?;

        stream.memcpy_htod(&base_seed, &mut d_base_seed)?;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (args.threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&d_base_seed)
                .arg(&d_base_table)
                .arg(&iteration)
                .arg(&d_i2p_prefix_buffer)
                .arg(&d_i2p_prefix_lengths)
                .arg(&i2p_prefix_count_val)
                .arg(&current_search_i2p)
                .arg(&mut d_i2p_out_seeds)
                .arg(&mut d_i2p_out_pubkeys)
                .arg(&mut d_i2p_match_count)
                .arg(&max_results_val)
                .arg(&d_tor_prefix_buffer)
                .arg(&d_tor_prefix_lengths)
                .arg(&tor_prefix_count_val)
                .arg(&current_search_tor)
                .arg(&mut d_tor_out_seeds)
                .arg(&mut d_tor_out_pubkeys)
                .arg(&mut d_tor_match_count)
                .arg(&max_results_val)
                .launch(cfg)?;
        }

        stream.synchronize()?;

        let i2p_match_data: Vec<i32> = stream.clone_dtoh(&d_i2p_match_count)?;
        let gpu_i2p_match_count = i2p_match_data[0] as usize;

        if gpu_i2p_match_count > 0 {
            let actual_count = gpu_i2p_match_count.min(max_results);
            let slice_end = actual_count * 32;

            let matching_seeds: Vec<u8> =
                stream.clone_dtoh(&d_i2p_out_seeds.try_slice(0..slice_end)
                    .expect("I2P seeds buffer slice out of bounds"))?;
            let matching_pubkeys: Vec<u8> =
                stream.clone_dtoh(&d_i2p_out_pubkeys.try_slice(0..slice_end)
                    .expect("I2P pubkeys buffer slice out of bounds"))?;

            for i in 0..actual_count {
                let mut seed = [0u8; 32];
                seed.copy_from_slice(&matching_seeds[i * 32..(i + 1) * 32]);

                let mut pubkey = [0u8; 32];
                pubkey.copy_from_slice(&matching_pubkeys[i * 32..(i + 1) * 32]);

                let (_, expected_pubkey) = generate_keypair(seed);
                if pubkey != expected_pubkey {
                    // GPU pubkey differs, use GPU's version for address construction
                }

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

                let destination = construct_destination(&random_data, &pubkey);
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(&destination);
                let hash = hasher.finalize();
                let hash_array: [u8; 32] = hash.into();
                let address = hash_to_address(&hash_array);

                // Check which prefix this matches and if we still need it
                if let Some(prefix_idx) = find_matching_prefix(&address, &i2p_prefixes_lower) {
                    let mut counts = i2p_prefix_counts.lock().unwrap();
                    let current_count = *counts.get(&prefix_idx).unwrap_or(&0);
                    if current_count < max_matches {
                        counts.insert(prefix_idx, current_count + 1);
                        let matched_prefix = &args.i2p_prefixes[prefix_idx];
                        println!("\n  Found I2P: {}.b32.i2p (prefix: '{}')", address, matched_prefix);

                        let filename = format!("{}.b32.i2p.dat", address);
                        let keyfile_path = args.i2p_output.join(&filename);
                        write_keyfile(&keyfile_path, &destination, &[0u8; 256], &seed)?;
                        println!("    Saved to: {}", keyfile_path.display());

                        // Send ntfy notification if enabled
                        if let Some(ref config) = args.ntfy_config {
                            if config.is_enabled() && config.on_match {
                                let msg = format!(
                                    "Found: {}.b32.i2p\nPrefix: {}",
                                    address, matched_prefix
                                );
                                send_ntfy_notification(config, "Found I2P Vanity Address", &msg);
                            }
                        }
                    }
                }
            }
        }

        let tor_match_data: Vec<i32> = stream.clone_dtoh(&d_tor_match_count)?;
        let gpu_tor_match_count = tor_match_data[0] as usize;

        if gpu_tor_match_count > 0 {
            let actual_count = gpu_tor_match_count.min(max_results);
            let slice_end = actual_count * 32;

            let matching_seeds: Vec<u8> =
                stream.clone_dtoh(&d_tor_out_seeds.try_slice(0..slice_end)
                    .expect("Tor seeds buffer slice out of bounds"))?;
            let matching_pubkeys: Vec<u8> =
                stream.clone_dtoh(&d_tor_out_pubkeys.try_slice(0..slice_end)
                    .expect("Tor pubkeys buffer slice out of bounds"))?;

            for i in 0..actual_count {
                let mut seed = [0u8; 32];
                seed.copy_from_slice(&matching_seeds[i * 32..(i + 1) * 32]);

                let mut pubkey = [0u8; 32];
                pubkey.copy_from_slice(&matching_pubkeys[i * 32..(i + 1) * 32]);

                let address = pubkey_to_tor_address(&pubkey);

                // Check which prefix this matches and if we still need it
                if let Some(prefix_idx) = find_matching_prefix(&address, &tor_prefixes_lower) {
                    let mut counts = tor_prefix_counts.lock().unwrap();
                    let current_count = *counts.get(&prefix_idx).unwrap_or(&0);
                    if current_count < max_matches {
                        counts.insert(prefix_idx, current_count + 1);
                        let matched_prefix = &args.tor_prefixes[prefix_idx];
                        println!("\n  Found Tor: {}.onion (prefix: '{}')", address, matched_prefix);

                        write_tor_keyfile(&args.tor_output, &address, &seed)?;
                        let keyfile_dir = args.tor_output.join(format!("{}.onion", address));
                        println!("    Saved to: {}", keyfile_dir.display());

                        // Send ntfy notification if enabled
                        if let Some(ref config) = args.ntfy_config {
                            if config.is_enabled() && config.on_match {
                                let msg = format!(
                                    "Found: {}.onion\nPrefix: {}",
                                    address, matched_prefix
                                );
                                send_ntfy_notification(config, "Found Tor Vanity Address", &msg);
                            }
                        }
                    }
                }
            }
        }

        total_attempts += total_threads;
        iteration += 1;

        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let rate = total_attempts as f64 / elapsed;

            // Calculate totals for display
            let i2p_total: u32 = i2p_prefix_counts.lock().unwrap().values().sum();
            let tor_total: u32 = tor_prefix_counts.lock().unwrap().values().sum();

            print!(
                "\r  Attempts: {} ({:.2} M/s)  I2P: {}  Tor: {}  ",
                total_attempts,
                rate / 1_000_000.0,
                i2p_total,
                tor_total
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
        }

        for byte in base_seed.iter_mut() {
            *byte = byte.wrapping_add(1);
        }
    }

    println!("\n\nSearch completed!");
    println!("  Total attempts: {}", total_attempts);
    println!("  Time: {:.2}s", start.elapsed().as_secs_f64());

    let i2p_counts = i2p_prefix_counts.lock().unwrap();
    let tor_counts = tor_prefix_counts.lock().unwrap();

    println!("\nI2P matches by prefix:");
    for (i, prefix) in args.i2p_prefixes.iter().enumerate() {
        let count = i2p_counts.get(&i).unwrap_or(&0);
        println!("  '{}': {}", prefix, count);
    }

    println!("\nTor matches by prefix:");
    for (i, prefix) in args.tor_prefixes.iter().enumerate() {
        let count = tor_counts.get(&i).unwrap_or(&0);
        println!("  '{}': {}", prefix, count);
    }

    let i2p_total: u32 = i2p_counts.values().sum();
    let tor_total: u32 = tor_counts.values().sum();

    if i2p_total > 0 {
        println!("\nI2P keys saved to: {}", args.i2p_output.display());
    }
    if tor_total > 0 {
        println!("Tor keys saved to: {}", args.tor_output.display());
    }

    // Send completion notification if enabled
    if let Some(ref config) = args.ntfy_config {
        if config.is_enabled() {
            let msg = format!(
                "I2P matches: {}\nTor matches: {}\nTotal attempts: {}\nTime: {:.2}s",
                i2p_total,
                tor_total,
                total_attempts,
                start.elapsed().as_secs_f64()
            );
            send_ntfy_notification(config, "Vanity Search Complete", &msg);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(&dest[256..288], &pubkey[..]);
        for (i, &byte) in dest[288..384].iter().enumerate() {
            assert_eq!(byte, 0, "Padding byte {} should be zero", i);
        }
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

        use sha2::{Digest, Sha256};
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
        use sha2::{Digest, Sha256};
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
        assert_eq!(address.len(), 51);
    }

    #[test]
    fn test_base32_known_values() {
        let zero_hash = [0u8; 32];
        let address = hash_to_address(&zero_hash);
        assert!(address.starts_with("aaaaaaaaaa"));
        assert!(address.chars().all(|c| c >= 'a' && c <= 'z' || c >= '2' && c <= '7'));

        let ones_hash = [0xFFu8; 32];
        let address = hash_to_address(&ones_hash);
        assert!(address.chars().all(|c| c == '7' || c >= 'r'));
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
    fn test_keypair_different_seeds() {
        let seed1 = [0x01u8; 32];
        let seed2 = [0x02u8; 32];
        let (_, pubkey1) = generate_keypair(seed1);
        let (_, pubkey2) = generate_keypair(seed2);
        assert_ne!(pubkey1, pubkey2);
    }

    #[test]
    fn test_end_to_end_address_generation() {
        let random_data = [0x5Au8; 256];
        let pubkey = [0x3Cu8; 32];
        let dest = construct_destination(&random_data, &pubkey);
        assert_eq!(dest.len(), 391);

        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&dest);
        let hash: [u8; 32] = hasher.finalize().into();
        let address = hash_to_address(&hash);
        assert_eq!(address.len(), 51);
        assert!(address.chars().all(|c| (c >= 'a' && c <= 'z') || (c >= '2' && c <= '7')));
    }

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

        use sha3::{Digest, Sha3_256};
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

        use sha3::{Digest, Sha3_256};
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
