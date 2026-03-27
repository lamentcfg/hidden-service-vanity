use anyhow::{Context, Result};
use clap::Parser;
use cudarc::driver::*;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use rand::RngCore;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

mod i2p;
mod ntfy;
mod precomp_table;
mod tor;
mod utils;

use precomp_table::BASE_TABLE;

// Kernel source embedded at compile time by build.rs
const KERNEL_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/combined_kernel.cu"));

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
    ntfy_config: Option<ntfy::NtfyConfig>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load prefixes from list files if specified
    let mut i2p_prefixes = args.i2p_prefixes.clone();
    let mut tor_prefixes = args.tor_prefixes.clone();

    if let Some(ref i2p_list_path) = args.i2p_list {
        let file_prefixes = utils::load_prefixes_from_file(i2p_list_path)?;
        i2p_prefixes.extend(file_prefixes);
    }

    if let Some(ref tor_list_path) = args.tor_list {
        let file_prefixes = utils::load_prefixes_from_file(tor_list_path)?;
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
        utils::validate_prefix(prefix)
            .with_context(|| format!("Invalid I2P prefix in list: '{}'", prefix))?;
    }
    for prefix in &tor_prefixes {
        utils::validate_prefix(prefix)
            .with_context(|| format!("Invalid Tor prefix in list: '{}'", prefix))?;
    }

    let i2p_output = args.output.join("i2p");
    let tor_output = args.output.join("tor");

    // Build ntfy config if topic is provided
    let ntfy_config = args.ntfy_topic.map(|topic| ntfy::NtfyConfig {
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

/// Process I2P GPU matches: compute addresses, check prefixes, write keyfiles, send notifications.
/// Returns true if any new match was found (caller should reseed).
fn process_i2p_matches(
    stream: &Arc<CudaStream>,
    d_i2p_out_seeds: &mut CudaSlice<u8>,
    d_i2p_out_pubkeys: &mut CudaSlice<u8>,
    d_i2p_match_count: &mut CudaSlice<i32>,
    max_results: usize,
    i2p_prefixes: &[String],
    i2p_prefixes_lower: &[String],
    i2p_prefix_counts: &Arc<std::sync::Mutex<HashMap<usize, u32>>>,
    i2p_output: &PathBuf,
    max_matches: u32,
    ntfy_config: &Option<ntfy::NtfyConfig>,
    rng: &mut rand::rngs::ThreadRng,
    base_seed: &mut [u8; 32],
) -> Result<bool> {
    let match_data: Vec<i32> = stream.clone_dtoh(d_i2p_match_count)?;
    let gpu_match_count = match_data[0] as usize;

    if gpu_match_count == 0 {
        return Ok(false);
    }

    let actual_count = gpu_match_count.min(max_results);
    let slice_end = actual_count * 32;

    let matching_seeds: Vec<u8> =
        stream.clone_dtoh(&d_i2p_out_seeds.try_slice(0..slice_end)
            .expect("I2P seeds buffer slice out of bounds"))?;
    let matching_pubkeys: Vec<u8> =
        stream.clone_dtoh(&d_i2p_out_pubkeys.try_slice(0..slice_end)
            .expect("I2P pubkeys buffer slice out of bounds"))?;

    let mut found = false;

    for i in 0..actual_count {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&matching_seeds[i * 32..(i + 1) * 32]);

        let mut pubkey = [0u8; 32];
        pubkey.copy_from_slice(&matching_pubkeys[i * 32..(i + 1) * 32]);

        let (address, destination) = i2p::i2p_seed_to_address(&seed, &pubkey);

        // Check which prefix this matches and if we still need it
        if let Some(prefix_idx) = utils::find_matching_prefix(&address, i2p_prefixes_lower) {
            let mut counts = i2p_prefix_counts.lock().unwrap();
            let current_count = *counts.get(&prefix_idx).unwrap_or(&0);
            if current_count < max_matches {
                counts.insert(prefix_idx, current_count + 1);
                let matched_prefix = &i2p_prefixes[prefix_idx];
                println!("\n  Found I2P: {}.b32.i2p (prefix: '{}')", address, matched_prefix);

                let filename = format!("{}.b32.i2p.dat", address);
                let keyfile_path = i2p_output.join(&filename);
                i2p::write_keyfile(&keyfile_path, &destination, &[0u8; 256], &seed)?;
                println!("    Saved to: {}", keyfile_path.display());

                // Reseed base_seed from OS RNG so future keypairs
                // are cryptographically independent of this one
                rng.fill_bytes(base_seed);
                found = true;

                // Send ntfy notification if enabled
                if let Some(ref config) = ntfy_config {
                    if config.is_enabled() && config.on_match {
                        let msg = format!(
                            "Found: {}.b32.i2p\nPrefix: {}",
                            address, matched_prefix
                        );
                        ntfy::send_ntfy_notification(config, "Found I2P Vanity Address", &msg);
                    }
                }
            }
        }
    }

    Ok(found)
}

/// Process Tor GPU matches: compute addresses, check prefixes, write keyfiles, send notifications.
/// Returns true if any new match was found (caller should reseed).
fn process_tor_matches(
    stream: &Arc<CudaStream>,
    d_tor_out_seeds: &mut CudaSlice<u8>,
    d_tor_out_pubkeys: &mut CudaSlice<u8>,
    d_tor_match_count: &mut CudaSlice<i32>,
    max_results: usize,
    tor_prefixes: &[String],
    tor_prefixes_lower: &[String],
    tor_prefix_counts: &Arc<std::sync::Mutex<HashMap<usize, u32>>>,
    tor_output: &PathBuf,
    max_matches: u32,
    ntfy_config: &Option<ntfy::NtfyConfig>,
    rng: &mut rand::rngs::ThreadRng,
    base_seed: &mut [u8; 32],
) -> Result<bool> {
    let match_data: Vec<i32> = stream.clone_dtoh(d_tor_match_count)?;
    let gpu_match_count = match_data[0] as usize;

    if gpu_match_count == 0 {
        return Ok(false);
    }

    let actual_count = gpu_match_count.min(max_results);
    let slice_end = actual_count * 32;

    let matching_seeds: Vec<u8> =
        stream.clone_dtoh(&d_tor_out_seeds.try_slice(0..slice_end)
            .expect("Tor seeds buffer slice out of bounds"))?;
    let matching_pubkeys: Vec<u8> =
        stream.clone_dtoh(&d_tor_out_pubkeys.try_slice(0..slice_end)
            .expect("Tor pubkeys buffer slice out of bounds"))?;

    let mut found = false;

    for i in 0..actual_count {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&matching_seeds[i * 32..(i + 1) * 32]);

        let mut pubkey = [0u8; 32];
        pubkey.copy_from_slice(&matching_pubkeys[i * 32..(i + 1) * 32]);

        let address = tor::pubkey_to_tor_address(&pubkey);

        // Check which prefix this matches and if we still need it
        if let Some(prefix_idx) = utils::find_matching_prefix(&address, tor_prefixes_lower) {
            let mut counts = tor_prefix_counts.lock().unwrap();
            let current_count = *counts.get(&prefix_idx).unwrap_or(&0);
            if current_count < max_matches {
                counts.insert(prefix_idx, current_count + 1);
                let matched_prefix = &tor_prefixes[prefix_idx];
                println!("\n  Found Tor: {}.onion (prefix: '{}')", address, matched_prefix);

                tor::write_tor_keyfile(tor_output, &address, &seed)?;
                let keyfile_dir = tor_output.join(format!("{}.onion", address));
                println!("    Saved to: {}", keyfile_dir.display());

                // Reseed base_seed from OS RNG so future keypairs
                // are cryptographically independent of this one
                rng.fill_bytes(base_seed);
                found = true;

                // Send ntfy notification if enabled
                if let Some(ref config) = ntfy_config {
                    if config.is_enabled() && config.on_match {
                        let msg = format!(
                            "Found: {}.onion\nPrefix: {}",
                            address, matched_prefix
                        );
                        ntfy::send_ntfy_notification(config, "Found Tor Vanity Address", &msg);
                    }
                }
            }
        }
    }

    Ok(found)
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
            let attempts = utils::estimate_attempts(prefix.len());
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
            let attempts = utils::estimate_attempts(prefix.len());
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

        process_i2p_matches(
            &stream,
            &mut d_i2p_out_seeds,
            &mut d_i2p_out_pubkeys,
            &mut d_i2p_match_count,
            max_results,
            &args.i2p_prefixes,
            &i2p_prefixes_lower,
            &i2p_prefix_counts,
            &args.i2p_output,
            max_matches,
            &args.ntfy_config,
            &mut rng,
            &mut base_seed,
        )?;

        process_tor_matches(
            &stream,
            &mut d_tor_out_seeds,
            &mut d_tor_out_pubkeys,
            &mut d_tor_match_count,
            max_results,
            &args.tor_prefixes,
            &tor_prefixes_lower,
            &tor_prefix_counts,
            &args.tor_output,
            max_matches,
            &args.ntfy_config,
            &mut rng,
            &mut base_seed,
        )?;

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
            ntfy::send_ntfy_notification(config, "Vanity Search Complete", &msg);
        }
    }

    Ok(())
}
