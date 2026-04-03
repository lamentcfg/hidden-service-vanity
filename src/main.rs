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

/// Supported CUDA toolkit version ranges (major, min_minor, max_minor).
/// Update this list when adding or dropping CUDA version support.
const SUPPORTED_CUDA_VERSIONS: &[(u32, u32, u32)] = &[
    (11, 4, 8),  // CUDA 11.4 – 11.8
    (12, 0, 9),  // CUDA 12.0 – 12.9
    (13, 0, 0),  // CUDA 13.0
];

fn format_supported_versions() -> String {
    SUPPORTED_CUDA_VERSIONS
        .iter()
        .map(|(major, lo, hi)| {
            if lo == hi {
                format!("{}.{}", major, lo)
            } else {
                format!("{}.{} – {}.{}", major, lo, major, hi)
            }
        })
        .collect::<Vec<_>>()
        .join("\n    ")
}

// Kernel source embedded at compile time by build.rs
const KERNEL_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/combined_kernel.cu"));

/// On Windows, cudarc dynamically loads CUDA DLLs by trying candidate filenames
/// (e.g. `nvrtc64.dll`, `nvrtc64_1104.dll`) via LoadLibraryExW with flags=0,
/// which searches PATH. The CUDA toolkit installs versioned DLLs like
/// `nvrtc64_1302.dll` that won't match the compile-time candidate names.
///
/// This function:
/// 1. Discovers all CUDA installations (CUDA_PATH, CUDA_PATH_V* env vars,
///    and the default install directory)
/// 2. Copies the latest versioned DLLs to a temp directory with the generic
///    names cudarc expects (e.g. `nvrtc64_1302.dll` → `nvrtc64.dll`)
/// 3. Prepends all CUDA bin dirs and the temp dir to PATH
#[cfg(target_os = "windows")]
fn ensure_cuda_on_path() {
    use std::fs;

    // Collect all CUDA installation directories
    let mut cuda_dirs: Vec<std::path::PathBuf> = Vec::new();

    // CUDA_PATH (current active CUDA, set by installer)
    if let Ok(p) = std::env::var("CUDA_PATH") {
        let pb = std::path::PathBuf::from(&p);
        if pb.exists() && !cuda_dirs.contains(&pb) {
            cuda_dirs.push(pb);
        }
    }

    // CUDA_PATH_V11_4, CUDA_PATH_V12_0, etc. (all installed versions)
    for (key, val) in std::env::vars() {
        if key.starts_with("CUDA_PATH_V") {
            let pb = std::path::PathBuf::from(&val);
            if pb.exists() && !cuda_dirs.contains(&pb) {
                cuda_dirs.push(pb);
            }
        }
    }

    // Default install location
    let default_base =
        std::path::PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    if default_base.exists() {
        if let Ok(entries) = fs::read_dir(&default_base) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && !cuda_dirs.contains(&path) {
                    cuda_dirs.push(path);
                }
            }
        }
    }

    if cuda_dirs.is_empty() {
        return;
    }

    // Sort descending by version (prefer latest CUDA) so we copy the newest DLLs
    cuda_dirs.sort_by(|a, b| {
        let va = a
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .trim_start_matches('v')
            .replace('.', "");
        let vb = b
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .trim_start_matches('v')
            .replace('.', "");
        // Compare as integers (e.g. "132" vs "114")
        let na: u32 = va.parse().unwrap_or(0);
        let nb: u32 = vb.parse().unwrap_or(0);
        nb.cmp(&na)
    });

    // Staging directory for DLL copies with generic names.
    // Clear it first so stale copies from a previous CUDA installation
    // (e.g. after upgrading/downgrading CUDA) don't take precedence.
    let staging = std::env::temp_dir().join("hidden-service-vanity-cuda");
    let _ = fs::remove_dir_all(&staging);
    let _ = fs::create_dir_all(&staging);

    // CUDA libraries that cudarc dynamically loads.
    // cudarc looks for names like {prefix}64.dll, {prefix}64_1104.dll, etc.
    // We copy the actual versioned DLL as the generic {prefix}64.dll.
    let cuda_libs = ["nvrtc", "nvcuda", "cublas", "cublasLt", "curand", "nvrtc-builtins"];

    // CUDA 13+ moved DLLs into bin\x64\; older versions keep them in bin\.
    // Scan both locations for each installation.
    for cuda_dir in &cuda_dirs {
        let search_dirs: Vec<std::path::PathBuf> = vec![
            cuda_dir.join("bin"),
            cuda_dir.join("bin").join("x64"),
        ];

        for bin_dir in &search_dirs {
            if !bin_dir.exists() {
                continue;
            }

            if let Ok(entries) = fs::read_dir(bin_dir) {
                let mut dlls: Vec<(String, std::path::PathBuf)> = Vec::new();
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if !name.to_lowercase().ends_with(".dll") {
                        continue;
                    }
                    dlls.push((name, entry.path()));
                }

                for prefix in &cuda_libs {
                    let generic_name = format!("{}64.dll", prefix);
                    let dest = staging.join(&generic_name);
                    if dest.exists() {
                        continue; // Already copied from a higher-version CUDA
                    }

                    // Find the versioned DLL matching this prefix.
                    // Names like: nvrtc64_1302.dll, nvrtc64_110.dll, nvcuda64_12.dll, etc.
                    let prefix_lower = format!("{}64_", prefix.to_lowercase());
                    let mut best: Option<std::path::PathBuf> = None;
                    for (name, path) in &dlls {
                        if name.to_lowercase().starts_with(&prefix_lower) {
                            best = Some(path.clone());
                        }
                    }

                    // Also try the unversioned name directly
                    if best.is_none() {
                        let unversioned = bin_dir.join(&generic_name);
                        if unversioned.exists() {
                            best = Some(unversioned);
                        }
                    }

                    if let Some(src) = best {
                        let _ = fs::copy(&src, &dest);
                        // NVRTC loads its builtins DLL by the exact versioned name
                        // (e.g. nvrtc-builtins64_132.dll), so also stage it with
                        // the original filename alongside the generic copy.
                        if prefix.contains("builtins") {
                            if let Some(name) = src.file_name() {
                                let versioned_dest = staging.join(name);
                                if !versioned_dest.exists() {
                                    let _ = fs::copy(&src, &versioned_dest);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Build new PATH: staging dir + all CUDA bin dirs (+ x64) + original PATH
    let mut path_parts: Vec<String> = Vec::new();
    path_parts.push(staging.display().to_string());
    for dir in &cuda_dirs {
        let bin = dir.join("bin");
        let bin_x64 = dir.join("bin").join("x64");
        if bin_x64.exists() {
            path_parts.push(bin_x64.display().to_string());
        }
        if bin.exists() {
            path_parts.push(bin.display().to_string());
        }
    }
    if let Ok(current_path) = std::env::var("PATH") {
        path_parts.push(current_path);
    }
    std::env::set_var("PATH", path_parts.join(";"));
}

/// Install a panic hook that catches cudarc library loading failures and
/// prints a user-friendly message with installation instructions.
fn install_cuda_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info.to_string();
        if msg.contains("Unable to dynamically load") && msg.contains("shared library") {
            eprintln!("\n========================================");
            eprintln!("  CUDA libraries not found");
            eprintln!("========================================\n");
            eprintln!("  This program requires the CUDA Toolkit to be installed.");
            eprintln!();
            eprintln!("  Supported CUDA versions:");
            eprintln!("    {}", format_supported_versions());
            eprintln!();
            eprintln!("  Download: https://developer.nvidia.com/cuda-downloads");
            eprintln!();
            #[cfg(target_os = "windows")]
            {
                eprintln!("  Make sure the CUDA_PATH environment variable is set (the installer");
                eprintln!("  does this automatically). If you have a CUDA version installed that");
                eprintln!("  differs from the one this program was built for, the DLL names may");
                eprintln!("  not match. Try installing CUDA 11.4 or later.");
            }
            #[cfg(not(target_os = "windows"))]
            eprintln!("  Make sure ldconfig is configured, or set LD_LIBRARY_PATH to include\n  the CUDA lib directory (e.g. /usr/local/cuda/lib64).");
            eprintln!();
            eprintln!("  Original error:\n    {}", msg);
            std::process::exit(1);
        } else {
            default_hook(info);
        }
    }));
}

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
    #[cfg(target_os = "windows")]
    ensure_cuda_on_path();
    install_cuda_panic_hook();

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
