# Hidden Service Vanity Address Generator (CUDA)

Generate custom vanity addresses for I2P (`.b32.i2p`) and Tor v3 (`.onion`) hidden services using GPU acceleration.

## Features

- **GPU Acceleration**: Uses CUDA for parallel address generation
- **Dual Network Support**: Generate addresses for I2P and Tor v3
- **Ed25519 Signatures**: Uses recommended Ed25519 signing keys
- **I2P Compatible**: Generates keys compatible with i2pd and Java I2P routers
- **Tor Compatible**: Generates keys compatible with tor hidden services
- **Case-insensitive Matching**: Search for prefixes regardless of case
- **Multi-prefix Search**: Search for multiple prefixes simultaneously
- **Push Notifications**: Optional ntfy.sh notifications when addresses are found

## Prerequisites

- **Rust** 1.70 or later
- **CUDA Toolkit** 11.4+ (for driver and nvrtc libraries)
- **NVIDIA GPU** with CUDA support (Compute capability 3.0+)
- **NVIDIA Drivers** compatible with your CUDA version

## Installation

```bash
# Clone the repository
git clone https://github.com/lamentcfg/hidden-service-vanity.git
cd hidden-service-vanity

# Build in release mode
cargo build --release
```

## Usage

### Basic Usage

```bash
# Search for I2P addresses starting with "test"
./target/release/hidden-service-vanity -i test -o ./keys

# Search for Tor addresses starting with "abc"
./target/release/hidden-service-vanity -t abc -o ./keys

# Search both networks simultaneously
./target/release/hidden-service-vanity -i test -t abc -o ./keys

# Search for multiple I2P prefixes
./target/release/hidden-service-vanity -i foo -i bar -o ./keys
```

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--i2p` | `-i` | (none) | I2P prefix to search for (base32: a-z, 2-7). Can be repeated. |
| `--i2p-list` | | (none) | File containing I2P prefixes (one per line, # for comments) |
| `--tor` | `-t` | (none) | Tor prefix to search for (base32: a-z, 2-7). Can be repeated. |
| `--tor-list` | | (none) | File containing Tor prefixes (one per line, # for comments) |
| `--output` | `-o` | `.` | Base output directory (creates `i2p/` or `tor/` subdirectories) |
| `--device` | `-d` | `0` | GPU device ID to use |
| `--threads` | | `256` | Threads per block |
| `--blocks` | | auto | Number of blocks (auto-detected from GPU) |
| `--batch-size` | | `1048576` | Keys per GPU launch |
| `--count` | `-n` | `1` | Number of addresses to generate per network |
| `--ntfy-host` | | `https://ntfy.sh` | Ntfy server URL |
| `--ntfy-topic` | | (none) | Ntfy topic (enables notifications when set) |
| `--ntfy-username` | | (none) | Ntfy username for authentication |
| `--ntfy-password` | | (none) | Ntfy password for authentication |
| `--ntfy-on-match` | | `true` | Notify on each match (true) or only on completion (false) |

### Examples

```bash
# Generate an I2P address starting with "test"
./target/release/hidden-service-vanity -i test -o ./keys

# Generate a Tor address starting with "abc"
./target/release/hidden-service-vanity -t abc -o ./keys

# Generate both I2P and Tor addresses at once (shares keypair generation)
./target/release/hidden-service-vanity -i foo -t bar -o ./keys

# Generate multiple I2P addresses
./target/release/hidden-service-vanity -i test -n 5 -o ./keys

# Search for multiple prefixes on the same network
./target/release/hidden-service-vanity -i abc -i def -i xyz -o ./keys

# Load prefixes from a file (one per line, # for comments)
./target/release/hidden-service-vanity --i2p-list prefixes.txt -o ./keys

# Mix command-line and file-based prefixes
./target/release/hidden-service-vanity -i foo --i2p-list prefixes.txt -t bar --tor-list tor_prefixes.txt -o ./keys

# Enable ntfy notifications when a match is found
./target/release/hidden-service-vanity -i test --ntfy-topic my-vanity-search -o ./keys

# Use custom ntfy server with authentication
./target/release/hidden-service-vanity -i test --ntfy-host https://ntfy.example.com --ntfy-topic my-topic --ntfy-username user --ntfy-password pass -o ./keys
```

## Performance Estimates

| Prefix Length | Expected Attempts | Estimated Time* |
|---------------|-------------------|-----------------|
| 3 chars | ~32,768 | seconds |
| 4 chars | ~1,048,576 | seconds |
| 5 chars | ~33,554,432 | minutes |
| 6 chars | ~1,073,741,824 | hours |
| 7 chars | ~34,359,738,368 | days |
| 8 chars | ~1,099,511,627,776 | weeks |

*Estimates vary significantly based on GPU model and clock speed.

## Contributing

This repository is not open to contributions. However, you are welcome to fork the project for your own use. If you encounter bugs or have feature requests, feel free to open an issue.

## License

MIT License
