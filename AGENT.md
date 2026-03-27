# AGENT.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hidden Service Vanity Address Generator with CUDA acceleration. Generates custom vanity addresses for I2P (`.b32.i2p`) and Tor v3 (`.onion`) hidden services by brute-forcing Ed25519 keypairs on GPU.

## Build & Run Commands

```bash
# Build (release mode required - debug builds are 100x+ slower due to Ed25519)
cargo build --release

# Run tests
cargo test

# I2P usage - outputs to ./keys/i2p/
./target/release/hidden-service-vanity -i test -o ./keys
./target/release/hidden-service-vanity -i abc -i def -o ./keys  # multiple prefixes

# Tor usage - outputs to ./keys/tor/
./target/release/hidden-service-vanity -t test -o ./keys

# Both I2P and Tor - outputs to ./keys/i2p/ and ./keys/tor/
./target/release/hidden-service-vanity -i abc -t xyz -o ./keys
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --i2p` | (none) | I2P prefix (base32: a-z, 2-7). Repeatable. |
| `-t, --tor` | (none) | Tor prefix (base32: a-z, 2-7). Repeatable. |
| `-o, --output` | `.` | Output directory (creates `i2p/` or `tor/` subdirs) |
| `-d, --device` | `0` | GPU device ID |
| `--threads` | `256` | Threads per block |
| `--blocks` | auto | GPU blocks (auto: SM count × 32) |
| `--batch-size` | `1048576` | Keys per GPU launch |
| `-n, --count` | `1` | Addresses to generate per network |

## Requirements

- Rust 1.70+
- CUDA Toolkit 11.4+ (driver + nvrtc libraries)
- NVIDIA GPU with compute capability 3.0+

## Architecture

### Two-Part System

1. **CPU (Rust)** - `src/main.rs`
   - CLI parsing with clap using flags (`-i/--i2p` and `-t/--tor`)
   - CUDA context management via `cudarc`
   - Match verification and key file output
   - Supports multiple prefix search (up to 16 prefixes per network)

2. **GPU - Combined Kernel** - `kernels/combined_vanity.cu`
   - Shared Ed25519 keypair generation (SHA-512 → scalar → scalar mult)
   - I2P path: Destination construction (256 random + 96 zero pad + 32 pubkey + 7 cert = 391 bytes) → SHA-256 → Base32 (52 chars)
   - Tor path: checksum = SHA3-256(".onion checksum" || pubkey || 0x03)[:2] → Base32
   - Multi-prefix matching with case-insensitive comparison for both networks

### CUDA Primitives - `kernels/primitives/`

| File | Purpose |
|------|---------|
| `sha256.cuh` | SHA-256 for I2P address hashing |
| `sha512.cuh` | SHA-512 for Ed25519 scalar derivation |
| `sha3.cuh` | SHA3-256 for Tor checksum |
| `base32.cuh` | RFC4648 Base32 encoding |
| `prng.cuh` | xorshift64 PRNG |
| `types.cuh` | Common type definitions |
| `ed25519/ed25519_ref10.cuh` | Ed25519 header with scalar derivation |
| `ed25519/common_ref10.cu` | Common field operations |
| `ed25519/fe_ref10.cu` | Field element arithmetic |
| `ed25519/ge_ref10.cu` | Group element operations and scalar mult |

### Key Generation Flow (Full GPU Pipeline)

```
xorshift64(tid, iteration) → 32-byte seed → SHA-512(seed) → scalar → ge_scalarmult_base_with_table → Ed25519 pubkey
```

### Precomputed Ed25519 Base Table

- `src/precomp_table.rs` - Embeds 30,720-byte precomputed base point table
- Loaded at runtime and copied to GPU device memory
- Enables fast scalar multiplication without constant memory limitations

### CUDA Kernel Compilation

- Kernels are JIT-compiled at runtime using nvrtc (NVIDIA Runtime Compiler)
- Kernel sources (`.cu` files) are embedded in the binary via `include_str!`
- Include paths for kernels: `kernels/` and `kernels/primitives/`

### I2P Address Derivation

```
Destination (391 bytes) → SHA-256 (32 bytes) → Base32 (52 chars) → ".b32.i2p"
```

#### Destination Binary Structure (ElGamal 2048 + Ed25519)

Total: **391 bytes**. Source: I2P router `KeysAndCert.writeBytes()`.

| Offset | Size | Field | Value |
|--------|------|-------|-------|
| 0–255 | 256 | Crypto Public Key (ElGamal) | Random data for vanity gen; unused for address validity |
| 256–351 | 96 | Padding | Zeros in our impl; real I2P routers use leading bytes of raw signing key slot |
| 352–383 | 32 | Signing Public Key (Ed25519) | Right-aligned in the 128-byte signing key slot |
| 384–390 | 7 | Key Certificate | Fixed bytes (see below) |

**Layout rule** (from I2P spec): "Crypto Public Key is aligned at the start and Signing Public Key is aligned at the end."

#### Key Certificate (7 bytes, fixed for ElGamal+Ed25519)

Source: I2P router `KeyCertificate.Ed25519Cert.ED_DATA`.

```
Byte 0:   0x05  — Certificate type (CERTIFICATE_TYPE_KEY)
Byte 1–2: 0x00 0x04 — Payload length (4 bytes, big-endian)
Byte 3–4: 0x00 0x07 — Sig type: EdDSA-SHA512-Ed25519 (code 7)
Byte 5–6: 0x00 0x00 — Crypto type: ElGamal-2048 (code 0)
```

**Other key type codes** (for reference):
- Sig types: DSA-SHA1=1, ECDSA-P256=6, **Ed25519=7**, Ed25519ph=8, RedDSA=11
- Crypto types: **ElGamal-2048=0**, ECIES-X25519=4

If crypto type were X25519 (code 4), the certificate would be `{0x05, 0x00, 0x04, 0x00, 0x07, 0x00, 0x04}` and the Destination would be 167 bytes (32+96+32+7).

#### Base32 Encoding

- Alphabet: `abcdefghijklmnopqrstuvwxyz234567` (RFC 3548, lowercase)
- 32-byte SHA-256 hash → 256 bits → 52 characters (51 full + 1 trailing bit)
- **No padding characters** (`=`) are used
- Output is always lowercase; decoding is case-insensitive

### Tor v3 Address Derivation

```
Ed25519 pubkey → checksum = SHA3-256(".onion checksum" || pubkey || 0x03)[:2] → 0x03 || pubkey || checksum (35 bytes) → Base32 → .onion address
```

### Output File Formats

**I2P** (679 bytes binary `.b32.i2p.dat`):
```
Destination (391) + PrivateKey (256, zeros for ElGamal) + SigningPrivateKey (32, Ed25519 seed)
```
The 679-byte keyfile layout is: bytes 0–390 = Destination, bytes 391–646 = ElGamal private key (zeros), bytes 647–678 = Ed25519 seed.

**Tor** (directory `<address>.onion/`):
```
hs_ed25519_secret_key: 32-byte header + 32-byte seed (64 bytes total)
hostname: <address>.onion
```

## Key Dependencies

- `cudarc` v0.19 - Safe CUDA bindings with `driver`, `nvrtc`, and `cuda-11040` features
  - Targets CUDA 11.4 for maximum compatibility
  - Uses dynamic-loading so no CUDA libraries needed at build time
- `ed25519-dalek` - CPU-side keypair verification
- `sha2`, `sha3` - CPU-side hash verification
- `clap` - CLI with derive macros using flags (not subcommands)

## Reference Documentation

- I2P Destination serialization: https://gitlab.com/i2p/i2p.i2p/-/blob/master/core/java/src/net/i2p/data/KeysAndCert.java (`writeBytes()`)
- I2P KeyCertificate: https://gitlab.com/i2p/i2p.i2p/-/blob/master/core/java/src/net/i2p/crypto/KeyCertificate.java
- I2P Base32: https://gitlab.com/i2p/i2p.i2p/-/blob/master/core/java/src/net/i2p/util/Base32.java
- Tor v3 spec: https://spec.torproject.org/rend-spec-v3

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- **Minimal comments** - only comment when doing something non-obvious or weird. No doc comments that repeat what the code says.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
