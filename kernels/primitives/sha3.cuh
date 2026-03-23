/**
 * SHA3-256 (Keccak) Implementation for CUDA
 *
 * Used for Tor v3 onion address checksum calculation.
 * Tor checksum: SHA3-256(".onion checksum" || pubkey || version)[:2]
 */

#ifndef SHA3_CUH
#define SHA3_CUH

#include "types.cuh"

// ============================================================================
// Keccak-f[1600] Round Constants
// ============================================================================

__constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// ============================================================================
// Rotation Function
// ============================================================================

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

// ============================================================================
// Keccak-f[1600] Permutation
// ============================================================================

__device__ void keccak_f1600(uint64_t* state) {
    // State is organized as 5x5 array of 64-bit lanes
    // state[i] corresponds to lane at (x=i%5, y=i/5)

    uint64_t& a00 = state[0];  uint64_t& a10 = state[1];  uint64_t& a20 = state[2];  uint64_t& a30 = state[3];  uint64_t& a40 = state[4];
    uint64_t& a01 = state[5];  uint64_t& a11 = state[6];  uint64_t& a21 = state[7];  uint64_t& a31 = state[8];  uint64_t& a41 = state[9];
    uint64_t& a02 = state[10]; uint64_t& a12 = state[11]; uint64_t& a22 = state[12]; uint64_t& a32 = state[13]; uint64_t& a42 = state[14];
    uint64_t& a03 = state[15]; uint64_t& a13 = state[16]; uint64_t& a23 = state[17]; uint64_t& a33 = state[18]; uint64_t& a43 = state[19];
    uint64_t& a04 = state[20]; uint64_t& a14 = state[21]; uint64_t& a24 = state[22]; uint64_t& a34 = state[23]; uint64_t& a44 = state[24];

    for (int round = 0; round < 24; round++) {
        // Theta step
        uint64_t c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;
        uint64_t c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;
        uint64_t c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;
        uint64_t c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;
        uint64_t c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;

        uint64_t d0 = c4 ^ rotl64(c1, 1);
        uint64_t d1 = c0 ^ rotl64(c2, 1);
        uint64_t d2 = c1 ^ rotl64(c3, 1);
        uint64_t d3 = c2 ^ rotl64(c4, 1);
        uint64_t d4 = c3 ^ rotl64(c0, 1);

        a00 ^= d0; a01 ^= d0; a02 ^= d0; a03 ^= d0; a04 ^= d0;
        a10 ^= d1; a11 ^= d1; a12 ^= d1; a13 ^= d1; a14 ^= d1;
        a20 ^= d2; a21 ^= d2; a22 ^= d2; a23 ^= d2; a24 ^= d2;
        a30 ^= d3; a31 ^= d3; a32 ^= d3; a33 ^= d3; a34 ^= d3;
        a40 ^= d4; a41 ^= d4; a42 ^= d4; a43 ^= d4; a44 ^= d4;

        // Rho and Pi steps (combined)
        // B[x,y] = ROT(A[(x+3y)%5, x], r[(x+3y)%5, x])
        // Using standard Keccak rotation offsets
        uint64_t b00 = a00;                    // ROT(A[0,0], 0)
        uint64_t b10 = rotl64(a11, 44);        // ROT(A[1,1], 44)
        uint64_t b20 = rotl64(a22, 43);        // ROT(A[2,2], 43)
        uint64_t b30 = rotl64(a33, 21);        // ROT(A[3,3], 21)
        uint64_t b40 = rotl64(a44, 14);        // ROT(A[4,4], 14)

        uint64_t b01 = rotl64(a30, 28);        // ROT(A[3,0], 28)
        uint64_t b11 = rotl64(a41, 20);        // ROT(A[4,1], 20)
        uint64_t b21 = rotl64(a02, 3);         // ROT(A[0,2], 3)
        uint64_t b31 = rotl64(a13, 45);        // ROT(A[1,3], 45)
        uint64_t b41 = rotl64(a24, 61);        // ROT(A[2,4], 61)

        uint64_t b02 = rotl64(a10, 1);         // ROT(A[1,0], 1)
        uint64_t b12 = rotl64(a21, 6);         // ROT(A[2,1], 6)
        uint64_t b22 = rotl64(a32, 25);        // ROT(A[3,2], 25)
        uint64_t b32 = rotl64(a43, 8);         // ROT(A[4,3], 8)
        uint64_t b42 = rotl64(a04, 18);        // ROT(A[0,4], 18)

        uint64_t b03 = rotl64(a40, 27);        // ROT(A[4,0], 27)
        uint64_t b13 = rotl64(a01, 36);        // ROT(A[0,1], 36)
        uint64_t b23 = rotl64(a12, 10);        // ROT(A[1,2], 10)
        uint64_t b33 = rotl64(a23, 15);        // ROT(A[2,3], 15)
        uint64_t b43 = rotl64(a34, 56);        // ROT(A[3,4], 56)

        uint64_t b04 = rotl64(a20, 62);        // ROT(A[2,0], 62)
        uint64_t b14 = rotl64(a31, 55);        // ROT(A[3,1], 55)
        uint64_t b24 = rotl64(a42, 39);        // ROT(A[4,2], 39)
        uint64_t b34 = rotl64(a03, 41);        // ROT(A[0,3], 41)
        uint64_t b44 = rotl64(a14, 2);         // ROT(A[1,4], 2)

        // Chi step
        a00 = b00 ^ (~b10 & b20);
        a01 = b01 ^ (~b11 & b21);
        a02 = b02 ^ (~b12 & b22);
        a03 = b03 ^ (~b13 & b23);
        a04 = b04 ^ (~b14 & b24);

        a10 = b10 ^ (~b20 & b30);
        a11 = b11 ^ (~b21 & b31);
        a12 = b12 ^ (~b22 & b32);
        a13 = b13 ^ (~b23 & b33);
        a14 = b14 ^ (~b24 & b34);

        a20 = b20 ^ (~b30 & b40);
        a21 = b21 ^ (~b31 & b41);
        a22 = b22 ^ (~b32 & b42);
        a23 = b23 ^ (~b33 & b43);
        a24 = b24 ^ (~b34 & b44);

        a30 = b30 ^ (~b40 & b00);
        a31 = b31 ^ (~b41 & b01);
        a32 = b32 ^ (~b42 & b02);
        a33 = b33 ^ (~b43 & b03);
        a34 = b34 ^ (~b44 & b04);

        a40 = b40 ^ (~b00 & b10);
        a41 = b41 ^ (~b01 & b11);
        a42 = b42 ^ (~b02 & b12);
        a43 = b43 ^ (~b03 & b13);
        a44 = b44 ^ (~b04 & b14);

        // Iota step
        a00 ^= KECCAK_RC[round];
    }
}

// ============================================================================
// SHA3-256 Implementation
// ============================================================================

// SHA3-256: rate = 1088 bits = 136 bytes, capacity = 512 bits
constexpr int SHA3_256_RATE = 136;
constexpr int SHA3_256_HASH_SIZE = 32;

/**
 * Compute SHA3-256 hash.
 *
 * @param data Input data to hash
 * @param len Length of input data
 * @param hash Output buffer (must be at least 32 bytes)
 */
__device__ void sha3_256(const uint8_t* data, size_t len, uint8_t* hash) {
    uint64_t state[25] = {0};

    // Absorb phase
    size_t absorbed = 0;
    while (absorbed < len) {
        size_t block_len = min((size_t)SHA3_256_RATE, len - absorbed);

        // XOR data into state (byte by byte into 64-bit lanes, little-endian)
        for (size_t i = 0; i < block_len; i++) {
            size_t lane_idx = i / 8;
            size_t byte_offset = i % 8;
            state[lane_idx] ^= ((uint64_t)data[absorbed + i]) << (byte_offset * 8);
        }

        absorbed += block_len;

        // Apply permutation if block is full, or this is the last block
        if (block_len == SHA3_256_RATE) {
            keccak_f1600(state);
        }
    }

    // Padding: append 0x06, then 0x80 at the end of the rate
    // The padding starts at position (len % rate)
    size_t pad_pos = len % SHA3_256_RATE;
    state[pad_pos / 8] ^= 0x06ULL << ((pad_pos % 8) * 8);
    state[SHA3_256_RATE / 8 - 1] ^= 0x8000000000000000ULL;

    // Final permutation
    keccak_f1600(state);

    // Squeeze phase (we only need the first 256 bits = 32 bytes)
    // Extract from state in little-endian order
    for (int i = 0; i < 4; i++) {
        uint64_t lane = state[i];
        hash[i * 8] = lane & 0xFF;
        hash[i * 8 + 1] = (lane >> 8) & 0xFF;
        hash[i * 8 + 2] = (lane >> 16) & 0xFF;
        hash[i * 8 + 3] = (lane >> 24) & 0xFF;
        hash[i * 8 + 4] = (lane >> 32) & 0xFF;
        hash[i * 8 + 5] = (lane >> 40) & 0xFF;
        hash[i * 8 + 6] = (lane >> 48) & 0xFF;
        hash[i * 8 + 7] = (lane >> 56) & 0xFF;
    }
}

// ============================================================================
// Tor v3 Checksum Helper
// ============================================================================

// Tor v3 onion address checksum constant
constexpr int TOR_CHECKSUM_STRING_LEN = 15;  // ".onion checksum"
constexpr int TOR_PUBKEY_SIZE = 32;

/**
 * Compute Tor v3 onion address checksum.
 *
 * Tor checksum = SHA3-256(".onion checksum" || pubkey || version)[:2]
 *
 * Note: The final address format is: base32(pubkey || version || checksum)
 * The version byte (0x03) comes AFTER the pubkey in the address data.
 *
 * @param pubkey 32-byte Ed25519 public key
 * @param checksum Output 2-byte checksum
 */
__device__ void tor_checksum(const uint8_t* pubkey, uint8_t* checksum) {
    // Build input: ".onion checksum" (15 bytes) || pubkey (32 bytes) || version (1 byte)
    uint8_t input[48];

    // ".onion checksum"
    input[0] = '.';
    input[1] = 'o';
    input[2] = 'n';
    input[3] = 'i';
    input[4] = 'o';
    input[5] = 'n';
    input[6] = ' ';
    input[7] = 'c';
    input[8] = 'h';
    input[9] = 'e';
    input[10] = 'c';
    input[11] = 'k';
    input[12] = 's';
    input[13] = 'u';
    input[14] = 'm';

    // Public key
    for (int i = 0; i < 32; i++) {
        input[15 + i] = pubkey[i];
    }

    // Version byte
    input[47] = TOR_VERSION;

    // Compute SHA3-256 and take first 2 bytes
    uint8_t hash[32];
    sha3_256(input, 48, hash);

    checksum[0] = hash[0];
    checksum[1] = hash[1];
}

#endif // SHA3_CUH
