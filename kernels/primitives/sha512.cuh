/**
 * SHA-512 Implementation for CUDA
 *
 * Used for Ed25519 scalar derivation:
 * SHA-512(seed[32]) -> hash[64]
 * scalar = hash[0..32] with clamping
 */

#ifndef SHA512_CUH
#define SHA512_CUH

// CUDA-compatible types (size_t provided by NVRTC)
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// SHA-512 constants (80 rounds)
__constant__ const uint64_t SHA512_K[80] = {
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817
};

// SHA-512 initial hash values
__constant__ const uint64_t SHA512_H[8] = {
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
};

// ============================================================================
// SHA-512 Utility Functions
// ============================================================================

__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

__device__ __forceinline__ uint64_t sha512_ch(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint64_t sha512_maj(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint64_t sigma0_512(uint64_t x) {
    return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39);
}

__device__ __forceinline__ uint64_t sigma1_512(uint64_t x) {
    return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41);
}

__device__ __forceinline__ uint64_t gamma0_512(uint64_t x) {
    return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7);
}

__device__ __forceinline__ uint64_t gamma1_512(uint64_t x) {
    return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6);
}

// ============================================================================
// SHA-512 Core Implementation
// ============================================================================

/**
 * SHA-512 hash function
 * @param data Input data
 * @param len Length of input data in bytes
 * @param hash Output buffer (64 bytes)
 */
__device__ void sha512(const uint8_t* data, size_t len, uint8_t* hash) {
    uint64_t h[8];
    for (int i = 0; i < 8; i++) {
        h[i] = SHA512_H[i];
    }

    // Calculate padded length (128-byte blocks)
    size_t padded_len = ((len + 17 + 127) / 128) * 128;
    int num_blocks = padded_len / 128;

    uint64_t w[80];

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Prepare message schedule (16 64-bit words from input)
        for (int i = 0; i < 16; i++) {
            int byte_idx = block_idx * 128 + i * 8;
            if (byte_idx + 7 < (int)len) {
                w[i] = ((uint64_t)data[byte_idx] << 56) |
                       ((uint64_t)data[byte_idx + 1] << 48) |
                       ((uint64_t)data[byte_idx + 2] << 40) |
                       ((uint64_t)data[byte_idx + 3] << 32) |
                       ((uint64_t)data[byte_idx + 4] << 24) |
                       ((uint64_t)data[byte_idx + 5] << 16) |
                       ((uint64_t)data[byte_idx + 6] << 8) |
                       ((uint64_t)data[byte_idx + 7]);
            } else if (byte_idx <= (int)len) {
                // Partial word with padding
                uint64_t word = 0;
                for (int j = 0; j < 8; j++) {
                    int idx = byte_idx + j;
                    if (idx < (int)len) {
                        word |= ((uint64_t)data[idx]) << (56 - j * 8);
                    } else if (idx == (int)len) {
                        word |= ((uint64_t)0x80) << (56 - j * 8);
                    }
                }
                w[i] = word;
            } else if (block_idx == num_blocks - 1 && i >= 14) {
                // Length in bits (128-bit value, stored as two 64-bit words)
                uint64_t bit_len = (uint64_t)len * 8;
                w[i] = (i == 14) ? 0 : bit_len;  // High bits are 0 for our use case
            } else {
                w[i] = 0;
            }
        }

        // Extend message schedule to 80 words
        for (int i = 16; i < 80; i++) {
            w[i] = gamma1_512(w[i - 2]) + w[i - 7] + gamma0_512(w[i - 15]) + w[i - 16];
        }

        // Compression
        uint64_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint64_t e = h[4], f = h[5], g = h[6], hh = h[7];

        for (int i = 0; i < 80; i++) {
            uint64_t t1 = hh + sigma1_512(e) + sha512_ch(e, f, g) + SHA512_K[i] + w[i];
            uint64_t t2 = sigma0_512(a) + sha512_maj(a, b, c);
            hh = g; g = f; f = e;
            e = d + t1;
            d = c; c = b; b = a;
            a = t1 + t2;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    // Output hash (big-endian)
    for (int i = 0; i < 8; i++) {
        hash[i * 8] = (h[i] >> 56) & 0xFF;
        hash[i * 8 + 1] = (h[i] >> 48) & 0xFF;
        hash[i * 8 + 2] = (h[i] >> 40) & 0xFF;
        hash[i * 8 + 3] = (h[i] >> 32) & 0xFF;
        hash[i * 8 + 4] = (h[i] >> 24) & 0xFF;
        hash[i * 8 + 5] = (h[i] >> 16) & 0xFF;
        hash[i * 8 + 6] = (h[i] >> 8) & 0xFF;
        hash[i * 8 + 7] = h[i] & 0xFF;
    }
}

/**
 * Derive Ed25519 scalar from seed
 * SHA-512(seed) -> hash, then clamp first 32 bytes
 *
 * Clamping:
 *   scalar[0] &= 248   (clear bottom 3 bits)
 *   scalar[31] &= 127  (clear top bit)
 *   scalar[31] |= 64   (set second-to-top bit)
 *
 * @param seed 32-byte Ed25519 seed
 * @param scalar_out 32-byte output scalar (clamped)
 */
__device__ void ed25519_derive_scalar(const uint8_t* seed, uint8_t* scalar_out) {
    uint8_t hash[64];
    sha512(seed, 32, hash);

    // Copy first 32 bytes
    for (int i = 0; i < 32; i++) {
        scalar_out[i] = hash[i];
    }

    // Clamp the scalar
    scalar_out[0] &= 248;
    scalar_out[31] &= 127;
    scalar_out[31] |= 64;
}

#endif // SHA512_CUH
