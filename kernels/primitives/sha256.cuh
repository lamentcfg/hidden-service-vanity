// SHA-256 for CUDA
#ifndef SHA256_CUH
#define SHA256_CUH

#include "types.cuh"

__constant__ uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__constant__ uint32_t SHA256_H[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t sha256_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t sha256_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

__device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

// --- Streaming SHA-256 API (avoids large stack allocations) ---

typedef struct {
    uint32_t h[8];
    uint8_t buffer[64];
    uint64_t total_len;
    uint32_t buf_len;
} sha256_ctx;

__device__ __forceinline__ void sha256_process_block(uint32_t* h, const uint8_t* block) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i*4] << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) |
               ((uint32_t)block[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = hh + sigma1(e) + sha256_ch(e, f, g) + SHA256_K[i] + w[i];
        uint32_t t2 = sigma0(a) + sha256_maj(a, b, c);
        hh = g; g = f; f = e;
        e = d + t1;
        d = c; c = b; b = a;
        a = t1 + t2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

__device__ __forceinline__ void sha256_init(sha256_ctx* ctx) {
    for (int i = 0; i < 8; i++) ctx->h[i] = SHA256_H[i];
    ctx->total_len = 0;
    ctx->buf_len = 0;
}

__device__ void sha256_update(sha256_ctx* ctx, const uint8_t* data, size_t len) {
    ctx->total_len += len;

    if (ctx->buf_len > 0) {
        uint32_t space = 64 - ctx->buf_len;
        uint32_t to_copy = (len < space) ? (uint32_t)len : space;
        for (uint32_t i = 0; i < to_copy; i++) {
            ctx->buffer[ctx->buf_len + i] = data[i];
        }
        ctx->buf_len += to_copy;
        data += to_copy;
        len -= to_copy;
        if (ctx->buf_len == 64) {
            sha256_process_block(ctx->h, ctx->buffer);
            ctx->buf_len = 0;
        }
    }

    while (len >= 64) {
        sha256_process_block(ctx->h, data);
        data += 64;
        len -= 64;
    }

    if (len > 0) {
        for (size_t i = 0; i < len; i++) {
            ctx->buffer[i] = data[i];
        }
        ctx->buf_len = (uint32_t)len;
    }
}

__device__ void sha256_final(sha256_ctx* ctx, uint8_t* hash) {
    uint64_t bit_len = ctx->total_len * 8;

    ctx->buffer[ctx->buf_len++] = 0x80;

    if (ctx->buf_len > 56) {
        while (ctx->buf_len < 64) {
            ctx->buffer[ctx->buf_len++] = 0;
        }
        sha256_process_block(ctx->h, ctx->buffer);
        ctx->buf_len = 0;
    }

    while (ctx->buf_len < 56) {
        ctx->buffer[ctx->buf_len++] = 0;
    }

    ctx->buffer[56] = (bit_len >> 56) & 0xFF;
    ctx->buffer[57] = (bit_len >> 48) & 0xFF;
    ctx->buffer[58] = (bit_len >> 40) & 0xFF;
    ctx->buffer[59] = (bit_len >> 32) & 0xFF;
    ctx->buffer[60] = (bit_len >> 24) & 0xFF;
    ctx->buffer[61] = (bit_len >> 16) & 0xFF;
    ctx->buffer[62] = (bit_len >> 8) & 0xFF;
    ctx->buffer[63] = bit_len & 0xFF;

    sha256_process_block(ctx->h, ctx->buffer);

    for (int i = 0; i < 8; i++) {
        hash[i * 4] = (ctx->h[i] >> 24) & 0xFF;
        hash[i * 4 + 1] = (ctx->h[i] >> 16) & 0xFF;
        hash[i * 4 + 2] = (ctx->h[i] >> 8) & 0xFF;
        hash[i * 4 + 3] = ctx->h[i] & 0xFF;
    }
}

// Convenience wrapper (for standalone test kernel)
__device__ void sha256(const uint8_t* data, size_t len, uint8_t* hash) {
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, hash);
}

#endif
