// xorshift64 PRNG
#ifndef PRNG_CUH
#define PRNG_CUH

#include "types.cuh"

__device__ __forceinline__ uint64_t xorshift64(uint64_t state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

__device__ void generate_random_variation(
    const uint8_t* base_random,
    uint64_t variation_id,
    uint8_t* output
) {
    uint64_t state = variation_id;
    for (int chunk = 0; chunk < 8; chunk++) {
        state = xorshift64(state);
        for (int i = 0; i < 32; i++) {
            output[chunk * 32 + i] = base_random[i] ^ ((state >> ((i % 8) * 8)) & 0xFF);
        }
    }
}

#endif
