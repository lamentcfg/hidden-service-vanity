// Base32 encoding for I2P (52 chars) and Tor (56 chars)
#ifndef BASE32_CUH
#define BASE32_CUH

#include "types.cuh"

__constant__ char BASE32_TABLE[33] = "abcdefghijklmnopqrstuvwxyz234567";

constexpr int BASE32_I2P_ADDRESS_LENGTH = 52;
constexpr int BASE32_TOR_ADDRESS_LENGTH = 56;

__device__ void base32_encode_generic(
    const uint8_t* input,
    int input_len,
    char* output,
    int output_len
) {
    uint64_t buffer = 0;
    int bits = 0;
    int out_idx = 0;

    for (int i = 0; i < input_len && out_idx < output_len; i++) {
        buffer = (buffer << 8) | input[i];
        bits += 8;

        while (bits >= 5 && out_idx < output_len) {
            bits -= 5;
            output[out_idx++] = BASE32_TABLE[(buffer >> bits) & 0x1F];
        }
    }

    if (bits > 0 && out_idx < output_len) {
        output[out_idx++] = BASE32_TABLE[(buffer << (5 - bits)) & 0x1F];
    }
}

__device__ void base32_encode(const uint8_t* input, char* output) {
    base32_encode_generic(input, 32, output, BASE32_I2P_ADDRESS_LENGTH);
}

__device__ void base32_encode_tor(const uint8_t* input, char* output) {
    base32_encode_generic(input, 35, output, BASE32_TOR_ADDRESS_LENGTH);
}

#endif
