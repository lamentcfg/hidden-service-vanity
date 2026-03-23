// Combined I2P + Tor Vanity Address Generator
// Ed25519 keypair generation (~95% compute) is shared, then each network has its own address derivation.

#include "primitives/sha256.cuh"
#include "primitives/sha512.cuh"
#include "primitives/sha3.cuh"
#include "primitives/base32.cuh"
#include "primitives/prng.cuh"
#include "primitives/ed25519/ed25519_ref10.cuh"

// Ed25519 sizes
constexpr int ED25519_SEED_SIZE = 32;
constexpr int ED25519_PUBLIC_KEY_SIZE = 32;

// I2P Destination sizes
constexpr int I2P_PUBKEY_SIZE = 256;
constexpr int I2P_SIGNING_PUBKEY_SLOT = 128;
constexpr int I2P_CERTIFICATE_SIZE = 7;
constexpr int I2P_DESTINATION_SIZE = I2P_PUBKEY_SIZE + I2P_SIGNING_PUBKEY_SLOT + I2P_CERTIFICATE_SIZE;

// Tor v3 sizes
constexpr int TOR_ADDRESS_DATA_SIZE = 35;

// I2P Destination: 256 random + 32 pubkey + 96 padding + 7 certificate = 391 bytes
__device__ void construct_i2p_destination(
    const uint8_t* random_data,
    const uint8_t* ed25519_pubkey,
    uint8_t* destination
) {
    for (int i = 0; i < I2P_PUBKEY_SIZE; i++) {
        destination[i] = random_data[i];
    }
    for (int i = 0; i < ED25519_PUBLIC_KEY_SIZE; i++) {
        destination[I2P_PUBKEY_SIZE + i] = ed25519_pubkey[i];
    }
    for (int i = 0; i < I2P_SIGNING_PUBKEY_SLOT - ED25519_PUBLIC_KEY_SIZE; i++) {
        destination[I2P_PUBKEY_SIZE + ED25519_PUBLIC_KEY_SIZE + i] = 0;
    }
    // Key Certificate for Ed25519
    destination[I2P_PUBKEY_SIZE + I2P_SIGNING_PUBKEY_SLOT] = 0x05;
    destination[I2P_PUBKEY_SIZE + I2P_SIGNING_PUBKEY_SLOT + 1] = 0x00;
    destination[I2P_PUBKEY_SIZE + I2P_SIGNING_PUBKEY_SLOT + 2] = 0x04;
    destination[I2P_PUBKEY_SIZE + I2P_SIGNING_PUBKEY_SLOT + 3] = 0x00;
    destination[I2P_PUBKEY_SIZE + I2P_SIGNING_PUBKEY_SLOT + 4] = 0x07;
    destination[I2P_PUBKEY_SIZE + I2P_SIGNING_PUBKEY_SLOT + 5] = 0x00;
    destination[I2P_PUBKEY_SIZE + I2P_SIGNING_PUBKEY_SLOT + 6] = 0x00;
}

__device__ void construct_tor_address_data(
    const uint8_t* pubkey,
    const uint8_t* checksum,
    uint8_t* output
) {
    for (int i = 0; i < ED25519_PUBLIC_KEY_SIZE; i++) {
        output[i] = pubkey[i];
    }
    output[32] = TOR_VERSION;
    output[33] = checksum[0];
    output[34] = checksum[1];
}

__device__ bool matches_prefix(const char* address, const char* prefix, int prefix_len) {
    for (int i = 0; i < prefix_len; i++) {
        char addr_char = address[i];
        char pref_char = prefix[i];
        if (addr_char >= 'A' && addr_char <= 'Z') addr_char += 32;
        if (pref_char >= 'A' && pref_char <= 'Z') pref_char += 32;
        if (addr_char != pref_char) return false;
    }
    return true;
}

__device__ bool matches_any_prefix(
    const char* address,
    const char* prefix_buffer,
    const int* prefix_lengths,
    int prefix_count
) {
    int offset = 0;
    for (int i = 0; i < prefix_count; i++) {
        int len = prefix_lengths[i];
        if (matches_prefix(address, prefix_buffer + offset, len)) {
            return true;
        }
        offset += len;
    }
    return false;
}

__device__ uint64_t generate_seed(
    const uint8_t* base_seed,
    int tid,
    uint64_t iteration,
    uint8_t* seed
) {
    uint64_t state = ((uint64_t)tid << 32) ^ iteration;

    for (int i = 0; i < 4; i++) {
        state = xorshift64(state);
        uint64_t base_word = ((uint64_t)base_seed[i*8] |
                              ((uint64_t)base_seed[i*8+1] << 8) |
                              ((uint64_t)base_seed[i*8+2] << 16) |
                              ((uint64_t)base_seed[i*8+3] << 24) |
                              ((uint64_t)base_seed[i*8+4] << 32) |
                              ((uint64_t)base_seed[i*8+5] << 40) |
                              ((uint64_t)base_seed[i*8+6] << 48) |
                              ((uint64_t)base_seed[i*8+7] << 56));
        state ^= base_word;

        seed[i*8 + 0] = state & 0xFF;
        seed[i*8 + 1] = (state >> 8) & 0xFF;
        seed[i*8 + 2] = (state >> 16) & 0xFF;
        seed[i*8 + 3] = (state >> 24) & 0xFF;
        seed[i*8 + 4] = (state >> 32) & 0xFF;
        seed[i*8 + 5] = (state >> 40) & 0xFF;
        seed[i*8 + 6] = (state >> 48) & 0xFF;
        seed[i*8 + 7] = (state >> 56) & 0xFF;
    }

    return state;
}

extern "C" __global__ void search_combined_kernel(
    const uint8_t* __restrict__ base_seed,
    const ge_precomp* __restrict__ base_table,
    uint64_t iteration,

    const char* __restrict__ i2p_prefix_buffer,
    const int* __restrict__ i2p_prefix_lengths,
    int i2p_prefix_count,
    bool search_i2p,
    uint8_t* __restrict__ i2p_out_seeds,
    uint8_t* __restrict__ i2p_out_pubkeys,
    int* __restrict__ i2p_match_count,
    int i2p_max_matches,

    const char* __restrict__ tor_prefix_buffer,
    const int* __restrict__ tor_prefix_lengths,
    int tor_prefix_count,
    bool search_tor,
    uint8_t* __restrict__ tor_out_seeds,
    uint8_t* __restrict__ tor_out_pubkeys,
    int* __restrict__ tor_match_count,
    int tor_max_matches
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (!search_i2p && !search_tor) return;

    // Ed25519 Keypair Generation
    uint8_t seed[32];
    uint64_t state = generate_seed(base_seed, tid, iteration, seed);

    uint8_t scalar[32];
    ed25519_derive_scalar(seed, scalar);

    ge_p3 pubkey_point;
    ge_scalarmult_base_with_table(&pubkey_point, scalar, base_table);

    uint8_t pubkey[32];
    ge_p3_tobytes(pubkey, &pubkey_point);

    // I2P PATH
    if (search_i2p && i2p_prefix_count > 0) {
        uint8_t random_data[I2P_PUBKEY_SIZE];
        uint64_t rand_state = ((uint64_t)seed[0] | ((uint64_t)seed[1] << 8) | ((uint64_t)seed[2] << 16) | ((uint64_t)seed[3] << 24) |
                              ((uint64_t)seed[4] << 32) | ((uint64_t)seed[5] << 40) | ((uint64_t)seed[6] << 48) | ((uint64_t)seed[7] << 56))
                            ^ ((uint64_t)seed[8] | ((uint64_t)seed[9] << 8) | ((uint64_t)seed[10] << 16) | ((uint64_t)seed[11] << 24) |
                              ((uint64_t)seed[12] << 32) | ((uint64_t)seed[13] << 40) | ((uint64_t)seed[14] << 48) | ((uint64_t)seed[15] << 56));
        for (int chunk = 0; chunk < 8; chunk++) {
            for (int i = 0; i < 4; i++) {
                rand_state = xorshift64(rand_state);
                int idx = chunk * 32 + i * 8;
                random_data[idx + 0] = rand_state & 0xFF;
                random_data[idx + 1] = (rand_state >> 8) & 0xFF;
                random_data[idx + 2] = (rand_state >> 16) & 0xFF;
                random_data[idx + 3] = (rand_state >> 24) & 0xFF;
                random_data[idx + 4] = (rand_state >> 32) & 0xFF;
                random_data[idx + 5] = (rand_state >> 40) & 0xFF;
                random_data[idx + 6] = (rand_state >> 48) & 0xFF;
                random_data[idx + 7] = (rand_state >> 56) & 0xFF;
            }
        }

        uint8_t destination[I2P_DESTINATION_SIZE];
        construct_i2p_destination(random_data, pubkey, destination);

        uint8_t hash[32];
        sha256(destination, I2P_DESTINATION_SIZE, hash);

        char i2p_address[BASE32_I2P_ADDRESS_LENGTH + 1];
        base32_encode(hash, i2p_address);
        i2p_address[BASE32_I2P_ADDRESS_LENGTH] = '\0';

        if (matches_any_prefix(i2p_address, i2p_prefix_buffer, i2p_prefix_lengths, i2p_prefix_count)) {
            int idx = atomicAdd(i2p_match_count, 1);
            if (idx < i2p_max_matches) {
                for (int i = 0; i < ED25519_SEED_SIZE; i++) {
                    i2p_out_seeds[idx * ED25519_SEED_SIZE + i] = seed[i];
                }
                for (int i = 0; i < ED25519_PUBLIC_KEY_SIZE; i++) {
                    i2p_out_pubkeys[idx * ED25519_PUBLIC_KEY_SIZE + i] = pubkey[i];
                }
            }
        }
    }

    // TOR PATH
    if (search_tor && tor_prefix_count > 0) {
        uint8_t checksum[2];
        tor_checksum(pubkey, checksum);

        uint8_t address_data[TOR_ADDRESS_DATA_SIZE];
        construct_tor_address_data(pubkey, checksum, address_data);

        char tor_address[BASE32_TOR_ADDRESS_LENGTH + 1];
        base32_encode_tor(address_data, tor_address);
        tor_address[BASE32_TOR_ADDRESS_LENGTH] = '\0';

        if (matches_any_prefix(tor_address, tor_prefix_buffer, tor_prefix_lengths, tor_prefix_count)) {
            int idx = atomicAdd(tor_match_count, 1);
            if (idx < tor_max_matches) {
                for (int i = 0; i < ED25519_SEED_SIZE; i++) {
                    tor_out_seeds[idx * ED25519_SEED_SIZE + i] = seed[i];
                }
                for (int i = 0; i < ED25519_PUBLIC_KEY_SIZE; i++) {
                    tor_out_pubkeys[idx * ED25519_PUBLIC_KEY_SIZE + i] = pubkey[i];
                }
            }
        }
    }
}

extern "C" __global__ void debug_combined_kernel(
    const uint8_t* __restrict__ base_seed,
    const ge_precomp* __restrict__ base_table,
    uint64_t iteration,

    char* __restrict__ debug_i2p_addresses,
    uint8_t* __restrict__ debug_i2p_pubkeys,
    char* __restrict__ debug_tor_addresses,
    uint8_t* __restrict__ debug_tor_pubkeys,
    int* __restrict__ debug_count,
    int max_debug
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= max_debug) return;

    uint8_t seed[32];
    uint64_t state = generate_seed(base_seed, tid, iteration, seed);

    uint8_t scalar[32];
    ed25519_derive_scalar(seed, scalar);

    ge_p3 pubkey_point;
    ge_scalarmult_base_with_table(&pubkey_point, scalar, base_table);

    uint8_t pubkey[32];
    ge_p3_tobytes(pubkey, &pubkey_point);

    for (int i = 0; i < 32; i++) {
        debug_i2p_pubkeys[tid * 32 + i] = pubkey[i];
        debug_tor_pubkeys[tid * 32 + i] = pubkey[i];
    }

    // I2P Address
    uint8_t random_data[I2P_PUBKEY_SIZE];
    uint64_t rand_state = ((uint64_t)seed[0] | ((uint64_t)seed[1] << 8) | ((uint64_t)seed[2] << 16) | ((uint64_t)seed[3] << 24) |
                          ((uint64_t)seed[4] << 32) | ((uint64_t)seed[5] << 40) | ((uint64_t)seed[6] << 48) | ((uint64_t)seed[7] << 56))
                        ^ ((uint64_t)seed[8] | ((uint64_t)seed[9] << 8) | ((uint64_t)seed[10] << 16) | ((uint64_t)seed[11] << 24) |
                          ((uint64_t)seed[12] << 32) | ((uint64_t)seed[13] << 40) | ((uint64_t)seed[14] << 48) | ((uint64_t)seed[15] << 56));
    for (int chunk = 0; chunk < 8; chunk++) {
        for (int i = 0; i < 4; i++) {
            rand_state = xorshift64(rand_state);
            int idx = chunk * 32 + i * 8;
            random_data[idx + 0] = rand_state & 0xFF;
            random_data[idx + 1] = (rand_state >> 8) & 0xFF;
            random_data[idx + 2] = (rand_state >> 16) & 0xFF;
            random_data[idx + 3] = (rand_state >> 24) & 0xFF;
            random_data[idx + 4] = (rand_state >> 32) & 0xFF;
            random_data[idx + 5] = (rand_state >> 40) & 0xFF;
            random_data[idx + 6] = (rand_state >> 48) & 0xFF;
            random_data[idx + 7] = (rand_state >> 56) & 0xFF;
        }
    }

    uint8_t destination[I2P_DESTINATION_SIZE];
    construct_i2p_destination(random_data, pubkey, destination);

    uint8_t hash[32];
    sha256(destination, I2P_DESTINATION_SIZE, hash);

    char i2p_address[BASE32_I2P_ADDRESS_LENGTH + 1];
    base32_encode(hash, i2p_address);

    for (int i = 0; i < BASE32_I2P_ADDRESS_LENGTH; i++) {
        debug_i2p_addresses[tid * BASE32_I2P_ADDRESS_LENGTH + i] = i2p_address[i];
    }

    // Tor Address
    uint8_t checksum[2];
    tor_checksum(pubkey, checksum);

    uint8_t address_data[TOR_ADDRESS_DATA_SIZE];
    construct_tor_address_data(pubkey, checksum, address_data);

    char tor_address[BASE32_TOR_ADDRESS_LENGTH + 1];
    base32_encode_tor(address_data, tor_address);

    for (int i = 0; i < BASE32_TOR_ADDRESS_LENGTH; i++) {
        debug_tor_addresses[tid * BASE32_TOR_ADDRESS_LENGTH + i] = tor_address[i];
    }

    if (tid == 0) {
        *debug_count = max_debug;
    }
}
