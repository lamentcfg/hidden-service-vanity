/**
 * Common utilities for ref10 implementation
 */

#ifndef COMMON_REF10_CU
#define COMMON_REF10_CU

// Load 3 bytes into a 64-bit integer (little-endian)
static __host__ __device__ uint64_t load_3(const unsigned char *in) {
    uint64_t result;
    result = (uint64_t) in[0];
    result |= ((uint64_t) in[1]) << 8;
    result |= ((uint64_t) in[2]) << 16;
    return result;
}

// Load 4 bytes into a 64-bit integer (little-endian)
static __host__ __device__ uint64_t load_4(const unsigned char *in) {
    uint64_t result;
    result = (uint64_t) in[0];
    result |= ((uint64_t) in[1]) << 8;
    result |= ((uint64_t) in[2]) << 16;
    result |= ((uint64_t) in[3]) << 24;
    return result;
}

#endif // COMMON_REF10_CU
