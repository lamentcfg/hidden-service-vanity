/**
 * Ed25519 Implementation using ref10 (from solanity/ChorusOne)
 *
 * This provides optimized Ed25519 scalar multiplication using precomputed tables.
 * Uses 10x32-bit limb representation (standard ref10).
 *
 * Original source: https://github.com/ChorusOne/solanity
 * License: Apache 2.0
 */

#ifndef ED25519_REF10_CUH
#define ED25519_REF10_CUH

// Include common utilities first
#include "primitives/ed25519/common_ref10.cu"

// Include type definitions
#include "primitives/ed25519/fixedint.h"

// Include field element header (renamed to avoid conflicts)
#ifndef FE_H
#define FE_H

typedef int32_t fe[10];

void __host__ __device__ fe_0(fe h);
void __device__ __host__ fe_1(fe h);
void __device__ __host__ fe_frombytes(fe h, const unsigned char *s);
void __device__ __host__ fe_tobytes(unsigned char *s, const fe h);
void __host__ __device__ fe_copy(fe h, const fe f);
int __host__ __device__ fe_isnegative(const fe f);
int __device__ __host__ fe_isnonzero(const fe f);
void __host__ __device__ fe_cmov(fe f, const fe g, unsigned int b);
void __device__ __host__ fe_neg(fe h, const fe f);
void __device__ __host__ fe_add(fe h, const fe f, const fe g);
void __device__ __host__ fe_invert(fe out, const fe z);
void __device__ __host__ fe_sq(fe h, const fe f);
void __host__ __device__ fe_sq2(fe h, const fe f);
void __device__ __host__ fe_mul(fe h, const fe f, const fe g);
void __device__ __host__ fe_pow22523(fe out, const fe z);
void __device__ __host__ fe_sub(fe h, const fe f, const fe g);

#endif // FE_H

// Include group element header
#ifndef GE_H
#define GE_H

typedef struct {
  fe X;
  fe Y;
  fe Z;
} ge_p2;

typedef struct {
  fe X;
  fe Y;
  fe Z;
  fe T;
} ge_p3;

typedef struct {
  fe X;
  fe Y;
  fe Z;
  fe T;
} ge_p1p1;

typedef struct {
  fe yplusx;
  fe yminusx;
  fe xy2d;
} ge_precomp;

typedef struct {
  fe YplusX;
  fe YminusX;
  fe Z;
  fe T2d;
} ge_cached;

void __host__ __device__ ge_p3_tobytes(unsigned char *s, const ge_p3 *h);
void __host__ __device__ ge_tobytes(unsigned char *s, const ge_p2 *h);
int  __host__ __device__ ge_frombytes_negate_vartime(ge_p3 *h, const unsigned char *s);
void __host__ __device__ ge_add(ge_p1p1 *r, const ge_p3 *p, const ge_cached *q);
void __host__ __device__ ge_sub(ge_p1p1 *r, const ge_p3 *p, const ge_cached *q);
void __host__ __device__ ge_madd(ge_p1p1 *r, const ge_p3 *p, const ge_precomp *q);
void __host__ __device__ ge_msub(ge_p1p1 *r, const ge_p3 *p, const ge_precomp *q);
void __host__ __device__ ge_scalarmult_base(ge_p3 *h, const unsigned char *a);
void __host__ __device__ ge_scalarmult_base_with_table(ge_p3 *h, const unsigned char *a, const ge_precomp *base_table);
void __host__ __device__ ge_p1p1_to_p2(ge_p2 *r, const ge_p1p1 *p);
void __host__ __device__ ge_p1p1_to_p3(ge_p3 *r, const ge_p1p1 *p);
void __host__ __device__ ge_p2_0(ge_p2 *h);
void __host__ __device__ ge_p2_dbl(ge_p1p1 *r, const ge_p2 *p);
void __host__ __device__ ge_p3_0(ge_p3 *h);
void __host__ __device__ ge_p3_dbl(ge_p1p1 *r, const ge_p3 *p);
void __host__ __device__ ge_p3_to_cached(ge_cached *r, const ge_p3 *p);
void __host__ __device__ ge_p3_to_p2(ge_p2 *r, const ge_p3 *p);

#endif // GE_H

// Include precomputed tables
#include "primitives/ed25519/precomp_data.h"

// Include implementations
#include "primitives/ed25519/fe_ref10.cu"
#include "primitives/ed25519/ge_ref10.cu"

// ============================================================================
// High-level API for I2P Vanity
// ============================================================================

/**
 * Derive Ed25519 public key from seed using optimized scalar multiplication.
 *
 * @param seed 32-byte Ed25519 seed
 * @param public_key_out 32-byte output public key
 */
__device__ void ed25519_publickey_ref10(const unsigned char *seed, unsigned char *public_key_out) {
    ge_p3 A;
    ge_scalarmult_base(&A, seed);
    ge_p3_tobytes(public_key_out, &A);
}

#endif // ED25519_REF10_CUH
