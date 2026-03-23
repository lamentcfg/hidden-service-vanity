/*
    Portable header to provide the 32 and 64 bits type.

    Modified for CUDA compatibility - avoids conflicts with existing type definitions.
*/

#ifndef FIXEDINT_H_INCLUDED
#define FIXEDINT_H_INCLUDED

// For CUDA, we use the built-in types
#ifdef __CUDACC__
    typedef unsigned int uint32_t;
    typedef int int32_t;
    typedef unsigned long long uint64_t;
    typedef long long int64_t;
#else
    // For host code, try to use stdint.h
    #if ((defined(__STDC__) && __STDC__ && __STDC_VERSION__ >= 199901L) || \
         (defined(__GNUC__) && (defined(_STDINT_H) || defined(_STDINT_H_) || defined(__UINT_FAST64_TYPE__))))
        #include <stdint.h>
    #else
        // Fallback definitions
        typedef unsigned int uint32_t;
        typedef int int32_t;
        typedef unsigned long long uint64_t;
        typedef long long int64_t;
    #endif
#endif

#endif // FIXEDINT_H_INCLUDED
