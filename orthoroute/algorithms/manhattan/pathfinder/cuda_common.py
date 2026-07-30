"""
Shared CUDA device-function preamble.

Every routing kernel that needs float atomic-min or the packed
(distance, parent) atomic key prepends DEVICE_PRELUDE to its source string
instead of carrying its own copy of these functions.  This is pure string
concatenation — the function bodies are the canonical copies that previously
lived duplicated inside cuda_dijkstra.py and persistent_kernel.py.

Pure Python string module: importable with or without CuPy installed.
"""

DEVICE_PRELUDE = r'''
// ── Shared device preamble (see cuda_common.py) ──────────────────────────

// Custom atomic min for float using compare-and-swap
__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        if (old_val <= value) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ unsigned int f2u(float x) {
    return __float_as_uint(x);
}

// Pack distance and parent into 64-bit key for atomic operations
__device__ __forceinline__ unsigned long long pack_key(float g, int p) {
    return ((unsigned long long)f2u(g) << 32) | (unsigned long long)(unsigned int)p;
}

// Atomically replace a packed (distance, parent) key only when distance
// strictly improves.  Comparing the whole key lets an equal float32 distance
// replace its parent based on node ID; after large penalties, sub-ULP edges
// then create flat-distance parent cycles.
__device__ __forceinline__ unsigned long long atomicMinDistanceKey(unsigned long long* address, unsigned long long val) {
    unsigned long long old = *address;
    unsigned long long assumed;
    do {
        assumed = old;
        const unsigned int old_dist = (unsigned int)(assumed >> 32);
        const unsigned int new_dist = (unsigned int)(val >> 32);
        if (new_dist >= old_dist) break;
        old = atomicCAS(address, assumed, val);
    } while (assumed != old);
    return old;
}

// ── End shared device preamble ───────────────────────────────────────────
'''
