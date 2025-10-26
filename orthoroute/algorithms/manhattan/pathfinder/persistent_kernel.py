"""
AGENT K: Persistent SSSP Kernel Implementation
Single-launch kernel that runs until convergence or destination found.
Eliminates kernel launch overhead by running persistently on device.
"""

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

# PERSISTENT SSSP KERNEL
# This kernel launches ONCE per net and runs until:
# 1. ANY destination in dst_bitmap is reached, OR
# 2. Frontier is empty (no path), OR
# 3. Max iterations reached
PERSISTENT_SSSP_KERNEL_CODE = r'''
#include <cooperative_groups.h>

// CUDA constants
#define CUDA_INFINITY __int_as_float(0x7f800000)

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

extern "C" __global__
void persistent_sssp_kernel(
    const int* indptr,                  // CSR indptr for full graph
    const int* indices,                 // CSR indices
    const float* weights,               // CSR weights
    const int num_nodes,                // Total nodes in graph
    const int* src_seeds,               // Source seed array
    const int num_srcs,                 // Number of sources
    const int* dst_targets,             // Array of destination indices
    const int num_dsts,                 // Number of destinations
    float* dist,                        // Distance array (num_nodes,)
    int* parent,                        // Parent array (num_nodes,)
    unsigned int* frontier_curr,        // Current frontier (bit-packed)
    unsigned int* frontier_next,        // Next frontier (bit-packed)
    const int frontier_words,           // Number of uint32 words for frontier
    int* settled_flag,                  // Flag: 1 when path found
    int* best_dst,                      // Output: best destination found
    float* best_dist,                   // Output: best distance found
    const int max_iterations,           // Maximum iterations
    int* iteration_count,               // Output: actual iterations performed
    int* has_active_global              // Global flag for frontier empty check
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_tid = bid * blockDim.x + tid;
    const int total_threads = blockDim.x * gridDim.x;

    // Get grid-wide cooperative group for synchronization across ALL blocks
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    // Thread 0: Initialize global state
    if (global_tid == 0) {
        *iteration_count = 0;
        *settled_flag = 0;
        *best_dst = -1;
        *best_dist = CUDA_INFINITY;
    }
    grid.sync();

    // Initialize source seeds (distributed across all threads)
    for (int s = global_tid; s < num_srcs; s += total_threads) {
        int seed = src_seeds[s];
        if (seed >= 0 && seed < num_nodes) {
            dist[seed] = 0.0f;
            parent[seed] = -1;

            // Add to frontier
            int word_idx = seed / 32;
            int bit_pos = seed % 32;
            atomicOr(&frontier_curr[word_idx], 1u << bit_pos);
        }
    }
    grid.sync();

    // Main persistent loop - runs until convergence
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        // Early exit if path found
        if (*settled_flag) break;

        // Update iteration counter (thread 0 only)
        if (global_tid == 0) {
            *iteration_count = iteration;
        }

        // Clear next frontier (distributed)
        for (int w = global_tid; w < frontier_words; w += total_threads) {
            frontier_next[w] = 0;
        }
        grid.sync();

        // Process frontier - each block handles different words
        for (int word_idx = bid; word_idx < frontier_words; word_idx += gridDim.x) {
            unsigned int word = frontier_curr[word_idx];
            if (word == 0) continue;

            // Each thread processes different bits
            for (int bit = tid; bit < 32; bit += blockDim.x) {
                if (!((word >> bit) & 1)) continue;

                int node = word_idx * 32 + bit;
                if (node >= num_nodes) continue;

                float node_dist = dist[node];
                if (isinf(node_dist)) continue;

                // Expand neighbors
                int e0 = indptr[node];
                int e1 = indptr[node + 1];

                for (int e = e0; e < e1; e++) {
                    int neighbor = indices[e];
                    if (neighbor < 0 || neighbor >= num_nodes) continue;

                    float edge_cost = weights[e];
                    float g_new = node_dist + edge_cost;

                    // Try to improve distance
                    float old_dist = dist[neighbor];
                    if (g_new >= old_dist) continue;

                    float old = atomicMinFloat(&dist[neighbor], g_new);
                    if (g_new + 1e-8f < old) {
                        // Update parent
                        atomicExch(&parent[neighbor], node);

                        // Add to next frontier
                        int nbr_word = neighbor / 32;
                        int nbr_bit = neighbor % 32;
                        atomicOr(&frontier_next[nbr_word], 1u << nbr_bit);
                    }
                }
            }
        }
        grid.sync();

        // Check if any destination reached
        for (int d = global_tid; d < num_dsts; d += total_threads) {
            int dst = dst_targets[d];
            if (dst < 0 || dst >= num_nodes) continue;

            float dst_dist = dist[dst];
            if (!isinf(dst_dist)) {
                // Found a path! Update best
                atomicMinFloat(best_dist, dst_dist);

                // Check if this is the best distance
                if (fabsf(dst_dist - *best_dist) < 1e-8f) {
                    atomicExch(best_dst, dst);
                    atomicExch(settled_flag, 1);
                }
            }
        }
        grid.sync();

        // Early exit if settled
        if (*settled_flag) break;

        // Check if frontier is empty (use global flag, not shared)
        if (global_tid == 0) {
            *has_active_global = 0;  // Reset global flag
        }
        grid.sync();  // Ensure reset is visible

        // All threads check their portion of frontier_next
        for (int w = global_tid; w < frontier_words; w += total_threads) {
            if (frontier_next[w] != 0) {
                atomicExch(has_active_global, 1);  // Global atomic write
            }
        }
        grid.sync();  // CRITICAL: Grid-wide sync to ensure all blocks finish checking

        if (*has_active_global == 0) {
            // No more work - terminate
            break;
        }

        // Swap frontiers for next iteration
        for (int w = global_tid; w < frontier_words; w += total_threads) {
            frontier_curr[w] = frontier_next[w];
        }
        grid.sync();  // CRITICAL: Grid-wide sync before next iteration
    }
}
'''


def create_persistent_kernel():
    """Create and return the compiled persistent SSSP kernel with cooperative groups enabled."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy not available - cannot create persistent kernel")

    # CRITICAL: Enable cooperative groups for grid.sync() to work
    return cp.RawKernel(
        PERSISTENT_SSSP_KERNEL_CODE,
        'persistent_sssp_kernel',
        enable_cooperative_groups=True  # Enables grid-wide synchronization
    )


def launch_persistent_kernel(
    kernel,
    indptr_gpu,
    indices_gpu,
    weights_gpu,
    num_nodes,
    src_seeds_gpu,
    dst_targets_gpu,
    dist_gpu,
    parent_gpu,
    frontier_words,
    max_iterations=2000
):
    """
    Launch the persistent kernel for single-shot SSSP.

    Args:
        kernel: Compiled RawKernel from create_persistent_kernel()
        indptr_gpu: CSR indptr array (CuPy)
        indices_gpu: CSR indices array (CuPy)
        weights_gpu: CSR weights array (CuPy)
        num_nodes: Total number of nodes
        src_seeds_gpu: Source seed array (CuPy int32)
        dst_targets_gpu: Destination targets array (CuPy int32)
        dist_gpu: Distance array (CuPy float32), pre-initialized to inf
        parent_gpu: Parent array (CuPy int32), pre-initialized to -1
        frontier_words: Number of uint32 words for frontier
        max_iterations: Maximum iterations before timeout

    Returns:
        Tuple of (best_dst, best_dist, iterations)
        best_dst: Index of best destination found (-1 if no path)
        best_dist: Distance to best destination (inf if no path)
        iterations: Number of iterations performed
    """
    import cupy as cp

    # Allocate frontier buffers
    frontier_curr = cp.zeros(frontier_words, dtype=cp.uint32)
    frontier_next = cp.zeros(frontier_words, dtype=cp.uint32)

    # Allocate output buffers
    settled_flag = cp.zeros(1, dtype=cp.int32)
    best_dst = cp.full(1, -1, dtype=cp.int32)
    best_dist = cp.full(1, cp.inf, dtype=cp.float32)
    iteration_count = cp.zeros(1, dtype=cp.int32)
    has_active_global = cp.zeros(1, dtype=cp.int32)  # Global frontier empty flag

    num_srcs = len(src_seeds_gpu)
    num_dsts = len(dst_targets_gpu)

    # Kernel launch config
    # Use many blocks for persistent execution
    threads_per_block = 256
    num_blocks = 80  # 80 SMs on RTX 4090, or adjust based on GPU

    # Launch kernel (SINGLE LAUNCH!)
    kernel(
        (num_blocks,),
        (threads_per_block,),
        (
            indptr_gpu,
            indices_gpu,
            weights_gpu,
            num_nodes,
            src_seeds_gpu,
            num_srcs,
            dst_targets_gpu,
            num_dsts,
            dist_gpu,
            parent_gpu,
            frontier_curr,
            frontier_next,
            frontier_words,
            settled_flag,
            best_dst,
            best_dist,
            max_iterations,
            iteration_count,
            has_active_global  # Added: Global frontier empty flag
        )
    )

    # Wait for completion
    cp.cuda.Stream.null.synchronize()

    # Extract results
    result_dst = int(best_dst[0])
    result_dist = float(best_dist[0])
    result_iters = int(iteration_count[0])

    return result_dst, result_dist, result_iters
