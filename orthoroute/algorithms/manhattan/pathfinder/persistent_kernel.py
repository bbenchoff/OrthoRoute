"""
AGENT K: Persistent SSSP Kernel Implementation
Single-launch kernel that runs until convergence or destination found.
Eliminates kernel launch overhead by running persistently on device.
"""

import logging

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Cooperative launches require every block to be resident on the device at
# once, so the grid size must respect this GPU's capacity rather than assume
# an RTX 4090.  Computed once per (kernel, block size) by
# _cooperative_grid_size() and cached here for the life of the process.
_launch_config_cache = {}

# Conservative fallback when occupancy queries fail: small enough to be
# resident on any CUDA GPU that can run this kernel at all.
_FALLBACK_NUM_BLOCKS = 32
_MAX_NUM_BLOCKS = 512


def _cooperative_grid_size(kernel, threads_per_block):
    """Return the number of blocks for a cooperative launch of ``kernel``.

    A cooperative kernel launch fails outright if the grid exceeds
    ``sm_count * max_active_blocks_per_sm`` for the kernel, so query both at
    runtime instead of hardcoding a grid sized for one specific GPU.  Any
    query failure falls back to a conservative fixed grid with a one-line
    warning; the result is cached per (function pointer, block size).
    """
    try:
        func_ptr = kernel.kernel.ptr
    except AttributeError:
        func_ptr = None

    cache_key = (func_ptr, threads_per_block)
    cached = _launch_config_cache.get(cache_key)
    if cached is not None:
        return cached

    num_blocks = _FALLBACK_NUM_BLOCKS
    try:
        sm_count = int(cp.cuda.Device().attributes['MultiProcessorCount'])
    except Exception as exc:
        logger.warning(
            "Persistent kernel: MultiProcessorCount query failed (%s); "
            "using fallback grid of %d blocks", exc, num_blocks
        )
        sm_count = 0

    if sm_count > 0 and func_ptr is not None:
        try:
            blocks_per_sm = int(
                cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                    func_ptr, threads_per_block, 0
                )
            )
            num_blocks = max(1, min(sm_count * blocks_per_sm, _MAX_NUM_BLOCKS))
        except Exception as exc:
            logger.warning(
                "Persistent kernel: occupancyMaxActiveBlocksPerMultiprocessor "
                "query failed (%s); using fallback grid of %d blocks",
                exc, num_blocks
            )
    elif sm_count > 0:
        logger.warning(
            "Persistent kernel: kernel function pointer unavailable for "
            "occupancy query; using fallback grid of %d blocks", num_blocks
        )

    logger.debug(
        "Persistent kernel launch config: %d blocks x %d threads",
        num_blocks, threads_per_block
    )
    _launch_config_cache[cache_key] = num_blocks
    return num_blocks

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

// Atomically replace a packed (distance, parent) key only when distance
// strictly improves.  Comparing the whole key lets an equal float32 distance
// replace its parent based on node ID; after large penalties, sub-ULP edges
// then create flat-distance parent cycles.
__device__ unsigned long long atomicMinDistanceKey(
    unsigned long long* address,
    unsigned long long val
) {
    unsigned long long old = *address;
    unsigned long long assumed;
    do {
        assumed = old;
        unsigned int old_dist = (unsigned int)(assumed >> 32);
        unsigned int new_dist = (unsigned int)(val >> 32);
        if (new_dist >= old_dist) break;
        old = atomicCAS(address, assumed, val);
    } while (assumed != old);
    return old;
}

// Pack distance and parent into 64-bit key for atomic operations
__device__ __forceinline__ unsigned long long pack_key(float dist, int parent) {
    unsigned int dist_bits = __float_as_uint(dist);
    return ((unsigned long long)dist_bits << 32) | (unsigned long long)(unsigned int)parent;
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
    unsigned long long* best_key,       // 64-bit atomic keys (dist+parent packed)
    unsigned int* frontier_curr,        // Current frontier (bit-packed)
    unsigned int* frontier_next,        // Next frontier (bit-packed)
    const int frontier_words,           // Number of uint32 words for frontier
    const unsigned int* allowed_bitmap, // Optional hard node filter
    const int bitmap_words,             // Words in allowed_bitmap
    const int use_bitmap,               // 1 = enforce bitmap
    const float* node_penalty,           // Optional cost for entering each node
    const int use_node_penalty,          // 1 = add node_penalty[neighbor]
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
            // Initialize best_key with SRC_KEY (dist=0.0, parent=-1)
            best_key[seed] = pack_key(0.0f, -1);

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

                    if (use_bitmap) {
                        int nbr_word = neighbor >> 5;
                        int nbr_bit = neighbor & 31;
                        if (nbr_word >= bitmap_words ||
                            ((allowed_bitmap[nbr_word] >> nbr_bit) & 1u) == 0) {
                            continue;
                        }
                    }

                    float edge_cost = weights[e];
                    if (use_node_penalty) {
                        edge_cost += node_penalty[neighbor];
                    }
                    float g_new = node_dist + edge_cost;

                    // Pack new key with distance and parent
                    unsigned long long new_key = pack_key(g_new, node);

                    // Single atomic operation on 64-bit key - eliminates race condition!
                    unsigned long long old_key = atomicMinDistanceKey(
                        &best_key[neighbor], new_key
                    );

                    // Only the winning thread proceeds
                    if ((unsigned int)(new_key >> 32)
                            < (unsigned int)(old_key >> 32)) {
                        // We won! Update dist and parent arrays (for compatibility)
                        dist[neighbor] = g_new;
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

# Queue-based successor to the bit-packed implementation above.  A monster
# graph has more than 250,000 frontier words, while a single wave normally
# contains only a few thousand nodes.  Scanning and clearing every word three
# times per wave made work proportional to the whole board rather than to the
# search frontier.
PERSISTENT_QUEUE_SSSP_KERNEL_CODE = r'''
#include <cooperative_groups.h>

#define CUDA_INFINITY __int_as_float(0x7f800000)

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

__device__ unsigned long long atomicMinDistanceKey(
    unsigned long long* address,
    unsigned long long value
) {
    unsigned long long old = *address;
    unsigned long long assumed;
    do {
        assumed = old;
        unsigned int old_dist = (unsigned int)(assumed >> 32);
        unsigned int new_dist = (unsigned int)(value >> 32);
        if (new_dist >= old_dist) break;
        old = atomicCAS(address, assumed, value);
    } while (assumed != old);
    return old;
}

__device__ __forceinline__ unsigned long long pack_key(
    float dist,
    int parent
) {
    return ((unsigned long long)__float_as_uint(dist) << 32)
        | (unsigned long long)(unsigned int)parent;
}

extern "C" __global__
void persistent_queue_sssp_kernel(
    const int* indptr,
    const int* indices,
    const float* weights,
    const int num_nodes,
    const int* src_seeds,
    const float* src_seed_costs,
    const int num_srcs,
    const int* dst_targets,
    const float* dst_target_costs,
    const int num_dsts,
    float* dist,
    int* parent,
    unsigned long long* best_key,
    int* queue_a,
    int* queue_b,
    int* queue_count_a,
    int* queue_count_b,
    unsigned int* queued_a,
    unsigned int* queued_b,
    const unsigned int* allowed_bitmap,
    const int bitmap_words,
    const int use_bitmap,
    const float* node_penalty,
    const int use_node_penalty,
    int* settled_flag,
    int* best_dst,
    float* best_dist,
    const int max_iterations,
    int* iteration_count,
    int* overflow_flag
) {
    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + tid;
    const int total_threads = blockDim.x * gridDim.x;
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    int* current_queue = queue_a;
    int* next_queue = queue_b;
    int* current_count = queue_count_a;
    int* next_count = queue_count_b;
    unsigned int* current_bits = queued_a;
    unsigned int* next_bits = queued_b;

    if (global_tid == 0) {
        *current_count = 0;
        *next_count = 0;
        *iteration_count = 0;
        *settled_flag = 0;
        *best_dst = -1;
        *best_dist = CUDA_INFINITY;
        *overflow_flag = 0;
    }
    grid.sync();

    for (int s = global_tid; s < num_srcs; s += total_threads) {
        int seed = src_seeds[s];
        if (seed < 0 || seed >= num_nodes) continue;

        float seed_cost = src_seed_costs[s];
        dist[seed] = seed_cost;
        parent[seed] = -1;
        best_key[seed] = pack_key(seed_cost, -1);

        int word_idx = seed >> 5;
        unsigned int mask = 1u << (seed & 31);
        unsigned int old_bits = atomicOr(&current_bits[word_idx], mask);
        if ((old_bits & mask) == 0) {
            int position = atomicAdd(current_count, 1);
            if (position < num_nodes) {
                current_queue[position] = seed;
            } else {
                atomicExch(overflow_flag, 1);
            }
        }
    }
    grid.sync();

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        if (*overflow_flag) break;

        if (global_tid == 0) {
            *next_count = 0;
        }
        grid.sync();

        int active_count = *current_count;
        for (int position = global_tid;
             position < active_count;
             position += total_threads) {
            int node = current_queue[position];
            if (node < 0 || node >= num_nodes) continue;

            int node_word = node >> 5;
            unsigned int node_mask = 1u << (node & 31);
            atomicAnd(&current_bits[node_word], ~node_mask);

            unsigned long long node_key = best_key[node];
            float node_dist = __uint_as_float(
                (unsigned int)(node_key >> 32)
            );
            if (isinf(node_dist)) continue;
            if (node_dist >= *best_dist) continue;

            int edge_start = indptr[node];
            int edge_end = indptr[node + 1];
            for (int edge = edge_start; edge < edge_end; edge++) {
                int neighbor = indices[edge];
                if (neighbor < 0 || neighbor >= num_nodes) continue;

                if (use_bitmap) {
                    int word_idx = neighbor >> 5;
                    int bit_idx = neighbor & 31;
                    if (word_idx >= bitmap_words
                        || ((allowed_bitmap[word_idx] >> bit_idx) & 1u) == 0) {
                        continue;
                    }
                }

                float edge_cost = weights[edge];
                if (use_node_penalty) {
                    edge_cost += node_penalty[neighbor];
                }
                float candidate = node_dist + edge_cost;
                if (candidate >= *best_dist) continue;
                unsigned long long new_key = pack_key(candidate, node);
                unsigned long long old_key = atomicMinDistanceKey(
                    &best_key[neighbor], new_key
                );

                if ((unsigned int)(new_key >> 32)
                        < (unsigned int)(old_key >> 32)) {
                    int word_idx = neighbor >> 5;
                    unsigned int mask = 1u << (neighbor & 31);
                    unsigned int old_bits = atomicOr(
                        &next_bits[word_idx], mask
                    );
                    if ((old_bits & mask) == 0) {
                        int next_position = atomicAdd(next_count, 1);
                        if (next_position < num_nodes) {
                            next_queue[next_position] = neighbor;
                        } else {
                            atomicExch(overflow_flag, 1);
                        }
                    }
                }
            }
        }
        grid.sync();

        // Establish or tighten a destination upper bound.  The search keeps
        // running below this cost until no improving nodes remain; stopping
        // at the first destination would minimize hop count, not weighted
        // path cost.
        for (int d = global_tid; d < num_dsts; d += total_threads) {
            int dst = dst_targets[d];
            if (dst < 0 || dst >= num_nodes) continue;

            unsigned long long dst_key = best_key[dst];
            float dst_dist = __uint_as_float(
                (unsigned int)(dst_key >> 32)
            );
            if (!isinf(dst_dist)) {
                atomicMinFloat(
                    best_dist, dst_dist + dst_target_costs[d]
                );
            }
        }
        grid.sync();

        if (global_tid == 0) {
            *best_dst = -1;
        }
        grid.sync();

        for (int d = global_tid; d < num_dsts; d += total_threads) {
            int dst = dst_targets[d];
            if (dst < 0 || dst >= num_nodes) continue;

            unsigned long long dst_key = best_key[dst];
            float dst_dist = __uint_as_float(
                (unsigned int)(dst_key >> 32)
            );
            if (!isinf(dst_dist)
                && fabsf(
                    dst_dist + dst_target_costs[d] - *best_dist
                ) < 1e-6f) {
                atomicCAS(best_dst, -1, dst);
                atomicExch(settled_flag, 1);
            }
        }
        grid.sync();

        if (global_tid == 0) {
            *iteration_count = iteration + 1;
        }
        grid.sync();

        if (*overflow_flag || *next_count == 0) break;

        int* queue_tmp = current_queue;
        current_queue = next_queue;
        next_queue = queue_tmp;

        int* count_tmp = current_count;
        current_count = next_count;
        next_count = count_tmp;

        unsigned int* bits_tmp = current_bits;
        current_bits = next_bits;
        next_bits = bits_tmp;
        grid.sync();
    }
}
'''


def create_persistent_kernel():
    """Create and return the compiled persistent SSSP kernel with cooperative groups enabled."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy not available - cannot create persistent kernel")

    # CRITICAL: Enable cooperative groups for grid.sync() to work
    return cp.RawKernel(
        PERSISTENT_QUEUE_SSSP_KERNEL_CODE,
        'persistent_queue_sssp_kernel',
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
    best_key_gpu,
    frontier_words,
    allowed_bitmap_gpu=None,
    use_bitmap=False,
    node_penalty_gpu=None,
    use_node_penalty=False,
    src_seed_costs_gpu=None,
    dst_target_costs_gpu=None,
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
        src_seed_costs_gpu: Initial cost aligned with each source seed
        dst_targets_gpu: Destination targets array (CuPy int32)
        dst_target_costs_gpu: Terminal cost aligned with each destination
        dist_gpu: Distance array (CuPy float32), pre-initialized to inf
        parent_gpu: Parent array (CuPy int32), pre-initialized to -1
        best_key_gpu: 64-bit atomic key array (CuPy uint64), packed dist+parent
        frontier_words: Number of uint32 words for frontier
        allowed_bitmap_gpu: Optional hard node-filter bitmap (CuPy uint32)
        use_bitmap: Whether to enforce allowed_bitmap_gpu
        node_penalty_gpu: Optional per-node entry cost (CuPy float32)
        use_node_penalty: Whether to add node_penalty_gpu during relaxation
        max_iterations: Maximum iterations before timeout

    Returns:
        Tuple of (best_dst, best_dist, iterations)
        best_dst: Index of best destination found (-1 if no path)
        best_dist: Distance to best destination (inf if no path)
        iterations: Number of iterations performed
    """
    import cupy as cp

    if allowed_bitmap_gpu is None or not use_bitmap:
        bitmap_words = frontier_words
        bitmap_gpu = cp.full(bitmap_words, 0xFFFFFFFF, dtype=cp.uint32)
        use_bitmap_flag = 0
    else:
        bitmap_gpu = cp.asarray(
            allowed_bitmap_gpu, dtype=cp.uint32
        ).ravel()
        bitmap_words = int(len(bitmap_gpu))
        use_bitmap_flag = 1

    if node_penalty_gpu is None or not use_node_penalty:
        penalty_gpu = cp.zeros(1, dtype=cp.float32)
        use_node_penalty_flag = 0
    else:
        penalty_gpu = cp.asarray(
            node_penalty_gpu, dtype=cp.float32
        ).ravel()
        if len(penalty_gpu) != num_nodes:
            raise ValueError(
                f"node_penalty has {len(penalty_gpu)} nodes, "
                f"expected {num_nodes}"
            )
        use_node_penalty_flag = 1

    # Queue entries are compact active node IDs.  Bitmaps only deduplicate
    # enqueue operations; the kernel never scans them.
    queue_a = cp.empty(num_nodes, dtype=cp.int32)
    queue_b = cp.empty(num_nodes, dtype=cp.int32)
    queue_count_a = cp.zeros(1, dtype=cp.int32)
    queue_count_b = cp.zeros(1, dtype=cp.int32)
    queued_a = cp.zeros(frontier_words, dtype=cp.uint32)
    queued_b = cp.zeros(frontier_words, dtype=cp.uint32)

    # Allocate output buffers
    settled_flag = cp.zeros(1, dtype=cp.int32)
    best_dst = cp.full(1, -1, dtype=cp.int32)
    best_dist = cp.full(1, cp.inf, dtype=cp.float32)
    iteration_count = cp.zeros(1, dtype=cp.int32)
    overflow_flag = cp.zeros(1, dtype=cp.int32)

    num_srcs = len(src_seeds_gpu)
    num_dsts = len(dst_targets_gpu)
    if src_seed_costs_gpu is None:
        source_costs_gpu = cp.zeros(num_srcs, dtype=cp.float32)
    else:
        source_costs_gpu = cp.asarray(
            src_seed_costs_gpu, dtype=cp.float32
        ).ravel()
        if len(source_costs_gpu) != num_srcs:
            raise ValueError(
                f"src_seed_costs has {len(source_costs_gpu)} entries, "
                f"expected {num_srcs}"
            )
    if dst_target_costs_gpu is None:
        target_costs_gpu = cp.zeros(num_dsts, dtype=cp.float32)
    else:
        target_costs_gpu = cp.asarray(
            dst_target_costs_gpu, dtype=cp.float32
        ).ravel()
        if len(target_costs_gpu) != num_dsts:
            raise ValueError(
                f"dst_target_costs has {len(target_costs_gpu)} entries, "
                f"expected {num_dsts}"
            )

    # Kernel launch config: cooperative launches need every block resident,
    # so size the grid from this GPU's occupancy instead of a hardcoded 80.
    threads_per_block = 256
    num_blocks = _cooperative_grid_size(kernel, threads_per_block)

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
            source_costs_gpu,
            num_srcs,
            dst_targets_gpu,
            target_costs_gpu,
            num_dsts,
            dist_gpu,
            parent_gpu,
            best_key_gpu,           # Added: 64-bit atomic keys for cycle-proof relaxation
            queue_a,
            queue_b,
            queue_count_a,
            queue_count_b,
            queued_a,
            queued_b,
            bitmap_gpu,
            bitmap_words,
            use_bitmap_flag,
            penalty_gpu,
            use_node_penalty_flag,
            settled_flag,
            best_dst,
            best_dist,
            max_iterations,
            iteration_count,
            overflow_flag
        )
    )

    # Wait for completion
    cp.cuda.Stream.null.synchronize()

    # Extract results
    result_dst = int(best_dst[0])
    result_dist = float(best_dist[0])
    result_iters = int(iteration_count[0])
    if int(overflow_flag[0]) != 0:
        raise RuntimeError("Persistent CUDA frontier queue overflowed")

    return result_dst, result_dist, result_iters
