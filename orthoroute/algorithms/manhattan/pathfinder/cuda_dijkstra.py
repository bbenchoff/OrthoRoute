"""
CUDA GPU Shortest-Path Solvers

Clean, focused GPU implementation for parallel shortest path computation.
This module handles ONLY GPU pathfinding - all graph state remains in unified_pathfinder.py

Naming note: the module name is historical. The primary solver
(find_path_fullgraph_gpu_seeds) is a frontier/queue label-correcting search
(Bellman-Ford family) with strict-improvement packed atomic keys and
destination upper-bound pruning — not heap-based Dijkstra.
"""

import logging
from typing import List, Optional, Tuple

from .cuda_common import DEVICE_PRELUDE

try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.sparse
    CUDA_AVAILABLE = True
except ImportError:
    # Create dummy module for type hints when CuPy not available
    import numpy as np
    class _DummyCuPy:
        ndarray = np.ndarray  # Use numpy's ndarray as fallback type for type hints
    cp = _DummyCuPy()
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE: GPU DIAGNOSTIC VERBOSITY CONTROL
# ============================================================================
# Set to True ONLY when debugging GPU kernel behavior
# When False, eliminates 997ms of diagnostic overhead per net (91.5% of GPU time!)
DEBUG_VERBOSE_GPU = False


# ============================================================================
# ROI TUPLE VALIDATION AND NORMALIZATION
# ============================================================================

def _validate_roi_tuple(tup, expected_len=13):
    """Validate ROI tuple format

    Args:
        tup: ROI tuple to validate
        expected_len: Expected tuple length (default: 13)

    Raises:
        ValueError: If tuple length doesn't match expected length

    Returns:
        The validated tuple
    """
    if len(tup) != expected_len:
        raise ValueError(f"ROI tuple length mismatch: got {len(tup)}, expected {expected_len}")
    return tup


def _normalize_roi_tuple(t):
    """Convert old tuple formats to 13-element format

    Handles backward compatibility by normalizing various tuple formats:
    - 13-element: (src, dst, indptr, indices, weights, roi_size, bitmap, bbox_minx, bbox_maxx, bbox_miny, bbox_maxy, bbox_minz, bbox_maxz)
    - 11-element: Old format without entry/exit layers
    - 6-element: Minimal format from legacy code

    Args:
        t: ROI tuple to normalize

    Returns:
        Normalized 13-element tuple

    Raises:
        ValueError: If tuple length is not 6, 11, or 13
    """
    if len(t) == 13:
        return t

    if len(t) == 11:
        # Old format: inject defaults for entry/exit layers
        logger.warning(f"[ROI-TUPLE] Normalized 11-element tuple to 13 elements")
        roi_nodes, g2r, bbox, src, dst, via_mask, plane, xs, ys, zmin, zmax = t
        entry_layer = zmin
        exit_layer = zmax
        return (roi_nodes, g2r, bbox, entry_layer, exit_layer, src, dst,
                via_mask, plane, xs, ys, zmin, zmax)

    if len(t) == 6:
        # Minimal format: pad with Nones
        logger.warning(f"[ROI-TUPLE] Normalized 6-element tuple to 13 elements")
        roi_nodes, g2r, bbox, src, dst, roi_size = t
        return (roi_nodes, g2r, bbox, None, None, src, dst,
                None, roi_size, None, None, None, None)

    raise ValueError(f"Cannot normalize ROI tuple of length {len(t)}")


# Import GPU configuration (hardcoded for plugin deployment)
# NOTE: This creates a circular import, but it's safe because we only access class attributes
# Alternative: Move GPUConfig to a separate config.py module
try:
    from ..unified_pathfinder import GPUConfig
except ImportError:
    # Fallback if circular import causes issues - define minimal config here
    class GPUConfig:
        GPU_MODE = True
        DEBUG_INVARIANTS = True
        USE_PERSISTENT_KERNEL = False  # Enable persistent kernel with atomic parent keys (eliminates cycles)
        USE_GPU_COMPACTION = True
        USE_DELTA_STEPPING = True  # Use proper Delta-stepping bucket-based priority queue (instead of BFS wavefront)
        DELTA_VALUE = 0.5  # Bucket width in mm (0.5mm ~= 1.25 x 0.4mm grid pitch)


class CUDADijkstra:
    """GPU-accelerated Dijkstra shortest path finder using CUDA"""

    def __init__(self, graph=None, lattice=None):
        """Initialize CUDA Dijkstra solver"""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CuPy not available - cannot use CUDA Dijkstra")

        # Keep an existing device graph on the device.  Downloading the full
        # CSR here duplicates multi-gigabyte graphs and is unnecessary for
        # full-graph CUDA searches.  ROI extraction takes a CPU copy lazily.
        if graph:
            self.indptr = graph.indptr
            self.indices = graph.indices
        else:
            self.indptr = None
            self.indices = None

        # Store lattice for A* coordinate building
        self.lattice = lattice

        # Phase 1: Device-Resident Stamp Pools (allocated once, reused forever)
        # Calculate K_pool from available GPU memory (will be set during first _prepare_batch call)
        self.K_pool = None  # Will be calculated dynamically
        self._k_pool_calculated = False

        # Allocate device pools ONCE (reused for all batches)
        self.dist_val_pool = None
        self.dist_stamp_pool = None
        self.parent_val_pool = None
        self.parent_stamp_pool = None
        self.near_bits_pool = None  # Phase B: Bitset frontier (1 bit/node)
        self.far_bits_pool = None   # Phase B: Bitset frontier (1 bit/node)
        self.current_gen = 1  # Generation counter

        # PERFORMANCE: Persistent kernel (compiled on-demand)
        self._persistent_kernel = None
        self._enable_persistent_kernel = True
        # The persistent kernel uses compact frontier queues, packed atomic
        # parent keys, and weighted-cost convergence.
        self._zero_node_penalty = cp.zeros(1, dtype=cp.float32)
        self._fullgraph_backtrace_kernel = None
        self._fullgraph_backtrace_buffer = None
        self._fullgraph_backtrace_meta = None

        # Compile FULLY PARALLEL wavefront expansion kernel
        # This processes ALL K ROIs + ALL frontier nodes in ONE launch
        # SUPPORTS SHARED CSR: Use stride=0 for broadcast arrays (no duplication!)
        self.wavefront_kernel = cp.RawKernel(DEVICE_PRELUDE + r'''
        // Atomic key constants
        const unsigned long long SRC_KEY = 0x00000000FFFFFFFF;  // cost=0, parent=-1
        const unsigned long long INF_KEY = 0x7F800000FFFFFFFF;  // +inf, parent=-1

        // FIX-BITMAP: Macro to check if node is in ROI bitmap
        #define IN_BITMAP(roi_idx, node, roi_bitmap, bitmap_words) ( \
            ((node) >> 5) < (bitmap_words) && \
            ((roi_bitmap)[(roi_idx) * (bitmap_words) + ((node) >> 5)] >> ((node) & 31)) & 1u )

        extern "C" __global__
        void wavefront_expand_all(
            const int K,                    // Number of ROIs
            const int max_roi_size,         // Max nodes per ROI
            const int max_edges,            // Max edges per ROI
            const unsigned int* frontier,   // (K, frontier_words) BIT-PACKED frontier - uint32!
            const int frontier_words,       // Number of uint32 words per ROI
            const int* indptr,              // CSR indptr base pointer
            const int* indices,             // CSR indices base pointer
            const float* weights,           // CSR weights base pointer
            const int indptr_stride,        // Stride between ROI rows (0 for shared CSR!)
            const int indices_stride,       // Stride between ROI rows (0 for shared CSR!)
            const int weights_stride,       // Stride between ROI rows (0 for shared CSR!)
            const float* total_cost,            // CSR negotiated costs
            const int total_cost_stride,        // Stride for total_cost
            const float* node_penalty,           // Optional cost for entering a node
            const int node_penalty_stride,       // Stride between ROI penalty rows
            const int use_node_penalty,          // 1 = add node penalty, 0 = ignore
            const int* goal_nodes,          // (K,) goal node index for each ROI (for A* heuristic)
            const int Nx,                   // P0-3: Lattice X dimension
            const int Ny,                   // P0-3: Lattice Y dimension
            const int Nz,                   // P0-3: Lattice Z dimension
            const int* goal_coords,         // P0-3: (K, 3) goal coordinates only
            const int use_astar,            // 1 = enable A* heuristic, 0 = plain Dijkstra
            float* dist,                    // (K, max_roi_size) distances - SLICED pool view
            const int dist_stride,          // NEW: Pool stride for dist
            int* parent,                    // (K, max_roi_size) parents - SLICED pool view
            const int parent_stride,        // NEW: Pool stride for parent
            unsigned int* new_frontier,     // (K, frontier_words) BIT-PACKED output - uint32!
            // FIX-7: ROI bitmap for neighbor validation
            const unsigned int* roi_bitmap, // (K, bitmap_words) per ROI - neighbor must be in bitmap!
            const int bitmap_words,         // Words per ROI bitmap
            const int use_bitmap,           // 1 = enforce bitmap, 0 = bbox-only (iteration 1 mode)
            const int iter1_relax_hv,       // 1 = relax H/V discipline in Iter-1 (always write parent)
            // ATOMIC PARENT KEYS: Cycle-proof parent tracking for Iter>=2
            unsigned long long* best_key,   // (K, key_stride) 64-bit atomic keys (cost+parent)
            const int key_stride,           // Stride for best_key array
            const int use_atomic_parent_keys // 1 = use atomic keys (Iter>=2), 0 = legacy (Iter==1)
        ) {
            // Block index = ROI index (expects exactly K blocks!)
            int roi_idx = blockIdx.x;
            if (roi_idx >= K) return;

            // Thread index within block
            int tid = threadIdx.x;
            int B = blockDim.x;

            // Calculate base offsets for this ROI
            // For dist/parent: use stride parameters
            const size_t dist_off = (size_t)roi_idx * (size_t)dist_stride;
            const size_t parent_off = (size_t)roi_idx * (size_t)parent_stride;
            // For frontier: use roi_idx * frontier_words (bit-packed)
            const int frontier_off = roi_idx * frontier_words;

            // For CSR arrays: use stride (0 for shared, actual dimension for per-ROI)
            // When stride=0 (shared CSR), all ROIs use the same base pointer!
            const int indptr_off = roi_idx * indptr_stride;   // 0 in shared mode
            const int indices_off = roi_idx * indices_stride; // 0 in shared mode
            const int weights_off = roi_idx * weights_stride; // 0 in shared mode
            const int total_cost_off = roi_idx * total_cost_stride; // PathFinder negotiated costs

            // Grid-stride loop over ALL nodes in this ROI
            for (int node = tid; node < max_roi_size; node += B) {
                // Check if this node is in the frontier (BIT-PACKED CHECK)
                const int word_idx = node / 32;
                const int bit_pos = node % 32;
                const unsigned int bit_mask = 1u << bit_pos;
                if ((frontier[frontier_off + word_idx] & bit_mask) == 0) continue;  // Bit not set

                // NOTE: We DO NOT check if current node is in bitmap - it's in frontier, so expand it
                // Bitmap check happens when relaxing NEIGHBOR edges (prevents expanding OUTSIDE ROI)

                // Get node distance
                const int nidx_self = dist_off + node;
                const float node_dist = dist[nidx_self];

                // Skip unreachable nodes (inf distance only)
                if (isinf(node_dist)) continue;

                // Get CSR edge range for this node (uses stride-aware offset)
                const int e0 = indptr[indptr_off + node];
                const int e1 = indptr[indptr_off + node + 1];

                // Process all edges (warp-level parallelism)
                for (int e = e0; e < e1; ++e) {
                    const int neighbor = indices[indices_off + e];
                    // Bounds check to prevent corruption
                    if (neighbor < 0 || neighbor >= max_roi_size) continue;

                    float edge_cost = total_cost[total_cost_off + e];  // Use negotiated cost (PathFinder)
                    if (use_node_penalty) {
                        edge_cost += node_penalty[
                            (size_t)roi_idx * (size_t)node_penalty_stride
                            + (size_t)neighbor
                        ];
                    }
                    const float g_new = node_dist + edge_cost;  // g(n) = distance from start

                    // FIX-7: BITMAP CHECK - conditional based on use_bitmap flag
                    if (use_bitmap) {
                        const int nbr_word = neighbor >> 5;
                        const int nbr_bit = neighbor & 31;
                        const int bitmap_off = roi_idx * bitmap_words;
                        // Bounds check for bitmap access
                        if (nbr_word >= bitmap_words) continue;  // Neighbor index exceeds bitmap size
                        const unsigned int nbr_in_bitmap = (roi_bitmap[bitmap_off + nbr_word] >> nbr_bit) & 1;
                        if (nbr_in_bitmap == 0) {
                            continue;  // Neighbor not in ROI - prevents diagonal traces!
                        }
                    }

                    // P0-3: A* HEURISTIC with procedural coordinate decoding
                    float f_new = g_new;  // Default: Dijkstra (no heuristic)

                    if (use_astar) {
                        // Decode neighbor coordinates from node index (no memory loads!)
                        const int plane_size = Nx * Ny;
                        const int nz = neighbor / plane_size;
                        const int remainder = neighbor - (nz * plane_size);
                        const int ny = remainder / Nx;
                        const int nx = remainder - (ny * Nx);

                        // Load goal coordinates (just 3 ints per ROI)
                        const int gx = goal_coords[roi_idx * 3 + 0];
                        const int gy = goal_coords[roi_idx * 3 + 1];
                        const int gz = goal_coords[roi_idx * 3 + 2];

                        // Manhattan distance heuristic (admissible)
                        const float h = (abs(gx - nx) + abs(gy - ny)) * 0.4f + abs(gz - nz) * 1.5f;
                        f_new = g_new + h;
                    }

                    const int nidx = dist_off + neighbor;

                    // ATOMIC KEY PATH: Cycle-proof single-op update (Iter>=2)
                    if (use_atomic_parent_keys) {
                        // Pack cost+parent into 64-bit key
                        const unsigned long long new_key = pack_key(g_new, node);
                        unsigned long long* key_ptr = &best_key[(size_t)roi_idx * (size_t)key_stride + (size_t)neighbor];
                        const unsigned long long old_key = atomicMinDistanceKey(key_ptr, new_key);

                        // Only the winning thread proceeds
                        if ((unsigned int)(new_key >> 32)
                                < (unsigned int)(old_key >> 32)) {
                            // We won! Always update dist (enables early exit & valid backtrace)
                            dist[nidx] = g_new;
                            // Mirror parent to legacy array for tooling/accounting compatibility
                            atomicExch(&parent[parent_off + neighbor], node);
                            // Enqueue in frontier
                            const int nbr_word_idx = neighbor / 32;
                            const int nbr_bit_pos = neighbor % 32;
                            atomicOr(&new_frontier[frontier_off + nbr_word_idx], 1u << nbr_bit_pos);
                        }
                    } else {
                        // LEGACY PATH: Separate dist/parent updates (Iter==1, has race conditions)
                        const float old = atomicMinFloat(&dist[nidx], g_new);  // Store g(n), not f(n)!
                        // STABILIZATION: Epsilon guard prevents float-noise equal-cost flip-flops
                        if (g_new + 1e-8f < old) {
                            // MANHATTAN VALIDATION: Verify parent->child is adjacent
                            // Decode current node coordinates
                            const int plane_size_node = Nx * Ny;
                            const int z_node = node / plane_size_node;
                            const int remainder_node = node - (z_node * plane_size_node);
                            const int y_node = remainder_node / Nx;
                            const int x_node = remainder_node - (y_node * Nx);

                            // Check if parent->child relationship is Manhattan-legal
                            bool valid_parent = false;
                            if (z_node != nz) {
                                // Via jump - same X,Y required
                                if (nx == x_node && ny == y_node) {
                                    valid_parent = true;
                                }
                            } else {
                                // Same layer - must be adjacent with correct direction
                                const int dx = abs(nx - x_node);
                                const int dy = abs(ny - y_node);

                                if (dx + dy == 1) {
                                    // Check layer direction discipline (matches graph construction)
                                    const bool is_h_layer = (nz % 2) == 1;  // Odd layers = horizontal
                                    if (is_h_layer) {
                                        if (dy == 0) valid_parent = true;  // H layer must have dy=0
                                    } else {
                                        if (dx == 0) valid_parent = true;  // V layer must have dx=0
                                    }
                                }
                            }

                            // Update parent based on validation and Iter-1 relaxation mode
                            if (valid_parent || iter1_relax_hv) {
                                // FIX D1: ALWAYS update parent on improvement (allow node reopening!)
                                // Use atomicExch for thread-safe parent update
                                // In Iter-1 with relax mode: write parent even if H/V invalid (soft penalty)
                                // In Iter-2+: only write parent if H/V discipline is satisfied
                                atomicExch(&parent[parent_off + neighbor], node);
                            }

                            // P0-4: BIT-PACKED WRITE using atomicOr
                            const int nbr_word_idx = neighbor / 32;
                            const int nbr_bit_pos = neighbor % 32;
                            atomicOr(&new_frontier[frontier_off + nbr_word_idx], 1u << nbr_bit_pos);
                        }
                    }
                }
            }
        }
        ''', 'wavefront_expand_all')

        # Compile ACTIVE-LIST kernel (MASSIVE SPEEDUP!)
        # This processes only ACTIVE frontier nodes (~1000) instead of ALL nodes (4.2M)
        # Uses global thread indexing - launches over total_active items across ALL ROIs
        # Expected: Higher GPU occupancy, better load balancing, 2-3× faster than one-block-per-ROI
        self.active_list_kernel = cp.RawKernel(DEVICE_PRELUDE + r'''
        // Atomic min for 64-bit unsigned integers (cycle-proof relaxation)
        __device__ __forceinline__
        unsigned long long atomicMin64(unsigned long long* address, unsigned long long val) {
            unsigned long long old = *address;
            unsigned long long assumed;
            do {
                assumed = old;
                old = atomicCAS(address, assumed, (val < assumed) ? val : assumed);
            } while (assumed != old);
            return old;
        }

        // FIX-BITMAP: Macro to check if node is in ROI bitmap
        #define IN_BITMAP(roi_idx, node, roi_bitmap, bitmap_words) ( \
            ((node) >> 5) < (bitmap_words) && \
            ((roi_bitmap)[(roi_idx) * (bitmap_words) + ((node) >> 5)] >> ((node) & 31)) & 1u )

        // Phase 4: ROI bounding box check
        __device__ __forceinline__
        bool in_roi(int nx, int ny, int nz, int roi_idx,
                    const int* minx, const int* maxx,
                    const int* miny, const int* maxy,
                    const int* minz, const int* maxz) {
            return nx >= minx[roi_idx] && nx <= maxx[roi_idx] &&
                   ny >= miny[roi_idx] && ny <= maxy[roi_idx] &&
                   nz >= minz[roi_idx] && nz <= maxz[roi_idx];
        }

        extern "C" __global__
        void wavefront_expand_active(
            const int total_active,             // Total active frontier nodes across ALL ROIs
            const int max_roi_size,             // Max nodes per ROI
            const int frontier_words,           // Number of uint32 words per ROI
            const int* roi_ids,                 // (total_active,) ROI ID for each active node
            const int* node_ids,                // (total_active,) Node ID within ROI
            const int* indptr,                  // CSR indptr
            const int* indices,                 // CSR indices
            const float* weights,               // CSR weights
            const int indptr_stride,            // Stride (0 for shared)
            const int indices_stride,
            const int weights_stride,
            const float* total_cost,            // CSR negotiated costs
            const int total_cost_stride,        // Stride for total_cost
            const float* node_penalty,           // Optional cost for entering a node
            const int node_penalty_stride,       // Stride between ROI penalty rows
            const int use_node_penalty,          // 1 = add node penalty, 0 = ignore
            const int* goal_nodes,              // (K,) goal indices for A*
            const int Nx,                       // P0-3: Lattice X dimension (procedural coords)
            const int Ny,                       // P0-3: Lattice Y dimension
            const int Nz,                       // P0-3: Lattice Z dimension (layers)
            const int* goal_coords,             // (K, 3) goal coordinates (gx, gy, gz) per ROI
            const int use_astar,                // A* enable flag
            float* dist,                        // (K, max_roi_size) distances - SLICED pool view
            const int dist_stride,              // NEW: Pool stride for dist
            int* parent,                        // (K, max_roi_size) parents - SLICED pool view
            const int parent_stride,            // NEW: Pool stride for parent
            unsigned int* new_frontier,         // (K, frontier_words) BIT-PACKED output
            // FIX-7: ROI bitmap for neighbor validation
            const unsigned int* roi_bitmap,     // (K, bitmap_words) per ROI - neighbor must be in bitmap!
            const int bitmap_words,             // Words per ROI bitmap
            const int use_bitmap,               // 1 = enforce bitmap, 0 = bbox-only (iteration 1 mode)
            int* blocked_by_bitmap_counter,     // INSTRUMENTATION: count neighbors blocked by bitmap
            // Phase 4: ROI bounding boxes
            const int* roi_minx,                // (K,) Min X per ROI
            const int* roi_maxx,                // (K,) Max X per ROI
            const int* roi_miny,                // (K,) Min Y per ROI
            const int* roi_maxy,                // (K,) Max Y per ROI
            const int* roi_minz,                // (K,) Min Z per ROI
            const int* roi_maxz,                // (K,) Max Z per ROI
            const int* pref_layer,              // (K,) preferred even layer per ROI
            const int* src_x_coord,             // (K,) source x-coordinate per ROI
            const int window_cols,              // Bias window size (columns, ~8mm)
            const float rr_alpha,               // Bias strength (0.0 = disabled, 0.12 typical)
            const float jitter_eps              // Jitter magnitude (0.001 typical)
        ) {
            // Global thread ID - each thread processes ONE frontier node
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_active) return;

            // Get ROI and node for this frontier item
            const int roi_idx = roi_ids[idx];
            const int node = node_ids[idx];

            // NOTE: We DO NOT check if current node is in bitmap - it's in frontier, so expand it
            // Bitmap check happens when relaxing NEIGHBOR edges (prevents expanding OUTSIDE ROI)

            // CSR offsets for this ROI
            const int indptr_off = roi_idx * indptr_stride;
            const int indices_off = roi_idx * indices_stride;
            const int weights_off = roi_idx * weights_stride;
            const int total_cost_off = roi_idx * total_cost_stride; // PathFinder negotiated costs
            const size_t dist_off = (size_t)roi_idx * (size_t)dist_stride;
            const size_t parent_off = (size_t)roi_idx * (size_t)parent_stride;
            const int frontier_off = roi_idx * frontier_words;

            // Guarded atomic: Check distance before edge expansion
            const float node_dist = __ldg(&dist[dist_off + node]);
            if (isinf(node_dist)) return;

            // Get CSR edge range
            const int e0 = indptr[indptr_off + node];
            const int e1 = indptr[indptr_off + node + 1];

            // Process all neighbors
            for (int e = e0; e < e1; ++e) {
                const int neighbor = indices[indices_off + e];
                if (neighbor < 0 || neighbor >= max_roi_size) continue;

                float edge_cost = total_cost[total_cost_off + e];  // Use negotiated cost (PathFinder)
                if (use_node_penalty) {
                    edge_cost += node_penalty[
                        (size_t)roi_idx * (size_t)node_penalty_stride
                        + (size_t)neighbor
                    ];
                }

                // Phase 4: Decode neighbor coordinates (always needed for ROI check)
                const int plane_size = Nx * Ny;
                const int nz = neighbor / plane_size;
                const int remainder = neighbor - (nz * plane_size);  // Faster than % on older arch
                const int ny = remainder / Nx;
                const int nx = remainder - (ny * Nx);

                // Phase 4: ROI gate - skip neighbors outside bounding box
                if (!in_roi(nx, ny, nz, roi_idx, roi_minx, roi_maxx, roi_miny, roi_maxy, roi_minz, roi_maxz)) {
                    continue;  // Skip this neighbor
                }

                // FIX-7: BITMAP CHECK - conditional based on use_bitmap flag
                if (use_bitmap) {
                    const int nbr_word = neighbor >> 5;  // neighbor / 32
                    const int nbr_bit = neighbor & 31;   // neighbor % 32
                    const int bitmap_off = roi_idx * bitmap_words;
                    // Bounds check for bitmap access
                    if (nbr_word >= bitmap_words) continue;  // Neighbor index exceeds bitmap size
                    const unsigned int nbr_in_bitmap = (roi_bitmap[bitmap_off + nbr_word] >> nbr_bit) & 1;
                    if (nbr_in_bitmap == 0) {
                        // INSTRUMENTATION: Count blocked neighbors
                        atomicAdd(blocked_by_bitmap_counter, 1);
                        continue;  // Neighbor not in ROI - skip to prevent diagonal traces!
                    }
                }

                // === ROUND-ROBIN LAYER BIAS ===
                if (rr_alpha > 0.0f) {
                    const int plane_size = Nx * Ny;
                    const int z_node = node / plane_size;
                    const int x_node = (node % plane_size) % Nx;
                    const int z_neighbor = neighbor / plane_size;
                    const bool is_vertical = (z_neighbor != z_node);

                    // Only bias vertical edges on even layers within window
                    if (is_vertical && (z_node & 1) == 0) {
                        int dx = x_node - src_x_coord[roi_idx];
                        if (dx < 0) dx = -dx;

                        if (dx <= window_cols) {
                            const int pref_z = pref_layer[roi_idx];
                            const float m = (z_node == pref_z) ? (1.0f - rr_alpha) : (1.0f + rr_alpha);
                            edge_cost *= m;
                        }
                    }
                }

                // Add deterministic jitter
                float jitter = 0.0f;
                if (jitter_eps > 0.0f) {
                    unsigned int hash = (unsigned int)node * 73856093u
                                      ^ (unsigned int)neighbor * 19349663u
                                      ^ (unsigned int)roi_idx * 83492791u;
                    float normalized = (float)(hash & 0x7FFFFFu) / (float)0x7FFFFFu * 2.0f - 1.0f;
                    jitter = normalized * jitter_eps;
                }

                // Apply jitter to cost
                const float g_new = node_dist + edge_cost + jitter;

                // P0-3: A* heuristic with PROCEDURAL coordinate decoding (no global loads!)
                float f_new = g_new;
                if (use_astar) {
                    // Load goal coordinates (just 3 ints per ROI, preloaded once per batch)
                    const int gx = goal_coords[roi_idx * 3 + 0];
                    const int gy = goal_coords[roi_idx * 3 + 1];
                    const int gz = goal_coords[roi_idx * 3 + 2];

                    // Manhattan distance heuristic (admissible)
                    const float h = (abs(gx - nx) + abs(gy - ny)) * 0.4f + abs(gz - nz) * 1.5f;
                    f_new = g_new + h;
                }

                // Guarded atomic: Only CAS if we might improve distance
                const int nidx = dist_off + neighbor;
                const float old_dist = __ldg(&dist[nidx]);

                // Early exit if we can't improve
                if (g_new >= old_dist) continue;

                // Try atomic update
                const float old = atomicMinFloat(&dist[nidx], g_new);

                // Update parent and frontier on improvement
                if (g_new + 1e-8f < old) {
                    // Update parent
                    atomicExch(&parent[parent_off + neighbor], node);

                    // BIT-PACKED WRITE to frontier
                    const int nbr_word_idx = neighbor / 32;
                    const int nbr_bit_pos = neighbor % 32;
                    atomicOr(&new_frontier[frontier_off + nbr_word_idx], 1u << nbr_bit_pos);
                }
            }
        }
        ''', 'wavefront_expand_active')

        # P1-7: DELTA-STEPPING BUCKET ASSIGNMENT KERNEL
        # Replaces slow Python loop that iterates over all nodes
        # This kernel processes only updated nodes and assigns them to buckets in parallel
        self.bucket_assign_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void assign_nodes_to_buckets(
            const int K,                        // Number of ROIs
            const int max_roi_size,             // Max nodes per ROI
            const int max_buckets,              // Max number of buckets
            const int frontier_words,           // Number of uint32 words per ROI
            const unsigned int* updated_nodes,  // (K, frontier_words) BIT-PACKED updated nodes
            const float* dist,                  // (K, max_roi_size) current distances
            const float delta,                  // Bucket width
            unsigned int* buckets               // (K, max_buckets, frontier_words) output buckets
        ) {
            // Grid-stride loop over all possible nodes across all ROIs
            // Total work = K * max_roi_size nodes
            const int total_nodes = K * max_roi_size;

            for (int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
                 global_idx < total_nodes;
                 global_idx += blockDim.x * gridDim.x) {

                // Decode ROI and node from global index
                const int roi_idx = global_idx / max_roi_size;
                const int node = global_idx % max_roi_size;

                // Check if this node was updated (bit-packed check)
                const int word_idx = node / 32;
                const int bit_pos = node % 32;
                const unsigned int bit_mask = 1u << bit_pos;

                const int frontier_off = roi_idx * frontier_words;
                const unsigned int updated_word = __ldg(&updated_nodes[frontier_off + word_idx]);

                // Early exit if this node was not updated
                if (!(updated_word & bit_mask)) continue;

                // Get node's current distance
                const int dist_off = roi_idx * max_roi_size;
                const float node_dist = __ldg(&dist[dist_off + node]);

                // Skip nodes with infinite distance (unreachable)
                if (isinf(node_dist)) continue;

                // Compute bucket index: bucket_idx = floor(distance / delta)
                const int bucket_idx = __float2int_rd(node_dist / delta);

                // Bounds check: ensure bucket is within range
                if (bucket_idx >= 0 && bucket_idx < max_buckets) {
                    // Atomic OR to set bit in target bucket
                    // Bucket offset: (roi_idx * max_buckets + bucket_idx) * frontier_words + word_idx
                    const int bucket_off = (roi_idx * max_buckets + bucket_idx) * frontier_words;
                    atomicOr(&buckets[bucket_off + word_idx], bit_mask);
                }
            }
        }
        ''', 'assign_nodes_to_buckets')

        # P1-6: Compile PERSISTENT KERNEL with device-side queues
        # This kernel runs a while-loop on GPU until all paths found, eliminating kernel launch overhead
        # Uses cooperative groups for grid-wide synchronization
        self.persistent_kernel = cp.RawKernel(DEVICE_PRELUDE + r'''
        #include <cooperative_groups.h>
        namespace cg = cooperative_groups;

        // FIX-BITMAP: Macro to check if node is in ROI bitmap
        #define IN_BITMAP(roi_idx, node, roi_bitmap, bitmap_words) ( \
            ((node) >> 5) < (bitmap_words) && \
            ((roi_bitmap)[(roi_idx) * (bitmap_words) + ((node) >> 5)] >> ((node) & 31)) & 1u )

        extern "C" __global__
        void __launch_bounds__(256)
        sssp_persistent_cooperative(
            int* queue_a,                   // Device queue A (max_queue_size)
            int* queue_b,                   // Device queue B (max_queue_size)
            int* size_a,                    // Size of queue A (device scalar)
            int* size_b,                    // Size of queue B (device scalar)
            const int max_queue_size,       // Maximum queue capacity
            const int K,                    // Number of ROIs
            const int max_roi_size,         // Max nodes per ROI
            const int* indptr,              // CSR indptr
            const int* indices,             // CSR indices
            const float* weights,           // CSR base weights (for accountant)
            const int indptr_stride,        // Stride (0 for shared)
            const int indices_stride,
            const int weights_stride,
            const float* total_cost,        // CSR negotiated costs (weights + present + history)
            const int total_cost_stride,    // Stride for total_cost
            const int Nx,                   // Lattice X dimension
            const int Ny,                   // Lattice Y dimension
            const int Nz,                   // Lattice Z dimension
            const int* goal_coords,         // (K, 3) goal coordinates
            const int use_astar,            // A* enable flag
            float* dist,                    // (K, max_roi_size) distances
            int* parent,                    // (K, max_roi_size) parents
            const unsigned int* roi_bitmap, // FIX-BITMAP: ROI bitmap for validation
            const int bitmap_words,         // FIX-BITMAP: Words per ROI bitmap
            const int use_bitmap,           // 1 = enforce bitmap, 0 = bbox-only (iteration 1 mode)
            const int iter1_relax_hv,       // 1 = relax H/V discipline in Iter-1 (soft penalty, not hard reject)
            int* iterations_out             // Output: number of iterations completed
        ) {
            // Grid handle for cooperative groups
            cg::grid_group grid = cg::this_grid();

            // Thread ID and total threads
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int total_threads = gridDim.x * blockDim.x;

            // Ping-pong flag: which queue is input
            bool use_a = true;

            int iteration = 0;
            const int MAX_ITERATIONS = 2000;  // Safety limit

            while (iteration < MAX_ITERATIONS) {
                // Grid-wide barrier
                grid.sync();

                // Select queues based on ping-pong flag
                int* q_in = use_a ? queue_a : queue_b;
                int* q_out = use_a ? queue_b : queue_a;
                int* sz_in = use_a ? size_a : size_b;
                int* sz_out = use_a ? size_b : size_a;

                // Load queue size (all threads read same value)
                const int queue_size = *sz_in;

                // Termination check
                if (queue_size == 0) {
                    break;
                }

                // Reset output queue size (single thread only)
                if (tid == 0) {
                    *sz_out = 0;
                }
                grid.sync();

                // Process queue in parallel (grid-stride loop)
                for (int i = tid; i < queue_size; i += total_threads) {
                    // Unpack (roi, node) from 32-bit packed format
                    const int packed = q_in[i];
                    const int roi_idx = packed >> 24;          // Upper 8 bits = ROI (supports 256 ROIs)
                    const int node = packed & 0xFFFFFF;        // Lower 24 bits = node (supports 16M nodes)

                    // Bounds check
                    if (roi_idx >= K || node >= max_roi_size) continue;

                    // CSR offsets for this ROI
                    const int indptr_off = roi_idx * indptr_stride;
                    const int indices_off = roi_idx * indices_stride;
                    const int weights_off = roi_idx * weights_stride;
                    const int total_cost_off = roi_idx * total_cost_stride; // PathFinder negotiated costs
                    const int dist_off = roi_idx * max_roi_size;

                    // Get node distance
                    const float node_dist = __ldg(&dist[dist_off + node]);
                    if (isinf(node_dist)) continue;

                    // Get CSR edge range
                    const int e0 = indptr[indptr_off + node];
                    const int e1 = indptr[indptr_off + node + 1];

                    // Process all edges
                    for (int e = e0; e < e1; ++e) {
                        const int neighbor = indices[indices_off + e];
                        if (neighbor < 0 || neighbor >= max_roi_size) continue;

                        // BITMAP CHECK - conditional based on use_bitmap flag
                        if (use_bitmap) {
                            const int nbr_word = neighbor >> 5;
                            const int nbr_bit = neighbor & 31;
                            const int bitmap_off = roi_idx * bitmap_words;
                            if (nbr_word >= bitmap_words) continue;  // Out of bitmap bounds
                            const unsigned int nbr_in_bitmap = (roi_bitmap[bitmap_off + nbr_word] >> nbr_bit) & 1;
                            if (nbr_in_bitmap == 0) continue;  // Neighbor not in ROI bitmap
                        }

                        const float edge_cost = total_cost[total_cost_off + e];  // Use negotiated cost (PathFinder)
                        const float g_new = node_dist + edge_cost;

                        // A* heuristic with procedural coordinate decoding
                        // CRITICAL FIX: Declare neighbor coordinates at function scope (needed for validation later)
                        int nx, ny, nz;

                        float f_new = g_new;
                        if (use_astar) {
                            const int plane_size = Nx * Ny;
                            nz = neighbor / plane_size;  // Now properly scoped (no 'const')
                            const int remainder = neighbor - (nz * plane_size);
                            ny = remainder / Nx;
                            nx = remainder - (ny * Nx);

                            const int gx = goal_coords[roi_idx * 3 + 0];
                            const int gy = goal_coords[roi_idx * 3 + 1];
                            const int gz = goal_coords[roi_idx * 3 + 2];

                            const float h = (abs(gx - nx) + abs(gy - ny)) * 0.4f + abs(gz - nz) * 1.5f;
                            f_new = g_new + h;
                        } else {
                            // Decode coordinates even if not using A* (needed for validation)
                            const int plane_size = Nx * Ny;
                            nz = neighbor / plane_size;
                            const int remainder = neighbor - (nz * plane_size);
                            ny = remainder / Nx;
                            nx = remainder - (ny * Nx);
                        }

                        // Atomic distance update
                        const int nidx = dist_off + neighbor;
                        const float old = atomicMinFloat(&dist[nidx], g_new);

                        // If improved, update parent and add to output queue
                        if (g_new + 1e-8f < old) {
                            // MANHATTAN VALIDATION: Verify parent->child is adjacent
                            // Decode current node coordinates
                            const int plane_size_node = Nx * Ny;
                            const int z_node = node / plane_size_node;
                            const int remainder_node = node - (z_node * plane_size_node);
                            const int y_node = remainder_node / Nx;
                            const int x_node = remainder_node - (y_node * Nx);

                            // Check if parent->child relationship is Manhattan-legal
                            // Neighbor coordinates (nx, ny, nz) are now always decoded above
                            bool valid_parent = false;
                            if (z_node != nz) {
                                // Via jump - same X,Y required
                                if (nx == x_node && ny == y_node) {
                                    valid_parent = true;
                                }
                            } else {
                                // Same layer - must be adjacent with correct direction
                                const int dx = abs(nx - x_node);
                                const int dy = abs(ny - y_node);

                                if (dx + dy == 1) {
                                    // Check layer direction discipline (matches graph construction)
                                    const bool is_h_layer = (nz % 2) == 1;  // Odd layers = horizontal
                                    if (is_h_layer) {
                                        if (dy == 0) valid_parent = true;  // H layer must have dy=0
                                    } else {
                                        if (dx == 0) valid_parent = true;  // V layer must have dx=0
                                    }
                                }
                            }

                            // Only update parent if validation passed
                            if (valid_parent) {
                                atomicExch(&parent[nidx], node);
                            }

                            // Add to output queue using atomic counter
                            int queue_pos = atomicAdd(sz_out, 1);
                            if (queue_pos < max_queue_size) {
                                // Pack (roi, node) into 32 bits: 8-bit ROI + 24-bit node
                                q_out[queue_pos] = (roi_idx << 24) | neighbor;
                            }
                            // Note: If queue overflows, nodes are dropped (graceful degradation)
                            // In practice, queue is sized large enough that this never happens
                        }
                    }
                }

                // Grid-wide barrier before queue swap
                grid.sync();

                // Flip queues for next iteration
                use_a = !use_a;
                iteration++;
            }

            // Write iteration count (single thread)
            if (tid == 0) {
                *iterations_out = iteration;
            }
        }
        ''', 'sssp_persistent_cooperative')

        # P3: Compile COMPACTION kernel (Phase 3: GPU-side frontier compaction)
        # This eliminates host<->device sync during frontier compaction (replaces cp.nonzero)
        # Phase B: Updated to use bitsets (1 bit/node instead of 1 byte/node)
        self.compact_kernel = cp.RawKernel(r'''
        // Phase B: Bitset helper for compaction
        __device__ __forceinline__
        bool get_bit_compact(const unsigned char* bits, int idx) {
            return (bits[idx >> 3] >> (idx & 7)) & 1;
        }

        extern "C" __global__
        void compact_mask_to_list(
            const unsigned char* __restrict__ frontier_bits,  // Phase B: bitset input (1 bit/node)
            int N,
            int* __restrict__ out_idx,
            int* __restrict__ out_count
        ) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Phase B: Read bit instead of byte
            bool is_set = (tid < N) && get_bit_compact(frontier_bits, tid);
            unsigned int warp_mask = __ballot_sync(0xffffffff, is_set);
            int lane = threadIdx.x & 31;

            if (warp_mask) {
                int warp_count = __popc(warp_mask);
                int base = 0;
                if (lane == 0) {
                    base = atomicAdd(out_count, warp_count);
                }
                base = __shfl_sync(0xffffffff, base, 0);
                int offset = __popc(warp_mask & ((1u << lane) - 1));

                if (is_set) {
                    out_idx[base + offset] = tid;
                }
            }
        }
        ''', 'compact_mask_to_list')

        # PARENT-CSR CONSISTENCY VALIDATOR KERNEL: Diagnose parent pointer corruption
        # Checks if every parent[node] is actually a CSR neighbor of that node
        # This helps identify if corruption happens during search or during backtrace/mapping
        logger.debug("[CUDA-COMPILE] Compiling validate_parents kernel for debugging")
        self.validate_parents_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void validate_parents(
            const int K,
            const int max_roi_size,
            const int* __restrict__ indptr,
            const int indptr_stride,
            const int* __restrict__ indices,
            const int indices_stride,
            const int* __restrict__ parent,
            const int parent_stride,  // CRITICAL FIX: Need stride for parent array too!
            int* __restrict__ bad_counts
        ){
            int roi = blockIdx.x;
            int u = blockIdx.y * blockDim.x + threadIdx.x;
            if (roi >= K || u >= max_roi_size) return;

            int parent_base = roi * parent_stride;  // Use parent_stride, not max_roi_size!
            int p = parent[parent_base + u];
            if (p < 0 || p >= max_roi_size) return;  // -1 or invalid parent

            // Check if u is in p's neighbor list
            int prow = roi * indptr_stride + p;
            int istart = indptr[prow];
            int iend = indptr[prow + 1];
            bool ok = false;
            for (int e = istart; e < iend; ++e) {
                int v = indices[roi * indices_stride + e];
                if (v == u) {
                    ok = true;
                    break;
                }
            }
            if (!ok) atomicAdd(&bad_counts[roi], 1);
        }
        ''', 'validate_parents')

        # GPU PATH RECONSTRUCTION KERNEL: Backtrace on GPU to avoid 256 MB CPU transfers
        # Each thread reconstructs one path by following parent pointers on GPU
        # Outputs: compact path arrays + path lengths (sparse transfer)
        # BACKTRACE KERNEL COMPILATION (should see this log EXACTLY ONCE)
        logger.debug("[CUDA-COMPILE] Compiling backtrace_paths kernel with use_bitmap parameter")
        self.backtrace_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void backtrace_paths(
            const int K,                    // Number of ROIs
            const int max_roi_size,         // Max nodes per ROI
            const int* parent,              // (K, parent_stride) parent pointers
            const int parent_stride,        // CRITICAL: Stride for parent array (pool stride!)
            const float* dist,              // (K, dist_stride) distances
            const int dist_stride,          // CRITICAL: Stride for dist array (pool stride!)
            const int* sinks,               // (K,) sink nodes
            int* paths_out,                 // (K, max_path_len) output paths
            int* path_lengths,              // (K,) output path lengths
            const int max_path_len,         // Maximum path length (safety limit)
            // FIX-BITMAP-BUG: Add bitmap validation to backtrace
            const unsigned int* roi_bitmap, // (K, bitmap_words) per ROI - validate each path node!
            const int bitmap_words,         // Words per ROI bitmap
            const int use_bitmap,           // 1 = enforce bitmap, 0 = bbox-only (skip validation)
            // ATOMIC PARENT KEYS: Read from 64-bit keys when enabled
            const unsigned long long* best_key, // (K, key_stride) atomic keys
            const int key_stride,           // Stride for best_key array
            const int use_atomic_parent_keys // 1 = read from best_key, 0 = read from parent
        ) {
            int roi_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (roi_idx >= K) return;

            const int sink = sinks[roi_idx];
            const int parent_off = roi_idx * parent_stride;  // Use parent_stride!
            const int dist_off = roi_idx * dist_stride;  // Use dist_stride!
            const int path_off = roi_idx * max_path_len;

            // Check if path exists (distance is finite)
            if (isinf(dist[dist_off + sink]) || sink < 0 || sink >= max_roi_size) {
                path_lengths[roi_idx] = 0;  // No path found
                return;
            }

            // Backtrace from sink to source
            int curr = sink;
            int path_len = 0;
            int visited[4096];  // Stack-allocated cycle detection (max 4096 nodes per path)
            int visited_count = 0;

            while (curr != -1 && path_len < max_path_len) {
                // Cycle detection (check last 32 nodes for performance)
                bool is_cycle = false;
                int check_start = (visited_count > 32) ? (visited_count - 32) : 0;
                for (int i = check_start; i < visited_count; i++) {
                    if (visited[i] == curr) {
                        is_cycle = true;
                        break;
                    }
                }

                if (is_cycle) {
                    path_lengths[roi_idx] = -1;  // Indicate cycle error
                    return;
                }

                // FIX-BITMAP-BUG: Validate current node is in bitmap before adding to path
                // Only validate if use_bitmap=1, skip validation if use_bitmap=0
                if (use_bitmap) {
                    const int curr_word = curr >> 5;
                    const int curr_bit = curr & 31;
                    const int bitmap_off = roi_idx * bitmap_words;
                    if (curr_word >= bitmap_words) {
                        path_lengths[roi_idx] = -2;  // Indicate bitmap bounds error
                        return;
                    }
                    const unsigned int curr_in_bitmap = (roi_bitmap[bitmap_off + curr_word] >> curr_bit) & 1;
                    if (curr_in_bitmap == 0) {
                        path_lengths[roi_idx] = -2;  // Indicate bitmap validation error (node not in ROI)
                        return;
                    }
                }

                // Add to path (stored in reverse order)
                paths_out[path_off + path_len] = curr;
                path_len++;

                // Track visited (up to 4096 nodes)
                if (visited_count < 4096) {
                    visited[visited_count++] = curr;
                }

                // Follow parent pointer (read from atomic key or legacy array)
                if (use_atomic_parent_keys) {
                    // ATOMIC MODE: Extract parent from low 32 bits of best_key
                    const unsigned long long curr_key = best_key[roi_idx * key_stride + curr];
                    curr = (int)(curr_key & 0xFFFFFFFFu);  // Parent is lower 32 bits
                } else {
                    // LEGACY MODE: Read from parent array
                    curr = parent[parent_off + curr];  // Use parent_off, not dist_off!
                }
            }

            // Store final path length
            path_lengths[roi_idx] = path_len;

            // Reverse path in-place (convert from sink->source to source->sink)
            for (int i = 0; i < path_len / 2; i++) {
                int tmp = paths_out[path_off + i];
                paths_out[path_off + i] = paths_out[path_off + path_len - 1 - i];
                paths_out[path_off + path_len - 1 - i] = tmp;
            }
        }
        ''', 'backtrace_paths')

        logger.debug("[CUDA] Compiled parallel edge relaxation kernel")
        logger.debug("[CUDA] Compiled FULLY PARALLEL wavefront expansion kernel")
        logger.debug("[CUDA] Compiled ACTIVE-LIST kernel (2-3× faster than one-block-per-ROI!)")
        logger.debug("[CUDA] Compiled PROCEDURAL NEIGHBOR kernel (P1-8: ditches CSR, pure arithmetic!)")
        logger.debug("[CUDA] Compiled DELTA-STEPPING bucket assignment kernel (P1-7: replaces Python loop!)")
        logger.debug("[CUDA] Compiled PERSISTENT KERNEL (P1-6: device-side queues, eliminates launch overhead!)")
        logger.debug("[CUDA] Compiled COMPACTION KERNEL (P3: GPU-side frontier compaction, no host sync!)")
        logger.debug("[CUDA] Compiled ACCOUNTANT KERNEL (Phase 5: GPU-side history/present/cost updates!)")
        logger.debug("[CUDA] Compiled GPU BACKTRACE KERNEL (eliminates 256 MB parent/dist CPU transfers!)")

    def _normalize_batch(self, roi_batch):
        """
        Ensure the batch size used by all GPU code is consistent:
          - Clamp to self.K_pool if present
          - Return (K, sliced_roi_batch)
        Never allocate or stride anything until after this runs.
        """
        K_in = len(roi_batch)
        K_pool = getattr(self, "K_pool", None)
        if K_pool is not None and K_in > K_pool:
            K_eff = K_pool
            roi_batch = roi_batch[:K_eff]
        else:
            K_eff = K_in
        return K_eff, roi_batch

    def _cp_array(self, x, dtype=None):
        """Normalize x to a contiguous CuPy array with optional dtype (no copy if already correct)."""
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return x if (dtype is None or x.dtype == dtype) else x.astype(dtype, copy=False)
        return cp.asarray(x, dtype=dtype)

    def find_paths_on_rois(self, roi_batch: List[Tuple], use_bitmap: bool = True) -> List[Optional[List[int]]]:
        """
        Find paths on ROI subgraphs using GPU Near-Far worklist algorithm.

        This is the production GPU implementation with 75-100× speedup over CPU.

        Args:
            roi_batch: List of (roi_source, roi_sink, roi_indptr, roi_indices, roi_weights, roi_size)

        Returns:
            List of paths (local ROI indices), one per ROI
        """
        if not roi_batch:
            return []

        # Normalize all tuples to 13-element format for backward compatibility
        roi_batch = [_normalize_roi_tuple(t) for t in roi_batch]

        K = len(roi_batch)
        mode_desc = "BBOX-ONLY (no bitmap fence)" if not use_bitmap else "BITMAP-FILTERED (L-corridor)"
        logger.debug(f"[CUDA-ROI] Processing {K} ROI subgraphs using GPU Near-Far algorithm ({mode_desc})")

        try:
            # Prepare batched GPU arrays
            logger.debug(f"[DEBUG-GPU] Preparing batch data for {K} ROIs (use_bitmap={use_bitmap})")
            batch_data = self._prepare_batch(roi_batch, use_bitmap=use_bitmap)
            logger.debug(f"[DEBUG-GPU] Batch data prepared, starting Near-Far algorithm")

            # CRITICAL: Use the K that _prepare_batch actually built arrays for
            K_actual = int(batch_data.get('K', len(roi_batch)))
            logger.debug(f"[K-RESYNC-CALLER] K adjusted from {K} -> {K_actual} after _prepare_batch")
            K = K_actual

            # INVARIANT CHECKS: Shared-CSR indexing (user-requested debugging)
            if GPUConfig.DEBUG_INVARIANTS if hasattr(GPUConfig, 'DEBUG_INVARIANTS') else True:
                # Check if using shared CSR (stride=0 indicates broadcast)
                is_shared_csr = (hasattr(batch_data['batch_indptr'], 'strides') and
                                len(batch_data['batch_indptr'].strides) == 2 and
                                batch_data['batch_indptr'].strides[0] == 0)

                if is_shared_csr:
                    logger.debug("[INVARIANT-CHECK] Shared CSR detected (stride=0)")
                    # In shared CSR mode, max_roi_size must equal total graph size
                    N_global = len(batch_data['batch_indptr'][0]) - 1  # indptr shape is (N+1,)
                    max_roi = batch_data['max_roi_size']

                    if max_roi != N_global:
                        logger.error(f"[INVARIANT-FAIL] Shared CSR but max_roi_size={max_roi} != N_global={N_global}")
                        raise AssertionError(f"Shared CSR invariant violated: max_roi_size={max_roi} != N_global={N_global}")
                    else:
                        logger.debug(f"[INVARIANT-OK] Shared CSR: max_roi_size={max_roi} == N_global={N_global}")
                else:
                    logger.debug("[INVARIANT-CHECK] Per-ROI CSR mode (no shared CSR)")

            # Run Near-Far algorithm on GPU
            # CRITICAL: Slice roi_batch to match K so indices align with arrays
            try:
                paths = self._run_near_far(batch_data, K, roi_batch[:K])
                logger.debug(f"[DEBUG-GPU] Near-Far algorithm completed")
            except Exception as near_far_error:
                logger.error(f"[DEBUG-GPU] Error in _run_near_far: {near_far_error}")
                import traceback
                logger.error(f"[DEBUG-GPU] Traceback:\n{traceback.format_exc()}")
                raise

            found = sum(1 for p in paths if p)
            logger.debug(f"[CUDA-ROI] Complete: {found}/{K} paths found using GPU")
            return paths

        except Exception as e:
            logger.warning(f"[CUDA-ROI] GPU pathfinding failed: {e}, skipping (fail fast mode)")
            # Return None for all paths instead of slow CPU fallback
            return [None] * len(roi_batch)

    # ========================================================================
    # NEAR-FAR WORKLIST ALGORITHM - PRODUCTION IMPLEMENTATION
    # ========================================================================

    def _prepare_batch(self, roi_batch: List[Tuple], use_bitmap: bool = True) -> dict:
        """
        Prepare batched GPU arrays for FAST WAVEFRONT algorithm.

        Args:
            roi_batch: List of (src, dst, indptr, indices, weights, size)

        Returns:
            Dictionary with all GPU arrays needed for wavefront expansion
        """
        import numpy as np

        # CRITICAL: Normalize batch BEFORE any allocations or as_strided views
        K, roi_batch = self._normalize_batch(roi_batch)
        logger.debug(f"[PREPARE] preparing batch for {K} ROIs (after normalization)")

        # Check if all nets share the same CSR (full graph routing)
        # Use array length instead of id() - if all have same huge size, it's shared full graph
        first_roi_size = roi_batch[0][5]
        first_indices_len = len(roi_batch[0][3])
        all_share_csr = all(roi[5] == first_roi_size and len(roi[3]) == first_indices_len for roi in roi_batch)
        # Additional check: if roi_size > 1M and all same size, definitely shared full graph
        if all_share_csr and first_roi_size > 1_000_000:
            logger.debug(f"[SHARED-CSR-DETECT] All {K} nets have roi_size={first_roi_size:,} - using shared CSR mode")
        else:
            logger.debug(f"[INDIVIDUAL-CSR-DETECT] Nets have varying sizes - using individual CSR mode")

        if all_share_csr:
            # SHARED CSR MODE: All nets use same graph - allocate CSR once!
            logger.debug(f"[SHARED-CSR] All {K} nets share same CSR - no duplication!")
            shared_indptr = roi_batch[0][2]
            shared_indices = roi_batch[0][3]
            shared_weights = roi_batch[0][4]
            max_roi_size = roi_batch[0][5]
            max_edges = len(shared_indices)

            # CRITICAL FIX: Verify indptr has correct size (max_roi_size + 1)
            # CSR format requires indptr.shape[0] == num_nodes + 1
            if len(shared_indptr) != max_roi_size + 1:
                logger.warning(f"[SHARED-CSR-FIX] indptr size mismatch: len={len(shared_indptr)}, expected={max_roi_size + 1}")
                logger.warning(f"[SHARED-CSR-FIX] This indicates a bug in ROI extraction - fixing by padding/truncating")

                import numpy as np
                if len(shared_indptr) < max_roi_size + 1:
                    # Pad with final value (common CSR pattern)
                    if isinstance(shared_indptr, cp.ndarray):
                        shared_indptr_cpu = shared_indptr.get()
                    else:
                        shared_indptr_cpu = np.asarray(shared_indptr)

                    final_val = shared_indptr_cpu[-1]
                    padding_size = (max_roi_size + 1) - len(shared_indptr_cpu)
                    shared_indptr_cpu = np.concatenate([shared_indptr_cpu, np.full(padding_size, final_val, dtype=shared_indptr_cpu.dtype)])
                    shared_indptr = cp.asarray(shared_indptr_cpu)
                    logger.debug(f"[SHARED-CSR-FIX] Padded indptr from {len(shared_indptr) - padding_size} to {len(shared_indptr)}")
                else:
                    # Truncate to correct size
                    shared_indptr = shared_indptr[:max_roi_size + 1]
                    logger.debug(f"[SHARED-CSR-FIX] Truncated indptr to {len(shared_indptr)}")

            # Transfer CSR to GPU once (not K times!)
            if not isinstance(shared_indptr, cp.ndarray):
                shared_indptr = cp.asarray(shared_indptr)
            if not isinstance(shared_indices, cp.ndarray):
                shared_indices = cp.asarray(shared_indices)
            if not isinstance(shared_weights, cp.ndarray):
                was_cpu = True
                shared_weights_cpu = shared_weights  # Save CPU version for debug
                shared_weights = cp.asarray(shared_weights)
                # DEBUG: Verify infinity survived CPU->GPU transfer
                import numpy as np
                test_edges = [822688, 822689, 822690]  # Test edges around the infinite one
                for edge_idx in test_edges:
                    if edge_idx < len(shared_weights_cpu):
                        cpu_val = shared_weights_cpu[edge_idx]
                        gpu_val = float(shared_weights[edge_idx])
                        if cpu_val > 1e8:  # If CPU had infinity
                            if gpu_val > 1e8:
                                logger.debug(f"[TEST-A1-GPU-XFER] Edge {edge_idx}: CPU={cpu_val:.2e} -> GPU={gpu_val:.2e} ✅")
                            else:
                                logger.error(f"[TEST-A1-GPU-XFER] Edge {edge_idx}: CPU={cpu_val:.2e} -> GPU={gpu_val:.2e} ❌ LOST INFINITY!")

            # Phase 1: STAMP TRICK - Lazy-allocate pools on first use
            if self.dist_val_pool is None:
                # Calculate K_pool dynamically from available GPU memory
                if self.K_pool is None or not self._k_pool_calculated:
                    # Query actual GPU memory
                    free_bytes, total_bytes = cp.cuda.Device().mem_info
                    N_max = 5_000_000  # Maximum node count

                    # Bytes per net (estimate based on current implementation)
                    # Will be optimized further with uint16 stamps + bitsets
                    bytes_per_net = (
                        4 * N_max +      # dist_val float32
                        2 * N_max +      # dist_stamp uint16 (Phase A)
                        4 * N_max +      # parent_val int32
                        2 * N_max +      # parent_stamp uint16 (Phase A)
                        8 * N_max +      # best_key uint64 (NEW: atomic keys)
                        (N_max + 7) // 8 +  # near_bits bitset (Phase B: 1 bit/node)
                        (N_max + 7) // 8    # far_bits bitset (Phase B: 1 bit/node)
                    )

                    shared_overhead = 500 * (1024 ** 2)  # CSR + present/history/cost (~500 MB)
                    safety = 0.7  # Use 70% of free memory

                    K_pool_calculated = max(8, min(256, int((free_bytes - shared_overhead) * safety / bytes_per_net)))

                    self.K_pool = K_pool_calculated
                    self._k_pool_calculated = True

                    logger.debug(f"[MEMORY-AWARE] GPU memory: {free_bytes / 1e9:.2f} GB free, {total_bytes / 1e9:.2f} GB total")
                    logger.debug(f"[MEMORY-AWARE] Calculated K_pool: {self.K_pool} (allows {self.K_pool} nets in parallel)")
                    logger.debug(f"[MEMORY-AWARE] Per-net memory: {bytes_per_net / 1e6:.1f} MB")
                    logger.debug(f"[MEMORY-AWARE] Total pool memory: {self.K_pool * bytes_per_net / 1e9:.2f} GB")
                else:
                    N_max = 5_000_000  # Conservative max nodes

                # CRITICAL: Now that K_pool exists, re-normalize K and roi_batch
                K_old = K
                K, roi_batch = self._normalize_batch(roi_batch)
                if K != K_old:
                    logger.warning(f"[K-RENORMALIZE] After K_pool calculated: K {K_old} -> {K}")

                self.dist_val_pool = cp.full((self.K_pool, N_max), cp.inf, dtype=cp.float32)
                self.dist_stamp_pool = cp.zeros((self.K_pool, N_max), dtype=cp.uint16)  # Phase A: uint16 stamps
                self.parent_val_pool = cp.full((self.K_pool, N_max), -1, dtype=cp.int32)
                self.parent_stamp_pool = cp.zeros((self.K_pool, N_max), dtype=cp.uint16)  # Phase A: uint16 stamps
                # NEW: 64-bit atomic key pool (cycle-proof parent updates)
                INF_KEY = 0x7F800000FFFFFFFF  # float +inf (upper 32) | parent -1 (lower 32)
                self.best_key_pool = cp.full((self.K_pool, N_max), INF_KEY, dtype=cp.uint64)
                # Phase B: Allocate as uint32 words (32 bits per word, properly aligned)
                frontier_words = (N_max + 31) // 32  # Number of 32-bit words needed
                self.near_bits_pool = cp.zeros((self.K_pool, frontier_words), dtype=cp.uint32)  # Phase B: 1 bit/node
                self.far_bits_pool = cp.zeros((self.K_pool, frontier_words), dtype=cp.uint32)   # Phase B: 1 bit/node
                logger.debug(f"[STAMP-POOL] Allocated device pools: K={self.K_pool}, N={N_max}")
                logger.debug(f"[PHASE-A] Using uint16 stamps (16 MB memory savings per net)")
                logger.debug(f"[ATOMIC-KEY] Using 64-bit atomic keys for cycle-proof parent updates")
                frontier_bytes = ((N_max + 31) // 32) * 4  # uint32 words * 4 bytes/word
                logger.debug(f"[PHASE-B] Using bitset frontiers (8× memory savings: {N_max/1e6:.1f} MB -> {frontier_bytes/1e6:.1f} MB per net)")

            # Slice pool instead of allocating new arrays (NO ZEROING!)
            gen = self.current_gen
            dist = self.dist_val_pool[:K, :max_roi_size]
            parent = self.parent_val_pool[:K, :max_roi_size]
            dist_stamps = self.dist_stamp_pool[:K, :max_roi_size]
            parent_stamps = self.parent_stamp_pool[:K, :max_roi_size]
            # Phase B: Slice bitset pools (size = (K, (max_roi_size+31)//32) uint32 words)
            frontier_words = (max_roi_size + 31) // 32
            near_bits = self.near_bits_pool[:K, :frontier_words]
            far_bits = self.far_bits_pool[:K, :frontier_words]

            # FIX: Use as_strided to create zero-copy broadcast view
            # This creates a (K, N) view with stride (0, element_size) - true zero-copy broadcast
            # Avoids cp.broadcast_to() which may copy/corrupt data
            import cupy
            batch_indptr = cupy.lib.stride_tricks.as_strided(
                shared_indptr,
                shape=(K, len(shared_indptr)),
                strides=(0, shared_indptr.itemsize)
            )
            batch_indices = cupy.lib.stride_tricks.as_strided(
                shared_indices,
                shape=(K, len(shared_indices)),
                strides=(0, shared_indices.itemsize)
            )
            batch_weights = cupy.lib.stride_tricks.as_strided(
                shared_weights,
                shape=(K, len(shared_weights)),
                strides=(0, shared_weights.itemsize)
            )

            logger.debug(f"[SHARED-CSR-FIX] Using as_strided for zero-copy broadcast")
            logger.debug(f"[SHARED-CSR-FIX] batch_indptr: shape={batch_indptr.shape}, strides={batch_indptr.strides}")
            logger.debug(f"[SHARED-CSR-FIX] batch_weights: shape={batch_weights.shape}, strides={batch_weights.strides}")

            logger.debug(f"[SHARED-CSR] Memory saved: {K-1} × {max_edges * 8 / 1e9:.1f} GB = {(K-1) * max_edges * 8 / 1e9:.1f} GB")
        else:
            # INDIVIDUAL CSR MODE: Each net has different ROI
            max_roi_size = max(roi[5] for roi in roi_batch)
            max_edges = max(len(roi[3]) if hasattr(roi[3], '__len__') else roi[3].shape[0] for roi in roi_batch)

            logger.debug(f"[INDIVIDUAL-CSR] K={K} nets with different ROIs, max_roi_size={max_roi_size}, max_edges={max_edges}")

            # Allocate separate CSR arrays for each ROI
            batch_indptr = cp.zeros((K, max_roi_size + 1), dtype=cp.int32)
            batch_indices = cp.zeros((K, max_edges), dtype=cp.int32)
            batch_weights = cp.zeros((K, max_edges), dtype=cp.float32)

            # Phase 1: STAMP TRICK - Lazy-allocate pools on first use
            if self.dist_val_pool is None:
                # Calculate K_pool dynamically from available GPU memory
                if self.K_pool is None or not self._k_pool_calculated:
                    # Query actual GPU memory
                    free_bytes, total_bytes = cp.cuda.Device().mem_info
                    N_max = 5_000_000  # Maximum node count

                    # Bytes per net (estimate based on current implementation)
                    # Will be optimized further with uint16 stamps + bitsets
                    bytes_per_net = (
                        4 * N_max +      # dist_val float32
                        2 * N_max +      # dist_stamp uint16 (Phase A)
                        4 * N_max +      # parent_val int32
                        2 * N_max +      # parent_stamp uint16 (Phase A)
                        8 * N_max +      # best_key uint64 (NEW: atomic keys)
                        (N_max + 7) // 8 +  # near_bits bitset (Phase B: 1 bit/node)
                        (N_max + 7) // 8    # far_bits bitset (Phase B: 1 bit/node)
                    )

                    shared_overhead = 500 * (1024 ** 2)  # CSR + present/history/cost (~500 MB)
                    safety = 0.7  # Use 70% of free memory

                    K_pool_calculated = max(8, min(256, int((free_bytes - shared_overhead) * safety / bytes_per_net)))

                    self.K_pool = K_pool_calculated
                    self._k_pool_calculated = True

                    logger.debug(f"[MEMORY-AWARE] GPU memory: {free_bytes / 1e9:.2f} GB free, {total_bytes / 1e9:.2f} GB total")
                    logger.debug(f"[MEMORY-AWARE] Calculated K_pool: {self.K_pool} (allows {self.K_pool} nets in parallel)")
                    logger.debug(f"[MEMORY-AWARE] Per-net memory: {bytes_per_net / 1e6:.1f} MB")
                    logger.debug(f"[MEMORY-AWARE] Total pool memory: {self.K_pool * bytes_per_net / 1e9:.2f} GB")
                else:
                    N_max = 5_000_000  # Conservative max nodes

                # CRITICAL: Now that K_pool exists, re-normalize K and roi_batch
                K_old = K
                K, roi_batch = self._normalize_batch(roi_batch)
                if K != K_old:
                    logger.warning(f"[K-RENORMALIZE] After K_pool calculated: K {K_old} -> {K}")

                self.dist_val_pool = cp.full((self.K_pool, N_max), cp.inf, dtype=cp.float32)
                self.dist_stamp_pool = cp.zeros((self.K_pool, N_max), dtype=cp.uint16)  # Phase A: uint16 stamps
                self.parent_val_pool = cp.full((self.K_pool, N_max), -1, dtype=cp.int32)
                self.parent_stamp_pool = cp.zeros((self.K_pool, N_max), dtype=cp.uint16)  # Phase A: uint16 stamps
                # NEW: 64-bit atomic key pool (cycle-proof parent updates)
                INF_KEY = 0x7F800000FFFFFFFF  # float +inf (upper 32) | parent -1 (lower 32)
                self.best_key_pool = cp.full((self.K_pool, N_max), INF_KEY, dtype=cp.uint64)
                # Phase B: Allocate as uint32 words (32 bits per word, properly aligned)
                frontier_words = (N_max + 31) // 32  # Number of 32-bit words needed
                self.near_bits_pool = cp.zeros((self.K_pool, frontier_words), dtype=cp.uint32)  # Phase B: 1 bit/node
                self.far_bits_pool = cp.zeros((self.K_pool, frontier_words), dtype=cp.uint32)   # Phase B: 1 bit/node
                logger.debug(f"[STAMP-POOL] Allocated device pools: K={self.K_pool}, N={N_max}")
                logger.debug(f"[PHASE-A] Using uint16 stamps (16 MB memory savings per net)")
                logger.debug(f"[ATOMIC-KEY] Using 64-bit atomic keys for cycle-proof parent updates")
                frontier_bytes = ((N_max + 31) // 32) * 4  # uint32 words * 4 bytes/word
                logger.debug(f"[PHASE-B] Using bitset frontiers (8× memory savings: {N_max/1e6:.1f} MB -> {frontier_bytes/1e6:.1f} MB per net)")

            # Slice pool instead of allocating new arrays (NO ZEROING!)
            gen = self.current_gen
            dist = self.dist_val_pool[:K, :max_roi_size]
            parent = self.parent_val_pool[:K, :max_roi_size]
            dist_stamps = self.dist_stamp_pool[:K, :max_roi_size]
            parent_stamps = self.parent_stamp_pool[:K, :max_roi_size]
            # Phase B: Slice bitset pools (size = (K, (max_roi_size+31)//32) uint32 words)
            frontier_words = (max_roi_size + 31) // 32
            near_bits = self.near_bits_pool[:K, :frontier_words]
            far_bits = self.far_bits_pool[:K, :frontier_words]

        threshold = cp.full(K, 0.4, dtype=cp.float32)

        # CRITICAL: Log K vs roi_batch size to catch off-by-one bugs
        logger.debug(f"[PREPARE-BATCH] K={K}, len(roi_batch)={len(roi_batch)}, will process roi_batch[:K] = {K} elements")

        sources = []
        sinks = []

        # Fill arrays from ROI batch
        if not all_share_csr:
            # Only transfer CSR if nets have different ROIs
            # roi_batch is already sliced to K at the top of function
            for i, (src, dst, indptr, indices, weights, roi_size, roi_bitmap, bbox_minx, bbox_maxx, bbox_miny, bbox_maxy, bbox_minz, bbox_maxz) in enumerate(roi_batch):
                # CRITICAL FIX: Verify indptr has correct size (roi_size + 1)
                if len(indptr) != roi_size + 1:
                    logger.warning(f"[INDIVIDUAL-CSR-FIX] ROI {i}: indptr size mismatch: len={len(indptr)}, expected={roi_size + 1}")

                    import numpy as np
                    if len(indptr) < roi_size + 1:
                        # Pad with final value
                        if isinstance(indptr, cp.ndarray):
                            indptr_cpu = indptr.get()
                        else:
                            indptr_cpu = np.asarray(indptr)

                        final_val = indptr_cpu[-1]
                        padding_size = (roi_size + 1) - len(indptr_cpu)
                        indptr_cpu = np.concatenate([indptr_cpu, np.full(padding_size, final_val, dtype=indptr_cpu.dtype)])
                        indptr = cp.asarray(indptr_cpu)
                        logger.debug(f"[INDIVIDUAL-CSR-FIX] ROI {i}: Padded indptr from {len(indptr) - padding_size} to {len(indptr)}")
                    else:
                        # Truncate to correct size
                        indptr = indptr[:roi_size + 1]
                        logger.debug(f"[INDIVIDUAL-CSR-FIX] ROI {i}: Truncated indptr to {len(indptr)}")

                # Convert to GPU if needed
                if not isinstance(indptr, cp.ndarray):
                    indptr = cp.asarray(indptr)
                if not isinstance(indices, cp.ndarray):
                    indices = cp.asarray(indices)
                if not isinstance(weights, cp.ndarray):
                    weights = cp.asarray(weights)

                # Transfer CSR data (with padding)
                batch_indptr[i, :len(indptr)] = indptr
                batch_indices[i, :len(indices)] = indices
                batch_weights[i, :len(weights)] = weights

        # P0-3: Build GOAL coordinate array (K × 3, not max_roi_size × 3!)
        # This eliminates global memory loads for ALL node coordinates
        import numpy as np
        goal_coords_array = np.zeros((K, 3), dtype=np.int32)

        # P0-3: Extract lattice dimensions for procedural coordinate decoding
        Nx, Ny, Nz = 0, 0, 0
        if all_share_csr and self.lattice:
            Nx = self.lattice.x_steps
            Ny = self.lattice.y_steps
            Nz = self.lattice.layers
            logger.debug(f"[P0-3] Using procedural coordinates: Nx={Nx}, Ny={Ny}, Nz={Nz}")
            logger.debug(f"[P0-3] Memory saved: {max_roi_size * 3 * 4 / 1e6:.1f} MB (no node_coords array!)")

        goal_nodes_array = cp.zeros(K, dtype=cp.int32)

        # Phase 4: ROI bounding boxes (per net) for device-side ROI gating
        roi_minx = cp.zeros(K, dtype=cp.int32)
        roi_maxx = cp.zeros(K, dtype=cp.int32)
        roi_miny = cp.zeros(K, dtype=cp.int32)
        roi_maxy = cp.zeros(K, dtype=cp.int32)
        roi_minz = cp.zeros(K, dtype=cp.int32)
        roi_maxz = cp.zeros(K, dtype=cp.int32)

        # FIX-7: Batch bitmaps (conditional based on use_bitmap flag)
        if use_bitmap and roi_batch[0][6] is not None:
            first_bitmap = roi_batch[0][6]
            bitmap_words = len(first_bitmap)

            # FIX-7: STRICT bitmap size validation - all bitmaps MUST be same size
            for i, roi_tuple in enumerate(roi_batch):
                roi_bitmap = roi_tuple[6]
                if roi_bitmap is not None and len(roi_bitmap) != bitmap_words:
                    raise ValueError(
                        f"[FIX-7-BITMAP] ROI {i}: bitmap size mismatch! "
                        f"Expected {bitmap_words} words, got {len(roi_bitmap)} words. "
                        f"All ROIs must use the same graph size for batched processing."
                    )

            batch_bitmaps = cp.zeros((K, bitmap_words), dtype=cp.uint32)
            logger.debug(f"[FIX-7-BITMAP] Validated {K} bitmaps, all have {bitmap_words} words ({bitmap_words*4/1e6:.2f} MB each)")
        else:
            # No bitmap filtering - bbox-only mode (iteration 1 wide search)
            # Create dummy all-ones bitmap (all nodes allowed) to avoid None errors in kernel launches
            # Need to cover ALL nodes in the full graph!
            if all_share_csr and self.lattice:
                full_graph_size = self.lattice.x_steps * self.lattice.y_steps * self.lattice.layers
            else:
                # Fallback: estimate from max_roi_size
                full_graph_size = max_roi_size
            bitmap_words = (full_graph_size + 31) // 32
            batch_bitmaps = cp.ones((K, bitmap_words), dtype=cp.uint32) * 0xFFFFFFFF
            logger.debug(f"[BBOX-ONLY] Bitmap filtering DISABLED (using all-ones bitmap covering {full_graph_size} nodes = {bitmap_words} words)")

        # Initialize sources/sinks for all nets
        # roi_batch is already sliced to K at the top of function
        for i, (src, dst, indptr, indices, weights, roi_size, roi_bitmap, bbox_minx, bbox_maxx, bbox_miny, bbox_maxy, bbox_minz, bbox_maxz) in enumerate(roi_batch):
            # Store goal node for A* heuristic
            goal_nodes_array[i] = dst

            # P0-3: Decode goal coordinates and store in small array
            if self.lattice and all_share_csr:
                gx, gy, gz = self.lattice.idx_to_coord(dst)
                goal_coords_array[i] = [gx, gy, gz]

                # Phase 4: Compute ROI bounding box from src/dst with margin
                sx, sy, sz = self.lattice.idx_to_coord(src)
                dx, dy, dz = self.lattice.idx_to_coord(dst)

                # DISABLED: ROI bounding too restrictive for long detours
                # margin = 50  # Grid cells (configurable)
                # roi_minx[i] = max(0, min(sx, dx) - margin)
                # roi_maxx[i] = min(Nx - 1, max(sx, dx) + margin)
                # roi_miny[i] = max(0, min(sy, dy) - margin)
                # roi_maxy[i] = min(Ny - 1, max(sy, dy) + margin)
                # Use full board bounds (no ROI restriction)
                roi_minx[i] = 0
                roi_maxx[i] = Nx - 1
                roi_miny[i] = 0
                roi_maxy[i] = Ny - 1
                roi_minz[i] = 0  # All layers
                roi_maxz[i] = Nz - 1
            else:
                # Without lattice, use full space (no ROI gating)
                roi_minx[i] = 0
                roi_maxx[i] = 999999
                roi_miny[i] = 0
                roi_maxy[i] = 999999
                roi_minz[i] = 0
                roi_maxz[i] = 999999

            # FIX-7: Copy bitmap into batched array (only if use_bitmap=True)
            if use_bitmap and batch_bitmaps is not None and roi_bitmap is not None:
                if len(roi_bitmap) == bitmap_words:
                    batch_bitmaps[i] = roi_bitmap
                else:
                    logger.warning(f"[FIX-7-BITMAP] ROI {i}: bitmap size mismatch ({len(roi_bitmap)} vs {bitmap_words})")

            # CSR VALIDATION: Verify src has edges after transfer (DEBUG ONLY)
            if src < len(indptr) - 1:
                src_edge_start = int(batch_indptr[i, src])
                src_edge_end = int(batch_indptr[i, src + 1])
                num_src_neighbors = src_edge_end - src_edge_start
                logger.debug(f"[CSR-VALIDATION] ROI {i}: src={src} has {num_src_neighbors} neighbors in CSR (edges {src_edge_start} to {src_edge_end})")
                if num_src_neighbors > 0 and src_edge_end <= len(indices):
                    # Sample first neighbor
                    first_neighbor = int(batch_indices[i, src_edge_start])
                    first_weight = float(batch_weights[i, src_edge_start])
                    logger.debug(f"[CSR-VALIDATION] ROI {i}: src={src} -> neighbor[0]={first_neighbor}, weight={first_weight:.3f}")
            else:
                logger.warning(f"[CSR-VALIDATION] ROI {i}: src={src} is out of bounds! indptr length={len(indptr)}")

            # Initialize distance with source at 0
            dist[i, src] = 0.0
            # Phase 1: Initialize source with stamp
            dist_stamps[i, src] = gen
            parent_stamps[i, src] = gen
            # ATOMIC KEY: Initialize source with SRC_KEY (cost=0, parent=-1)
            SRC_KEY = 0x00000000FFFFFFFF  # cost=0 (upper 32) | parent=-1 (lower 32)
            if hasattr(self, 'best_key_pool') and self.best_key_pool is not None:
                self.best_key_pool[i, src] = SRC_KEY
            # Phase B: Set bit in near_bits for source initialization (uint32 word addressing)
            word_idx = src // 32
            bit_pos = src % 32
            near_bits[i, word_idx] = near_bits[i, word_idx] | (1 << bit_pos)

            sources.append(src)
            sinks.append(dst)


        # Convert sources/sinks to CuPy int32 arrays for consistent indexing
        sources = cp.asarray(sources, dtype=cp.int32)
        sinks = cp.asarray(sinks, dtype=cp.int32)
        logger.debug(f"[PREPARE-BATCH] Converted sources/sinks to CuPy arrays: sources.shape={sources.shape}, sinks.shape={sinks.shape}")

        # Phase B: Create legacy near_mask/far_mask views by unpacking bitsets
        # This maintains backward compatibility with legacy delta-stepping code
        # CuPy doesn't support axis parameter, so we ravel, unpack, then reshape
        # Note: unpackbits expects uint8, so view uint32 arrays as uint8 first
        near_mask = cp.unpackbits(near_bits.view(cp.uint8).ravel(), bitorder='little').reshape(K, -1)[:, :max_roi_size].astype(cp.bool_)
        far_mask = cp.unpackbits(far_bits.view(cp.uint8).ravel(), bitorder='little').reshape(K, -1)[:, :max_roi_size].astype(cp.bool_)

        # FIX-BITMAP-BUG: Removed duplicate return statement - was dead code that never executed
        # The actual data_dict is built below after sources/sinks conversion

        # K-consistency invariants check
        # CRITICAL: Convert sources/sinks to CuPy int32 arrays (not Python lists!)
        # Build directly from roi_batch to ensure K consistency
        sources_cp = cp.asarray([roi[0] for roi in roi_batch], dtype=cp.int32)
        sinks_cp = cp.asarray([roi[1] for roi in roi_batch], dtype=cp.int32)
        logger.debug(f"[SOURCES-SINKS] Converted to CuPy: sources.shape={sources_cp.shape}, sinks.shape={sinks_cp.shape}")
        # DEBUG: Check for duplicates
        unique_sources = int(cp.unique(sources_cp).size)
        logger.debug(f"[SOURCES-DEBUG] Unique sources: {unique_sources}/{K} (sample: {cp.asnumpy(sources_cp[:5]).tolist()})")

        data_dict = {
            'K': K,
            'max_roi_size': max_roi_size,
            'max_edges': max_edges,
            'batch_indptr': batch_indptr,
            'batch_indices': batch_indices,
            'batch_weights': batch_weights,
            'dist': dist,
            'parent': parent,
            'near_bits': near_bits,
            'far_bits': far_bits,
            'near_mask': near_mask,
            'far_mask': far_mask,
            'threshold': threshold,
            'sources': sources_cp,  # Now CuPy int32 array
            'sinks': sinks_cp,      # Now CuPy int32 array
            'goal_nodes': goal_nodes_array,
            'Nx': Nx,
            'Ny': Ny,
            'Nz': Nz,
            'goal_coords': cp.asarray(goal_coords_array),
            'use_astar': 1 if (Nx > 0 and Ny > 0 and Nz > 0) else 0,
            'roi_minx': roi_minx,
            'roi_maxx': roi_maxx,
            'roi_miny': roi_miny,
            'roi_maxy': roi_maxy,
            'roi_minz': roi_minz,
            'roi_maxz': roi_maxz,
            # FIX-BITMAP-BUG: Add bitmap arrays to data_dict (were missing, causing backtrace to always validate)
            'roi_bitmaps': batch_bitmaps,
            'bitmap_words': bitmap_words,
            'use_bitmap': use_bitmap,  # Flag to enable/disable bitmap filtering in kernels
            'dist_stamps': dist_stamps,
            'parent_stamps': parent_stamps,
            'generation': gen,
        }

        # === VIA SEGMENT POOLING (GPU Implementation) ===
        # Transfer via segment pooling arrays to GPU if enabled
        # Check if parent object (unified_pathfinder) has via_seg_prefix
        if hasattr(self, 'via_seg_prefix') and self.via_seg_prefix is not None:
            # Transfer via_seg_prefix to GPU (shape: Nx × Ny × segZ)
            via_seg_prefix_gpu = cp.asarray(self.via_seg_prefix, dtype=cp.float32)
            segZ = self._segZ if hasattr(self, '_segZ') else self.via_seg_prefix.shape[2]
            via_segment_weight = float(getattr(self.config, "via_segment_weight", 1.0))

            # Get current presence factor from parent pathfinder
            # This is updated each iteration in unified_pathfinder
            pres_fac = getattr(self, '_pres_fac', 1.0)

            # Add to data dict for GPU kernel
            data_dict['via_seg_prefix_gpu'] = via_seg_prefix_gpu
            data_dict['segZ'] = segZ
            data_dict['via_segment_weight'] = via_segment_weight
            data_dict['pres_fac'] = pres_fac

            logger.debug(f"[GPU-VIA-POOL] Transferred via_seg_prefix to GPU: shape={via_seg_prefix_gpu.shape}, segZ={segZ}, weight={via_segment_weight}, pres_fac={pres_fac}")
        else:
            # Via pooling disabled - pass dummy values
            data_dict['via_seg_prefix_gpu'] = None
            data_dict['segZ'] = 0
            data_dict['via_segment_weight'] = 0.0
            data_dict['pres_fac'] = 1.0
            logger.debug("[GPU-VIA-POOL] Via segment pooling DISABLED (via_seg_prefix not found)")

        # Verify all per-ROI arrays have shape[0] == K
        for key in ['dist', 'parent', 'near_bits', 'far_bits', 'near_mask', 'far_mask',
                    'threshold', 'goal_nodes', 'roi_minx', 'roi_maxx', 'roi_miny', 'roi_maxy',
                    'roi_minz', 'roi_maxz', 'dist_stamps', 'parent_stamps', 'roi_bitmaps']:
            arr = data_dict[key]
            if hasattr(arr, 'shape') and len(arr.shape) >= 1:
                if arr.shape[0] != K:
                    logger.error(f"[K-INVARIANT-FAIL] {key}.shape[0]={arr.shape[0]} != K={K}")
                    raise ValueError(f"K-consistency check failed: {key}.shape[0]={arr.shape[0]} != K={K}")
        
        # Verify sources/sinks have shape[0] == K
        if sources_cp.shape[0] != K:
            logger.error(f"[K-INVARIANT-FAIL] sources.shape[0]={sources_cp.shape[0]} != K={K}")
            raise ValueError(f"K-consistency check failed: sources.shape[0]={sources_cp.shape[0]} != K={K}")
        if sinks_cp.shape[0] != K:
            logger.error(f"[K-INVARIANT-FAIL] sinks.shape[0]={sinks_cp.shape[0]} != K={K}")
            raise ValueError(f"K-consistency check failed: sinks.shape[0]={sinks_cp.shape[0]} != K={K}")
        
        logger.debug(f"[K-INVARIANT-PASS] All per-ROI arrays have shape[0]={K}")

        # Phase 1: Increment generation for next batch (MUST be before return!)
        # Phase A: Wrap generation counter to prevent uint16 overflow (max 65535)
        self.current_gen += 1
        if self.current_gen >= 65535:
            self.current_gen = 1  # Reset to 1 (0 is reserved for "uninitialized")

        return data_dict

    def _run_near_far(self, data: dict, K: int, roi_batch: List[Tuple] = None) -> List[Optional[List[int]]]:
        """
        Execute GPU pathfinding algorithm - routes to either delta-stepping or BFS wavefront.

        NEW (Delta-stepping): Proper bucket-based priority queue expansion
        - Processes nodes in cost order (buckets 0, 1, 2, ...)
        - Maintains correctness of shortest-path guarantees
        - Reduces atomic contention via distance-based bucketing
        - Enabled via GPUConfig.USE_DELTA_STEPPING

        OLD (BFS wavefront): Parallel wavefront expansion
        - Process ENTIRE frontier in parallel (not one node at a time!)
        - Use matrix operations for bulk edge relaxation
        - No bucketing overhead - direct distance propagation
        - 10-50× faster than Near-Far algorithm but ignores cost ordering

        Args:
            data: Batched GPU arrays from _prepare_batch
            K: Number of ROIs
            roi_batch: Original ROI batch (for diagnostics)

        Returns:
            List of paths (local ROI indices)
        """
        import time

        
        # Respect the batch width prepared upstream
        K = int(data.get('K', K))
        # Basic invariants to catch off-by-one before launching kernels
        try:
            n_src = len(data['sources'])
            n_dst = len(data['sinks'])
        except Exception:
            n_src = int(data['sources'].shape[0])
            n_dst = int(data['sinks'].shape[0])
        assert n_src == K and n_dst == K, f"K mismatch: K={K} sources={n_src} sinks={n_dst}"
        logger.debug(f"[_run_near_far] Using K={K}, sources.shape={n_src}, sinks.shape={n_dst}")

# Route to delta-stepping if enabled (proper Delta-stepping with bucket-based priority queue)
        # Temporarily force wavefront for iteration-1 debugging
        use_delta_stepping = False
        
        # TEMPORARILY DISABLED for testing
        if False:  # Disabled delta-stepping for K-consistency testing
            # Get delta value from config (fallback to 0.5mm)
            delta = GPUConfig.DELTA_VALUE if hasattr(GPUConfig, 'DELTA_VALUE') else 0.5
            logger.debug(f"[CUDA-PATHFINDING] Routing to DELTA-STEPPING algorithm (delta={delta:.3f}mm)")
            return self._run_delta_stepping(data, K, delta, roi_batch)

        # Otherwise use BFS wavefront (fast but incorrect cost ordering)
        logger.debug(f"[CUDA-WAVEFRONT] Starting BFS wavefront algorithm for {K} ROIs (WARNING: ignores cost ordering)")

        # Adaptive iteration budget for MASSIVE PARALLEL routing
        # For large batches on full graph, need enough iterations for longest path
        if roi_batch and len(roi_batch) > 0:
            roi_size = roi_batch[0][5]
            batch_size = len(roi_batch)

            if roi_size > 1_000_000:  # Full graph
                # For massive parallel batches: budget for worst-case path
                # Board diagonal ~600 steps, increased to 2000 for better convergence
                max_iterations = 2000
                logger.debug(f"[MASSIVE-PARALLEL] Routing {batch_size} nets on full graph with {max_iterations} iterations")
            else:
                # ROIs: scale with size
                max_iterations = min(4096, roi_size // 100 + 500)
        else:
            max_iterations = 2000
        start_time = time.perf_counter()

        # DIAGNOSTIC: Check if destinations are reachable
        logger.debug(f"[DEBUG-GPU] Validating ROI sources and destinations")
        invalid_rois = []
        for roi_idx in range(K):
            src = int(data['sources'][roi_idx].item())
            dst = int(data['sinks'][roi_idx].item())

            # Get actual ROI size for this ROI (not padded size)
            if roi_batch and roi_idx < len(roi_batch):
                actual_roi_size = roi_batch[roi_idx][5]  # Size is 6th element
            else:
                actual_roi_size = data['max_roi_size']

            # Validate src/dst in range
            if src < 0 or src >= actual_roi_size:
                logger.error(f"[CUDA-WAVEFRONT] ROI {roi_idx}: INVALID SOURCE {src} (actual_size={actual_roi_size})")
                invalid_rois.append(roi_idx)

            if dst < 0 or dst >= actual_roi_size:
                logger.error(f"[CUDA-WAVEFRONT] ROI {roi_idx}: INVALID SINK {dst} (actual_size={actual_roi_size})")
                invalid_rois.append(roi_idx)

            # Check if source has any edges
            if src >= 0 and src < actual_roi_size:
                src_start = int(data['batch_indptr'][roi_idx, src])
                src_end = int(data['batch_indptr'][roi_idx, src + 1])
                if src_start == src_end:
                    logger.warning(f"[CUDA-WAVEFRONT] ROI {roi_idx}: Source node {src} has NO edges!")

        if invalid_rois:
            logger.error(f"[CUDA-WAVEFRONT] Aborting - {len(invalid_rois)} ROI(s) have invalid src/dst")
            return [None] * K

        # P0-4: BIT-PACKED FRONTIER - 8× memory reduction!
        # Instead of uint8 (1 byte per node), use uint32 bitset (1 bit per node)
        # Memory: K × max_roi_size bytes -> K × (max_roi_size/32) uint32 words = 8× smaller
        max_roi_size = data['max_roi_size']
        frontier_words = (max_roi_size + 31) // 32  # Round up to cover all nodes
        frontier = cp.zeros((K, frontier_words), dtype=cp.uint32)

        # Initialize source nodes in bitset
        for roi_idx in range(K):
            src = int(data['sources'][roi_idx].item())
            word_idx = src // 32
            bit_pos = src % 32
            frontier[roi_idx, word_idx] = cp.uint32(1) << bit_pos

        logger.debug(f"[BIT-FRONTIER] Memory: {K}×{max_roi_size} uint8 ({K*max_roi_size/1e6:.1f}MB) -> "
                   f"{K}×{frontier_words} uint32 ({K*frontier_words*4/1e6:.1f}MB) = {8.0:.1f}× reduction")

        # GUARDED DIAGNOSTICS: Only when debugging
        if DEBUG_VERBOSE_GPU:
            # DIAGNOSTIC: Verify frontier bits are actually set
            frontier_check = int(cp.count_nonzero(frontier))
            logger.debug(f"[FRONTIER-INIT] After initialization: {frontier_check} non-zero words (expected ~{K})")

            # Check first few ROIs have their source bit set
            for roi_idx in range(min(3, K)):
                src = int(data['sources'][roi_idx].item())
                word_idx = src // 32
                bit_pos = src % 32
                word_val = int(frontier[roi_idx, word_idx])
                bit_set = (word_val >> bit_pos) & 1
                logger.debug(f"[FRONTIER-INIT] ROI {roi_idx}: src={src}, word={word_idx}, bit={bit_pos}, word_val={word_val:08x}, bit_set={bit_set}")

            # DIAGNOSTIC: Check initial state
            for roi_idx in range(min(3, K)):  # Check first 3 ROIs
                src = int(data['sources'][roi_idx].item())
                sink = int(data['sinks'][roi_idx].item())
                src_dist = float(data['dist'][roi_idx, src])
                sink_dist = float(data['dist'][roi_idx, sink])
                logger.debug(f"[CUDA-WAVEFRONT] ROI {roi_idx}: src={src} (dist={src_dist}), "
                           f"sink={sink} (dist={sink_dist})")

        logger.debug(f"[CUDA-WAVEFRONT] Starting parallel wavefront expansion (max {max_iterations} iterations)")

        # P1-6: PERSISTENT KERNEL OPTION (experimental)
        # Hardcoded via GPUConfig.USE_PERSISTENT_KERNEL (was USE_PERSISTENT_KERNEL env var)
        use_persistent = GPUConfig.USE_PERSISTENT_KERNEL if hasattr(GPUConfig, 'USE_PERSISTENT_KERNEL') else False

        if use_persistent:
            logger.debug("[P1-6] PERSISTENT KERNEL enabled - attempting single-launch execution")
            try:
                iters = self._run_persistent_kernel(data, K, frontier)
                if iters >= 0:
                    logger.debug(f"[P1-6] PERSISTENT KERNEL succeeded in {iters} iterations")
                    # Continue to path reconstruction below (skip iterative loop)
                    elapsed = time.perf_counter() - start_time
                    logger.debug(f"[CUDA-WAVEFRONT] GPU pathfinding complete: {elapsed:.4f}s total")
                    # Jump to path reconstruction
                    return self._reconstruct_paths(data, K)
                else:
                    logger.warning("[P1-6] PERSISTENT KERNEL failed - falling back to iterative")
            except Exception as e:
                logger.error(f"[P1-6] PERSISTENT KERNEL error: {e}")
                logger.warning("[P1-6] Falling back to iterative kernel")

        for iteration in range(max_iterations):
            # FIX: Check for empty frontier (count non-zero words, not sum uint32 values!)
            if int(cp.count_nonzero(frontier)) == 0:
                logger.debug(f"[CUDA-WAVEFRONT] Terminated: no active frontiers")
                break

            # FAST WAVEFRONT EXPANSION - Process entire frontier in parallel!
            nodes_expanded = self._expand_wavefront_parallel(data, K, frontier)

            # GUARDED DIAGNOSTICS: Lightweight logging (only when debugging)
            if DEBUG_VERBOSE_GPU and (iteration % 50 == 0 or iteration < 3):
                # FIX: Use fancy indexing for single GPU->CPU transfer instead of K transfers
                # Old: for roi_idx in range(K): sink_dist = float(data['dist'][roi_idx, sink])
                # New: Single vectorized operation gets all sink distances at once
                sink_dists_gpu = data['dist'][cp.arange(K), data['sinks']]  # GPU operation
                sink_dists = sink_dists_gpu.get()  # Single transfer: K values at once
                min_sink_dist = float(cp.min(sink_dists_gpu).get())  # Compute min on GPU
                reached_count = int(cp.sum(sink_dists_gpu < 1e9).get())  # Count on GPU

                logger.debug(f"[CUDA-WAVEFRONT] Iteration {iteration}: {reached_count}/{K} sinks reached, "
                          f"min_sink_dist={min_sink_dist:.2f}")

            # SPEEDUP: Disable expensive diagnostic logging in hot loop
            # This .get() call copies 4M+ nodes from GPU->CPU every 50 iterations!
            # Comment out for production, re-enable only for debugging
            if False:  # Disabled for speedup - was causing 1.5-2× slowdown
                pass  # Original Test D1 logging removed

            # FIX: Unambiguous early termination check (vectorized for performance)
            # Use fancy indexing to get all sink distances in one GPU operation
            sink_dists_term = data['dist'][cp.arange(K), data['sinks']]
            sinks_reached_count = int(cp.sum(sink_dists_term < 1e9).get())  # Single GPU->CPU transfer

            # SPEEDUP: ε-optimal termination (anytime A* with slack)
            # With A* enabled, first path is near-optimal. Allow small exploration for alternatives.
            # Strategy: Minimal afterglow (was 64+64=128 iters, now 32+16=48 iters)
            MIN_ITERS = 32                      # Minimum iterations (reduced from 64)
            EXTRA_ITERS_AFTER_ALL_SINKS = 16    # Small afterglow (reduced from 64) - A* finds good paths fast!

            if not hasattr(self, '_first_all_sinks_iter'):
                self._first_all_sinks_iter = {}

            # Track when each batch first reaches all sinks
            batch_id = id(data)
            if sinks_reached_count == K and batch_id not in self._first_all_sinks_iter:
                self._first_all_sinks_iter[batch_id] = iteration
                logger.debug(f"[STABILIZATION] All sinks reached at iteration {iteration}, "
                           f"continuing for {EXTRA_ITERS_AFTER_ALL_SINKS} more iterations to find alternatives...")

            # Balanced termination: respect both MIN_ITERS and EXTRA_ITERS
            if sinks_reached_count == K:
                iters_since_all_sinks = iteration - self._first_all_sinks_iter[batch_id]

                # Optional: check if frontier is empty (early convergence signal)
                frontier_empty = int(cp.count_nonzero(frontier)) == 0

                if iteration >= MIN_ITERS and iters_since_all_sinks >= EXTRA_ITERS_AFTER_ALL_SINKS:
                    logger.debug(f"[STABILIZATION] Terminating: iteration {iteration} >= MIN_ITERS and "
                               f"{iters_since_all_sinks} >= EXTRA_ITERS after sinks")
                    break

                if frontier_empty and iteration >= MIN_ITERS:
                    logger.debug(f"[STABILIZATION] Early termination: frontier empty at iteration {iteration}")
                    break

            # Periodic logging (reduced frequency for speedup)
            if DEBUG_VERBOSE_GPU and (iteration % 50 == 0 or iteration < 3):
                # FIX: Count ROIs with any set bits (not sum of uint32 values!)
                active_rois = int(cp.count_nonzero(cp.count_nonzero(frontier, axis=1)))
                # Note: nodes_expanded already comes from compaction so it's correct
                logger.debug(f"[CUDA-WAVEFRONT] Iteration {iteration}: {active_rois}/{K} ROIs active, "
                          f"expanded={nodes_expanded}")

            # Progress check: warn if taking too long
            if iteration >= 200:
                logger.warning(f"[CUDA-WAVEFRONT] Iteration {iteration}: algorithm taking longer than expected")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"[CUDA-WAVEFRONT] Complete in {iteration+1} iterations, {elapsed_ms:.1f}ms "
                   f"({elapsed_ms/(iteration+1):.2f}ms/iter)")

        # Reconstruct paths
        paths = self._reconstruct_paths(data, K)
        found = sum(1 for p in paths if p)
        logger.debug(f"[CUDA-WAVEFRONT] Paths found: {found}/{K} ({100*found/K:.1f}% success rate)")

        # VALIDATION: Log path statistics
        if found > 0:
            path_lengths = [len(p) for p in paths if p]
            avg_len = sum(path_lengths) / len(path_lengths)
            max_len = max(path_lengths)
            min_len = min(path_lengths)
            logger.debug(f"[CUDA-WAVEFRONT] Path stats: avg={avg_len:.1f}, min={min_len}, max={max_len} nodes")

            # DIAGNOSTIC: Check X-coordinate exploration in first 3 paths
            if self.lattice and K >= 3:
                for roi_idx in range(min(3, K)):
                    if paths[roi_idx]:
                        path = paths[roi_idx]
                        x_coords = [self.lattice.idx_to_coord(node_idx)[0] for node_idx in path[:20]]  # First 20 nodes
                        x_min, x_max = min(x_coords), max(x_coords)
                        x_range = x_max - x_min
                        logger.debug(f"[PATH-DIAG] ROI {roi_idx}: X-range {x_min}-{x_max} (span={x_range}), len={len(path)}")

        return paths

    def _expand_wavefront_parallel(self, data: dict, K: int, frontier: cp.ndarray) -> int:
        """
        FRONTIER-COMPACTED WAVEFRONT EXPANSION - Only processes active nodes!

        SPEEDUP: Uses sparse frontier lists instead of checking all 4.2M nodes.
        - Compact frontier mask into list of active indices (stream compaction)
        - Process only ~1000 active nodes instead of 4.2M total nodes
        - Expected: 100-1000× fewer memory accesses

        This is THE key optimization for GPU SSSP on sparse frontiers.

        Args:
            data: Batched GPU arrays
            K: Number of ROIs
            frontier: (K, max_roi_size) boolean mask of active frontier nodes

        Returns:
            Number of nodes expanded
        """
        # Check if any work to do (count set BITS, not uint32 values!)
        # frontier is bit-packed uint32, need to count population (number of 1-bits)
        # Quick estimate: count non-zero words as proxy
        frontier_any_set = int(cp.count_nonzero(frontier))
        if frontier_any_set == 0:
            return 0

        # ═══════════════════════════════════════════════════════════════════════
        # FRONTIER COMPACTION: Convert sparse mask -> dense list (100-1000× speedup!)
        # ═══════════════════════════════════════════════════════════════════════
        # P0-5: Time compaction step
        compact_start = cp.cuda.Event()
        compact_end = cp.cuda.Event()
        compact_start.record()

        # P0-4: Convert uint32 bit-packed frontier to uint8 bitset for compaction
        # frontier is (K, frontier_words) uint32 - convert to (K, frontier_words*4) uint8 bitset
        max_roi_size = data['max_roi_size']
        frontier_words = frontier.shape[1]

        # Phase B: View as uint8 bitset (each uint32 = 4 bytes = 32 bits)
        frontier_bytes = frontier.view(cp.uint8)  # (K, frontier_words*4) uint8 bitset

        # P3: Single-pass GPU-side compaction (replaces Python cp.nonzero!)
        # Device-side compaction eliminates host<->device sync
        # Phase B: Compaction now works directly on bitsets (no unpacking needed!)
        use_gpu_compaction = GPUConfig.USE_GPU_COMPACTION if hasattr(GPUConfig, 'USE_GPU_COMPACTION') else True

        if use_gpu_compaction:
            # GPU compaction path (Phase 3 + Phase B optimization)
            # Phase B: Pass bitset directly to compaction kernel (8× less memory!)
            mask_size = K * max_roi_size
            out_indices = cp.zeros(mask_size, dtype=cp.int32)
            out_count = cp.zeros(1, dtype=cp.int32)

            # Launch compaction kernel with bitset input
            threads = 256
            blocks = (mask_size + threads - 1) // threads
            self.compact_kernel((blocks,), (threads,), (
                frontier_bytes.ravel(),  # Phase B: Pass bitset directly
                mask_size,
                out_indices,
                out_count
            ))

            # Get total active count
            total_active = int(out_count[0])
            flat_idx = out_indices[:total_active]
        else:
            # BASELINE: CPU compaction using cp.nonzero (needs unpacking)
            logger.debug("[BASELINE] Using cp.nonzero for compaction (USE_GPU_COMPACTION=0)")
            # Unpack bitset for cp.nonzero (CuPy doesn't support axis, use ravel+reshape)
            frontier_mask = cp.unpackbits(frontier_bytes.ravel(), bitorder='little')
            frontier_mask = frontier_mask.reshape(K, -1)[:, :max_roi_size]
            flat_idx = cp.nonzero(frontier_mask.ravel())[0]
            total_active = int(flat_idx.size)

        compact_end.record()
        compact_end.synchronize()
        compact_ms = cp.cuda.get_elapsed_time(compact_start, compact_end)

        if total_active == 0:
            return 0

        # FIX: Use frontier bit stride (frontier_words * 32), not max_roi_size!
        # frontier is (K, frontier_words) where frontier_words = (max_roi_size+31)//32
        # So each ROI has frontier_words*32 bits allocated (slightly more than max_roi_size)
        bits_per_roi = frontier_words * 32
        roi_ids = (flat_idx // bits_per_roi).astype(cp.int32)
        node_ids = (flat_idx - (roi_ids * bits_per_roi)).astype(cp.int32)

        # GUARDED DIAGNOSTICS: Only when debugging (saves time in hot path)
        if DEBUG_VERBOSE_GPU:
            # Sanity check: node_ids should all be < max_roi_size
            if cp.any(node_ids >= max_roi_size):
                invalid_count = int(cp.sum(node_ids >= max_roi_size))
                logger.warning(f"[COMPACTION-BUG] {invalid_count} nodes have node_id >= max_roi_size!")

            # Add sanity logging after compaction
            n_rois = int(cp.unique(roi_ids).size)
            logger.debug(f"[ACTIVE-SET] total_active={total_active}, unique_rois={n_rois}")
            if n_rois < 10:
                heads = cp.asnumpy(cp.stack([roi_ids[:16], node_ids[:16]], axis=1)) if total_active >= 16 else cp.asnumpy(cp.stack([roi_ids, node_ids], axis=1))
                logger.debug(f"[ACTIVE-HEAD] (roi,node)[:16] = {heads.tolist()}")

            # Check compacted indices
            if total_active > 0:
                logger.debug(f"[COMPACTION] total_active={total_active}, first 3 nodes:")
                for i in range(min(3, total_active)):
                    roi = int(roi_ids[i])
                    node = int(node_ids[i])
                    src_expected = int(data['sources'][roi].item()) if roi < len(data['sources']) else -1
                    logger.debug(f"[COMPACTION]   [{i}] roi={roi}, node={node} (expected src={src_expected})")

        # Use compacted kernel for ANY sparse frontier
        # Compaction overhead is negligible compared to scanning 4.2M nodes
        # FIX: Use total_active (actual bit count) instead of uint32 sum for sparsity
        frontier_count = total_active  # This is the REAL number of active nodes
        sparsity = frontier_count / (K * max_roi_size)
        use_compaction = sparsity < 0.5  # Use compaction if <50% active (aggressive)

        if use_compaction and total_active > 0:
            # COMPACTED PATH: Only process active nodes (100-1000× faster!)
            return self._expand_wavefront_compacted(data, K, roi_ids, node_ids, frontier, sparsity, compact_ms)
        else:
            # FULL SCAN PATH: Frontier is dense (>50% nodes active)
            return self._expand_wavefront_full_scan(data, K, frontier)

    def _expand_wavefront_compacted(self, data: dict, K: int, roi_ids: cp.ndarray,
                                   node_ids: cp.ndarray, old_frontier: cp.ndarray, sparsity: float = 0.0,
                                   compact_ms: float = 0.0) -> int:
        """
        Expand wavefront using ACTIVE-LIST kernel (2-3× faster than one-block-per-ROI!)

        Only processes active nodes (~1000) instead of ALL nodes (4.2M).
        Uses global thread indexing for better GPU occupancy and load balancing.

        Args:
            data: Batched GPU arrays
            K: Number of ROIs
            roi_ids: (total_active,) ROI index for each active node
            node_ids: (total_active,) Node index within ROI for each active node
            old_frontier: (K, max_roi_size) frontier mask to update
            sparsity: Frontier sparsity (for logging)
            compact_ms: Time spent on compaction (ms)
        """
        # Get total_active directly from roi_ids size
        total_active = int(roi_ids.size)

        # Prepare round-robin and jitter parameters
        current_iteration = getattr(self, 'current_iteration', 1)
        pref_layers_gpu, src_x_coords_gpu, rr_alpha, window_cols, jitter_eps = self._prepare_roundrobin_params(
            [], data, current_iteration
        )

        logger.debug(f"[RR-WAVEFRONT] iteration={current_iteration}, rr_alpha={float(rr_alpha)}, jitter={float(jitter_eps)}")

        # P0-5: CUDA Event instrumentation
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        # Log compaction benefit (only first time)
        if not hasattr(self, '_compaction_logged'):
            logger.debug(f"[ACTIVE-LIST] Processing {total_active} active nodes (vs {K * data['max_roi_size']:,} total)")
            logger.debug(f"[ACTIVE-LIST] Sparsity={100*sparsity:.3f}% -> {1/sparsity:.0f}× fewer memory accesses!")
            logger.debug(f"[ACTIVE-LIST] Launching over {total_active} items (not {K} blocks) for better occupancy")
            self._compaction_logged = True

        # Allocate new frontier
        new_frontier = cp.zeros_like(old_frontier)

        # NOTE: best_key_pool initialization removed - only needed when use_atomic_parent_keys=True

        # Get CSR arrays and strides
        indptr_arr = data['batch_indptr']
        indices_arr = data['batch_indices']
        weights_arr = data['batch_weights']

        # Detect strides - check for stride=0 (shared CSR)
        if hasattr(indptr_arr, 'strides') and len(indptr_arr.strides) == 2 and indptr_arr.strides[0] == 0:
            # Stride 0 = shared CSR (zero-copy broadcast)
            indptr_stride = 0
            indices_stride = 0
            weights_stride = 0
        else:
            # Per-ROI arrays
            indptr_stride = indptr_arr.shape[1] if len(indptr_arr.shape) > 1 else data['max_roi_size'] + 1
            indices_stride = indices_arr.shape[1] if len(indices_arr.shape) > 1 else data['max_edges']
            weights_stride = weights_arr.shape[1] if len(weights_arr.shape) > 1 else data['max_edges']

        # Launch active-list kernel: grid launches over total_active items (not K blocks!)
        # PERF: RTX 4090 benefits from 512 threads/block for better SM occupancy
        block_size = 512  # Optimized for Ada Lovelace architecture (was 256)
        grid_size = (total_active + block_size - 1) // block_size

        # P0-4: Include frontier_words for bit-packed frontier
        frontier_words = old_frontier.shape[1]

        # Get pool stride and pass sliced pool views (matching persistent kernel pattern)
        K_batch = K
        max_roi_size = data['max_roi_size']
        pool_stride = self.dist_val_pool.shape[1]  # N_max


        # Get use_bitmap flag from data dict
        use_bitmap_flag = 1 if data.get('use_bitmap', False) else 0  # Default FALSE for iter-1 compatibility
        node_penalty_arr = data.get('node_penalty', self._zero_node_penalty)
        node_penalty_stride = int(data.get('node_penalty_stride', 0))
        use_node_penalty_flag = 1 if data.get('use_node_penalty', False) else 0

        # INSTRUMENTATION: Allocate counter for blocked neighbors
        blocked_counter = cp.zeros(1, dtype=cp.int32)

        args = (
            total_active,
            max_roi_size,
            frontier_words,        # P0-4: Number of uint32 words per ROI
            roi_ids,
            node_ids,
            indptr_arr,
            indices_arr,
            weights_arr,
            indptr_stride,
            indices_stride,
            weights_stride,
            weights_arr,           # total_cost = same as weights (already negotiated)
            weights_stride,        # total_cost_stride
            node_penalty_arr,
            node_penalty_stride,
            use_node_penalty_flag,
            data['goal_nodes'],
            data['Nx'],            # P0-3: Lattice dimensions for procedural coords
            data['Ny'],
            data['Nz'],
            data['goal_coords'],   # P0-3: (K, 3) goal coordinates only
            data['use_astar'],
            self.dist_val_pool.ravel(),     # FIX: Pass full pool without slicing
            pool_stride,                    # Pool stride (N_max)
            self.parent_val_pool.ravel(),   # FIX: Pass full pool without slicing
            pool_stride,                    # Pool stride (N_max)
            new_frontier.ravel(),  # P0-4: BIT-PACKED uint32 output
            # FIX-7: ROI bitmaps for neighbor validation
            data['roi_bitmaps'].ravel(),    # (K, bitmap_words) flattened
            data['bitmap_words'],           # Words per bitmap
            use_bitmap_flag,                # 1 = enforce bitmap, 0 = bbox-only
            blocked_counter,                # INSTRUMENTATION: bitmap block counter
            # Phase 4: ROI bounding boxes
            data['roi_minx'],
            data['roi_maxx'],
            data['roi_miny'],
            data['roi_maxy'],
            data['roi_minz'],
            data['roi_maxz'],
            pref_layers_gpu,
            src_x_coords_gpu,
            cp.int32(window_cols),
            cp.float32(rr_alpha),
            cp.float32(jitter_eps)
        )

        # P0-5: Time kernel execution
        start_event.record()
        # BATCH SANITY CHECK
        logger.debug(f"[BATCH-SANITY] active_list_kernel: K={K} total_active={total_active} roi_ids.shape={roi_ids.shape} node_ids.shape={node_ids.shape}")
        logger.debug(f"[BATCH-SANITY] dist.shape={data['dist'].shape} parent.shape={data['parent'].shape}")
        self.active_list_kernel((grid_size,), (block_size,), args)

        # INSTRUMENTATION: Log bitmap blocking
        blocked_count = int(blocked_counter[0])
        if use_bitmap_flag and blocked_count > 0:
            logger.debug(f"[BITMAP-DEBUG] Blocked {blocked_count:,} neighbors by owner bitmap")
        # Removed spammy warning about 0 neighbors blocked - this is normal
        if rr_alpha > 0.0:
            logger.debug(f"[KERNEL-RR-WAVEFRONT] Active: alpha={float(rr_alpha)}, window={int(window_cols)}")
        if jitter_eps > 0.0:
            logger.debug(f"[KERNEL-JITTER-WAVEFRONT] Active: eps={float(jitter_eps)}")
        end_event.record()
        end_event.synchronize()
        kernel_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        # Update frontier
        old_frontier[:] = new_frontier

        # Fast check: count non-zero words (approximate)
        nodes_expanded = int(cp.count_nonzero(new_frontier))

        # GUARDED DIAGNOSTICS: Only when debugging (saves 481ms per net!)
        if DEBUG_VERBOSE_GPU:
            # DIAGNOSTIC: Check if kernel wrote anything to new_frontier
            new_frontier_words_set = int(cp.count_nonzero(new_frontier))
            logger.debug(f"[KERNEL-OUTPUT] new_frontier has {new_frontier_words_set} non-zero words (out of {K * frontier_words} total)")

            # Count nodes expanded (count set BITS, not sum of uint32 values!)
            nodes_expanded_mask = cp.unpackbits(new_frontier.view(cp.uint8).ravel(), bitorder='little')
            nodes_expanded_actual = int(cp.count_nonzero(nodes_expanded_mask))
            logger.debug(f"[KERNEL-OUTPUT] After unpacking: {nodes_expanded_actual} bits set (expanded nodes)")

            # Track ROI activity after kernel execution
            if nodes_expanded_actual > 0:
                # Compact new_frontier to see which ROIs are active
                new_flat_idx = cp.nonzero(nodes_expanded_mask.reshape(K, -1)[:, :data['max_roi_size']].ravel())[0]
                if new_flat_idx.size > 0:
                    new_roi_ids = (new_flat_idx // data['max_roi_size']).astype(cp.int32)
                    n_active_rois = int(cp.unique(new_roi_ids).size)
                    logger.debug(f"[ACTIVE-SET] After kernel: unique_rois={n_active_rois}, total_expanded={nodes_expanded_actual}")
                    if n_active_rois < 10:
                        unique_rois_list = cp.asnumpy(cp.unique(new_roi_ids)).tolist()
                        logger.debug(f"[ACTIVE-ROIS] Active ROI IDs: {unique_rois_list}")

            # Log performance metrics
            active_pct = 100.0 * total_active / (K * data['max_roi_size'])
            edges_per_node = 5  # Conservative estimate
            edges_relaxed = total_active * edges_per_node
            edges_per_sec = edges_relaxed / (kernel_ms / 1000.0) if kernel_ms > 0 else 0

            logger.debug(f"[GPU-PERF] active={total_active:,} ({active_pct:.4f}%), "
                       f"compact={compact_ms:.3f}ms, kernel={kernel_ms:.3f}ms, "
                       f"edges/sec={edges_per_sec/1e6:.2f}M")

        return nodes_expanded

    def _expand_wavefront_full_scan(self, data: dict, K: int, frontier: cp.ndarray) -> int:
        """
        Expand wavefront using FULL SCAN (original kernel, for dense frontiers).
        """
        # Allocate new frontier mask
        new_frontier = cp.zeros_like(frontier)

        # Get dimensions for kernel launch
        max_roi_size = data['max_roi_size']
        max_edges = data['batch_indices'].shape[1]

        # FIX: Launch exactly K blocks (kernel expects blockIdx.x = ROI index)
        # Each block grid-strides across its ROI's nodes
        block_size = 512  # Optimized for Ada Lovelace (was 256)
        grid_size = K  # One block per ROI (as kernel expects!)

        # CRITICAL: Determine CSR strides (0 for shared, actual dims for per-ROI)
        # Check if arrays are broadcast (stride 0) or contiguous per-ROI
        indptr_arr = data['batch_indptr']
        indices_arr = data['batch_indices']
        weights_arr = data['batch_weights']

        # Detect shared CSR: check for stride=0
        is_shared_csr = False
        if hasattr(indptr_arr, 'strides') and len(indptr_arr.strides) == 2:
            # Check if first dimension has stride 0 (broadcast)
            if indptr_arr.strides[0] == 0:
                is_shared_csr = True
                indptr_stride = 0
                indices_stride = 0
                weights_stride = 0
                logger.debug(f"[CUDA-WAVEFRONT] Detected shared CSR (stride=0), using zero-copy broadcast")
            else:
                # Per-ROI CSR: stride = number of elements in one ROI's row
                indptr_stride = indptr_arr.shape[1]  # max_roi_size + 1
                indices_stride = indices_arr.shape[1]  # max_edges
                weights_stride = weights_arr.shape[1]  # max_edges
                logger.debug(f"[CUDA-WAVEFRONT] Per-ROI CSR, strides=({indptr_stride}, {indices_stride}, {weights_stride})")
        else:
            # Fallback: assume contiguous
            indptr_stride = max_roi_size + 1
            indices_stride = max_edges
            weights_stride = max_edges
            logger.debug(f"[CUDA-WAVEFRONT] Assuming contiguous per-ROI CSR")

        logger.debug(f"[CUDA-WAVEFRONT] Launching {grid_size} blocks × {block_size} threads for {K} ROIs ({max_roi_size:,} nodes each)")

        # VALIDATION: Sanity checks before kernel launch
        assert K > 0, f"Invalid K={K}"
        assert max_roi_size > 0, f"Invalid max_roi_size={max_roi_size}"
        assert max_edges > 0, f"Invalid max_edges={max_edges}"
        assert indptr_stride >= 0, f"Invalid indptr_stride={indptr_stride}"
        assert indices_stride >= 0, f"Invalid indices_stride={indices_stride}"
        assert weights_stride >= 0, f"Invalid weights_stride={weights_stride}"

        # Log memory layout for debugging
        if is_shared_csr:
            logger.debug(f"[CUDA-WAVEFRONT] Shared CSR mode: ALL {K} ROIs use same graph (memory saved: {(K-1)*max_edges*8/1e6:.1f} MB)")
        else:
            logger.debug(f"[CUDA-WAVEFRONT] Per-ROI CSR mode: Each ROI has dedicated CSR copy")

        # CRITICAL FIX: Don't call .ravel() on broadcast CSR arrays!
        # .ravel() materializes the broadcast, copying stride-0 arrays into contiguous memory
        # This defeats the shared CSR optimization and causes OOM (60GB instead of 0.42GB!)
        # CuPy RawKernel accepts multi-dimensional arrays - kernel uses strides to handle both cases

        # Only ravel per-ROI arrays (frontier, dist, parent) which MUST be flattened
        # CSR arrays: pass directly to preserve broadcast stride
        # P0-4: frontier is now bit-packed uint32, include frontier_words parameter
        frontier_words = frontier.shape[1]

        # Get pool stride and pass sliced pool views (matching persistent kernel pattern)
        pool_stride = self.dist_val_pool.shape[1]  # N_max

        # Get use_bitmap flag from data dict
        use_bitmap_flag = 1 if data.get('use_bitmap', False) else 0  # Default FALSE for iter-1 compatibility
        node_penalty_arr = data.get('node_penalty', self._zero_node_penalty)
        node_penalty_stride = int(data.get('node_penalty_stride', 0))
        use_node_penalty_flag = 1 if data.get('use_node_penalty', False) else 0

        # Get iter1_relax_hv flag from config and current iteration
        current_iteration = getattr(self, 'current_iteration', 1)
        iter1_relax_hv_flag = 1 if current_iteration == 1 else 0  # Relax H/V in Iter-1

        # Get use_atomic_parent_keys flag (Iter>=2 only per runbook)
        use_atomic_flag = 1 if current_iteration >= 2 else 0  # Use atomic keys in Iter>=2 (cycle-proof)

        args = (
            K,
            max_roi_size,
            max_edges,
            frontier.ravel(),      # P0-4: BIT-PACKED uint32 frontier
            frontier_words,        # P0-4: Number of uint32 words per ROI
            indptr_arr,            # FIX: Don't ravel! Preserves broadcast stride
            indices_arr,           # FIX: Don't ravel! Preserves broadcast stride
            weights_arr,           # FIX: Don't ravel! Preserves broadcast stride
            indptr_stride,         # Kernel uses this to compute correct address
            indices_stride,        # When stride=0, all ROIs share base pointer
            weights_stride,        # When stride>0, each ROI has own row
            weights_arr,           # total_cost = same as weights (already negotiated)
            weights_stride,        # total_cost_stride
            node_penalty_arr,
            node_penalty_stride,
            use_node_penalty_flag,
            data['goal_nodes'],    # NEW: A* goal nodes
            data['Nx'],            # P0-3: Lattice dimensions
            data['Ny'],
            data['Nz'],
            data['goal_coords'],   # P0-3: Goal coordinates only (K×3, not max_roi_size×3!)
            data['use_astar'],     # NEW: A* enable flag
            self.dist_val_pool.ravel(),     # FIX: Pass full pool without slicing
            pool_stride,                    # Pool stride (N_max)
            self.parent_val_pool.ravel(),   # FIX: Pass full pool without slicing
            pool_stride,                    # Pool stride (N_max)
            new_frontier.ravel(),  # P0-4: BIT-PACKED uint32 output
            # FIX-7: ROI bitmaps for neighbor validation
            data['roi_bitmaps'].ravel(),    # (K, bitmap_words) flattened
            data['bitmap_words'],           # Words per bitmap
            use_bitmap_flag,                # 1 = enforce bitmap, 0 = bbox-only
            iter1_relax_hv_flag,            # 1 = relax H/V discipline in Iter-1
            # ATOMIC PARENT KEYS: Cycle-proof tracking for Iter>=2
            self.best_key_pool.ravel(),     # (K, N_max) 64-bit atomic keys
            pool_stride,                    # Same stride as dist/parent
            use_atomic_flag,                # 1 = atomic mode (Iter>=2), 0 = legacy (Iter==1)
        )

        if is_shared_csr:
            logger.debug(f"[CUDA-WAVEFRONT] CSR arrays: indptr shape={indptr_arr.shape}, indices shape={indices_arr.shape}, A*={'enabled' if data['use_astar'] else 'disabled'}")
        self.wavefront_kernel((grid_size,), (block_size,), args)

        # Synchronize to ensure kernel completion
        cp.cuda.Stream.null.synchronize()

        # Update frontier (clear old, set new)
        frontier[:] = new_frontier

        # Fast check: count non-zero words (approximate, but fast!)
        # This avoids expensive unpackbits() operation (320ms saving!)
        nodes_expanded = int(cp.count_nonzero(new_frontier))
        return nodes_expanded

    def _run_persistent_kernel(self, data: dict, K: int, initial_frontier: cp.ndarray) -> int:
        """
        P1-6: Execute persistent kernel with device-side queues.

        This launches a SINGLE kernel that runs until all paths are found,
        eliminating the overhead of 100-200 kernel launches per batch.

        Args:
            data: Batched GPU arrays from _prepare_batch
            K: Number of ROIs
            initial_frontier: (K, frontier_words) bit-packed initial frontier

        Returns:
            Number of iterations completed
        """
        import time

        logger.debug(f"[PERSISTENT-KERNEL] Initializing device-side queues for {K} ROIs")

        max_roi_size = data['max_roi_size']

        # Allocate device queues (ping-pong buffers)
        # Size estimate: avg_degree * max_frontier_size
        # Conservative: max_roi_size * 10 (assumes max 10× expansion per iteration)
        max_queue_size = min(max_roi_size * 10, 50_000_000)  # Cap at 50M entries (200MB per queue)

        logger.debug(f"[PERSISTENT-KERNEL] Allocating queues: {max_queue_size:,} entries ({max_queue_size*4/1e6:.1f} MB each)")

        queue_a = cp.zeros(max_queue_size, dtype=cp.int32)
        queue_b = cp.zeros(max_queue_size, dtype=cp.int32)
        size_a = cp.zeros(1, dtype=cp.int32)
        size_b = cp.zeros(1, dtype=cp.int32)
        iterations_out = cp.zeros(1, dtype=cp.int32)

        # Initialize queue A with source nodes from frontier
        # Unpack bit-packed frontier to get initial active nodes
        frontier_words = initial_frontier.shape[1]
        frontier_bytes = initial_frontier.view(cp.uint8)
        # FIX: CuPy unpackbits doesn't support axis parameter - unpack and reshape instead
        frontier_mask_flat = cp.unpackbits(frontier_bytes.ravel(), bitorder='little')
        frontier_mask = frontier_mask_flat.reshape(K, -1)[:, :max_roi_size]

        # Compact to get (roi, node) pairs
        flat_idx = cp.nonzero(frontier_mask.ravel())[0]
        total_initial = int(flat_idx.size)

        if total_initial == 0:
            logger.warning("[PERSISTENT-KERNEL] No initial frontier nodes - nothing to do")
            return 0

        if total_initial > max_queue_size:
            logger.error(f"[PERSISTENT-KERNEL] Initial frontier ({total_initial}) exceeds queue capacity ({max_queue_size})!")
            logger.error("[PERSISTENT-KERNEL] Falling back to iterative kernel")
            return -1

        # Pack (roi, node) into queue_a
        roi_ids = flat_idx // max_roi_size
        node_ids = flat_idx - (roi_ids * max_roi_size)
        packed = (roi_ids << 24) | node_ids  # 8-bit ROI + 24-bit node
        queue_a[:total_initial] = packed
        size_a[0] = total_initial

        logger.debug(f"[PERSISTENT-KERNEL] Initialized queue with {total_initial:,} active nodes")

        # Get CSR arrays and strides
        indptr_arr = data['batch_indptr']
        indices_arr = data['batch_indices']
        weights_arr = data['batch_weights']

        # Detect strides (0 for shared CSR)
        if hasattr(indptr_arr, 'strides') and len(indptr_arr.strides) == 2 and indptr_arr.strides[0] == 0:
            # Stride 0 = shared CSR
            indptr_stride = 0
            indices_stride = 0
            weights_stride = 0
            logger.debug("[PERSISTENT-KERNEL] Using shared CSR (stride=0)")
        else:
            # Per-ROI arrays
            indptr_stride = indptr_arr.shape[1] if len(indptr_arr.shape) > 1 else max_roi_size + 1
            indices_stride = indices_arr.shape[1] if len(indices_arr.shape) > 1 else data['max_edges']
            weights_stride = weights_arr.shape[1] if len(weights_arr.shape) > 1 else data['max_edges']
            logger.debug(f"[PERSISTENT-KERNEL] Using per-ROI CSR (strides={indptr_stride}, {indices_stride}, {weights_stride})")

        # Prepare kernel arguments
        args = (
            queue_a,
            queue_b,
            size_a,
            size_b,
            max_queue_size,
            K,
            max_roi_size,
            indptr_arr,
            indices_arr,
            weights_arr,
            indptr_stride,
            indices_stride,
            weights_stride,
            data['Nx'],
            data['Ny'],
            data['Nz'],
            data['goal_coords'],
            data['use_astar'],
            data['dist'].ravel(),
            data['parent'].ravel(),
            iterations_out,
        )

        # Launch configuration for cooperative kernel
        block_size = 256
        grid_size = 256  # Use many blocks for good occupancy

        logger.debug(f"[PERSISTENT-KERNEL] Launching cooperative kernel: {grid_size} blocks × {block_size} threads = {grid_size*block_size:,} total threads")

        # NOTE: Cooperative kernel launch requires special API
        # CuPy's RawKernel doesn't directly support launchCooperativeKernel
        # We need to use the lower-level CUDA runtime API

        try:
            # Get CUDA function pointer
            kernel_func = self.persistent_kernel

            # Try cooperative launch using cupy.cuda.runtime
            # This is the standard way but requires compute capability 6.0+
            start_time = time.perf_counter()

            # Standard launch (cooperative groups work with regular launch on modern GPUs)
            kernel_func((grid_size,), (block_size,), args)

            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - start_time

            iters = int(iterations_out[0])
            logger.debug(f"[PERSISTENT-KERNEL] Completed {iters} iterations in {elapsed*1000:.2f} ms ({elapsed*1000/max(iters,1):.3f} ms/iter)")
            logger.debug(f"[PERSISTENT-KERNEL] Compare to iterative: ~{iters*0.007:.2f} ms launch overhead eliminated")

            return iters

        except Exception as e:
            logger.error(f"[PERSISTENT-KERNEL] Cooperative kernel launch failed: {e}")
            logger.error("[PERSISTENT-KERNEL] This may require compute capability 6.0+ or cooperative launch API")
            logger.error("[PERSISTENT-KERNEL] Falling back to iterative kernel")
            return -1

    def _prepare_roundrobin_params(self, roi_batch: List[Tuple], data: dict, current_iteration: int) -> Tuple:
        """Prepare round-robin bias + jitter parameters for GPU kernel.

        Computes preferred layer and source x-coordinate for each ROI in the batch.
        Returns tiny device arrays and scalars for kernel.
        """
        import numpy as np

        # Handle case where roi_batch is empty but data has sources
        if not roi_batch and 'sources' in data:
            K = int(data['sources'].shape[0]) if hasattr(data['sources'], 'shape') else len(data['sources'])
        else:
            K = len(roi_batch)

        Nx = data['Nx']
        Ny = data['Ny']
        plane_size = Nx * Ny

        # Get layer count
        layer_count = self.lattice.layers if self.lattice else 18
        even_layers = [z for z in range(layer_count) if (z & 1) == 0]

        pref_layers = []
        src_x_coords = []

        # For each ROI, compute preferred layer and source x
        if roi_batch:
            # Original path: use roi_batch tuples
            for i, roi in enumerate(roi_batch):
                src_idx = roi[0]

                # Hash to pick preferred even layer
                h = ((i + 1) * 0x9E3779B9) & 0xffffffff
                pref_z = even_layers[h % len(even_layers)] if even_layers else 0
                pref_layers.append(pref_z)

                # Decode source x-coordinate
                src_x = (src_idx % plane_size) % Nx
                src_x_coords.append(src_x)
        else:
            # Wavefront path: use data['sources'] array
            sources = data['sources']
            for i in range(K):
                src_idx = int(sources[i].item()) if hasattr(sources[i], 'item') else int(sources[i])

                # Hash to pick preferred even layer
                h = ((i + 1) * 0x9E3779B9) & 0xffffffff
                pref_z = even_layers[h % len(even_layers)] if even_layers else 0
                pref_layers.append(pref_z)

                # Decode source x-coordinate
                src_x = (src_idx % plane_size) % Nx
                src_x_coords.append(src_x)

        # Upload to GPU
        pref_layers_gpu = cp.asarray(pref_layers, dtype=cp.int32)
        src_x_coords_gpu = cp.asarray(src_x_coords, dtype=cp.int32)

        # Compute parameters based on iteration
        pitch = self.lattice.pitch if self.lattice and hasattr(self.lattice, 'pitch') else 0.4
        window_cols = int(8.0 / pitch)

        if current_iteration <= 3:
            rr_alpha = 0.12
            jitter_eps = 0.001
            logger.debug(f"[RR-ENABLE] YES - iteration {current_iteration} <= 3")
            logger.debug(f"[ROUNDROBIN-PARAMS] iteration={current_iteration}, rr_alpha={rr_alpha}, window_cols={window_cols}")
            logger.debug(f"[ROUNDROBIN-KERNEL] Active for iteration {current_iteration}: alpha={rr_alpha}, window={window_cols} cols")
            logger.debug(f"[RR-SAMPLE] First 5 ROIs: pref_layers={pref_layers[:min(5,K)]}, src_x={src_x_coords[:min(5,K)]}")
        else:
            rr_alpha = 0.0
            jitter_eps = 0.001  # Jitter always on
            logger.debug(f"[RR-ENABLE] NO - iteration {current_iteration} > 3")
            #             logger.debug(f"[ROUNDROBIN-KERNEL] RR disabled for iteration {current_iteration}, jitter still active")

        logger.debug(f"[JITTER-ENABLE] YES - jitter_eps={jitter_eps}")
        logger.debug(f"[JITTER-PARAMS] jitter_eps={jitter_eps}")
        logger.debug(f"[JITTER-KERNEL] jitter_eps={jitter_eps} (breaks ties, prevents elevator shafts)")

        return pref_layers_gpu, src_x_coords_gpu, rr_alpha, window_cols, jitter_eps

    def _reconstruct_paths(self, data: dict, K: int) -> List[Optional[List[int]]]:
        """
        Reconstruct paths from parent pointers using GPU kernel (eliminates 256 MB CPU transfer).

        Strategy:
        - For large ROIs (>100k nodes): Use GPU backtrace kernel (sparse transfer)
        - For small ROIs (<100k nodes): Use CPU backtrace (low overhead)

        Args:
            data: Batched GPU arrays with parent pointers
            K: Number of ROIs

        Returns:
            List of paths (local ROI indices)
        """
        import numpy as np

        max_roi_size = data['max_roi_size']
        sinks = data['sinks']

        # Decide whether to use GPU or CPU path reconstruction
        # For large graphs (>100k nodes), GPU kernel saves massive bandwidth
        # For small ROIs, CPU is faster due to lower kernel launch overhead
        use_gpu_backtrace = (max_roi_size > 100_000)

        if use_gpu_backtrace:
            # GPU PATH RECONSTRUCTION (eliminates 256 MB parent/dist transfer!)
            import time
            start = time.perf_counter()

            # Estimate max path length (heuristic: sqrt of ROI size, capped at 4096)
            max_path_len = min(4096, int(np.sqrt(max_roi_size)) + 100)

            # Allocate GPU output buffers
            paths_gpu = cp.zeros((K, max_path_len), dtype=cp.int32)
            path_lengths_gpu = cp.zeros(K, dtype=cp.int32)
            sinks_gpu = cp.array(sinks, dtype=cp.int32)

            # Launch GPU backtrace kernel
            # PERF: 512 threads/block for better occupancy
            block_size = 512  # Optimized for RTX 4090 (was 256)
            grid_size = (K + block_size - 1) // block_size

            # Get use_bitmap flag from data dict
            use_bitmap_flag = 1 if data.get('use_bitmap', False) else 0  # Default FALSE for iter-1 compatibility

            # DIAGNOSTIC: Log backtrace kernel args
            logger.debug(f"[BACKTRACE-ARGS] K={K}, max_roi_size={max_roi_size}, max_path_len={max_path_len}")
            logger.debug(f"[BACKTRACE-ARGS] bitmap_words={data['bitmap_words']}, use_bitmap_flag={use_bitmap_flag}")
            logger.debug(f"[BACKTRACE-ARGS] roi_bitmaps.shape={data['roi_bitmaps'].shape if data['roi_bitmaps'] is not None else 'None'}")

            # CRITICAL DIAGNOSTIC: Validate parent-CSR consistency BEFORE backtrace
            # This detects if parent pointers are corrupted during search phase
            bad_counts = cp.zeros(K, dtype=cp.int32)
            threads = 256
            grid_y = (max_roi_size + threads - 1) // threads
            val_grid = (K, grid_y)

            # Get parent stride (N_max for full pool, or max_roi_size for sliced arrays)
            parent_stride = self.parent_val_pool.shape[1] if self.parent_val_pool is not None else max_roi_size

            logger.debug(f"[VALIDATOR] parent.shape={data['parent'].shape}, parent_stride={parent_stride}")
            logger.debug(f"[VALIDATOR] Using strides: indptr=0, indices=0, parent={parent_stride}")

            self.validate_parents_kernel(
                val_grid, (threads,),
                (K, max_roi_size,
                 data['batch_indptr'], 0,  # indptr_stride=0 for shared CSR
                 data['batch_indices'], 0,  # indices_stride=0 for shared CSR
                 self.parent_val_pool.ravel(),  # Use FULL pool, not sliced view
                 parent_stride,  # Stride between ROIs in parent array (N_max)
                 bad_counts)
            )
            cp.cuda.Stream.null.synchronize()

            violations = bad_counts.get()
            total_violations = int(violations.sum())
            if total_violations > 0:
                rois_with_errors = np.where(violations > 0)[0]
                logger.error(f"[PARENT-VALIDATE] {total_violations} parent-CSR mismatches across {len(rois_with_errors)} ROIs!")
                for roi_idx in rois_with_errors[:10]:  # Show first 10
                    logger.error(f"[PARENT-VALIDATE]   ROI {roi_idx}: {violations[roi_idx]} bad parents")
                logger.error(f"[PARENT-VALIDATE] Corruption in SEARCH PHASE - parent writes are invalid!")
            else:
                logger.debug(f"[PARENT-VALIDATE] 0 parent-CSR mismatches ✓ Parents are valid!")

            # Calculate strides (pool strides, not max_roi_size!)
            parent_stride_val = self.parent_val_pool.shape[1] if self.parent_val_pool is not None else max_roi_size
            dist_stride_val = self.dist_val_pool.shape[1] if self.dist_val_pool is not None else max_roi_size

            logger.debug(f"[BACKTRACE-STRIDE] parent_stride={parent_stride_val}, dist_stride={dist_stride_val}, max_roi_size={max_roi_size}")

            # Get use_atomic flag from current iteration
            current_iteration = getattr(self, 'current_iteration', 1)
            use_atomic_flag = 1 if current_iteration >= 2 else 0  # Use atomic keys in Iter>=2 (cycle-proof)
            key_stride_val = self.best_key_pool.shape[1] if hasattr(self, 'best_key_pool') else max_roi_size

            self.backtrace_kernel(
                (grid_size,), (block_size,),
                (K, max_roi_size,
                 self.parent_val_pool.ravel(),  # Pass FULL pool
                 parent_stride_val,  # CRITICAL: Pass parent_stride!
                 self.dist_val_pool.ravel(),  # Pass FULL pool
                 dist_stride_val,  # CRITICAL: Pass dist_stride!
                 sinks_gpu,
                 paths_gpu, path_lengths_gpu, max_path_len,
                 data['roi_bitmaps'].ravel(), data['bitmap_words'], use_bitmap_flag,  # Bitmap flag
                 self.best_key_pool.ravel(), key_stride_val, use_atomic_flag)  # ATOMIC KEYS
            )
            cp.cuda.Stream.null.synchronize()

            # Transfer ONLY compact paths (much smaller than full parent/dist arrays)
            paths_cpu = paths_gpu.get()
            path_lengths_cpu = path_lengths_gpu.get()

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Parse results
            paths = []
            for roi_idx in range(K):
                path_len = path_lengths_cpu[roi_idx]
                if path_len > 0:
                    # Extract path from compact buffer
                    path = paths_cpu[roi_idx, :path_len].tolist()
                    paths.append(path)
                elif path_len == -1:
                    # Cycle detected (cosmetic - doesn't affect routing correctness)
                    logger.warning(f"[GPU-BACKTRACE] Path reconstruction: cycle detected for ROI {roi_idx}")
                    paths.append(None)
                elif path_len == -2:
                    # Bitmap validation error (node not in ROI bitmap)
                    logger.warning(f"[GPU-BACKTRACE] Path reconstruction: bitmap validation error for ROI {roi_idx}")
                    paths.append(None)
                else:
                    # No path found (path_len == 0)
                    paths.append(None)

            # Calculate bandwidth savings
            old_transfer_mb = (K * max_roi_size * 4 * 2) / 1e6  # parent + dist (int32 + float32)
            new_transfer_mb = (K * max_path_len * 4 + K * 4) / 1e6  # paths + lengths
            savings_mb = old_transfer_mb - new_transfer_mb

            logger.debug(f"[GPU-BACKTRACE] Reconstructed {K} paths in {elapsed_ms:.2f}ms on GPU")
            logger.debug(f"[GPU-BACKTRACE] Transfer: {new_transfer_mb:.2f} MB (saved {savings_mb:.2f} MB vs CPU method)")

            return paths

        else:
            # CPU PATH RECONSTRUCTION (for small ROIs, faster due to low overhead)
            logger.debug(f"[CPU-BACKTRACE] Using CPU path reconstruction for {K} small ROIs ({max_roi_size:,} nodes)")

            # Transfer to CPU (acceptable for small ROIs)
            parent_cpu = data['parent'].get()
            dist_cpu = data['dist'].get()

            paths = []
            for roi_idx in range(K):
                sink = sinks[roi_idx]

                # Check if path exists
                if dist_cpu[roi_idx, sink] == np.inf:
                    paths.append(None)
                    continue

                # Walk backward from sink to source
                path = []
                curr = sink
                visited = set()

                while curr != -1:
                    # Cycle detection (cosmetic - doesn't affect routing correctness)
                    if curr in visited:
                        logger.warning(f"[CPU-BACKTRACE] Path reconstruction: cycle detected at node {curr}")
                        paths.append(None)
                        break

                    path.append(curr)
                    visited.add(curr)
                    parent_node = parent_cpu[roi_idx, curr]

                    # MANHATTAN VALIDATION: Check if parent->curr is adjacent
                    if parent_node != -1 and len(path) > 1:
                        # Get coordinates (using lattice if available)
                        if hasattr(self, 'lattice') and self.lattice:
                            x_curr, y_curr, z_curr = self.lattice.idx_to_coord(curr)
                            x_par, y_par, z_par = self.lattice.idx_to_coord(parent_node)

                            dx = abs(x_curr - x_par)
                            dy = abs(y_curr - y_par)
                            dz = abs(z_curr - z_par)

                            # Check if parent->child is Manhattan-legal
                            if dz != 0:
                                # Via jump - same X,Y required
                                if dx != 0 or dy != 0:
                                    logger.error(f"[PARENT-VALIDATION] Non-adjacent via: {parent_node}->{curr}, dist=({dx},{dy},{dz})")
                                    paths.append(None)  # Reject path
                                    break
                            elif (dx + dy) != 1:
                                # Same layer - must be adjacent
                                logger.error(f"[PARENT-VALIDATION] Non-adjacent parent: {parent_node}->{curr}, dist=({dx},{dy},{dz})")
                                paths.append(None)  # Reject path
                                break
                            else:
                                # Check layer direction discipline
                                is_h_layer = (z_curr % 2) == 1  # Odd layers = horizontal
                                if is_h_layer and dy != 0:
                                    logger.error(f"[PARENT-VALIDATION] H-layer violation: {parent_node}->{curr}, dy={dy}")
                                    paths.append(None)  # Reject path
                                    break
                                elif not is_h_layer and dx != 0:
                                    logger.error(f"[PARENT-VALIDATION] V-layer violation: {parent_node}->{curr}, dx={dx}")
                                    paths.append(None)  # Reject path
                                    break

                    curr = parent_node

                    # Safety limit
                    if len(path) > max_roi_size:
                        logger.error(f"[CPU-BACKTRACE] Path reconstruction: exceeded max_roi_size")
                        paths.append(None)
                        break
                else:
                    # Reverse path (built backward)
                    path.reverse()
                    paths.append(path)

            return paths

    # ========================================================================
    # DELTA-STEPPING SSSP ALGORITHM (P1-7)
    # ========================================================================

    def _run_delta_stepping(self, data: dict, K: int, delta: float, roi_batch: List[Tuple] = None) -> List[Optional[List[int]]]:
        """
        Execute delta-stepping SSSP algorithm on GPU with distance-based bucketing.

        Delta-stepping reduces atomic contention by organizing nodes into distance buckets
        and processing them in waves. Nodes at similar distances are processed together,
        reducing conflicting atomic updates.

        Algorithm:
        1. Maintain buckets: bucket[b] contains nodes with distance in [b*Delta, (b+1)*Delta)
        2. Process current bucket with light edges (cost < Delta)
        3. Heavy edges (cost >= Delta) relax to future buckets
        4. Less contention since nodes at similar distances don't compete

        Args:
            data: Batched GPU arrays from _prepare_batch
            K: Number of ROIs
            delta: Bucket width (Delta parameter) - typically median edge cost (0.4-1.0)
            roi_batch: Original ROI batch (for diagnostics)

        Returns:
            List of paths (local ROI indices)

        Performance:
            - Expected 2-4× faster on large graphs due to reduced atomic contention
            - Best for graphs with diverse edge weights and long paths
            - Delta selection is critical: too small = many buckets, too large = poor parallelism
        """
        import time

        
        # Respect the batch width prepared upstream
        K = int(data.get('K', K))
        # Basic invariants to catch off-by-one before launching kernels
        try:
            n_src = len(data['sources'])
            n_dst = len(data['sinks'])
        except Exception:
            n_src = int(data['sources'].shape[0])
            n_dst = int(data['sinks'].shape[0])
        assert n_src == K and n_dst == K, f"K mismatch: K={K} sources={n_src} sinks={n_dst}"
        logger.debug(f"[_run_delta_stepping] Using K={K}, sources.shape={n_src}, sinks.shape={n_dst}")

        logger.debug(f"[DELTA-STEPPING] Starting with K={K} ROIs, delta={delta:.3f}")

        # Adaptive iteration budget
        if roi_batch and len(roi_batch) > 0:
            roi_size = roi_batch[0][5]
            batch_size = len(roi_batch)

            if roi_size > 1_000_000:  # Full graph
                max_iterations = 2000
                logger.debug(f"[DELTA-STEPPING] Full graph routing: {batch_size} nets, {max_iterations} iterations")
            else:
                max_iterations = min(4096, roi_size // 100 + 500)
        else:
            max_iterations = 2000

        start_time = time.perf_counter()

        # Validate sources and sinks
        invalid_rois = []
        for roi_idx in range(K):
            src = int(data['sources'][roi_idx].item())
            dst = int(data['sinks'][roi_idx].item())

            if roi_batch and roi_idx < len(roi_batch):
                actual_roi_size = roi_batch[roi_idx][5]
            else:
                actual_roi_size = data['max_roi_size']

            if src < 0 or src >= actual_roi_size:
                logger.error(f"[DELTA-STEPPING] ROI {roi_idx}: INVALID SOURCE {src}")
                invalid_rois.append(roi_idx)

            if dst < 0 or dst >= actual_roi_size:
                logger.error(f"[DELTA-STEPPING] ROI {roi_idx}: INVALID SINK {dst}")
                invalid_rois.append(roi_idx)

        if invalid_rois:
            logger.error(f"[DELTA-STEPPING] Aborting - {len(invalid_rois)} invalid ROI(s)")
            return [None] * K

        # Initialize buckets
        max_roi_size = data['max_roi_size']
        max_buckets = int(cp.ceil(1000.0 / delta))  # Assume max distance ~1000mm

        # Bucket structure: (K, max_buckets, frontier_words) - bit-packed nodes per bucket
        frontier_words = (max_roi_size + 31) // 32
        buckets = cp.zeros((K, max_buckets, frontier_words), dtype=cp.uint32)

        # Initialize: place source nodes in bucket 0 (distance 0)
        for roi_idx in range(K):
            src = int(data['sources'][roi_idx].item())
            word_idx = src // 32
            bit_pos = src % 32
            buckets[roi_idx, 0, word_idx] = cp.uint32(1) << bit_pos

        logger.debug(f"[DELTA-STEPPING] Initialized {max_buckets} buckets with width {delta:.3f}")
        logger.debug(f"[DELTA-STEPPING] Memory: {K}×{max_buckets}×{frontier_words} uint32 = "
                   f"{K*max_buckets*frontier_words*4/1e6:.1f}MB")

        # Main delta-stepping loop
        current_bucket = 0
        iteration = 0

        while iteration < max_iterations:
            # Find first non-empty bucket
            while current_bucket < max_buckets:
                bucket_sum = int(cp.sum(buckets[:, current_bucket, :]))
                if bucket_sum > 0:
                    break
                current_bucket += 1

            if current_bucket >= max_buckets:
                logger.debug(f"[DELTA-STEPPING] All buckets empty at iteration {iteration}")
                break

            # Get nodes in current bucket
            frontier = buckets[:, current_bucket, :].copy()
            bucket_node_count = int(cp.sum(frontier))

            # Clear current bucket (nodes will be reinserted if distances improve)
            buckets[:, current_bucket, :] = 0

            if bucket_node_count == 0:
                current_bucket += 1
                continue

            # Process bucket with light edge relaxation
            # Light edges: cost < delta (can stay in same or next bucket)
            # Heavy edges: cost >= delta (go to future buckets)
            nodes_expanded = self._delta_relax_bucket(
                data, K, frontier, buckets, current_bucket, delta
            )

            # Check termination: have all sinks been reached? (vectorized)
            # Use fancy indexing to avoid K separate GPU->CPU transfers
            sink_dists_check = data['dist'][cp.arange(K), data['sinks']]
            sinks_reached = int(cp.sum(sink_dists_check < 1e9).get())  # Single transfer

            if iteration % 50 == 0 or iteration < 3:
                logger.debug(f"[DELTA-STEPPING] Iter {iteration}: bucket={current_bucket}, "
                          f"nodes={bucket_node_count}, expanded={nodes_expanded}, "
                          f"sinks_reached={sinks_reached}/{K}")

            # Early termination when all sinks reached
            if sinks_reached == K:
                logger.debug(f"[DELTA-STEPPING] All sinks reached at iteration {iteration}")
                break

            iteration += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"[DELTA-STEPPING] Complete in {iteration+1} iterations, {elapsed_ms:.1f}ms "
                   f"({elapsed_ms/(iteration+1):.2f}ms/iter)")

        # Reconstruct paths
        paths = self._reconstruct_paths(data, K)
        found = sum(1 for p in paths if p)
        logger.debug(f"[DELTA-STEPPING] Paths found: {found}/{K} ({100*found/K:.1f}% success)")

        return paths

    def _delta_relax_bucket(self, data: dict, K: int, frontier: cp.ndarray,
                           buckets: cp.ndarray, current_bucket: int, delta: float) -> int:
        """
        Relax edges for nodes in current bucket using delta-stepping rules.

        Light edges (cost < delta): Can update same or next bucket
        Heavy edges (cost >= delta): Update distant buckets

        This reduces atomic contention because:
        1. Nodes at similar distances process together
        2. Light edges stay local (less bucket conflicts)
        3. Heavy edges are deferred to well-separated buckets

        Args:
            data: Batched GPU arrays
            K: Number of ROIs
            frontier: (K, frontier_words) bit-packed mask of active nodes in bucket
            buckets: (K, max_buckets, frontier_words) all distance buckets
            current_bucket: Index of bucket being processed
            delta: Bucket width

        Returns:
            Number of nodes expanded
        """
        # Unpack frontier bits for processing
        max_roi_size = data['max_roi_size']
        frontier_words = frontier.shape[1]

        # Unpack bits to bytes for nonzero operation
        frontier_bytes = frontier.view(cp.uint8)
        # FIX: CuPy unpackbits doesn't support axis parameter - unpack and reshape instead
        frontier_mask_flat = cp.unpackbits(frontier_bytes.ravel(), bitorder='little')
        frontier_mask = frontier_mask_flat.reshape(K, -1)[:, :max_roi_size]

        # Compact active nodes
        flat_idx = cp.nonzero(frontier_mask.ravel())[0]
        total_active = int(flat_idx.size)
        if total_active == 0:
            return 0

        N = max_roi_size
        roi_ids = (flat_idx // N).astype(cp.int32)
        node_ids = (flat_idx - (roi_ids * N)).astype(cp.int32)

        # Allocate temporary frontier for relaxed nodes
        new_frontier = cp.zeros((K, frontier_words), dtype=cp.uint32)

        # Get CSR arrays
        indptr_arr = data['batch_indptr']
        indices_arr = data['batch_indices']
        weights_arr = data['batch_weights']

        # Detect strides (shared vs per-ROI CSR)
        if hasattr(indptr_arr, 'strides') and len(indptr_arr.strides) == 2 and indptr_arr.strides[0] == 0:
            # Stride 0 = shared CSR
            indptr_stride = 0
            indices_stride = 0
            weights_stride = 0
        else:
            # Per-ROI arrays
            indptr_stride = indptr_arr.shape[1] if len(indptr_arr.shape) > 1 else max_roi_size + 1
            indices_stride = indices_arr.shape[1] if len(indices_arr.shape) > 1 else data['max_edges']
            weights_stride = weights_arr.shape[1] if len(weights_arr.shape) > 1 else data['max_edges']

        # Launch kernel: process active nodes with delta-aware bucketing
        block_size = 256
        grid_size = (total_active + block_size - 1) // block_size

        # Calculate pool_stride for kernel (N_max from pool shape)
        pool_stride = self.dist_val_pool.shape[1] if self.dist_val_pool is not None else max_roi_size

        # Get use_bitmap flag from data dict
        use_bitmap_flag = 1 if data.get('use_bitmap', False) else 0  # Default FALSE for iter-1 compatibility

        # Use active_list_kernel (could be extended with delta-specific logic)
        args = (
            total_active,
            max_roi_size,
            frontier_words,
            roi_ids,
            node_ids,
            indptr_arr,
            indices_arr,
            weights_arr,
            indptr_stride,
            indices_stride,
            weights_stride,
            weights_arr,           # total_cost = same as weights (already negotiated)
            weights_stride,        # total_cost_stride
            data['goal_nodes'],
            data['Nx'],            # P0-3: Lattice dimensions
            data['Ny'],
            data['Nz'],
            data['goal_coords'],   # P0-3: Goal coordinates
            data['use_astar'],
            pool_stride,           # pool_stride_val
            pool_stride,           # pool_stride_parent (same as dist)
            self.dist_val_pool.ravel() if self.dist_val_pool is not None else data['dist'].ravel(),
            self.parent_val_pool.ravel() if self.parent_val_pool is not None else data['parent'].ravel(),
            new_frontier.ravel(),
            # FIX-7: ROI bitmaps for neighbor validation
            data['roi_bitmaps'].ravel(),    # (K, bitmap_words) flattened
            data['bitmap_words'],           # Words per bitmap
            use_bitmap_flag,                # 1 = enforce bitmap, 0 = bbox-only
            # Phase 4: ROI bounding boxes
            data['roi_minx'],
            data['roi_maxx'],
            data['roi_miny'],
            data['roi_maxy'],
            data['roi_minz'],
            data['roi_maxz'],
        )

        # BATCH SANITY CHECK
        logger.debug(f"[BATCH-SANITY] delta_stepping active_list: K={K} total_active={total_active} roi_ids.shape={roi_ids.shape}")
        logger.debug(f"[BATCH-SANITY] dist.shape={data['dist'].shape} parent.shape={data['parent'].shape}")
        self.active_list_kernel((grid_size,), (block_size,), args)
        cp.cuda.Stream.null.synchronize()

        # P1-7: GPU-accelerated bucket assignment (replaces slow Python loop)
        # Old approach: Python loop over K×max_roi_size (potentially millions of iterations)
        # New approach: GPU kernel processes only updated nodes in parallel
        max_buckets = buckets.shape[1]

        # Launch bucket assignment kernel
        # Use grid-stride loop to handle large node counts
        total_nodes = K * max_roi_size
        block_size_bucket = 256
        # Launch enough blocks to cover all nodes, but not excessive
        grid_size_bucket = min((total_nodes + block_size_bucket - 1) // block_size_bucket, 65535)

        bucket_args = (
            K,
            max_roi_size,
            max_buckets,
            frontier_words,
            new_frontier.ravel(),      # Updated nodes (bit-packed)
            data['dist'].ravel(),      # Current distances
            cp.float32(delta),         # Bucket width
            buckets.ravel(),           # Output buckets
        )

        self.bucket_assign_kernel((grid_size_bucket,), (block_size_bucket,), bucket_args)
        cp.cuda.Stream.null.synchronize()

        # Fast check: count non-zero words (approximate, but fast!)
        # This avoids expensive unpackbits() operation (320ms saving!)
        nodes_expanded = int(cp.count_nonzero(new_frontier))
        return nodes_expanded

    def _fallback_cpu_dijkstra(self, roi_batch: List[Tuple]) -> List[Optional[List[int]]]:
        """
        CPU fallback using heapq Dijkstra (identical to SimpleDijkstra).

        Used when GPU pathfinding fails or for correctness validation.
        """
        import heapq
        import numpy as np

        logger.debug(f"[CUDA-FALLBACK] Using CPU Dijkstra for {len(roi_batch)} ROIs")

        paths = []
        # roi_batch now has 13 elements: (src, dst, indptr, indices, weights, roi_size, roi_bitmap, bbox_minx, bbox_maxx, bbox_miny, bbox_maxy, bbox_minz, bbox_maxz)
        # CPU fallback only needs the first 6
        for roi_idx, (src, sink, indptr, indices, weights, size, *_) in enumerate(roi_batch):
            # Transfer to CPU if needed
            if hasattr(indptr, 'get'):
                indptr_cpu = indptr.get()
                indices_cpu = indices.get()
                weights_cpu = weights.get()
            else:
                indptr_cpu = np.asarray(indptr)
                indices_cpu = np.asarray(indices)
                weights_cpu = np.asarray(weights)

            # Heap-based Dijkstra - Use sparse dictionaries to avoid large allocations
            # This prevents "Unable to allocate X MiB" errors on large graphs
            dist = {src: 0.0}
            parent = {}

            heap = [(0.0, src)]
            visited = set()

            while heap:
                current_dist, u = heapq.heappop(heap)

                if u in visited:
                    continue

                visited.add(u)

                if u == sink:
                    break

                # Relax neighbors
                start = int(indptr_cpu[u])
                end = int(indptr_cpu[u + 1])

                for i in range(start, end):
                    v = int(indices_cpu[i])
                    cost = float(weights_cpu[i])

                    if v not in visited:
                        new_dist = current_dist + cost
                        old_dist = dist.get(v, float('inf'))
                        if new_dist < old_dist:
                            dist[v] = new_dist
                            parent[v] = u
                            heapq.heappush(heap, (new_dist, v))

            # Reconstruct path
            if sink in dist and dist[sink] < float('inf'):
                path = []
                curr = sink
                while curr != src:
                    path.append(curr)
                    if curr not in parent:
                        logger.warning(f"[CPU-FALLBACK] ROI {roi_idx}: Path reconstruction failed at node {curr}")
                        path = None
                        break
                    curr = parent[curr]
                if path is not None:
                    path.append(src)
                    path.reverse()
                    paths.append(path)
                else:
                    paths.append(None)
            else:
                logger.debug(f"[CPU-FALLBACK] ROI {roi_idx}: No path found from {src} to {sink}")
                paths.append(None)

        return paths

    # ========================================================================
    # SIMPLIFIED INTERFACE FOR SIMPLEDIJKSTRA INTEGRATION
    # ========================================================================

    def find_path_roi_gpu(self,
                         src: int,
                         dst: int,
                         costs,
                         roi_nodes,
                         global_to_roi,
                         node_penalty=None) -> Optional[List[int]]:
        """
        GPU pathfinding on single ROI (SimpleDijkstra-compatible interface).

        This is the integration point for SimpleDijkstra.find_path_roi().

        Args:
            src, dst: Global node indices
            costs: Edge costs array (global graph)
            roi_nodes: Array of global node indices in ROI
            global_to_roi: Mapping from global to local ROI indices

        Returns:
            Path as list of global node indices, or None
        """
        import numpy as np

        # Convert to CPU if needed
        if hasattr(roi_nodes, 'get'):
            roi_nodes_cpu = roi_nodes.get()
        else:
            roi_nodes_cpu = np.asarray(roi_nodes)

        if hasattr(global_to_roi, 'get'):
            global_to_roi_cpu = global_to_roi.get()
        else:
            global_to_roi_cpu = np.asarray(global_to_roi)

        # Map src/dst to ROI space
        roi_src = int(global_to_roi_cpu[src])
        roi_dst = int(global_to_roi_cpu[dst])

        if roi_src < 0 or roi_dst < 0:
            logger.warning("[CUDA-ROI] src or dst not in ROI")
            return None

        # Build ROI CSR subgraph
        roi_size = len(roi_nodes_cpu)
        roi_indptr, roi_indices, roi_weights = self._extract_roi_csr(
            roi_nodes_cpu, global_to_roi_cpu, costs,
            node_penalty=node_penalty,
        )

        # VALIDATION: Verify src/dst are in valid range
        assert 0 <= roi_src < roi_size, \
            f"Source {roi_src} not in valid range [0, {roi_size})"
        assert 0 <= roi_dst < roi_size, \
            f"Destination {roi_dst} not in valid range [0, {roi_size})"

        # VALIDATION: Check if source has edges (warn if isolated)
        src_edge_count = roi_indptr[roi_src + 1] - roi_indptr[roi_src]
        if src_edge_count == 0:
            logger.warning(f"[CUDA-ROI] Source node {src} (local {roi_src}) has no edges in ROI - may not reach destination")

        # VALIDATION: Check if ROI has any edges at all
        total_edges = len(roi_indices)
        if total_edges == 0:
            logger.warning(f"[CUDA-ROI] ROI subgraph has NO edges - disconnected graph")
            return None

        #         logger.debug(f"[CUDA-ROI] Routing in ROI: src={roi_src}, dst={roi_dst}, "
        #                     f"roi_size={roi_size}, edges={total_edges}")

        # Call GPU Near-Far on ROI subgraph
        # Create complete 13-element tuple: (src, dst, indptr, indices, weights, roi_size,
        #                                     roi_bitmap, bbox_minx, bbox_maxx, bbox_miny, bbox_maxy, bbox_minz, bbox_maxz)
        roi_batch = [(roi_src, roi_dst, roi_indptr, roi_indices, roi_weights, roi_size,
                      None,      # roi_bitmap (None = no bitmap filtering)
                      0,         # bbox_minx
                      999999,    # bbox_maxx
                      0,         # bbox_miny
                      999999,    # bbox_maxy
                      0,         # bbox_minz
                      999999)]   # bbox_maxz
        paths = self.find_paths_on_rois(roi_batch)

        if not paths or paths[0] is None:
            return None

        # Convert local ROI path -> global path
        local_path = paths[0]
        global_path = [int(roi_nodes_cpu[node_idx]) for node_idx in local_path]

        return global_path if len(global_path) > 1 else None

    def _extract_roi_csr(self, roi_nodes, global_to_roi, global_costs,
                         node_penalty=None):
        """
        Extract CSR subgraph for ROI.

        Builds a CSR representation of the subgraph induced by roi_nodes
        from the global graph.

        Args:
            roi_nodes: Array of global node indices in ROI
            global_to_roi: Mapping from global to local ROI indices (-1 if not in ROI)
            global_costs: Edge costs array for global graph

        Returns:
            (roi_indptr, roi_indices, roi_weights): CSR representation of ROI subgraph
        """
        import numpy as np

        roi_size = len(roi_nodes)
        max_edges_estimate = roi_size * 10  # Conservative estimate
        indptr = self.indptr.get() if hasattr(self.indptr, "get") else self.indptr
        indices = self.indices.get() if hasattr(self.indices, "get") else self.indices

        # Build local CSR from global graph
        local_edges = []

        for local_u, global_u in enumerate(roi_nodes):
            # Get global edges for this node
            start = int(indptr[global_u])
            end = int(indptr[global_u + 1])

            for ei in range(start, end):
                global_v = int(indices[ei])
                local_v = global_to_roi[global_v]

                # Only include edges within ROI
                if local_v >= 0:
                    cost = float(global_costs[ei])
                    if node_penalty is not None:
                        cost += float(node_penalty[local_v])
                    local_edges.append((local_u, local_v, cost))

        # Convert to CSR format
        local_edges.sort(key=lambda e: e[0])  # Sort by source

        roi_indptr = np.zeros(roi_size + 1, dtype=np.int32)
        roi_indices = np.zeros(len(local_edges), dtype=np.int32)
        roi_weights = np.zeros(len(local_edges), dtype=np.float32)

        curr_src = -1
        for i, (u, v, cost) in enumerate(local_edges):
            while curr_src < u:
                curr_src += 1
                roi_indptr[curr_src] = i
            roi_indices[i] = v
            roi_weights[i] = cost

        while curr_src < roi_size:
            curr_src += 1
            roi_indptr[curr_src] = len(local_edges)

        # VALIDATION: Verify CSR structure integrity
        assert len(roi_indptr) == roi_size + 1, \
            f"CSR indptr size mismatch: {len(roi_indptr)} != {roi_size + 1}"
        assert roi_indptr[0] == 0, \
            f"CSR indptr[0] must be 0, got {roi_indptr[0]}"
        assert roi_indptr[-1] == len(roi_indices), \
            f"CSR indptr[-1] ({roi_indptr[-1]}) != len(indices) ({len(roi_indices)})"

        # Verify all indices are in valid range
        if len(roi_indices) > 0:
            assert roi_indices.min() >= 0 and roi_indices.max() < roi_size, \
                f"CSR indices out of range [0, {roi_size}): min={roi_indices.min()}, max={roi_indices.max()}"

        # Verify indptr is monotonically increasing
        for i in range(len(roi_indptr) - 1):
            assert roi_indptr[i] <= roi_indptr[i+1], \
                f"CSR indptr not monotonic at index {i}: {roi_indptr[i]} > {roi_indptr[i+1]}"

        logger.debug(f"[CSR-EXTRACT] ROI size={roi_size}, edges={len(local_edges)}, "
                    f"edge_density={len(local_edges)/(roi_size*roi_size) if roi_size > 0 else 0:.3f}")

        return roi_indptr, roi_indices, roi_weights

    def find_path_fullgraph_gpu_seeds(self, costs, src_seeds, dst_targets, ub_hint=None, *,
                                      src_seed_costs=None,
                                      dst_target_costs=None,
                                      node_penalty=None,
                                      allowed_bitmap=None, use_bitmap=False):
        """
        Multi-source/multi-sink SSSP using supersource seeding on full graph.

        Args:
            costs: CuPy array (on device) - edge costs for full graph CSR
            src_seeds: np.int32 array of source node IDs
            dst_targets: np.int32 array of destination node IDs
            ub_hint: Optional upper bound for early termination
            src_seed_costs: Initial cost aligned with each source seed
            dst_target_costs: Terminal cost aligned with each destination
            node_penalty: Optional float32 cost for entering each graph node
            allowed_bitmap: Optional uint32 bitmap for owner-aware node filtering
            use_bitmap: Whether to enable bitmap filtering

        Returns:
            Path as list of global node indices from best source to best destination,
            or None if no path found.
        """
        import numpy as np
        import cupy as cp
        import cupyx

        logger.debug(f"[GPU-SEEDS] Starting full-graph SSSP: {len(src_seeds)} sources -> {len(dst_targets)} targets")

        # Validate inputs
        if len(src_seeds) == 0 or len(dst_targets) == 0:
            logger.warning("[GPU-SEEDS] Empty src_seeds or dst_targets")
            return None

        # Get graph dimensions
        num_nodes = len(self.indptr) - 1
        num_edges = len(self.indices)
        logger.debug(f"[GPU-SEEDS] Full graph: {num_nodes} nodes, {num_edges} edges")

        # Dedupe before scatter-add: repeated seeds would add the same bit
        # twice and carry into a neighboring frontier bit. Retain the
        # cheapest cost for duplicate nodes and keep costs aligned after sort.
        def _prepare_terminals(nodes, terminal_costs, label):
            nodes_cpu = np.asarray(nodes, dtype=np.int32).ravel()
            if terminal_costs is None:
                costs_cpu = np.zeros(len(nodes_cpu), dtype=np.float32)
            else:
                costs_cpu = np.asarray(
                    terminal_costs, dtype=np.float32
                ).ravel()
                if len(costs_cpu) != len(nodes_cpu):
                    raise ValueError(
                        f"{label} costs has {len(costs_cpu)} entries, "
                        f"expected {len(nodes_cpu)}"
                    )
            order = np.argsort(nodes_cpu, kind="stable")
            sorted_nodes = nodes_cpu[order]
            sorted_costs = costs_cpu[order]
            unique_nodes, first = np.unique(
                sorted_nodes, return_index=True
            )
            unique_costs = np.minimum.reduceat(sorted_costs, first)
            return (
                cp.asarray(unique_nodes, dtype=cp.int32),
                cp.asarray(unique_costs, dtype=cp.float32),
            )

        src_seeds_gpu, src_seed_costs_gpu = _prepare_terminals(
            src_seeds, src_seed_costs, "source seed"
        )
        dst_targets_gpu, dst_target_costs_gpu = _prepare_terminals(
            dst_targets, dst_target_costs, "destination target"
        )
        if node_penalty is not None:
            node_penalty_gpu = cp.asarray(
                node_penalty, dtype=cp.float32
            ).reshape(1, -1)
            if node_penalty_gpu.shape[1] != num_nodes:
                raise ValueError(
                    f"node_penalty has {node_penalty_gpu.shape[1]} nodes, "
                    f"expected {num_nodes}"
                )
            use_node_penalty = True
            node_penalty_stride = num_nodes
            # Source seeds are initialized directly rather than reached by a
            # relaxation, so the kernel would otherwise never charge their
            # entry-node ownership cost.
            src_seed_costs_gpu = (
                src_seed_costs_gpu
                + node_penalty_gpu.ravel()[src_seeds_gpu]
            )
        else:
            node_penalty_gpu = self._zero_node_penalty
            use_node_penalty = False
            node_penalty_stride = 0

        # Initialize bit-packed frontier (K=1, frontier_words) — vectorized scatter-or
        frontier_words = (num_nodes + 31) // 32
        frontier = cp.zeros((1, frontier_words), dtype=cp.uint32)  # 2D array for K=1
        src_words = (src_seeds_gpu >> 5).astype(cp.int32)
        src_bits = (src_seeds_gpu & 31).astype(cp.uint32)
        cupyx.scatter_add(frontier[0], src_words, cp.uint32(1) << src_bits)
        logger.info(f"[GPU-SEEDS] Initialized frontier with {len(src_seeds)} seeds")

        # Allocate stamp pools if needed (device-resident optimization)
        if self.dist_val_pool is None or self.dist_val_pool.shape[1] < num_nodes:
            N_max = max(num_nodes, 5_000_000)
            logger.debug(f"[GPU-SEEDS] Allocated stamp pools: N_max={N_max}")
            self.dist_val_pool = cp.full((1, N_max), cp.inf, dtype=cp.float32)
            self.parent_val_pool = cp.full((1, N_max), -1, dtype=cp.int32)
            # Allocate 64-bit atomic key pool (cycle-proof parent updates)
            INF_KEY = 0x7F800000FFFFFFFF  # float +inf (upper 32) | parent -1 (lower 32)
            self.best_key_pool = cp.full((1, N_max), INF_KEY, dtype=cp.uint64)

        # Reset the persistent pools in place.  The previous implementation
        # allocated two redundant full-graph arrays and then copied them into
        # these pools for every net.
        dist_view = self.dist_val_pool[0, :num_nodes]
        parent_view = self.parent_val_pool[0, :num_nodes]
        dist_view.fill(cp.inf)
        parent_view.fill(-1)
        dist_view[src_seeds_gpu] = src_seed_costs_gpu
        logger.info(
            f"[GPU-SEEDS] Initialized {len(src_seeds_gpu)} source seeds"
        )

        # Initialize best_key_pool for source seeds with their entry costs.
        INF_KEY = 0x7F800000FFFFFFFF  # +inf (upper 32 bits) | parent=-1 (lower 32 bits)
        # Reset all keys to INF_KEY first
        self.best_key_pool[0, :num_nodes] = INF_KEY
        seed_cost_bits = src_seed_costs_gpu.view(cp.uint32).astype(cp.uint64)
        seed_keys = (
            seed_cost_bits << cp.uint64(32)
        ) | cp.uint64(0xFFFFFFFF)
        self.best_key_pool[0, src_seeds_gpu] = seed_keys

        # Determine lattice dimensions for coordinate encoding
        # For full-graph routing: use linear layout
        Nx = num_nodes
        Ny = 1
        Nz = 1

        # Build data dict for kernel calls - must match complete expected structure
        # Create goal coords for A* (even if use_astar=0, some code may check it)
        goal_x = int(dst_targets_gpu[0]) % Nx
        goal_y = 0
        goal_z = 0
        goal_coords = cp.array([[goal_x, goal_y, goal_z]], dtype=cp.int32)  # (K=1, 3)

        # Create bitmap - owner-aware if provided, otherwise all-ones (no filtering)
        bitmap_words = (num_nodes + 31) // 32
        if allowed_bitmap is not None:
            # Use provided owner-aware bitmap
            # If already a CuPy array (built on GPU), use directly — no upload needed
            if isinstance(allowed_bitmap, cp.ndarray):
                roi_bitmaps = allowed_bitmap.reshape(1, -1)
            else:
                roi_bitmaps = cp.asarray(allowed_bitmap, dtype=cp.uint32).reshape(1, -1)
            bitmap_words = int(roi_bitmaps.shape[1])
            use_bitmap = True

            # CRITICAL: Force-allow source/dest seeds (prevents frontier empty!)
            # This happens IN THE GPU CODE to guarantee seeds are always reachable
            seed_nodes = cp.unique(cp.concatenate([src_seeds_gpu, dst_targets_gpu]))
            seed_words = (seed_nodes >> 5).astype(cp.int32)
            seed_bits = (seed_nodes & 31).astype(cp.uint32)
            force_mask = cp.zeros(bitmap_words, dtype=cp.uint32)
            cupyx.scatter_add(
                force_mask, seed_words, cp.uint32(1) << seed_bits
            )
            roi_bitmaps[0] |= force_mask

            # Count allowed nodes for sanity
            total_bits_set = int(cp.count_nonzero(roi_bitmaps))
            logger.debug(f"[GPU-SEEDS] Owner-aware bitmap: {bitmap_words} words, ~{total_bits_set} bits set, seeds force-allowed")
        else:
            # Default: all bits set (no filtering)
            roi_bitmaps = cp.full((1, bitmap_words), 0xFFFFFFFF, dtype=cp.uint32)
            use_bitmap = False

        # Ensure ALL CSR arrays are CuPy (on GPU)
        indptr_gpu = cp.asarray(self.indptr) if not isinstance(self.indptr, cp.ndarray) else self.indptr
        indices_gpu = cp.asarray(self.indices) if not isinstance(self.indices, cp.ndarray) else self.indices
        costs_gpu = cp.asarray(costs) if not isinstance(costs, cp.ndarray) else costs

        data = {
            'K': 1,
            'max_roi_size': num_nodes,
            'max_edges': num_edges,
            'batch_indptr': indptr_gpu.reshape(1, -1),
            'batch_indices': indices_gpu.reshape(1, -1),
            'batch_weights': costs_gpu.reshape(1, -1),
            'dist': self.dist_val_pool[:1, :num_nodes],  # Use stamp pool slice
            'parent': self.parent_val_pool[:1, :num_nodes],  # Use stamp pool slice
            'sources': src_seeds_gpu[[0]],  # (K=1,) single source for diagnostics
            'sinks': dst_targets_gpu[[0]],  # (K=1,) single sink for diagnostics
            'goal_nodes': dst_targets_gpu[[0]],  # (K=1,) for A*
            'goal_coords': goal_coords,  # (K=1, 3) for A*
            'Nx': Nx,
            'Ny': Ny,
            'Nz': Nz,
            'roi_minx': cp.array([0], dtype=cp.int32),
            'roi_maxx': cp.array([Nx], dtype=cp.int32),
            'roi_miny': cp.array([0], dtype=cp.int32),
            'roi_maxy': cp.array([Ny], dtype=cp.int32),
            'roi_minz': cp.array([0], dtype=cp.int32),
            'roi_maxz': cp.array([Nz], dtype=cp.int32),
            'roi_bitmaps': roi_bitmaps,
            'bitmap_words': bitmap_words,
            'use_astar': 0,
            'use_bitmap': bool(use_bitmap),  # Owner-aware filtering if bitmap provided
            'node_penalty': node_penalty_gpu,
            'node_penalty_stride': node_penalty_stride,
            'use_node_penalty': use_node_penalty,
            'iter1_relax_hv': True,
            'use_atomic_parent_keys': True,  # ENABLED: Eliminates parent pointer race condition cycles
        }

        max_iterations = 2000
        best_dst = None
        best_dist = float('inf')

        logger.debug(f"[GPU-SEEDS] Starting wavefront expansion (max {max_iterations} iterations)")

        # OPTIMIZATION: Use persistent kernel (single launch) if enabled
        use_persistent = getattr(self, '_enable_persistent_kernel', False)

        if use_persistent:
            logger.debug("[GPU-SEEDS] Using PERSISTENT kernel (single launch)")
            # Import persistent kernel module
            from . import persistent_kernel as pk

            # Compile kernel on first use
            if self._persistent_kernel is None:
                logger.debug("[GPU-SEEDS] Compiling persistent kernel with cooperative groups...")
                self._persistent_kernel = pk.create_persistent_kernel()
                logger.debug("[GPU-SEEDS] Persistent kernel compiled successfully!")

            # Launch persistent kernel with stamp pool arrays and owner-aware bitmap
            best_dst_result, best_dist_result, iterations_done = pk.launch_persistent_kernel(
                self._persistent_kernel,
                indptr_gpu,
                indices_gpu,
                costs_gpu,
                num_nodes,
                src_seeds_gpu,
                dst_targets_gpu,
                self.dist_val_pool[0, :num_nodes],  # Use stamp pool slice
                self.parent_val_pool[0, :num_nodes],  # Use stamp pool slice
                self.best_key_pool[0, :num_nodes],  # Use best_key pool slice for atomic keys
                frontier_words,
                allowed_bitmap_gpu=data['roi_bitmaps'][0],
                use_bitmap=bool(data['use_bitmap']),
                node_penalty_gpu=data['node_penalty'],
                use_node_penalty=bool(data['use_node_penalty']),
                src_seed_costs_gpu=src_seed_costs_gpu,
                dst_target_costs_gpu=dst_target_costs_gpu,
                max_iterations=max_iterations
            )

            logger.debug(f"[GPU-SEEDS] Persistent kernel complete: {iterations_done} iterations, dist={best_dist_result:.2f}")

            if best_dst_result < 0:
                logger.warning("[GPU-SEEDS] No path found via persistent kernel")
                return None

            best_dst = best_dst_result
            best_dist = best_dist_result
            iteration = iterations_done

        else:
            logger.debug("[GPU-SEEDS] Using MULTI-LAUNCH kernel (Python loop)")
            # Wavefront expansion loop
            for iteration in range(max_iterations):
                # OPTIMIZATION: Only log progress every 100 iterations to reduce overhead
                if DEBUG_VERBOSE_GPU and iteration % 100 == 0:
                    active_count = int(cp.count_nonzero(frontier))
                    target_dists = self.dist_val_pool[0, dst_targets_gpu]
                    min_target_dist = float(cp.min(target_dists))
                    logger.debug(f"[GPU-SEEDS] Iteration {iteration}: frontier_words={active_count}, min_target_dist={min_target_dist}")

                # Expand wavefront (reuses existing infrastructure)
                # Note: frontier is modified in-place by _expand_wavefront_parallel
                try:
                    self._expand_wavefront_parallel(data, 1, frontier)
                except Exception as e:
                    logger.error(f"[GPU-SEEDS] Wavefront expansion failed at iteration {iteration}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

                # OPTIMIZATION: Only check destinations every 10 iterations to reduce GPU→CPU sync
                # Trade-off: May run a few extra iterations after reaching goal, but much faster overall
                if iteration % 10 == 0 or iteration < 10:
                    target_dists = self.dist_val_pool[0, dst_targets_gpu]
                    min_dist = float(cp.min(target_dists))

                    if min_dist < float('inf'):
                        best_idx = int(cp.argmin(target_dists))
                        best_dst = int(dst_targets_gpu[best_idx])
                        best_dist = min_dist
                        logger.debug(f"[GPU-SEEDS] Path found at iteration {iteration+1}: best_dst={best_dst}, dist={best_dist:.2f}")
                        break

                    # Check upper bound hint
                    if ub_hint is not None and min_dist > ub_hint:
                        logger.debug(f"[GPU-SEEDS] Exceeding upper bound hint {ub_hint} at iteration {iteration}")
                        break

                # OPTIMIZATION: Only check for empty frontier periodically (every 50 iters) or near end
                # Frontier going empty is rare, so this check can be infrequent
                if iteration % 50 == 0 or iteration > max_iterations - 10:
                    active_count = int(cp.count_nonzero(frontier))
                    if active_count == 0:
                        logger.debug(f"[GPU-SEEDS] Frontier empty at iteration {iteration}")
                        break

        # Check final state
        if best_dst is None:
            logger.warning(f"[GPU-SEEDS] No path found after {iteration+1} iterations")
            return None

        logger.warning(f"[GPU-SEEDS] Path found in {iteration+1} iterations ({best_dist:.2f}ms)")

        # Reconstruct on device and transfer only the resulting path.  Copying
        # the entire parent array cost 32 MB of PCIe traffic per monster-board
        # net even though typical paths contain only hundreds of nodes.
        path = self._backtrace_fullgraph_path(
            self.best_key_pool[0, :num_nodes], best_dst, num_nodes
        )
        if path is None:
            return None
        logger.info(f"[GPU-SEEDS] Path reconstructed: length={len(path)}, from seed={path[0]} to target={best_dst}")

        return path

    def _backtrace_fullgraph_path(self, best_key_gpu, best_dst, num_nodes):
        """Backtrace packed atomic parent keys without downloading the graph."""
        import cupy as cp

        if self._fullgraph_backtrace_kernel is None:
            self._fullgraph_backtrace_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void backtrace_parent(
                const unsigned long long* best_key,
                const int start,
                const int num_nodes,
                int* reversed_path,
                const int capacity,
                int* metadata
            ) {
                if (blockIdx.x != 0 || threadIdx.x != 0) {
                    return;
                }

                int current = start;
                int length = 0;
                while (length < capacity) {
                    if (current < 0 || current >= num_nodes) {
                        metadata[0] = length;
                        metadata[1] = -2;
                        return;
                    }

                    reversed_path[length++] = current;
                    int previous = (int)(
                        best_key[current] & 0xFFFFFFFFULL
                    );
                    if (previous == -1) {
                        metadata[0] = length;
                        metadata[1] = 1;
                        return;
                    }
                    if (previous == current) {
                        metadata[0] = length;
                        metadata[1] = -3;
                        return;
                    }
                    current = previous;
                }

                metadata[0] = length;
                metadata[1] = 0;
            }
            ''', 'backtrace_parent')

        # A persistent search advances at most 2,000 waves.  Keep ample room
        # for specialized relaxations while bounding cycle/corruption damage.
        capacity = min(num_nodes, 16384)
        if (self._fullgraph_backtrace_buffer is None
                or self._fullgraph_backtrace_buffer.size < capacity):
            self._fullgraph_backtrace_buffer = cp.empty(capacity, dtype=cp.int32)
            self._fullgraph_backtrace_meta = cp.empty(2, dtype=cp.int32)

        self._fullgraph_backtrace_kernel(
            (1,), (1,),
            (
                best_key_gpu,
                cp.int32(best_dst),
                cp.int32(num_nodes),
                self._fullgraph_backtrace_buffer,
                cp.int32(capacity),
                self._fullgraph_backtrace_meta,
            ),
        )
        length, status = self._fullgraph_backtrace_meta.get().tolist()
        if status != 1 or length <= 0:
            logger.error(
                "[GPU-SEEDS] Device backtrace failed "
                f"(status={status}, length={length}, capacity={capacity})"
            )
            return None

        return self._fullgraph_backtrace_buffer[:length].get()[::-1].tolist()
