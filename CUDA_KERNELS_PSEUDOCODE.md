# CUDA Kernel Pseudocode for PCB Routing Pathfinding

**Companion to:** CUDA_DIJKSTRA_ARCHITECTURE.md
**Algorithm:** Near-Far Worklist (2-bucket delta-stepping)

---

## Kernel 1: `relax_near_bucket` (Parallel Edge Relaxation)

**Purpose:** Process all nodes in Near bucket, relax their outgoing edges in parallel.

**Thread Assignment:** One thread per node in Near bucket (across all K ROIs)

**Input:**
- `K`: Number of ROIs in batch
- `max_roi_size`: Maximum nodes per ROI (for padding)
- `max_edges`: Maximum edges per ROI
- `batch_indptr[K][max_roi_size+1]`: CSR row pointers (read-only)
- `batch_indices[K][max_edges]`: CSR column indices (read-only)
- `batch_weights[K][max_edges]`: Edge costs (read-only)
- `near_mask[K][max_roi_size]`: Near bucket membership (read-only)

**Output:**
- `dist[K][max_roi_size]`: Distance labels (read-write, atomic)
- `parent[K][max_roi_size]`: Predecessor pointers (read-write)
- `far_mask[K][max_roi_size]`: Far bucket membership (read-write)

**Pseudocode:**

```c
__global__ void relax_near_bucket(
    int K,
    int max_roi_size,
    int max_edges,
    const int* batch_indptr,
    const int* batch_indices,
    const float* batch_weights,
    const bool* near_mask,
    float* dist,
    int* parent,
    bool* far_mask
) {
    // Map thread to (roi_idx, node)
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int roi_idx = global_id / max_roi_size;
    int node = global_id % max_roi_size;

    // Bounds check
    if (roi_idx >= K || node >= max_roi_size) {
        return;
    }

    // Linearized index for 2D arrays
    int idx = roi_idx * max_roi_size + node;

    // Skip nodes not in Near bucket
    if (!near_mask[idx]) {
        return;
    }

    // Get this node's distance
    float u_dist = dist[idx];

    // Get CSR edge range for this node
    int indptr_offset = roi_idx * (max_roi_size + 1);
    int start = batch_indptr[indptr_offset + node];
    int end = batch_indptr[indptr_offset + node + 1];

    // Relax all outgoing edges
    int edge_offset = roi_idx * max_edges;
    for (int edge_idx = start; edge_idx < end; edge_idx++) {
        // Get neighbor and edge cost
        int v = batch_indices[edge_offset + edge_idx];
        float cost = batch_weights[edge_offset + edge_idx];
        float new_dist = u_dist + cost;

        // Atomic min-update for distance
        int v_idx = roi_idx * max_roi_size + v;
        float old_dist = atomicMinFloat(&dist[v_idx], new_dist);

        // If we improved distance, update parent and add to Far
        if (new_dist < old_dist) {
            parent[v_idx] = node;
            far_mask[v_idx] = true;
        }
    }
}

// Helper: Atomic min for float (not natively supported in CUDA)
__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;

    do {
        assumed = old;
        float old_float = __int_as_float(assumed);
        float new_float = fminf(old_float, value);
        int new_int = __float_as_int(new_float);
        old = atomicCAS(addr_as_int, assumed, new_int);
    } while (assumed != old);

    return __int_as_float(old);
}
```

**Key Implementation Details:**

1. **Thread Mapping:** `global_id = blockIdx.x * blockDim.x + threadIdx.x`
   - Threads are mapped to (ROI, node) pairs
   - `roi_idx = global_id / max_roi_size`
   - `node = global_id % max_roi_size`

2. **Memory Layout:** Row-major 2D arrays flattened to 1D
   - `batch_indptr[roi_idx][node]` → `batch_indptr[roi_idx * (max_roi_size + 1) + node]`
   - `dist[roi_idx][node]` → `dist[roi_idx * max_roi_size + node]`

3. **Atomic Operations:**
   - **Why needed:** Multiple threads may relax edges to same neighbor `v`
   - **`atomicMinFloat`:** Compare-and-swap loop (no native float atomic min)
   - **Contention:** Minimized by Near/Far bucketing (limits concurrent updates)

4. **Coalesced Memory Access:**
   - CSR arrays accessed sequentially within `for` loop
   - GPU warps (32 threads) read contiguous memory → high bandwidth

5. **Work Efficiency:**
   - Only processes nodes in Near bucket (active nodes)
   - Skips nodes with `near_mask[idx] == false`

**Complexity:** O(|Near| × avg_degree) work, O(1) depth per iteration

**Performance Estimate:**
- 100K nodes, avg degree 6 → 600K edges
- Near bucket: ~12.5K nodes per iteration (assuming uniform cost distribution)
- Edges relaxed: 12.5K × 6 = 75K edge operations
- Time: ~0.1 ms on RTX 3090 (memory-bound, coalesced access)

---

## Kernel 2: `split_near_far` (Bucket Re-assignment)

**Purpose:** After relaxation, re-bucket nodes based on updated distances vs threshold.

**Thread Assignment:** One thread per node (across all K ROIs)

**Input:**
- `K`: Number of ROIs
- `max_roi_size`: Max nodes per ROI
- `dist[K][max_roi_size]`: Updated distance labels (read-only)
- `threshold[K]`: Current distance threshold per ROI (read-only)
- `far_mask[K][max_roi_size]`: Far bucket membership (read-write)

**Output:**
- `near_mask[K][max_roi_size]`: Near bucket membership (write-only, cleared then refilled)

**Pseudocode:**

```c
__global__ void split_near_far(
    int K,
    int max_roi_size,
    const float* dist,
    const float* threshold,
    bool* near_mask,
    bool* far_mask
) {
    // Map thread to (roi_idx, node)
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int roi_idx = global_id / max_roi_size;
    int node = global_id % max_roi_size;

    // Bounds check
    if (roi_idx >= K || node >= max_roi_size) {
        return;
    }

    // Linearized index
    int idx = roi_idx * max_roi_size + node;

    // Get this node's distance and ROI threshold
    float d = dist[idx];
    float t = threshold[roi_idx];

    // Clear Near bucket (all nodes processed in previous iteration)
    near_mask[idx] = false;

    // Re-bucket nodes from Far based on threshold
    if (far_mask[idx]) {
        if (d < t) {
            // Distance below threshold → move to Near
            near_mask[idx] = true;
            far_mask[idx] = false;
        }
        // else: stays in Far
    }
}
```

**Key Implementation Details:**

1. **Threshold Semantics:**
   - `threshold[roi_idx]` = min(dist[v] for v in Far bucket)
   - Computed on host via CuPy reduction: `threshold = far_dists.min(axis=1)`

2. **Near Bucket Clearing:**
   - All Near nodes processed in previous iteration → clear Near
   - No synchronization needed (each thread handles independent node)

3. **Far → Near Migration:**
   - Nodes with `dist < threshold` move from Far to Near
   - These will be processed in next iteration

4. **Memory Access Pattern:**
   - Sequential access to `dist`, `threshold`, masks
   - Coalesced reads/writes → high memory bandwidth

**Complexity:** O(V) work, O(1) depth

**Performance Estimate:**
- 100K nodes, 8 ROIs → 800K threads
- Time: ~0.05 ms (memory-bound, simple operation)

---

## Kernel 3: `advance_threshold` (CuPy Reduction)

**Purpose:** When Near bucket is empty, advance threshold to minimum distance in Far bucket.

**Implementation:** No custom kernel needed - use CuPy reduction.

**Pseudocode (Python):**

```python
# Mask dist array: inf where not in Far, dist[v] where in Far
far_dists = cp.where(far_mask, dist, cp.inf)

# Find minimum per ROI (reduction along axis=1)
threshold = far_dists.min(axis=1)  # Shape: (K,)

# Check if any ROI has work remaining
active = (threshold < cp.inf)  # Shape: (K,), bool
```

**Key Implementation Details:**

1. **CuPy `cp.where`:**
   - Element-wise ternary operator: `far_mask ? dist : inf`
   - Output shape: `(K, max_roi_size)`

2. **CuPy `min(axis=1)`:**
   - Reduction along rows (per-ROI minimum)
   - Internally uses CUDA `cub::DeviceReduce` (highly optimized)

3. **Termination Check:**
   - If `threshold[roi_idx] == inf`, Far bucket is empty → ROI is done
   - Active ROIs: `threshold < inf`

**Complexity:** O(V) work, O(log V) depth (tree reduction)

**Performance Estimate:**
- 100K nodes, 8 ROIs → 800K elements
- Time: ~0.02 ms (CuPy optimized reduction)

---

## Kernel 4: `reconstruct_path` (CPU-based)

**Purpose:** Walk parent pointers backward from sink to source to build path.

**Implementation:** CPU-side (transfer parent array to host).

**Pseudocode (Python):**

```python
def reconstruct_path_batch(parent, dist, sources, sinks, K, max_roi_size):
    """
    Reconstruct paths for all ROIs in batch.

    Args:
        parent: (K, max_roi_size) int32 array (CPU NumPy)
        dist: (K, max_roi_size) float32 array (CPU NumPy)
        sources: List[int] - source node per ROI
        sinks: List[int] - sink node per ROI
        K: Number of ROIs
        max_roi_size: Max nodes per ROI

    Returns:
        List of paths (each path is List[int] of local ROI indices)
    """
    paths = []

    for roi_idx in range(K):
        sink = sinks[roi_idx]

        # Check if path exists
        if dist[roi_idx, sink] == np.inf:
            paths.append(None)
            continue

        # Walk backward from sink to source
        path = []
        curr = sink
        visited = set()  # Detect cycles

        while curr != -1:
            if curr in visited:
                # Cycle detected (should never happen with correct algorithm)
                logger.error(f"[GPU] Path reconstruction: cycle detected at node {curr}")
                paths.append(None)
                break

            path.append(curr)
            visited.add(curr)
            curr = parent[roi_idx, curr]

            # Safety limit
            if len(path) > max_roi_size:
                logger.error(f"[GPU] Path reconstruction: exceeded max_roi_size")
                paths.append(None)
                break
        else:
            # Reverse path (built backward)
            path.reverse()
            paths.append(path)

    return paths
```

**Key Implementation Details:**

1. **CPU vs GPU:**
   - Path reconstruction is sequential (inherently serial)
   - CPU is faster for this task (no GPU parallelism benefit)
   - Transfer `parent` and `dist` to CPU: `parent.get()`, `dist.get()`

2. **Cycle Detection:**
   - Should never happen with correct Dijkstra
   - Safety check to prevent infinite loops

3. **Path Length:**
   - Typical PCB paths: 50-500 nodes
   - Worst case: O(V) nodes

**Complexity:** O(path_length) per ROI, serial

**Performance Estimate:**
- 8 ROIs, avg path length 100 nodes
- Time: ~0.5 ms (CPU, negligible compared to search)

---

## Multi-Source Initialization (Portal Routing)

**Purpose:** Initialize Near bucket with multiple source seeds (18 portal layers).

**Implementation:** Extend initialization in `_prepare_batch`.

**Pseudocode (Python):**

```python
def _prepare_batch_multisource(self, roi_batch_multisource):
    """
    Prepare batch with multi-source initialization for portal routing.

    Args:
        roi_batch_multisource: List of (src_seeds, dst_targets, roi_indptr, ...)
            src_seeds: List[(node, initial_cost)] - multiple sources with entry costs
            dst_targets: List[node] - multiple possible destinations
    """
    K = len(roi_batch_multisource)

    # [Same array allocation as single-source]
    # ...

    sources_list = []
    sinks_list = []

    for i, (src_seeds, dst_targets, indptr, indices, weights, roi_size) in enumerate(roi_batch_multisource):
        # Transfer CSR data
        batch_indptr[i, :len(indptr)] = indptr
        batch_indices[i, :len(indices)] = indices
        batch_weights[i, :len(weights)] = weights

        # MULTI-SOURCE INITIALIZATION
        for (node, initial_cost) in src_seeds:
            dist[i, node] = initial_cost  # Discounted via cost (e.g., 0.45 for portals)
            near_mask[i, node] = True     # Add all sources to Near bucket

        # Store all destinations (terminate when ANY reached)
        sources_list.append([s[0] for s in src_seeds])
        sinks_list.append(dst_targets)

    return {
        # [Same as single-source, plus:]
        'sources': sources_list,  # List of lists
        'sinks': sinks_list       # List of lists
    }
```

**Termination Condition (Multi-Sink):**

```python
def _run_near_far_multisink(self, data, K):
    """Near-Far with multi-sink termination"""

    for iteration in range(max_iterations):
        # [Same as single-sink...]

        # Check if ANY destination reached for each ROI
        for roi_idx in range(K):
            if data['active'][roi_idx]:
                dst_targets = data['sinks'][roi_idx]
                for dst in dst_targets:
                    if data['dist'][roi_idx, dst] < cp.inf:
                        # Path found to one destination → mark ROI as done
                        data['active'][roi_idx] = False
                        break

        # Check global termination
        if not data['active'].any():
            break

    return self._reconstruct_paths_multisink(data, K)
```

**Key Differences from Single-Source:**

1. **Initialization:**
   - Multiple `dist[roi_idx, src] = initial_cost` updates
   - All sources added to Near bucket

2. **Termination:**
   - Early exit when **ANY** destination reached (not specific sink)
   - Portal routing: 18 sources → 18 possible destinations (across layers)

3. **Path Reconstruction:**
   - Walk backward from whichever destination was reached first
   - Return `(path, entry_layer, exit_layer)` for portal geometry

**Complexity:** Same as single-source (no algorithmic change)

---

## Launch Configuration Recommendations

### Block Size Selection

**Heuristic:** 256 threads per block (good balance for most kernels)

**Reasoning:**
- NVIDIA GPUs have 32-thread warps
- 256 threads = 8 warps per block
- Allows 4 blocks per SM (streaming multiprocessor) on most GPUs
- Good occupancy without excessive register pressure

**Tuning:** Benchmark with 128, 256, 512 threads per block.

### Grid Size Calculation

```python
threads_per_block = 256
total_threads = K * max_roi_size
num_blocks = (total_threads + threads_per_block - 1) // threads_per_block

# Example: K=8, max_roi_size=100K
total_threads = 8 * 100,000 = 800,000
num_blocks = 800,000 / 256 = 3,125 blocks
```

**Launch:**
```python
kernel[num_blocks, threads_per_block](args...)
```

### Memory Bandwidth Optimization

**Coalescing:**
- Threads in a warp access contiguous memory addresses
- CSR arrays stored in row-major order (C-style)
- Flattened 2D arrays ensure coalesced access

**Example:**
```
Thread 0: Accesses dist[0]   (roi_idx=0, node=0)
Thread 1: Accesses dist[1]   (roi_idx=0, node=1)
...
Thread 31: Accesses dist[31] (roi_idx=0, node=31)
→ Single 128-byte memory transaction (32 × 4 bytes)
```

---

## Kernel Complexity Summary

| Kernel | Work | Depth | Memory | Bottleneck |
|--------|------|-------|--------|------------|
| `relax_near_bucket` | O(E_near) | O(1) | 2 MB read + 1 MB write | Compute (atomics) |
| `split_near_far` | O(V) | O(1) | 1 MB read + 0.5 MB write | Memory-bound |
| `advance_threshold` | O(V) | O(log V) | 1 MB read | Memory-bound |
| Path reconstruction | O(L) | O(L) | 0.4 MB read | CPU serial |

**Total per iteration:**
- Work: O(E + V) (dominated by edge relaxation)
- Depth: O(1) (fully parallel)
- Memory: ~5 MB read/write per iteration

**Expected iterations:** O(max_cost / min_cost) = O(3.0 / 0.4) = ~8 iterations

---

## Atomic Operation Optimization

**Problem:** `atomicMinFloat` can cause contention if many threads update same node.

**Mitigation Strategies:**

1. **Near-Far Bucketing:**
   - Limits concurrent updates (only nodes in current Near frontier)
   - Reduces contention by ~10-100× vs unbucketed approach

2. **Atomic-Free Alternatives (Future):**
   - Use per-thread local minima, then reduce (2-phase approach)
   - More complex but eliminates atomics entirely

3. **Hardware Support:**
   - NVIDIA Volta+ has native `atomicMin` for floats (use if available)

**Performance Impact:**
- Worst case: 5× slowdown due to atomic contention
- Typical case: 2× slowdown (sparse graphs, limited contention)

---

## Memory Layout Diagram

```
GPU Memory Layout (Batch Size K=4, max_roi_size=8, max_edges=16)

batch_indptr: (4, 9) int32
┌─────────────────────────────────────────────┐
│ ROI 0: [0, 2, 4, 6, 8, 10, 12, 14, 16]     │ CSR row pointers
│ ROI 1: [0, 1, 3, 5, 7,  9, 11, 13, 15]     │
│ ROI 2: [0, 2, 4, 6, 8, 10, 12, 14, 16]     │
│ ROI 3: [0, 3, 6, 9, 12, 14, 15, 16, 16]    │
└─────────────────────────────────────────────┘

batch_indices: (4, 16) int32
┌─────────────────────────────────────────────┐
│ ROI 0: [1, 2, 3, 4, 5, 6, 7, 0, ...]       │ CSR column indices
│ ROI 1: [2, 3, 4, 5, 6, 7, 0, 1, ...]       │
│ ROI 2: [1, 3, 2, 4, 5, 6, 7, 0, ...]       │
│ ROI 3: [1, 2, 3, 4, 5, 6, 7, 0, ...]       │
└─────────────────────────────────────────────┘

batch_weights: (4, 16) float32
┌─────────────────────────────────────────────┐
│ ROI 0: [0.4, 0.4, 0.4, 3.0, 0.4, ...]      │ Edge costs
│ ROI 1: [0.4, 0.4, 3.0, 0.4, 0.4, ...]      │
│ ROI 2: [0.4, 0.4, 0.4, 0.4, 0.4, ...]      │
│ ROI 3: [0.4, 3.0, 0.4, 0.4, 0.4, ...]      │
└─────────────────────────────────────────────┘

dist: (4, 8) float32
┌─────────────────────────────────────────────┐
│ ROI 0: [0.0, inf, inf, inf, inf, inf, ...] │ Distance labels
│ ROI 1: [0.0, inf, inf, inf, inf, inf, ...] │
│ ROI 2: [0.0, inf, inf, inf, inf, inf, ...] │
│ ROI 3: [0.0, inf, inf, inf, inf, inf, ...] │
└─────────────────────────────────────────────┘

near_mask: (4, 8) bool
┌─────────────────────────────────────────────┐
│ ROI 0: [T, F, F, F, F, F, F, F]            │ Near bucket
│ ROI 1: [T, F, F, F, F, F, F, F]            │
│ ROI 2: [T, F, F, F, F, F, F, F]            │
│ ROI 3: [T, F, F, F, F, F, F, F]            │
└─────────────────────────────────────────────┘

far_mask: (4, 8) bool
┌─────────────────────────────────────────────┐
│ ROI 0: [F, F, F, F, F, F, F, F]            │ Far bucket
│ ROI 1: [F, F, F, F, F, F, F, F]            │
│ ROI 2: [F, F, F, F, F, F, F, F]            │
│ ROI 3: [F, F, F, F, F, F, F, F]            │
└─────────────────────────────────────────────┘

Total Memory: 4 × (9×4 + 16×4 + 16×4 + 8×4 + 8×4 + 8×1 + 8×1)
            = 4 × (36 + 64 + 64 + 32 + 32 + 8 + 8)
            = 4 × 244 bytes
            = 976 bytes (for 4 tiny ROIs)

Realistic: K=8, max_roi_size=100K, max_edges=600K
         = 8 × (100K×4 + 100K×4 + 600K×4 + 600K×4 + 100K×1 + 100K×1)
         = 8 × (0.4MB + 0.4MB + 2.4MB + 2.4MB + 0.1MB + 0.1MB)
         = 8 × 5.8 MB
         = 46.4 MB (easily fits on modern GPUs)
```

---

**END OF PSEUDOCODE DOCUMENT**
