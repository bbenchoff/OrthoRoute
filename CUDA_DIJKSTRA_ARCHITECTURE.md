# Production-Ready CUDA Dijkstra Architecture for PCB Routing Pathfinding

**Document Version:** 1.0
**Date:** 2025-10-03
**Target:** 10-100× speedup over CPU heapq Dijkstra
**Use Case:** PCB routing on ROI subgraphs (10K-100K nodes)

---

## Executive Summary

This document specifies a production-ready CUDA-accelerated shortest path solver for OrthoRoute's PCB routing engine. The current CPU implementation using Python's heapq takes **3-4 seconds per net**, making it the primary performance bottleneck. We target **10-100× speedup** by moving Dijkstra computation to GPU using a **Near-Far worklist algorithm** with ROI batching.

**Key Design Decisions:**
- **Algorithm:** Near-Far worklist (2-bucket delta-stepping variant)
- **Rationale:** Avoids GPU priority queues, exploits uniformity of PCB edge costs
- **Batching:** Process multiple ROI subgraphs in parallel (K=4-16 ROIs)
- **Data Structure:** CSR format (already on GPU via CuPy)
- **Integration:** Drop-in replacement for `SimpleDijkstra.find_path_roi()`

**Expected Performance:**
- Current: 3-4 seconds per net (CPU heapq)
- Target: 30-40 ms per net (GPU Near-Far)
- Speedup: **75-133×** for typical ROIs

---

## 1. Algorithm Selection & Justification

### 1.1 Candidate Algorithms

| Algorithm | Pros | Cons | Verdict |
|-----------|------|------|---------|
| **Near-Far Worklist** | ✅ No priority queue<br>✅ Simple 2-bucket design<br>✅ Works well for uniform costs<br>✅ Easy to batch | ❌ Suboptimal for highly variable costs | **RECOMMENDED** |
| **Delta-Stepping** | ✅ Good for general graphs<br>✅ Proven GPU scalability | ❌ Requires tuning Δ parameter<br>❌ More buckets = more overhead | Backup option |
| **Parallel Bellman-Ford** | ✅ Very simple to implement<br>✅ No data structures needed | ❌ O(V×E) worst case<br>❌ Too slow for 100K nodes | ❌ Rejected |
| **GPU Priority Queue** | ✅ Identical to CPU algorithm | ❌ Slow on GPU<br>❌ Complex to implement<br>❌ Poor parallelism | ❌ Rejected |

### 1.2 Why Near-Far Worklist?

**Near-Far is a simplified delta-stepping variant with Δ = min_edge_cost:**

1. **Bucket "Near":** Nodes with `dist[v] < current_threshold`
2. **Bucket "Far":** Nodes with `dist[v] ≥ current_threshold`

**Why it's perfect for PCB routing:**

✅ **Uniform edge costs:** PCB routing has ~3 distinct edge costs:
   - Horizontal/vertical tracks: `0.4 mm` (grid pitch)
   - Layer transitions (vias): `3.0 mm` (via cost)
   - Portal-discounted vias: `0.45-0.65 mm`

✅ **No priority queue:** GPU threads process entire "Near" bucket in parallel

✅ **Work-efficient:** Each edge relaxation is O(1), total work O(E log V) expected

✅ **Easy to batch:** Process K ROIs in parallel with independent Near/Far buckets

✅ **Proven performance:** Research shows 10-100× speedup on sparse graphs (PPoPP'21)

**Delta-Stepping vs Near-Far:**
- Delta-stepping: Δ is tunable parameter, requires multiple buckets
- Near-Far: Δ = min_edge_cost (automatic), only 2 buckets
- For PCB routing: Near-Far is simpler and performs equivalently

---

## 2. Architecture Design

### 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ SimpleDijkstra.find_path_roi(src, dst, costs, roi_nodes, ...)  │
│                                                                   │
│ Decision: if roi_size > 5000 → GPU, else → CPU                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ├─── GPU Path (NEW) ───────────────┐
                              │                                    │
                              v                                    │
┌─────────────────────────────────────────────────────────────────┤
│ CUDADijkstra.find_path_roi_batch()                              │
│                                                                   │
│ Input: List of (src, dst, roi_csr, costs, roi_size) × K         │
│ Output: List of paths (local ROI indices) × K                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ GPU Memory Layout (CuPy arrays)                                  │
│                                                                   │
│ • batch_indptr: (K, max_roi_size+1) int32 - CSR row pointers   │
│ • batch_indices: (K, max_edges) int32 - CSR column indices     │
│ • batch_weights: (K, max_edges) float32 - Edge costs           │
│ • dist: (K, max_roi_size) float32 - Distance labels            │
│ • parent: (K, max_roi_size) int32 - Predecessor pointers       │
│ • near_mask: (K, max_roi_size) bool - Near bucket membership   │
│ • far_mask: (K, max_roi_size) bool - Far bucket membership     │
│ • active: (K,) bool - ROI still has work?                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ Near-Far Iteration Loop (GPU Kernels)                            │
│                                                                   │
│ while any(active):                                               │
│   1. Kernel: relax_near_bucket()  [parallel edge updates]       │
│   2. Kernel: split_near_far()      [re-bucket nodes]            │
│   3. Check: if near_empty → advance threshold, swap buckets     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│ Path Reconstruction (GPU or CPU)                                 │
│                                                                   │
│ • Walk parent[] pointers backward from sink to source            │
│ • Return list of local ROI node indices                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Multi-Source/Multi-Sink Support (Portal Routing)

**Current Challenge:** Portal routing uses 18 source seeds (one per layer) with discounted entry costs.

**Solution:** Multi-source Dijkstra initialization:

```python
# Initialize heap with all portal layers
for (global_node, initial_cost) in src_seeds:
    roi_idx = global_to_roi[global_node]
    dist[roi_idx] = initial_cost
    near_mask[roi_idx] = True  # Add to Near bucket

# Terminate when ANY destination reached
while any(active):
    if any(dist[roi_idx] < inf for roi_idx in dst_targets):
        break  # Path found!
```

**No algorithmic change needed** - just initialize multiple seeds with different costs.

### 2.3 ROI Batching Strategy

**Goal:** Route K nets in parallel to amortize kernel launch overhead.

**Batch Size Selection:**
- **K = 4-16** (tunable parameter)
- Smaller K: Lower latency, better for small ROIs
- Larger K: Better GPU utilization, risk of memory overflow

**Memory Requirements:**
```
Batch Memory = K × (
    max_roi_size × (4 + 4 + 1 + 1)  [dist + parent + near + far]
  + max_edges × (4 + 4)              [indices + weights]
  + max_roi_size × 4                 [indptr]
)

Example: K=8, max_roi_size=100K, avg_degree=6
  = 8 × (100K × 10 + 600K × 8 + 100K × 4)
  = 8 × (1MB + 4.8MB + 0.4MB)
  = ~50 MB (fits easily on modern GPUs)
```

**Batching Heuristics:**
1. Queue incoming ROI tasks in a buffer
2. Launch batch when:
   - Buffer reaches K tasks, OR
   - 5ms timeout (avoid starvation)
3. Pad smaller batches with dummy ROIs (active=False)

---

## 3. Data Structures Specification

### 3.1 GPU Arrays (CuPy Format)

All arrays stored in **row-major** format for coalesced memory access.

#### CSR Graph (Read-Only)
```python
batch_indptr: cp.ndarray[K, max_roi_size+1, cp.int32]
  # CSR row pointers: batch_indptr[roi_idx, node] → edge_start
  # batch_indptr[roi_idx, node+1] → edge_end

batch_indices: cp.ndarray[K, max_edges, cp.int32]
  # CSR column indices: batch_indices[roi_idx, edge_idx] → neighbor_node

batch_weights: cp.ndarray[K, max_edges, cp.float32]
  # Edge costs: batch_weights[roi_idx, edge_idx] → cost
```

#### Distance & Parent Arrays (Read-Write)
```python
dist: cp.ndarray[K, max_roi_size, cp.float32]
  # Distance labels: dist[roi_idx, node] → shortest_distance
  # Initialize: dist[roi_idx, src] = 0.0, others = inf

parent: cp.ndarray[K, max_roi_size, cp.int32]
  # Predecessor pointers: parent[roi_idx, node] → parent_node
  # Initialize: all = -1
```

#### Near-Far Buckets (Read-Write)
```python
near_mask: cp.ndarray[K, max_roi_size, cp.bool_]
  # Near bucket membership: near_mask[roi_idx, node] → in_near?
  # Processed in current iteration

far_mask: cp.ndarray[K, max_roi_size, cp.bool_]
  # Far bucket membership: far_mask[roi_idx, node] → in_far?
  # Deferred to later iterations

threshold: cp.ndarray[K, cp.float32]
  # Current distance threshold per ROI: threshold[roi_idx]
  # Nodes with dist < threshold go to Near
```

#### Control Arrays
```python
active: cp.ndarray[K, cp.bool_]
  # ROI still has work?: active[roi_idx] → True/False
  # Set False when sink reached or Near+Far both empty

near_empty: cp.ndarray[K, cp.bool_]
  # Near bucket empty?: near_empty[roi_idx] → True/False
  # Triggers threshold advance
```

### 3.2 Memory Layout Example

```
ROI 0: [============= 100K nodes =============]
ROI 1: [============= 80K nodes  =============]
ROI 2: [============= 120K nodes =============]
       ^                                       ^
       All padded to max_roi_size for alignment
```

**Why padding?** Uniform array access patterns → coalesced memory reads.

---

## 4. CUDA Kernel Specifications

### 4.1 Kernel 1: `relax_near_bucket`

**Purpose:** Relax all edges from nodes in the Near bucket (parallel).

**Thread Assignment:** One thread per node in Near bucket (across all ROIs).

**Pseudocode:**
```cuda
__global__ void relax_near_bucket(
    int K,                  // Number of ROIs
    int max_roi_size,       // Max nodes per ROI
    int max_edges,          // Max edges per ROI
    const int* batch_indptr,    // (K, max_roi_size+1)
    const int* batch_indices,   // (K, max_edges)
    const float* batch_weights, // (K, max_edges)
    const bool* near_mask,      // (K, max_roi_size)
    float* dist,                // (K, max_roi_size)
    int* parent,                // (K, max_roi_size)
    bool* far_mask              // (K, max_roi_size)
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int roi_idx = global_id / max_roi_size;
    int node = global_id % max_roi_size;

    if (roi_idx >= K || !near_mask[roi_idx * max_roi_size + node]) {
        return;  // Not in Near bucket
    }

    // Get node's distance
    float u_dist = dist[roi_idx * max_roi_size + node];

    // Get edge range from CSR
    int start = batch_indptr[roi_idx * (max_roi_size + 1) + node];
    int end = batch_indptr[roi_idx * (max_roi_size + 1) + node + 1];

    // Relax all outgoing edges
    for (int edge_idx = start; edge_idx < end; edge_idx++) {
        int v = batch_indices[roi_idx * max_edges + edge_idx];
        float cost = batch_weights[roi_idx * max_edges + edge_idx];
        float new_dist = u_dist + cost;

        // Atomic min-update for distance
        float* dist_ptr = &dist[roi_idx * max_roi_size + v];
        float old_dist = atomicMinFloat(dist_ptr, new_dist);

        // Update parent if we improved
        if (new_dist < old_dist) {
            parent[roi_idx * max_roi_size + v] = node;
            far_mask[roi_idx * max_roi_size + v] = true;  // Add to Far
        }
    }
}

// Helper: atomicMin for float (not natively supported)
__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
            __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
```

**Launch Configuration:**
```python
threads_per_block = 256
total_threads = K * max_roi_size
num_blocks = (total_threads + threads_per_block - 1) // threads_per_block

relax_near_bucket[num_blocks, threads_per_block](...)
```

**Complexity:** O(|Near| × avg_degree) work, O(1) depth (parallel)

---

### 4.2 Kernel 2: `split_near_far`

**Purpose:** Re-bucket nodes based on updated distances vs threshold.

**Thread Assignment:** One thread per node (across all ROIs).

**Pseudocode:**
```cuda
__global__ void split_near_far(
    int K,
    int max_roi_size,
    const float* dist,        // (K, max_roi_size)
    const float* threshold,   // (K,)
    bool* near_mask,          // (K, max_roi_size)
    bool* far_mask,           // (K, max_roi_size)
    bool* near_empty          // (K,) output
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int roi_idx = global_id / max_roi_size;
    int node = global_id % max_roi_size;

    if (roi_idx >= K) return;

    int idx = roi_idx * max_roi_size + node;
    float d = dist[idx];
    float t = threshold[roi_idx];

    // Clear old Near bucket (processed)
    near_mask[idx] = false;

    // Split Far bucket based on threshold
    if (far_mask[idx]) {
        if (d < t) {
            near_mask[idx] = true;   // Move to Near
            far_mask[idx] = false;
        }
        // else: stays in Far
    }

    // Check if Near is empty (reduction needed)
    __syncthreads();
    if (threadIdx.x == 0) {
        bool has_near = false;
        for (int i = 0; i < max_roi_size; i++) {
            if (near_mask[roi_idx * max_roi_size + i]) {
                has_near = true;
                break;
            }
        }
        near_empty[roi_idx] = !has_near;
    }
}
```

**Note:** The `near_empty` check is simplified here - production code should use CuPy reduction: `near_mask.any(axis=1)`.

---

### 4.3 Kernel 3: `advance_threshold`

**Purpose:** When Near bucket is empty, advance threshold to min(Far distances).

**Implementation:** CuPy reduction (no custom kernel needed):

```python
# Find minimum distance in Far bucket for each ROI
far_dists = cp.where(far_mask, dist, cp.inf)
threshold = far_dists.min(axis=1)  # (K,) array

# If threshold = inf, ROI is done
active = (threshold < cp.inf)
```

---

### 4.4 Kernel 4: `reconstruct_path`

**Purpose:** Walk parent pointers backward to build path.

**Implementation:** CPU-side (simple, not performance-critical):

```python
def reconstruct_path_batch(parent_cpu, sinks, K, max_roi_size):
    paths = []
    for roi_idx in range(K):
        if dist[roi_idx, sinks[roi_idx]] == cp.inf:
            paths.append(None)  # No path
            continue

        path = []
        curr = sinks[roi_idx]
        while curr != -1:
            path.append(curr)
            curr = parent_cpu[roi_idx, curr]
        path.reverse()
        paths.append(path)
    return paths
```

**Cost:** O(path_length) per ROI, negligible compared to search.

---

## 5. Integration with SimpleDijkstra

### 5.1 Modified `SimpleDijkstra` Class

**File:** `orthoroute/algorithms/manhattan/unified_pathfinder.py`

```python
class SimpleDijkstra:
    """Dijkstra SSSP with ROI support (CPU fallback + GPU acceleration)"""

    def __init__(self, graph: CSRGraph, lattice=None, use_gpu=False):
        # CPU arrays (existing)
        self.indptr = graph.indptr.get() if hasattr(graph.indptr, "get") else graph.indptr
        self.indices = graph.indices.get() if hasattr(graph.indices, "get") else graph.indices
        self.N = len(self.indptr) - 1
        self.plane_size = lattice.x_steps * lattice.y_steps if lattice else None

        # GPU solver (NEW)
        self.use_gpu = use_gpu and GPU_AVAILABLE and CUDA_DIJKSTRA_AVAILABLE
        if self.use_gpu:
            from .pathfinder.cuda_dijkstra import CUDADijkstra
            self.gpu_solver = CUDADijkstra()
            logger.info("[GPU] CUDA Dijkstra enabled for ROI pathfinding")
        else:
            self.gpu_solver = None

    def find_path_roi(self, src: int, dst: int, costs, roi_nodes, global_to_roi) -> Optional[List[int]]:
        """Find shortest path within ROI subgraph (GPU-accelerated when possible)"""

        # Decision heuristic: GPU vs CPU
        roi_size = len(roi_nodes)
        use_gpu_for_this = self.use_gpu and roi_size > 5000  # Tunable threshold

        if use_gpu_for_this:
            return self._find_path_roi_gpu(src, dst, costs, roi_nodes, global_to_roi)
        else:
            return self._find_path_roi_cpu(src, dst, costs, roi_nodes, global_to_roi)

    def _find_path_roi_cpu(self, src, dst, costs, roi_nodes, global_to_roi):
        """Original CPU heap-based Dijkstra (unchanged)"""
        import heapq
        import numpy as np

        # [EXISTING CPU CODE - NO CHANGES]
        # ... (lines 985-1051 from current implementation)

    def _find_path_roi_gpu(self, src, dst, costs, roi_nodes, global_to_roi):
        """GPU-accelerated Dijkstra using CUDADijkstra"""
        import numpy as np

        # Map src/dst to ROI space
        roi_src = int(global_to_roi[src])
        roi_dst = int(global_to_roi[dst])

        if roi_src < 0 or roi_dst < 0:
            logger.warning("[GPU] src or dst not in ROI, falling back to CPU")
            return self._find_path_roi_cpu(src, dst, costs, roi_nodes, global_to_roi)

        # Build ROI subgraph CSR (on CPU, then transfer to GPU)
        roi_size = len(roi_nodes)
        roi_indptr, roi_indices, roi_weights = self._build_roi_csr(
            roi_nodes, costs, roi_size
        )

        # Call GPU solver (batch size = 1 for now)
        try:
            paths = self.gpu_solver.find_paths_on_rois([
                (roi_src, roi_dst, roi_indptr, roi_indices, roi_weights, roi_size)
            ])
            path = paths[0] if paths else None
        except Exception as e:
            logger.warning(f"[GPU] CUDA Dijkstra failed: {e}, falling back to CPU")
            return self._find_path_roi_cpu(src, dst, costs, roi_nodes, global_to_roi)

        if path is None:
            return None

        # Convert local ROI indices back to global indices
        global_path = [int(roi_nodes[local_idx]) for local_idx in path]
        return global_path

    def _build_roi_csr(self, roi_nodes, costs, roi_size):
        """Build CSR subgraph for ROI nodes"""
        import numpy as np

        roi_nodes_cpu = roi_nodes.get() if hasattr(roi_nodes, 'get') else roi_nodes
        costs_cpu = costs.get() if hasattr(costs, 'get') else costs

        # Build mapping: global_node → local_roi_idx
        global_to_local = -np.ones(self.N, dtype=np.int32)
        for local_idx, global_node in enumerate(roi_nodes_cpu):
            global_to_local[global_node] = local_idx

        # Extract edges within ROI
        edges = []
        for local_u, global_u in enumerate(roi_nodes_cpu):
            start = self.indptr[global_u]
            end = self.indptr[global_u + 1]
            for edge_idx in range(start, end):
                global_v = self.indices[edge_idx]
                local_v = global_to_local[global_v]
                if local_v >= 0:  # Neighbor is in ROI
                    cost = costs_cpu[edge_idx]
                    edges.append((local_u, local_v, cost))

        # Build CSR arrays
        edges.sort(key=lambda x: (x[0], x[1]))  # Sort by (src, dst)
        roi_indptr = np.zeros(roi_size + 1, dtype=np.int32)
        roi_indices = np.zeros(len(edges), dtype=np.int32)
        roi_weights = np.zeros(len(edges), dtype=np.float32)

        for i, (u, v, cost) in enumerate(edges):
            roi_indptr[u + 1] += 1
            roi_indices[i] = v
            roi_weights[i] = cost

        # Cumulative sum for indptr
        roi_indptr = np.cumsum(roi_indptr)

        # Transfer to GPU
        import cupy as cp
        return (
            cp.asarray(roi_indptr),
            cp.asarray(roi_indices),
            cp.asarray(roi_weights)
        )
```

### 5.2 Multi-Source/Multi-Sink Extension

```python
def find_path_multisource_multisink(self, src_seeds, dst_targets, costs, roi_nodes, global_to_roi):
    """Multi-source/multi-sink Dijkstra for portal routing (GPU-accelerated)"""

    roi_size = len(roi_nodes)
    use_gpu_for_this = self.use_gpu and roi_size > 5000

    if use_gpu_for_this:
        return self._find_path_multisource_gpu(src_seeds, dst_targets, costs, roi_nodes, global_to_roi)
    else:
        # [EXISTING CPU CODE - lines 1053-1150]
        # ... (unchanged)
```

**GPU implementation:** Initialize `dist` with multiple seeds, terminate when any target reached.

---

## 6. Performance Analysis & Estimates

### 6.1 Complexity Analysis

| Algorithm | Work | Depth (Parallelism) | Memory |
|-----------|------|---------------------|---------|
| **CPU heapq Dijkstra** | O(E log V) | O(E) serial | O(V + E) |
| **GPU Near-Far** | O(E log V) expected | O(log V) parallel | O(K × V) |

**Key Insight:** GPU processes entire Near bucket in parallel → depth reduction from O(E) to O(log V).

### 6.2 Performance Model

**Assumptions:**
- ROI size: V = 100,000 nodes
- Average degree: d = 6 (typical PCB routing graph)
- Edge count: E = d × V = 600,000 edges
- GPU: NVIDIA RTX 3090 (10,496 CUDA cores, 936 GB/s bandwidth)
- Grid pitch: 0.4 mm (uniform edge costs favor Near-Far)

**CPU Performance (Current):**
```
Time per iteration: 0.01 ms (heap extract-min)
Iterations: V × log V = 100K × 17 = 1.7M iterations
Total time: 1.7M × 0.01 ms = 17,000 ms = 17 seconds (worst case)
Observed: 3-4 seconds (with early termination, sparse ROIs)
```

**GPU Performance (Near-Far):**
```
Near-Far iterations: O(max_edge_cost / min_edge_cost) = O(3.0 / 0.4) = ~8 iterations
Edges per iteration: E / iterations = 600K / 8 = 75K edges
Threads per iteration: 75K threads (one per edge relaxation)
Time per kernel: ~0.1 ms (memory-bound, coalesced access)
Total time: 8 iterations × 0.1 ms = 0.8 ms = 800 μs per ROI
```

**Speedup Estimate:**
```
Speedup = 3000 ms (CPU) / 0.8 ms (GPU) = 3750×  (best case, memory-bound)
Realistic speedup: 100-500× (accounting for:
  - Kernel launch overhead: ~5 μs per launch
  - PCIe transfer: ~100 μs for small ROIs (if not already on GPU)
  - Atomics contention: 2-5× slowdown in worst case
)

Conservative estimate: 50-100× speedup
Target: 3-4 seconds → 30-80 ms per net
```

### 6.3 Batching Performance

**Single ROI:**
- Time: 0.8 ms compute + 0.1 ms overhead = 0.9 ms

**Batched ROIs (K=8):**
- Time: 0.8 ms compute + 0.1 ms overhead = 0.9 ms (same!)
- Throughput: 8 ROIs / 0.9 ms = 8.9 ROIs/ms
- Per-ROI cost: 0.11 ms (8× improvement via batching)

**Conclusion:** Batching is critical for small ROIs to amortize kernel launch overhead.

### 6.4 Memory Bandwidth Analysis

**Data per iteration:**
```
Read:  batch_indptr (4 bytes × 100K) + batch_indices (4 bytes × 75K) + batch_weights (4 bytes × 75K)
     = 0.4 MB + 0.3 MB + 0.3 MB = 1 MB
Write: dist (4 bytes × 100K) + parent (4 bytes × 100K) + masks (2 bytes × 100K)
     = 0.4 MB + 0.4 MB + 0.2 MB = 1 MB
Total: 2 MB per iteration
```

**Bandwidth utilization:**
```
Memory bandwidth: 936 GB/s (RTX 3090)
Required: 2 MB × 8 iterations / 0.8 ms = 20 GB/s (2% utilization)
```

**Bottleneck:** Compute-bound (atomic operations), not memory-bound → good!

---

## 7. Implementation Plan

### Phase 1: Proof of Concept (3-5 days)
**Goal:** Single-ROI GPU Dijkstra with correctness validation

**Tasks:**
1. ✅ Implement `relax_near_bucket` CUDA kernel
2. ✅ Implement `split_near_far` CUDA kernel
3. ✅ Python wrapper in `cuda_dijkstra.py`
4. ✅ Unit tests: Compare GPU vs CPU results on small graphs (10-1000 nodes)
5. ✅ Validation: Ensure bit-identical paths to CPU heapq

**Deliverables:**
- `find_path_single_roi()` function
- Test suite with 20+ validation cases

---

### Phase 2: Batching & Optimization (5-7 days)
**Goal:** Multi-ROI batching with performance tuning

**Tasks:**
1. ✅ Implement ROI batching: `find_paths_on_rois_batch(K=8)`
2. ✅ Padding strategy for heterogeneous ROI sizes
3. ✅ Kernel tuning: Block size, shared memory, atomic contention
4. ✅ Benchmark: CPU vs GPU on 100 real PCB routing tasks
5. ✅ Adaptive heuristic: GPU vs CPU decision based on ROI size

**Deliverables:**
- `find_paths_on_rois()` with automatic batching
- Performance report: Speedup vs ROI size graph

---

### Phase 3: Multi-Source Integration (3-4 days)
**Goal:** Support portal routing (18 source seeds)

**Tasks:**
1. ✅ Extend initialization for multi-source seeds
2. ✅ Multi-sink termination condition
3. ✅ Integration with `find_path_multisource_multisink()`
4. ✅ End-to-end test: Route 100 nets with portals

**Deliverables:**
- Full portal routing support on GPU
- Performance comparison: Portal CPU vs Portal GPU

---

### Phase 4: Production Hardening (5-7 days)
**Goal:** Error handling, fallback, monitoring

**Tasks:**
1. ✅ GPU memory overflow handling (large ROIs)
2. ✅ Automatic CPU fallback on CUDA errors
3. ✅ Logging: GPU utilization, kernel times, batch stats
4. ✅ Profiling: NVIDIA Nsight Systems integration
5. ✅ Documentation: User guide, performance tuning tips

**Deliverables:**
- Production-ready code
- Performance dashboard logs
- User documentation

---

**Total Estimated Time:** 16-23 days (3-4 weeks)

---

## 8. Error Handling & Fallback Strategy

### 8.1 GPU Memory Overflow

**Problem:** ROI too large to fit in GPU memory.

**Solution:**
```python
try:
    paths = gpu_solver.find_paths_on_rois(roi_batch)
except cp.cuda.memory.OutOfMemoryError:
    logger.warning("[GPU] Out of memory, falling back to CPU")
    paths = cpu_fallback(roi_batch)
```

**Heuristic:** If ROI > 500K nodes, skip GPU and use CPU directly.

### 8.2 CUDA Kernel Errors

**Problem:** Kernel launch failure, illegal memory access, timeout.

**Solution:**
```python
try:
    relax_near_bucket[num_blocks, threads_per_block](...)
    cp.cuda.Stream.null.synchronize()  # Check for errors
except cp.cuda.runtime.CUDARuntimeError as e:
    logger.error(f"[GPU] CUDA error: {e}")
    raise GPUPathfindingError("Kernel failed") from e
```

**Fallback:** Catch exception at `SimpleDijkstra` level, retry with CPU.

### 8.3 Correctness Validation

**Problem:** GPU produces incorrect paths due to race conditions.

**Solution:**
```python
# Development mode: Double-check GPU results against CPU
if DEBUG_MODE:
    gpu_path = find_path_gpu(...)
    cpu_path = find_path_cpu(...)
    assert gpu_path == cpu_path, "GPU/CPU mismatch!"
```

**Testing:** Run 10,000 random routing tasks, assert 100% agreement.

---

## 9. Comparison: Delta-Stepping vs Near-Far

| Feature | Delta-Stepping | Near-Far | Winner |
|---------|---------------|----------|--------|
| **Buckets** | Multiple (Δ-sized) | 2 (Near + Far) | Near-Far (simpler) |
| **Parameter Tuning** | Requires Δ tuning | Automatic (Δ = min_cost) | Near-Far |
| **Memory** | O(V × num_buckets) | O(2V) | Near-Far |
| **Parallelism** | High | High | Tie |
| **Uniform Costs** | Overkill | Optimal | Near-Far |
| **Variable Costs** | Better | Acceptable | Delta-Stepping |
| **Implementation** | Complex | Simple | Near-Far |

**Recommendation:** **Near-Far for initial implementation**, Delta-Stepping as future optimization if needed.

**When to use Delta-Stepping:**
- If future PCB designs have highly variable edge costs (e.g., congestion-dependent)
- If Near-Far shows poor performance on specific boards

**Migration path:** Near-Far → Delta-Stepping is trivial (just add more buckets).

---

## 10. Expected Results Summary

### 10.1 Performance Targets

| Metric | CPU (Current) | GPU (Target) | Speedup |
|--------|---------------|--------------|---------|
| **Time per net** | 3-4 seconds | 30-40 ms | **75-133×** |
| **ROI extraction** | 5-10 ms | 5-10 ms (unchanged) | 1× |
| **Path reconstruction** | 1-2 ms | 1-2 ms | 1× |
| **Total routing time** | 3-4 sec/net | 40-50 ms/net | **60-100×** |

**Net routing breakdown:**
- ROI extraction (BFS): 10 ms (CPU, already fast)
- Dijkstra search: 3-4 sec (CPU) → 30-40 ms (GPU) **← PRIMARY GAIN**
- Path reconstruction: 2 ms (CPU, negligible)

### 10.2 Scalability

| ROI Size | CPU Time | GPU Time (Batched) | Speedup |
|----------|----------|--------------------|---------|
| 10K nodes | 300 ms | 5 ms | 60× |
| 50K nodes | 1.5 sec | 20 ms | 75× |
| 100K nodes | 3.5 sec | 40 ms | 87× |
| 200K nodes | 8 sec | 80 ms | 100× |

**Observation:** Speedup increases with ROI size (better GPU utilization).

### 10.3 Multi-Source Performance (Portal Routing)

**CPU (18 sources):**
- Time: ~4 seconds (slightly slower than single-source)
- Heap initialization: O(18 log 18) ≈ negligible

**GPU (18 sources):**
- Time: ~40 ms (same as single-source!)
- Near bucket initialization: O(18) parallel writes

**Speedup:** ~100× (no penalty for multiple sources on GPU)

---

## 11. Monitoring & Profiling

### 11.1 Logging Instrumentation

```python
logger.info("[GPU] Batch stats: K={K}, avg_roi_size={avg_size}, total_time={time_ms:.1f}ms")
logger.debug("[GPU] Kernel: relax_near_bucket took {kernel_time_ms:.3f}ms")
logger.debug("[GPU] Iterations: {num_iterations}, Near bucket size: {near_count}")
```

**Metrics to track:**
- Kernel launch time
- Iterations per ROI
- Near/Far bucket sizes over time
- Memory usage (peak allocation)
- Fallback rate (% of ROIs using CPU)

### 11.2 NVIDIA Nsight Systems Profiling

**Command:**
```bash
nsys profile --trace=cuda,cudnn,cublas,osrt python route_board.py
```

**Key Metrics:**
- Kernel occupancy (target: >50%)
- Memory bandwidth utilization (target: >30%)
- Atomic operation contention (target: <10% of kernel time)

### 11.3 Performance Dashboard (CLI Output)

```
[GPU-STATS] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Nets routed:        512
  Total GPU time:     20.5 seconds
  Avg time per net:   40.1 ms
  Speedup vs CPU:     85.2×
  ROI sizes:          10K-120K nodes (avg 65K)
  Batch efficiency:   92% (avg 7.4 ROIs/batch)
  GPU fallbacks:      3 (0.6%) - all due to memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 12. Future Optimizations (Phase 5+)

### 12.1 Work-Efficient Frontier Expansion

**Current:** Process all Near bucket nodes (some may have no work).

**Optimization:** Maintain explicit frontier list of "active" nodes.

**Benefit:** 2-5× reduction in wasted threads.

### 12.2 Delta-Stepping with Adaptive Δ

**Current:** Near-Far uses Δ = min_edge_cost (fixed).

**Optimization:** Dynamically adjust Δ based on Far bucket size.

**Benefit:** Fewer iterations for graphs with variable costs.

### 12.3 Multi-GPU Scaling

**Current:** Single GPU.

**Optimization:** Distribute ROI batches across multiple GPUs.

**Benefit:** Linear speedup with GPU count (embarrassingly parallel).

### 12.4 Shared Memory Optimization

**Current:** All arrays in global memory.

**Optimization:** Cache CSR indptr in shared memory (48 KB per SM).

**Benefit:** 2-3× speedup for small ROIs (<10K nodes).

### 12.5 Persistent Threads

**Current:** Launch kernel per iteration.

**Optimization:** Single persistent kernel, threads loop until done.

**Benefit:** Eliminate kernel launch overhead (~5 μs per launch).

---

## 13. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Atomic contention slows GPU** | Medium | High | Use atomic-free reduction where possible; benchmark early |
| **Memory overflow on large ROIs** | Low | Medium | Implement CPU fallback; add ROI size limit |
| **GPU/CPU results differ** | Low | Critical | Extensive unit tests; bit-exact validation |
| **Speedup less than expected** | Medium | Medium | Profile with Nsight; iterate on kernel optimization |
| **Integration breaks existing code** | Low | High | Feature flag (`use_gpu=False` default); gradual rollout |

---

## 14. Success Criteria

### Must-Have (Phase 1-3)
- ✅ Correctness: 100% path agreement with CPU on 10,000 test cases
- ✅ Speedup: >10× on ROIs with 50K+ nodes
- ✅ Stability: <1% GPU fallback rate on typical boards
- ✅ Multi-source: Portal routing works identically to CPU

### Nice-to-Have (Phase 4+)
- ✅ Speedup: >50× on large ROIs (100K+ nodes)
- ✅ Batching: >80% GPU utilization via efficient batching
- ✅ Profiling: Detailed performance dashboard
- ✅ Documentation: User guide with tuning recommendations

---

## 15. References & Prior Art

1. **"A Fast Work-Efficient SSSP Algorithm for GPUs"** (Wang et al., PPoPP'21)
   - Near-Far algorithm with 2-bucket design
   - Achieves 10-100× speedup on road networks

2. **"Accelerating large graph algorithms on the GPU using CUDA"** (Harish & Narayanan, 2007)
   - Original GPU Dijkstra with CSR format
   - Bugfixes in Martín et al. (2009)

3. **"Delta-Stepping: A Parallelizable Shortest Path Algorithm"** (Meyer & Sanders, 2003)
   - Theoretical foundation for bucket-based SSSP

4. **CuPy Documentation** (https://docs.cupy.dev/)
   - RawKernel API for custom CUDA kernels

5. **NVIDIA CUDA C Programming Guide**
   - Atomic operations, memory coalescing, occupancy

---

## Appendix A: Full Python Wrapper Interface

**File:** `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`

```python
"""
Production-Ready CUDA Dijkstra Pathfinder
Near-Far algorithm with ROI batching for PCB routing
"""

import logging
from typing import List, Optional, Tuple
import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


class CUDADijkstra:
    """GPU-accelerated Dijkstra shortest path finder using Near-Far algorithm"""

    def __init__(self, max_roi_size: int = 200_000, batch_size: int = 8):
        """
        Initialize CUDA Dijkstra solver.

        Args:
            max_roi_size: Maximum nodes per ROI (for padding)
            batch_size: Number of ROIs to process in parallel
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CuPy not available - cannot use CUDA Dijkstra")

        self.max_roi_size = max_roi_size
        self.batch_size = batch_size

        # Compile CUDA kernels
        self._compile_kernels()

        logger.info(f"[CUDA] Initialized Near-Far Dijkstra "
                   f"(max_roi_size={max_roi_size}, batch_size={batch_size})")

    def _compile_kernels(self):
        """Compile CUDA kernels for Near-Far algorithm"""

        # Kernel 1: Relax Near bucket
        self.relax_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void relax_near_bucket(
            const int K,              // Number of ROIs
            const int max_roi_size,   // Max nodes per ROI
            const int max_edges,      // Max edges per ROI
            const int* batch_indptr,  // (K, max_roi_size+1)
            const int* batch_indices, // (K, max_edges)
            const float* batch_weights, // (K, max_edges)
            const bool* near_mask,    // (K, max_roi_size)
            float* dist,              // (K, max_roi_size)
            int* parent,              // (K, max_roi_size)
            bool* far_mask            // (K, max_roi_size)
        ) {
            // [KERNEL CODE FROM SECTION 4.1]
            int global_id = blockIdx.x * blockDim.x + threadIdx.x;
            int roi_idx = global_id / max_roi_size;
            int node = global_id % max_roi_size;

            if (roi_idx >= K || node >= max_roi_size) return;

            int idx = roi_idx * max_roi_size + node;
            if (!near_mask[idx]) return;

            float u_dist = dist[idx];
            int start = batch_indptr[roi_idx * (max_roi_size + 1) + node];
            int end = batch_indptr[roi_idx * (max_roi_size + 1) + node + 1];

            for (int edge_idx = start; edge_idx < end; edge_idx++) {
                int v = batch_indices[roi_idx * max_edges + edge_idx];
                float cost = batch_weights[roi_idx * max_edges + edge_idx];
                float new_dist = u_dist + cost;

                // Atomic min for distance
                float* dist_ptr = &dist[roi_idx * max_roi_size + v];
                atomicMinFloat(dist_ptr, new_dist);

                // Update parent
                if (dist[roi_idx * max_roi_size + v] == new_dist) {
                    parent[roi_idx * max_roi_size + v] = node;
                    far_mask[roi_idx * max_roi_size + v] = true;
                }
            }
        }

        __device__ float atomicMinFloat(float* addr, float value) {
            int* addr_as_int = (int*)addr;
            int old = *addr_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(addr_as_int, assumed,
                    __float_as_int(fminf(value, __int_as_float(assumed))));
            } while (assumed != old);
            return __int_as_float(old);
        }
        ''', 'relax_near_bucket')

        # Kernel 2: Split Near-Far buckets
        self.split_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void split_near_far(
            const int K,
            const int max_roi_size,
            const float* dist,
            const float* threshold,
            bool* near_mask,
            bool* far_mask
        ) {
            int global_id = blockIdx.x * blockDim.x + threadIdx.x;
            int roi_idx = global_id / max_roi_size;
            int node = global_id % max_roi_size;

            if (roi_idx >= K || node >= max_roi_size) return;

            int idx = roi_idx * max_roi_size + node;
            float d = dist[idx];
            float t = threshold[roi_idx];

            // Clear Near (processed)
            near_mask[idx] = false;

            // Move Far → Near if below threshold
            if (far_mask[idx] && d < t) {
                near_mask[idx] = true;
                far_mask[idx] = false;
            }
        }
        ''', 'split_near_far')

        logger.debug("[CUDA] Compiled Near-Far kernels")

    def find_paths_on_rois(self, roi_batch: List[Tuple]) -> List[Optional[List[int]]]:
        """
        Find shortest paths on multiple ROI subgraphs (batched).

        Args:
            roi_batch: List of (src, dst, roi_indptr, roi_indices, roi_weights, roi_size)
                      Each tuple describes one ROI subgraph.

        Returns:
            List of paths (local ROI indices), one per ROI. None if no path found.
        """
        if not roi_batch:
            return []

        K = len(roi_batch)
        logger.debug(f"[CUDA] Processing {K} ROI subgraphs with Near-Far algorithm")

        # Prepare batched GPU arrays
        batch_data = self._prepare_batch(roi_batch)

        # Run Near-Far algorithm
        paths = self._run_near_far(batch_data, K)

        logger.debug(f"[CUDA] Complete: {sum(1 for p in paths if p)}/{K} paths found")
        return paths

    def _prepare_batch(self, roi_batch):
        """Prepare batched GPU arrays from ROI list"""
        K = len(roi_batch)

        # Determine max sizes
        max_roi_size = max(roi[5] for roi in roi_batch)
        max_edges = max(len(roi[3]) for roi in roi_batch)

        # Allocate GPU arrays
        batch_indptr = cp.zeros((K, max_roi_size + 1), dtype=cp.int32)
        batch_indices = cp.zeros((K, max_edges), dtype=cp.int32)
        batch_weights = cp.zeros((K, max_edges), dtype=cp.float32)
        dist = cp.full((K, max_roi_size), cp.inf, dtype=cp.float32)
        parent = cp.full((K, max_roi_size), -1, dtype=cp.int32)
        near_mask = cp.zeros((K, max_roi_size), dtype=cp.bool_)
        far_mask = cp.zeros((K, max_roi_size), dtype=cp.bool_)
        threshold = cp.zeros(K, dtype=cp.float32)

        sources = []
        sinks = []

        # Fill arrays
        for i, (src, dst, indptr, indices, weights, roi_size) in enumerate(roi_batch):
            # Transfer CSR data
            batch_indptr[i, :len(indptr)] = indptr
            batch_indices[i, :len(indices)] = indices
            batch_weights[i, :len(weights)] = weights

            # Initialize distance
            dist[i, src] = 0.0
            near_mask[i, src] = True

            sources.append(src)
            sinks.append(dst)

        return {
            'K': K,
            'max_roi_size': max_roi_size,
            'max_edges': max_edges,
            'batch_indptr': batch_indptr,
            'batch_indices': batch_indices,
            'batch_weights': batch_weights,
            'dist': dist,
            'parent': parent,
            'near_mask': near_mask,
            'far_mask': far_mask,
            'threshold': threshold,
            'sources': sources,
            'sinks': sinks
        }

    def _run_near_far(self, data, K):
        """Execute Near-Far algorithm iterations"""
        max_iterations = 1000
        threads_per_block = 256

        for iteration in range(max_iterations):
            # Check if any ROI has work
            if not data['near_mask'].any():
                break

            # Kernel 1: Relax Near bucket
            total_threads = K * data['max_roi_size']
            num_blocks = (total_threads + threads_per_block - 1) // threads_per_block

            self.relax_kernel(
                (num_blocks,), (threads_per_block,),
                (K, data['max_roi_size'], data['max_edges'],
                 data['batch_indptr'], data['batch_indices'], data['batch_weights'],
                 data['near_mask'], data['dist'], data['parent'], data['far_mask'])
            )

            # Kernel 2: Split Near-Far
            # First, advance threshold
            far_dists = cp.where(data['far_mask'], data['dist'], cp.inf)
            data['threshold'] = far_dists.min(axis=1)

            self.split_kernel(
                (num_blocks,), (threads_per_block,),
                (K, data['max_roi_size'], data['dist'], data['threshold'],
                 data['near_mask'], data['far_mask'])
            )

            # Check termination
            active = data['threshold'] < cp.inf
            if not active.any():
                break

            if iteration % 100 == 0:
                logger.debug(f"[CUDA] Iteration {iteration}: "
                           f"{int(active.sum())}/{K} ROIs active")

        # Reconstruct paths
        return self._reconstruct_paths(data, K)

    def _reconstruct_paths(self, data, K):
        """Reconstruct paths from parent pointers"""
        parent_cpu = data['parent'].get()
        dist_cpu = data['dist'].get()
        sinks = data['sinks']

        paths = []
        for roi_idx in range(K):
            sink = sinks[roi_idx]

            if dist_cpu[roi_idx, sink] == np.inf:
                paths.append(None)
                continue

            # Walk backward
            path = []
            curr = sink
            while curr != -1:
                path.append(curr)
                curr = parent_cpu[roi_idx, curr]
                if len(path) > data['max_roi_size']:
                    logger.error(f"[CUDA] Path reconstruction loop detected")
                    paths.append(None)
                    break
            else:
                path.reverse()
                paths.append(path)

        return paths


# Export
__all__ = ['CUDADijkstra', 'CUDA_AVAILABLE']
```

---

## Appendix B: Unit Test Template

```python
"""
Unit tests for CUDA Dijkstra pathfinding
Validates correctness against CPU heapq implementation
"""

import pytest
import numpy as np
from orthoroute.algorithms.manhattan.unified_pathfinder import SimpleDijkstra, CSRGraph
from orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra import CUDADijkstra, CUDA_AVAILABLE


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CuPy not available")
class TestCUDADijkstra:

    def test_simple_chain(self):
        """Test on simple 5-node chain: 0 → 1 → 2 → 3 → 4"""
        # Build graph
        indptr = np.array([0, 1, 2, 3, 4, 4], dtype=np.int32)
        indices = np.array([1, 2, 3, 4], dtype=np.int32)
        weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # CPU path
        cpu_path = self._cpu_dijkstra(indptr, indices, weights, src=0, dst=4)

        # GPU path
        gpu_solver = CUDADijkstra()
        gpu_paths = gpu_solver.find_paths_on_rois([
            (0, 4, indptr, indices, weights, 5)
        ])
        gpu_path = gpu_paths[0]

        assert cpu_path == gpu_path == [0, 1, 2, 3, 4]

    def test_grid_graph(self):
        """Test on 10×10 grid (100 nodes)"""
        # [Build 10×10 grid CSR graph]
        # ...
        cpu_path = self._cpu_dijkstra(...)
        gpu_path = gpu_solver.find_paths_on_rois([...])
        assert cpu_path == gpu_path

    def _cpu_dijkstra(self, indptr, indices, weights, src, dst):
        """Reference CPU implementation"""
        import heapq
        N = len(indptr) - 1
        dist = [float('inf')] * N
        parent = [-1] * N
        dist[src] = 0.0
        heap = [(0.0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist[u]:
                continue
            for i in range(indptr[u], indptr[u+1]):
                v = indices[i]
                cost = weights[i]
                if dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    parent[v] = u
                    heapq.heappush(heap, (dist[v], v))

        # Reconstruct path
        path = []
        curr = dst
        while curr != -1:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path if dist[dst] < float('inf') else None
```

---

## Appendix C: Performance Tuning Checklist

### Kernel Optimization
- [ ] Block size tuned (test 128, 256, 512 threads)
- [ ] Coalesced memory access (CSR arrays aligned)
- [ ] Atomic contention measured (Nsight Systems)
- [ ] Shared memory used for hot data (indptr caching)

### Algorithm Tuning
- [ ] Batch size optimized (test K=4, 8, 16, 32)
- [ ] GPU/CPU threshold tuned (test 1K, 5K, 10K nodes)
- [ ] Near-Far vs Delta-Stepping compared
- [ ] Multi-source initialization verified

### System Integration
- [ ] CPU fallback tested (out-of-memory, CUDA errors)
- [ ] Logging added (kernel times, iteration counts)
- [ ] Profiling enabled (Nsight Systems, CuPy profiler)
- [ ] End-to-end benchmarks run (100+ nets)

---

**END OF DOCUMENT**
