# CUDA Dijkstra Integration Plan

**Purpose:** Step-by-step plan to integrate GPU pathfinding into OrthoRoute's SimpleDijkstra class
**Target Files:**
- `orthoroute/algorithms/manhattan/unified_pathfinder.py`
- `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py` (new file)

---

## Integration Overview

### Current Architecture

```
PathFinderRouter
    ├─ route_multiple_nets()
    │   └─ _pathfinder_negotiation()
    │       └─ _route_all()  [routes hotset nets]
    │           └─ _route_single_net()
    │               ├─ ROIExtractor.extract_roi()  [BFS to build ROI subgraph]
    │               └─ SimpleDijkstra.find_path_roi()  [CPU heapq Dijkstra]
    │                   └─ Python heapq (3-4 seconds per net) ← BOTTLENECK
```

### Target Architecture (GPU-Accelerated)

```
PathFinderRouter
    ├─ route_multiple_nets()
    │   └─ _pathfinder_negotiation()
    │       └─ _route_all()  [routes hotset nets]
    │           └─ _route_single_net()
    │               ├─ ROIExtractor.extract_roi()  [BFS - unchanged]
    │               └─ SimpleDijkstra.find_path_roi()  [GPU/CPU decision]
    │                   ├─ if roi_size > 5K → CUDADijkstra.find_paths_on_rois()
    │                   │   └─ GPU Near-Far (30-40 ms) ← NEW!
    │                   └─ else → CPU heapq (fast for small ROIs)
```

---

## Phase 1: Proof of Concept (Days 1-5)

### Goal
Implement single-ROI GPU Dijkstra with correctness validation.

### Tasks

#### Task 1.1: Create CUDA Dijkstra Module (Day 1)
**File:** `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`

**Action:** Create new file with skeleton class.

```python
"""
CUDA GPU Dijkstra Pathfinding
Near-Far algorithm for parallel shortest path computation
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
    """GPU-accelerated Dijkstra using Near-Far worklist algorithm"""

    def __init__(self, max_roi_size: int = 200_000, batch_size: int = 8):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CuPy not available")

        self.max_roi_size = max_roi_size
        self.batch_size = batch_size
        self._compile_kernels()

        logger.info(f"[CUDA] Initialized Near-Far Dijkstra "
                   f"(max_roi_size={max_roi_size}, batch={batch_size})")

    def _compile_kernels(self):
        """Compile CUDA kernels"""
        # [Kernel compilation code from CUDA_KERNELS_PSEUDOCODE.md]
        pass

    def find_paths_on_rois(self, roi_batch: List[Tuple]) -> List[Optional[List[int]]]:
        """
        Find shortest paths on multiple ROI subgraphs (batched).

        Args:
            roi_batch: List of (src, dst, roi_indptr, roi_indices, roi_weights, roi_size)

        Returns:
            List of paths (local ROI indices), one per ROI
        """
        # [Implementation from CUDA_DIJKSTRA_ARCHITECTURE.md Appendix A]
        pass
```

**Checkpoint:** File compiles without errors, imports successfully.

---

#### Task 1.2: Implement `relax_near_bucket` Kernel (Day 1-2)

**Action:** Add CUDA kernel for parallel edge relaxation.

**Code:** (from CUDA_KERNELS_PSEUDOCODE.md, Kernel 1)

```python
def _compile_kernels(self):
    self.relax_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void relax_near_bucket(
        const int K,
        const int max_roi_size,
        const int max_edges,
        const int* batch_indptr,
        const int* batch_indices,
        const float* batch_weights,
        const bool* near_mask,
        float* dist,
        int* parent,
        bool* far_mask
    ) {
        int global_id = blockIdx.x * blockDim.x + threadIdx.x;
        int roi_idx = global_id / max_roi_size;
        int node = global_id % max_roi_size;

        if (roi_idx >= K || node >= max_roi_size) return;

        int idx = roi_idx * max_roi_size + node;
        if (!near_mask[idx]) return;

        // [Full kernel code from pseudocode doc]
    }
    ''', 'relax_near_bucket')
```

**Test:**
```python
# Test on 5-node chain: 0 → 1 → 2 → 3 → 4
indptr = cp.array([0, 1, 2, 3, 4, 4], dtype=cp.int32)
indices = cp.array([1, 2, 3, 4], dtype=cp.int32)
weights = cp.array([1.0, 1.0, 1.0, 1.0], dtype=cp.float32)

solver = CUDADijkstra()
paths = solver.find_paths_on_rois([(0, 4, indptr, indices, weights, 5)])
assert paths[0] == [0, 1, 2, 3, 4]
```

**Checkpoint:** Kernel compiles and runs on toy graph.

---

#### Task 1.3: Implement `split_near_far` Kernel (Day 2)

**Action:** Add kernel for bucket re-assignment.

**Code:** (from CUDA_KERNELS_PSEUDOCODE.md, Kernel 2)

**Test:** Verify Near/Far buckets update correctly after relaxation.

**Checkpoint:** Both kernels run in sequence without errors.

---

#### Task 1.4: Implement Near-Far Main Loop (Day 3)

**Action:** Add Python wrapper for iteration loop.

```python
def _run_near_far(self, data, K):
    """Execute Near-Far algorithm iterations"""
    max_iterations = 1000
    threads_per_block = 256

    for iteration in range(max_iterations):
        # Check termination
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

        # Advance threshold (CuPy reduction)
        far_dists = cp.where(data['far_mask'], data['dist'], cp.inf)
        data['threshold'] = far_dists.min(axis=1)

        # Kernel 2: Split Near-Far
        self.split_kernel(
            (num_blocks,), (threads_per_block,),
            (K, data['max_roi_size'], data['dist'], data['threshold'],
             data['near_mask'], data['far_mask'])
        )

        # Check active ROIs
        active = data['threshold'] < cp.inf
        if not active.any():
            break

    return self._reconstruct_paths(data, K)
```

**Checkpoint:** Full Near-Far loop runs on simple graphs.

---

#### Task 1.5: Implement Path Reconstruction (Day 3)

**Action:** Add CPU-based path reconstruction.

```python
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

        path = []
        curr = sink
        while curr != -1:
            path.append(curr)
            curr = parent_cpu[roi_idx, curr]
        path.reverse()
        paths.append(path)

    return paths
```

**Checkpoint:** End-to-end GPU pathfinding works on toy graphs.

---

#### Task 1.6: Unit Tests for Correctness (Day 4-5)

**File:** `tests/test_cuda_dijkstra.py`

**Test Cases:**
1. Simple chain: `0 → 1 → 2 → 3 → 4`
2. Small grid: 5×5 grid (25 nodes)
3. Medium grid: 10×10 grid (100 nodes)
4. Disconnected graph: No path from src to dst
5. Self-loop: src == dst
6. Multiple costs: Mixed edge weights (0.4, 3.0, 0.45)

**Validation:** Compare GPU path to CPU heapq path (must be identical).

```python
import pytest
from orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra import CUDADijkstra, CUDA_AVAILABLE

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CuPy not available")
class TestCUDADijkstra:

    def test_simple_chain(self):
        # [Test code]
        cpu_path = self._cpu_dijkstra(...)
        gpu_path = gpu_solver.find_paths_on_rois([...])
        assert cpu_path == gpu_path[0]

    def test_grid_10x10(self):
        # [Build 10×10 grid, compare CPU vs GPU]
        pass

    def _cpu_dijkstra(self, indptr, indices, weights, src, dst):
        # [Reference heapq implementation]
        pass
```

**Checkpoint:** All 6 test cases pass with 100% path agreement.

---

## Phase 2: Integration with SimpleDijkstra (Days 6-10)

### Goal
Connect GPU solver to existing SimpleDijkstra class with CPU fallback.

### Tasks

#### Task 2.1: Add GPU Solver to SimpleDijkstra (Day 6)

**File:** `orthoroute/algorithms/manhattan/unified_pathfinder.py`

**Action:** Modify `SimpleDijkstra.__init__()` to instantiate GPU solver.

**Location:** Around line 977 (class SimpleDijkstra)

**Code:**

```python
class SimpleDijkstra:
    """Dijkstra SSSP with ROI support (CPU fallback + GPU acceleration)"""

    def __init__(self, graph: CSRGraph, lattice=None, use_gpu=False):
        # Copy CSR to CPU if on GPU (existing code)
        self.indptr = graph.indptr.get() if hasattr(graph.indptr, "get") else graph.indptr
        self.indices = graph.indices.get() if hasattr(graph.indices, "get") else graph.indices
        self.N = len(self.indptr) - 1
        self.plane_size = lattice.x_steps * lattice.y_steps if lattice else None

        # GPU solver initialization (NEW)
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            try:
                from .pathfinder.cuda_dijkstra import CUDADijkstra, CUDA_AVAILABLE
                if CUDA_AVAILABLE:
                    self.gpu_solver = CUDADijkstra(max_roi_size=200_000, batch_size=8)
                    logger.info("[GPU] CUDA Dijkstra enabled for ROI pathfinding")
                else:
                    self.gpu_solver = None
                    self.use_gpu = False
            except Exception as e:
                logger.warning(f"[GPU] Failed to initialize CUDA Dijkstra: {e}")
                self.gpu_solver = None
                self.use_gpu = False
        else:
            self.gpu_solver = None
```

**Checkpoint:** SimpleDijkstra instantiates with GPU solver when `use_gpu=True`.

---

#### Task 2.2: Add GPU/CPU Decision Heuristic (Day 6)

**Action:** Modify `find_path_roi()` to choose GPU vs CPU based on ROI size.

**Location:** Line 985 (find_path_roi method)

**Code:**

```python
def find_path_roi(self, src: int, dst: int, costs, roi_nodes, global_to_roi) -> Optional[List[int]]:
    """Find shortest path within ROI subgraph (GPU-accelerated when possible)"""

    # Decision heuristic: GPU vs CPU
    roi_size = len(roi_nodes)
    GPU_THRESHOLD = 5000  # Tunable parameter (benchmark to optimize)

    use_gpu_for_this = self.use_gpu and self.gpu_solver and roi_size > GPU_THRESHOLD

    if use_gpu_for_this:
        try:
            return self._find_path_roi_gpu(src, dst, costs, roi_nodes, global_to_roi)
        except Exception as e:
            logger.warning(f"[GPU] CUDA Dijkstra failed: {e}, falling back to CPU")
            # Fall through to CPU

    # CPU fallback (existing implementation)
    return self._find_path_roi_cpu(src, dst, costs, roi_nodes, global_to_roi)
```

**Checkpoint:** Routing switches between GPU/CPU based on ROI size.

---

#### Task 2.3: Implement `_find_path_roi_cpu()` (Refactor Existing) (Day 7)

**Action:** Extract existing CPU code into separate method (no logic changes).

**Location:** Line 985-1051 (existing find_path_roi code)

**Code:**

```python
def _find_path_roi_cpu(self, src: int, dst: int, costs, roi_nodes, global_to_roi) -> Optional[List[int]]:
    """Original CPU heap-based Dijkstra (O(E log V))"""
    import numpy as np
    import heapq

    # [MOVE EXISTING CODE HERE - lines 985-1051]
    # No changes to algorithm, just refactored into separate method

    costs = costs.get() if hasattr(costs, "get") else costs
    roi_nodes = roi_nodes.get() if hasattr(roi_nodes, "get") else roi_nodes
    global_to_roi = global_to_roi.get() if hasattr(global_to_roi, "get") else global_to_roi

    roi_src = int(global_to_roi[src])
    roi_dst = int(global_to_roi[dst])

    if roi_src < 0 or roi_dst < 0:
        logger.warning("src or dst not in ROI")
        return None

    # [Rest of existing CPU Dijkstra code...]
```

**Checkpoint:** CPU pathfinding still works identically (regression test).

---

#### Task 2.4: Implement `_find_path_roi_gpu()` (Day 7-8)

**Action:** Add GPU pathfinding method with CSR subgraph extraction.

**Code:**

```python
def _find_path_roi_gpu(self, src: int, dst: int, costs, roi_nodes, global_to_roi) -> Optional[List[int]]:
    """GPU-accelerated Dijkstra using CUDADijkstra"""
    import numpy as np

    # Map src/dst to ROI space
    roi_nodes_cpu = roi_nodes.get() if hasattr(roi_nodes, 'get') else roi_nodes
    global_to_roi_cpu = global_to_roi.get() if hasattr(global_to_roi, 'get') else global_to_roi

    roi_src = int(global_to_roi_cpu[src])
    roi_dst = int(global_to_roi_cpu[dst])

    if roi_src < 0 or roi_dst < 0:
        logger.warning("[GPU] src or dst not in ROI, falling back to CPU")
        raise ValueError("src/dst not in ROI")

    # Build ROI subgraph CSR
    roi_size = len(roi_nodes_cpu)
    roi_indptr, roi_indices, roi_weights = self._build_roi_csr(
        roi_nodes_cpu, costs, roi_size
    )

    # Call GPU solver (batch size = 1 for now)
    paths = self.gpu_solver.find_paths_on_rois([
        (roi_src, roi_dst, roi_indptr, roi_indices, roi_weights, roi_size)
    ])
    path = paths[0] if paths else None

    if path is None:
        return None

    # Convert local ROI indices back to global indices
    global_path = [int(roi_nodes_cpu[local_idx]) for local_idx in path]
    return global_path
```

**Checkpoint:** GPU pathfinding returns correct global paths.

---

#### Task 2.5: Implement `_build_roi_csr()` Helper (Day 8)

**Action:** Extract CSR subgraph from global graph for ROI nodes.

**Code:**

```python
def _build_roi_csr(self, roi_nodes, costs, roi_size):
    """Build CSR subgraph for ROI nodes (transfer to GPU)"""
    import numpy as np
    import cupy as cp

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
    edges.sort(key=lambda x: (x[0], x[1]))
    roi_indptr = np.zeros(roi_size + 1, dtype=np.int32)
    roi_indices = np.zeros(len(edges), dtype=np.int32)
    roi_weights = np.zeros(len(edges), dtype=np.float32)

    for i, (u, v, cost) in enumerate(edges):
        roi_indptr[u + 1] += 1
        roi_indices[i] = v
        roi_weights[i] = cost

    roi_indptr = np.cumsum(roi_indptr)

    # Transfer to GPU
    return (
        cp.asarray(roi_indptr),
        cp.asarray(roi_indices),
        cp.asarray(roi_weights)
    )
```

**Checkpoint:** CSR subgraphs built correctly, GPU receives valid data.

---

#### Task 2.6: End-to-End Integration Test (Day 9-10)

**File:** `tests/test_integration_cuda.py`

**Test:** Route 10 real nets on test board with GPU enabled.

```python
def test_route_with_gpu():
    # Load test board
    board = Board(...)

    # Initialize router with GPU
    router = PathFinderRouter(
        board=board,
        config=PathfinderConfig(use_gpu=True)
    )

    # Route nets
    results = router.route_multiple_nets(nets)

    # Validate results
    assert results['success_rate'] == 1.0
    assert all(path is not None for path in router.net_paths.values())
```

**Checkpoint:** GPU routing produces valid paths, no regressions.

---

## Phase 3: Multi-Source Support (Days 11-14)

### Goal
Support portal routing with 18 source seeds (multi-source/multi-sink).

### Tasks

#### Task 3.1: Extend CUDADijkstra for Multi-Source (Day 11-12)

**Action:** Modify `_prepare_batch()` to handle multiple source seeds.

**Code:**

```python
def find_paths_multisource(self, roi_batch_multisource: List[Tuple]):
    """
    Find paths with multi-source initialization (portal routing).

    Args:
        roi_batch_multisource: List of (src_seeds, dst_targets, roi_indptr, ...)
            src_seeds: List[(node, initial_cost)]
            dst_targets: List[node]
    """
    # [Implementation from CUDA_KERNELS_PSEUDOCODE.md Multi-Source section]
    pass
```

**Checkpoint:** GPU handles 18 source seeds correctly.

---

#### Task 3.2: Integrate with SimpleDijkstra.find_path_multisource_multisink() (Day 13)

**Action:** Add GPU path to existing multi-source method.

**Location:** Line 1053 (find_path_multisource_multisink)

**Code:**

```python
def find_path_multisource_multisink(self, src_seeds, dst_targets, costs, roi_nodes, global_to_roi):
    """Multi-source/multi-sink Dijkstra for portal routing (GPU-accelerated)"""

    roi_size = len(roi_nodes)
    use_gpu_for_this = self.use_gpu and self.gpu_solver and roi_size > 5000

    if use_gpu_for_this:
        try:
            return self._find_path_multisource_gpu(src_seeds, dst_targets, costs, roi_nodes, global_to_roi)
        except Exception as e:
            logger.warning(f"[GPU] Multi-source failed: {e}, falling back to CPU")

    # CPU fallback (existing code - lines 1053-1150)
    return self._find_path_multisource_cpu(src_seeds, dst_targets, costs, roi_nodes, global_to_roi)
```

**Checkpoint:** Portal routing uses GPU for large ROIs.

---

#### Task 3.3: End-to-End Portal Test (Day 14)

**Test:** Route 50 nets with portal escapes, verify GPU/CPU produce same paths.

```python
def test_portal_routing_gpu():
    router = PathFinderRouter(
        config=PathfinderConfig(
            use_gpu=True,
            portal_enabled=True
        )
    )

    # Route nets with portals
    results = router.route_multiple_nets(nets)

    # Validate portal geometry
    assert len(router._geometry_payload['vias']) > 0
```

**Checkpoint:** Portal routing works on GPU, produces correct via stacks.

---

## Phase 4: ROI Batching (Days 15-18)

### Goal
Route multiple nets in parallel by batching ROI subgraphs.

### Tasks

#### Task 4.1: Implement ROI Queue (Day 15)

**Action:** Accumulate ROI tasks in a queue before launching GPU batch.

**Code:**

```python
class SimpleDijkstra:
    def __init__(self, ...):
        # [Existing init code...]
        self._roi_queue = []  # Queue of pending ROI tasks
        self._batch_size = 8  # Max ROIs per batch

    def find_path_roi(self, src, dst, costs, roi_nodes, global_to_roi):
        """Queue ROI task for batched processing"""

        roi_size = len(roi_nodes)
        if self.use_gpu and roi_size > 5000:
            # Add to queue
            task = (src, dst, costs, roi_nodes, global_to_roi)
            self._roi_queue.append(task)

            # Flush batch if full
            if len(self._roi_queue) >= self._batch_size:
                return self._flush_gpu_batch()

        # CPU path (immediate)
        return self._find_path_roi_cpu(...)
```

**Checkpoint:** ROI tasks queued, batch flushed when full.

---

#### Task 4.2: Implement Batch Flush Logic (Day 16)

**Action:** Process queued ROIs in single GPU batch.

**Code:**

```python
def _flush_gpu_batch(self):
    """Process all queued ROI tasks in single GPU batch"""
    if not self._roi_queue:
        return []

    # Build batch
    roi_batch = []
    for (src, dst, costs, roi_nodes, global_to_roi) in self._roi_queue:
        roi_csr = self._build_roi_csr(roi_nodes, costs, len(roi_nodes))
        roi_src = int(global_to_roi[src])
        roi_dst = int(global_to_roi[dst])
        roi_batch.append((roi_src, roi_dst, *roi_csr, len(roi_nodes)))

    # Call GPU solver (batched)
    paths = self.gpu_solver.find_paths_on_rois(roi_batch)

    # Clear queue
    self._roi_queue.clear()

    return paths
```

**Checkpoint:** Multiple ROIs processed in single GPU call.

---

#### Task 4.3: Benchmark Batching (Day 17-18)

**Test:** Compare single-ROI vs batched performance.

**Expected Results:**
- Single ROI: 0.9 ms (0.8 ms compute + 0.1 ms overhead)
- Batched (K=8): 0.9 ms total → 0.11 ms per ROI (8× improvement)

**Checkpoint:** Batching achieves >5× speedup on overhead.

---

## Phase 5: Production Hardening (Days 19-23)

### Goal
Error handling, logging, profiling, documentation.

### Tasks

#### Task 5.1: GPU Memory Overflow Handling (Day 19)

**Action:** Catch `OutOfMemoryError` and fall back to CPU.

**Code:**

```python
def _find_path_roi_gpu(self, ...):
    try:
        # [GPU code...]
    except cp.cuda.memory.OutOfMemoryError:
        logger.warning("[GPU] Out of memory (ROI too large), falling back to CPU")
        raise ValueError("GPU OOM")  # Caught by find_path_roi()
```

**Test:** Route net with 500K-node ROI, verify CPU fallback.

**Checkpoint:** No crashes on large ROIs.

---

#### Task 5.2: CUDA Error Handling (Day 19)

**Action:** Catch kernel launch failures.

**Code:**

```python
def _run_near_far(self, ...):
    try:
        self.relax_kernel[num_blocks, threads_per_block](...)
        cp.cuda.Stream.null.synchronize()  # Check for errors
    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.error(f"[GPU] CUDA kernel error: {e}")
        raise GPUPathfindingError("Kernel failed") from e
```

**Checkpoint:** Kernel errors logged and caught gracefully.

---

#### Task 5.3: Performance Logging (Day 20)

**Action:** Log GPU utilization, batch stats, kernel times.

**Code:**

```python
# In _run_near_far()
start_time = time.perf_counter()
# [Run kernels...]
elapsed_ms = (time.perf_counter() - start_time) * 1000

logger.info(f"[GPU-PERF] Batch: K={K}, avg_roi_size={avg_size}, "
           f"time={elapsed_ms:.1f}ms, throughput={K/elapsed_ms*1000:.1f} ROIs/sec")
```

**Checkpoint:** Detailed performance metrics in logs.

---

#### Task 5.4: NVIDIA Nsight Profiling (Day 21)

**Action:** Profile with Nsight Systems, identify bottlenecks.

**Command:**
```bash
nsys profile --trace=cuda,cudnn,cublas,osrt python route_board.py
nsys-ui report1.nsys-rep
```

**Analyze:**
- Kernel occupancy (target: >50%)
- Memory bandwidth (target: >30%)
- Atomic contention (target: <10%)

**Checkpoint:** Profiling data collected, optimization targets identified.

---

#### Task 5.5: User Documentation (Day 22)

**File:** `docs/GPU_PATHFINDING.md`

**Contents:**
- How to enable GPU pathfinding (`use_gpu=True`)
- GPU requirements (CuPy, CUDA 11+, 4GB+ VRAM)
- Performance tuning (batch size, GPU threshold)
- Troubleshooting (fallback behavior, logging)

**Checkpoint:** User-facing documentation complete.

---

#### Task 5.6: Developer Documentation (Day 23)

**File:** `docs/CUDA_DIJKSTRA_DEVELOPER_GUIDE.md`

**Contents:**
- Architecture overview
- Kernel descriptions
- Adding new features (e.g., Delta-Stepping)
- Debugging tips (CUDA-GDB, printf debugging)
- Performance optimization checklist

**Checkpoint:** Developer documentation complete.

---

## Configuration & Feature Flags

### PathfinderConfig Changes

**File:** `orthoroute/algorithms/manhattan/unified_pathfinder.py`

**Location:** Line 531 (PathfinderConfig dataclass)

**Add:**

```python
@dataclass
class PathfinderConfig:
    # [Existing fields...]

    # GPU Acceleration
    use_gpu: bool = False                 # Enable GPU pathfinding
    gpu_roi_threshold: int = 5000         # Min ROI size for GPU (nodes)
    gpu_batch_size: int = 8               # Max ROIs per GPU batch
    gpu_max_roi_size: int = 200_000       # Max ROI size (memory limit)
```

### Environment Variables (Optional)

**File:** `orthoroute/shared/configuration/settings.py`

**Add:**

```python
# GPU settings
ORTHOROUTE_USE_GPU = os.getenv('ORTHOROUTE_USE_GPU', 'false').lower() == 'true'
ORTHOROUTE_GPU_BATCH_SIZE = int(os.getenv('ORTHOROUTE_GPU_BATCH_SIZE', '8'))
```

---

## Testing Strategy

### Unit Tests (tests/test_cuda_dijkstra.py)

**Coverage:**
- [x] Simple chain graph (5 nodes)
- [x] Small grid (5×5 = 25 nodes)
- [x] Medium grid (10×10 = 100 nodes)
- [x] Large grid (100×100 = 10K nodes)
- [x] Disconnected graph (no path)
- [x] Self-loop (src == dst)
- [x] Multi-source (18 seeds)
- [x] Multi-sink (18 targets)

### Integration Tests (tests/test_integration_cuda.py)

**Coverage:**
- [x] Route 10 nets with GPU (small board)
- [x] Route 100 nets with GPU (medium board)
- [x] Portal routing with GPU
- [x] GPU/CPU fallback on large ROI
- [x] Batching (process 8 ROIs in parallel)
- [x] Correctness: GPU == CPU paths

### Performance Tests (tests/test_cuda_performance.py)

**Benchmarks:**
- [x] Single ROI: CPU vs GPU (10K, 50K, 100K nodes)
- [x] Batched ROIs: Speedup vs batch size (K=1, 4, 8, 16)
- [x] End-to-end: Route 512 nets (CPU vs GPU)
- [x] Memory usage: Peak GPU allocation

---

## Rollout Plan

### Stage 1: Developer Testing (Week 1)
- Feature flag: `use_gpu=False` (default off)
- Accessible via: `PathfinderConfig(use_gpu=True)`
- Testing: Internal validation on test boards

### Stage 2: Opt-In Beta (Week 2-3)
- Feature flag: `use_gpu=False` (default off)
- Documentation: User guide published
- Testing: External beta testers, collect feedback

### Stage 3: Opt-Out (Week 4)
- Feature flag: `use_gpu=True` (default on if GPU available)
- Fallback: Automatic CPU fallback on errors
- Monitoring: Track GPU success rate, performance

### Stage 4: Full Deployment (Week 5+)
- Feature flag: Removed (GPU always used when available)
- Optimization: Continuous tuning based on metrics
- Support: Address user issues, add features

---

## Risk Mitigation

### Risk 1: GPU/CPU Path Mismatch

**Mitigation:**
- Extensive unit tests (1000+ random graphs)
- Double-check mode (compare GPU/CPU in debug builds)
- Logging: Log any path differences

### Risk 2: GPU Memory Overflow

**Mitigation:**
- Pre-check ROI size before GPU allocation
- Automatic CPU fallback on `OutOfMemoryError`
- Configurable limit: `gpu_max_roi_size=200_000`

### Risk 3: Slowdown on Small ROIs

**Mitigation:**
- GPU threshold: Only use GPU for ROIs >5K nodes
- Benchmark-tuned threshold (measure crossover point)
- CPU remains default for small ROIs

### Risk 4: CUDA Errors

**Mitigation:**
- Try/catch all CUDA operations
- CPU fallback on any CUDA error
- Logging: Detailed error messages for debugging

### Risk 5: Integration Breaks Existing Code

**Mitigation:**
- Feature flag: Default off during development
- Regression tests: Ensure CPU path unchanged
- Gradual rollout: Opt-in → Opt-out → Default

---

## Success Metrics

### Phase 1 (Proof of Concept)
- ✅ Correctness: 100% path agreement on 20+ test cases
- ✅ Speedup: >10× on 50K-node ROI
- ✅ Stability: No crashes on toy graphs

### Phase 2 (Integration)
- ✅ Correctness: 100% path agreement on 100+ real nets
- ✅ Speedup: >50× on 100K-node ROI
- ✅ Fallback: <1% GPU fallback rate

### Phase 3 (Multi-Source)
- ✅ Correctness: Portal routing identical to CPU
- ✅ Performance: No penalty for 18 sources

### Phase 4 (Batching)
- ✅ Throughput: >5× speedup via batching (K=8)
- ✅ GPU Utilization: >80%

### Phase 5 (Production)
- ✅ Reliability: <0.1% GPU error rate in production
- ✅ Documentation: User + developer guides complete
- ✅ Profiling: Performance dashboard operational

---

## Timeline Summary

| Phase | Days | Deliverable |
|-------|------|-------------|
| **Phase 1: Proof of Concept** | 5 | Single-ROI GPU Dijkstra with validation |
| **Phase 2: Integration** | 5 | GPU/CPU decision logic in SimpleDijkstra |
| **Phase 3: Multi-Source** | 4 | Portal routing support |
| **Phase 4: Batching** | 4 | Multi-ROI parallel processing |
| **Phase 5: Hardening** | 5 | Error handling, logging, docs |
| **Total** | **23 days** | **Production-ready GPU pathfinding** |

**Estimated Calendar Time:** 4-5 weeks (with buffer for testing/iteration)

---

## Appendix: File Checklist

### New Files
- [ ] `orthoroute/algorithms/manhattan/pathfinder/__init__.py`
- [ ] `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`
- [ ] `tests/test_cuda_dijkstra.py`
- [ ] `tests/test_integration_cuda.py`
- [ ] `tests/test_cuda_performance.py`
- [ ] `docs/GPU_PATHFINDING.md`
- [ ] `docs/CUDA_DIJKSTRA_DEVELOPER_GUIDE.md`

### Modified Files
- [ ] `orthoroute/algorithms/manhattan/unified_pathfinder.py`
  - SimpleDijkstra.__init__() - Add GPU solver
  - SimpleDijkstra.find_path_roi() - Add GPU/CPU decision
  - SimpleDijkstra.find_path_multisource_multisink() - Add GPU path
  - PathfinderConfig - Add GPU config fields

---

**END OF INTEGRATION PLAN**
