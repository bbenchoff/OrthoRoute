# CUDA Near-Far Integration Checklist

**Purpose:** Step-by-step checklist for integrating GPU Near-Far algorithm into SimpleDijkstra
**Status:** GPU implementation complete, integration pending
**Time Estimate:** 2-3 hours

---

## Prerequisites

- [ ] CuPy installed (`pip install cupy-cuda11x` or `cupy-cuda12x`)
- [ ] CUDA toolkit installed (11.0+ or 12.0+)
- [ ] NVIDIA GPU available (4GB+ VRAM recommended)
- [ ] Tests pass: `python test_cuda_near_far.py`

---

## Step 1: Verify GPU Implementation Works

### 1.1 Run Unit Tests

```bash
cd C:\Users\Benchoff\Documents\GitHub\OrthoRoute
python test_cuda_near_far.py
```

**Expected Output:**
```
============================================================
CUDA Near-Far Algorithm Tests
============================================================
Test 1: Simple chain (5 nodes)
  ✓ PASS
Test 2: Small grid (3×3 = 9 nodes)
  ✓ PASS
...
============================================================
ALL TESTS PASSED ✓
============================================================
```

- [ ] All 5 tests pass
- [ ] No CUDA errors
- [ ] Paths match CPU reference

### 1.2 Run Benchmarks (Optional)

```bash
python benchmark_cuda_near_far.py
```

**Expected:** Speedup increases with ROI size (10× at 5K, 100× at 100K)

- [ ] Benchmarks complete without errors
- [ ] Speedup > 10× on ROIs with 10K+ nodes
- [ ] GPU memory doesn't overflow

---

## Step 2: Add GPU Solver to SimpleDijkstra

### 2.1 Modify `__init__()` Method

**File:** `orthoroute/algorithms/manhattan/unified_pathfinder.py`
**Location:** Line 977 (class SimpleDijkstra)

**Add after existing CPU initialization:**

```python
def __init__(self, graph: CSRGraph, lattice=None, use_gpu=False):
    # Existing CPU initialization (KEEP AS IS)
    self.indptr = graph.indptr.get() if hasattr(graph.indptr, "get") else graph.indptr
    self.indices = graph.indices.get() if hasattr(graph.indices, "get") else graph.indices
    self.N = len(self.indptr) - 1
    self.plane_size = lattice.x_steps * lattice.y_steps if lattice else None

    # GPU solver initialization (NEW - ADD THIS)
    self.use_gpu = use_gpu
    if self.use_gpu:
        try:
            from .pathfinder.cuda_dijkstra import CUDADijkstra, CUDA_AVAILABLE
            if CUDA_AVAILABLE:
                self.gpu_solver = CUDADijkstra()
                logger.info("[GPU] CUDA Dijkstra enabled for ROI pathfinding")
            else:
                logger.warning("[GPU] CuPy not available, GPU disabled")
                self.gpu_solver = None
                self.use_gpu = False
        except Exception as e:
            logger.warning(f"[GPU] Failed to initialize CUDA Dijkstra: {e}")
            self.gpu_solver = None
            self.use_gpu = False
    else:
        self.gpu_solver = None
```

**Checklist:**
- [ ] Code added after line 983
- [ ] Indentation correct (8 spaces for method body)
- [ ] No syntax errors
- [ ] Imports work (`from .pathfinder.cuda_dijkstra import CUDADijkstra`)

---

## Step 3: Add GPU/CPU Decision Logic

### 3.1 Modify `find_path_roi()` Method

**File:** `orthoroute/algorithms/manhattan/unified_pathfinder.py`
**Location:** Line 985 (method `find_path_roi`)

**Replace entire method with:**

```python
def find_path_roi(self, src: int, dst: int, costs, roi_nodes, global_to_roi) -> Optional[List[int]]:
    """Find shortest path within ROI subgraph (GPU-accelerated when possible)"""

    # Decision heuristic: GPU vs CPU based on ROI size
    roi_size = len(roi_nodes)
    GPU_THRESHOLD = 5000  # Use GPU for ROIs >5K nodes (tunable)

    use_gpu_for_this = self.use_gpu and self.gpu_solver and roi_size > GPU_THRESHOLD

    if use_gpu_for_this:
        try:
            return self._find_path_roi_gpu(src, dst, costs, roi_nodes, global_to_roi)
        except Exception as e:
            logger.warning(f"[GPU] CUDA Dijkstra failed: {e}, falling back to CPU")
            # Fall through to CPU

    # CPU fallback (call refactored method)
    return self._find_path_roi_cpu(src, dst, costs, roi_nodes, global_to_roi)
```

**Checklist:**
- [ ] Method signature unchanged
- [ ] GPU threshold set (5000 is recommended starting point)
- [ ] Exception handling in place
- [ ] Falls back to CPU on error

---

## Step 4: Refactor Existing CPU Code

### 4.1 Create `_find_path_roi_cpu()` Method

**File:** `orthoroute/algorithms/manhattan/unified_pathfinder.py`
**Location:** After `find_path_roi()` (around line 1020)

**Copy existing code from old `find_path_roi()` (lines 987-1051) into new method:**

```python
def _find_path_roi_cpu(self, src: int, dst: int, costs, roi_nodes, global_to_roi) -> Optional[List[int]]:
    """Original CPU heap-based Dijkstra (O(E log V))"""
    import numpy as np
    import heapq

    # [PASTE ENTIRE OLD find_path_roi BODY HERE - lines 987-1051]
    # This is just moving existing code to a separate method
    # NO LOGIC CHANGES!

    costs = costs.get() if hasattr(costs, "get") else costs
    roi_nodes = roi_nodes.get() if hasattr(roi_nodes, "get") else roi_nodes
    global_to_roi = global_to_roi.get() if hasattr(global_to_roi, "get") else global_to_roi

    roi_src = int(global_to_roi[src])
    roi_dst = int(global_to_roi[dst])

    if roi_src < 0 or roi_dst < 0:
        logger.warning("src or dst not in ROI")
        return None

    roi_size = len(roi_nodes)
    dist = np.full(roi_size, np.inf, dtype=np.float32)
    parent = np.full(roi_size, -1, dtype=np.int32)
    visited = np.zeros(roi_size, dtype=bool)
    dist[roi_src] = 0.0

    heap = [(0.0, roi_src)]

    while heap:
        du, u_roi = heapq.heappop(heap)

        if visited[u_roi]:
            continue

        visited[u_roi] = True

        if u_roi == roi_dst:
            break

        u_global = int(roi_nodes[u_roi])

        s, e = int(self.indptr[u_global]), int(self.indptr[u_global + 1])
        for ei in range(s, e):
            v_global = int(self.indices[ei])
            v_roi = int(global_to_roi[v_global])

            if v_roi < 0 or visited[v_roi]:
                continue

            alt = du + float(costs[ei])
            if alt < dist[v_roi]:
                dist[v_roi] = alt
                parent[v_roi] = u_roi
                heapq.heappush(heap, (alt, v_roi))

    if not np.isfinite(dist[roi_dst]):
        return None

    # Reconstruct path in global coordinates
    path, cur = [], roi_dst
    while cur != -1:
        path.append(int(roi_nodes[cur]))
        cur = int(parent[cur])
    path.reverse()

    return path if len(path) > 1 else None
```

**Checklist:**
- [ ] Method created after `find_path_roi()`
- [ ] All original code preserved (no changes!)
- [ ] Indentation correct
- [ ] Returns same type as before

---

## Step 5: Add GPU Pathfinding Method

### 5.1 Create `_find_path_roi_gpu()` Method

**File:** `orthoroute/algorithms/manhattan/unified_pathfinder.py`
**Location:** After `_find_path_roi_cpu()` (around line 1080)

```python
def _find_path_roi_gpu(self, src: int, dst: int, costs, roi_nodes, global_to_roi) -> Optional[List[int]]:
    """GPU-accelerated Dijkstra using CUDADijkstra"""
    import numpy as np

    # Convert to CPU if needed
    roi_nodes_cpu = roi_nodes.get() if hasattr(roi_nodes, 'get') else roi_nodes
    global_to_roi_cpu = global_to_roi.get() if hasattr(global_to_roi, 'get') else global_to_roi

    # Map src/dst to ROI space
    roi_src = int(global_to_roi_cpu[src])
    roi_dst = int(global_to_roi_cpu[dst])

    if roi_src < 0 or roi_dst < 0:
        logger.warning("[GPU] src or dst not in ROI, falling back to CPU")
        raise ValueError("src/dst not in ROI")

    # Build ROI CSR subgraph
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

**Checklist:**
- [ ] Method created after `_find_path_roi_cpu()`
- [ ] Calls `_build_roi_csr()` (next step)
- [ ] Converts local → global path
- [ ] Raises ValueError on error (caught by `find_path_roi()`)

---

## Step 6: Add CSR Subgraph Extraction

### 6.1 Create `_build_roi_csr()` Helper Method

**File:** `orthoroute/algorithms/manhattan/unified_pathfinder.py`
**Location:** After `_find_path_roi_gpu()` (around line 1110)

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

**Checklist:**
- [ ] Method created after `_find_path_roi_gpu()`
- [ ] Extracts edges within ROI
- [ ] Builds CSR format (indptr, indices, weights)
- [ ] Transfers to GPU (CuPy arrays)

---

## Step 7: Test Integration

### 7.1 Create Simple Integration Test

**File:** `test_integration_gpu.py` (new file)

```python
"""
Integration test for GPU pathfinding in SimpleDijkstra.
"""

import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    print("CuPy not available")
    CUDA_AVAILABLE = False
    exit(1)

from orthoroute.algorithms.manhattan.unified_pathfinder import SimpleDijkstra, CSRGraph


def test_simpledijkstra_gpu():
    """Test SimpleDijkstra with GPU enabled"""
    print("Testing SimpleDijkstra GPU integration...")

    # Create simple graph: 0 -> 1 -> 2 -> 3 -> 4
    indptr = np.array([0, 1, 2, 3, 4, 4], dtype=np.int32)
    indices = np.array([1, 2, 3, 4], dtype=np.int32)
    costs = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Create CSRGraph
    graph = CSRGraph(indptr=indptr, indices=indices)

    # Initialize with GPU
    dijkstra = SimpleDijkstra(graph, lattice=None, use_gpu=True)

    # Create ROI (all nodes)
    roi_nodes = np.arange(5, dtype=np.int32)
    global_to_roi = np.arange(5, dtype=np.int32)

    # Find path
    path = dijkstra.find_path_roi(0, 4, costs, roi_nodes, global_to_roi)

    print(f"Path: {path}")

    assert path == [0, 1, 2, 3, 4], f"Expected [0,1,2,3,4], got {path}"
    print("✓ PASS")


if __name__ == "__main__":
    test_simpledijkstra_gpu()
```

**Run:**
```bash
python test_integration_gpu.py
```

**Checklist:**
- [ ] Test passes
- [ ] Path correct: [0, 1, 2, 3, 4]
- [ ] No errors or warnings
- [ ] GPU logs visible: "[GPU] CUDA Dijkstra enabled"

---

## Step 8: Test on Real Routing Task

### 8.1 Route Single Net with GPU

**Test:** Route one net from your PCB with GPU enabled

```python
from orthoroute.algorithms.manhattan.unified_pathfinder import PathFinderRouter
from orthoroute.shared.configuration import PathfinderConfig

# Enable GPU
config = PathfinderConfig(
    use_gpu=True,
    gpu_roi_threshold=5000  # Use GPU for ROIs >5K nodes
)

# Initialize router
router = PathFinderRouter(
    board=board,
    config=config
)

# Route nets
results = router.route_multiple_nets(nets)
```

**Checklist:**
- [ ] Net routes successfully
- [ ] Path looks correct (visualize if possible)
- [ ] GPU logs show GPU usage
- [ ] No CUDA errors

### 8.2 Compare CPU vs GPU Results

**Test:** Route same net with CPU and GPU, verify identical

```python
# CPU route
config_cpu = PathfinderConfig(use_gpu=False)
router_cpu = PathFinderRouter(board=board, config=config_cpu)
results_cpu = router_cpu.route_multiple_nets(nets)

# GPU route
config_gpu = PathfinderConfig(use_gpu=True)
router_gpu = PathFinderRouter(board=board, config=config_gpu)
results_gpu = router_gpu.route_multiple_nets(nets)

# Compare
assert results_cpu['success_rate'] == results_gpu['success_rate']
print("✓ CPU and GPU produce same results")
```

**Checklist:**
- [ ] CPU and GPU success rates match
- [ ] Paths are identical (or very similar)
- [ ] No regressions

---

## Step 9: Benchmark Performance

### 9.1 Measure Speedup on Real ROIs

**Test:** Route 100 nets, measure time

```python
import time

config = PathfinderConfig(use_gpu=True)
router = PathFinderRouter(board=board, config=config)

start = time.time()
results = router.route_multiple_nets(nets)
elapsed = time.time() - start

print(f"Routed {len(nets)} nets in {elapsed:.1f}s")
print(f"Average: {elapsed/len(nets)*1000:.1f}ms per net")
```

**Expected:**
- 100K-node ROIs: 30-50 ms per net (GPU) vs 3-4 seconds (CPU)
- Speedup: 60-130×

**Checklist:**
- [ ] GPU faster than CPU
- [ ] Speedup > 10× on large ROIs
- [ ] No performance regressions on small ROIs

---

## Step 10: Production Configuration

### 10.1 Add Configuration Flags

**File:** `orthoroute/shared/configuration/settings.py` (or equivalent)

```python
# GPU pathfinding settings
USE_GPU_PATHFINDING = True         # Enable GPU acceleration
GPU_ROI_THRESHOLD = 5000           # Min ROI size for GPU (nodes)
GPU_BATCH_SIZE = 8                 # Max ROIs per GPU batch
GPU_MAX_ROI_SIZE = 200_000         # Max ROI size (memory limit)
```

**Checklist:**
- [ ] Configuration added
- [ ] Defaults set
- [ ] Documentation updated

### 10.2 Add Environment Variables (Optional)

```python
import os

USE_GPU = os.getenv('ORTHOROUTE_USE_GPU', 'true').lower() == 'true'
GPU_THRESHOLD = int(os.getenv('ORTHOROUTE_GPU_THRESHOLD', '5000'))
```

**Checklist:**
- [ ] Environment variables work
- [ ] Can disable GPU via env var

---

## Step 11: Final Validation

### 11.1 Run Full Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run GPU-specific tests
python test_cuda_near_far.py
python test_integration_gpu.py
```

**Checklist:**
- [ ] All tests pass
- [ ] No regressions
- [ ] GPU tests pass

### 11.2 Route Large Board

**Test:** Route 500+ nets with GPU

```bash
python route_board.py --board large_test_board.json --use-gpu
```

**Checklist:**
- [ ] All nets route successfully
- [ ] Performance improved vs CPU
- [ ] No CUDA errors or crashes

---

## Troubleshooting

### Issue: "CuPy not available"

**Solution:**
```bash
pip install cupy-cuda11x  # For CUDA 11.x
# OR
pip install cupy-cuda12x  # For CUDA 12.x
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size or ROI threshold
```python
config = PathfinderConfig(
    gpu_roi_threshold=10000,  # Increase (use GPU less often)
    gpu_max_roi_size=100000   # Decrease (reject huge ROIs)
)
```

### Issue: GPU slower than CPU

**Possible Causes:**
1. ROI too small (overhead dominates) → Increase `gpu_roi_threshold`
2. PCIe transfer overhead → Check if graph already on GPU
3. Small batch size → Increase to K=8 or K=16

### Issue: GPU/CPU paths differ

**Solution:** This is a correctness bug - report immediately!
```python
# Debug: Compare paths
print(f"CPU: {cpu_path}")
print(f"GPU: {gpu_path}")

# Check distances
print(f"CPU cost: {cpu_cost}")
print(f"GPU cost: {gpu_cost}")
```

---

## Success Criteria

- [ ] All unit tests pass (5/5)
- [ ] Integration test passes
- [ ] GPU routes nets correctly
- [ ] GPU faster than CPU on large ROIs (>10× speedup)
- [ ] No regressions on CPU path
- [ ] No CUDA errors in production
- [ ] Documentation updated

---

## Rollback Plan

If GPU integration causes issues:

1. Set `use_gpu=False` by default
2. CPU fallback still works (no functionality lost)
3. Can disable GPU via config flag
4. No breaking changes to existing code

**Safe to deploy incrementally.**

---

## Documentation Updates

- [ ] Update README with GPU requirements
- [ ] Add GPU section to user guide
- [ ] Document configuration flags
- [ ] Add performance benchmarks
- [ ] Update troubleshooting guide

---

## Timeline

| Task | Time | Status |
|------|------|--------|
| Verify GPU implementation | 30 min | ✓ Complete |
| Add GPU solver to SimpleDijkstra | 30 min | TODO |
| Refactor CPU code | 30 min | TODO |
| Add GPU pathfinding method | 30 min | TODO |
| Add CSR extraction | 30 min | TODO |
| Integration testing | 1 hour | TODO |
| Performance benchmarking | 1 hour | TODO |
| Documentation | 30 min | TODO |
| **Total** | **5-6 hours** | **In Progress** |

---

**Status:** GPU implementation complete (✓), integration pending (TODO)
**Next Step:** Step 2 - Add GPU solver to SimpleDijkstra.__init__()

