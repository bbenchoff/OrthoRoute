# CUDA Dijkstra Implementation Recommendations

**Purpose:** Specific recommendations for improving the current cuda_dijkstra.py implementation
**Current Status:** CPU fallback implementation (lines 124-208 in cuda_dijkstra.py)
**Target:** Replace with GPU Near-Far algorithm for 75-100× speedup

---

## Current Implementation Analysis

### What Exists (cuda_dijkstra.py)

**Current Approach (Lines 124-208):**
```python
def find_paths_on_rois(self, roi_batch: List[Tuple]):
    # DECISION: For correctness, use CPU heap-based Dijkstra on each ROI
    # This ensures identical results to CPU mode
    # GPU parallelism comes from processing multiple ROIs concurrently (not shown here)

    import heapq
    for roi_idx, (src, sink, indptr, indices, weights, size) in enumerate(roi_batch):
        # Transfer GPU arrays to CPU
        indptr_cpu = indptr.get()
        # Run CPU heapq Dijkstra
        # [lines 149-192: standard heap-based algorithm]
```

**Analysis:**
- ❌ **Not actually using GPU** - runs CPU heapq inside GPU module
- ❌ **No speedup** - transfers arrays to CPU, runs serially
- ❌ **Misleading** - CUDADijkstra class but CPU implementation
- ✅ **Correct** - produces valid paths (matches CPU SimpleDijkstra)

**Status:** This is a **correctness placeholder**, not a GPU implementation.

---

## Immediate Recommendations

### Recommendation 1: Replace with Near-Far Algorithm ⭐⭐⭐

**Priority:** CRITICAL (this is the entire point of CUDA integration)

**Action:** Replace `find_paths_on_rois()` with GPU Near-Far implementation.

**File:** `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`

**Replace:** Lines 124-208 (current CPU heapq code)

**With:** GPU Near-Far implementation from `CUDA_DIJKSTRA_ARCHITECTURE.md` Appendix A (lines 80-320).

**Expected Impact:** 75-100× speedup on 100K-node ROIs

**Effort:** 3-5 days (Phase 1 of integration plan)

**Blocker:** None - can be done immediately

---

### Recommendation 2: Add Near-Far Kernels ⭐⭐⭐

**Priority:** CRITICAL (required for Recommendation 1)

**Action:** Add `relax_near_bucket` and `split_near_far` CUDA kernels.

**Current State:** Kernel at lines 31-74 (`relax_edges_parallel`) is incomplete.

**Problems with Current Kernel:**
```cuda
// Line 31-74: relax_edges_parallel
// Issues:
// 1. Processes one min_node at a time (not parallel Near bucket)
// 2. Requires external loop to feed min_nodes (defeats GPU parallelism)
// 3. No bucketing logic (Near/Far)
// 4. atomicMin uses float-as-int (correct but not documented)
```

**Replace With:** Kernels from `CUDA_KERNELS_PSEUDOCODE.md`:
- **Kernel 1:** `relax_near_bucket` (lines 28-92) - parallel edge relaxation
- **Kernel 2:** `split_near_far` (lines 138-184) - bucket re-assignment
- **Helper:** `atomicMinFloat` (lines 82-92) - atomic min for floats

**Expected Impact:** Enable GPU Near-Far algorithm

**Effort:** 1-2 days (kernel compilation + testing)

---

### Recommendation 3: Remove find_path_batch() ⭐

**Priority:** LOW (unused code)

**Action:** Delete `find_path_batch()` method (lines 210-319).

**Reason:**
- This is a different algorithm (frontier-based Dijkstra with argmin)
- Not compatible with ROI batching (processes nodes globally)
- Inefficient (argmin every iteration = O(V²) complexity)
- Unused (not called by SimpleDijkstra)

**Alternative:** Keep as reference/backup, but comment as "DEPRECATED - DO NOT USE".

**Impact:** Cleaner codebase, less confusion

**Effort:** 5 minutes

---

### Recommendation 4: Fix Atomic Operations ⭐⭐

**Priority:** MEDIUM (correctness issue in current kernel)

**Problem:** Line 66 uses unsafe atomic pattern:
```cuda
// Line 66: Incorrect atomic min pattern
atomicMin((int*)dist_ptr, __float_as_int(new_dist));  // WRONG!
```

**Why Wrong:**
- `atomicMin` for int doesn't preserve float comparison semantics
- Example: `atomicMin(-1, 1)` = -1 (as int), but -1 as float < 1 as float (WRONG!)

**Fix:** Use proper `atomicMinFloat` with CAS loop:
```cuda
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

**Location:** Add as helper function, use in `relax_near_bucket` kernel.

**Impact:** Correct distance updates (prevents path errors)

**Effort:** 30 minutes

---

### Recommendation 5: Add Batching Support ⭐⭐

**Priority:** HIGH (60% overhead reduction)

**Action:** Implement ROI batching (K=8 ROIs in parallel).

**Current State:** `find_paths_on_rois()` processes ROIs serially (for loop).

**Target:** Process all K ROIs in single GPU batch (parallel).

**Changes:**
1. Pre-allocate batched arrays: `batch_indptr[K, max_roi_size+1]`
2. Pad smaller ROIs to `max_roi_size` (uniform array dimensions)
3. Launch kernels once for all K ROIs (not K times)

**Implementation:** See `CUDA_DIJKSTRA_ARCHITECTURE.md` Section 2.3 (ROI Batching).

**Impact:** 1.58× speedup via overhead reduction (287 μs → 182 μs per ROI)

**Effort:** 2-3 days (Phase 4 of integration plan)

---

### Recommendation 6: Add Multi-Source Initialization ⭐⭐

**Priority:** HIGH (portal routing requirement)

**Action:** Support 18 source seeds for portal routing.

**Current State:** `find_paths_on_rois()` expects single (src, sink) per ROI.

**Target:** Accept `List[(node, initial_cost)]` for multi-source seeds.

**Changes:**
1. Modify function signature: `(src_seeds, dst_targets, ...)`
2. Initialize Near bucket with all source seeds: `dist[src] = initial_cost`
3. Terminate when ANY destination reached: `if any(dist[dst] < inf): break`

**Implementation:** See `CUDA_KERNELS_PSEUDOCODE.md` Multi-Source section.

**Impact:** Portal routing works on GPU (same as CPU)

**Effort:** 1-2 days (Phase 3 of integration plan)

---

## Detailed Implementation Checklist

### Phase 1: Near-Far Kernel Implementation (Days 1-5)

#### Day 1: Setup & Kernel Skeleton
- [ ] Read CUDA_KERNELS_PSEUDOCODE.md (Kernel 1 & 2)
- [ ] Copy `atomicMinFloat` helper (lines 82-92)
- [ ] Copy `relax_near_bucket` kernel skeleton (lines 28-81)
- [ ] Copy `split_near_far` kernel skeleton (lines 138-184)
- [ ] Test compilation (ensure no syntax errors)

**Checkpoint:** Kernels compile without errors

#### Day 2: Data Preparation
- [ ] Implement `_prepare_batch()` method:
  - Allocate batched GPU arrays: `batch_indptr`, `batch_indices`, `batch_weights`
  - Allocate distance/parent arrays: `dist`, `parent`
  - Allocate bucket masks: `near_mask`, `far_mask`
  - Initialize sources: `dist[roi_idx, src] = 0.0`, `near_mask[roi_idx, src] = True`
- [ ] Test memory allocation (ensure no OOM errors)

**Checkpoint:** Batched arrays allocated correctly

#### Day 3: Near-Far Main Loop
- [ ] Implement `_run_near_far()` method:
  - Iteration loop (max 1000 iterations)
  - Kernel 1 launch: `relax_near_bucket`
  - Threshold advance: `threshold = far_dists.min(axis=1)`
  - Kernel 2 launch: `split_near_far`
  - Termination check: `if not near_mask.any(): break`
- [ ] Test on 5-node chain (expect 4 iterations)

**Checkpoint:** Near-Far loop runs without crashes

#### Day 4: Path Reconstruction
- [ ] Implement `_reconstruct_paths()` method:
  - Transfer `parent` to CPU: `parent_cpu = parent.get()`
  - Walk backward from sink: `curr = sink; while curr != -1: ...`
  - Reverse path: `path.reverse()`
- [ ] Test on 5-node chain (expect path = [0, 1, 2, 3, 4])

**Checkpoint:** End-to-end GPU pathfinding works on toy graph

#### Day 5: Unit Tests
- [ ] Test simple chain (5 nodes)
- [ ] Test small grid (5×5 = 25 nodes)
- [ ] Test medium grid (10×10 = 100 nodes)
- [ ] Test disconnected graph (no path)
- [ ] Test self-loop (src == dst)
- [ ] Compare GPU vs CPU (100% agreement)

**Checkpoint:** All unit tests pass with correctness

**Deliverable:** Working GPU Near-Far Dijkstra (single-ROI)

---

### Phase 2: Integration with SimpleDijkstra (Days 6-10)

#### Day 6: SimpleDijkstra Changes
- [ ] File: `orthoroute/algorithms/manhattan/unified_pathfinder.py`
- [ ] Line 977: Modify `SimpleDijkstra.__init__()`:
  - Add: `self.use_gpu = use_gpu and GPU_AVAILABLE`
  - Add: `self.gpu_solver = CUDADijkstra() if self.use_gpu else None`
- [ ] Line 985: Modify `find_path_roi()`:
  - Add GPU/CPU decision: `if self.use_gpu and roi_size > 5000: use_gpu`
  - Call: `self._find_path_roi_gpu(...)` or `self._find_path_roi_cpu(...)`
- [ ] Test: SimpleDijkstra instantiates with GPU solver

**Checkpoint:** SimpleDijkstra can call GPU solver

#### Day 7: CPU Path Refactoring
- [ ] Extract existing CPU code (lines 985-1051) into `_find_path_roi_cpu()`
- [ ] No logic changes (just refactoring)
- [ ] Test: CPU pathfinding still works (regression test)

**Checkpoint:** CPU path unchanged, no regressions

#### Day 8: GPU Path Implementation
- [ ] Implement `_find_path_roi_gpu()`:
  - Map src/dst to ROI space: `roi_src = global_to_roi[src]`
  - Build ROI CSR: `roi_indptr, roi_indices, roi_weights = self._build_roi_csr(...)`
  - Call GPU solver: `paths = self.gpu_solver.find_paths_on_rois([...])`
  - Convert to global indices: `global_path = [roi_nodes[i] for i in path]`
- [ ] Implement `_build_roi_csr()`:
  - Extract edges within ROI
  - Build CSR arrays (indptr, indices, weights)
  - Transfer to GPU

**Checkpoint:** GPU pathfinding returns correct global paths

#### Day 9-10: Integration Testing
- [ ] Route 10 real nets on test board (GPU enabled)
- [ ] Compare GPU vs CPU results (100% path agreement)
- [ ] Measure speedup (expect 50-100×)
- [ ] Test CPU fallback (large ROI, GPU OOM)

**Checkpoint:** GPU routing produces valid paths, no regressions

**Deliverable:** GPU-accelerated SimpleDijkstra

---

### Phase 3: Multi-Source Support (Days 11-14)

#### Day 11-12: Multi-Source Kernel
- [ ] Modify `_prepare_batch()` for multi-source:
  - Accept `src_seeds: List[(node, initial_cost)]`
  - Initialize multiple sources: `for (node, cost) in src_seeds: dist[node] = cost`
- [ ] Modify `_run_near_far()` for multi-sink:
  - Terminate when ANY destination reached
  - Return entry/exit layers

**Checkpoint:** GPU handles 18 source seeds correctly

#### Day 13: SimpleDijkstra Multi-Source
- [ ] Modify `find_path_multisource_multisink()` (line 1053):
  - Add GPU path: `if self.use_gpu and roi_size > 5000: use_gpu`
  - Call: `self._find_path_multisource_gpu(...)`
- [ ] Implement `_find_path_multisource_gpu()`:
  - Build ROI CSR (same as single-source)
  - Map source seeds to ROI space
  - Call GPU solver with multi-source

**Checkpoint:** Portal routing uses GPU for large ROIs

#### Day 14: Portal Testing
- [ ] Route 50 nets with portal escapes
- [ ] Compare GPU vs CPU results (100% agreement)
- [ ] Verify via stacks correct (entry/exit layers)
- [ ] Measure performance (expect no penalty for 18 sources)

**Checkpoint:** Portal routing works on GPU

**Deliverable:** Full multi-source support

---

### Phase 4: ROI Batching (Days 15-18)

#### Day 15: Batch Queue
- [ ] Add to `SimpleDijkstra.__init__()`:
  - `self._roi_queue = []`
  - `self._batch_size = 8`
- [ ] Modify `find_path_roi()`:
  - Queue ROI tasks: `self._roi_queue.append(task)`
  - Flush batch: `if len(self._roi_queue) >= 8: self._flush_gpu_batch()`

**Checkpoint:** ROI tasks queued, batch flushed when full

#### Day 16: Batch Processing
- [ ] Implement `_flush_gpu_batch()`:
  - Build batch: `roi_batch = [...]`
  - Call GPU solver: `paths = self.gpu_solver.find_paths_on_rois(roi_batch)`
  - Clear queue
- [ ] Modify `find_paths_on_rois()` to handle K ROIs in parallel
  - Pad smaller ROIs to max_roi_size
  - Launch kernels once for all K ROIs

**Checkpoint:** Multiple ROIs processed in single GPU call

#### Day 17-18: Batching Benchmarks
- [ ] Benchmark single-ROI vs batched (K=1, 4, 8, 16)
- [ ] Measure overhead reduction (expect 1.58× for K=8)
- [ ] Tune batch size (optimal K=8)

**Checkpoint:** Batching achieves >1.5× speedup on overhead

**Deliverable:** Batched GPU pathfinding

---

### Phase 5: Production Hardening (Days 19-23)

#### Day 19: Error Handling
- [ ] GPU memory overflow: Catch `cp.cuda.memory.OutOfMemoryError`
- [ ] CUDA errors: Catch `cp.cuda.runtime.CUDARuntimeError`
- [ ] CPU fallback: Log warnings, retry with CPU

**Checkpoint:** No crashes on large ROIs or GPU errors

#### Day 20: Logging
- [ ] Add performance logs:
  - GPU batch stats (K, avg_roi_size, time)
  - Kernel times (relax, split, threshold)
  - Speedup vs CPU
- [ ] Add debug logs:
  - Iteration count
  - Near/Far bucket sizes
  - Memory usage

**Checkpoint:** Detailed performance metrics in logs

#### Day 21: Profiling
- [ ] Profile with Nsight Systems:
  - `nsys profile --trace=cuda python route_board.py`
- [ ] Analyze occupancy, bandwidth, atomics
- [ ] Identify bottlenecks

**Checkpoint:** Profiling data collected

#### Day 22: User Documentation
- [ ] Write `docs/GPU_PATHFINDING.md`:
  - How to enable GPU (`use_gpu=True`)
  - GPU requirements (CuPy, CUDA 11+)
  - Performance tuning (batch size, GPU threshold)
  - Troubleshooting (fallback, logging)

**Checkpoint:** User-facing docs complete

#### Day 23: Developer Documentation
- [ ] Write `docs/CUDA_DIJKSTRA_DEVELOPER_GUIDE.md`:
  - Architecture overview
  - Kernel descriptions
  - Adding features (Delta-Stepping)
  - Debugging tips (CUDA-GDB)

**Checkpoint:** Developer docs complete

**Deliverable:** Production-ready GPU pathfinding

---

## Code Quality Checklist

### Kernel Code
- [ ] All CUDA kernels compile without warnings
- [ ] Atomic operations use proper CAS loop (`atomicMinFloat`)
- [ ] Bounds checks prevent illegal memory access
- [ ] Thread indexing correct: `roi_idx = global_id / max_roi_size`
- [ ] Memory layout documented (row-major, coalesced access)

### Python Wrapper
- [ ] Type hints for all function arguments
- [ ] Docstrings for all public methods
- [ ] Error handling with try/except (CPU fallback)
- [ ] Logging at INFO (user-facing) and DEBUG (developer) levels
- [ ] No hardcoded magic numbers (use constants)

### Testing
- [ ] Unit tests for all kernels (toy graphs)
- [ ] Integration tests for SimpleDijkstra (real nets)
- [ ] Performance tests (CPU vs GPU benchmarks)
- [ ] Correctness validation (10,000+ random graphs)
- [ ] Edge cases (disconnected, self-loop, large ROI)

### Documentation
- [ ] Architecture docs (this file + companion docs)
- [ ] User guide (how to enable GPU)
- [ ] Developer guide (how to modify kernels)
- [ ] API reference (function signatures)
- [ ] Performance tuning (optimization checklist)

---

## Performance Optimization Checklist

### After Phase 1 (Single-ROI)
- [ ] Measure speedup (expect 50-100×)
- [ ] Profile with Nsight (check occupancy >50%)
- [ ] Verify correctness (100% agreement with CPU)

### After Phase 4 (Batching)
- [ ] Measure batching improvement (expect 1.5-2×)
- [ ] Tune batch size (try K=4, 8, 16)
- [ ] Measure GPU utilization (expect >75%)

### Phase 5+ (Advanced)
- [ ] Shared memory caching (indptr array) → 2× speedup
- [ ] Block size tuning (try 128, 256, 512 threads) → 10-20% gain
- [ ] Persistent threads (eliminate launch overhead) → 20% gain
- [ ] Delta-Stepping (if Near-Far insufficient) → 20-50% gain

---

## Common Pitfalls & Solutions

### Pitfall 1: Atomic Contention Slowdown

**Symptom:** GPU slower than expected (10× instead of 100×)

**Cause:** Many threads updating same node (atomic serialization)

**Solution:**
- Use Near-Far bucketing (limits concurrent updates)
- Profile with Nsight: Check "atomic operation %" (target <10%)
- Consider work-efficient frontier (Phase 5+)

### Pitfall 2: Memory Layout Wrong

**Symptom:** Kernel crashes with illegal memory access

**Cause:** Incorrect array indexing (row-major vs column-major confusion)

**Solution:**
- Verify layout: `batch_indptr[roi_idx * (max_roi_size + 1) + node]`
- Use helper macros: `#define IDX2D(i, j, width) ((i) * (width) + (j))`
- Add bounds checks: `if (roi_idx >= K || node >= max_roi_size) return;`

### Pitfall 3: PCIe Bottleneck

**Symptom:** GPU time dominated by transfers (50% overhead)

**Cause:** Re-transferring CSR graph every call

**Solution:**
- Keep graph on GPU (CuPy arrays, already done)
- Only transfer ROI subgraph (smaller)
- Batch transfers (send K ROIs at once)

### Pitfall 4: Near-Far Iterations Too Many

**Symptom:** 100+ iterations instead of 8

**Cause:** Highly variable edge costs (100:1 ratio)

**Solution:**
- Check cost distribution: `max_cost / min_cost` (target <10)
- Consider Delta-Stepping with smaller Δ
- Normalize costs (if applicable)

### Pitfall 5: Path Reconstruction Incorrect

**Symptom:** GPU path != CPU path (correctness failure)

**Cause:** Parent pointers cycle (infinite loop)

**Solution:**
- Add cycle detection: `if curr in visited: break`
- Verify parent initialization: `parent[roi_idx, :] = -1`
- Test on known graphs (chain, grid)

---

## Success Criteria Validation

### Correctness (Must-Have)
- [ ] 100% path agreement with CPU on 10,000 test cases
- [ ] Portal routing identical to CPU (entry/exit layers)
- [ ] No crashes on typical boards (500+ nets)
- [ ] CPU fallback works (large ROI, GPU error)

### Performance (Must-Have)
- [ ] Speedup >50× on 100K-node ROIs (conservative)
- [ ] Speedup >100× on 100K-node ROIs (realistic)
- [ ] Batching improvement >1.5× (K=8 vs K=1)
- [ ] GPU utilization >75% (Nsight profiling)

### Reliability (Must-Have)
- [ ] GPU fallback rate <1% on typical boards
- [ ] No memory leaks (peak allocation stable)
- [ ] Handles disconnected graphs (no crashes)

### Usability (Nice-to-Have)
- [ ] Detailed performance logs (batch stats, kernel times)
- [ ] User documentation complete (GPU guide)
- [ ] Developer documentation complete (kernel guide)

---

## Final Recommendations Summary

### Top 3 Priorities (Do First)

1. **Replace find_paths_on_rois() with Near-Far** ⭐⭐⭐
   - Impact: 75-100× speedup (the entire point)
   - Effort: 3-5 days
   - Risk: Low (well-understood algorithm)

2. **Add Near-Far CUDA Kernels** ⭐⭐⭐
   - Impact: Enables Near-Far algorithm
   - Effort: 1-2 days
   - Risk: Low (reference implementation exists)

3. **Integrate with SimpleDijkstra** ⭐⭐⭐
   - Impact: Makes GPU accessible to users
   - Effort: 3-5 days
   - Risk: Medium (integration points)

### Next 3 Priorities (Do Soon)

4. **Add Multi-Source Support** ⭐⭐
   - Impact: Portal routing on GPU
   - Effort: 2-3 days
   - Risk: Low (minor kernel change)

5. **Implement ROI Batching** ⭐⭐
   - Impact: 1.58× speedup via overhead reduction
   - Effort: 2-3 days
   - Risk: Low (straightforward padding)

6. **Fix Atomic Operations** ⭐⭐
   - Impact: Correctness (prevents path errors)
   - Effort: 30 minutes
   - Risk: Low (well-known pattern)

### Later Priorities (Phase 5+)

7. **Remove find_path_batch()** ⭐
   - Impact: Code cleanup
   - Effort: 5 minutes
   - Risk: None

8. **Optimize Kernels** (shared memory, persistent threads)
9. **Add Delta-Stepping** (if Near-Far insufficient)
10. **Multi-GPU Scaling** (future work)

---

## Next Steps

**For Immediate Action:**
1. Read `CUDA_INTEGRATION_PLAN.md` Phase 1 (Days 1-5)
2. Start with Recommendation 1: Replace find_paths_on_rois()
3. Follow day-by-day checklist in Section "Phase 1: Near-Far Kernel Implementation"

**For Questions:**
- Architecture: See `CUDA_DIJKSTRA_ARCHITECTURE.md`
- Kernels: See `CUDA_KERNELS_PSEUDOCODE.md`
- Performance: See `CUDA_PERFORMANCE_ANALYSIS.md`

---

**STATUS:** Ready to implement (design complete, plan validated)

**BLOCKERS:** None (all dependencies documented)

**RISK:** Low (conservative estimates, robust fallback)

---

**END OF RECOMMENDATIONS**
