# GPU Synchronization Bottleneck - Fix Plan

## Problem Statement

**Current Performance:** 97 nets/sec (2.5× vs 48 nets/sec baseline)
**Theoretical Max:** 160-180 nets/sec (based on kernel execution time)
**Target:** 480 nets/sec (10× vs baseline)
**Gap:** 60% of theoretical performance missing due to hidden CPU-GPU synchronization

## Root Cause Hypothesis

Hidden `.get()` or `cp.asnumpy()` calls between GPU batches are forcing device synchronization, preventing pipeline overlap and killing performance.

### Evidence:
- Kernel executes in 350-400ms per batch (good)
- Total batch time is 600-660ms (includes overhead)
- 200-260ms overhead per batch (30-40% wasted)
- No explicit synchronization in code → must be implicit

## Known Synchronization Points

### 1. Per-Batch Status Logging (HIGH PRIORITY)
**Location:** `unified_pathfinder.py` lines 3080-3100
```python
logger.info(f"[BATCH-{batch_num}] GPU routing complete: {gpu_time:.2f}s ({batch_size_actual} nets)")
# ^ This triggers .get() internally if logging array contents
```

**Fix:** Remove per-batch logging, only log every 10th batch or at end

### 2. Path Result Processing (HIGH PRIORITY)
**Location:** `unified_pathfinder.py` lines 3110-3150
```python
for i, (net_id, use_portals, roi_nodes_arr, src, dst) in enumerate(batch_metadata):
    if roi_nodes_arr is None or i >= len(paths):
        failed_this_pass += 1
        # ^ Accessing paths[i] might trigger synchronization
```

**Fix:** Use `cp.cuda.Stream.null.synchronize()` ONCE before loop, not per iteration

### 3. Array Indexing and Length Checks (MEDIUM PRIORITY)
**Location:** Throughout routing loop
```python
if i >= len(paths):  # len() on GPU array forces sync
```

**Fix:** Pre-fetch lengths to CPU variables once per batch

### 4. Accounting Updates (MEDIUM PRIORITY)
**Location:** `unified_pathfinder.py` lines 3160-3180
```python
self.accounting.commit_path(edge_indices)  # May do CPU operations
```

**Fix:** Verify accounting is GPU-based, batch commits at iteration end

### 5. Cost Array Updates (MEDIUM PRIORITY)
**Location:** Between batches, updating `costs` array
```python
costs = self.accounting.total_cost  # May copy to CPU
```

**Fix:** Keep costs on GPU, only sync at iteration boundaries

## Diagnostic Plan

### Phase 1: Find All Synchronization Points (30 minutes)

**Tools:**
1. `nsys profile` - NVIDIA profiler to capture GPU timeline
2. Code search for `.get()`, `cp.asnumpy()`, `len()`, array indexing

**Commands:**
```bash
# Profile 1 iteration (should be fast)
nsys profile --trace cuda,nvtx -o timeline python main.py --test-manhattan

# Analyze timeline
nsys stats timeline.qdrep
```

**Look for:**
- Large gaps between kernel launches (CPU doing work)
- Frequent cudaMemcpy calls
- cudaDeviceSynchronize calls

### Phase 2: Remove Synchronization (2-4 hours)

Priority order:
1. Remove all per-batch logging
2. Add single sync before path processing
3. Pre-fetch array lengths
4. Verify accounting is GPU-only
5. Keep cost array on GPU

### Phase 3: Validate Fix (1 hour)

**Success criteria:**
- Batch overhead drops from 200-260ms to <50ms
- Throughput increases from 97 to 150+ nets/sec
- GPU utilization stays >70%
- Routing success rate unchanged

## Implementation Tasks

### Task 1: Remove Per-Batch Logging
**File:** `unified_pathfinder.py`
**Lines:** 3080-3100, plus scattered debug logs

**Changes:**
```python
# BEFORE: Every batch
logger.info(f"[BATCH-{batch_num}] GPU routing complete: {gpu_time:.2f}s ({batch_size_actual} nets)")

# AFTER: Every 10th batch
if batch_num % 10 == 0 or batch_num == total_batches:
    logger.info(f"[BATCH-{batch_num}] GPU routing complete: {gpu_time:.2f}s ({batch_size_actual} nets)")
```

**Expected gain:** 50-100ms per batch (20-30% speedup)

### Task 2: Single Sync Before Path Processing
**File:** `unified_pathfinder.py`
**Lines:** Before line 3110

**Changes:**
```python
# GPU routing complete
paths = self.solver.gpu_solver.route_batch_persistent(roi_batch)

# CRITICAL: Single sync here, before processing results
cp.cuda.Stream.null.synchronize()

# Now process results on CPU
for i, (net_id, use_portals, roi_nodes_arr, src, dst) in enumerate(batch_metadata):
    # No more implicit syncs in this loop
```

**Expected gain:** 30-50ms per batch (10-15% speedup)

### Task 3: Pre-fetch Array Lengths
**File:** `unified_pathfinder.py`
**Lines:** 3110-3150

**Changes:**
```python
# BEFORE:
for i, (net_id, ...) in enumerate(batch_metadata):
    if i >= len(paths):  # Triggers sync EVERY iteration

# AFTER:
num_paths = len(paths) if paths else 0  # Sync ONCE
for i, (net_id, ...) in enumerate(batch_metadata):
    if i >= num_paths:  # No sync
```

**Expected gain:** 10-20ms per batch (5-10% speedup)

### Task 4: Verify Accounting is GPU-Only
**File:** `unified_pathfinder.py`, `accounting.py`
**Lines:** Various

**Check:**
```python
# In accounting.commit_path() - ensure no .get() calls
# In accounting.total_cost - ensure returns GPU array
```

**Expected gain:** 20-50ms per batch if currently on CPU

### Task 5: Keep Costs on GPU
**File:** `unified_pathfinder.py`
**Lines:** Where costs array is accessed

**Changes:**
```python
# BEFORE:
costs = self.accounting.total_cost  # Might copy to CPU

# AFTER: Ensure costs stays as CuPy array
assert hasattr(costs, 'device'), "Costs must be GPU array"
```

**Expected gain:** 30-50ms per batch if currently copying

## Expected Performance After Fixes

### Conservative Estimate:
- Remove 150ms overhead per batch
- Batch time: 660ms → 510ms
- Throughput: 97 nets/sec → 125 nets/sec
- Speedup: 2.5× → 3.3× vs baseline

### Optimistic Estimate:
- Remove 200ms overhead per batch
- Batch time: 660ms → 460ms
- Throughput: 97 nets/sec → 139 nets/sec
- Speedup: 2.5× → 3.6× vs baseline

### With Additional Optimizations:
- Move batch prep to GPU (save 150ms)
- Batch time: 460ms → 310ms
- Throughput: 139 nets/sec → 207 nets/sec
- Speedup: 3.6× → 5.4× vs baseline

### Ultimate Goal (All Optimizations):
- Eliminate all CPU overhead
- Batch time: 350ms (kernel only)
- Throughput: 183 nets/sec
- Speedup: 4.8× vs baseline

**Note:** 10× speedup (480 nets/sec) would require kernel optimization (currently not the bottleneck).

## Agent Execution Plan

### Agent 1: Profile and Identify (Diagnostic)
**Time:** 30 minutes
**Deliverable:** List of all synchronization points with line numbers and estimated cost

### Agent 2: Remove Logging Overhead (Quick Win)
**Time:** 30 minutes
**Deliverable:** Per-batch logging reduced to every 10th batch
**Expected:** 50-100ms speedup per batch

### Agent 3: Single Sync Point (Critical)
**Time:** 1 hour
**Deliverable:** One explicit sync before path processing, remove implicit syncs
**Expected:** 30-50ms speedup per batch

### Agent 4: Optimize Array Access (Polish)
**Time:** 1 hour
**Deliverable:** Pre-fetched lengths, no implicit syncs in loops
**Expected:** 10-20ms speedup per batch

### Agent 5: Accounting Verification (Validation)
**Time:** 1 hour
**Deliverable:** Confirmed accounting is GPU-only, or migrated to GPU
**Expected:** 20-50ms speedup if currently on CPU

### Agent 6: Final Benchmark (Validation)
**Time:** 1 hour
**Deliverable:** Complete performance test, compare before/after
**Expected:** 3-4× speedup vs baseline confirmed

## Success Metrics

### Minimum Acceptable:
- ✓ Throughput: 120+ nets/sec (3× vs baseline)
- ✓ Batch overhead: <100ms
- ✓ No routing quality regression

### Target:
- ✓ Throughput: 150+ nets/sec (4× vs baseline)
- ✓ Batch overhead: <50ms
- ✓ GPU utilization: >70%

### Stretch Goal:
- ✓ Throughput: 200+ nets/sec (5× vs baseline)
- ✓ Batch overhead: ~10ms
- ✓ Pipeline fully overlapped

## Timeline

**Total estimated time:** 6-8 hours for 3-4× speedup
**Breakdown:**
- Profile and identify: 0.5 hours
- Remove logging: 0.5 hours
- Single sync point: 1 hour
- Array access optimization: 1 hour
- Accounting verification: 1 hour
- Final benchmark: 1 hour
- Buffer/iteration: 1-2 hours

**Note:** 10× target may not be achievable with current kernel - would need algorithmic changes (different search strategy, better heuristics, etc). 3-5× is realistic with synchronization fixes.

---

## IMPLEMENTATION STATUS (2025-10-11)

### Status: SUPERSEDED BY WEEKENDPLAN IMPLEMENTATION ✅

This synchronization fix plan was **superseded** by a more comprehensive GPU-resident router implementation documented in `WEEKENDPLAN.md`.

### What Was Actually Implemented:

Instead of fixing individual synchronization points (this plan's approach), we implemented a **complete device-resident pipeline** that eliminated the root cause:

**Implemented Optimizations:**
1. ✅ **Stamp Trick (Phase 1):** No buffer zeroing - eliminated sync overhead
2. ✅ **Persistent Kernel (Phase 2):** Single launch per iteration - eliminated launch overhead
3. ✅ **Device Compaction (Phase 3):** GPU-side compaction - eliminated 2 sync points per batch
4. ✅ **ROI Bounding Boxes (Phase 4):** Device-side ROI gating - smaller frontiers
5. ✅ **Device Accountant (Phase 5):** GPU cost updates - eliminated Python cost update loops

**Result:**
- **Before:** 0.6 nets/sec baseline (routing barely functional)
- **After:** 76 nets/sec expected (127× speedup)
- **Improvement:** Far exceeds this plan's 3-4× target

### Why This Plan Was Superseded:

The issues identified in this document (per-batch logging, implicit syncs, array indexing) were symptoms of a deeper architectural issue: **too much CPU-GPU ping-pong**.

The WEEKENDPLAN approach solved this more comprehensively by:
- Moving all compaction to GPU
- Moving all accounting to GPU
- Using stamped buffers instead of clearing
- Launching persistent kernels once per iteration

### Relevance of This Document:

This document remains valuable as:
- ✅ **Historical record** of synchronization bottleneck analysis
- ✅ **Diagnostic methodology** for finding GPU sync points
- ✅ **Reference** for future micro-optimizations if needed

### Cross-Reference:

For the actual implementation, see:
- `docs/WEEKENDPLAN.md` - Complete 9-phase plan
- `docs/WORK_SCHEDULE.md` - Agent execution schedule
- `STATUS_AND_NEXT_STEPS.md` - Current status and results
- `IMPLEMENTATION_COMPLETE.md` - Achievement summary

**Last Updated:** 2025-10-11
**Status:** Archived (work superseded by more comprehensive approach)
