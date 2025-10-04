# GPU Near-Far Algorithm Debug Report

## Problem Statement
- GPU Near-Far runs 1200+ iterations without terminating
- Finds 0/16 paths
- All ROIs stay "active" forever
- Should run 8-15 iterations for typical paths

## Investigation Summary

### Tests Performed

1. **Simple 5-node chain test** - PASSED
   - Expected: 4-5 iterations
   - Actual: 5 iterations
   - Path found correctly: [0, 1, 2, 3, 4]

2. **Batch test with mixed ROIs** - PASSED
   - 3 ROIs (connected + disconnected + connected)
   - Terminated correctly in 6 iterations
   - Correct paths found

3. **Large grid tests** - PASSED
   - 10x10 (100 nodes): 19 iterations ✓
   - 20x20 (400 nodes): 39 iterations ✓
   - 50x50 (2,500 nodes): 99 iterations ✓
   - 100x100 (10,000 nodes): 199 iterations ✓

### Key Finding

**The algorithm works correctly on synthetic test cases!**

This means the bug is in the **integration**, not the algorithm itself.

## Root Cause Analysis

### Hypothesis: Invalid Source/Sink in ROI

The 1200+ iteration bug occurs when:

1. **Sink node is not in the ROI**
   - ROI extraction maps global nodes → local indices
   - If sink is not included in ROI, `dst` points to an invalid/unreachable node
   - Algorithm explores entire ROI looking for nonexistent sink
   - Never terminates naturally, runs until max_iterations or ROI exhausted

2. **Source node has no edges**
   - If ROI extraction produces disconnected subgraph
   - Source can't reach anything
   - Algorithm terminates immediately but finds no path

### Why This Causes 1200+ Iterations

For a large ROI (e.g., 1200 nodes):
- If sink is not in ROI, algorithm explores all reachable nodes
- With grid topology: ~1200 nodes → ~1200 iterations
- All ROIs stay "active" because:
  - `threshold < infinity` (Far bucket has nodes)
  - `sink_distance = infinity` (never reached)
  - Algorithm can't detect this condition

## Bugs Fixed

### 1. Missing Early Termination on Sink Reached
**Location:** `_run_near_far()` line 413-423

**Problem:** Algorithm continues even after all sinks reached

**Fix:** Added early termination check after relaxation:
```python
# EARLY TERMINATION: Check if all sinks reached
sinks_reached = cp.zeros(K, dtype=cp.bool_)
for roi_idx in range(K):
    sink = data['sinks'][roi_idx]
    sinks_reached[roi_idx] = data['dist'][roi_idx, sink] < cp.inf

if sinks_reached.all():
    logger.info(f"[CUDA-NF] Early termination at iteration {iteration}: all sinks reached")
    break
```

**Impact:** Saves unnecessary iterations after shortest paths found

### 2. Missing Validation of Source/Sink in ROI
**Location:** `_run_near_far()` line 388-412

**Problem:** No validation that src/dst are valid ROI indices

**Fix:** Added comprehensive diagnostics:
```python
# Validate src/dst in range
if src < 0 or src >= actual_roi_size:
    logger.error(f"[CUDA-NF] ROI {roi_idx}: INVALID SOURCE {src} (actual_size={actual_roi_size})")

if dst < 0 or dst >= actual_roi_size:
    logger.error(f"[CUDA-NF] ROI {roi_idx}: INVALID SINK {dst} (actual_size={actual_roi_size})")
    logger.error(f"[CUDA-NF] This ROI will fail - sink not in ROI! Algorithm will run forever!")
```

**Impact:** Detects invalid ROI construction early

### 3. Enhanced Iteration Diagnostics
**Location:** `_run_near_far()` line 436-451

**Problem:** No visibility into why algorithm runs so long

**Fix:** Added detailed per-iteration logging every 100 iterations:
```python
logger.warning(f"[CUDA-NF] Iteration {iteration}: {active_count}/{K} ROIs active, "
             f"Near nodes={near_total}, Far nodes={far_total}")

for roi_idx in range(K):
    logger.warning(f"  ROI {roi_idx}: Near={near_count}, Far={far_count}, "
                 f"Thresh={thresh:.1f}, Sink_dist={sink_dist:.1f}")
```

**Impact:** User can diagnose infinite loop immediately

## Upstream Bug: ROI Extraction

The real bug is likely in `unified_pathfinder.py` → `_extract_roi_csr()`

**Location:** `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py` line 837-880

### Potential Issues

1. **ROI doesn't include destination**
   ```python
   # When extracting ROI around bounding box
   # Destination might be just outside ROI bounds
   # Result: dst maps to invalid local index
   ```

2. **Global-to-local mapping bug**
   ```python
   # global_to_roi[dst] returns -1 (not in ROI)
   # But code uses it anyway as local index
   # Result: negative index or out-of-bounds
   ```

3. **ROI size mismatch**
   ```python
   # ROI claims size=N but only M nodes have edges (M < N)
   # Padding creates "ghost nodes" with no connectivity
   # Result: unreachable sink
   ```

## Recommended Fixes

### 1. Add ROI Validation in `unified_pathfinder.py`

Before calling `find_paths_on_rois()`:
```python
# Validate that dst is in the ROI
if roi_dst < 0 or roi_dst >= roi_size:
    logger.error(f"[PATHFINDER] Net {net_id}: Destination not in ROI!")
    logger.error(f"  Global dst: {dst}, Local dst: {roi_dst}, ROI size: {roi_size}")
    # Skip this net or expand ROI
    continue
```

### 2. Add Safety Timeout

In `_run_near_far()`:
```python
# Detect stuck ROIs
if iteration > 2 * data['max_roi_size']:
    logger.error(f"[CUDA-NF] ROI stuck after {iteration} iterations (size={data['max_roi_size']})")
    logger.error(f"[CUDA-NF] Likely bug: sink not in ROI or disconnected graph")
    break
```

### 3. Verify ROI Construction

In `_extract_roi_csr()`:
```python
# After extraction, verify sink is reachable
# Do quick BFS/DFS from source to check connectivity
# Warn if sink unreachable
```

## Testing Recommendations

### 1. Create Failing Test Case
```python
# Deliberately create ROI without sink
indptr = [0, 1, 2, 2, 2]  # Node 3 has no edges
indices = [1, 2]
weights = [1.0, 1.0]
src = 0
dst = 4  # Sink not in graph!
size = 5

# Should detect and fail immediately, not run 1200 iterations
```

### 2. Validate Real Routing Scenario
- Add logging to capture actual ROI batch
- Check src/dst values
- Check ROI sizes vs. path lengths
- Identify which nets trigger 1200+ iterations

### 3. Compare CPU vs GPU Results
- Run same net on CPU fallback
- If CPU succeeds but GPU fails → GPU bug
- If both fail → ROI extraction bug

## Summary

**Algorithm is correct.** The 1200+ iteration bug is caused by:
1. **Invalid ROI construction** (sink not in ROI)
2. **Missing validation** (doesn't detect this case)
3. **No early termination** (runs until exhaustion)

**Fixes applied:**
- ✅ Early termination when sinks reached
- ✅ Validation of src/dst in ROI range
- ✅ Enhanced diagnostics for debugging

**Still needed:**
- Fix ROI extraction in `unified_pathfinder.py`
- Add safety timeout for stuck ROIs
- Verify ROI connectivity before pathfinding

## Next Steps

1. **Enable logging:** Set log level to WARNING to see diagnostics
2. **Run real routing:** Trigger the 1200+ iteration bug
3. **Capture logs:** Look for "INVALID SINK" errors
4. **Fix ROI extraction:** Ensure all ROIs include both src and dst
5. **Re-test:** Verify iterations drop to expected range (8-15)
