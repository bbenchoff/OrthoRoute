# CSR Graph Structure Validation Report

## Issue Identified

**CRITICAL BUG FOUND AND FIXED** in `unified_pathfinder.py` line 2460.

### The Problem

When using the full graph (all 4.2M nodes), the code was passing **ROI-local indices** to the routing kernel instead of **global indices**.

**Before Fix (Line 2430-2460):**
```python
roi_src = int(global_to_roi[src])  # Convert to ROI-local index
roi_dst = int(global_to_roi[dst])

if roi_size == full_graph_size:
    # Full graph: use existing CSR structure
    roi_indptr = indptr  # Full graph CSR (4.2M nodes)
    roi_indices = indices
    roi_weights = costs_cpu

# BUG: Passing ROI-local indices with global CSR!
roi_batch.append((roi_src, roi_dst, roi_indptr, roi_indices, roi_weights, roi_size))
```

### Why This Failed

- **CSR arrays**: Full graph with 4,157,244 nodes and 54,030,036 edges
- **Source/Dest**: Were converted to ROI-local indices (e.g., src=0, dst=1)
- **Kernel**: Tried to expand from node 0 in a 4.2M-node graph
- **Result**: Only explored ~14 neighbors, couldn't find paths

The wavefront expansion was stuck because it started from the wrong nodes!

### The Fix

**After Fix:**
```python
if roi_size == full_graph_size:
    # Full graph: use existing CSR structure
    roi_indptr = indptr
    roi_indices = indices
    roi_weights = costs_cpu
    # CRITICAL FIX: For full graph, use GLOBAL indices (not ROI-local)
    actual_roi_src = src  # Use global src
    actual_roi_dst = dst  # Use global dst
else:
    # ROI subgraph: extract CSR
    roi_indptr, roi_indices, roi_weights = ...
    # For ROI subgraph, use local indices
    actual_roi_src = roi_src
    actual_roi_dst = roi_dst

roi_batch.append((actual_roi_src, actual_roi_dst, roi_indptr, roi_indices, roi_weights, roi_size))
```

## Validation Results

### CSR Structure (VERIFIED CORRECT)

From test logs:
```
CSR: 4157244 nodes, 54030036 edges
Vias: 45,729,684 edges (22864842 bidirectional pairs)
```

**Expected edge count:**
- Grid: 611×567×12 layers
- H-layers (6 layers): 6 × 567 × 610 × 2 = 4.15M edges
- V-layers (6 layers): 6 × 611 × 566 × 2 = 4.15M edges
- Via edges: 45.73M edges
- **Total**: 54.03M edges ✅ **MATCHES!**

### CSR Transfer Validation (NO CORRUPTION)

From CSR-VALIDATION logs:
```
[CSR-VALIDATION] ROI 0: src=2105061, dst=1926945, roi_size=4157244
[CSR-VALIDATION] indptr_len=4157245, indices_len=54030036
[CSR-VALIDATION] src=2105061 has 13 neighbors in CSR
[CSR-VALIDATION] src=2105061 -> neighbor[0]=26439, weight=5.252
```

**Verified:**
- ✅ `indptr_len = roi_size + 1` (correct CSR format)
- ✅ `indices_len = 54,030,036` (no truncation)
- ✅ Source nodes have valid edges (13 neighbors = 12 via edges + 1 lateral edge)
- ✅ Edge weights are reasonable (5.25 = via cost with span penalty)

### Wavefront Expansion (NOW WORKING!)

**Before fix:**
```
Iteration 0: finite_dists=14/4157244 (stuck at 14 nodes)
Iteration 10: finite_dists=14/4157244 (no progress)
Iteration 100: finite_dists=14/4157244 (completely stuck)
```

**After fix:**
```
Iteration 0: finite_dists=14/4157244 (starting from source)
Iteration 130: finite_dists=398623/4157244 (398K nodes explored!)
Iteration 200: finite_dists=659029/4157244 (659K nodes explored!)
```

The wavefront is now **properly expanding** through the graph at ~4000 nodes/iteration!

## Root Cause Analysis

### Why This Bug Existed

The code path for "full graph optimization" was added to skip CPU-based CSR extraction (a 3-second bottleneck). However, it failed to account for the index space difference:

1. **ROI subgraph path**: Extracts subset of nodes → remaps to local indices [0, N-1] → passes local indices
2. **Full graph path**: Uses full CSR → BUT STILL REMAPPED TO LOCAL INDICES → **WRONG!**

The full graph case should have used identity mapping (global indices = local indices), but instead it used the ROI extractor's mapping which was incorrect for a full graph.

### Impact

- **Symptom**: All routes failed with "no path found"
- **Cause**: Kernel searched from wrong starting nodes
- **Duration**: Present since full-graph optimization was added
- **Severity**: CRITICAL - routing completely non-functional

## Recommendations

1. ✅ **Fix Applied**: Use global indices for full graph case
2. ✅ **Validation Added**: CSR transfer validation logs
3. ⚠️ **Next Steps**:
   - Verify paths are now being found successfully
   - Check if portal placement is correct (routes still failing may indicate escape issues)
   - Consider adding unit tests for CSR index space handling

## Files Modified

1. **`orthoroute/algorithms/manhattan/unified_pathfinder.py`** (lines 2437-2467)
   - Added `actual_roi_src` and `actual_roi_dst` variables
   - Conditional logic: full graph uses global indices, ROI uses local indices

2. **`orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`** (lines 402-411, 443-455)
   - Added CSR validation logging
   - Verifies array sizes match expected dimensions
   - Logs source node connectivity after transfer

## Conclusion

**The CSR graph structure is CORRECT and VALID.**

The issue was not with graph construction or CSR transfer, but with **passing the wrong indices** to the routing kernel. The fix ensures that when using the full graph CSR, we pass global node indices instead of ROI-local indices.

The wavefront expansion now works properly, exploring hundreds of thousands of nodes as expected.
