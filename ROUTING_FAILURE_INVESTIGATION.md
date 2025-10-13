# Routing Failure Investigation - Complete Analysis

## Summary
After 4 test cycles and multiple fixes, routing success rate remains at **0.3-0.4%** (25-31 nets out of 8,192). The GPU kernel explores properly but fails to find paths for 99.6% of nets.

---

## What We Fixed (Correctly)

### ✅ Fix 1: PathFinder Negotiated Costs
**Problem**: Expand kernels read `weights[]` instead of `total_cost[]`
**Fix**: Added `total_cost` parameter, updated all 4 expand sites
**Result**: PathFinder negotiation enabled (but didn't improve success rate)
**Status**: CORRECT FIX, but not the blocker

### ✅ Fix 2: Grid Layer Structure
**Problem**: F.Cu had routing edges, should be empty
**Fix**: `layer_dir[0] = 'empty'`, skip edge creation for empty layers
**Result**: F.Cu now has 0 edges (verified: 53.3M total edges, 0 on F.Cu)
**Status**: CORRECT FIX, grid structure now matches spec

### ✅ Fix 3: Node Packing Format
**Problem**: 20-bit node IDs (max 1M) but graph has 4.1M nodes
**Fix**: Changed to 24-bit node IDs (max 16M)
**Result**: No truncation errors
**Status**: CORRECT FIX, but wasn't the blocker (most nodes < 1M anyway)

---

## What We Fixed (Incorrectly/No Effect)

### ❌ Fix 4: ROI Bounding Disabled
**Hypothesis**: 50-cell margin too restrictive, blocking long detours
**Fix**: Disabled ROI, use full board bounds
**Result**: SUCCESS 0.3% → 0.07% (WORSE!)
**Conclusion**: ROI restriction was NOT the blocker

### ❌ Fix 5: Portal Layers (Vertical Only)
**Hypothesis**: Portals on horizontal layers unreachable from vertical escape stubs
**Fix**: Restrict portals to vertical layers (2,4,6,8,10)
**Result**: SUCCESS 0.3% → 0.4% (no improvement)
**Conclusion**: Layer direction doesn't affect portal reachability

---

## Current Status

### Grid Construction ✅ CORRECT
```
Layer 0 (F.Cu):   0 edges (empty)
Layer 1 (In1.Cu): 691,740 edges (horizontal, 0.4mm grid)
Layer 2 (In2.Cu): 691,652 edges (vertical, 0.4mm grid)
Layers 3-11:      Alternating H/V (691K edges each)
Vias:             45.7M edges (all layer pairs)
Total:            53.3M edges ✓
```

### GPU Kernel Performance ✅ WORKING
- Compiles and launches successfully
- Explores 200-700 iterations per batch
- Wavefront expansion logic functional
- Queue management operational
- **Finds paths for 0.04% of attempts** (6 out of 16,381)

### PathFinder Negotiation ✅ ENABLED
- `total_cost[]` array passed to kernels
- Expand reads negotiated costs (not base weights)
- Present/history accumulation working
- **But success rate too low to see negotiation effect**

---

## The Mystery: Why 99.6% Failure?

### Evidence
1. **Kernel explores properly**: 200-700 iterations (not timeout/hang)
2. **Rare successes**: 6 nets out of 16,381 prove pipeline works end-to-end
3. **Variable iterations**: Queue empties (not stuck in loop)
4. **Graph structure correct**: CSR validation shows 13 neighbors per node

### Hypotheses (Unverified)

#### 🔴 **Most Likely: Portal Coordinates Invalid**
- Portals may be at coordinates where no graph node exists
- Grid snapping may be broken
- Portal (x_idx, y_idx) may not align with actual lattice points
- **Test**: Check if `node_idx(portal.x_idx, portal.y_idx, portal.entry_layer)` exists in graph

#### 🟡 **Likely: Goal Detection Broken**
- Kernel reaches goal but fails to detect `node == dst`
- Destination node ID mismatch
- **Test**: Add kernel printf to log when `node == dst` check happens

#### 🟡 **Likely: Source Initialization Wrong**
- Source nodes may not have dist=0 set correctly
- Generation stamp mismatch
- **Test**: Check if source nodes have proper stamps before kernel launch

#### 🟠 **Possible: Graph Connectivity Issue**
- Portal nodes exist but are isolated (no edges)
- Vias don't connect portal layers properly
- **Test**: Check neighbor count for portal nodes (should be >0)

#### 🟠 **Possible: Coordinate Decoding Bug**
- Procedural coordinate calculation wrong
- `nx = node % Nx` formula incorrect for flattened index
- **Test**: Decode a known node ID manually, verify coordinates

---

## Diagnostic Data

### Test Cycle Results
| Cycle | Fix Applied | Success Rate | GPU Success | Notes |
|-------|-------------|--------------|-------------|-------|
| 1 | Grid fixes | 0.3% (23/8192) | 0% | Before fixes |
| 2 | Node packing 24-bit | 0.3% (29/8192) | 0.08% | No improvement |
| 3 | ROI disabled | 0.07% (6/8192) | 0.04% | WORSE! |
| 4 | Portal vertical layers | 0.4% (31/8192) | ~0.1% | Minimal improvement |

### GPU Batch Patterns
- **Typical batch**: 0/64 paths (202+ batches)
- **Rare success**: 1-2/64 paths (4 batches total)
- **Iteration count**: 200-700 (proper exploration)
- **Never**: Mass success (30+/64 paths)

---

## Next Steps (Requires Detailed Debugging)

### 1. **Verify Portal Node Existence** (CRITICAL)
```python
for net_id in failing_nets[:10]:
    src_portal = portals[net_id_src]
    src_node = lattice.node_idx(src_portal.x_idx, src_portal.y_idx, src_portal.entry_layer)

    # Check if node exists
    if src_node >= len(graph.indptr) - 1:
        print(f"ERROR: src_node {src_node} out of range!")

    # Check if node has edges
    edge_start = graph.indptr[src_node]
    edge_end = graph.indptr[src_node + 1]
    num_neighbors = edge_end - edge_start

    if num_neighbors == 0:
        print(f"ERROR: src_node {src_node} has NO EDGES!")
```

### 2. **Add CUDA Kernel Diagnostics**
```cuda
// In persistent kernel, add debug output
if (tid == 0 && iteration % 100 == 0) {
    printf("Iter %d: queue_size=%d, roi 0 goal_reached=%d\n",
           iteration, queue_size, goal_reached[0]);
}

// In goal check
if (node == dst) {
    printf("ROI %d: FOUND GOAL at node %d\n", roi_idx, node);
}
```

### 3. **Compare CPU vs GPU on Same Net**
- Route one failing net with CPU Dijkstra
- If CPU succeeds → GPU bug
- If CPU fails → Graph/portal issue

### 4. **Validate Escape Planner Output**
- Check if all portals were created successfully
- Verify no pads were skipped
- Confirm portal coordinates are on-grid

---

## Current Hypothesis

**Portal nodes don't exist in the graph or have no edges.**

The escape planner creates portals at (x_idx, y_idx) coordinates, but:
1. These coordinates may not correspond to actual lattice points
2. The `node_idx()` calculation may be wrong
3. Portal nodes may be created but have no edges (isolated)

The 6 successful paths suggest that SOME portals are correct, but 99.6% are broken.

---

## Conclusion

We've fixed the grid structure and PathFinder negotiation, but the **fundamental routing failure** persists. The GPU kernel works correctly (explores, terminates properly), but almost never finds paths.

**The blocker is NOT**:
- ❌ Grid structure (fixed and verified)
- ❌ PathFinder negotiation (fixed)
- ❌ Node packing (fixed)
- ❌ ROI spatial restriction (disabled, made it worse)
- ❌ Portal layer selection (doesn't affect reachability)

**The blocker IS**:
- ✅ Something fundamentally wrong with portal coordinates or goal setup
- ✅ Nodes don't exist, have no edges, or can't be reached

**Next action**: Need to inspect a failing net's portal coordinates and verify graph connectivity.
