# Iteration 1: BFS-Only Approach

## Status: BROKEN - Sink nodes unreachable

## Test Configuration:
- Removed L-corridor entirely
- Using BFS for all nets (no radius cap)
- 300K node ROI limit
- Full blind/buried via support
- Grid pitch: 0.4mm
- Lattice: 611x567x12 = 4,157,244 nodes

## Metrics:

### ROI Sizes:
- BFS expansion before truncation: 2.7M-3M nodes (good, exploring most of board)
- After truncation: 300K nodes (at limit, as expected)
- Edge density in ROI: ~700K-750K edges per ROI

### Performance:
- ROI extraction: ~8-10 seconds per net (SLOW!)
- Total time for 32 nets: ~5 minutes just for ROI extraction
- Estimated total time: 8192 nets * 10 sec = 22 hours just for iteration 1!

### Sink Reachability: **0% - ALL INFINITE**
- Every single net shows `Sink_dist=inf`
- Example warnings:
  ```
  ROI 0: Near=1, Far=747, Thresh=13.2, Sink_dist=inf
  ROI 1: Near=1, Far=445, Thresh=110.1, Sink_dist=inf
  ROI 20: Near=1, Far=486, Thresh=30.9, Sink_dist=inf
  ```
- Near-Far thresholds are reasonable (12-156 mm)
- But sinks are not included in the truncated ROI!

### Paths Found:
- 0/32 in first batch (still running iterations, no paths found)
- Test appears stuck in infinite CUDA Dijkstra iterations

## Issues Found:

### 1. **CRITICAL: Sink nodes not preserved in truncated ROI**
The code claims "Verified 2/2 critical nodes preserved" but the CUDA Dijkstra shows `Sink_dist=inf` for ALL nets. This means:
- The BFS truncation is not actually including the sink nodes in the final ROI
- OR the sink nodes are being mapped to wrong IDs during ROI extraction
- OR the portal nodes (not raw pads) need to be preserved instead

### 2. **PERFORMANCE: ROI extraction too slow**
- 8-10 seconds per net for BFS + truncation
- Most of this is the 2.7M-3M node BFS expansion
- Need to limit BFS search radius BEFORE truncation

### 3. **BLOCKER: Cannot complete test in reasonable time**
- 22+ hours just for iteration 1 is unacceptable
- Need much faster ROI extraction

## Root Cause Analysis:

The BFS ROI extraction has a critical bug in how it preserves critical nodes:

1. BFS expands from source, finds 2.7M-3M nodes
2. Truncation keeps the "closest" 300K nodes by BFS distance
3. Code claims to verify sink nodes are preserved
4. BUT: The verification is checking the wrong thing!

Looking at the code flow:
- `_extract_bfs_roi_with_truncation()` preserves critical_nodes
- But it's preserving the **pad** nodes, not the **portal** nodes
- The actual routing happens portal-to-portal
- So even if the pad nodes are in the ROI, the portal nodes might not be!

## Fix Applied:

Need to check the code and ensure we're preserving PORTAL nodes, not just pad nodes.

Also need to add BFS radius limit BEFORE truncation to speed things up.

## Next Steps:

1. Stop current test (it's broken and slow)
2. Fix sink node preservation to use portal nodes
3. Add BFS radius cap (e.g., 200mm) before truncation
4. Re-run Iteration 2 with fixes
