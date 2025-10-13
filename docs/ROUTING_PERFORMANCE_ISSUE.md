# Routing Performance Issue - Investigation Needed

## Current Status

**Date**: 2025-10-12
**What's Working**: Grid bounds fix, GPU sort optimization
**What's Broken**: Routing success rate is extremely low (<1%)

## Recent Fixes Applied

### ✅ Fix 1: Grid Bounds (WORKING)
**File**: `orthoroute/algorithms/manhattan/unified_pathfinder.py:1563-1615`
**Change**: Modified `_calc_bounds()` to extract pads from `board.nets` instead of falling back to full board bounds

**Results**:
- Lattice reduced from 4,157,244 nodes to 2,474,712 nodes (40% reduction)
- Bounds correctly constrained to pad area + 3mm margin
- Before: (34-278mm × 29-256mm) - entire board
- After: (96.3-257.4mm × 32.8-237.6mm) - actual routing area

### ✅ Fix 2: GPU Sort (WORKING)
**File**: `orthoroute/algorithms/manhattan/unified_pathfinder.py:646-667`
**Change**: Fixed CuPy structured array issue by extracting 'src' field for GPU sorting

**Results**:
- GPU sort: 0.9 seconds at 34.9M edges/sec
- Before: 25+ seconds CPU fallback
- 28× speedup in graph construction phase

## The Problem

**Routing success rate is catastrophically low**: Only 1 out of 244 nets routed in first 2 batches (10 minutes runtime)

### Test Results (10-minute run)

```
[BOUNDS] Extracted 16384 pads from 8192 nets
[BOUNDS] Pad area: (99.3, 35.8) to (254.4, 234.6)
[BOUNDS] Final with 3.0mm margin: (96.3, 32.8) to (257.4, 237.6)
Lattice: 402×513×12 = 2,474,712 nodes
[GPU-SORT] GPU sort completed in 0.9 seconds (34.9M edges/sec)

[BATCH-1] Complete: 0/244 routed (0.0%), 177.78s total, 1.4 nets/sec
[BATCH-2] Complete: 1/244 routed (0.4%), 296.78s total, 0.8 nets/sec
```

### Key Observations

1. **Extremely slow routing**: ~1 net per 3 minutes (should be much faster)
2. **High failure rate**: 243 out of 244 nets failed to route
3. **Log spam**: Constant "BFS ROI 76,852 > 50,000, truncating" warnings
4. **ROI size issue**: Every net is hitting the 50k ROI limit and getting truncated

### Log Pattern (repeats for every net)

```
WARNING - BFS ROI 76,852 > 50,000, truncating (preserving 2 critical nodes)
INFO - Verified 2/2 critical nodes preserved
DEBUG - [CSR-EXTRACT] ROI size=50000, edges=434352, edge_density=0.000
INFO - [DEBUG] Net B10B11_114: Extracted ROI CSR subgraph
```

## Root Cause Hypothesis

The **ROI (Region of Interest) extraction** is the bottleneck:

1. **ROI too large**: BFS is finding 76,852 nodes for every net (well over the 50k limit)
2. **Indiscriminate truncation**: Cutting ROI to 50k may be removing critical routing paths
3. **Performance impact**: Processing 76k+ nodes per net even with truncation is slow
4. **Success rate**: Truncated ROIs may not contain valid paths, causing routing failures

## Investigation Areas

### 1. Why is BFS ROI so large?

**File to check**: `orthoroute/algorithms/manhattan/unified_pathfinder.py`
**Method**: `_extract_roi_subgraph()` or similar ROI extraction logic

**Questions**:
- Why is BFS expanding to 76,852 nodes for every net?
- Is BFS expansion unbounded or poorly constrained?
- Should ROI be spatially constrained (e.g., bounding box around src/dst)?
- Is the 50k limit appropriate for this design?

### 2. Is ROI truncation breaking paths?

**Questions**:
- When truncating from 76k to 50k, are we keeping the right nodes?
- Does the truncation preserve connectivity between src and dst?
- Should we use spatial heuristics (distance to src/dst) instead of BFS order?

### 3. Can we use smarter ROI extraction?

**Possible improvements**:
- Spatial bounding box: Only expand BFS within X mm of src/dst pads
- Adaptive limits: Increase ROI size if routing fails, decrease if successful
- Layer-aware expansion: Prioritize nodes on layers that connect src/dst
- Distance-based pruning: Keep nodes closer to direct path between endpoints

### 4. Is the actual pathfinding failing?

**Questions**:
- Are nets failing because paths don't exist in truncated ROI?
- Are nets failing due to congestion/overlaps?
- Is the GPU Dijkstra implementation correct?
- Are the extracted subgraphs valid (connectivity preserved)?

## Test Data

### Board Statistics
- **Nets**: 8192 routable nets
- **Pads**: 16384 pads total (2 pads per net average)
- **Lattice**: 402×513×12 = 2,474,712 nodes
- **Edges**: 32,160,276 edges total

### Routing Configuration
- Grid pitch: 0.4mm
- Layers: 12 (F.Cu + 10 internal + B.Cu)
- Via cost: 3.0
- Max iterations: 10
- ROI limit: 50,000 nodes

## Reproduction Steps

```bash
# Run test with full logging
ORTHO_CPU_ONLY=1 timeout 600 python main.py --test-manhattan 2>&1 | tee routing_test.txt

# Check key metrics
grep -E "(BOUNDS|Lattice:|GPU-SORT|BATCH.*Complete)" routing_test.txt

# Check ROI truncation frequency
grep "BFS ROI.*truncating" routing_test.txt | wc -l

# Check routing success
grep "BATCH.*Complete" routing_test.txt | tail -5
```

## Expected vs Actual

### Expected Behavior
- ROI size: 10k-30k nodes per net (localized search)
- Routing speed: 10-50 nets/sec
- Success rate: >90% of nets routed
- Runtime: Complete 8192 nets in 2-5 minutes

### Actual Behavior
- ROI size: 76,852 nodes per net (entire lattice?)
- Routing speed: 0.3 nets/sec (1 net per 3 minutes)
- Success rate: <1% (1 out of 244 nets)
- Runtime: Would take 7+ hours to complete at current rate

## Recommended Next Steps

1. **Investigate ROI extraction**: Find why BFS expands to 76k nodes
2. **Add spatial constraints**: Limit ROI to bounding box around src/dst + margin
3. **Profile routing time**: Identify if time is spent in BFS, CSR extraction, or Dijkstra
4. **Verify subgraph validity**: Ensure truncated ROIs still contain valid paths
5. **Add ROI size histogram**: Log distribution of ROI sizes across different nets

## Files to Check

### Primary suspects
- `orthoroute/algorithms/manhattan/unified_pathfinder.py` - ROI extraction logic
- `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py` - GPU pathfinding

### Search patterns
```bash
# Find ROI extraction code
grep -n "BFS ROI" orthoroute/algorithms/manhattan/unified_pathfinder.py

# Find ROI limit constants
grep -n "50000\|50_000" orthoroute/algorithms/manhattan/unified_pathfinder.py

# Find BFS expansion logic
grep -n "def.*roi\|def.*bfs" orthoroute/algorithms/manhattan/unified_pathfinder.py -i
```

## Success Criteria

When fixed, we should see:
- ROI sizes: <30k nodes for most nets
- Routing speed: >10 nets/sec
- Success rate: >90% of nets routed in iteration 1
- Runtime: Complete 8192 nets in <5 minutes
- No truncation warnings (or very few)

## Additional Context

- User confirmed routing area should be "bounding box of all vias + 3mm" (now correctly implemented)
- Grid bounds fix was successful (lattice size reduced 40%)
- GPU acceleration is working (sort is 28× faster)
- The bottleneck has shifted from graph construction to actual routing

## Test Output Location

Full test log available at: `test_10min_full_run.txt`
