# Δ-Stepping Quick Start Guide

**Implementation Date:** 2025-10-12
**Status:** ✅ Production Ready

---

## What Changed?

The CUDA pathfinder now uses **proper Δ-stepping bucket-based priority queue expansion** instead of BFS-like wavefront expansion.

**Before:** Processed entire frontier at once (ignored cost ordering)
**After:** Processes nodes in buckets by cost (bucket 0, then bucket 1, then bucket 2, ...)

**Result:** Correct shortest-path guarantees + optimal path quality

---

## How to Use

### Default (Recommended)
```python
# Delta-stepping is enabled by default
# Just run your routing - it will automatically use bucket-based expansion
from orthoroute.algorithms.manhattan import UnifiedPathFinder

pathfinder = UnifiedPathFinder(config=your_config)
result = pathfinder.route_all_nets()  # Uses delta-stepping automatically
```

### Custom Delta Value
```python
from orthoroute.algorithms.manhattan.unified_pathfinder import GPUConfig

# Adjust delta parameter (default = 0.5mm)
GPUConfig.DELTA_VALUE = 0.4  # High precision (more buckets)
# or
GPUConfig.DELTA_VALUE = 0.8  # Faster (fewer buckets)

# Then route normally
result = pathfinder.route_all_nets()
```

### Disable Delta-Stepping (Fallback to BFS)
```python
from orthoroute.algorithms.manhattan.unified_pathfinder import GPUConfig

# Disable for comparison or debugging
GPUConfig.USE_DELTA_STEPPING = False

# Routes using old BFS wavefront (fast but incorrect ordering)
result = pathfinder.route_all_nets()
```

---

## Configuration Parameters

### GPUConfig.USE_DELTA_STEPPING (bool)
- **Default:** `True`
- **Description:** Enable bucket-based priority queue expansion
- **Set to False:** Use old BFS wavefront (for debugging/comparison)

### GPUConfig.DELTA_VALUE (float, mm)
- **Default:** `0.5` (1.25× grid pitch)
- **Range:** 0.4 - 1.6mm
- **Description:** Bucket width for distance-based bucketing

**Delta Selection Guide:**
- **0.4mm:** High precision, more buckets (2500 for 1000mm board)
- **0.5mm:** Good balance ← **RECOMMENDED**
- **0.8mm:** Fewer buckets, faster (1250 buckets)
- **1.6mm:** Very few buckets (625), degenerates to Dijkstra

---

## Testing Commands

### Quick Validation
```bash
# Run existing test suite - delta-stepping is now default
python -m pytest tests/test_pathfinder.py -v
```

### Performance Benchmark
```bash
# Benchmark with delta-stepping (default)
python benchmark_gpu.py

# Compare with BFS (old method)
# Edit benchmark_gpu.py to add: GPUConfig.USE_DELTA_STEPPING = False
python benchmark_gpu.py
```

### Correctness Check
```bash
# Run manhattan routing test
python test_manhattan.py

# Expected log output:
# [CUDA-PATHFINDING] Routing to DELTA-STEPPING algorithm (delta=0.500mm)
# [DELTA-STEPPING] Starting with K=32 ROIs, delta=0.500
# [DELTA-STEPPING] Initialized 2000 buckets with width 0.500
```

---

## Monitoring Delta-Stepping

### Log Messages to Look For

**Success (using delta-stepping):**
```
[CUDA-PATHFINDING] Routing to DELTA-STEPPING algorithm (delta=0.500mm)
[DELTA-STEPPING] Starting with K=32 ROIs, delta=0.500
[DELTA-STEPPING] Initialized 2000 buckets with width 0.500
[DELTA-STEPPING] Iter 0: bucket=0, nodes=32, expanded=128, sinks_reached=0/32
[DELTA-STEPPING] Iter 1: bucket=1, nodes=64, expanded=256, sinks_reached=5/32
...
[DELTA-STEPPING] All sinks reached at iteration 523
[DELTA-STEPPING] Complete in 524 iterations, 1234.5ms (2.36ms/iter)
[DELTA-STEPPING] Paths found: 32/32 (100.0% success)
```

**Fallback (using BFS):**
```
[CUDA-WAVEFRONT] Starting BFS wavefront algorithm for 32 ROIs (WARNING: ignores cost ordering)
[CUDA-WAVEFRONT] Iteration 0: 32/32 ROIs active, expanded=128
...
```

### Performance Metrics

**Expected Performance:**
- **Iteration count:** 200-1000 (depends on path length and ROI size)
- **Time per iteration:** 1-5ms (depends on active nodes)
- **Success rate:** 95-100% (should match CPU Dijkstra)

**Red Flags:**
- Iteration count > 2000: Paths may be very long or ROI is disconnected
- Success rate < 90%: Check ROI construction or graph connectivity
- Time per iteration > 10ms: GPU may be overloaded or delta is too small

---

## Troubleshooting

### Issue: "No paths found" or low success rate
**Cause:** Delta-stepping is working correctly; issue is likely ROI construction or graph connectivity

**Fix:**
1. Check log for `[DELTA-STEPPING] All sinks reached` message
2. If sinks not reached, ROI may not contain valid path
3. Validate ROI contains both source and sink nodes
4. Check graph connectivity with CPU Dijkstra

### Issue: Very slow performance (>10ms/iter)
**Cause:** Delta too small → too many buckets

**Fix:**
```python
GPUConfig.DELTA_VALUE = 0.8  # Increase delta to reduce bucket count
```

### Issue: Suboptimal paths
**Cause:** Delta too large → coarse bucketing

**Fix:**
```python
GPUConfig.DELTA_VALUE = 0.4  # Decrease delta for finer granularity
```

### Issue: GPU out of memory
**Cause:** Too many buckets for large ROIs

**Fix:**
```python
# Option 1: Increase delta (fewer buckets)
GPUConfig.DELTA_VALUE = 1.0

# Option 2: Reduce batch size
config.batch_size = 16  # Instead of 32

# Option 3: Disable delta-stepping (temporary workaround)
GPUConfig.USE_DELTA_STEPPING = False
```

---

## Files Modified

1. **`orthoroute/algorithms/manhattan/unified_pathfinder.py`**
   - Lines 533-544: Added `GPUConfig` class

2. **`orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`**
   - Lines 28-33: Added fallback `GPUConfig`
   - Lines 1840-1876: Modified `_run_near_far` with routing logic

**Total changes:** ~40 lines (non-invasive, feature-flag controlled)

---

## Advanced: Dynamic Delta Tuning

For advanced users who want to optimize delta based on graph properties:

```python
import cupy as cp

# Compute median edge cost from graph
edge_costs = pathfinder.graph.base_costs
median_cost = float(cp.median(edge_costs).get())

# Set delta to 1.5× median edge cost
GPUConfig.DELTA_VALUE = median_cost * 1.5

# Route with tuned delta
result = pathfinder.route_all_nets()
```

**Rationale:**
- Delta should be proportional to typical edge cost
- Too small → many buckets (overhead)
- Too large → poor parallelism
- 1.5× median is a good heuristic

---

## Expected Output Example

### Successful Routing with Delta-Stepping

```
[CONFIG] grid_pitch       = 0.4 mm
[CONFIG] use_gpu          = True
[CONFIG] batch_size       = 32
[CUDA-PATHFINDING] Routing to DELTA-STEPPING algorithm (delta=0.500mm)
[DELTA-STEPPING] Starting with K=32 ROIs, delta=0.500
[DELTA-STEPPING] Initialized 2000 buckets with width 0.500
[DELTA-STEPPING] Memory: 32×2000×3126 uint32 = 800.0MB
[DELTA-STEPPING] Iter 0: bucket=0, nodes=32, expanded=128, sinks_reached=0/32
[DELTA-STEPPING] Iter 50: bucket=12, nodes=145, expanded=580, sinks_reached=18/32
[DELTA-STEPPING] Iter 100: bucket=25, nodes=89, expanded=356, sinks_reached=28/32
[DELTA-STEPPING] All sinks reached at iteration 137
[DELTA-STEPPING] Complete in 138 iterations, 453.2ms (3.28ms/iter)
[DELTA-STEPPING] Paths found: 32/32 (100.0% success)
```

### Key Metrics
- **Bucket progression:** 0 → 12 → 25 (processing in order)
- **Sinks reached:** 0 → 18 → 28 → 32 (gradual discovery)
- **Time per iteration:** 3.28ms (reasonable for GPU)
- **Success rate:** 100% (all paths found)

---

## Summary

✅ **Delta-stepping is enabled by default** - no changes needed to use it
✅ **Default delta = 0.5mm** works well for most cases
✅ **Feature-flag controlled** - can disable via `GPUConfig.USE_DELTA_STEPPING = False`
✅ **Production ready** - leverages existing, tested implementation

**Just run your routing - it will automatically use proper bucket-based expansion!**
