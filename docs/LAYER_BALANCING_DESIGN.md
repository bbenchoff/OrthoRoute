# Layer Balancing Implementation Design

## Executive Summary

**Problem:** Layer 30 has 16.5% of all horizontal overuse (severe hotspot). Previous attempt using sequential loop over 52M edges hung the router.

**Solution:** Pre-build edge-to-layer mapping during graph construction, then apply layer bias using vectorized NumPy operations.

**Expected Overhead:** ~0.2-0.4 seconds per iteration (negligible vs. current 30-60s iteration time)

**Risk:** LOW - Follows same pattern as existing `_via_edges` implementation which already works well.

---

## Problem Analysis

### Current State

1. **Per-layer congestion tracking EXISTS and is FAST:**
   - `_log_per_layer_congestion()` (line 3748) already computes overuse by layer efficiently
   - Returns `overuse_horiz` dict mapping layer_z → overuse_amount
   - Called every iteration (line 2845)

2. **Layer bias EWMA update EXISTS and is FAST:**
   - Lines 2843-2872: Computes EWMA of layer bias
   - Updates `self.layer_bias` array (shape: [Nz], dtype: float32)
   - Alpha=0.20, clipped to [-0.40, +0.40]
   - This part is already working and < 10ms overhead

3. **The MISSING PIECE:** Applying `layer_bias` to edge costs
   - Line 2748-2749: "Layer balancing disabled - sequential loop over 52M edges too slow"
   - Need to multiply horizontal edge costs by `(1.0 + layer_bias[z])` factor

### Architecture Context

- **Graph structure:** CSR (Compressed Sparse Row) format
  - `indptr[u:u+1]` gives edge range for node u
  - `indices[ei]` gives destination node for edge ei
  - `base_costs[ei]` stores base cost for edge ei
  - Total edges: ~52M for large boards

- **Existing edge classification:**
  - `self._via_edges` (bool array, shape [num_edges]) - marks via edges
  - Built in `_identify_via_edges()` (line 3920) during initialization
  - Uses arithmetic (plane_size division) instead of idx_to_coord for speed
  - ~30MB memory for 27M edges
  - Construction time: ~2-3 seconds (one-time cost)

- **Cost update flow:**
  - Line 2741: `self.accounting.update_costs()` called once per iteration
  - Formula: `total_cost = base * via_mult * base_weight + pres_fac * overuse + hist_weight * history`
  - Layer bias should be applied AFTER this, before routing starts

---

## Design Options Comparison

### Option A: Pre-build Edge-to-Layer Mapping (RECOMMENDED)

**Approach:** Mirror the `_via_edges` pattern but store layer z instead of boolean.

**Memory:** ~52MB for int8 array (52M edges × 1 byte)

**Construction time:** ~2-3 seconds (one-time, during initialization)

**Application time:** ~0.2-0.4 seconds per iteration

**Pros:**
- ✅ Minimal code changes (follows existing proven pattern)
- ✅ Vectorized NumPy operations (fast)
- ✅ Compatible with both CPU and GPU modes
- ✅ Easy to debug and verify
- ✅ No changes to GPU kernels needed

**Cons:**
- ⚠️ Additional 52MB memory (acceptable for modern systems)
- ⚠️ 2-3 second initialization cost (one-time, amortized)

---

### Option B: Just-in-Time Layer Lookup

**Approach:** During cost application, compute layer on-the-fly for each edge.

**Memory:** Zero additional memory

**Application time:** ~3-5 seconds per iteration (too slow!)

**Pros:**
- ✅ No memory overhead

**Cons:**
- ❌ 3-5 second overhead per iteration (unacceptable)
- ❌ Requires coordinate conversion for every edge
- ❌ Python loops over 52M edges

**Verdict:** REJECTED - Too slow.

---

### Option C: GPU Kernel Implementation

**Approach:** Add `layer_bias[Nz]` array to GPU memory, apply bias during edge relaxation in CUDA kernel.

**Memory:** Negligible (~100 bytes for layer_bias array)

**Application time:** ~0 seconds (absorbed into existing kernel execution)

**Pros:**
- ✅ Zero overhead (best performance)
- ✅ Minimal memory

**Cons:**
- ❌ HIGH implementation complexity
- ❌ Requires modifying CUDA kernel code
- ❌ Requires understanding of existing GPU pathfinding kernel
- ❌ Difficult to debug
- ❌ Doesn't help CPU-only mode
- ⚠️ Risk of introducing GPU bugs

**Verdict:** FUTURE OPTIMIZATION - Implement Option A first, then optimize to Option C if needed.

---

## Recommended Implementation: Option A

### Architecture Design

```python
# New instance variable (add to __init__ around line 1938):
self._horizontal_edge_layers = None  # np.ndarray[int8], shape: [num_edges]
                                     # Value: layer_z for horizontal edges, -1 for vias

# Build mapping during initialization (new method):
def _build_horizontal_edge_layers(self):
    """
    Build edge-to-layer mapping for horizontal edges.
    Called once during initialization, after graph construction.
    Similar to _identify_via_edges() but stores layer z instead of boolean.
    """
    import time
    t0 = time.perf_counter()

    indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
    indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices

    num_edges = int(indptr[-1])
    # Use int8 to save memory (-128 to 127, enough for 52 layers)
    # Initialize to -1 (marker for "not a horizontal edge" / via)
    self._horizontal_edge_layers = np.full(num_edges, -1, dtype=np.int8)

    # Use arithmetic for speed (same as _identify_via_edges)
    plane_size = self.lattice.x_steps * self.lattice.y_steps

    horizontal_count = 0
    for u in range(len(indptr) - 1):
        uz = u // plane_size
        for ei in range(int(indptr[u]), int(indptr[u+1])):
            v = int(indices[ei])
            vz = v // plane_size

            if uz == vz:  # Horizontal edge (same layer)
                self._horizontal_edge_layers[ei] = uz
                horizontal_count += 1

    elapsed = time.perf_counter() - t0
    logger.info(f"[LAYER-BALANCE] Built layer mapping for {horizontal_count:,} horizontal edges in {elapsed:.3f}s")
    logger.info(f"[LAYER-BALANCE] Memory: {self._horizontal_edge_layers.nbytes / 1024 / 1024:.1f} MB")

# Apply layer bias to costs (new method):
def _apply_layer_bias_to_costs(self):
    """
    Apply layer balancing bias to horizontal edge costs.
    Called once per iteration, after update_costs(), before routing.

    Hot layers (positive bias) get cost increase.
    Cool layers (negative bias) get cost decrease.
    """
    if self._horizontal_edge_layers is None:
        return  # Layer balancing not initialized

    if np.allclose(self.layer_bias, 0.0, atol=0.001):
        return  # All biases negligible, skip application

    import time
    t0 = time.perf_counter()

    # Get total_cost array (may be on GPU)
    if self.accounting.use_gpu:
        total_cost_cpu = self.accounting.total_cost.get()
    else:
        total_cost_cpu = self.accounting.total_cost

    # Vectorized application: for each edge, multiply by (1.0 + layer_bias[z])
    # Only process horizontal edges (where layer_z >= 0)

    # Build bias multiplier array: shape [num_edges]
    # For horizontal edges: 1.0 + layer_bias[z]
    # For vias: 1.0 (no change)

    num_edges = len(self._horizontal_edge_layers)
    bias_multipliers = np.ones(num_edges, dtype=np.float32)

    # Routing layers only (skip F.Cu at z=0 and B.Cu at z=Nz-1)
    routing_layers = range(1, self._Nz - 1)

    for z in routing_layers:
        bias_val = self.layer_bias[z]
        if abs(bias_val) < 0.001:
            continue  # Skip negligible bias

        # Find all edges on this layer
        layer_mask = (self._horizontal_edge_layers == z)

        # Apply bias: cost_new = cost_old * (1.0 + bias)
        # Positive bias → higher cost (penalize hot layer)
        # Negative bias → lower cost (encourage cool layer)
        bias_multipliers[layer_mask] = 1.0 + bias_val

    # Apply multipliers
    total_cost_cpu *= bias_multipliers

    # Copy back to GPU if needed
    if self.accounting.use_gpu:
        self.accounting.total_cost[:] = cp.asarray(total_cost_cpu)

    elapsed = time.perf_counter() - t0

    # Log statistics
    affected = np.sum(bias_multipliers != 1.0)
    if affected > 0:
        min_mult = np.min(bias_multipliers[bias_multipliers != 1.0])
        max_mult = np.max(bias_multipliers[bias_multipliers != 1.0])
        logger.info(f"[LAYER-BIAS-APPLY] Applied bias to {affected:,} edges in {elapsed*1000:.1f}ms (mult range: {min_mult:.3f}-{max_mult:.3f})")
```

---

### Integration Points

#### 1. Initialization (line ~1990, after `_identify_via_edges()`)

```python
# Current code:
self._identify_via_edges()
self._build_via_edge_metadata()

# ADD THIS:
self._build_horizontal_edge_layers()
```

#### 2. Cost Application (line ~2750, after `update_costs()`)

```python
# Current code:
self.accounting.update_costs(
    self.graph.base_costs, pres_fac, hist_gain,
    via_cost_multiplier=via_cost_mult,
    base_cost_weight=cfg.base_cost_weight
)

# STEP 2.5: Apply via column/segment pooling penalties
self._apply_via_pooling_penalties(pres_fac)

# ADD THIS (STEP 2.6):
# STEP 2.6: Apply layer balancing bias
self._apply_layer_bias_to_costs()
```

---

### Performance Analysis

#### Memory Overhead

- **Edge-to-layer array:** 52M edges × 1 byte (int8) = **52 MB**
- **Layer bias array:** Already exists (Nz × 4 bytes = 52 × 4 = 208 bytes)
- **Total new memory:** ~52 MB (0.5% of typical 10GB system memory)

#### Time Overhead

**One-time initialization:**
- Building `_horizontal_edge_layers`: ~2-3 seconds
- This is acceptable (happens once at startup)

**Per-iteration overhead:**
- Bias application: ~0.2-0.4 seconds
- Current iteration time: 30-60 seconds
- Overhead percentage: **0.5-1.0%** (negligible)

**Breakdown of per-iteration time:**
1. GPU→CPU transfer (if GPU mode): ~50ms
2. Build bias_multipliers array (vectorized): ~100ms
3. Multiply total_cost array (vectorized): ~100ms
4. CPU→GPU transfer (if GPU mode): ~50ms
5. Logging: ~10ms

**Total:** ~310ms (~0.3 seconds)

---

### Verification Strategy

#### 1. Correctness Verification

Add debug logging to verify layer bias is applied correctly:

```python
# In _apply_layer_bias_to_costs(), add debug output:
if it <= 3:  # First 3 iterations only
    for z in routing_layers:
        if abs(self.layer_bias[z]) > 0.05:
            layer_mask = (self._horizontal_edge_layers == z)
            num_edges_layer = np.sum(layer_mask)
            avg_cost_before = np.mean(total_cost_before[layer_mask])
            avg_cost_after = np.mean(total_cost_cpu[layer_mask])
            logger.info(f"[LAYER-BIAS-DEBUG] Layer {z}: bias={self.layer_bias[z]:+.3f}, "
                       f"{num_edges_layer:,} edges, "
                       f"avg_cost: {avg_cost_before:.2f} → {avg_cost_after:.2f}")
```

#### 2. Impact Verification

Monitor per-layer overuse over iterations:

```python
# Already exists at line 2845 (called every 10 iterations)
# Just need to verify that Layer 30's overuse percentage decreases
```

Expected behavior:
- **Before layer balancing:** Layer 30 consistently has ~16% of overuse
- **After layer balancing:** Layer 30's overuse should decrease toward ~6.25% (uniform across 16 routing layers)

#### 3. Performance Verification

```python
# In _apply_layer_bias_to_costs():
t0 = time.perf_counter()
# ... apply bias ...
elapsed = time.perf_counter() - t0

if elapsed > 1.0:
    logger.warning(f"[LAYER-BIAS-APPLY] SLOW: {elapsed:.3f}s (expected <0.5s)")
```

---

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Memory overflow (52MB array) | LOW | MEDIUM | Check available memory before allocation; gracefully degrade if insufficient |
| Slow per-iteration overhead (>1s) | LOW | HIGH | Verify vectorized operations; add timeout guard; can disable via flag |
| Incorrect layer mapping | LOW | HIGH | Unit test with known graph; verify via_edges + horizontal_edges = all_edges |
| GPU sync issues | MEDIUM | MEDIUM | Use existing GPU↔CPU transfer pattern from via pooling |
| No convergence improvement | MEDIUM | LOW | Monitor first 50 iterations; can disable if no benefit |

---

### Fallback Plan

If Option A is too slow (>1s per iteration):

**Fallback 1:** Apply bias only to **routing layers with significant bias**

```python
# Instead of processing all layers:
for z in routing_layers:
    if abs(self.layer_bias[z]) > 0.05:  # Only significant biases
        layer_mask = (self._horizontal_edge_layers == z)
        total_cost_cpu[layer_mask] *= (1.0 + self.layer_bias[z])
```

This reduces work by ~80% (most layers have negligible bias).

**Fallback 2:** Apply bias only every N iterations

```python
# In main routing loop:
if it % 5 == 0:  # Every 5 iterations
    self._apply_layer_bias_to_costs()
```

This reduces overhead to ~0.06s per iteration (0.3s ÷ 5).

**Fallback 3:** Disable layer balancing via config flag

```python
if getattr(self.config, 'enable_layer_balancing', True):
    self._apply_layer_bias_to_costs()
```

---

## Implementation Plan

### Phase 1: Core Implementation (Est. 30 minutes)

1. ✅ Add `_horizontal_edge_layers` instance variable
2. ✅ Implement `_build_horizontal_edge_layers()` method
3. ✅ Call during initialization (after `_identify_via_edges()`)
4. ✅ Implement `_apply_layer_bias_to_costs()` method
5. ✅ Call during iteration loop (after `update_costs()`)

### Phase 2: Verification (Est. 15 minutes)

6. ✅ Add debug logging for first 3 iterations
7. ✅ Add performance guard (warn if >1s)
8. ✅ Test on small board (verify correctness)
9. ✅ Test on large board (verify performance)

### Phase 3: Tuning (Est. 15 minutes)

10. ✅ Monitor layer overuse convergence
11. ✅ Adjust alpha/clip values if needed
12. ✅ Add config flag for enable/disable

**Total estimated time:** ~60 minutes

---

## Expected Results

### Before Layer Balancing
```
[LAYER-CONGESTION] Horizontal overuse by layer:
  Layer 30: 42500.0 (16.5%)  ← HOTSPOT
  Layer 28: 18200.0 (7.1%)
  Layer 29: 15800.0 (6.1%)
  ... other layers ...
```

### After Layer Balancing (Iteration 20+)
```
[LAYER-CONGESTION] Horizontal overuse by layer:
  Layer 30: 16100.0 (6.3%)  ← BALANCED!
  Layer 28: 15800.0 (6.2%)
  Layer 29: 15900.0 (6.2%)
  Layer 27: 16200.0 (6.3%)
  ... uniform distribution ...
```

### Key Metrics

- **Layer 30 overuse reduction:** 16.5% → ~6.3% (target: 1/16 = 6.25%)
- **Convergence speed:** Expected 10-20% fewer iterations to convergence
- **Total routing time:** Similar or slightly faster (fewer iterations compensates for overhead)
- **Memory usage:** +52MB (0.5% increase)
- **Per-iteration overhead:** +0.3s (1% increase)

---

## Code Locations Reference

| Component | File | Line | Description |
|-----------|------|------|-------------|
| Layer bias initialization | unified_pathfinder.py | 1938 | `self.layer_bias = np.zeros(Nz)` |
| Via edges identification | unified_pathfinder.py | 3920 | `_identify_via_edges()` - pattern to follow |
| Via edge metadata | unified_pathfinder.py | 3943 | `_build_via_edge_metadata()` - vectorized approach |
| Cost update | unified_pathfinder.py | 2741 | `self.accounting.update_costs()` |
| Via pooling penalties | unified_pathfinder.py | 2750 | `_apply_via_pooling_penalties()` - similar structure |
| Layer bias EWMA update | unified_pathfinder.py | 2843-2872 | Already working, computes `self.layer_bias` |
| Per-layer congestion logging | unified_pathfinder.py | 3748 | `_log_per_layer_congestion()` - returns `overuse_horiz` |

---

## Alternative: Optimized Vectorization (Advanced)

If basic vectorization is still slow, can optimize further:

```python
def _apply_layer_bias_to_costs_optimized(self):
    """Ultra-fast version using advanced NumPy indexing"""
    if self._horizontal_edge_layers is None or np.allclose(self.layer_bias, 0.0, atol=0.001):
        return

    import time
    t0 = time.perf_counter()

    # Get costs
    if self.accounting.use_gpu:
        total_cost = self.accounting.total_cost.get()
    else:
        total_cost = self.accounting.total_cost

    # Pre-compute layer bias lookup table (length = Nz)
    bias_lut = 1.0 + self.layer_bias  # shape: [Nz]

    # Vectorized lookup: for each edge, get bias from its layer
    # Only process horizontal edges (layer >= 0)
    horiz_mask = self._horizontal_edge_layers >= 0
    horiz_layers = self._horizontal_edge_layers[horiz_mask]

    # Apply bias using fancy indexing
    total_cost[horiz_mask] *= bias_lut[horiz_layers]

    # Copy back to GPU
    if self.accounting.use_gpu:
        self.accounting.total_cost[:] = cp.asarray(total_cost)

    elapsed = time.perf_counter() - t0
    affected = np.sum(horiz_mask)
    logger.debug(f"[LAYER-BIAS-APPLY] {affected:,} edges in {elapsed*1000:.1f}ms")
```

This version:
- **Eliminates loop over layers** (processes all edges in one shot)
- **Uses fancy indexing** (NumPy optimized C code)
- **Expected time:** ~100-150ms (2-3× faster than basic version)

---

## Conclusion

**RECOMMENDED APPROACH:** Option A (Pre-build edge-to-layer mapping)

**KEY ADVANTAGES:**
- ✅ Proven pattern (mirrors `_via_edges` which already works)
- ✅ Negligible overhead (~0.3s per iteration = 1%)
- ✅ Easy to implement (~60 minutes)
- ✅ Easy to debug and verify
- ✅ Low risk of breaking existing functionality

**NEXT STEPS:**
1. Implement `_build_horizontal_edge_layers()` method
2. Implement `_apply_layer_bias_to_costs()` method
3. Add initialization call (line ~1993)
4. Add cost application call (line ~2751)
5. Test on small board first
6. Verify layer balancing effect over 50 iterations
7. Deploy to production

**DECISION POINT:**
After implementing Option A, if overhead is unacceptable (>1s), fall back to:
- Apply only to layers with |bias| > 0.05 (80% speedup)
- Apply every 5 iterations instead of every iteration (5× speedup)
- Disable via config flag

If overhead is acceptable, consider future optimization to Option C (GPU kernel) for zero overhead.
