# Layer Balancing - Step-by-Step Implementation Guide

## Overview

This guide provides exact code snippets and locations for implementing fast layer balancing.

**Goal:** Balance horizontal routing load across layers to eliminate Layer 30 hotspot (currently 16.5% of overuse).

**Approach:** Pre-build edge→layer mapping during initialization, apply bias using vectorized NumPy operations.

**Expected overhead:** ~0.2-0.4 seconds per iteration (0.5-1% of iteration time).

---

## Implementation Steps

### Step 1: Add Instance Variable

**Location:** `unified_pathfinder.py`, line ~1938 (in `__init__` method, after layer_bias initialization)

**Current code:**
```python
# Layer balancing (EWMA of per-layer horizontal overuse)
self.layer_bias = np.zeros(Nz, dtype=np.float32)  # Index by z (0..Nz-1)
logger.info(f"[LAYER-BALANCE] Initialized for {Nz} layers")
```

**Add after this:**
```python
# Layer balancing: edge-to-layer mapping for fast bias application
self._horizontal_edge_layers = None  # Will be built during initialize_graph()
```

---

### Step 2: Create Edge-to-Layer Mapping Method

**Location:** `unified_pathfinder.py`, add new method after `_identify_via_edges()` (after line ~3941)

**Insert this complete method:**
```python
def _build_horizontal_edge_layers(self):
    """
    Build edge-to-layer mapping for horizontal edges.
    This enables fast vectorized layer bias application.

    Similar to _identify_via_edges() but stores layer z for horizontal edges.
    Via edges are marked with -1.

    Memory: ~52MB for 52M edges (int8 array)
    Time: ~2-3 seconds (one-time initialization cost)
    """
    import time
    t0 = time.perf_counter()

    # Get CSR graph structure
    indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
    indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices

    num_edges = int(indptr[-1])

    # Initialize to -1 (marker for via edges)
    # Use int8 to save memory (can represent -128 to 127, enough for any realistic layer count)
    self._horizontal_edge_layers = np.full(num_edges, -1, dtype=np.int8)

    # Use arithmetic instead of idx_to_coord for speed (same as _identify_via_edges)
    plane_size = self.lattice.x_steps * self.lattice.y_steps

    horizontal_count = 0
    via_count = 0

    # Loop over all nodes and their outgoing edges
    for u in range(len(indptr) - 1):
        uz = u // plane_size  # Source layer

        for ei in range(int(indptr[u]), int(indptr[u+1])):
            v = int(indices[ei])
            vz = v // plane_size  # Destination layer

            if uz == vz:
                # Horizontal edge: store layer z
                self._horizontal_edge_layers[ei] = uz
                horizontal_count += 1
            else:
                # Via edge: keep as -1
                via_count += 1

    elapsed = time.perf_counter() - t0
    memory_mb = self._horizontal_edge_layers.nbytes / (1024 * 1024)

    logger.info(f"[LAYER-BALANCE] Built edge-to-layer mapping in {elapsed:.3f}s")
    logger.info(f"[LAYER-BALANCE]   Horizontal edges: {horizontal_count:,}")
    logger.info(f"[LAYER-BALANCE]   Via edges: {via_count:,}")
    logger.info(f"[LAYER-BALANCE]   Memory: {memory_mb:.1f} MB")

    # Sanity check
    assert horizontal_count + via_count == num_edges, \
        f"Edge count mismatch: {horizontal_count} + {via_count} != {num_edges}"
```

---

### Step 3: Create Bias Application Method

**Location:** `unified_pathfinder.py`, add new method after `_build_horizontal_edge_layers()`

**Insert this complete method:**
```python
def _apply_layer_bias_to_costs(self):
    """
    Apply layer balancing bias to horizontal edge costs.

    Called once per iteration, after update_costs(), before routing.

    Hot layers (positive bias) → higher cost → routes avoid them
    Cool layers (negative bias) → lower cost → routes prefer them

    Time: ~0.2-0.4 seconds (vectorized NumPy operations)
    """
    # Skip if not initialized
    if self._horizontal_edge_layers is None:
        return

    # Skip if all biases are negligible
    if np.allclose(self.layer_bias, 0.0, atol=0.001):
        return

    import time
    t0 = time.perf_counter()

    # Get total_cost array (may be on GPU)
    if self.accounting.use_gpu:
        total_cost = self.accounting.total_cost.get()  # Transfer GPU → CPU
    else:
        total_cost = self.accounting.total_cost

    # Build bias multiplier for each edge
    # For horizontal edges on layer z: multiplier = 1.0 + layer_bias[z]
    # For via edges: multiplier = 1.0 (no change)

    # Method 1: Loop over routing layers (clear and debuggable)
    routing_layers = range(1, self._Nz - 1)  # Skip F.Cu (0) and B.Cu (Nz-1)

    edges_affected = 0

    for z in routing_layers:
        bias = self.layer_bias[z]

        # Skip negligible biases (performance optimization)
        if abs(bias) < 0.001:
            continue

        # Find all horizontal edges on this layer
        layer_mask = (self._horizontal_edge_layers == z)

        # Apply bias: new_cost = old_cost × (1.0 + bias)
        # Positive bias → cost increases (penalize hot layer)
        # Negative bias → cost decreases (encourage cool layer)
        bias_factor = 1.0 + bias
        total_cost[layer_mask] *= bias_factor

        edges_affected += np.sum(layer_mask)

    # Copy back to GPU if needed
    if self.accounting.use_gpu:
        self.accounting.total_cost[:] = cp.asarray(total_cost)

    elapsed = time.perf_counter() - t0

    # Log performance
    if edges_affected > 0:
        logger.info(f"[LAYER-BIAS-APPLY] Applied bias to {edges_affected:,} edges in {elapsed*1000:.1f}ms")

        # Performance warning
        if elapsed > 1.0:
            logger.warning(f"[LAYER-BIAS-APPLY] SLOW: {elapsed:.3f}s (expected <0.5s)")
            logger.warning(f"[LAYER-BIAS-APPLY] Consider disabling layer balancing or using optimized version")
```

**Alternative: Optimized Version (use if basic version is too slow):**
```python
def _apply_layer_bias_to_costs_optimized(self):
    """
    Ultra-fast version using advanced NumPy indexing.
    Expected time: ~0.1-0.2 seconds (2-3× faster than basic version)
    """
    if self._horizontal_edge_layers is None or np.allclose(self.layer_bias, 0.0, atol=0.001):
        return

    import time
    t0 = time.perf_counter()

    # Get costs
    if self.accounting.use_gpu:
        total_cost = self.accounting.total_cost.get()
    else:
        total_cost = self.accounting.total_cost

    # Create bias lookup table: bias_lut[z] = 1.0 + layer_bias[z]
    bias_lut = 1.0 + self.layer_bias  # shape: [Nz]

    # Find horizontal edges (layer >= 0)
    horiz_mask = self._horizontal_edge_layers >= 0
    horiz_layers = self._horizontal_edge_layers[horiz_mask]

    # Vectorized lookup and multiply (fancy indexing)
    total_cost[horiz_mask] *= bias_lut[horiz_layers]

    # Copy back to GPU
    if self.accounting.use_gpu:
        self.accounting.total_cost[:] = cp.asarray(total_cost)

    elapsed = time.perf_counter() - t0
    logger.debug(f"[LAYER-BIAS-APPLY] {np.sum(horiz_mask):,} edges in {elapsed*1000:.1f}ms")
```

---

### Step 4: Call Mapping Builder During Initialization

**Location:** `unified_pathfinder.py`, line ~1990 (in `initialize_graph()` method)

**Current code:**
```python
# Identify via edges for via-specific accounting
self._identify_via_edges()

# Build via edge metadata for vectorized penalty application
self._build_via_edge_metadata()

self._map_pads(board)
```

**Change to:**
```python
# Identify via edges for via-specific accounting
self._identify_via_edges()

# Build via edge metadata for vectorized penalty application
self._build_via_edge_metadata()

# Build horizontal edge layers for layer balancing
self._build_horizontal_edge_layers()

self._map_pads(board)
```

---

### Step 5: Call Bias Application During Iteration

**Location:** `unified_pathfinder.py`, line ~2750 (in main routing loop, after `update_costs()`)

**Current code:**
```python
# STEP 2: Update costs (with history weight and via annealing)
self.accounting.update_costs(
    self.graph.base_costs, pres_fac, hist_gain,
    via_cost_multiplier=via_cost_mult,
    base_cost_weight=cfg.base_cost_weight
)

# STEP 2.5: Apply via column/segment pooling penalties
# NOTE: Layer balancing disabled - sequential loop over 52M edges too slow
# TODO: Implement layer balancing in GPU kernel or vectorize
self._apply_via_pooling_penalties(pres_fac)

# STEP 3: Route (hotset incremental after iter 1)
```

**Change to:**
```python
# STEP 2: Update costs (with history weight and via annealing)
self.accounting.update_costs(
    self.graph.base_costs, pres_fac, hist_gain,
    via_cost_multiplier=via_cost_mult,
    base_cost_weight=cfg.base_cost_weight
)

# STEP 2.5: Apply via column/segment pooling penalties
self._apply_via_pooling_penalties(pres_fac)

# STEP 2.6: Apply layer balancing bias
self._apply_layer_bias_to_costs()

# STEP 3: Route (hotset incremental after iter 1)
```

**IMPORTANT:** Remove the old comment about layer balancing being disabled:
```python
# DELETE THIS COMMENT:
# NOTE: Layer balancing disabled - sequential loop over 52M edges too slow
# TODO: Implement layer balancing in GPU kernel or vectorize
```

---

## Verification and Testing

### Step 6: Add Debug Logging (First 3 Iterations)

**Location:** Inside `_apply_layer_bias_to_costs()`, after bias application

**Add this debug block:**
```python
# Debug logging for first 3 iterations
if self.iteration <= 3:
    logger.info(f"[LAYER-BIAS-DEBUG] Iteration {self.iteration} - Bias application details:")

    for z in routing_layers:
        bias = self.layer_bias[z]
        if abs(bias) > 0.05:  # Only log significant biases
            layer_mask = (self._horizontal_edge_layers == z)
            num_edges = np.sum(layer_mask)

            if num_edges > 0:
                # Sample a few edges to verify correct application
                sample_costs = total_cost[layer_mask][:10]
                logger.info(f"    Layer {z:2d}: bias={bias:+.3f}, {num_edges:,} edges, "
                           f"sample_costs={sample_costs.tolist()}")
```

---

### Step 7: Verify Layer Balancing Effect

**Monitor these logs over iterations:**

```
# Before layer balancing (iteration 1-5):
[LAYER-CONGESTION] Horizontal overuse by layer:
  Layer 30: 42500.0 (16.5%)  ← HOTSPOT
  Layer 28: 18200.0 (7.1%)

[LAYER-BIAS] Hot layers: L30:+0.103, L28:+0.013, L29:-0.008

# After layer balancing kicks in (iteration 10-20):
[LAYER-CONGESTION] Horizontal overuse by layer:
  Layer 30: 24800.0 (9.8%)   ← IMPROVING
  Layer 28: 16500.0 (6.5%)

[LAYER-BIAS] Hot layers: L30:+0.263, L28:+0.045, L27:-0.031

# Convergence (iteration 30-50):
[LAYER-CONGESTION] Horizontal overuse by layer:
  Layer 30: 15900.0 (6.3%)   ← BALANCED!
  Layer 28: 15700.0 (6.2%)
  Layer 29: 16100.0 (6.4%)

[LAYER-BIAS] Hot layers: L30:+0.015, L27:-0.018, L28:+0.008
```

**Success criteria:**
- Layer 30 overuse drops from ~16% to ~6% (uniform distribution)
- Layer bias values stabilize (oscillate around 0)
- Total overuse converges faster (fewer iterations to zero)

---

### Step 8: Performance Validation

**Check these metrics:**

1. **Initialization time increase:**
   - Look for `[LAYER-BALANCE] Built edge-to-layer mapping in X.XXXs`
   - Expected: 2-3 seconds (acceptable one-time cost)
   - If > 5 seconds: Warning, but still acceptable

2. **Per-iteration time increase:**
   - Look for `[LAYER-BIAS-APPLY] Applied bias to X edges in X.Xms`
   - Expected: 200-400ms
   - If > 1000ms: Switch to optimized version

3. **Total iteration time:**
   - Before: ~30-60 seconds per iteration
   - After: ~30-61 seconds per iteration (< 2% increase)
   - Acceptable if convergence improves

4. **Memory usage:**
   - Look for `[LAYER-BALANCE] Memory: X.X MB`
   - Expected: ~52MB
   - Should be < 1% of total system memory

---

## Troubleshooting

### Issue 1: Slow Bias Application (>1 second)

**Diagnosis:**
```
[LAYER-BIAS-APPLY] SLOW: 2.345s (expected <0.5s)
```

**Solution:**
Replace `_apply_layer_bias_to_costs()` with `_apply_layer_bias_to_costs_optimized()` version.

---

### Issue 2: No Convergence Improvement

**Diagnosis:**
After 50 iterations, Layer 30 still has ~15% overuse (no improvement).

**Possible causes:**
1. Bias alpha too small (not aggressive enough)
2. Bias clipping too tight (prevents correction)
3. Other bottlenecks dominating (portal issues, via congestion)

**Solution:**
Tune parameters in the EWMA update (line 2861):
```python
# Try more aggressive alpha
alpha = 0.30  # Was 0.20

# Try wider clip range
self.layer_bias = np.clip(self.layer_bias, -0.60, +0.60)  # Was [-0.40, +0.40]
```

---

### Issue 3: Layer Bias Oscillation

**Diagnosis:**
Layer bias values oscillate wildly (e.g., L30: +0.40 → -0.40 → +0.40).

**Solution:**
Reduce alpha for smoother convergence:
```python
alpha = 0.10  # Was 0.20 (slower but more stable)
```

---

### Issue 4: Memory Allocation Failure

**Diagnosis:**
```
MemoryError: Unable to allocate 52.4 MiB for array
```

**Solution:**
Gracefully degrade (disable layer balancing):
```python
def _build_horizontal_edge_layers(self):
    try:
        # ... allocation code ...
    except MemoryError:
        logger.warning("[LAYER-BALANCE] Insufficient memory for edge-to-layer mapping")
        logger.warning("[LAYER-BALANCE] Layer balancing DISABLED")
        self._horizontal_edge_layers = None
        return
```

---

## Configuration Options (Future Enhancement)

Add config flag to enable/disable:

```python
# In PathFinderConfig:
enable_layer_balancing: bool = True
layer_bias_alpha: float = 0.20
layer_bias_clip: float = 0.40
```

```python
# In _apply_layer_bias_to_costs():
if not getattr(self.config, 'enable_layer_balancing', True):
    return
```

---

## Expected Timeline

| Phase | Task | Time | Total |
|-------|------|------|-------|
| 1 | Add instance variable | 2 min | 2 min |
| 2 | Create `_build_horizontal_edge_layers()` | 10 min | 12 min |
| 3 | Create `_apply_layer_bias_to_costs()` | 10 min | 22 min |
| 4 | Add initialization call | 2 min | 24 min |
| 5 | Add iteration call | 2 min | 26 min |
| 6 | Add debug logging | 5 min | 31 min |
| 7 | Test on small board | 10 min | 41 min |
| 8 | Test on large board | 15 min | 56 min |
| 9 | Tune parameters if needed | 10 min | 66 min |

**Total: ~1 hour**

---

## Testing Checklist

- [ ] Code compiles without errors
- [ ] `_build_horizontal_edge_layers()` called during initialization
- [ ] Log shows: `[LAYER-BALANCE] Built edge-to-layer mapping in X.XXXs`
- [ ] Log shows: `[LAYER-BALANCE] Horizontal edges: X,XXX,XXX`
- [ ] Log shows: `[LAYER-BALANCE] Memory: X.X MB`
- [ ] `_apply_layer_bias_to_costs()` called each iteration
- [ ] Log shows: `[LAYER-BIAS-APPLY] Applied bias to X,XXX,XXX edges in XXXms`
- [ ] Bias application time < 500ms
- [ ] Debug logging works for first 3 iterations
- [ ] Layer 30 overuse percentage decreases over iterations
- [ ] Layer bias values stabilize (not oscillating)
- [ ] Total iteration time increase < 2%
- [ ] No memory errors
- [ ] Convergence improves (fewer iterations to zero overuse)

---

## Success Metrics

**Primary metric:** Layer 30 overuse percentage
- **Before:** ~16.5%
- **Target:** ~6.25% (uniform across 16 routing layers)
- **Acceptable:** <10% (2× improvement)

**Secondary metrics:**
- Iterations to convergence: 10-20% reduction expected
- Total routing time: Similar or faster (fewer iterations)
- Per-iteration overhead: <2% increase

---

## Rollback Plan

If layer balancing causes issues:

1. **Immediate rollback:** Comment out the call to `_apply_layer_bias_to_costs()` (line ~2751)
2. **Partial rollback:** Only apply bias every N iterations instead of every iteration
3. **Full rollback:** Remove all changes (git revert)

The implementation is isolated and can be cleanly disabled without affecting other systems.

---

## Future Optimizations

Once Option A is working and validated:

1. **GPU kernel integration (Option C):**
   - Move bias application into CUDA kernel
   - Zero overhead (absorbed into edge relaxation)
   - Requires CUDA expertise

2. **Adaptive alpha:**
   - Increase alpha when far from convergence
   - Decrease alpha when close to convergence
   - Faster initial response, more stable final state

3. **Per-layer capacity modeling:**
   - Track actual capacity of each layer (considering width, spacing rules)
   - Adjust target distribution based on real capacity

4. **Via-aware layer balancing:**
   - Consider via landing penalties when biasing layers
   - Coordinate with via pooling system
