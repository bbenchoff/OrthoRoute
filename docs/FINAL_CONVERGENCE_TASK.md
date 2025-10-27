# PathFinder Final Convergence Task

**Date:** 2025-10-26
**Priority:** HIGH
**Estimated Time:** 2-3 hours
**Goal:** Achieve convergence to 0 overuse on 32-layer test board

---

## Context - What's Been Done

### ✅ Via Model Fixed (100% Complete)
The original diagnosis was correct - via model issues were causing convergence problems. All fixes implemented:

1. **Full blind/buried vias enabled** - 870 via pairs (was 58 adjacent-only)
   - File: `unified_pathfinder.py:951-980`
   - Allows any routing layer to connect to any other (L1↔L30, L5↔L25, etc.)

2. **Via keepouts disabled** - No intermediate layer blocking
   - File: `config.py:117`
   - `enable_buried_via_keepouts = False`

3. **Hotset size optimized** - 64-180 adaptive (was 300+)
   - File: `unified_pathfinder.py:3621-3628`

4. **Parameters tuned** - Sweet spot found
   - `pres_fac_mult = 1.25`
   - `hist_gain = 1.5`
   - `via_cost = 0.7`

5. **Performance optimized** - 31% faster
   - Screenshots every 10 iterations (was every iteration)
   - GPU transfers cached

### 📊 Current Results:
- **Best convergence:** 1,911 overuse @ iteration 8
- **Baseline improvement:** 80% better (was diverging to 49K+)
- **Performance:** 31 seconds per iteration, GPU at 25ms/net ⚡
- **Status:** Oscillates 2-6K (stable, not diverging)

---

## ⚠️ The Remaining Problem

**Via conflicts = 0.1% of overuse** (via model is fixed!)
**Horizontal overuse = 99.9%** with severe layer imbalance:

```
Layer 30: 1,570 overuse (16.5% of total!) 🔥 HOTSPOT
Layer 10: 1,195 overuse (12.6%)
Layer  6: 1,143 overuse (12.0%)
Three layers = 41% of all overuse
```

**Issue:** PathFinder can't rebalance layers with just pres_fac and history alone. Needs explicit per-layer cost bias.

**Previous attempt:** Agent tried layer balancing but used sequential loop over 52M edges → hung the router or made convergence worse

**Solution:** Implement lightweight vectorized layer balancing (this task)

---

## Your Mission

Implement fast vectorized layer balancing to eliminate the Layer 30 hotspot and achieve final convergence below 1,000 overuse.

**Success Criteria:**
- Layer 30 share drops from 16.5% → 6-9% (uniform distribution)
- Best overuse improves from 1,911 → <1,000
- No performance regression (keep ~31s per iteration)
- All 512 nets continue routing successfully

---

## Implementation Plan

### Step 1: Modify EdgeAccountant.update_costs() to Accept Layer Bias

**Location:** `orthoroute/algorithms/manhattan/unified_pathfinder.py` around line 845

**Find:**
```python
def update_costs(self, base_costs, pres_fac: float, hist_weight: float = 1.0, add_jitter: bool = True, via_cost_multiplier: float = 1.0, base_cost_weight: float = 0.01):
```

**Replace with:**
```python
def update_costs(
    self,
    base_costs,
    pres_fac: float,
    hist_weight: float = 1.0,
    add_jitter: bool = True,
    via_cost_multiplier: float = 1.0,
    base_cost_weight: float = 0.01,
    *,
    edge_layer=None,          # np/cp array [E] with source layer per edge
    layer_bias_per_layer=None # np/cp array [L] with multiplicative bias
):
    """
    total = (base * via_mult * base_weight * layer_bias) + pres_fac*overuse + hist*history + jitter
    """
    xp = self.xp
    over = xp.maximum(0, self.present - self.capacity)

    # Vectorized per-edge layer bias (single gather operation)
    if (edge_layer is not None) and (layer_bias_per_layer is not None):
        if hasattr(base_costs, "device"):  # CuPy
            layer_bias = cp.asarray(layer_bias_per_layer) if not hasattr(layer_bias_per_layer, "get") else layer_bias_per_layer
            edge_layer_arr = cp.asarray(edge_layer) if not hasattr(edge_layer, "get") else edge_layer
        else:  # NumPy
            layer_bias = layer_bias_per_layer
            edge_layer_arr = edge_layer
        per_edge_bias = layer_bias[edge_layer_arr]
    else:
        per_edge_bias = 1.0

    adjusted_base = base_costs * via_cost_multiplier * base_cost_weight * per_edge_bias
    self.total_cost = adjusted_base + pres_fac * over + hist_weight * self.history

    if add_jitter:
        E = len(self.total_cost)
        jitter = xp.arange(E, dtype=xp.float32) % 9973
        self.total_cost += jitter * xp.float32(1e-6)
```

**Why this works:** One vectorized gather (`layer_bias[edge_layer]`) + one multiply. Zero loops, ~1ms overhead.

---

### Step 2: Add _compute_layer_bias() Helper Method

**Location:** Add after existing helper methods, around line 3600

**Add this complete function:**
```python
def _compute_layer_bias(self, accountant, graph, num_layers: int, alpha: float = 0.9, max_boost: float = 1.8):
    """
    Compute per-layer multiplicative bias (shape [L]) based on overuse distribution.
    Hot layers get bias > 1.0, cool layers stay at 1.0.

    Args:
        accountant: EdgeAccountant with present/capacity arrays
        graph: CSRGraph with edge_layer mapping
        num_layers: Number of layers
        alpha: EMA smoothing factor (0..1, higher = smoother)
        max_boost: Maximum bias multiplier (1.0 to max_boost)

    Returns:
        Layer bias array or None if layer mapping unavailable
    """
    xp = accountant.xp

    # Check if edge_layer mapping exists
    edge_layer = getattr(graph, "edge_layer_gpu", None) if accountant.use_gpu else getattr(graph, "edge_layer", None)
    if edge_layer is None:
        return None

    # Get overuse array
    usage = accountant.present.get() if accountant.use_gpu else accountant.present
    cap = accountant.capacity.get() if accountant.use_gpu else accountant.capacity
    over = np.maximum(0, usage - cap) if not accountant.use_gpu else cp.maximum(0, usage - cap)

    # Sum overuse per layer (ONE bincount - very fast)
    per_layer_over = (cp if accountant.use_gpu else np).bincount(
        edge_layer, weights=over, minlength=num_layers
    )

    # Normalize to create bias factors
    maxv = float(per_layer_over.max().get() if accountant.use_gpu else per_layer_over.max())
    if maxv <= 1e-9:
        raw_bias = (cp.ones(num_layers, dtype=cp.float32) if accountant.use_gpu
                    else np.ones(num_layers, dtype=np.float32))
    else:
        shortfall = per_layer_over / maxv
        raw_bias = 1.0 + 0.75 * shortfall  # Bias = 1.0 to 1.75

    # EMA smoothing to prevent oscillation
    if not hasattr(self, "_layer_bias_ema"):
        self._layer_bias_ema = raw_bias.astype(cp.float32 if accountant.use_gpu else np.float32)
    else:
        self._layer_bias_ema = (alpha * self._layer_bias_ema +
                                (1.0 - alpha) * raw_bias).astype(self._layer_bias_ema.dtype)

    # Clamp to prevent extreme penalties
    if accountant.use_gpu:
        self._layer_bias_ema = cp.clip(self._layer_bias_ema, 1.0, max_boost)
    else:
        self._layer_bias_ema = np.clip(self._layer_bias_ema, 1.0, max_boost)

    return self._layer_bias_ema
```

---

### Step 3: Wire Layer Bias into Cost Update

**Location:** In `_pathfinder_negotiation()` around line 2730-2740 (where `accounting.update_costs()` is called)

**Find the existing call:**
```python
self.accounting.update_costs(
    self.graph.base_costs, pres_fac, hist_gain,
    via_cost_multiplier=via_cost_mult,
    base_cost_weight=cfg.base_cost_weight
)
```

**Replace with:**
```python
# Compute layer bias once per iteration
layer_bias = self._compute_layer_bias(
    self.accounting, self.graph,
    num_layers=self.lattice.layers,
    alpha=0.9,  # High smoothing for stability
    max_boost=1.8  # Limit penalty to 1.8x
)

# Apply bias through existing cost update mechanism
self.accounting.update_costs(
    self.graph.base_costs, pres_fac, hist_gain,
    via_cost_multiplier=via_cost_mult,
    base_cost_weight=cfg.base_cost_weight,
    edge_layer=(self.graph.edge_layer_gpu if self.accounting.use_gpu else self.graph.edge_layer),
    layer_bias_per_layer=layer_bias
)
```

---

### Step 4: Add Damping Mechanisms

**Location:** Same negotiation loop

**4a) History Decay - Find line ~2709:**
```python
# FIND:
decay_factor=1.0  # or 0.98

# REPLACE WITH:
decay_factor=0.995  # Gentle decay prevents oscillation
```

**4b) Adaptive Hotset - Around line 3624:**
```python
# FIND:
adaptive_cap = min(self.config.hotset_cap, base_target)

# ADD BEFORE IT:
# Track improvement trend
if not hasattr(self, '_prev_overuse'):
    self._prev_overuse = float('inf')

delta = (self._prev_overuse - over_sum) / max(1, self._prev_overuse)
if delta < 0.02:  # <2% improvement - enlarge hotset
    base_target = min(240, int(base_target * 1.25))
elif delta > 0.08:  # >8% improvement - shrink hotset
    base_target = max(64, int(base_target * 0.80))

self._prev_overuse = over_sum
adaptive_cap = min(self.config.hotset_cap, base_target)
```

---

### Step 5: Verify Edge-Layer Mapping Exists

**Location:** Check `CSRGraph.finalize()` around line 750

**Verify this code exists:**
```python
# Build edge-to-layer mapping
plane_size = num_nodes // self.num_layers if hasattr(self, 'num_layers') else 0
if plane_size > 0:
    edge_sources = np.zeros(E, dtype=np.int32)
    for u in range(num_nodes):
        start, end = indptr[u], indptr[u+1]
        if end > start:
            edge_sources[start:end] = u

    self.edge_layer = (edge_sources // plane_size).astype(np.uint8)
    if self.use_gpu:
        self.edge_layer_gpu = cp.asarray(self.edge_layer)
```

**If missing:** The agent should have already added this. If not found, add it to `finalize()` method.

---

## Testing Instructions

**Run test:**
```bash
ORTHO_CPU_ONLY=1 timeout 1200 python main.py --test-manhattan 2>&1 | tee test_FINAL_CONVERGENCE.log
```

**Monitor progress:**
```bash
# Every 5 minutes, check:
tail -20 test_FINAL_CONVERGENCE.log | grep "ITER\|LAYER-BIAS"
```

**After completion, verify:**

1. **Layer balancing active:**
```bash
grep "\[LAYER-MAP\]" test_FINAL_CONVERGENCE.log  # Should show edge mapping built
grep "layer_bias" test_FINAL_CONVERGENCE.log | head -20  # Should show bias updates
```

2. **Layer 30 hotspot reduction:**
```bash
grep -A 20 "\[LAYER-CONGESTION\]" test_FINAL_CONVERGENCE.log | grep "Layer 30"
```

Expected: Layer 30 share decreases from ~16% toward ~6-9% over iterations

3. **Convergence improvement:**
```bash
grep -E "\[ITER [0-9]+\].*overuse=" test_FINAL_CONVERGENCE.log
```

Expected: Best overuse < 1,500 (improved from 1,911), final < 3,000 (improved from 6K)

---

## Expected Results

**Iteration 1-5:**
```
Layer 30: ~1,500 overuse (16%)  layer_bias[30]=1.10
Best: ~5,000 overuse total
```

**Iteration 10-15:**
```
Layer 30: ~900 overuse (10%)  layer_bias[30]=1.18
Best: ~1,500 overuse total (IMPROVING!)
```

**Iteration 20-30:**
```
Layer 30: ~600 overuse (7%)  layer_bias[30]=1.12 (stabilizing)
Best: ~800 overuse total (TARGET APPROACHING!)
```

**Iteration 35-40:**
```
Layer 30: ~500 overuse (6.5%)  layer_bias[30]=1.05
Final: <1,000 overuse ✅ CONVERGED!
```

---

## Troubleshooting

**If layer balancing doesn't activate:**
- Check `[LAYER-MAP] Built edge→layer mapping` appears in log
- Verify `edge_layer` array exists in CSRGraph
- Check layer_bias is being passed to update_costs()

**If performance regresses (>40s per iteration):**
- Layer bias computation should be <1ms (one bincount)
- If slow, check for accidental loops over 52M edges
- Ensure using vectorized NumPy/CuPy operations only

**If convergence gets worse:**
- Reduce max_boost from 1.8 → 1.5
- Reduce bias strength from 0.75 → 0.5
- Increase alpha from 0.9 → 0.95 (more smoothing)

**If Layer 30 stays hot:**
- Increase max_boost from 1.8 → 2.0
- Increase bias strength from 0.75 → 1.0
- Check if hotspot is geometric (single corridor bottleneck)

---

## Key Files

**Main implementation:**
- `orthoroute/algorithms/manhattan/unified_pathfinder.py`

**Config (if tuning needed):**
- `orthoroute/algorithms/manhattan/pathfinder/config.py`

**Reference (working baseline from this morning):**
- `test_CONVERGENCE_FINAL.log` - 26 iterations, best 1.9K, oscillating 2-6K

---

## Critical Notes

1. **Do NOT use sequential loops over 52M edges** - Only vectorized NumPy/CuPy operations
2. **Layer bias multiplies BASE costs only** - Not pres_fac or history terms
3. **EMA smoothing is critical** - Prevents oscillation from over-correction
4. **Test after each step** - Verify syntax before running full test

---

## Final Validation

**Success = All of these:**
- ✅ Layer 30 share < 10% by iteration 20
- ✅ Best overuse < 1,500 (better than 1,911 baseline)
- ✅ Final overuse < 3,000 (better than 6K baseline)
- ✅ Iteration time 29-35s (no regression from 31s)
- ✅ All 512 nets route successfully

**Bonus (stretch goal):**
- Best overuse < 1,000
- Final overuse < 1,500
- Layer distribution uniform (all layers 5-8% range)

---

## Context Preservation

**What works (DON'T BREAK):**
- Full blind/buried vias (870 pairs) - unified_pathfinder.py:951-980
- Via keepouts disabled - config.py:117
- GPU performance (25ms/net)
- Hotset sizing and shuffling
- Screenshot/logging optimizations

**What to implement:**
- Vectorized layer balancing (this task)
- History decay (already set to 0.995)
- Adaptive hotset trend-based sizing

---

## End Goal

**32-layer test board (512 nets) should:**
- Converge below 1,000 overuse in 25-35 iterations
- Complete in 12-18 minutes
- Route all nets successfully
- Maintain 25-31s per iteration performance

**This is the final step** to complete PathFinder convergence debugging. The via model work is done, this is pure algorithmic tuning.

Good luck!
