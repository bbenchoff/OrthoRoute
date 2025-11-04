# PathFinder Parameter Tuning Guide

This guide explains all the tunable parameters in OrthoRoute's PathFinder algorithm and how to adjust them for different board types and convergence issues.

## Quick Reference Table

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `pres_fac_mult` | 1.10-1.35 | 1.05-2.0 | How fast overused edges become expensive |
| `pres_fac_max` | 6.0-16.0 | 4.0-64.0 | Maximum pressure on overused edges |
| `hist_gain` | 0.15-0.20 | 0.05-0.35 | How fast history accumulates |
| `hist_cost_weight` | 10.0-16.0 | 5.0-25.0 | Relative importance of history vs present |
| `via_cost` | 0.5-0.7 | 0.3-2.0 | Penalty for using vias |
| `max_iterations` | 40-100 | 20-200 | When to give up |
| `BARREL_PHASE_1_ITERS` | 10-50 | 0-100 | When barrel penalties stop |

---

## Core PathFinder Parameters

### 1. Present Factor Multiplier (`pres_fac_mult`)

**What it does:** Controls how aggressively PathFinder escalates costs on overused edges each iteration.

**Formula:**
```
pres_fac(iter) = pres_fac_init × (pres_fac_mult)^(iter-1)

Example with mult=1.35:
Iter 1: pres_fac = 1.0
Iter 5: pres_fac = 3.3
Iter 10: pres_fac = 18.9
Iter 15: pres_fac = 107.9  (hits max)
```

**How to tune:**

| Value | Board Type | Effect |
|-------|-----------|---------|
| 1.05-1.10 | Sparse (ρ < 0.5) | Gentle, slow convergence |
| 1.15-1.25 | Normal (ρ 0.5-0.8) | Balanced |
| **1.30-1.40** | **Tight (ρ 0.8-1.0)** | **Aggressive, forces rerouting** |
| 1.50-2.00 | Over-congested (ρ > 1.0) | Very aggressive (may oscillate) |

**When to adjust:**
- **Diverging overuse?** Increase (try 1.35-1.50)
- **Oscillating?** Decrease (try 1.10-1.15)
- **Slow convergence?** Increase slightly

**Example from today:**
- 8,192-net board with ρ=0.915 was diverging (overuse 2.4M → 4.4M)
- **Fix:** Changed from 1.10 → 1.35
- **Result:** Should converge in ~50 iterations instead of diverging

---

### 2. Present Factor Maximum (`pres_fac_max`)

**What it does:** Caps how high `pres_fac` can grow, preventing infinite costs.

**Default:** 6.0-16.0 depending on board density

**How to tune:**

| Value | Use Case |
|-------|----------|
| 4.0-6.0 | Sparse boards - don't need extreme pressure |
| 8.0-16.0 | Tight boards - need strong rerouting signals |
| 32.0-64.0 | Desperate - board barely routable |

**Trade-off:**
- **Too low:** PathFinder gives up early, accepts overuse
- **Too high:** Routes become extreme detours, slow convergence

**Rule of thumb:** Set to `pres_fac_mult^N` where N = expected iterations to convergence

---

### 3. History Gain (`hist_gain`)

**What it does:** Controls how fast PathFinder "remembers" overused edges across iterations.

**Formula:**
```
history[edge] += hist_gain × overuse[edge]  (each iteration)
total_cost[edge] = present[edge] + history[edge]
```

**The balance problem:**
- **Too HIGH** (> 0.25): History dominates, fights present costs → oscillation
- **Too LOW** (< 0.10): Present dominates, no learning → repeated mistakes
- **Just right** (0.12-0.18): Balanced

**How to tune:**

| Value | Effect | When to Use |
|-------|--------|-------------|
| 0.05-0.10 | Minimal memory | Very responsive, but forgets quickly |
| **0.12-0.18** | **Balanced** | **Most boards** |
| 0.20-0.25 | Strong memory | Dense boards, but risks oscillation |
| 0.30-0.35 | Very strong | Rarely needed, often causes problems |

**Diagnostic:** Check `hist/pres ratio` in logs:
```
hist/pres ratio: 0.994  ← Good (history ~ present)
hist/pres ratio: 0.05   ← BAD: hist_gain too low!
hist/pres ratio: 10.0   ← BAD: hist_gain too high!
```

**Target ratio:** 0.5 - 2.0

---

### 4. History Cost Weight (`hist_cost_weight`)

**What it does:** Multiplies history before adding to total cost.

**Formula:**
```
total_cost = present + (hist_cost_weight × history)
```

**Board-adaptive formula:**
```python
hist_cost_weight = 7.0                    # Base
                 + (1.0 per 4 layers)     # Layer bonus
                 + (6.0 × (ρ - 0.7))     # Congestion bonus
```

**How to tune:**
- **Higher weight** (15-25): Emphasizes history, good for dense boards
- **Lower weight** (5-10): Emphasizes present, good for sparse boards
- **Default (10-15)**: Usually fine, auto-tuned based on ρ

**Rule:** If hist/pres ratio is off, adjust `hist_gain` first, not weight.

---

## Convergence Tuning

### 5. Maximum Iterations (`max_iterations`)

**What it does:** Stops routing after N iterations even if not converged.

**Board-adaptive defaults:**
```
ρ < 0.5:  40 iterations  (converges fast)
ρ 0.5-0.8: 60 iterations  (moderate)
ρ 0.8-1.0: 100 iterations (tight, needs time)
ρ > 1.0:  150+ iterations (may never converge)
```

**How to tune:**
- Check when test board converges (e.g., 75 iterations)
- Set max to 1.5× that value (safety margin)
- For production: Lower to save time if partial routing acceptable

---

### 6. Stagnation Patience (`stagnation_patience`)

**What it does:** Stops early if overuse doesn't improve for N iterations.

**Default:** 5 iterations

**Trade-off:**
- **Too low** (2-3): May quit before convergence
- **Too high** (10+): Wastes time on hopeless boards

**Example:**
```
If overuse is: 1000 → 980 → 975 → 977 → 978 → 979
After 5 iterations without significant improvement, stop.
```

---

## Via Parameters

### 7. Via Cost (`via_cost`)

**What it does:** Base penalty for using a via (layer change).

**Default:** 0.5-0.7 (half a grid step)

**How to tune:**

| Value | Effect |
|-------|--------|
| 0.3-0.4 | Encourage vias (more layer changes) |
| 0.5-0.7 | Balanced |
| 0.8-1.5 | Discourage vias (prefer planar routing) |
| 2.0+ | Strongly avoid vias (may cause congestion) |

**When to adjust:**
- **Too many vias?** Increase to 0.8-1.0
- **Routes not using layers?** Decrease to 0.4-0.5
- **Via barrel conflicts high?** Increase (makes vias less attractive)

---

## Barrel Conflict Parameters

### 8. Barrel Phase 1 Duration (`BARREL_PHASE_1_ITERS`)

**What it does:** How many iterations to apply barrel conflict penalties before disabling them.

**Location:** `unified_pathfinder.py` line ~3757

**How to tune:**

| Iterations | Board Size | Rationale |
|------------|-----------|-----------|
| 0 | Any | Disable barrel penalties entirely |
| 5-10 | Large (8k+ nets) | Get to Phase 2 fast |
| 20-30 | Medium (2k-5k nets) | Balanced approach |
| 40-50 | Small (< 1k nets) | Can afford longer Phase 1 |

**Phase 2 (penalties OFF) allows PathFinder to optimize without barrel constraint fighting convergence.**

**From today's experience:**
- Test board (512 nets): Phase 1 = 50 iterations worked perfectly
- Large board (8,192 nets): Phase 1 = 10 iterations better (less divergence)

---

### 9. Barrel Conflict Penalty Strength

**What it does:** How much to penalize edges passing through other nets' via barrels.

**Location:** `unified_pathfinder.py` line ~3765

**Formula:**
```python
conflict_penalty = min(multiplier × pres_fac, cap)
```

**Current settings:**
```python
# Phase 1
multiplier = 5.0   # Scale with pres_fac
cap = 50.0         # Maximum penalty
```

**How to tune:**

| Multiplier | Cap | Effect |
|------------|-----|--------|
| 2.0-5.0 | 30-50 | Light (large boards) |
| 10.0 | 100 | Medium (default) |
| 20.0 | 200+ | Aggressive (small boards) |

**Trade-off:**
- **Too strong:** Forces divergence (routes avoid barrels at all costs)
- **Too weak:** Doesn't prevent barrel conflicts

**Diagnostic:** If overuse grows in Phase 1, reduce multiplier/cap.

---

## Advanced Tuning

### 10. Layer Bias Alpha (`layer_bias` alpha)

**What it does:** Adjusts costs to balance load across layers dynamically.

**Location:** `_compute_layer_bias()` - alpha parameter

**Default:** 0.88 (strong smoothing)

**Range:** 0.5-0.95

- **Lower (0.5-0.7):** Quickly shifts routes to cooler layers
- **Higher (0.9-0.95):** Gradual shifts, prevents whiplash

---

### 11. ROI Extraction Threshold

**What it does:** Distance threshold for using focused ROI vs full-graph routing.

**Location:** `_route_all()` line ~4470

**Default:** 125 steps (50mm @ 0.4mm grid)

```python
ROI_THRESHOLD_STEPS = 125
```

**How to tune:**
- **Smaller (50-100):** More nets use ROI = faster but may fail long routes
- **Larger (150-200):** More nets use full graph = slower but more reliable

---

## Diagnostic Tuning Workflow

### Step 1: Check Congestion Ratio

```
ρ < 1.0 → Board is routable, tune for speed
ρ > 1.0 → Board is impossible, add layers FIRST
```

### Step 2: Run 10-20 Iterations and Check Trend

**If overuse is DECREASING:**
✅ Parameters are good, let it run

**If overuse is INCREASING:**
❌ Too gentle - increase `pres_fac_mult` to 1.3-1.5

**If overuse OSCILLATES (up/down/up/down):**
⚠️ Balance issue:
- Reduce `hist_gain` (try 0.12-0.15)
- Reduce barrel penalty if in Phase 1

### Step 3: Monitor Iteration Timing

With all GPU optimizations (from today):

| Board Size | Iter 1 | Iter 2+ | What's Slow? |
|------------|--------|---------|--------------|
| 512 nets | 2 min | 20 sec | Normal |
| 8,192 nets | 100 min | 2-3 min | Iteration 1 is inherently slow |

**If iterations 2+ are slow (>10 min):**
- Check if GPU optimizations are active
- Look for "BARREL-CONFLICT.*minutes" in logs
- Ensure Python cache is cleared

---

## Parameter Sets for Common Scenarios

### Scenario 1: Sparse Board (ρ < 0.5)

**Goal:** Fast routing

```python
pres_fac_mult = 1.15
pres_fac_max = 6.0
hist_gain = 0.12
max_iterations = 40
BARREL_PHASE_1_ITERS = 30
```

### Scenario 2: Normal Board (ρ 0.5-0.8)

**Goal:** Reliable convergence

```python
pres_fac_mult = 1.20
pres_fac_max = 10.0
hist_gain = 0.15
max_iterations = 60
BARREL_PHASE_1_ITERS = 30
```

### Scenario 3: Tight Board (ρ 0.8-1.0) ← YOUR BOARD

**Goal:** Force convergence without divergence

```python
pres_fac_mult = 1.35        # Aggressive!
pres_fac_max = 16.0         # High ceiling
hist_gain = 0.15            # Balanced
max_iterations = 100
BARREL_PHASE_1_ITERS = 10   # Short Phase 1
barrel_penalty_mult = 5.0   # Light penalties
barrel_penalty_cap = 50.0
```

### Scenario 4: Over-Congested (ρ > 1.0)

**Goal:** Add layers first, then tune

```
⚠️ STOP! Add layers until ρ < 1.0
No amount of parameter tuning will make impossible boards routable.
```

---

## Two-Phase Barrel Conflict Strategy

**Phase 1:** Apply barrel conflict penalties
- Reduces barrel conflicts quickly
- May increase overall congestion temporarily
- **Keep short** (10-20 iterations for large boards)

**Phase 2:** Remove penalties, let PathFinder optimize
- Converges to minimal overuse
- Barrel conflicts may increase slightly but stay manageable
- **Where actual convergence happens**

**Example from test board (512 nets):**
```
Phase 1 (iters 1-50):
  Barrel conflicts: 2,611 → 290 (89% reduction!)

Phase 2 (iters 51-75):
  Barrel conflicts: stable at ~300
  Overall overuse: 25,000 → 0 (CONVERGED!)
```

**Example from large board (8,192 nets):**
```
Phase 1 (iters 1-10): Short to avoid divergence
Phase 2 (iters 11+): Where convergence should happen
```

---

## Advanced: History vs Present Balance

The **hist/pres ratio** in logs shows the balance:

```
[HISTORY-DEBUG] Iter 10:
  hist/pres ratio: 0.994  ← GOOD
```

**Target: 0.5 - 2.0**

**If ratio < 0.1** (history too weak):
- Increase `hist_gain` (try +0.05)
- OR increase `hist_cost_weight` (try +3.0)

**If ratio > 5.0** (history too strong):
- Decrease `hist_gain` (try -0.05)
- OR decrease `hist_cost_weight` (try -3.0)

**If ratio oscillates wildly:**
- History and present are fighting
- Reduce `hist_gain` to 0.10-0.12
- May need to reduce `pres_fac_mult` slightly

---

## GPU vs CPU Performance Tuning

### Memory vs Speed Trade-offs

**22 layers on 8,192-net board:**
- 181M edges
- ~10 GB GPU memory needed
- If GPU OOM: Reduce layers or use CPU fallback

**CPU fallback implications:**
- Routing still works
- ~2-3× slower
- Via pooling penalties disabled (minor)

### When GPU Memory is Tight

1. **Reduce layers** (20 → 18 saves ~2 GB)
2. **Disable via pooling** (already done for >100M edges)
3. **Use smaller grid pitch** (0.4mm → 0.5mm = fewer nodes)

---

## Common Issues and Fixes

### Issue: Overuse Growing Each Iteration

**Symptoms:**
```
Iter 1: 2.4M overuse
Iter 5: 4.4M overuse (getting worse!)
```

**Fixes:**
1. **Increase `pres_fac_mult`** to 1.35-1.50
2. **Reduce barrel penalty** if in Phase 1
3. Check if ρ > 1.0 (may be impossible to route)

---

### Issue: Oscillation (Overuse Ping-Pongs)

**Symptoms:**
```
Iter 10: 5,000
Iter 11: 2,000
Iter 12: 6,000
Iter 13: 1,500
(never settles)
```

**Fixes:**
1. **Reduce `hist_gain`** to 0.10-0.12
2. **Reduce `pres_fac_mult`** to 1.15-1.20
3. Check if in Phase 1 - may need to reach Phase 2

---

### Issue: Slow Iterations (>10 min for iter 2+)

**Symptoms:**
- Iteration 2+ taking 30-60 minutes
- Should be 2-5 minutes with GPU

**Fixes:**
1. Clear Python cache: `find . -name "*.pyc" -delete`
2. Check GPU optimizations are active (look for "GPU" in logs)
3. Restart Python (clears memory pools)
4. Check barrel conflict timing - should be <1 second

---

## Testing Your Changes

After tuning parameters:

1. **Test on small board first** (512 nets, 18 layers)
   - Should converge in < 2 hours
   - Validates parameters work

2. **Then try large board** (8,192 nets)
   - Expect ~4-8 hours
   - Monitor first 10 iterations

3. **Check convergence trend:**
```bash
grep "\[ITER.*\] routed" logs/latest.log
```

Look for **decreasing overuse**:
```
Iter 1: 2,451,307
Iter 5: 1,800,000 ✓ (decreasing = good!)
Iter 10: 950,000 ✓
```

---

## Emergency: Board Won't Converge

If you've tried everything and it still won't converge:

### Option 1: Check Physical Constraints
```bash
grep "Congestion ratio" logs/latest.log
```

If ρ > 1.0: **Add more layers** (not a parameter problem!)

### Option 2: Disable Barrel Penalties Entirely
```python
BARREL_PHASE_1_ITERS = 0  # Skip Phase 1 completely
```

This may leave barrel conflicts but allows other routing to converge.

### Option 3: Reduce Grid Pitch
Coarser grid = fewer nodes = easier routing:
```
grid_pitch: 0.4mm → 0.5mm or 0.6mm
```

Trade-off: Less routing precision.

---

## File Locations for Parameter Changes

### Automatic (board-adaptive):
- `orthoroute/algorithms/manhattan/parameter_derivation.py`
  - Lines 70-95: pres_fac_mult by congestion
  - Lines 128-130: hist_gain
  - Lines 120-126: hist_cost_weight

### Manual overrides:
- `orthoroute/algorithms/manhattan/unified_pathfinder.py`
  - Line 3757: BARREL_PHASE_1_ITERS
  - Line 3765: Barrel penalty formula

### Config file (future):
- Could add to `orthoroute.json` for per-project tuning

---

## References

- McMurchie & Ebeling, "PathFinder: A Negotiation-Based Performance-Driven Router for FPGAs" (1995)
- Betz & Rose, "VPR: A New Packing, Placement and Routing Tool for FPGA Research" (1997)
- OrthoRoute extends these FPGA concepts to heterogeneous PCB routing

---

**Pro Tip:** Document your parameter changes in git commits. It's easy to forget what you tried!

**Example commit message:**
```
Tune for 8k-net board convergence

- pres_fac_mult: 1.10 → 1.35 (aggressive rerouting)
- hist_gain: 0.20 → 0.15 (reduce history dominance)
- BARREL_PHASE_1_ITERS: 50 → 10 (faster to Phase 2)

Board: ρ=0.915, 22 layers, 8,192 nets
Expected: Convergence in ~50-75 iterations
```
