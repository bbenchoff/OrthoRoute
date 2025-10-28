# HANDOFF: GPU CUDA Kernel Debug - Critical Fix Needed

## CURRENT STATUS (2025-10-27 23:30)

**PROBLEM**: GPU CUDA kernel is returning negative infinity (`-340282346638528859811704183484516925440.00`) causing catastrophic path reconstruction failures.

**IMPACT**: GPU routing completely broken - all pathfinding fails, falling back to ROI routing but paths are invalid.

**GOAL**: Fix GPU kernel to work with current codebase and achieve convergence on test board within a few iterations.

---

## WHAT TO KEEP (DO NOT REVERT)

These fixes from today's session are **working correctly** and should be preserved:

### ✅ History Accounting Fixes
1. **history_decay**: 1.0 (no decay - full memory)
2. **use_raw_present**: True (instead of present_ema)
3. **hist_gain**: 0.2-0.3 (balanced accumulation)
4. **HIST_COST_WEIGHT**: 10.0 (in `pathfinder/config.py`)
5. **history_cap_multiplier**: 100.0

**Location**: `orthoroute/algorithms/manhattan/unified_pathfinder.py` lines ~862-923 (Accounting class)

### ✅ Hotset Improvements
1. **Fixed size**: 100 nets (disabled adaptive sizing)
2. **60/40 mix**: Top scorers + random selection
3. **Cooldown**: Prevents consecutive rerouting
4. **History-aware selection**: Includes historical congestion

**Location**: `orthoroute/algorithms/manhattan/unified_pathfinder.py` lines ~4055-4087 (`_build_hotset`)

### ✅ Parameter Tuning
1. **pres_fac_mult**: 1.10 (gentler escalation)
2. **pres_fac_max**: 8.0 (capped to keep history competitive)
3. **base_cost_weight**: 0.05 (willing to detour for spreading)
4. **VIA_COST**: 1.5 (encourages horizontal spreading)

**Location**: `orthoroute/algorithms/manhattan/pathfinder/config.py`

### ✅ Auto-Configuration System
1. **board_analyzer.py** - Analyzes board characteristics (ρ, aspect ratio, capacity)
2. **parameter_derivation.py** - Derives optimal parameters automatically
3. **Unified config** - Single PathFinderConfig in `pathfinder/config.py`

**These are all working!**

---

## WHAT'S BROKEN (NEEDS FIX)

### ❌ GPU CUDA Kernel

**Error Output**:
```
[GPU-SEEDS] Persistent kernel complete: 183 iterations, dist=-340282346638528859811704183484516925440.00
[GPU-SEEDS] Path found in 184 iterations (-340282346638528859811704183484516925440.00ms)
[GPU-SEEDS] Path reconstruction exceeded max length 892944
```

**Analysis**:
- Distance value is -340282... = `-FLT_MAX` in float32
- GPU kernel is returning negative infinity
- Path reconstruction tries to build 892,944 node path (graph only has 446,472 nodes!)
- Happens on EVERY net, regardless of board configuration

**What Changed Today That Could Affect GPU**:
1. Cost validation code added (lines 983-984, 1004-1006) - but shouldn't cause -inf
2. Spatial smoothing (now disabled) - wasn't the cause
3. Initial jitter (now disabled) - wasn't the cause
4. Layer balancing strength changes (lines 3064-3080) - could affect costs
5. Outside-in hotset selection (lines 4237-4247) - shouldn't affect pathfinding

**Root Cause Hypothesis**:
The GPU kernel itself isn't broken - something is passing **invalid edge costs** to GPU. The kernel sees inf/NaN costs, interprets them as -FLT_MAX, and pathfinding explodes.

**Likely culprit**: Layer bias or cost calculation creating inf/NaN that slips past validation.

---

## HOW TO DEBUG

### Step 1: Verify Cost Array Before GPU Transfer

In `unified_pathfinder.py` around line 2920-2950 (where costs are updated before routing):

Add diagnostic:
```python
# BEFORE calling GPU pathfinding
cost_cpu = self.accounting.total_cost.get() if hasattr(self.accounting.total_cost, 'get') else self.accounting.total_cost
cost_min = float(cost_cpu.min())
cost_max = float(cost_cpu.max())
cost_has_nan = bool(np.isnan(cost_cpu).any())
cost_has_inf = bool(np.isinf(cost_cpu).any())

logger.error(f"[GPU-DEBUG] Cost array stats before GPU:")
logger.error(f"  min={cost_min}, max={cost_max}")
logger.error(f"  has_nan={cost_has_nan}, has_inf={cost_has_inf}")

if cost_has_nan or cost_has_inf:
    logger.error("[GPU-DEBUG] INVALID COSTS DETECTED - GPU will fail!")
    # Find where
    if cost_has_nan:
        nan_indices = np.where(np.isnan(cost_cpu))[0]
        logger.error(f"  NaN at edges: {nan_indices[:10]}")
    if cost_has_inf:
        inf_indices = np.where(np.isinf(cost_cpu))[0]
        logger.error(f"  Inf at edges: {inf_indices[:10]}")
```

**Expected**: If you see NaN or Inf, you've found the source.

### Step 2: Check Layer Bias Calculation

The layer bias multiplier (lines 3064-3080) could create inf if division by zero:

```python
# In layer bias update:
mean_overuse = np.mean(layer_overuse[layer_overuse > 0]) + 1e-9  # Check this has epsilon!
pressure = layer_overuse / mean_overuse  # Could be inf if mean is 0

# Verify:
logger.error(f"[GPU-DEBUG] Layer bias: mean_overuse={mean_overuse}, max_pressure={pressure.max()}")
```

### Step 3: Simplify Cost Calculation

Temporarily remove layer bias to isolate:

```python
# Line ~2920, comment out layer bias
# layer_bias = ...
layer_bias = None  # Disable for testing
```

If GPU works without layer bias → that's your culprit.

### Step 4: Check GPU Cost Transfer

The costs might be valid on CPU but corrupting during GPU transfer:

```python
# After costs computed, before GPU routing:
costs_before = self.accounting.total_cost.get()
# ... GPU routing happens ...
# Check if costs changed:
costs_after = self.accounting.total_cost.get()
if not np.array_equal(costs_before, costs_after):
    logger.error("[GPU-DEBUG] Costs changed during GPU operation!")
```

---

## TEST PROCEDURE

### Test 1: Baseline (Verify Current Failure)

```bash
cd /c/Users/Benchoff/Documents/GitHub/OrthoRoute
python main.py --test-manhattan 2>&1 | grep "dist=-340\|Path found\|ITER [1-3]" | head -30
```

**Expected**: See `dist=-340282...` errors

### Test 2: With Diagnostic Logging

Add the debug logging from Step 1 above, then:

```bash
python main.py --test-manhattan 2>&1 | grep "GPU-DEBUG\|ITER 1" | head -50
```

**Expected**: See "INVALID COSTS DETECTED" and which edges have NaN/inf

### Test 3: Without Layer Bias

Disable layer bias (set to None), then:

```bash
python main.py --test-manhattan 2>&1 | grep "dist=\|ITER [1-3]" | head -20
```

**Expected**: If distances are positive (e.g., `dist=45.23`), layer bias was the problem

### Test 4: CPU-Only Baseline

To verify pathfinding logic works:

```bash
ORTHO_CPU_ONLY=1 python main.py --test-manhattan 2>&1 | grep "ITER [1-5].*routed" | head -10
```

**Expected**: Should route successfully and converge

---

## SUCCESS CRITERIA

✅ **GPU kernel returns positive distances** (e.g., `dist=45.23ms`)
✅ **No "Path reconstruction exceeded" errors**
✅ **Paths found successfully** (length < 1000 nodes)
✅ **Iteration 1 completes** (512 nets routed)
✅ **Convergence by iteration 21** (overuse = 0 or close)

---

## CURRENT BOARD CONFIGURATION

**Test Board**: `TestBackplane.kicad_pcb`
- Dimensions: 41.9mm × 93.3mm (or 23.8mm × 188.5mm depending on load)
- Layers: 12-18 (varies by test)
- Nets: 512
- Auto-detected strategy: SPARSE or DENSE (depends on aspect ratio)

**Parameters in effect**:
- ρ (congestion): 0.16-1.2 (board-dependent)
- pres_fac_max: 6.0-13.0 (layer-adaptive)
- hist_weight: 10.0-23.0 (layer + aspect adaptive)
- Hotset: 76-128 nets (15-25%)

---

## FILES MODIFIED TODAY

**Core Routing**:
- `orthoroute/algorithms/manhattan/unified_pathfinder.py` (~500 lines changed)
  - History accounting fixes
  - Hotset improvements
  - Best-result caching
  - Iteration logging
  - Spread features (spatial smoothing, jitter) - **NOW DISABLED**
  - Cost validation (nan_to_num)

**Configuration**:
- `orthoroute/algorithms/manhattan/pathfinder/config.py` (~30 lines changed)
  - base_cost_weight: 0.3 → 0.05
  - VIA_COST: 0.7 → 1.5
  - PRES_FAC_MULT: 1.10 → 1.12
  - HIST_COST_WEIGHT: 1.0 → 10.0
  - Added base_cost_weight attribute

**New Modules** (working correctly):
- `orthoroute/algorithms/manhattan/board_analyzer.py` (284 lines)
- `orthoroute/algorithms/manhattan/parameter_derivation.py` (265 lines)

**GPU Code** (unchanged - but broken):
- `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py` (NO changes made)

---

## RECOMMENDED FIX APPROACH

### Quick Win (1 hour):

1. **Add comprehensive cost validation**
   - Before GPU transfer, check for any inf/NaN
   - Log exact indices and values
   - Identify which operation creates invalid costs

2. **Isolate the culprit**
   - Disable layer bias → test GPU
   - Disable history update → test GPU
   - Disable cost validation (nan_to_num) → test GPU
   - Binary search to find exact line causing issue

3. **Fix the root cause**
   - Add epsilon to prevent division by zero
   - Clamp intermediate values
   - Ensure GPU arrays stay synchronized with CPU

### Nuclear Option (if quick win fails):

**Git revert to last working GPU state**:
```bash
git log --oneline --since="2025-10-27 14:00"  # Find commit before GPU broke
git diff <last_good_commit> orthoroute/algorithms/manhattan/unified_pathfinder.py > changes.patch
# Manually extract and keep only history/hotset fixes
```

---

## KEY INSIGHTS FROM TODAY'S SESSION

### What Worked:
1. **Randomized hotset** (60/40 + cooldown) **eliminated phase-locking** → smooth convergence
2. **Auto-configuration** correctly identifies board constraints (narrow, dense, etc.)
3. **Best-result caching** saves partial solutions (e.g., 492/512 nets at 1k overuse)
4. **Aspect ratio penalty** prevents misclassifying narrow boards as SPARSE

### What Needs Work:
1. **GPU kernel** - broken by unknown change
2. **12-layer narrow boards** - physically impossible to fully converge (accept best result)
3. **Spatial smoothing** - good idea but needs CPU-only implementation first

### Performance Benchmarks (when GPU worked):
- **32-layer board**: Converged to 0 in 21 iterations (~10 minutes)
- **12-layer board**: Best = 1,036 overuse at iter 29 (492/512 nets)
- **Iteration time**: 30-60 seconds (GPU), 60-120 seconds (CPU)

---

## QUICK REFERENCE COMMANDS

**Test GPU status**:
```bash
python main.py --test-manhattan 2>&1 | grep "dist=-340" | wc -l
# If > 0, GPU still broken
```

**Test with CPU**:
```bash
ORTHO_CPU_ONLY=1 python main.py --test-manhattan
```

**Check latest routing log**:
```bash
ls -lt routing_log_*.txt | head -1
cat routing_log_*.txt | head -50
```

**Git status**:
```bash
git status
git diff orthoroute/algorithms/manhattan/unified_pathfinder.py | wc -l
# ~500 lines changed
```

---

## CONTEXT FOR NEXT CLAUDE

**What we achieved today**:
- Fixed 12 convergence bugs
- Implemented auto-configuration
- Added best-result caching
- Created iteration logging system
- Eliminated phase-locking oscillation
- Achieved convergence on 32-layer board

**What broke**:
- GPU CUDA kernel (unknown root cause)
- Returning -FLT_MAX distances
- Likely a cost array corruption issue

**What's needed**:
- 1-2 hours focused GPU debugging
- Find which cost calculation creates inf/NaN
- Fix without reverting history/hotset improvements

**Expected outcome**:
- GPU returns positive distances (5-50ms typical)
- Paths found successfully
- Current test board converges in 10-30 iterations

---

## CONTACT/REFERENCE

**Last working GPU commit**: dcab665 (before today's session)
**Current commit**: HEAD (GPU broken but everything else improved)
**Test command**: `python main.py --test-manhattan`
**Log location**: `orthoroute_debug.log` and `routing_log_*.txt`

**Key diagnostic**:
```bash
# If you see this, GPU is broken:
dist=-340282346638528859811704183484516925440.00

# If you see this, GPU is fixed:
dist=45.23ms
```

Good luck! The codebase is 95% there - just need to fix the GPU kernel issue.
