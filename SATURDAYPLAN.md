# 🔥 SATURDAY PLAN: Make Sequential Routing BLAZINGLY FAST

**Date**: 2025-10-25 (for Saturday execution)
**Current Status**: Sequential routing works (90.8% success) but SLOW (2 nets/sec)
**Target**: 10-20 nets/sec (10× speedup)
**Estimated Time**: 2-3 hours

---

## 🎯 MISSION: ELIMINATE THE REAL BOTTLENECKS

Based on code inspection, here are the **actual** problems preventing fast routing:

### ❌ **What's Still Wrong:**

1. **Batching is still active after iteration 1** - Sequential only runs for iter 1, then falls back to legacy batching
2. **Per-net 216 MB GPU→CPU cost transfer** - `costs.get()` called in sequential loop
3. **GPU path exists but never reached** - `find_path_roi()` doesn't detect GPU costs properly
4. **CUDA logs thrash I/O** - INFO-level spam from kernel compilation
5. **GPU ROI threshold too high** - Needs to be 1000, not 5000

---

## 🚀 HIGH-IMPACT FIXES (DO THESE FIRST)

### **FIX #1: Kill Per-Net GPU→CPU Cost Copy** ⚡ CRITICAL
**Impact**: Eliminates 216 MB × 512 = 110 GB of PCIe traffic per iteration
**Time**: 10 minutes
**Difficulty**: Easy

**Problem Location**: `unified_pathfinder.py` in sequential routing loop

**Current Code (BAD)**:
```python
# Line ~2780-2785 (in sequential routing)
costs = self.accounting.total_cost.get() if self.accounting.use_gpu else self.accounting.total_cost
```

**Fixed Code (GOOD)**:
```python
# Keep costs on GPU (no .get() transfer!)
costs = self.accounting.total_cost  # Leave on GPU if CuPy
```

**Why This Matters**:
- This single `.get()` call transfers 216 MB from GPU to CPU
- Called once per net × 512 nets = 110 GB per iteration
- Takes ~7 seconds per iteration just in PCIe transfers
- **This is the #1 bottleneck**

---

### **FIX #2: Make find_path_roi() GPU-Aware** ⚡ CRITICAL
**Impact**: Enables GPU pathfinding, 2-3× faster than CPU Dijkstra
**Time**: 20 minutes
**Difficulty**: Medium

**Problem**: `find_path_roi()` doesn't detect GPU costs, always uses CPU fallback

**Location**: `unified_pathfinder.py` lines ~1543-1625

**Current Code (BAD)**:
```python
def find_path_roi(self, src, dst, costs, roi_nodes, global_to_roi, force_cpu=False):
    # ... setup ...

    # Always converts to CPU
    costs = costs.get() if hasattr(costs, "get") else costs
    roi_nodes = roi_nodes.get() if hasattr(roi_nodes, "get") else roi_nodes

    # Then routes on CPU
    return SimpleDijkstra.find_path(...)
```

**Fixed Code (GOOD)**:
```python
def find_path_roi(self, src, dst, costs, roi_nodes, global_to_roi, force_cpu=False):
    import numpy as np

    # Detect ROI size
    roi_size = len(roi_nodes) if hasattr(roi_nodes, '__len__') else roi_nodes.shape[0]

    # Check GPU availability
    gpu_threshold = getattr(self.config, 'gpu_roi_min_nodes', 1000)
    use_gpu = (not force_cpu) and hasattr(self, 'gpu_solver') and self.gpu_solver and roi_size > gpu_threshold

    # Detect if costs are on GPU
    costs_on_gpu = hasattr(costs, "device")  # CuPy arrays have .device

    if use_gpu and costs_on_gpu:
        # FAST PATH: Zero-copy GPU routing
        logger.debug(f"[GPU-FAST] Routing with GPU-resident costs (no transfer)")
        return self.gpu_solver.find_path_roi_gpu(src, dst, costs, roi_nodes, global_to_roi)

    # Fallback: Transfer to CPU if needed
    if costs_on_gpu:
        logger.debug(f"[GPU→CPU] Transferring costs for CPU pathfinding")
        costs = costs.get()

    # Convert other arrays if needed
    roi_nodes = roi_nodes.get() if hasattr(roi_nodes, "get") else roi_nodes
    global_to_roi = global_to_roi.get() if hasattr(global_to_roi, "get") else global_to_roi

    # CPU Dijkstra fallback
    # ... existing CPU code ...
```

**Why This Matters**:
- Enables GPU pathfinding for 50-70% of nets
- GPU Near-Far algorithm is 2-3× faster than CPU heap Dijkstra
- Keeps entire pipeline on GPU (no transfers)

---

### **FIX #3: Force Sequential for ALL Iterations** ⚡ CRITICAL
**Impact**: Ensures PathFinder convergence works correctly
**Time**: 15 minutes
**Difficulty**: Easy

**Problem**: Sequential only runs for iteration 1, then switches to batching

**Location**: `unified_pathfinder.py` in `_route_all()` method

**Current Code (BAD)**:
```python
def _route_all(self, tasks, all_tasks, pres_fac=1.0):
    # ... setup ...

    if self.iteration == 1:
        return self._route_all_sequential_gpu(...)  # Only iter 1!
    else:
        return self._route_all_batched_gpu(...)  # Iter 2+ uses batching (WRONG!)
```

**Fixed Code (GOOD)**:
```python
def _route_all(self, tasks, all_tasks, pres_fac=1.0):
    # ... setup ...

    # ALWAYS use sequential (PathFinder requires per-net cost updates)
    cpu_mode = os.getenv('ORTHO_CPU_ONLY', '0') == '1'

    if cpu_mode:
        return self._route_all_sequential_cpu(...)
    else:
        return self._route_all_sequential_gpu(...)
```

**Why This Matters**:
- PathFinder algorithm requires costs updated after EVERY net
- Batching freezes costs → breaks convergence
- This was causing success rate to decline from 90.8% → 60% after iteration 1

---

### **FIX #4: Use GPU-Native ROI CSR Extraction** 🔥 HIGH IMPACT
**Impact**: Eliminates CPU roundtrip for ROI building
**Time**: 10 minutes
**Difficulty**: Easy

**Problem**: GPU path calls CPU extractor, forcing GPU→CPU→GPU roundtrip

**Location**: `cuda_dijkstra.py` in `find_path_roi_gpu()`

**Current Code (BAD)**:
```python
def find_path_roi_gpu(self, src, dst, costs_gpu, roi_nodes_gpu, global_to_roi_gpu):
    # Calls CPU extractor (bad!)
    roi_csr = self._extract_roi_csr(roi_nodes, global_to_roi, costs)  # Forces transfer
```

**Fixed Code (GOOD)**:
```python
def find_path_roi_gpu(self, src, dst, costs_gpu, roi_nodes_gpu, global_to_roi_gpu):
    # Check if inputs are on GPU
    if hasattr(costs_gpu, 'device') and hasattr(roi_nodes_gpu, 'device'):
        # Use GPU extractor (stays on device!)
        roi_csr = self._extract_roi_csr_gpu(roi_nodes_gpu, global_to_roi_gpu, costs_gpu)
    else:
        # Fallback to CPU extractor
        roi_nodes = roi_nodes_gpu.get() if hasattr(roi_nodes_gpu, 'get') else roi_nodes_gpu
        global_to_roi = global_to_roi_gpu.get() if hasattr(global_to_roi_gpu, 'get') else global_to_roi_gpu
        costs = costs_gpu.get() if hasattr(costs_gpu, 'get') else costs_gpu
        roi_csr = self._extract_roi_csr(roi_nodes, global_to_roi, costs)
```

**Why This Matters**:
- Keeps entire ROI extraction on GPU
- Eliminates 216 MB GPU→CPU transfer
- Only transfers small ROI CSR (~200 KB vs 216 MB) = **1000× reduction**

---

### **FIX #5: Lower GPU ROI Threshold** 🎯 QUICK WIN
**Impact**: 50-70% more nets use GPU pathfinding
**Time**: 5 minutes
**Difficulty**: Trivial

**Problem**: Threshold is 5000 nodes, but most nets have 1000-5000 node ROIs

**Already Fixed**: Agent 2 completed this
- `config.py` has `gpu_roi_min_nodes = 1000` ✅
- `find_path_roi()` uses `getattr(self.config, 'gpu_roi_min_nodes', 1000)` ✅

**Verification**: Just confirm it's working in logs:
```
[GPU] Using GPU pathfinding for ROI size=2500 (threshold=1000)
```

---

### **FIX #6: Trim Noisy CUDA Logs** 🔇 QUALITY OF LIFE
**Impact**: Reduces I/O overhead, cleaner logs
**Time**: 10 minutes
**Difficulty**: Easy

**Problem**: CUDA kernel compilation logs spam at INFO level

**Locations**:
- `cuda_dijkstra.py` lines with `logger.info("[CUDA]")` or `logger.info("[CUDA-COMPILE]")`

**Fix**: Change to DEBUG level:
```python
# BEFORE:
logger.info("[CUDA-COMPILE] Compiling validate_parents kernel...")

# AFTER:
logger.debug("[CUDA-COMPILE] Compiling validate_parents kernel...")
```

**Or**: Comment out entirely if not needed:
```python
# logger.info("[CUDA-COMPILE] Compiling validate_parents kernel...")  # Too noisy
```

---

## 🔧 CORRECTNESS CHECKS (DON'T BREAK THESE)

### ✅ **Keep ROI_THRESHOLD_STEPS = 125**
**Location**: 3 places in `unified_pathfinder.py`
- Line ~2877
- Line ~3520
- Line ~3637

**Why**: Higher values create massive ROIs (>200K nodes) → truncation → 0% success

**Action**: VERIFY it's still 125, don't change it

---

### ✅ **Keep global_to_roi as NumPy Array**
**Location**: Sequential routing loop

**Current (CORRECT)**:
```python
global_to_roi = np.arange(self.N, dtype=np.int32)  # NumPy array
```

**Don't Do This**:
```python
global_to_roi = {i: i for i in range(self.N)}  # Dict causes indexing errors!
```

**Action**: VERIFY it's an array, not a dict

---

### ✅ **Don't Touch GPU Pool Reset**
**Location**: `cuda_dijkstra.py` lines 2471-2485

This critical fix eliminates cycle detection:
```python
# Reset distance pool to infinity
self.dist_val_pool[:K, :max_roi_size] = cp.inf
# Reset parent pool to -1
self.parent_val_pool[:K, :max_roi_size] = -1
# Reset atomic key pool (CRITICAL!)
if hasattr(self, 'best_key_pool') and self.best_key_pool is not None:
    self.best_key_pool[:K, :max_roi_size] = INF_KEY
```

**Action**: DO NOT MODIFY THIS CODE

---

### ✅ **Verify _extract_roi_csr_gpu() Does Bulk Transfer**
**Location**: `cuda_dijkstra.py` lines ~4925-4980

**Good Implementation** (bulk transfer):
```python
def _extract_roi_csr_gpu(self, roi_nodes_gpu, global_to_roi_gpu, costs_gpu):
    # Transfer ONCE in bulk
    roi_nodes = roi_nodes_gpu.get() if hasattr(roi_nodes_gpu, 'device') else roi_nodes_gpu
    global_to_roi = global_to_roi_gpu.get() if hasattr(global_to_roi_gpu, 'device') else global_to_roi_gpu
    costs = costs_gpu.get() if hasattr(costs_gpu, 'device') else costs_gpu

    # Use CPU extractor (fast, does work once)
    return self._extract_roi_csr(roi_nodes, global_to_roi, costs)
```

**Bad Implementation** (per-element transfers - if you see this, fix it):
```python
def _extract_roi_csr_gpu(self, roi_nodes_gpu, global_to_roi_gpu, costs_gpu):
    # DON'T DO THIS - Python loop with .get() per element!
    for node in roi_nodes_gpu:
        val = node.get()  # BAD - transfers one element at a time
```

**Action**: VERIFY bulk transfer is used, not per-element

---

## 📋 CONFIG SANITY CHECKS

### ✅ **config.py Settings**

Verify these are set correctly:

```python
# GPU Configuration
use_gpu: bool = False  # Safe default (override with USE_GPU=1)
use_gpu_sequential: bool = True  # Sequential routing enabled
use_incremental_cost_update: bool = False  # Opt-in feature
gpu_roi_min_nodes: int = 1000  # GPU threshold (lowered from 5000)

# Micro-batch DISABLED
use_micro_batch_negotiation: bool = False

# PathFinder parameters
pres_fac_init: float = 1.0
pres_fac_mult: float = 2.0
hist_gain: float = 2.5
```

**Action**: Just verify, these are already correct from Agent 2's work

---

## 🧪 TESTING PROTOCOL

After each fix, run this quick test:

### Quick Test (3 iterations, ~5-10 minutes):
```bash
cd C:\Users\Benchoff\Documents\GitHub\OrthoRoute
set SEQUENTIAL_ALL=1
set MAX_ITERATIONS=3
python main.py --test-manhattan > test_saturday.log 2>&1
```

### Check Results:
```bash
# Success rate (must be ≥88%)
grep "ITER [1-3].*routed=" test_saturday.log

# Speed (target: 8-15 nets/sec)
grep "nets/sec" test_saturday.log | tail -5

# GPU usage (should see GPU being used)
grep "GPU-FAST\|GPU-NATIVE\|Using GPU pathfinding" test_saturday.log | head -10

# No errors
grep -i "error\|exception\|failed" test_saturday.log | grep -v "failed=47"
```

### Expected Output:
```
ITER 1: routed=465 (90.8%)
ITER 2: routed=366 (71.5%)
ITER 3: routed=356 (69.5%)

[GPU-FAST] Routing with GPU-resident costs (no transfer)
[GPU] Using GPU pathfinding for ROI size=2500 (threshold=1000)

GPU-SEQ completed: 8.5 nets/sec
```

---

## 📈 EXPECTED PERFORMANCE GAINS

### Current Baseline:
- **Speed**: 2 nets/sec
- **Time**: ~8-10 minutes per iteration
- **Success**: 90.8% (iteration 1)

### After Fix #1 (Kill .get()):
- **Speed**: 5-8 nets/sec (3-4× faster)
- **Time**: ~2-4 minutes per iteration
- **Reason**: Eliminates 110 GB of PCIe transfers

### After Fix #2 (GPU-aware find_path_roi):
- **Speed**: 8-12 nets/sec (5-6× faster)
- **Time**: ~1-2 minutes per iteration
- **Reason**: GPU pathfinding for 50-70% of nets

### After Fix #3 (Sequential all iterations):
- **Success**: Maintains 70-80% through convergence
- **Convergence**: Proper PathFinder negotiation
- **Reason**: Correct algorithm behavior

### After All Fixes:
- **Speed**: 10-20 nets/sec (10× faster) 🔥
- **Time**: <1 minute per iteration
- **Success**: 88-92% maintained
- **Convergence**: Proper (overuse decreases)

---

## 🗂️ FILES TO MODIFY

### Primary Files:
1. **`unified_pathfinder.py`** - Main routing logic
   - Fix #1: Remove `.get()` in sequential loop
   - Fix #2: Make `find_path_roi()` GPU-aware
   - Fix #3: Force sequential for all iterations

2. **`cuda_dijkstra.py`** - GPU pathfinding
   - Fix #4: Use GPU-native ROI extraction
   - Fix #6: Reduce log noise

3. **`config.py`** - Configuration (verify only, already fixed)
   - Check `gpu_roi_min_nodes = 1000` ✅

### Verification Files:
- `pathfinding_mixin.py` - Check no other `.get()` calls
- `roi_extractor_mixin.py` - Verify ROI format

---

## ⚠️ CRITICAL REMINDERS

### DON'T:
- ❌ Change ROI_THRESHOLD_STEPS from 125
- ❌ Modify GPU pool reset code (lines 2471-2485)
- ❌ Use dict for global_to_roi (must be array)
- ❌ Add per-element `.get()` calls in loops
- ❌ Break sequential mode for any iteration

### DO:
- ✅ Keep costs on GPU throughout pipeline
- ✅ Use bulk transfers (not per-element)
- ✅ Test after every change
- ✅ Verify 88-92% success maintained
- ✅ Check GPU logs confirm GPU usage

---

## 🎯 SUCCESS CRITERIA

### Must Achieve:
1. ✅ **Speed**: 10-20 nets/sec (vs 2 nets/sec baseline)
2. ✅ **Success Rate**: 88-92% (maintain or improve)
3. ✅ **GPU Usage**: 50-70% of nets use GPU pathfinding
4. ✅ **Convergence**: Overuse decreases across iterations
5. ✅ **Sequential Mode**: Used for ALL iterations

### Verification:
```bash
# Speed check
grep "nets/sec" test_saturday.log | tail -1
# Should show: 10-20 nets/sec

# Success check
grep "ITER 1.*routed=" test_saturday.log
# Should show: 450-470 routed (88-92%)

# GPU usage check
grep "GPU" test_saturday.log | grep -c "Using GPU pathfinding"
# Should show: 250+ (>50% of nets)

# Sequential check
grep "SEQUENTIAL\|BATCH" test_saturday.log
# Should show: SEQUENTIAL only, no BATCH
```

---

## 🚀 IMPLEMENTATION ORDER

Execute in this exact order for maximum safety:

### Phase 1: GPU Pipeline (30-45 minutes)
1. ✅ Fix #5 - Verify GPU threshold is 1000 (already done)
2. 🔧 Fix #1 - Remove `.get()` in sequential loop
3. 🔧 Fix #2 - Make `find_path_roi()` GPU-aware
4. 🔧 Fix #4 - Use GPU-native ROI extraction
5. 🧪 Test - Verify speed improved, success maintained

### Phase 2: Correctness (20 minutes)
6. 🔧 Fix #3 - Force sequential for all iterations
7. 🔧 Verify - Check all "don't break" items
8. 🧪 Test - Verify convergence works properly

### Phase 3: Cleanup (10 minutes)
9. 🔧 Fix #6 - Trim noisy logs
10. 📊 Final Test - Full 60-iteration validation

---

## 📝 EXACT LINE NUMBERS (Based on Current Code)

### unified_pathfinder.py:
- **Line ~2780-2785**: Remove `.get()` - FIX #1
- **Line ~1543-1625**: Make GPU-aware - FIX #2
- **Line ~2730-2850**: Force sequential - FIX #3
- **Lines 2877, 3520, 3637**: Verify ROI_THRESHOLD_STEPS = 125

### cuda_dijkstra.py:
- **Line ~4790-4830**: Use GPU extractor - FIX #4
- **Line ~2471-2485**: DON'T TOUCH (pool reset)
- **Lines with [CUDA-COMPILE]**: Reduce logging - FIX #6
- **Line ~4925-4980**: Verify bulk transfer

### config.py:
- **Line 192**: Verify `gpu_roi_min_nodes: int = 1000`
- **Line 186**: Verify `use_gpu_sequential: bool = True`
- **Line 180**: Verify `use_micro_batch_negotiation: bool = False`

---

## 🎁 BONUS: OPTIONAL OPTIMIZATIONS (If Time Permits)

### 🔧 Pre-Allocate GPU Scratch Buffers (30-45 minutes)
**Impact**: Eliminates per-call malloc/free overhead
**Complexity**: Medium

Currently: ROI buffers allocated per call (`batch_indptr`, `dist`, `parent`)
Better: Pre-allocate once in `CUDADijkstra.__init__()`, reuse via slicing

### 🔧 Implement True Incremental Cost Updates (1-2 hours)
**Impact**: 9× speedup on cost calculations
**Complexity**: High

Already wired at iteration level, needs implementation in `EdgeAccountant`:
```python
def update_costs_incremental(self, changed_edges, base_costs, pres_fac, ...):
    # Only update costs for edges in changed_edges set (~2K edges)
    # Instead of all 54M edges
```

### 🔧 A* Pathfinding with Manhattan Heuristic (2-3 hours)
**Impact**: 2× speedup on pathfinding
**Complexity**: High

Use admissible heuristic: `h(n) = abs(x_goal - x_n) + abs(y_goal - y_n) + via_penalty`

---

## 💾 BACKUP BEFORE STARTING

```bash
cd C:\Users\Benchoff\Documents\GitHub\OrthoRoute
git add -A
git commit -m "Backup before Saturday optimization sprint"
git branch saturday-optimization
```

If something breaks:
```bash
git checkout main
git reset --hard HEAD
```

---

## 🏁 FINAL CHECKLIST

Before declaring victory:

- [ ] All 6 fixes implemented
- [ ] Test passes with 88-92% success
- [ ] Speed ≥10 nets/sec achieved
- [ ] GPU usage confirmed in logs (50-70% of nets)
- [ ] Sequential mode used for all iterations
- [ ] Convergence working (overuse decreasing)
- [ ] No errors or warnings in log
- [ ] Code committed to git
- [ ] Performance documented in new test log

---

## 📊 WHERE TO FIND THINGS

### Current Test Logs:
- `test_final_validation.log` - Latest validation attempt
- `test_saturday.log` - Create this for today's test
- `orthoroute_debug.log` - Detailed debug output

### Documentation:
- `FRIDAYSUMMARY.md` - Yesterday's investigation
- `NEXT_ITERATION_GUIDE.md` - Original optimization guide
- `HANDOFF_FOR_NEXT_CLAUDE.md` - Handoff document
- `AUTONOMOUS_OPTIMIZATION_COMPLETE.md` - Agent work summary

### Screenshots:
- `debug_output/run_*/` - Visual routing quality verification

---

## 🎯 LET'S MAKE IT BLAZINGLY FAST! 🔥

**Estimated Total Time**: 2-3 hours
**Expected Speedup**: 10× (from 2 nets/sec → 20 nets/sec)
**Risk**: Low (all changes reversible, test after each fix)
**Reward**: BLAZINGLY FAST sequential routing! 🚀

---

**Ready to execute? Let's go!** 💪
