# 🔥 GPU OPTIMIZATION PLAN: Make It STUPIDLY FAST (REVISED)

**Date**: 2025-10-25 (Revised with user feedback)
**Goal**: 20× speedup via GPU acceleration
**Strategy**: ONE FIX AT A TIME (incremental, tested)
**Current Speed**: 0.85 nets/sec (~10 min/iteration)
**Target Speed**: 15-20 nets/sec (~30 sec/iteration)

---

## 🎯 BOTTLENECK (From Real Test Data)

```
Per-Net Breakdown:
ROI Extraction:    0.001s  ( 0.1%)
Portal Setup:      0.000s  ( 0.0%)
Pathfinding:       0.930s  (93.0%)  ← THE KILLER
Accounting:        0.070s  ( 7.0%)
────────────────────────────────────
Total:             1.000s
```

**93% of time = CPU Dijkstra on full 518K node graph**

**GPU would do this in 0.05s = 20× faster!**

---

## 🚨 TOP 5 IMMEDIATE BLOCKERS (Must Fix First)

### **BLOCKER #1: Batch/Micro-Batch Dispatch Still Exists**
**Problem**: Legacy code paths can still route via batch mode
**Location**: Check `_route_all()` for any batch routing returns
**Fix**: Delete or hard-gate with `if False:`
**Impact**: Ensures sequential mode is ONLY path

### **BLOCKER #2: .get() Calls on total_cost in Per-Net Path**
**Problem**: Costs are pulled to CPU before routing
**Location**: Check `_route_all()` sequential loop for `costs.get()`
**Fix**: Remove `.get()`, keep as CuPy throughout
**Impact**: Enables GPU pipeline (costs stay on device)

### **BLOCKER #3: find_path_roi() Not GPU-Aware**
**Problem**: Doesn't detect CuPy costs and call GPU path
**Location**: `unified_pathfinder.py` line ~1550-1593
**Fix**:
```python
# Early in find_path_roi(...)
costs_on_gpu = hasattr(costs, 'device')
if not force_cpu and costs_on_gpu and hasattr(self, 'gpu_solver') and self.gpu_solver:
    return self.gpu_solver.find_path_roi_gpu(src, dst, costs, roi_nodes, global_to_roi)
# else: CPU fallback
```
**Impact**: Enables GPU pathfinding (20× faster)

### **BLOCKER #4: GPU ROI Extractor Not Called**
**Problem**: GPU path calls CPU extractor, forcing transfer
**Location**: `cuda_dijkstra.py` in `find_path_roi_gpu()`
**Fix**: Use `_extract_roi_csr_gpu()` with bulk transfer (already exists!)
**Impact**: Eliminates per-net 216 MB transfer

### **BLOCKER #5: force_cpu=True Hardcoded**
**Problem**: GPU path is blocked even when it would work
**Location**: Check for `find_path_roi(..., force_cpu=True)` calls
**Fix**: Remove force_cpu=True once GPU is stable
**Impact**: Allows GPU to be used

---

## 📋 DETAILED FIX LIST (Priority Order)

### 🔥 **FIX #1: Kill Batch Dispatch** (10 min)
**Impact**: CRITICAL - Prevents regression to batch mode

**Current State**: Need to verify no batch calls exist
**Action**:
```bash
# Search for batch routing calls
grep -n "_route_all_batched\|_route_all_microbatch" unified_pathfinder.py
```

**If found**: Delete or disable with `if False:`
**If not found**: Verify _route_all() only calls sequential routing

**Test**: Run quick test, verify logs show [SEQUENTIAL] only, no [BATCH]

---

### 🔥 **FIX #2: Remove .get() in Sequential Loop** (15 min)
**Impact**: CRITICAL - Enables GPU pipeline

**Location**: `unified_pathfinder.py` line ~2893

**Current Code**:
```python
costs = self.accounting.total_cost  # Already correct!
```

**Verify**: This line does NOT have `.get()` call
**Check**: Ensure costs are passed to pathfinding WITHOUT .get()

**If found**: Remove any `.get()` calls on costs in sequential loop

**Test**: Run quick test, verify logs show [GPU-COSTS] message

---

### 🔥 **FIX #3: Make find_path_roi() GPU-Aware** (30 min)
**Impact**: CRITICAL - This enables GPU pathfinding

**Location**: `unified_pathfinder.py` line 1550-1593

**Current Code** (needs fix):
```python
def find_path_roi(self, src, dst, costs, roi_nodes, global_to_roi, force_cpu=False):
    # Currently forces CPU or doesn't detect GPU properly
```

**Fixed Code**:
```python
def find_path_roi(self, src, dst, costs, roi_nodes, global_to_roi, force_cpu=False):
    import numpy as np

    # Detect if costs are on GPU
    costs_on_gpu = hasattr(costs, 'device')

    # Try GPU first if costs are on GPU and GPU available
    if not force_cpu and costs_on_gpu and hasattr(self, 'gpu_solver') and self.gpu_solver:
        try:
            logger.debug(f"[GPU-PATH] Attempting GPU pathfinding (costs on GPU)")
            path = self.gpu_solver.find_path_roi_gpu(src, dst, costs, roi_nodes, global_to_roi)
            if path:
                logger.info(f"[GPU-PATH] SUCCESS - GPU pathfinding returned path length={len(path)}")
                if hasattr(self, '_gpu_path_count'):
                    self._gpu_path_count += 1
                return path
        except Exception as e:
            logger.warning(f"[GPU-PATH] FAILED: {e}, falling back to CPU")
            import traceback
            logger.debug(traceback.format_exc())

    # CPU Fallback
    if costs_on_gpu:
        logger.debug(f"[CPU-FALLBACK] Transferring costs for CPU pathfinding")
        costs = costs.get()

    # Convert arrays to NumPy
    roi_nodes = roi_nodes.get() if hasattr(roi_nodes, "get") else np.asarray(roi_nodes)
    global_to_roi = global_to_roi.get() if hasattr(global_to_roi, "get") else np.asarray(global_to_roi)

    # Track CPU usage
    if hasattr(self, '_cpu_path_count'):
        self._cpu_path_count += 1

    # ... rest of CPU Dijkstra code ...
```

**Test**: Run 10 nets, check for [GPU-PATH] SUCCESS messages

---

### 🔥 **FIX #4: Use GPU ROI Extractor** (20 min)
**Impact**: HIGH - Eliminates 216 MB transfer

**Location**: `cuda_dijkstra.py` in `find_path_roi_gpu()`

**Current Code** (check if it exists):
```python
def find_path_roi_gpu(self, src, dst, costs, roi_nodes, global_to_roi):
    # May call CPU extractor or have bugs
```

**Fixed Code**:
```python
def find_path_roi_gpu(self, src, dst, costs_gpu, roi_nodes_gpu, global_to_roi_gpu):
    """GPU pathfinding with GPU-resident costs"""
    import cupy as cp

    # Check if inputs are on GPU
    costs_on_gpu = hasattr(costs_gpu, 'device')
    roi_on_gpu = hasattr(roi_nodes_gpu, 'device')

    if costs_on_gpu and roi_on_gpu:
        # Fast path: Use GPU extractor (bulk transfer)
        logger.debug(f"[GPU-ROI-EXTRACT] Using GPU extraction with bulk transfer")
        roi_csr = self._extract_roi_csr_gpu(roi_nodes_gpu, global_to_roi_gpu, costs_gpu)
    else:
        # Fallback: Transfer and use CPU extractor
        logger.debug(f"[GPU-ROI-EXTRACT] Transferring to CPU for extraction")
        roi_nodes = roi_nodes_gpu.get() if hasattr(roi_nodes_gpu, 'get') else roi_nodes_gpu
        global_to_roi = global_to_roi_gpu.get() if hasattr(global_to_roi_gpu, 'get') else global_to_roi_gpu
        costs = costs_gpu.get() if hasattr(costs_gpu, 'get') else costs_gpu
        roi_csr = self._extract_roi_csr(roi_nodes, global_to_roi, costs)

    # Continue with GPU pathfinding using roi_csr
    # ... GPU Near-Far algorithm ...
```

**Verify**: `_extract_roi_csr_gpu()` exists and does bulk transfer (should be lines ~4975-5062)

**Test**: Check logs for [GPU-ROI-EXTRACT] messages

---

### ⚡ **FIX #5: Verify GPU Pool Reset Exists** (10 min)
**Impact**: CRITICAL - Without this, GPU has 0% success (cycle bugs)

**Location**: `cuda_dijkstra.py` lines 2471-2485

**Must Have This Code**:
```python
# Reset distance pool to infinity
self.dist_val_pool[:K, :max_roi_size] = cp.inf
# Reset parent pool to -1
self.parent_val_pool[:K, :max_roi_size] = -1
# Reset atomic key pool (CRITICAL!)
if hasattr(self, 'best_key_pool') and self.best_key_pool is not None:
    self.best_key_pool[:K, :max_roi_size] = INF_KEY
```

**Action**: READ lines 2471-2485 and verify this code exists
**If missing**: Add it back (this was a critical bug fix)
**If exists**: Confirm it's being called before GPU pathfinding

**Test**: GPU should NOT have "cycle detected" errors

---

### 🎯 **FIX #6: Remove force_cpu=True** (5 min)
**Impact**: HIGH - Enables GPU for all compatible nets

**Location**: Search for `find_path_roi(..., force_cpu=True)` calls

**Action**:
```bash
grep -n "force_cpu=True" orthoroute/algorithms/manhattan/unified_pathfinder.py
```

**Fix**: Change to `force_cpu=False` or remove parameter entirely

**Test**: GPU should be attempted (not skipped)

---

### 🔍 **FIX #7: Add GPU Usage Observability** (15 min)
**Impact**: MEDIUM - Helps debug and verify GPU is working

**Location**: End of `_route_all()` method

**Add Before Return**:
```python
# Log GPU vs CPU usage statistics
total_routed = routed_this_pass
gpu_count = getattr(self, '_gpu_path_count', 0)
cpu_count = getattr(self, '_cpu_path_count', 0)
gpu_pct = 100 * gpu_count / max(1, total_routed)
cpu_pct = 100 * cpu_count / max(1, total_routed)

logger.info(f"[ROUTING-STATS] GPU: {gpu_count} nets ({gpu_pct:.1f}%), CPU: {cpu_count} nets ({cpu_pct:.1f}%)")

# Reset counters for next iteration
self._gpu_path_count = 0
self._cpu_path_count = 0
```

**Test**: Check logs for [ROUTING-STATS] with GPU percentage >0%

---

### 🧹 **FIX #8: Delete Dead Code** (20 min)
**Impact**: LOW - Cleanup only

**Files**: `unified_pathfinder.py`

**Functions to DELETE**:
- `_route_all_batched_gpu()` (if still exists)
- `_route_all_microbatch()` (if still exists)
- Any batch-related helper functions

**Verification**:
```bash
grep -c "def _route_all_batched\|def _route_all_microbatch" unified_pathfinder.py
# Should return: 0
```

**Test**: Code still compiles and runs

---

### 🔧 **FIX #9: Verify No Per-Element .get() in Loops** (15 min)
**Impact**: HIGH - Prevents catastrophic slowdown

**Location**: `cuda_dijkstra.py` in `_extract_roi_csr_gpu()`

**BAD Pattern** (check for this):
```python
for ei in range(len(indices)):
    cost = global_costs[ei]  # If global_costs is CuPy, this transfers per-element!
```

**GOOD Pattern** (should be like this):
```python
# Transfer ONCE in bulk
global_costs_cpu = global_costs.get() if hasattr(global_costs, 'device') else global_costs

# Then loop on CPU array
for ei in range(len(indices)):
    cost = global_costs_cpu[ei]  # Fast CPU access
```

**Action**: Read `_extract_roi_csr_gpu()` and verify bulk transfer pattern

**Test**: Should see only ONE "Transferred costs" log per net, not thousands

---

### 🎯 **FIX #10: Conservative GPU Threshold** (5 min)
**Impact**: MEDIUM - Ensures GPU only attempts when ready

**Location**: `unified_pathfinder.py` line ~1557 and `config.py` line 192

**Action**:
```python
# Start with HIGH threshold (conservative)
gpu_roi_min_nodes: int = 50000  # Only massive ROIs use GPU initially
```

**Then gradually lower**:
1. Test with 50000 - verify GPU works
2. Lower to 20000 - test
3. Lower to 10000 - test
4. Lower to 5000 - test
5. Lower to 1000 - test

**Stop lowering** when GPU starts failing or performance degrades

**Test**: Monitor GPU success rate at each threshold

---

## 📝 IMPLEMENTATION PLAN (SEQUENTIAL)

### **PHASE 1: Make GPU Work** (1-2 hours)

Execute these fixes IN ORDER, testing after each:

1. ✅ Verify batch dispatch is dead → FIX #1
2. ✅ Verify no .get() in sequential loop → FIX #2
3. 🔧 Make find_path_roi() GPU-aware → FIX #3 (CRITICAL)
4. 🔧 Ensure GPU ROI extractor used → FIX #4
5. 🔧 Verify GPU pool reset exists → FIX #5
6. 🔧 Remove force_cpu=True → FIX #6
7. 🔧 Add GPU observability → FIX #7

**After Phase 1**:
- GPU should work for large ROIs (>50K nodes)
- Should see 10-20× speedup on those nets
- GPU success rate >50%

---

### **PHASE 2: Optimize Everything Else** (30-60 min)

8. 🔧 Set conservative GPU threshold → FIX #10
9. 🔧 Verify no per-element transfers → FIX #9
10. 🔧 Delete dead batch code → FIX #8

**After Phase 2**:
- All GPU-eligible nets use GPU
- No unnecessary transfers
- Clean codebase

---

### **PHASE 3: Optional Further Speedups** (If Needed)

Only do these if Phase 1-2 don't hit 15-20 nets/sec:

11. Increase ROI threshold (125 → 200) carefully
12. Pre-allocate GPU buffers
13. Implement A* heuristic
14. Incremental cost updates

---

## 🧪 TESTING PROTOCOL

### After EACH Fix:

**Step 1: Clean Cache**
```bash
find orthoroute -name "*.pyc" -delete
find orthoroute -name "__pycache__" -type d -exec rm -rf {} +
```

**Step 2: Quick Test (10 nets, ~10-20 seconds)**
```python
# Modify main.py or create test script
MAX_ITERATIONS = 1
MAX_NETS_TO_ROUTE = 10  # Only route first 10 nets
```

**Step 3: Check Results**
```bash
# GPU usage
grep "\[GPU-PATH\].*SUCCESS" test_quick.log | wc -l
# Should be >0 if GPU working

# Errors
grep -i "error\|exception" test_quick.log | head -10
# Should be minimal

# Performance
grep "Path=" test_quick.log
# Should see <0.2s if GPU, ~0.9s if CPU
```

**If GPU works for 10 nets**: Proceed to next fix
**If GPU fails**: Debug this fix before moving on

---

## 📊 EXPECTED RESULTS

### Phase 1 Complete:
```
GPU Success Rate: 60-80%
GPU Path Time: 0.05-0.1s per net (vs 0.93s CPU)
Average Speed: 8-12 nets/sec
Iteration Time: ~1-2 minutes
Speedup: 10-15×
```

### Phase 2 Complete:
```
GPU Success Rate: 80-95%
GPU Path Time: 0.05s per net
Average Speed: 15-20 nets/sec
Iteration Time: ~30 seconds
Speedup: 18-20×
```

---

## ⚠️ CRITICAL: What NOT To Change

Based on your docs, these are PROVEN WORKING - **DON'T TOUCH**:

1. **ROI_THRESHOLD_STEPS = 125** (lines 2970, 3520, 3637)
   - Higher values cause ROI inflation → truncation → 0% success
   - Leave at 125 until GPU is stable

2. **GPU Pool Reset** (cuda_dijkstra.py lines 2471-2485)
   - This fixes critical cycle detection bug
   - Without it: 0% GPU success
   - Must stay in place

3. **Sequential Loop Structure** (lines 2914-3100)
   - Routes one net at a time
   - Updates costs after each (via commit_path)
   - This is correct PathFinder algorithm
   - Don't change to batch

4. **global_to_roi as NumPy array** (not dict)
   - Dicts cause indexing errors
   - Must remain as np.array

---

## 🐛 BUGS ALREADY FIXED (Don't Re-Introduce)

These were fixed during iterative testing:

1. ✅ `_cpu_path_count` AttributeError → hasattr check added
2. ✅ CuPy comparison TypeError → float conversion added
3. ✅ `.item()` on int errors → hasattr checks (7 locations)
4. ✅ `cp` import scope → moved to function top
5. ✅ Missing Nx/Ny/Nz keys → added to batch_data

**Verify these fixes remain** in any new code

---

## 🎯 AGENT EXECUTION STRATEGY

### **ONE Agent Per Fix** (Sequential):

**Agent 1**: Execute FIX #3 (Make find_path_roi GPU-aware)
- Implement the fix
- Clean cache
- Test with 10 nets
- Report GPU success rate
- **WAIT FOR APPROVAL** before proceeding

**Agent 2**: Execute FIX #4 (GPU ROI extractor) - ONLY IF AGENT 1 SUCCEEDS
- Implement the fix
- Test
- Report results
- Wait for approval

**Agent 3**: Execute FIX #6 (Remove force_cpu=True) - ONLY IF AGENT 2 SUCCEEDS
- And so on...

### **NO Parallel Agents!**
Each agent must:
1. Complete its fix
2. Test thoroughly
3. Report results
4. Get approval before next agent starts

---

## 📈 SUCCESS METRICS

### Must Achieve (Phase 1):
- ✅ GPU pathfinding works (not 100% failure)
- ✅ GPU success rate >50%
- ✅ Speed >5 nets/sec (vs 0.85 current)
- ✅ No crashes or critical errors
- ✅ Success rate maintained (88-92%)

### Stretch Goals (Phase 2):
- 🎯 GPU success rate >80%
- 🎯 Speed 15-20 nets/sec
- 🎯 Iteration time <1 minute

---

## 🔥 PRIORITY TL;DR

**Do these IN ORDER, testing after each:**

1. ✅ Verify batch dispatch is dead (FIX #1)
2. ✅ Verify no .get() in sequential loop (FIX #2)
3. 🔧 Make find_path_roi() GPU-aware (FIX #3) ← **START HERE**
4. 🔧 Use GPU ROI extractor (FIX #4)
5. 🔧 Verify pool reset exists (FIX #5)
6. 🔧 Remove force_cpu=True (FIX #6)
7. 🔧 Add GPU stats logging (FIX #7)

**Expected Result**: **15-20× SPEEDUP** 🔥

**Each fix takes 5-30 minutes. Total: 2-3 hours to STUPIDLY FAST routing.**

---

**Ready to execute? Approve and I'll start with ONE agent on FIX #3.**
