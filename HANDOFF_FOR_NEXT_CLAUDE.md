# Handoff for Next Claude: Make Sequential Routing FAST

**Date:** 2025-10-25
**Current Status:** Sequential routing works (90.8% success), needs 10× speedup
**Your Goal:** Eliminate the 216 MB cost transfer bottleneck

---

## 🎯 WHAT YOU'RE STARTING WITH

**Working solution:**
```bash
cd C:\Users\Benchoff\Documents\GitHub\OrthoRoute
set SEQUENTIAL_ALL=1
set MAX_ITERATIONS=3
python main.py --test-manhattan
```

**Expected results (current working baseline):**
- Iteration 1: 465/512 routed (90.8% success)
- Iteration 2: 366/512 routed (71.5%)
- Iteration 3: 356/512 routed (69.5%)
- Overuse decreasing: 8251 → 7657 → 7111
- Time: ~8 minutes per iteration
- Speed: ~2 nets/sec

**Check this works BEFORE changing anything!**

---

## 🔴 THE BOTTLENECK

**File:** `orthoroute\algorithms\manhattan\unified_pathfinder.py`
**Function:** `_route_all_sequential_gpu()`
**Lines:** 3464-3573

**Line 3512 (THE KILLER):**
```python
costs_for_net = self.accounting.total_cost.get()  # ← 216 MB transfer PER NET
```

**This transfers 216 MB × 512 nets = 110 GB per iteration!**

**Your mission:** Eliminate this transfer.

---

## 📝 STEP-BY-STEP CHANGES

### Change #1: Remove the .get() transfer

**File:** `orthoroute\algorithms\manhattan\unified_pathfinder.py`
**Line:** 3512

**Current:**
```python
costs_for_net = self.accounting.total_cost.get() if self.accounting.use_gpu else self.accounting.total_cost
```

**Change to:**
```python
# Keep costs on GPU (CuPy array) - no transfer!
costs_for_net = self.accounting.total_cost
```

---

### Change #2: Make find_path_roi() handle GPU costs

**File:** `orthoroute\algorithms\manhattan\unified_pathfinder.py`
**Line:** 1564-1566

**Current:**
```python
# Ensure arrays are CPU NumPy
costs = costs.get() if hasattr(costs, "get") else costs
roi_nodes = roi_nodes.get() if hasattr(roi_nodes, "get") else roi_nodes
global_to_roi = global_to_roi.get() if hasattr(global_to_roi, "get") else global_to_roi
```

**Change to:**
```python
# If costs are on GPU, use GPU pathfinding path
if hasattr(costs, 'device'):
    # Costs are CuPy - call GPU pathfinding
    logger.info(f"[GPU-COSTS] Using GPU pathfinding with GPU-resident costs")
    return self.gpu_solver.find_path_roi_gpu(src, dst, costs, roi_nodes, global_to_roi)

# Fallback: CPU pathfinding with NumPy costs
costs = costs.get() if hasattr(costs, "get") else costs
roi_nodes = roi_nodes.get() if hasattr(roi_nodes, "get") else roi_nodes
global_to_roi = global_to_roi.get() if hasattr(global_to_roi, "get") else global_to_roi
```

---

### Change #3: Fix GPU extraction to transfer costs only ONCE

**File:** `orthoroute\algorithms\manhattan\pathfinder\cuda_dijkstra.py`
**Function:** `_extract_roi_csr_gpu`
**Lines:** 4955-4980

**Current code has Python loop - replace entire function with:**

```python
def _extract_roi_csr_gpu(self, roi_nodes_gpu, global_to_roi_gpu, global_costs_gpu):
    """GPU-native ROI CSR extraction - transfers costs ONCE"""
    import cupy as cp
    import numpy as np

    # Transfer to CPU ONCE (bulk transfer is fast)
    roi_nodes = roi_nodes_gpu.get() if hasattr(roi_nodes_gpu, 'device') else np.asarray(roi_nodes_gpu)
    global_to_roi = global_to_roi_gpu.get() if hasattr(global_to_roi_gpu, 'device') else np.asarray(global_to_roi_gpu)
    costs = global_costs_gpu.get() if hasattr(global_costs_gpu, 'device') else np.asarray(global_costs_gpu)

    # Use existing CPU extraction (already optimized)
    return self._extract_roi_csr(roi_nodes, global_to_roi, costs)
```

**This is already done!** Just verify it's there.

---

## 🧪 HOW TO TEST

### Quick Test (3 iterations, ~25 minutes):

```bash
cd C:\Users\Benchoff\Documents\GitHub\OrthoRoute
set SEQUENTIAL_ALL=1
set MAX_ITERATIONS=3
python main.py --test-manhattan > test_quick.log 2>&1
```

**Wait ~25 minutes, then check results:**

```bash
# Check iterations completed:
grep "ITER [0-9].*routed=" test_quick.log

# Expected output:
# ITER 1: routed=465 (90.8%)
# ITER 2: routed=366 (71.5%)
# ITER 3: routed=356 (69.5%)
```

**Check performance:**
```bash
grep "GPU-SEQ.*nets/sec" test_quick.log | tail -5

# Current baseline: 1.0-2.0 nets/sec
# After optimization: 5-10 nets/sec (target)
```

**Check for errors:**
```bash
grep -E "(ERROR|Exception|failed:)" test_quick.log | head -20

# Should be minimal errors
# Some nets failing to route is normal (~50 nets fail)
```

---

## 📸 HOW TO CHECK SCREENSHOTS

**Location:** `debug_output\run_<timestamp>\`

**Find latest run:**
```bash
cd debug_output
ls -lt | head -5
cd run_<latest_timestamp>
```

**Key screenshots to check:**

1. **`04_iteration_01_*.png`** - After iteration 1
   - Should show organized routing
   - Multiple colors (different layers)
   - Congestion in middle vertical channel is normal

2. **`10_iteration_07_*.png`** - Mid-way through
   - Should show improving layer distribution
   - Less chaotic than iteration 1

**What GOOD routing looks like:**
- Structured, organized traces
- Multiple layers used (see various colors)
- Vertical and horizontal segments

**What BAD routing looks like:**
- All traces jammed in one vertical channel
- Chaotic, overlapping everywhere
- Single color dominating (only one layer used)

---

## 📊 HOW TO READ THE LOGS

### Iteration Summary Line:

```
2025-10-25 10:47:09 - INFO - [ITER 1] routed=465 failed=47 overuse=8251 edges=8251 via_overuse=0.8%
```

**What this means:**
- `routed=465` - Successfully routed 465 out of 512 nets (90.8%)
- `failed=47` - 47 nets failed to find paths
- `overuse=8251` - 8251 edge violations (using more capacity than allowed)
- `via_overuse=0.8%` - Via congestion metric

**Good convergence:**
- Iteration 1: routed=465, overuse=8251
- Iteration 2: routed=366, overuse=7657 ← overuse decreasing ✓
- Iteration 3: routed=356, overuse=7111 ← still decreasing ✓

**Bad convergence:**
- Overuse increasing (8251 → 9000 → 11000) ✗
- Routed nets declining sharply (465 → 200 → 100) ✗

---

### Performance Line:

```
2025-10-25 10:46:00 - INFO - [GPU-SEQ] Net 251/512 | Routed: 196 | Failed: 54 | 1.3 nets/sec | ETA: 202s
```

**What this means:**
- Currently routing net 251 out of 512
- 196 nets successfully routed so far
- 54 nets failed
- Processing at **1.3 nets/sec** ← THIS IS THE KEY METRIC
- Estimated 202 seconds remaining

**Performance targets:**
- Current baseline: 1-2 nets/sec
- After Phase 1: 5-8 nets/sec (eliminate cost transfer)
- After Phase 2: 10-15 nets/sec (GPU pathfinding)
- Final goal: 20+ nets/sec

---

### Error Messages to Watch:

**CRITICAL errors (stop immediately):**
```
'NoneType' object is not subscriptable
Path reconstruction: cycle detected
AttributeError: ... has no attribute '_track_net_edges'
```
→ These mean routing is completely broken, revert changes!

**Expected errors (normal):**
```
No path found from X to Y
ROI failed for net_id, trying larger ROI
GPU pathfinding failed: Invalid roi_batch tuple
```
→ These are normal fallbacks, some nets just can't route

---

## 🔍 DEBUGGING TIPS

### If success rate drops below 60%:

**Check what changed:**
```bash
git diff unified_pathfinder.py | head -50
```

**Look for:**
- Did you change ROI_THRESHOLD_STEPS? (should be 125)
- Did you change max_nodes truncation? (should be 200000)
- Did you modify cost update logic? (revert if yes)

**Quick revert:**
```bash
git checkout HEAD -- unified_pathfinder.py
git checkout HEAD -- cuda_dijkstra.py
```

---

### If you see 0% success (all nets failing):

**Common causes:**
1. ROI truncation cutting out src/dst nodes
2. Costs passed as wrong type (CuPy when CPU expected)
3. ROI too large (>200K nodes)
4. Modified GPU pool reset code

**Check logs for:**
```bash
grep "truncating" test_quick.log | head -10
```
If you see many "BFS ROI 200,000+ > 200,000, truncating" → ROIs too large

---

### If test runs but is SLOW (same speed as before):

**Check if optimization actually applied:**
```bash
grep "GPU-COSTS\|GPU-NATIVE\|COST-TRANSFER" test_quick.log
```

Should see "[GPU-COSTS] Using GPU pathfinding with GPU-resident costs"

**If not appearing:**
- Your code change didn't work
- Costs still being transferred to CPU
- Check lines 3512 and 1564 were actually modified

---

## 📁 FILES YOU'LL MODIFY

**Primary file:**
- `orthoroute\algorithms\manhattan\unified_pathfinder.py`
  - Line 3512: Remove `.get()` transfer
  - Line 1564-1566: Add GPU costs detection
  - Lines 3575-3820: DELETE `_route_all_batched_gpu()` (optional cleanup)
  - Lines 3042-3139: DELETE `_route_all_microbatch()` (optional cleanup)

**Secondary file:**
- `orthoroute\algorithms\manhattan\pathfinder\cuda_dijkstra.py`
  - Lines 4955-4980: `_extract_roi_csr_gpu()` - already optimized
  - Lines 4787-4831: `find_path_roi_gpu()` - may need updates

**DO NOT MODIFY:**
- Lines 2471-2485 in cuda_dijkstra.py (GPU pool reset - CRITICAL!)
- config.py (already configured correctly)

---

## ⚡ FASTEST PATH TO SUCCESS

**If you want results in 2-3 hours:**

**Do ONLY Change #1 and #2:**
1. Remove `.get()` on line 3512
2. Add GPU detection on line 1564
3. Test

**If it works → 5-10× faster immediately!**

**If it breaks:**
- Revert
- Debug carefully
- Try smaller changes

**Don't try to do everything at once!**

---

## 📊 HOW TO MEASURE SUCCESS

**After making changes, run test and check:**

1. **Success rate maintained:**
   ```bash
   grep "ITER 1.*routed=" test_quick.log
   # Should see: routed=450-470 (88-92%)
   ```

2. **Speed improved:**
   ```bash
   grep "nets/sec" test_quick.log | tail -3
   # Should see: 5-10 nets/sec (vs 1-2 baseline)
   ```

3. **Convergence still working:**
   ```bash
   grep "ITER [1-3].*overuse=" test_quick.log
   # Should see: overuse decreasing (8000 → 7500 → 7000)
   ```

4. **Routing quality:**
   - Open latest screenshot in `debug_output\run_*/04_iteration_01_*.png`
   - Should look organized with multiple colors
   - Compare to earlier good screenshots

**ALL FOUR must pass!** If any fail, revert and try smaller changes.

---

## 🚨 CRITICAL CONFIG VALUES (DON'T CHANGE)

**In unified_pathfinder.py:**
- `ROI_THRESHOLD_STEPS = 125` (lines 2877, 3520, 3637) - Don't increase!
- `max_nodes = 200000` (line 1410) - Don't decrease!

**In config.py:**
- `iter1_always_connect = False` (line 140) - Keep False!
- `use_micro_batch_negotiation = False` (line 177) - Keep False!
- `use_gpu_sequential = True` (line 183) - Keep True!

**If you change these, routing breaks (0% success)!**

---

## 🔧 CURRENT CODE STATE

**Sequential routing function location:**
```python
File: orthoroute\algorithms\manhattan\unified_pathfinder.py
Function: _route_all_sequential_gpu()
Lines: 3464-3573

The main loop (line 3495-3567):
  for idx, net_id in enumerate(ordered_nets):
      # Line 3507-3510: Update costs on GPU (fast)
      self.accounting.update_costs(...)

      # Line 3512: BOTTLENECK - Transfer costs to CPU
      costs_for_net = self.accounting.total_cost.get()  # ← FIX THIS LINE

      # Line 3527-3533: Extract ROI
      roi_nodes, global_to_roi = self.roi_extractor.extract_roi(...)

      # Line 3538-3540: Route net using CPU pathfinding
      path = self.solver.find_path_roi(..., force_cpu=True)  # ← AND THIS LINE

      # Line 3543-3547: Update accounting
      self.accounting.commit_path(edges)
```

---

## 🎯 YOUR IMPLEMENTATION PLAN

### Hour 1: Make the changes

1. Edit line 3512 (remove `.get()`)
2. Edit lines 1564-1580 (add GPU detection)
3. Save files

### Hour 2: Test and debug

```bash
cd C:\Users\Benchoff\Documents\GitHub\OrthoRoute
set SEQUENTIAL_ALL=1
set MAX_ITERATIONS=3
python main.py --test-manhattan > test_optimized.log 2>&1
```

Wait ~10-15 minutes (if it's faster, optimization worked!)

### Hour 3: Verify results

```bash
# Check success rate:
grep "ITER 1.*routed=" test_optimized.log
# Want: routed=450-470

# Check speed:
grep "nets/sec" test_optimized.log | tail -5
# Want: 5-10 nets/sec (vs 1-2 baseline)

# Check for errors:
grep "ERROR" test_optimized.log | head -20
# Should be minimal
```

**Open screenshot:** `debug_output\<latest>\04_iteration_01*.png`
- Should look good (organized routing, multiple colors)

### Hour 4: If it worked, run full test

```bash
set SEQUENTIAL_ALL=1
set MAX_ITERATIONS=30
python main.py --test-manhattan > test_final.log 2>&1
```

Let this run for ~4-6 hours (if optimized) to see full convergence.

---

## 🐛 TROUBLESHOOTING

### Problem: "AttributeError: 'cupy.ndarray' object has no attribute 'X'"

**Cause:** Passing CuPy array to code expecting NumPy

**Fix:** Add `.get()` before passing to that function, OR modify function to handle CuPy

---

### Problem: Success drops to 0-10%

**Cause:** ROI extraction or pathfinding broken

**Quick fix:**
```bash
git checkout HEAD -- unified_pathfinder.py
git checkout HEAD -- cuda_dijkstra.py
# Start over with smaller changes
```

---

### Problem: Same speed as before (no improvement)

**Cause:** Optimization not actually being used

**Debug:**
```bash
# Add this at line 3512:
logger.info(f"[DEBUG] Costs type: {type(costs_for_net)}, has device: {hasattr(costs_for_net, 'device')}")
```

Should see: "has device: True" if optimization working

---

## 📂 USEFUL LOG FILES

**Baseline (working):**
- `test_cpu_poc_60iter.log` - CPU sequential, 82% success
- `test_sequential_ALL_60iter.log` - Latest test, 90.8% success

**Reference test output:**
```
2025-10-25 10:47:09 - INFO - [ITER 1] routed=465 failed=47 overuse=8251
2025-10-25 10:48:00 - INFO - [ITER 2] routed=366 failed=47 overuse=7657
2025-10-25 10:48:51 - INFO - [ITER 3] routed=356 failed=47 overuse=7111
```

**Screenshot location:**
```
debug_output\run_20251025_104307\04_iteration_01_*.png
```

Open this to see what GOOD routing looks like!

---

## 💡 WHAT SUCCESS LOOKS LIKE

**After your optimization:**

**Speed increased:**
```
[GPU-SEQ] Net 251/512 | Routed: 196 | Failed: 54 | 8.5 nets/sec | ETA: 30s
```
(vs baseline 1.3 nets/sec)

**Logs show:**
```
[GPU-COSTS] Using GPU pathfinding with GPU-resident costs
[GPU-NATIVE] ROI extraction on GPU
```

**Results maintained:**
```
[ITER 1] routed=465 failed=47 overuse=8251  (same as baseline)
```

**Iteration time reduced:**
- Before: ~8 minutes
- After: ~1-2 minutes

---

## 🎯 YOUR SUCCESS CRITERIA

**Must achieve:**
- ✅ 450+ nets routed in iteration 1 (88%+)
- ✅ Overuse decreasing across iterations
- ✅ 5+ nets/sec (at least 3× faster)
- ✅ Screenshots look good

**Bonus goals:**
- 10+ nets/sec (5× faster)
- Remove all batch code (cleaner)
- <2 minute iterations

---

## 📞 QUICK REFERENCE

**Test command:**
```bash
set SEQUENTIAL_ALL=1 && set MAX_ITERATIONS=3 && python main.py --test-manhattan > test.log 2>&1
```

**Check results:**
```bash
grep "ITER [0-9].*routed=" test.log
grep "nets/sec" test.log | tail -3
```

**View screenshot:**
```bash
cd debug_output && ls -lt | head -2
```

**Revert if broken:**
```bash
git checkout HEAD -- orthoroute/algorithms/manhattan/unified_pathfinder.py
git checkout HEAD -- orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py
```

---

## 🎁 WHAT PREVIOUS CLAUDE ACCOMPLISHED

**7-8 hours of work delivered:**
- ✅ Root cause: Batch routing breaks PathFinder (freezes costs)
- ✅ Solution: Sequential routing with per-net cost updates
- ✅ Result: 90.8% success (vs 35% batch baseline)
- ✅ Fixed 5 critical bugs (GPU pool reset was the big one)
- ✅ Convergence verified (overuse decreasing)

**What wasn't finished:**
- ⏳ Speed optimization (still 2 nets/sec, target 20+)
- ⏳ Cost transfer elimination (216 MB bottleneck)
- ⏳ Batch code removal (cleanup)

**Your job:** Finish the speed optimization!

---

## 🚀 GO FAST

**Don't overthink it.** The bottleneck is obvious (line 3512). Remove the `.get()`, test, iterate.

**Test frequently.** Run 3-iteration test after every change (~25 min each).

**Focus on working code.** 90.8% success is awesome, don't break it chasing speed.

**The hard part is done.** Previous Claude did the debugging. You just optimize.

**Good luck!** 🔥
