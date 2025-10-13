# NEXT SESSION QUICK START GUIDE

**Session Goal:** Find and fix the hidden bottleneck preventing 10× speedup
**Current Status:** 2.5× speedup achieved, need 4× more to meet minimum goal
**Estimated Time:** 4-8 hours to reach production-ready state

---

## TL;DR - Start Here

**Most likely issue:** Hidden CPU-GPU synchronization preventing full speedup

**Quick test to run first:**
```bash
# 1. Profile to find bottleneck
nsys profile -o profile_gpu python main.py --test-manhattan

# 2. Check for hidden synchronization
grep -r "cp.asnumpy\|\.get()\|cuda.synchronize" orthoroute/algorithms/manhattan/

# 3. Measure GPU utilization
nvidia-smi dmon -s u -d 5 > gpu_util.txt &
python main.py --test-manhattan
pkill nvidia-smi
cat gpu_util.txt
```

**Expected finding:** CPU-side loops or synchronization between kernel launches that are limiting parallelism.

---

## Background (Read This First)

### What We Achieved:
- ✅ GPU persistent router fully operational
- ✅ All 5 weekend plan phases implemented
- ✅ 2.5× speedup vs optimized CPU baseline (48 → 97 nets/sec)
- ✅ 53× fewer kernel launches (8,000+ → 150)
- ✅ System stable, no crashes

### What We Need:
- ⚠️ 10× speedup minimum (48 → 480 nets/sec)
- ⚠️ Currently only achieving 2.5×
- ⚠️ Missing 4× additional speedup

### Why We Think It's Fixable:
Looking at the timing breakdown:
- Kernel execution: 350-400ms per 64 nets = 160-180 nets/sec theoretical
- Observed throughput: 97 nets/sec = only 60% of theoretical
- **Gap:** 40% overhead somewhere (likely CPU-side)

---

## Diagnostic Steps (In Order)

### Step 1: Profile with nsys/nvprof (30 minutes)

**Goal:** Find where time is being spent

**Commands:**
```bash
# Install nsys if needed
# On Windows with CUDA Toolkit: already installed

# Profile a short test run
cd /c/Users/Benchoff/Documents/GitHub/OrthoRoute
nsys profile -o profile_gpu --trace=cuda,cudnn,cublas,osrt,nvtx python main.py --test-manhattan

# View report
nsys stats profile_gpu.qdrep

# Look for:
# - CPU gaps between GPU kernels (synchronization)
# - Unexpected host-device transfers
# - Long CPU-side operations during GPU idle
```

**What to look for:**
1. **CPU gaps in GPU timeline** - Indicates synchronization overhead
2. **Frequent D2H transfers** - Indicates hidden `.get()` or `cp.asnumpy()` calls
3. **CPU operations during GPU idle** - Indicates serial execution not parallel

**Expected finding:** CPU-side accounting or state management between batches causing serialization.

---

### Step 2: Check for Hidden Synchronization (15 minutes)

**Goal:** Find implicit CPU-GPU sync points

**Commands:**
```bash
cd /c/Users/Benchoff/Documents/GitHub/OrthoRoute

# Search for synchronization points
grep -n "cp.asnumpy\|\.get()\|cuda.synchronize\|cuda.stream_wait" \
  orthoroute/algorithms/manhattan/unified_pathfinder.py \
  orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py

# Search for array indexing that might trigger sync
grep -n "\[.*\].*=" orthoroute/algorithms/manhattan/unified_pathfinder.py | head -50

# Look for device-to-host transfers in the batch loop
grep -A5 -B5 "for.*batch" orthoroute/algorithms/manhattan/unified_pathfinder.py
```

**What to look for:**
1. `cp.asnumpy()` calls in the batch loop
2. Device array indexing like `arr[i]` that forces transfer
3. `if` statements checking device array values
4. `.get()` calls on CuPy arrays
5. Calls to `.item()` on scalars

**Common culprits:**
```python
# BAD - forces sync
if device_array[0] > 0:  # Forces D2H transfer
    do_something()

# GOOD - stays on device
kernel_that_checks_condition[...](device_array, ...)
```

---

### Step 3: Measure GPU Utilization (10 minutes)

**Goal:** Confirm GPU is being kept busy

**Commands:**
```bash
cd /c/Users/Benchoff/Documents/GitHub/OrthoRoute

# Start monitoring in background
nvidia-smi dmon -s u -d 5 -c 60 > gpu_util.txt &
GPU_PID=$!

# Run test
python main.py --test-manhattan 2>&1 | tee test_with_monitoring.txt &
TEST_PID=$!

# Wait 5 minutes then check
sleep 300
kill $TEST_PID
kill $GPU_PID

# Analyze utilization
echo "GPU Utilization Summary:"
awk '{sum+=$3; count++} END {print "Average:", sum/count "%"}' gpu_util.txt
grep -v "^#" gpu_util.txt | awk '{print $3}' | sort -n | head -5
grep -v "^#" gpu_util.txt | awk '{print $3}' | sort -n | tail -5
```

**What to look for:**
- **Good:** >70% average GPU utilization
- **Bad:** <50% average GPU utilization (indicates idle time)
- **Pattern:** Spiky utilization (kernel runs, then idle, then kernel) indicates serialization

---

### Step 4: Analyze Batch Loop (30 minutes)

**Goal:** Find CPU-side overhead in the batch iteration

**Files to examine:**
1. `orthoroute/algorithms/manhattan/unified_pathfinder.py` (lines ~700-900)
2. Look for the batch processing loop in `route_nets()` method

**What to check:**
```python
# EXAMPLE - What you might find:

for batch in batches:
    # GOOD - prep on GPU
    roi_data = prepare_batch_gpu(batch)

    # GOOD - kernel launch
    paths = cuda_dijkstra.route_batch(batch, roi_data)

    # BAD - this forces sync!
    num_routed = int(paths.success_count.get())  # Forces D2H transfer

    # BAD - this too!
    if paths.failed_nets.size > 0:  # .size on device array forces sync
        retry_nets.append(paths.failed_nets.get())  # Another sync

    # BAD - updating CPU state
    for net_id in range(len(batch)):  # CPU loop
        self.route_status[net_id] = check_route(paths[net_id])  # More syncs
```

**How to fix:**
```python
# GOOD - defer all status checks to end of iteration
for batch in batches:
    roi_data = prepare_batch_gpu(batch)
    paths = cuda_dijkstra.route_batch(batch, roi_data)
    all_paths.append(paths)  # Just accumulate on GPU

# Single sync at end
total_routed = sum([p.success_count.get() for p in all_paths])  # One sync
```

---

### Step 5: Fix Most Obvious Issue (1-2 hours)

Based on profiling, likely fixes:

**Fix A: Remove batch-level status checks**
```python
# BEFORE (in unified_pathfinder.py)
for batch in batches:
    result = self._route_batch_gpu(batch)
    success_count = int(result['success_count'].get())  # SYNC!
    self._log_batch_status(batch_num, success_count)  # CPU work

# AFTER
for batch in batches:
    result = self._route_batch_gpu(batch)
    batch_results.append(result)  # Stay on GPU

# Single sync at end
all_success = sum([r['success_count'].get() for r in batch_results])
```

**Fix B: Move accounting to GPU between iterations**
```python
# BEFORE (in unified_pathfinder.py)
for iteration in range(max_iters):
    for batch in batches:
        route_batch(batch)
    update_costs_cpu()  # CPU-side, forces sync

# AFTER
for iteration in range(max_iters):
    for batch in batches:
        route_batch(batch)
    update_costs_gpu()  # GPU kernel, no sync
```

**Fix C: Defer hotset computation**
```python
# BEFORE
for iteration in range(max_iters):
    route_all_batches()
    hotset = compute_hotset_cpu(overuse_map.get())  # SYNC!

# AFTER
for iteration in range(max_iters):
    route_all_batches()
    hotset = compute_hotset_gpu(overuse_map)  # On GPU
```

---

### Step 6: Validate Fix (30 minutes)

**After applying fix:**
```bash
# Run test again
python main.py --test-manhattan 2>&1 | tee test_after_fix.txt

# Extract performance
grep "nets/sec" test_after_fix.txt | tail -20

# Compare to baseline
echo "Before fix:"
grep "Total time:.*nets/sec" test_final.txt | tail -3
echo "After fix:"
grep "Total time:.*nets/sec" test_after_fix.txt | tail -3

# Look for improvement
# Target: 200+ nets/sec (4× current 50 nets/sec)
```

---

## Expected Timeline

| Step | Duration | Goal |
|------|----------|------|
| Profile with nsys | 30 min | Find bottleneck location |
| Check synchronization | 15 min | Find hidden sync points |
| Measure GPU util | 10 min | Confirm underutilization |
| Analyze batch loop | 30 min | Identify specific fixes |
| Implement fix | 1-2 hr | Remove synchronization |
| Validate | 30 min | Measure improvement |
| **Total** | **3-4 hr** | **Reach 10× target** |

---

## Most Likely Scenario

Based on code structure and common pitfalls:

**Hypothesis:** Batch status logging is forcing synchronization

**Location:** `unified_pathfinder.py`, lines ~800-850, in the batch processing loop

**Issue:** After each batch completes, code logs success rate:
```python
success_count = int(result['success_count'].get())  # Forces sync
logger.info(f"Batch {i}: {success_count}/{len(batch)} routed")
```

**Impact:** This forces GPU to wait for kernel completion and transfer result to CPU after EVERY batch (150+ times per iteration), preventing parallel batch execution.

**Fix:** Remove per-batch logging, only log at iteration end:
```python
# Accumulate results on GPU
batch_results.append(result)  # No sync

# At iteration end (one sync)
total = sum([r['success_count'].get() for r in batch_results])
logger.info(f"Iteration {iter}: {total}/{total_nets} routed")
```

**Expected gain:** 3-5× speedup (removes 150+ syncs per iteration)

---

## If Above Doesn't Work

### Alternative Hypotheses:

**Hypothesis B: ROI prep forcing sync**
- Location: `unified_pathfinder.py`, ROI bounding box computation
- Check: Is ROI computed on CPU from device arrays?
- Fix: Move ROI computation to GPU kernel

**Hypothesis C: Graph updates forcing sync**
- Location: `cuda_dijkstra.py`, cost/history updates
- Check: Are updates applied on CPU between batches?
- Fix: Ensure updates happen entirely on GPU

**Hypothesis D: CuPy array indexing**
- Location: Anywhere array indexing occurs: `arr[i] = value`
- Check: Does indexing trigger implicit synchronization?
- Fix: Use kernel launches for array updates

---

## Success Criteria

After optimization session, you should see:

1. **Performance:** 400+ nets/sec (8-10× vs 48 baseline)
2. **GPU utilization:** >80% average
3. **Synchronization:** <10 D2H transfers per iteration (down from 150+)
4. **Stability:** Still no crashes or errors

**If achieved:** Ready for production deployment ✅

**If not achieved:** Iterate on profiling and optimization

---

## Files to Focus On

### Primary (most likely to need changes):
1. `orthoroute/algorithms/manhattan/unified_pathfinder.py`
   - Lines ~700-900: Batch processing loop
   - Look for: `.get()`, `.item()`, status checks

### Secondary (if primary doesn't reveal issue):
2. `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`
   - Lines ~2274-2486: Persistent kernel launch
   - Check: Any implicit sync in host code?

### Don't touch (working correctly):
3. CUDA kernel code itself (already optimized)
4. GPU configuration (already correct)

---

## Pro Tips

1. **Start with profiling** - Don't guess, measure
2. **Look for patterns** - If throughput is exactly 2.5×, there's likely one bottleneck
3. **Check for batch effects** - If first batch is fast, later slow = memory issue
4. **Watch for sync patterns** - Spiky GPU util = serialization
5. **Test incrementally** - Fix one thing, measure, fix next

---

## Emergency Fallback

If you can't find the bottleneck after 4 hours:

**Plan B: Improve baseline instead**
- Current optimized baseline: 48 nets/sec
- With better ROI policy: Could reach 100+ nets/sec
- GPU would then be 97/100 = ~1× speedup
- But absolute performance would still improve

**Plan C: Use GPU for batch-level parallelism**
- Instead of batching 64 nets sequentially
- Route all 8,192 nets in parallel (memory limit: ~500 nets)
- Could achieve 10× speedup through pure parallelism

---

**Good luck! The bottleneck is there, you just need to find it. Start with profiling - it will tell you exactly where the time is going.**

---

## Quick Reference Commands

```bash
# Profile
nsys profile -o profile python main.py --test-manhattan
nsys stats profile.qdrep

# Monitor GPU
nvidia-smi dmon -s u -d 5 > gpu.txt &

# Search for sync points
grep -rn "\.get()\|cp.asnumpy" orthoroute/algorithms/manhattan/

# Test performance
python main.py --test-manhattan 2>&1 | tee test.txt
grep "nets/sec" test.txt | tail -20

# Compare results
diff test_before.txt test_after.txt | grep "nets/sec"
```
