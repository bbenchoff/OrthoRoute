# Monday Plan: Autonomous Debug and Fix of OrthoRoute PathFinder

## 🎯 Mission

Debug and fix the OrthoRoute PCB router to achieve >90% routing success in iteration 1 (currently stuck at 33%). Work autonomously: create agents, run tests, read logs, iterate until fixed.

---

## 📋 Current State (As of Friday Evening)

### What Was Implemented
- ✅ **9 elevator shaft fixes** (diffused history, column soft-cap, etc.)
- ✅ **3 critical bug fixes** (stride consistency, backtrace guards, logging)
- ✅ **4 policy fixes** (always-connect, connectivity check, ROI widening, overhead reduction)
- ✅ **Total**: 16 fixes, ~850 lines of code modified

### Performance Achieved
- ✅ **Speed**: 189 nets/sec (was 1.9 nets/sec) - **100× faster!**
- ✅ **Iteration 1 time**: 2.7 seconds (was 78+ seconds)
- ✅ **Connectivity checks**: Working (detecting disconnected ROIs)

### Problems Remaining
- ❌ **Success rate**: 33.2% (170/512) - Need >90%
- ❌ **Cycle errors**: 225 - Should be <10
- ❌ **Fixes not loading**: NONE of the key features are activating

### The Smoking Gun
```
ITER-1-POLICY count: 0       ← Always-connect policy NOT running
KERNEL-VERSION count: 0      ← Kernels using OLD cached version
KERNEL-RR count: 0           ← Round-robin bias NOT active
KERNEL-JITTER count: 0       ← Jitter NOT active
ATOMIC-KEY count: 0          ← 64-bit keys NOT being used
```

**Root cause**: Code changes are present in files, but **not being executed** due to caching or wrong code paths.

---

## 🔍 Your Mission: Debug Why Fixes Aren't Loading

### Phase 1: Verify Which Code Path Is Being Used

**Task**: Determine if the router is using the code paths where fixes were added.

**Method**:
1. Add print statements to verify execution
2. Check if `negotiation_mixin.py` is being used (has always-connect fix)
3. Check if `cuda_dijkstra.py` persistent kernel is being used (has RR/jitter)
4. Verify iteration tracking is working

**Debugging Code to Add**:

```python
# In negotiation_mixin.py line 311 (where always-connect should log):
print("DEBUG: _pathfinder_negotiation called, iteration=", it)
print("DEBUG: Checking always-connect condition:", it == 1, cfg.iter1_always_connect)
if it == 1 and cfg.iter1_always_connect:
    print("DEBUG: INSIDE ALWAYS-CONNECT BLOCK!")
    logger.info("[ITER-1-POLICY] Always-connect mode: soft costs only (no hard blocks)")

# In negotiation_mixin.py line 618 (where costs are updated):
print(f"DEBUG: _update_edge_total_costs called, iteration={self.current_iteration}")
print(f"DEBUG: iter1_always_connect={self.config.iter1_always_connect}")
if self.config.iter1_always_connect and self.current_iteration == 1:
    print("DEBUG: USING SOFT COSTS (×1000)")
else:
    print("DEBUG: USING HARD BLOCKS (np.inf)")

# In cuda_dijkstra.py line 3505 (where RR params should be prepared):
print("DEBUG: About to call _prepare_roundrobin_params")
print("DEBUG: Has attribute?", hasattr(self, '_prepare_roundrobin_params'))
pref_layers_gpu, src_x_coords_gpu, rr_alpha, window_cols, jitter_eps = self._prepare_roundrobin_params(...)
print(f"DEBUG: RR params: alpha={rr_alpha}, window={window_cols}, jitter={jitter_eps}")
```

**Expected output** if code is reached:
```
DEBUG: _pathfinder_negotiation called, iteration= 1
DEBUG: Checking always-connect condition: True True
DEBUG: INSIDE ALWAYS-CONNECT BLOCK!
[ITER-1-POLICY] Always-connect mode: soft costs only (no hard blocks)
DEBUG: _update_edge_total_costs called, iteration=1
DEBUG: USING SOFT COSTS (×1000)
DEBUG: About to call _prepare_roundrobin_params
DEBUG: Has attribute? True
DEBUG: RR params: alpha=0.12, window=20, jitter=0.001
```

**If these don't appear**: The code path is wrong - fixes are in dead code!

---

### Phase 2: Force Kernel Recompilation

**Task**: Ensure CUDA kernels recompile with new code.

**Method**:
1. Bump kernel version number (forces recompile)
2. Add compilation timestamp
3. Verify kernel source includes RR/jitter logic

**Steps**:

**1. Edit cuda_dijkstra.py line 1245** (kernel version print):
```cpp
// Change from v3.0 to v4.0 with timestamp
printf("[KERNEL-VERSION] v4.0-MONDAY-DEBUG compiled=%s %s\\n", __DATE__, __TIME__);
```

**2. Check kernel source has RR logic** (should be around line 1346):
```bash
grep -A20 "ROUND-ROBIN LAYER BIAS" orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py
```

If this returns nothing → **RR logic was never added to this kernel!**

**3. Verify kernel signature has parameters** (line 1219-1227):
```bash
grep "const int\* pref_layer\|const float rr_alpha\|const float jitter_eps\|unsigned long long\* best_key" orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py
```

Should return 4 matches. If not → **kernel signature incomplete!**

**4. Force Python to reimport**:
```python
import sys
# Remove cached modules
for mod in list(sys.modules.keys()):
    if 'orthoroute' in mod:
        del sys.modules[mod]

# Now import fresh
from orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra import CUDADijkstra
```

---

### Phase 3: Verify Always-Connect Policy

**Task**: Ensure iteration 1 uses soft costs, not hard blocks.

**Method**:
1. Check if `_update_edge_total_costs` is being called
2. Verify `current_iteration` attribute exists and equals 1
3. Verify `iter1_always_connect` config is True

**Debugging Steps**:

**1. Find where cost updates happen**:
```bash
grep -n "_update_edge_total_costs" orthoroute/algorithms/manhattan/*.py orthoroute/algorithms/manhattan/pathfinder/*.py
```

**2. Check if method is being called in negotiation**:
```bash
grep -B5 -A5 "_update_edge_total_costs" orthoroute/algorithms/manhattan/pathfinder/negotiation_mixin.py | head -30
```

**3. Add assertion to verify**:
```python
# In negotiation_mixin.py at start of _update_edge_total_costs:
assert hasattr(self, 'current_iteration'), "current_iteration attribute missing!"
assert hasattr(self.config, 'iter1_always_connect'), "iter1_always_connect config missing!"
print(f"DEBUG: Cost update - iteration={self.current_iteration}, iter1_always_connect={self.config.iter1_always_connect}")
```

**4. Check config value**:
```bash
grep "iter1_always_connect" orthoroute/algorithms/manhattan/pathfinder/config.py
```

Should show: `iter1_always_connect: bool = True`

---

### Phase 4: Test Minimal Fix First

**Task**: Test ONLY the always-connect fix in isolation.

**Method**:
1. Disable all other fixes temporarily
2. Test if always-connect alone gets >90% in iteration 1
3. If yes → re-enable other fixes one by one
4. If no → debug why always-connect isn't working

**Minimal Test**:

**1. Create test script** `test_always_connect.py`:
```python
#!/usr/bin/env python
"""Minimal test of always-connect policy"""

# Force fresh imports
import sys
for mod in list(sys.modules.keys()):
    if 'orthoroute' in mod:
        del sys.modules[mod]

# Import
from orthoroute.algorithms.manhattan.pathfinder.config import PathFinderConfig
from orthoroute.algorithms.manhattan.pathfinder.negotiation_mixin import NegotiationMixin

# Check config
cfg = PathFinderConfig()
print(f"iter1_always_connect = {cfg.iter1_always_connect}")

# Check if NegotiationMixin has the fix
import inspect
source = inspect.getsource(NegotiationMixin._update_edge_total_costs)
if "iter1_always_connect" in source:
    print("✓ Always-connect fix IS in the code")
else:
    print("✗ Always-connect fix NOT in the code")

if "PATHFINDER ITERATION 1 POLICY" in source:
    print("✓ Comment block present")
else:
    print("✗ Comment block missing")
```

**2. Run it**:
```bash
python test_always_connect.py
```

**Expected output**:
```
iter1_always_connect = True
✓ Always-connect fix IS in the code
✓ Comment block present
```

**If you get ✗**: The fix was lost or overwritten!

---

### Phase 5: Autonomous Testing Loop

**Task**: Create agents to test, analyze logs, and iterate.

**Workflow**:

```python
def autonomous_debug_loop():
    """Main debugging loop - run until problem is solved or max iterations reached"""

    max_iterations = 10

    for attempt in range(max_iterations):
        print(f"\n=== ATTEMPT {attempt + 1}/{max_iterations} ===\n")

        # Step 1: Clear all caches
        clear_all_caches()

        # Step 2: Run test
        result = run_routing_test(log_file=f"test_attempt_{attempt + 1}.log")

        # Step 3: Analyze log
        analysis = analyze_log(result.log_file)

        print(f"Iteration 1 success: {analysis['iter1_success_rate']}%")
        print(f"Cycles: {analysis['cycle_count']}")
        print(f"Features active: {analysis['features_active']}")

        # Step 4: Check success criteria
        if analysis['iter1_success_rate'] > 90 and analysis['cycle_count'] < 10:
            print("✓ SUCCESS! Router is working!")
            return True

        # Step 5: Diagnose and fix
        diagnosis = diagnose_issues(analysis)

        if diagnosis['fix_available']:
            print(f"Applying fix: {diagnosis['fix_description']}")
            apply_fix(diagnosis['fix_code'])
        else:
            print("No automated fix available - needs human intervention")
            print(f"Issues: {diagnosis['issues']}")
            return False

    print("Max iterations reached without success")
    return False
```

**Functions to implement**:

```python
def clear_all_caches():
    """Delete all Python and CuPy caches"""
    import subprocess
    subprocess.run(["find", ".", "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"])
    # Delete CuPy cache
    # etc.

def run_routing_test(log_file):
    """Run routing test and capture output"""
    import subprocess
    result = subprocess.run(
        ["python", "main.py", "--test-manhattan"],
        capture_output=True,
        text=True,
        timeout=600
    )
    with open(log_file, 'w') as f:
        f.write(result.stdout)
        f.write(result.stderr)
    return result

def analyze_log(log_file):
    """Extract key metrics from log"""
    with open(log_file, 'r', encoding='utf-16-le') as f:
        content = f.read()

    # Extract metrics
    iter1_match = re.search(r'Iteration 1.*?(\d+)/(\d+) routed \((\d+\.\d+)%\)', content)
    iter1_success_rate = float(iter1_match.group(3)) if iter1_match else 0

    cycle_count = content.count('cycle detected')

    features = {
        'ITER-1-POLICY': content.count('ITER-1-POLICY') > 0,
        'KERNEL-VERSION': content.count('KERNEL-VERSION') > 0,
        'KERNEL-RR': content.count('KERNEL-RR') > 0,
        'KERNEL-JITTER': content.count('KERNEL-JITTER') > 0,
        'ATOMIC-KEY': content.count('ATOMIC-KEY') > 0
    }

    return {
        'iter1_success_rate': iter1_success_rate,
        'cycle_count': cycle_count,
        'features_active': features
    }

def diagnose_issues(analysis):
    """Diagnose what's wrong and suggest fix"""
    issues = []
    fix_code = None

    if not analysis['features_active']['KERNEL-VERSION']:
        issues.append("CUDA kernels using cached version")
        fix_code = "bump_kernel_version_and_add_debug_prints()"

    if not analysis['features_active']['ITER-1-POLICY']:
        issues.append("Always-connect policy not activating")
        fix_code = "add_debug_prints_to_negotiation_path()"

    if analysis['cycle_count'] > 100:
        issues.append(f"{analysis['cycle_count']} cycle errors - atomic keys not working")

    if analysis['iter1_success_rate'] < 50:
        issues.append(f"Only {analysis['iter1_success_rate']}% success in iteration 1")

    return {
        'issues': issues,
        'fix_available': fix_code is not None,
        'fix_code': fix_code,
        'fix_description': issues[0] if issues else None
    }
```

---

## 🔧 Debugging Tasks (In Priority Order)

### Task 1: Verify Always-Connect Code Path (HIGH PRIORITY)

**Problem**: `[ITER-1-POLICY]` message never appears, meaning always-connect code isn't executing.

**Investigation**:

**1. Check if `_pathfinder_negotiation` is the right entry point**:
```bash
grep -n "def _pathfinder_negotiation" orthoroute/algorithms/manhattan/unified_pathfinder.py
```

**2. Verify it calls code that updates costs**:
```bash
# Find iteration loop
grep -A50 "def _pathfinder_negotiation" orthoroute/algorithms/manhattan/unified_pathfinder.py | grep "for it in range"

# Check what it calls during iteration
grep -A100 "for it in range.*max_iterations" orthoroute/algorithms/manhattan/unified_pathfinder.py | grep "_route_all\|_update.*cost"
```

**3. Find WHERE costs are actually updated**:
```bash
# Search all possible cost update locations
grep -rn "total\[~legal\] = np.inf\|total\[over_mask\] = np.inf" orthoroute/algorithms/manhattan/
```

**If you find multiple locations**: The fix might be in the WRONG location!

**Fix**: Add always-connect logic to ALL locations where costs are set to infinity.

---

### Task 2: Force Kernel Recompilation (HIGH PRIORITY)

**Problem**: Kernels using cached version without RR/jitter/atomic-key code.

**Investigation**:

**1. Verify kernel has the new code**:
```bash
# Check if RR logic exists in kernel
grep -n "ROUND-ROBIN LAYER BIAS" orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py

# Check if jitter logic exists
grep -n "BLUE-NOISE JITTER" orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py

# Check if atomic key logic exists
grep -n "atomicMin64\|best_key" orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py | grep "14[0-9][0-9]:"
```

**2. Verify kernel signature**:
```bash
grep -A60 "void sssp_persistent_stamped" orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py | grep "pref_layer\|src_x_coord\|rr_alpha\|jitter_eps\|best_key"
```

Should return 6 lines (pref_layer, src_x_coord, window_cols, rr_alpha, jitter_eps, best_key).

**3. Check args tuple matches**:
```bash
# Count kernel parameters
grep -A60 "void sssp_persistent_stamped" orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py | grep -c "const\|int\*\|float\*\|unsigned"

# Count args passed
grep -A50 "args = (" orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py | grep -v "^--" | head -60 | grep -c ","
```

**If counts don't match**: Parameter mismatch will cause silent failures!

**Fix**:
```python
# In cuda_dijkstra.py, bump version and add timestamp:
# Line 1245:
printf("[KERNEL-VERSION] v4.0-MONDAY %s %s\\n", __DATE__, __TIME__);
printf("[KERNEL-DEBUG] rr_alpha=%.3f jitter_eps=%.6f\\n", rr_alpha, jitter_eps);
```

This FORCES recompilation because source code changed.

---

### Task 3: Check Iteration Tracking (MEDIUM PRIORITY)

**Problem**: Kernel might not know what iteration it is.

**Investigation**:

**1. Verify current_iteration is set**:
```bash
grep -n "self.current_iteration = it\|gpu_solver.current_iteration = it" orthoroute/algorithms/manhattan/unified_pathfinder.py
```

**2. Check if CUDADijkstra has the attribute**:
```bash
grep -n "self.current_iteration = 1" orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py
```

Should be in `__init__` method around line 69.

**3. Verify it's passed to helper method**:
```bash
grep "_prepare_roundrobin_params.*current_iteration" orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py
```

**If missing**: Add iteration tracking everywhere.

---

### Task 4: Test Connectivity Check Overhead (LOW PRIORITY)

**Problem**: Connectivity checks are VERY frequent (865 times) and might be slowing things down.

**Current behavior**: Iteration 2+ is doing BFS checks on every net.

**Optimization**:
- Only check if previous attempt failed
- Only check in iteration 2+ (iteration 1 uses full graph anyway)
- Cache results more aggressively

**Code location**: `unified_pathfinder.py` around line 3000-3050

**Fix**:
```python
# Only check in iteration 2+, skip in iteration 1
if self.iteration > 1 and len(roi_nodes) > 0:
    # connectivity check
```

Change to:
```python
# Skip connectivity check entirely in iteration 1 (uses full graph)
if self.iteration == 1:
    pass  # No check needed
elif self.iteration > 1 and len(roi_nodes) > 0:
    # connectivity check
```

---

## 🧪 Testing Protocol

### Autonomous Test Procedure

**1. Create test runner script** `autonomous_test.py`:

```python
#!/usr/bin/env python3
"""Autonomous testing and debugging script"""

import subprocess
import re
import time
from pathlib import Path

def run_test(iteration_num):
    """Run routing test and return results"""
    print(f"\n{'='*60}")
    print(f"TEST ITERATION {iteration_num}")
    print(f"{'='*60}\n")

    log_file = f"autonomous_test_{iteration_num}.log"

    # Clear caches
    print("Clearing caches...")
    subprocess.run(["find", ".", "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
                   stderr=subprocess.DEVNULL)

    # Run test
    print("Running routing test...")
    start_time = time.time()

    result = subprocess.run(
        ["python", "main.py", "--test-manhattan"],
        capture_output=True,
        text=True,
        timeout=600
    )

    elapsed = time.time() - start_time

    # Save log
    with open(log_file, 'w') as f:
        f.write(result.stdout)
        f.write("\n\n=== STDERR ===\n\n")
        f.write(result.stderr)

    print(f"Test completed in {elapsed:.1f}s")
    print(f"Log saved to: {log_file}")

    # Analyze
    analysis = analyze_log(log_file)

    return {
        'log_file': log_file,
        'elapsed': elapsed,
        'analysis': analysis,
        'success': analysis['iter1_success_rate'] > 90 and analysis['cycle_count'] < 10
    }

def analyze_log(log_file):
    """Extract metrics from log"""
    with open(log_file, 'r', encoding='utf-16-le', errors='ignore') as f:
        content = f.read()

    # Extract iteration 1 success rate
    iter1_match = re.search(r'Iteration 1.*?(\d+)/(\d+) routed.*?\((\d+\.\d+)%\)', content, re.DOTALL)
    iter1_success_rate = float(iter1_match.group(3)) if iter1_match else 0
    iter1_routed = int(iter1_match.group(1)) if iter1_match else 0
    iter1_total = int(iter1_match.group(2)) if iter1_match else 512

    # Count messages
    features = {
        'ITER-1-POLICY': content.count('ITER-1-POLICY'),
        'KERNEL-VERSION': content.count('KERNEL-VERSION'),
        'KERNEL-RR': content.count('KERNEL-RR'),
        'KERNEL-JITTER': content.count('KERNEL-JITTER'),
        'ATOMIC-KEY': content.count('ATOMIC-KEY'),
    }

    cycle_count = content.count('cycle detected')

    return {
        'iter1_success_rate': iter1_success_rate,
        'iter1_routed': iter1_routed,
        'iter1_total': iter1_total,
        'cycle_count': cycle_count,
        'features': features
    }

# Main loop
if __name__ == "__main__":
    for i in range(1, 6):
        result = run_test(i)

        print(f"\nRESULTS:")
        print(f"  Iteration 1: {result['analysis']['iter1_routed']}/{result['analysis']['iter1_total']} ({result['analysis']['iter1_success_rate']}%)")
        print(f"  Cycles: {result['analysis']['cycle_count']}")
        print(f"  Features active: {sum(result['analysis']['features'].values())}/5")

        for feature, count in result['analysis']['features'].items():
            status = "✓" if count > 0 else "✗"
            print(f"    {status} {feature}: {count}")

        if result['success']:
            print("\n✓ SUCCESS! Problem solved!")
            break
        else:
            print("\n✗ Not yet working, will try next iteration...")
            # Here you would apply fixes based on diagnosis
```

**2. Run autonomous test**:
```bash
python autonomous_test.py
```

This will run up to 5 tests, analyzing each one and showing what's missing.

---

## 📁 Key Files to Check

### Files That MUST Have the Fixes

1. **negotiation_mixin.py**:
   - Line 311-313: Always-connect logging
   - Line 618-636: Soft cost implementation
   - Should contain: `if self.config.iter1_always_connect and self.current_iteration == 1:`

2. **cuda_dijkstra.py**:
   - Line 1219-1227: Kernel signature with RR/jitter params
   - Line 1244-1248: Kernel version print
   - Line 1346-1378: RR + jitter logic in kernel
   - Line 1434-1449: 64-bit atomic key usage
   - Line 3300-3358: `_prepare_roundrobin_params` method
   - Line 3505-3558: Method call and args tuple

3. **unified_pathfinder.py**:
   - Line 2209-2211: Iteration tracking for GPU solver
   - Line 1457-1501: Connectivity check method in ROIExtractor class
   - Line 3012, 3042: Calls to connectivity check

4. **config.py**:
   - Line 136: `iter1_always_connect: bool = True`
   - Line 120: `column_present_beta: float = 0.12`

---

## 🎯 Success Criteria

**Test is successful when**:
- ✅ Iteration 1 success: >90% (currently 33%)
- ✅ Cycle errors: <10 (currently 225)
- ✅ Features active: 5/5 (currently 0/5)
- ✅ Performance: >100 nets/sec (currently 189 - already good!)

**Iteration 1 Results Should Show**:
```
[ITER-1-POLICY] Always-connect mode: soft costs only
[KERNEL-VERSION] v4.0-MONDAY compiled=Oct 21 2025 22:00:00
[KERNEL-RR] ACTIVE alpha=0.120 window=20
[KERNEL-JITTER] ACTIVE eps=0.001000
[ATOMIC-KEY] Initialized 64-bit keys for 150 ROIs
[GPU-BATCH-SUMMARY] Iteration 1 complete:
  Batch result: 485/512 routed (94.7%), 27 failed
```

---

## 📝 Deliverables

At end of Monday session, provide:

1. **Status report**:
   - Final iteration 1 success rate
   - Final cycle count
   - Which fixes are actually working

2. **Root cause analysis**:
   - Why fixes weren't loading
   - What code paths were wrong
   - What was fixed

3. **Recommendation**:
   - If >90% achieved: Document success, create git commit
   - If 50-90% achieved: List remaining issues, estimated fix time
   - If <50%: Recommend alternative approach or architecture change

---

## 📚 Reference Documents

**Created Friday** (in repo):
- `ELEVATOR_SHAFT_FIXES.md` - Overview of all 9 elevator shaft fixes
- `ALL_FIXES_COMPLETE.md` - Summary of 12 total fixes
- `POLICY_FIXES_COMPLETE.md` - Summary of 4 policy fixes
- `EXPERT_RECOMMENDATIONS_IMPLEMENTED.md` - Expert guidance followed
- `COMPLETE_VERIFICATION.md` - Verification checklist
- `READY_TO_TEST.md` - Testing instructions

**Key insights from expert**:
- Iteration 1 should route ~100% of nets (ignore congestion)
- 33% success means hard blocks are preventing valid paths
- Always-connect policy is THE solution
- Connectivity checks are good but secondary

---

## 🚀 Quick Start for Monday Claude

**Step 1**: Read this document completely

**Step 2**: Read `COMPLETE_VERIFICATION.md` to understand what SHOULD be implemented

**Step 3**: Create debugging agent:
```
I need to debug why the PathFinder always-connect policy isn't activating.
The code is in negotiation_mixin.py lines 311-313 and 618-636, but the log
shows zero [ITER-1-POLICY] messages. Add debug prints to trace execution
and verify the code path is correct.
```

**Step 4**: Run test, read log, diagnose

**Step 5**: Create fix agent based on diagnosis

**Step 6**: Iterate until success criteria met or max 10 attempts

---

## ⚠️ Known Issues to Avoid

1. **File encoding**: Logs use UTF-16-LE, need `encoding='utf-16-le'` when reading
2. **Cache locations**:
   - Python: `**/__pycache__/` directories
   - CuPy: `%APPDATA%/cupy/kernel_cache` (Windows)
3. **Class structure**: `ROIExtractor` is in `unified_pathfinder.py`, NOT `roi_extractor_mixin.py`
4. **Iteration tracking**: Must update BOTH `self.iteration` AND `self.solver.gpu_solver.current_iteration`

---

## 💡 If All Else Fails

**Nuclear option**: Create a MINIMAL test that ONLY tests always-connect:

```python
# test_minimal.py
from orthoroute.algorithms.manhattan.pathfinder.negotiation_mixin import NegotiationMixin
import numpy as np

# Create mock instance
class MockConfig:
    iter1_always_connect = True
    phase_block_after = 2
    strict_overuse_block = True

class MockRouter(NegotiationMixin):
    def __init__(self):
        self.config = MockConfig()
        self.current_iteration = 1

# Test
router = MockRouter()
legal = np.array([True, False, True, False])
usage = np.array([0.5, 2.0, 1.0, 3.0])
cap = np.ones(4)
total = np.ones(4)

# This should use soft costs in iteration 1
result = router._update_edge_total_costs(legal, usage, cap, total)

print("Legal edges:", result[legal])
print("Illegal edges:", result[~legal])
print("Expected illegal: ~1000.0 (soft), not inf (hard)")
print("Actual:", result[~legal])

if not np.isinf(result[~legal]).any():
    print("✓ Always-connect WORKING!")
else:
    print("✗ Always-connect NOT working - still using hard blocks")
```

This tests the fix in isolation without running full routing.

---

## 🎯 Final Note

**The code IS better** - you went from 1.9 → 189 nets/sec. The fixes exist in the files. They're just not executing for some reason (caching, wrong code path, import issues).

**Monday's job**: Figure out WHY they're not executing and make them execute.

**You have all the tools** - the fixes are correct, they're just not running. Find the disconnect and you're done!

Good luck! 🚀
