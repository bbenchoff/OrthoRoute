# OrthoRoute Optimization Workflow

Complete guide for making performance optimizations with automated validation and regression detection.

**Target audience:** Contributors making performance improvements to OrthoRoute  
**Last updated:** April 12, 2026

---

## **Overview**

This workflow ensures that every optimization:
1. ✅ **Maintains correctness** — all nets routed, zero overuse
2. ✅ **Improves performance** — faster than previous baseline
3. ✅ **Avoids regressions** — validated against golden metrics
4. ✅ **Is reproducible** — profiling data captured for future reference

**Tools:**
- [scripts/optimize_and_validate.ps1](../../scripts/optimize_and_validate.ps1) — Automated test + validation
- [scripts/analyze_log.py](../../scripts/analyze_log.py) — Log parsing and metric extraction
- [tests/regression/smoke_metrics.json](../../tests/regression/smoke_metrics.json) — Fast validation baseline (100 nets, <30s)
- [tests/regression/golden_metrics.json](../../tests/regression/golden_metrics.json) — Production baseline (512 nets, 11-18 min)

---

## **Quick Start**

```powershell
# 1. Edit code (e.g., add @profile_time decorator)
# 2. Run quick validation
.\scripts\optimize_and_validate.ps1 -Compare tests/regression/smoke_metrics.json

# 3. If passed, run full validation
.\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane -Compare tests/regression/golden_metrics.json

# 4. Analyze profiling data
python scripts/analyze_log.py --compare tests/regression/golden_metrics.json

# 5. If successful, commit
git commit -m "optimization: <description>"
```

---

## **The Optimization Cycle**

### **Phase 1: Identify Bottleneck**

**Goal:** Understand what's slow before making changes

#### **Step 1.1: Run baseline with profiling**

```powershell
# Full backplane test with ORTHO_DEBUG=1
.\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane
```

**Expected output:** Routing completes, logs saved to `logs/latest.log`

#### **Step 1.2: Analyze profiling data**

```powershell
python scripts/analyze_log.py --compare tests/regression/golden_metrics.json
```

**Look for:**
- **PROFILING DATA** section — functions sorted by total time
- High-frequency functions (e.g., called 512× or 73× iterations)
- Functions taking >100ms per call

**Example output:**
```
PROFILING DATA (Top 10 by total time)
--------------------------------------------------------------------------------
  _build_owner_bitmap_for_fullgraph              67.2s  (  73 calls,   920.0ms avg)  ← BOTTLENECK
  commit_path                                     8.5s  ( 512 calls,    16.6ms avg)
  _path_to_edges                                  4.2s  ( 512 calls,     8.2ms avg)
  ...
```

**Decision:**
- Functions with **total time >10s** are high-priority targets
- Functions called **per-iteration** (73×) or **per-net** (512×) compound quickly
- GPU kernel calls should be <5ms; CPU overhead should be <50ms per net

#### **Step 1.3: Review code**

Read the bottleneck function in [orthoroute/algorithms/manhattan/unified_pathfinder.py](../../orthoroute/algorithms/manhattan/unified_pathfinder.py):

- Is there unnecessary computation?
- Can loops be vectorized?
- Are there redundant allocations?
- Can GPU operations replace CPU work?

**Document hypothesis:** "Function X takes 67s because it does Y in a loop. We can optimize by Z."

---

### **Phase 2: Implement Optimization**

**Goal:** Make the change while maintaining correctness

#### **Step 2.1: Create feature branch (optional)**

```powershell
git checkout -b optimization/reduce-bitmap-overhead
```

#### **Step 2.2: Make code changes**

**Example:** Add `@profile_time` decorator to measure new code paths

```python
from orthoroute.shared.utils.performance_utils import profile_time

class UnifiedPathFinder:
    
    @profile_time  # Logs "[PROFILE] function_name: XXXms" when ORTHO_DEBUG=1
    def _optimized_bitmap_build(self, seed_nodes):
        # New vectorized implementation
        ...
```

**Best practices:**
- One optimization at a time
- Add profiling to new code paths
- Keep changes small and testable
- Comment why the optimization works

#### **Step 2.3: Fast validation (smoke test)**

```powershell
# Quick check: does it still route correctly?
.\scripts\optimize_and_validate.ps1 -Compare tests/regression/smoke_metrics.json
```

**Exit code 0?** ✅ Proceed to step 3  
**Exit code 1?** ❌ Fix bugs (nets not routed, convergence failed)  
**Exit code 2?** ⚠️ Performance regression detected (check if expected)

**If test fails:**
```powershell
# Re-run with full debug logs
.\scripts\optimize_and_validate.ps1 -ProfileMode -ShowLog

# Review errors
Get-Content logs/latest.log | Select-String -Pattern "ERROR|FAIL|Exception"
```

---

### **Phase 3: Measure Performance Impact**

**Goal:** Quantify the improvement

#### **Step 3.1: Full test with profiling**

```powershell
# Run full backplane test with profiling
.\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane -Compare tests/regression/golden_metrics.json
```

**Expected time:** 11-18 min (512 nets, 18 layers)

#### **Step 3.2: Analyze results**

```powershell
python scripts/analyze_log.py --compare tests/regression/golden_metrics.json
```

**Look for:**
- **Total time:** Did it improve vs. golden (1106.6s)?
- **Avg iteration:** Did it decrease (was 15.2s)?
- **Profiling data:** Did target function time go down?
- **Comparison status:** PASS (no regression) or WARN (acceptable trade-off)?

**Document results:**
```
Before: _build_owner_bitmap_for_fullgraph = 67.2s (73 calls, 920ms avg)
After:  _build_owner_bitmap_for_fullgraph =  8.5s (73 calls, 116ms avg)
Improvement: 58.7s saved (87% reduction), 8× faster
```

#### **Step 3.3: Compare with previous baseline**

**Manual comparison:**
```powershell
# Read previous baseline doc
cat docs/optimization/golden_result_2026-04-10.md

# Compare key metrics:
# - Total time: 1106.6s (baseline) vs <your time> (new)
# - Iterations: 73 (baseline) vs <your iters> (new)
# - Convergence: must still be zero overuse
```

**Automated comparison:** (if using --compare flag)
```
GOLDEN COMPARISON
--------------------------------------------------------------------------------
  Overall status: PASS
  
  ✓ iterations          actual=      70 threshold=      88 => PASS  (improvement!)
  ✓ total_time_s        actual=   950.2 threshold=  1328.0 => PASS  (improvement!)
  ✓ barrel_conflicts    actual=     340 threshold=     450 => PASS
```

---

### **Phase 4: Validate & Document**

**Goal:** Ensure reproducibility and preserve knowledge

#### **Step 4.1: Run final validation** ✅

```powershell
# Clean run without debug overhead (verify release performance)
.\scripts\optimize_and_validate.ps1 -TestBoard backplane -Compare tests/regression/golden_metrics.json
```

**Must pass:** Exit code 0 (PASS) or exit code 2 (WARN acceptable if explained)

#### **Step 4.2: Document the optimization**

**For minor improvements (<10% speedup):**
- Update [docs/optimization/OPTIMIZATION_QUICK_REF.md](OPTIMIZATION_QUICK_REF.md) "Completed Optimizations" table

**For major improvements (>10% speedup or new baseline):**
- Create new baseline doc: `docs/optimization/optimization_baseline_$(Get-Date -Format 'yyyy-MM-dd').md`
- Use template: [docs/optimization/baseline_template.md](baseline_template.md)
- Include:
  - Before/after metrics
  - Profiling data comparison
  - Root cause explanation
  - Validation results (smoke + backplane)

**Example commit message:**
```
optimization: vectorize bitmap construction (8× faster bitmap builds)

Replaced per-seed GPU→CPU sync loop with single vectorized scatter-add.
Reduces _build_owner_bitmap_for_fullgraph from 920ms to 116ms avg.

Before: 73 iters, 1106.6s total (15.2s avg)
After:  70 iters,  950.2s total (13.6s avg)
Speedup: 14% total routing time reduction

Validation:
- Smoke test: PASS (100/100 nets, <30s)
- Backplane: PASS (512/512 nets, zero overuse, 70 iters)
```

#### **Step 4.3: Update golden metrics (if new baseline)**

If this is a new performance baseline that should be the new target:

```powershell
# Backup old golden
cp tests/regression/golden_metrics.json tests/regression/golden_metrics_$(Get-Date -Format 'yyyy-MM-dd').json.bak

# Update golden with new thresholds (manual edit)
# Set thresholds = actual_value × 1.20 for headroom
```

**Example:**
- Actual: 950.2s → Threshold: 1140s (950.2 × 1.20)
- Actual: 70 iters → Threshold: 84 (70 × 1.20)

**Re-validate:**
```powershell
python scripts/analyze_log.py --compare tests/regression/golden_metrics.json
# Should show PASS for all metrics (not WARN)
```

#### **Step 4.4: Commit changes**

```powershell
git add -A
git commit -m "optimization: <one-line description>"
git push origin optimization/reduce-bitmap-overhead  # or main
```

---

## **Decision Tree: Which Test to Run?**

```
┌─ Making a code change?
│
├─ Quick bug fix / refactor (no perf impact expected)
│  └─> .\scripts\optimize_and_validate.ps1
│     (smoke test, no profiling, fast feedback <30s)
│
├─ Performance optimization (targeted speedup)
│  ├─> .\scripts\optimize_and_validate.ps1 -Compare tests/regression/smoke_metrics.json
│  │  (validate correctness + check for major regression)
│  └─> .\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane
│     (measure actual speedup, capture profiling data)
│
├─ Risky change (algorithm modification, GPU kernel change)
│  ├─> .\scripts\optimize_and_validate.ps1 -ProfileMode -Compare tests/regression/smoke_metrics.json
│  │  (smoke test with profiling to catch early issues)
│  └─> .\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane -Compare tests/regression/golden_metrics.json
│     (full validation before commit)
│
└─ Establishing new baseline (after major optimization)
   ├─> .\scripts\optimize_and_validate.ps1 -TestBoard backplane  (clean run, no debug)
   ├─> .\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane  (profiling run)
   └─> Document in docs/optimization/optimization_baseline_YYYY-MM-DD.md
```

---

## **Rollback Procedure**

If an optimization introduces a regression or bug:

### **Option 1: Git revert (recommended)**

```powershell
git log --oneline -5  # Find commit hash
git revert <commit_hash>
git commit -m "revert: rollback <optimization name> due to <reason>"
```

### **Option 2: Manual rollback**

```powershell
git restore <file_path>
.\scripts\optimize_and_validate.ps1 -Compare tests/regression/smoke_metrics.json
# Verify rollback succeeded
```

### **Option 3: Bisect to find regression**

```powershell
git bisect start
git bisect bad   # Current version fails
git bisect good optimization_baseline_2026-04-10  # Known good version

# Git will checkout commits for testing
# For each commit:
.\scripts\optimize_and_validate.ps1 -TestBoard smoke
git bisect good   # if test passes
git bisect bad    # if test fails

# Git identifies the breaking commit
git bisect reset
```

---

## **Common Scenarios**

### **Scenario 1: Optimization helps smoke test but regresses backplane**

**Symptoms:**
- Smoke test: PASS (30s → 25s, 17% faster)
- Backplane: WARN or FAIL (1106s → 1250s, 13% slower)

**Possible causes:**
- Optimization adds overhead that dominates at scale
- Different convergence behavior on larger boards
- Edge case only visible with 512 nets

**Action:**
1. Analyze profiling data: what new overhead appeared?
2. Check if iterations increased (slower per-iter is acceptable if fewer iters)
3. If net regression: rollback and investigate

### **Scenario 2: Test passes but profiling shows no improvement**

**Symptoms:**
- Exit code 0 (PASS)
- Total time unchanged
- Target function still slow

**Possible causes:**
- Optimization not executed (code path not reached)
- Bottleneck shifted to different function
- Compiler optimization already did it

**Action:**
1. Verify code path: add temporary `logger.warning("CHECKPOINT")` markers
2. Check profiling output: did target function time go down at all?
3. Use `--ShowLog` to review full execution trace

### **Scenario 3: Smoke test passes, backplane test times out**

**Symptoms:**
- Smoke test: PASS (20 nets in 15s)
- Backplane: No log output after 30+ min

**Possible causes:**
- Infinite loop introduced
- Deadlock in GPU kernel
- Excessive memory allocation (swap thrashing)

**Action:**
1. Kill the test: Ctrl+C in PowerShell
2. Review last log lines: `Get-Content logs/latest.log | Select-Object -Last 100`
3. Check iteration progress: did it freeze on a specific net or iteration?
4. Re-run with reduced net count: modify conftest.py `_HEADLESS_SAMPLE_NETS = 50`

---

## **Profiling Best Practices**

### **Adding profiling to code**

```python
from orthoroute.shared.utils.performance_utils import profile_time

@profile_time  # Only logs when ORTHO_DEBUG=1, zero overhead otherwise
def my_function(self, args):
    # ...implementation...
```

**When to add `@profile_time`:**
- ✅ Functions in hot paths (called per-net or per-iteration)
- ✅ Functions you're actively optimizing
- ✅ New code paths you want to measure
- ❌ Functions called once (e.g., initialization) — adds noise
- ❌ Trivial functions (<1ms) — clutters output

### **Analyzing profiling output**

**Focus on:**
1. **Total time** — functions at the top of the list
2. **Call count × avg time** — high-frequency functions compound
3. **Max time** — outliers indicate edge cases

**Examples:**

**Good target** (high impact):
```
_build_owner_bitmap_for_fullgraph    67.2s  (73 calls, 920ms avg)
→ Called per-iteration, 920ms is too slow, 73× compounds to 67s total
→ HIGH PRIORITY: 8× speedup → saves 58s total routing time
```

**Poor target** (low impact):
```
_init_data_structures               0.8s  (1 call, 800ms avg)
→ Called once, already fast enough, one-time cost
→ LOW PRIORITY: even 2× speedup only saves 0.4s total
```

### **Before/after comparison**

**Manual comparison:**
```powershell
# Save baseline profiling
python scripts/analyze_log.py > baseline_profile.txt

# Make optimization

# Compare
python scripts/analyze_log.py > optimized_profile.txt
code --diff baseline_profile.txt optimized_profile.txt
```

**Automated (using JSON output):**
```powershell
# Baseline
python scripts/analyze_log.py --json > baseline.json

# Optimized
python scripts/analyze_log.py --json > optimized.json

# Compare (requires custom script or manual JSON diff)
```

---

## **Validation Checklist**

Before committing an optimization, verify:

- [ ] **Smoke test passes** (100 nets, <60s)
  ```powershell
  .\scripts\optimize_and_validate.ps1 -Compare tests/regression/smoke_metrics.json
  ```

- [ ] **Backplane test passes** (512 nets, zero overuse)
  ```powershell
  .\scripts\optimize_and_validate.ps1 -TestBoard backplane -Compare tests/regression/golden_metrics.json
  ```

- [ ] **No correctness regressions**
  - All nets routed: `nets_routed == total_nets`
  - Converged: `converged == True`
  - Zero overuse: `overuse_edges == 0`

- [ ] **Performance improved or unchanged**
  - Total time ≤ golden threshold
  - Iterations ≤ golden threshold (or explained if higher)
  - No new bottlenecks introduced (check profiling data)

- [ ] **Profiling data captured** (for future reference)
  ```powershell
  .\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane
  python scripts/analyze_log.py > docs/optimization/profile_$(Get-Date -Format 'yyyy-MM-dd').txt
  ```

- [ ] **Changes documented**
  - Update OPTIMIZATION_QUICK_REF.md (for minor changes)
  - Create new baseline doc (for major improvements)
  - Update golden_metrics.json (if new baseline)

- [ ] **Commit message includes metrics**
  ```
  Before: <time>
  After:  <time>
  Speedup: <percentage>
  ```

---

## **FAQ**

### **Q: When should I update golden_metrics.json?**

**A:** Only when establishing a new baseline **after** validating the optimization is stable and reproducible. Not after every small improvement.

**Update golden when:**
- Major optimization (>20% speedup)
- New algorithm implementation
- Quarterly baseline refresh (even if no change)

**Don't update golden when:**
- Minor tweak (<10% speedup)
- Experimental change (not yet proven)
- Regression (never update golden to allow worse performance!)

---

### **Q: How do I handle "WARN" status in golden comparison?**

**A:** `WARN` means soft performance regression (slower but still within acceptable range).

**Investigate:**
1. Check if the slowdown is intentional (e.g., more iterations for better convergence)
2. Review profiling data: what's taking longer?
3. Decide: is the trade-off acceptable?

**Actions:**
- **Acceptable trade-off** (e.g., +5% time for better quality): Document in commit message, proceed
- **Unintentional regression**: Rollback and investigate

---

### **Q: My optimization helps iteration time but increases iteration count. Is this good?**

**A:** Depends on total time.

**Example:**
- Before: 73 iters × 15.2s = 1106.6s total
- After:  80 iters × 13.0s = 1040.0s total

**Result:** ✅ **Good** — 6% speedup overall despite more iterations

**Explanation:** Faster per-iteration can change convergence behavior. If total time improves, it's a net win.

**Bad example:**
- Before: 73 iters × 15.2s = 1106.6s total
- After:  90 iters × 13.0s = 1170.0s total

**Result:** ❌ **Bad** — 6% slower overall despite faster per-iteration

---

### **Q: Can I skip smoke test and go straight to backplane?**

**A:** You can, but it's inefficient.

**Smoke test advantages:**
- **Fast feedback** (<30s vs. 11-18 min)
- **Catch obvious bugs early** (before wasting 15 min on backplane)
- **Good enough for correctness validation** (100 nets still exercises full pipeline)

**Recommended workflow:**
1. Smoke test first (quick sanity check)
2. If smoke passes → backplane test (full validation)
3. If smoke fails → fix and re-run smoke (don't waste time on backplane)

---

### **Q: How do I measure memory usage?**

**A:** Use `timing_context()` from performance_utils.py:

```python
from orthoroute.shared.utils.performance_utils import timing_context

with timing_context("my_operation") as ctx:
    # ...code...
    pass

# ctx.metrics has: time_s, memory_mb, cpu_percent
logger.info(f"Memory: {ctx.metrics.memory_mb:.1f} MB")
```

Or review Task Manager / Resource Monitor during backplane test (manual observation).

---

## **References**

- [scripts/README.md](../../scripts/README.md) — Scripts usage guide
- [OPTIMIZATION_QUICK_REF.md](OPTIMIZATION_QUICK_REF.md) — Quick reference cheat sheet
- [golden_result_2026-04-10.md](golden_result_2026-04-10.md) — Current golden baseline
- [tests/run_golden_regression.md](../../tests/run_golden_regression.md) — Golden regression test details
- [../../README.md](../../README.md) — OrthoRoute project overview
