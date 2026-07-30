# OrthoRoute Optimization Baseline — [DATE]

**Status:** [✅ IMPROVEMENT | ⚠️ REGRESSION | 📊 BASELINE]

<!-- 
  ✅ IMPROVEMENT = Performance improved vs previous baseline
  ⚠️ REGRESSION  = Performance degraded (document why if intentional)
  📊 BASELINE    = Initial baseline or major refactor
-->

## Summary

[One-paragraph summary of the optimization and its impact]

**Example:**
> This optimization vectorized bitmap construction in `_build_owner_bitmap_for_fullgraph`, reducing per-net overhead from 920ms to 116ms (8× speedup). Total routing time improved from 1106.6s to 950.2s (14% faster). All 512 nets routed successfully with zero overuse.

---

## Change Description

**Files modified:**
- [orthoroute/algorithms/manhattan/unified_pathfinder.py](../../orthoroute/algorithms/manhattan/unified_pathfinder.py) — [brief description]
- [orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py](../../orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py) — [brief description]

**Optimization type:**
- [ ] Algorithm improvement (logic change)
- [ ] Vectorization (loop → GPU kernel)
- [ ] Memory optimization (reduced allocations/copies)
- [ ] Caching (precompute vs. recompute)
- [ ] Other: _____________

**Root cause of previous slowdown:**
[Explain what was slow and why]

**Example:**
> The `_build_owner_bitmap_for_fullgraph` function iterated over seed nodes in a Python loop, calling `int(seed)` for each (100+ calls per net). Each `int()` triggered a GPU→CPU sync (~1ms each), accumulating to ~920ms per call × 73 iterations = 67.2s total (6% of routing time).

**Solution implemented:**
[Explain the fix and why it works]

**Example:**
> Replaced the loop with vectorized CuPy operations: `seed_words = seed_nodes // 32`, `masks = cp.uint32(1) << seed_bits`, then used `cupyx.scatter_add()` to build the bitmap in a single GPU kernel call. This eliminates all GPU→CPU round-trips, reducing overhead from ~920ms to ~116ms.

---

## Performance Metrics

### Before vs After

| Metric | Before | After | Change | Status |
|--------|--------|-------|--------|--------|
| **Total time** | 1106.6s | 950.2s | -156.4s (-14%) | ✅ |
| **Avg iteration** | 15.2s | 13.6s | -1.6s (-11%) | ✅ |
| **Iterations** | 73 | 70 | -3 (-4%) | ✅ |
| **Nets routed** | 512/512 | 512/512 | — | ✅ |
| **Converged** | True | True | — | ✅ |
| **Overuse** | 0 | 0 | — | ✅ |
| **Barrel conflicts** | 367 | 340 | -27 (-7%) | ✅ |

**Interpretation:**
[Explain if the results match expectations, any surprises, trade-offs]

**Example:**
> The optimization met expectations with a ~14% total speedup. Iteration count decreased slightly (73→70) as a side effect of faster convergence. Barrel conflicts also decreased, likely due to better net ordering from faster early iterations.

### Profiling Data Comparison

**Target function** (what was optimized):

| Function | Before | After | Improvement |
|----------|--------|-------|-------------|
| `_build_owner_bitmap_for_fullgraph` | 67.2s (73 calls, 920ms avg) | 8.5s (73 calls, 116ms avg) | **87% reduction** |

**Top 5 functions** (current bottlenecks):

| Function | Total Time | Call Count | Avg Time | Notes |
|----------|------------|------------|----------|-------|
| `commit_path` | 8.5s | 512 | 16.6ms | Next optimization target |
| `_path_to_edges` | 4.2s | 512 | 8.2ms | Acceptable |
| `find_path_fullgraph_gpu_seeds` | 3.8s | 512 | 7.4ms | GPU kernel overhead |
| ... | ... | ... | ... | ... |

**Next optimization candidates:**
1. `commit_path` — 8.5s total, called per-net (512×)
2. [other candidates...]

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| **Date** | [YYYY-MM-DD @ HH:MM:SS] |
| **Board** | [TestBackplane.kicad_pcb] |
| **Mode** | [KiCad Plugin / Headless / CLI] |
| **Hardware** | [NVIDIA GPU model / CPU-only] |
| **Repository** | [Commit hash or branch] |
| **KiCad Version** | [e.g., 9.0.0] |

---

## Validation Results

### Smoke Test (100 nets, fast validation)

```powershell
.\scripts\optimize_and_validate.ps1 -Compare tests/regression/smoke_metrics.json
```

**Result:** [✅ PASS | ⚠️ WARN | ❌ FAIL]

| Metric | Actual | Threshold | Status |
|--------|--------|-----------|--------|
| Nets routed | 100/100 | 100 | ✅ |
| Converged | True | True | ✅ |
| Iterations | 18 | ≤30 | ✅ |
| Total time | 25.3s | ≤90s | ✅ |

### Backplane Test (512 nets, full validation)

```powershell
.\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane -Compare tests/regression/golden_metrics.json
```

**Result:** [✅ PASS | ⚠️ WARN | ❌ FAIL]

| Metric | Actual | Threshold | Status |
|--------|--------|-----------|--------|
| Nets routed | 512/512 | 512 | ✅ |
| Converged | True | True | ✅ |
| Iterations | 70 | ≤88 | ✅ |
| Total time | 950.2s | ≤1328s | ✅ |
| Barrel conflicts | 340 | ≤450 | ✅ |

**Log analysis:**
```powershell
python scripts/analyze_log.py --compare tests/regression/golden_metrics.json
```

[Paste summary output or highlight key findings]

---

## Regression Thresholds

**Should these metrics become the new golden baseline?** [YES / NO / TBD]

If YES, update [tests/regression/golden_metrics.json](../../tests/regression/golden_metrics.json):

```json
{
  "gpu": {
    "iterations_max": 84,        // 70 × 1.20
    "total_time_s_max": 1140,    // 950.2 × 1.20
    "barrel_conflicts_max": 408  // 340 × 1.20
  }
}
```

**Rationale:**
[Why this should/shouldn't become the new baseline]

**Example:**
> YES — This optimization achieved significant improvement (14% speedup) with no correctness regressions across smoke and backplane tests. Profiling confirms the root cause was addressed. Update golden metrics to reflect the new baseline and prevent future regressions back to the slower implementation.

---

## Reproducibility

### Commands Used

**Baseline measurement (before optimization):**
```powershell
git checkout <before_commit_hash>
.\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane
python scripts/analyze_log.py > baseline_profile.txt
```

**Optimization measurement (after changes):**
```powershell
git checkout <after_commit_hash>
.\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane
python scripts/analyze_log.py > optimized_profile.txt
```

**Comparison:**
```powershell
code --diff baseline_profile.txt optimized_profile.txt
```

### Logs Archived

- [ ] Baseline log: `docs/optimization/logs/baseline_YYYY-MM-DD.log`
- [ ] Optimized log: `docs/optimization/logs/optimized_YYYY-MM-DD.log`
- [ ] Profiling comparison: `docs/optimization/profile_YYYY-MM-DD.txt`

---

## Commit Information

**Branch:** [e.g., `optimization/vectorize-bitmap-construction`]

**Commit message:**
```
optimization: [one-line description]

[Detailed explanation of what changed and why]

Before: [key metric]
After:  [key metric]
Speedup: [percentage or absolute improvement]

Validation:
- Smoke test: PASS (100/100 nets, <30s)
- Backplane: PASS (512/512 nets, zero overuse, X iters, Xs total)
```

**Example:**
```
optimization: vectorize bitmap construction (8× faster bitmap builds)

Replaced per-seed GPU→CPU sync loop with single vectorized scatter-add.
Reduces _build_owner_bitmap_for_fullgraph from 920ms to 116ms avg.

Before: 73 iters, 1106.6s total (15.2s avg)
After:  70 iters,  950.2s total (13.6s avg)
Speedup: 14% total routing time reduction

Validation:
- Smoke test: PASS (100/100 nets, 25.3s)
- Backplane: PASS (512/512 nets, zero overuse, 70 iters, 950.2s)
```

---

## Known Issues / Limitations

[Document any trade-offs, limitations, or edge cases introduced by this optimization]

**Example:**
- None identified — optimization is a pure speedup with no correctness impact
- OR: Increased iteration count by 5% but overall time still 10% faster due to faster per-iteration execution

---

## References

- [Previous baseline: optimization_baseline_YYYY-MM-DD.md](optimization_baseline_YYYY-MM-DD.md)
- [Golden result: golden_result_2026-04-10.md](golden_result_2026-04-10.md)
- [Optimization workflow guide](optimization_workflow.md)
- [GitHub issue/PR: #XXX](#) (if applicable)

---

## Checklist

Before publishing this baseline doc:

- [ ] Smoke test validation completed (PASS)
- [ ] Backplane test validation completed (PASS or explained)
- [ ] Profiling data captured and analyzed
- [ ] Before/after metrics documented with evidence
- [ ] Logs archived (if significant baseline)
- [ ] Golden metrics updated (if this becomes new baseline)
- [ ] Code committed with proper commit message
- [ ] OPTIMIZATION_QUICK_REF.md updated (for minor changes)
- [ ] This baseline doc added to docs/optimization/
