# OrthoRoute Performance Optimization

This folder contains all performance optimization documentation, baselines, and migration guides for OrthoRoute.

---

## 📊 Performance Baselines

### [Baseline: April 8, 2026](optimization_baseline_2026-04-08.md) ⚠️ **CURRENT**
**TestBackplane routing analysis (512 nets, 18 layers, 17.5 minutes)**

Performance regression detected vs. April 5 baseline:
- 46% slower than best performance (11.96 min → 17.5 min)
- Average iteration time: 15.7s (vs. 11.0s on April 5)
- Root cause investigation required
- Full profiling data with `ORTHO_DEBUG=1` needed

### [Baseline: April 5, 2026](optimization_baseline_2026-04-05.md) ✅ **BEST PERFORMANCE**
**TestBackplane routing analysis (512 nets, 18 layers, 11.96 minutes)**

Best documented performance:
- 4× faster than April 3 baseline
- 2× faster than April 3 persistent kernel
- 11.0s average iteration time
- 512/512 nets routed with zero overuse

### [Baseline: April 3, 2026](optimization_baseline_2026-04-03.md)
**TestBackplane routing analysis (512 nets, 32 layers, 44.23 minutes)**

Original baseline with:
- Current performance measurements
- Profiled bottlenecks and priorities
- Phase-by-phase optimization roadmap
- Success metrics and validation criteria
- Quick start commands for optimization work

**Quick Reference**: [OPTIMIZATION_QUICK_REF.md](OPTIMIZATION_QUICK_REF.md) - Single-page cheat sheet

---

## 🔧 Optimization Tools & Systems

### [Optimization Workflow](optimization_workflow.md) ✨ **NEW**
**Complete guide for making performance optimizations with automated validation**

- Automated optimization cycle: deploy → test → validate
- Regression detection with golden metrics comparison
- Profiling best practices and bottleneck identification
- Before/after measurement and documentation standards

### [Scripts (scripts/)](../../scripts/)
**Automation tools for optimization workflows**

- [analyze_log.py](../../scripts/analyze_log.py) — Standalone log parser with golden comparison
- [optimize_and_validate.ps1](../../scripts/optimize_and_validate.ps1) — Automated test + validation wrapper
- Exit codes: 0=PASS, 1=FAIL, 2=WARN (performance regression), 3=ERROR

### [Current Logging Review](CURRENT_LOGGING_REVIEW.md)
**Analysis of existing logging patterns in `unified_pathfinder.py`**

- 287 logger calls analyzed and reclassified (134 debug, 61 info, 66 warning, 26 error)
- Existing tag system identified (`[ITER X]`, `[CONVERGENCE]`, `[GPU]`, etc.)
- Log bloat root cause fixed — console shows ~66 WARNING lines per run

---

## 🎯 Quick Start

### Optimization Workflow (Recommended)
```powershell
# 1. Quick smoke test validation (100 nets, <30s)
.\scripts\optimize_and_validate.ps1 -Compare tests/regression/smoke_metrics.json

# 2. Full backplane test with profiling (512 nets, 11-18 min)
.\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane -Compare tests/regression/golden_metrics.json

# 3. Analyze results
python scripts/analyze_log.py --compare tests/regression/golden_metrics.json
```

**See:** [optimization_workflow.md](optimization_workflow.md) for complete workflow guide

### Manual Baseline Test
```powershell
# Standard performance test
python main.py cli TestBoards/TestBackplane.kicad_pcb
```

### Analyze Performance
```powershell
# NEW: Standalone log parser (recommended)
python scripts/analyze_log.py
python scripts/analyze_log.py --compare tests/regression/golden_metrics.json

# Legacy manual analysis (for custom queries)
```
```python
import re, os
log = r"<plugin_dir>/logs/latest.log"
lines = open(log, encoding="utf-8", errors="ignore").readlines()
iters = [l for l in lines if re.search(r"WARNING.*\[ITER\s+\d+\]", l)]
print(f"Iters: {len(iters)}")
vals = [float(re.search(r": ([\d.]+)ms", l).group(1)) for l in lines if "[PROFILE]" in l and re.search(r": ([\d.]+)ms", l)]
print(f"Rebuild total: {sum(vals)/1000:.1f}s avg={sum(vals)/max(1,len(vals)):.0f}ms")
gpu = [float(re.search(r"\(([\d.]+)ms\)", l).group(1)) for l in lines if "Path found in" in l and "ms)" in l]
tot = [float(re.search(r"in ([\d.]+)s", l).group(1)) for l in lines if "SUCCESS! Path found in" in l and re.search(r"in ([\d.]+)s", l)]
print(f"GPU kernel: {sum(gpu)/max(1,len(gpu)):.1f}ms avg,  Total/net: {sum(tot)/max(1,len(tot))*1000:.0f}ms avg")
```

---

## 📈 Optimization Targets — Status Update

### Historical Progression

| Date | Total Time | Avg Iter | Target achieved | Notes |
|------|------------|----------|-----------------|-------|
| Apr 3 | 47.9 min | ~39s | Baseline | Multi-launch kernel |
| Apr 3 | 25 min | ~20s | **2× improvement** | Persistent kernel enabled |
| Apr 5 | **11.96 min** | **11.0s** | **4× improvement** ✅ | **Best performance** |
| Apr 8 | 17.5 min | 15.7s | ⚠️ **46% regression** | Regression investigation needed |

### Current Investigation Priorities (April 8, 2026)

| Priority | Target | Status | Action |
|----------|--------|--------|--------|
| 🔴 **#1** | Identify Apr 8 regression root cause | Unknown | Profile with `ORTHO_DEBUG=1` |
| 🔴 **#2** | Verify GPU persistent kernel usage | Unclear | Check actual kernel mode in logs |
| 🟡 **#3** | Measure debug screenshot overhead | ~5-10% | Test without `ORTHO_SCREENSHOT_FREQ` |
| 🟢 **#4** | Code diff analysis Apr 5 → Apr 8 | Pending | Git log review |

**Target**: Restore **11.96 min** performance from April 5 baseline

---

## 🚀 Optimization Phases

### Phase 1: Instrumentation ✅ COMPLETE
- ✅ `@profile_time` decorator in `shared/utils/performance_utils.py`
- ✅ Logging reclassified — console shows milestones (WARNING), file captures detail (DEBUG)
- ✅ GPU timing profiled — MULTI-LAUNCH overhead is the real bottleneck

### Phase 2: PERSISTENT KERNEL Activation 🔴 HIGH PRIORITY
**Target**: 44 min → 3-5 min (10-20× speedup)  
**File**: `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`
- Find why MULTI-LAUNCH is selected instead of PERSISTENT KERNEL
- Persistent kernel already compiled — routing selection logic needs fixing
- Port JITTER + ROUNDROBIN features if not yet in persistent variant
- Validate routing quality (zero overuse must be maintained)

### Phase 3: Graph Initialization
**Target**: 20.9s → 15s (30-40% improvement)
- Profile `initialize_graph()` sub-operations
- GPU-accelerated CSR construction
- Parallel edge array building

### Phase 4: Via Rebuild (Deferred)
**Status**: Incremental implementation deployed. 1% of runtime — not worth further work
until Phase 2 is resolved.

---

## 📚 Related Documentation

- [Algorithm Architecture](../algorithm_architecture.md): Deep dive into PathFinder algorithm
- [Performance Profiling Example](../performance_profiling_example.md): @profile_time decorator usage
- [Contributing Guide](../contributing.md): Development practices
- [Tuning Guide](../tuning_guide.md): Routing parameter optimization

---

## 🤖 Specialized Agents

Use these agents for optimization work:

```markdown
# GPU optimization
@GPU Performance Agent - analyze via rebuild performance with CuPy

# Safe refactoring
@Refactoring Agent - extract function from unified_pathfinder.py

# Architecture validation
@Architecture Reviewer - check dependencies in optimization branch

# Test creation
@Testing Agent - create unit tests for via pooling logic
```

---

## 📝 Contribution Guidelines

When adding optimization documentation:

1. **Name baselines with date**: `optimization_baseline_YYYY-MM-DD.md`
2. **Include test board details**: Board name, complexity, routing results
3. **Provide reproducible commands**: PowerShell/bash commands for verification
4. **Document before/after metrics**: Always compare to established baseline
5. **Link to related PRs/commits**: Connect documentation to code changes

---

## 🔍 Archive

As new baselines are established, previous baselines will be moved to an `archive/` subdirectory but kept for historical comparison.

**Current Active Baseline**: April 3, 2026 (TestBackplane, v0.2.0)

---

**Last Updated**: April 3, 2026  
**Maintainer**: Performance Optimization Team
