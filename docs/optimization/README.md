# OrthoRoute Performance Optimization

This folder contains all performance optimization documentation, baselines, and migration guides for OrthoRoute.

---

## 📊 Performance Baselines

### [Baseline: April 3, 2026](optimization_baseline_2026-04-03.md)
**TestBackplane routing analysis (512 nets, 32 layers, 44.23 minutes)**

Complete performance baseline with:
- Current performance measurements
- Profiled bottlenecks and priorities
- Phase-by-phase optimization roadmap
- Success metrics and validation criteria
- Quick start commands for optimization work

**Quick Reference**: [OPTIMIZATION_QUICK_REF.md](OPTIMIZATION_QUICK_REF.md) - Single-page cheat sheet

---

## 🔧 Optimization Tools & Systems

### [Current Logging Review](CURRENT_LOGGING_REVIEW.md)
**Analysis of existing logging patterns in `unified_pathfinder.py`**

- 287 logger calls analyzed and reclassified (134 debug, 61 info, 66 warning, 26 error)
- Existing tag system identified (`[ITER X]`, `[CONVERGENCE]`, `[GPU]`, etc.)
- Log bloat root cause fixed — console shows ~66 WARNING lines per run

---

## 🎯 Quick Start

### Run Baseline Test
```powershell
# Standard performance test
python main.py cli TestBoards/TestBackplane.kicad_pcb
```

### Analyze Performance
```python
# Use Python — avoids PowerShell multi-line paste issues with large logs
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

## 📈 Optimization Targets (Revised — Live Run April 3, 2026)

> Priority #1 was corrected after profiling 40 iterations and 3,311 GPU paths.
> The via rebuild is only 1% of runtime. The Python→CUDA kernel launch loop is 95%.

| Priority | Target | Measured Cost | % of Total | Est. Impact |
|----------|--------|--------------|------------|-------------|
| 🔴 **#1** | GPU MULTI-LAUNCH overhead (Python loop per net) | ~980s/run | **~95%** | **10-20× speedup** |
| 🟡 **#2** | `initialize_graph()` | 20.9s once | one-time | **6-8s** |
| 🟢 **#3** | `_rebuild_via_usage_from_committed()` | ~11s/run | **~1%** | deferred |

**Revised Target**: 44 min → **3-5 min** (10-20× faster, conditional on persistent kernel)

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
