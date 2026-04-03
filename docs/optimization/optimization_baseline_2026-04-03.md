# OrthoRoute Performance Baseline & Optimization Roadmap

**Date**: April 3, 2026  
**Version**: OrthoRoute v0.2.0  
**Test Board**: TestBackplane.kicad_pcb  
**Purpose**: Establish performance baseline for future optimization work

---

## Executive Summary

Successfully routed a **18-layer backplane** with **512 nets**, **1,604 pads** in **44.23 minutes** initially, reduced to **25 minutes** after persistent kernel optimization.

**Run history:**

| Date | Change | Time | Iter | Result |
|------|--------|------|------|--------|
| Apr 3 (baseline) | Multi-launch kernel | 47.9 min | 74 | 512/512 ✓ |
| Apr 3 (persistent kernel) | Persistent CUDA kernel + bitmap fix | **25 min** | 70 | 512/512 ✓ |

**Current bottleneck**: Python bitmap construction — `int(seed)` loop in `find_path_fullgraph_gpu_seeds()` triggers ~100 GPU→CPU syncs per net, dominating per-net time at ~50ms vs 3–24ms kernel GPU time. See [OPTIMIZATION_QUICK_REF.md](OPTIMIZATION_QUICK_REF.md) for the vectorized fix.

**Initial Finding (corrected)**: `_rebuild_via_usage_from_committed()` was believed to be the #1 target.
**Live-run correction (April 3, 2026)**: After profiling 40 iterations and 3,311 GPU paths, the via rebuild consumes **~1% of routing time**. The real bottleneck was the **MULTI-LAUNCH Python→CUDA loop** — now replaced by the persistent kernel. New bottleneck is Python bitmap construction (~50ms/net).

---

## Test Board Characteristics

### Board Complexity
- **Layers**: 32 copper layers
- **Nets**: 512 nets to route
- **Pads**: 3,200 pads
- **Via Pairs**: 870 via pairs
- **Design Type**: High-density backplane

### Routing Environment
- **Grid Resolution**: Manhattan lattice (configured in `orthoroute.json`)
- **Keepout Areas**: Multiple rule areas with track/via constraints
- **Algorithm**: PathFinder negotiated congestion routing
- **Convergence**: 64 iterations to zero overuse

### Final Results
- **Success Rate**: 512/512 nets routed (100%)
- **Overuse**: Zero final overuse
- **Barrel Conflicts**: 310 remaining (acceptable)
- **Output**: 4,118 tracks + 2,585 vias

---

## Performance Baseline (Total: 44.23 minutes)

### Phase Breakdown

#### 1. Initialization Phase (~25 seconds)
| Operation | Time | Frequency | Notes |
|-----------|------|-----------|-------|
| `initialize_graph` | 20.9s | Once | CSR graph construction |
| `finalize` | 3.95s | Once | Route commitment |
| `_apply_keepout_obstacles` | 0.15ms | Once | Keepout enforcement (highly optimized) |

#### 2. Routing Phase (~39 minutes for 64 iterations)
**Per-Iteration Costs**:

| Operation | Count | Min | Max | Avg | Total | % of Routing |
|-----------|-------|-----|-----|-----|-------|--------------|
| `_rebuild_via_usage_from_committed` | 64 | 0.37ms | 485ms | 347ms | 22.2s | ~13.9% |
| `_apply_via_pooling_penalties` | 64 | 2.58ms | 101ms | 25.5ms | 1.6s | ~1.0% |

**Note**: The above percentages are conservative estimates based on profiled operations. The majority of routing time (85%+) is spent in **unprofiled PathFinder algorithm code** within `unified_pathfinder.py`.

#### 3. Visualization & Output (~4 minutes)
- **Iteration Visualizations**: 66 PNG files generated (debug_output/)
- **KiCad Export**: 4,118 tracks + 2,585 vias written to board
- **Commit to KiCad**: ~40 seconds

---

## Performance Bottlenecks (Prioritized)

> Updated April 3, 2026 — persistent kernel active.

| Priority | Target | Measured Cost | % of per-net | Status |
|----------|--------|--------------|--------------|--------|
| ✅ **DONE** | MULTI-LAUNCH Python→CUDA overhead | ~980s / run | was ~95% | **Fixed — persistent kernel** |
| 🔴 **#1 (new)** | Python bitmap construction (int-loop GPU→CPU syncs) | ~50ms/net | **~65%** | Vectorize with `cp.scatter_add` |
| 🟡 **#2** | GPU→CPU convergence check | ~10ms/net | **~13%** | Check less frequently or on-device |
| 🟡 **#3** | `initialize_graph()` | 20.9s once | one-time | GPU-accelerated CSR build |
| 🟢 **#4 (was #1)** | `_rebuild_via_usage_from_committed()` | ~11s / run | **~1%** | Already incremental; low priority |

---

### 🔴 Priority 1 (NEW): GPU MULTI-LAUNCH Overhead

**Measured (40 iterations, 3,311 paths):**

```
GPU hardware:    RTX Turing (compute capability 75)
                 4.3 GB total VRAM / 3.0 GB free at start
                 14,281,664 edges, 446,472 nodes

Per-net timing:
  GPU Dijkstra kernel (MULTI-LAUNCH loop):   ~16ms   (5%)
  Python ↔ CUDA launch overhead:            ~296ms  (95%)
  Total per net:                             ~312ms

Projection (full run):
  3,311 paths × 312ms = 1,032s routing time (so far, 40 iters)
  If persistent kernel: 3,311 × 16ms = ~53s  → 20× speedup
```

**Root cause**: The MULTI-LAUNCH model runs a Python `for` loop (~150 iterations per net)
where each loop iteration launches a CUDA kernel and syncs back to Python to check
termination. This creates ~150 round-trips per net between CPU and GPU.

The PERSISTENT KERNEL is already compiled (confirmed from logs):
```
[CUDA] Compiled PERSISTENT KERNEL (P1-6: device-side queues, eliminates launch overhead!)
```
But every routing call logs:
```
[GPU-SEEDS] Using MULTI-LAUNCH kernel (Python loop)
```
meaning the selection logic never takes the persistent path.

**How to fix** (`cuda_dijkstra.py`):
1. Find the condition that selects between MULTI-LAUNCH and PERSISTENT KERNEL
2. Determine why the persistent path is not activated (flag not set? size threshold? bug?)
3. Wire in the persistent kernel for the main routing loop
4. Validate: routing quality must be identical, per-net total time should drop from ~312ms to ~30-50ms

**Estimated impact**: **10-20× routing speedup**; full run 44 min → 3-5 min

---

### 🟡 Priority 2: Graph Initialization (20.9s one-time)
**Function**: `initialize_graph()`  
**Impact**: Medium - one-time cost but significant for large boards  
**Current**: CSR sparse matrix construction

**Optimization Opportunities**:
1. **GPU Acceleration**: Check if CSR construction can use CuPy
2. **Parallel Construction**: Build edge arrays in parallel
3. **Memory Allocation**: Pre-allocate arrays if sizes are known
4. **Profiling Needed**: Break down 20s into sub-operations

**Estimated Impact**: 30-40% reduction → **~6-8s savings**

---

### 🟢 Priority 3: Via Usage Rebuild (~11s total, was Priority 1)
**Function**: `_rebuild_via_usage_from_committed()`  
**Revised assessment**: After live profiling, this is ~1% of runtime — not worth further
optimization until the kernel overhead (#1) is resolved.

**Current state (April 3, 2026)**:
- Incremental implementation deployed (40% threshold to full rebuild)
- 40 iters measured: 28 calls, 275ms avg, 11s total
- Full rebuild fires nearly every iteration (dirty set > threshold in early convergence)
- Incremental path activates only after routing stabilizes (~iter 30+)

**Action**: Defer. Revisit after persistent kernel is activated.

---

### ⚪ Not a Bottleneck: Keepout Obstacles (0.15ms)
**Status**: ✅ Already highly optimized (vectorized NumPy ray-casting)  
**Action**: No optimization needed

---

### 🟢 Priority 3: Finalization (3.95s one-time)
**Function**: `finalize()`  
**Impact**: Low - one-time cost, smaller magnitude  

**Optimization Opportunities**:
1. **Analyze what finalize() does** - may already be optimized
2. **Lower priority than iterative operations**

**Estimated Impact**: 25-30% reduction → **~1s savings**

---

### ⚪ Not a Bottleneck: Keepout Obstacles (0.15ms)
**Function**: `_apply_keepout_obstacles()`  
**Status**: ✅ Already highly optimized (vectorized NumPy ray-casting)  
**Action**: No optimization needed

### ⚪ Not a Bottleneck: Finalization (3.95s one-time)
**Function**: `finalize()`  
**Impact**: Low — one-time cost, smaller magnitude. Defer.

---

## GPU Performance Analysis — Live Run (April 3, 2026)

Collected from a run started at 15:04:39, observed through 40 completed iterations (3,311 paths routed, log ~612 MB, run still in progress at time of writing).

### Hardware
- GPU: RTX (CUDA compute capability 75 = Turing)
- VRAM: 4.3 GB total, 3.0 GB free at start
- Graph size: 446,472 nodes, 14,281,664 edges
- Graph sort: 1.0s (GPU radix sort, one-time, 14M edges/sec)

### Compiled Kernels (all confirmed present in logs)
| Kernel | Description |
|--------|-------------|
| ACTIVE-LIST | Sparse wavefront — "446,472× fewer memory accesses" |
| MULTI-LAUNCH | Python loop drives wavefront — **current bottleneck** |
| PERSISTENT KERNEL | Device-side queues, eliminates Python launch loop — **compiled, not activated** |
| COMPACTION | GPU-side frontier compaction, no host sync |
| ACCOUNTANT | GPU-side history/present/cost updates |
| GPU BACKTRACE | Eliminates 256 MB parent/dist CPU transfer |
| RR-WAVEFRONT | Round-robin layer preference |
| DELTA-STEPPING | Bucket assignment |

### Timing Measurements (3,311 paths)
| Metric | Value |
|--------|-------|
| GPU kernel time (CUDA Dijkstra only) | 15.8ms avg |
| Total per-net time (kernel + Python overhead) | 312ms avg |
| Python/CUDA launch overhead | **~296ms per net (95%)** |
| Total routing time (40 iters) | 1,032s |
| Via rebuild (`_rebuild_via_usage_from_committed`) | 275ms avg / 11s total (1%) |

### What 296ms of overhead consists of (inferred)
The `[GPU-SEEDS] Using MULTI-LAUNCH kernel (Python loop)` log line appears for every path.
The Python loop runs ~150 iterations per net (seen: 121–161 iters), and each iteration:
1. Calls `rr_wavefront_kernel` (JITTER + ROUNDROBIN parameters built in Python)
2. Reads frontier size back to CPU to check termination
3. Launches ACTIVE-LIST kernel with updated params

Each CPU↔GPU round-trip adds ~1-2ms latency on PCIe, giving 150 × ~2ms ≈ 300ms.

### The PERSISTENT KERNEL path
The persistent kernel (`[CUDA] Compiled PERSISTENT KERNEL (P1-6: device-side queues,
eliminates launch overhead!)`) stores the work queue on-device. Python launches it once,
the kernel loops internally until convergence or timeout, then returns. This replaces the
150-iteration Python loop with a single launch.

**Expected result**: per-net time drops from ~312ms → ~30-50ms (kernel time + one-time setup).

### Next step
In `cuda_dijkstra.py`, find the condition:
```python
logger.info("[GPU-SEEDS] Using MULTI-LAUNCH kernel (Python loop)")
```
and understand why the persistent kernel branch is not taken. The fix is likely a flag,
a size threshold, or a missing feature flag (e.g., `ROUNDROBIN` or `JITTER` not yet
ported to the persistent kernel variant).

---

## Unprofiled Algorithm Code (Largely Resolved)

**Update (April 3, 2026)**: The "unprofiled 2,316s" mystery is now explained.
95% of routing time is the MULTI-LAUNCH Python→CUDA overhead (see GPU analysis above).
The algorithm itself (Dijkstra wavefront) takes ~16ms per path on GPU — it is fast.
There is no hidden algorithm bottleneck to uncover with more `@profile_time` decorators.

**Remaining unknowns**: The `_route_all` batch dispatch loop and ROI construction overhead.
These are likely small (<5% each) relative to the kernel overhead, but can be confirmed
by adding `@profile_time` to `_route_all` and `_build_roi`.

**Action Required (revised)**: Instead of adding profile decorators to routing functions,
focus on activating the PERSISTENT KERNEL in `cuda_dijkstra.py`.

---

## Log File Analysis

- **Log Size (baseline)**: 799.9 MB for 44-minute run
- **Root Cause** (fixed April 3, 2026): 160 `logger.info` calls with no verbosity gate
- **Fix applied**: 18 INFO→WARNING, 81 INFO→DEBUG, 1 WARNING→DEBUG — 100 calls reclassified
- **Expected**: ~25-40 MB after re-test (re-run needed to confirm)
- See [CURRENT_LOGGING_REVIEW.md](CURRENT_LOGGING_REVIEW.md) for full analysis

---

## Routing Algorithm Behavior (Observed)

### Convergence Pattern
- **Total Iterations**: 64
- **Strategy**: PathFinder negotiated congestion
- **Exclusion/Retry**: Algorithm temporarily excludes difficult nets, retries every 10 iterations
- **Final State**: Zero overuse after 64 iterations

### Temporary Net Exclusions (Normal Behavior)
The algorithm correctly uses exclusion/retry strategy:
- Iteration 26: Excluded 1 net (`B03B07_005`) after 5 failed attempts
- Iteration 30: **RETRY** - gave 3 excluded nets another chance
- Iteration 34: Excluded 2 nets (`B00B04_010`, `B01B03_002`)
- Iteration 40: **RETRY** - gave 2 excluded nets another chance
- Iteration 45: Excluded 3 nets
- Iteration 50: **RETRY** - gave 3 excluded nets another chance
- **Final Result**: All 512 nets successfully routed

**This is correct behavior** - not errors or failures.

---

## System Health Status ✅

### Error Analysis
- **Errors Found**: 0
- **Critical Issues**: 0
- **Exceptions**: 0
- **Traceback**: 0

### Warnings Analysis
- **Total Warnings**: 12 (all temporary net exclusions - expected)
- **Severity**: Low (normal PathFinder algorithm behavior)

### Routing Quality
- **Completion**: 512/512 nets (100%)
- **Overuse**: 0 (perfect convergence)
- **Barrel Conflicts**: 310 (acceptable)
- **Keepout Violations**: 0 (enforced correctly)

**Conclusion**: System is stable and healthy. Ready for optimization work.

---

## Optimization Strategy & Roadmap

### Phase 1: Instrumentation (COMPLETE)
1. ✅ **Logging reclassification** — 100 calls reclassified (18 INFO→WARNING, 81 INFO→DEBUG)
2. ✅ **`@profile_time` decorator** — added and exported from `shared/utils/performance_utils.py`
3. ✅ **`_rebuild_via_usage_from_committed` profiled** — confirmed 1% of runtime
4. ✅ **GPU timing profiled** — MULTI-LAUNCH overhead identified as #1 bottleneck

---

### Phase 2: PERSISTENT KERNEL Activation (HIGH PRIORITY)
**Goal**: Eliminate Python→CUDA launch loop overhead (95% of routing time)  
**Target file**: `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`

1. ⬜ **Find kernel selection logic** — search for `MULTI-LAUNCH` selection condition
2. ⬜ **Identify why persistent path is skipped** — flag? threshold? missing JITTER/RR support?
3. ⬜ **Port JITTER + ROUNDROBIN to persistent kernel** (if not already done)
4. ⬜ **Activate persistent kernel** for ROI ≥ some node threshold
5. ⬜ **Validate routing quality** — zero overuse must be maintained
6. ⬜ **Benchmark** — expect per-net time to drop from ~312ms → ~30-50ms

**Target**: 44 min → 3-5 min (**10-20× speedup**)

---

### Phase 3: Graph Initialization Optimization
**Goal**: Reduce 20.9s to ~13-15s (30-40% improvement)

1. ⬜ **Profile `initialize_graph()` sub-operations**
2. ⬜ **Investigate GPU-accelerated CSR construction**
3. ⬜ **Parallel edge array building**
4. ⬜ **Memory pre-allocation**

**Target**: 6-8s savings (only meaningful once kernel overhead is resolved)

---

### Phase 4: Via Rebuild (Deferred)
**Status**: Incremental implementation already deployed (see `_rebuild_via_usage_from_committed`).
At 1% of runtime, further optimization is not worthwhile until Phase 2 is complete.

---

### Success Metrics (Revised)

**Performance Targets**:
- **Phase 2** (persistent kernel): 44 min → **3-5 min** (10-20× faster)
- **Phase 3** (init): no change to routing time, saves ~6-8s startup
- **Phase 4** (via rebuild): deferred

**Quality Requirements** (Must Maintain):
- 100% routing success rate (512/512 nets)
- Zero final overuse
- All keepout constraints enforced
- Correct barrel conflict detection

---

## Test Environment

### Hardware
- **OS**: Windows (OneDrive-enabled)
- **Python**: 3.11+
- **GPU**: NVIDIA with CUDA (CuPy available)
- **Memory**: Sufficient for 799MB logs + GPU operations

### Software Stack
- **OrthoRoute**: v0.2.0
- **KiCad**: 9.0+ (IPC API enabled)
- **Dependencies**: NumPy, CuPy, PyQt6
- **Profiling**: `@profile_time` decorator (performance_utils.py)

### Development Tools
- **VS Code**: Dual workspace (source + deployed plugin)
- **Specialized Agents**:
  - Refactoring Agent (`.github/agents/refactoring.agent.md`)
  - GPU Performance Agent (`.github/agents/gpu-performance.agent.md`)
  - Architecture Reviewer (`.github/agents/architecture-review.agent.md`)
  - Testing Agent (`.github/agents/testing.agent.md`)

---

## Quick Start for Optimization Work

### 1. Run with DEBUG_LEVEL=4
```powershell
# Set environment variable
$env:ORTHOROUTE_DEBUG_LEVEL=4

# Run plugin with detailed logging
python main.py plugin --no-gui
```

### 2. Analyze New Profile Data
```powershell
# Extract profile timings
Get-Content logs/latest.log | Select-String -Pattern "\[PROFILE\]"

# Count iterations
Get-Content logs/latest.log | Select-String -Pattern "\[ITER" | Measure-Object
```

### 3. Compare Performance
```powershell
# Log file size
(Get-Item logs/latest.log).Length / 1MB

# Total execution time (check log timestamps)
```

### 4. Use Specialized Agents
```markdown
# For algorithm optimization
@GPU Performance Agent - analyze via rebuild performance

# For refactoring unified_pathfinder.py
@Refactoring Agent - extract function X from unified_pathfinder

# For validation
@Architecture Reviewer - check dependencies in optimization branch

# For testing
@Testing Agent - create characterization tests for via rebuild
```

---

## References

### Documentation
- [Algorithm Architecture](../algorithm_architecture.md) - Deep dive into PathFinder
- [DEBUG_LEVEL System Guide](debug_level_system.md) - Complete logging guide
- [Performance Profiling Example](../performance_profiling_example.md) - @profile_time usage
- [Tuning Guide](../tuning_guide.md) - Parameter optimization

### Code Files
- **Main Router**: `orthoroute/algorithms/manhattan/unified_pathfinder.py` (3,936 lines)
- **Profiling Utils**: `orthoroute/shared/utils/performance_utils.py`
- **DEBUG_LEVEL Core**: `orthoroute/shared/utils/debug_levels.py`
- **Migration Examples**: `docs/optimization/examples/debug_level_migration_unified_pathfinder.py`

### Test Data
- **Test Board**: `TestBoards/TestBackplane.kicad_pcb`
- **Baseline Log**: `logs/run_20260403_100453.log` (799.9 MB)
- **Iteration Visualizations**: `debug_output/run_20260403_100514/` (66 PNGs)
- **KiCad Export**: `debug_output/kicad_export_20260403_104820.json`

---

## Notes & Observations

### Design Complexity is Adequate
The TestBackplane board with **512 nets** and **32 layers** provides excellent test coverage:
- Complex enough to show real performance issues
- Large enough for meaningful optimization gains
- Diverse net topology (short/long, simple/complex)
- Sufficient iteration count (64) for statistical analysis

### GPU Utilization Unknown
- Profile data doesn't indicate CPU vs GPU execution
- Need to verify which operations are GPU-accelerated
- Consult GPU Performance Agent for optimization guidance

### Log Verbosity is Critical
- 799 MB logs make analysis impossible
- DEBUG_LEVEL system implementation was essential
- Future optimization work depends on Level 4 logging

### PathFinder Algorithm is Mature
- Zero errors in complex routing proves stability
- Exclusion/retry strategy works correctly
- Ready for performance optimization without correctness concerns

---

## Baseline Established: Ready for Optimization ✅

This document establishes the performance baseline for OrthoRoute on a production-representative test case. All measurements are reproducible, system health is confirmed, and optimization targets are identified.

**Next Immediate Action**: Migrate `unified_pathfinder.py` to DEBUG_LEVEL system to enable detailed algorithm profiling.

---

**Document Version**: 1.0  
**Last Updated**: April 3, 2026  
**Author**: Performance Analysis  
**Status**: Baseline Established
