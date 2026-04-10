# OrthoRoute Performance Baseline — April 8, 2026

**Date**: April 8, 2026  
**Version**: OrthoRoute (current main branch)  
**Test Board**: TestBackplane.kicad_pcb  
**Purpose**: Track routing performance and identify regression vs. April 5 baseline

---

## Executive Summary

Successfully routed an **18-layer backplane** with **512 nets**, **1,604 pads** in **17.5 minutes (1,049.8s)** — a **46% regression vs. April 5 baseline (11.96 min)**.

**Run history (all documented runs):**

| Run | Date | Change | Total Time | Iter avg | Iters | Tracks | Vias | Barrel | Status |
|-----|------|--------|------------|----------|-------|--------|------|--------|--------|
| 1 | Apr 3 | Baseline — multi-launch kernel | 47.9 min | ~39s | 74 | 4,118 | 2,585 | — | ✓ |
| 2 | Apr 3 | Persistent CUDA kernel enabled | 25 min | ~20s | 70 | 4,118 | 2,585 | — | ✓ 2× gain |
| 3 | Apr 5 | **Best performance** | **11.96 min** | **11.0s** | **65** | **4,290** | **2,754** | **444** | **✓ 4× baseline** |
| 4 | Apr 8 | Current state | 17.5 min | 15.7s | 67 | 4,224 | 2,688 | 359 | ✓ **⚠️ 46% regression** |

**Primary concern**: Per-iteration time increased from 11.0s (Apr 5) to 15.7s (Apr 8), indicating possible algorithm or GPU kernel regression.

---

## Test Board Characteristics

| Property | Value |
|----------|-------|
| **Board file** | TestBackplane.kicad_pcb |
| **Copper layers** | 18 (F.Cu + 16 internal + B.Cu) |
| **Pads** | 1,604 |
| **Routable nets** | 512 |
| **Total nets** | 1,088 |
| **Components** | 12 footprints |
| **Board size** | 73.1 × 97.3 mm |
| **Pre-existing tracks** | 9,605 |
| **Pre-existing vias** | 6,021 |
| **Lattice nodes** | ~446,472 |
| **Lattice edges** | ~14,281,664 |

---

## Routing Run — Measured Results (April 8, 2026)

**Hardware**: NVIDIA GPU (CUDA-capable, exact model not captured in log)  
**Log file**: `com_github_bbenchoff_orthoroute/logs/latest.log` (April 8, 2026, 6:23 PM - 6:48 PM)  
**Log size**: 65,264 lines (~9.8 MB)  
**Debug mode**: OFF (no `[PROFILE]` timing data — requires `ORTHO_DEBUG=1`)

### Final Metrics

| Metric | Value |
|--------|-------|
| **Success rate** | 512/512 nets (100%) |
| **Total routing time** | 1,049.8s (17.5 minutes) |
| **Iterations** | 67 |
| **Iteration 1 time** | ~29s (estimated from iter 2 start time) |
| **Average iteration time** | 15.7s |
| **Fastest iteration** | 3.6s (iter 66) |
| **Slowest iteration** | ~29s (iter 1 — includes graph init) |
| **Final edge overuse** | 0 ✓ (fully converged) |
| **Barrel conflicts** | 359 (acceptable) |
| **Tracks written** | 4,224 (2,048 escapes + 2,176 routed) |
| **Vias written** | 2,688 |

### Convergence Trend (April 8, 2026 run)

| Iter | Nets active | Overuse (edges) | Barrel | Iter time | Total time | Notes |
|------|-------------|-----------------|--------|-----------|------------|-------|
| 1 | — | — | — | ~29s | ~29s | Graph init + first routing pass |
| 48 | 491/512 | 40,767 | 416 | 7.7s | 801.6s | Still high overuse mid-convergence |
| 49 | 495/512 | 47,586 | 484 | 7.8s | 816.6s | Overuse spike (normal oscillation) |
| 50 | 499/512 | 50,161 | 517 | 8.7s | 832.2s | Peak overuse during convergence |
| 51 | 504/512 | 1,515 | 517 | 8.3s | 849.6s | Sharp drop in overuse |
| 52 | 507/512 | 1,292 | 517 | 8.6s | 865.1s | Continued improvement |
| 53 | 511/512 | 1,545 | 517 | 9.0s | 880.9s | Minor oscillation |
| 54 | 511/512 | 1,523 | 517 | 8.3s | 896.2s | |
| 55 | 511/512 | 1,061 | 517 | 7.0s | 909.5s | |
| 56 | 511/512 | 864 | 517 | 5.8s | 921.6s | |
| 57 | 511/512 | 789 | 517 | 5.7s | 935.1s | |
| 58 | 511/512 | 626 | 517 | 5.8s | 948.3s | |
| 59 | 511/512 | 341 | 517 | 5.3s | 960.6s | |
| 60 | 512/512 | 173 | 359 | 4.7s | 972.1s | All nets active |
| 61 | 512/512 | 44 | 359 | 4.2s | 985.0s | Near convergence |
| 62 | 512/512 | 73 | 359 | 4.3s | 995.7s | Minor oscillation |
| 63 | 512/512 | 2 | 359 | 4.0s | 1006.7s | Almost converged |
| 64 | 512/512 | 2 | 359 | 3.7s | 1017.1s | |
| 65 | 512/512 | 21 | 359 | 3.8s | 1027.3s | |
| 66 | 512/512 | 26 | 359 | 3.6s | 1038.4s | |
| **67** | **512/512** | **0** ✓ | **359** | **3.9s** | **1,049.8s** | **CONVERGED** |

**Key observations:**
- Iteration 50 shows peak overuse (50,161) before rapid convergence
- Barrel conflicts stabilized at 359 by iteration 60
- Final iterations (60-67) averaged ~4s as overuse approached zero
- All 512 nets successfully routed with zero edge overuse

---

## GPU Performance

**Configuration (from log):**
```
[CONFIG] use_gpu=True
[CONFIG] use_gpu_sequential=True
[CONFIG] use_incremental_cost_update=False
PathFinder (GPU=True, Portals=True)
```

**GPU operations logged:**
- ✅ CUDA GPU provider initialized
- ✅ Persistent kernel compiled successfully (`[GPU-SEEDS] Persistent kernel compiled successfully!`)
- ✅ Via kernels compiled (`[VIA-KERNELS] CUDA kernels compiled successfully`)
- ✅ GPU radix sort: 0.9s for 16.2M edges (16.2M edges/sec)
- ✅ Via column pooling enabled (capacity=4 per x,y location)

**Average pathfinding times (from log samples):**
- [GPU-SEEDS] SUCCESS: ~70-90ms per net typical
- Range: 20ms (simple nets) to 190ms (complex nets)

**No GPU fallbacks to CPU reported** — all routing used GPU acceleration.

---

## Performance Regression Analysis

### Comparison vs. April 5 Baseline

| Metric | Apr 5 (best) | Apr 8 (current) | Delta | % Change |
|--------|--------------|-----------------|-------|----------|
| **Total time** | 11.96 min (717.5s) | 17.5 min (1,049.8s) | **+332.3s** | **+46%** ⚠️ |
| **Iterations** | 65 | 67 | +2 | +3% |
| **Avg iter time** | 11.0s | 15.7s | **+4.7s** | **+43%** ⚠️ |
| **Iter 1 time** | 29.2s | ~29s | ~0s | ~0% |
| **Final overuse** | 0 | 0 | 0 | = |
| **Tracks** | 4,290 | 4,224 | -66 | -1.5% |
| **Vias** | 2,754 | 2,688 | -66 | -2.4% |
| **Barrel conflicts** | 444 | 359 | -85 | -19% ✓ |
| **Success rate** | 100% | 100% | = | = |

### Root Cause Analysis

**Primary regression**: Per-iteration time increased by **43%** (11.0s → 15.7s)

**Possible causes:**
1. **GPU kernel mode**: April 5 used "PERSISTENT kernel" — need to verify April 8 is using the same mode (log confirms compiled, but check actual usage pattern)
2. **Algorithm changes**: Code changes between April 5 and April 8 may have added overhead
3. **Parameter differences**: Different PathFinder parameters (pres_fac_mult, hist_gain) could affect iteration count/time
4. **Hardware differences**: Different GPU model or available VRAM could impact performance
5. **Debug overhead**: April 8 generated 69 debug screenshots — visual debugging active

### Positive changes:
- ✅ Barrel conflicts reduced: 444 → 359 (-19%)
- ✅ Still 100% routing success
- ✅ Still achieves zero overuse convergence

---

## Routing Strategy

**Configuration (from log):**
```
STRATEGY: SPARSE (fast convergence)
Convergence: max_iters=250, patience=5
```

**Congestion ratio**: Not captured in log without `ORTHO_DEBUG=1`

---

## Debug Output

**Location**: `com_github_bbenchoff_orthoroute/debug_output/run_20260408_182319/`  
**Files**: 69 screenshots

**Screenshot sequence:**
1. `01_board_with_airwires` — Initial board state
2. `02_board_no_airwires` — Board without airwires
3. `03_board_with_escapes` — Portal escape vias placed
4. `04_iteration_01` through `70_iteration_67` — Per-iteration routing progress

---

## Recommendations

### Immediate Actions

1. **Enable profiling** — Run with `ORTHO_DEBUG=1` to capture `[PROFILE]` timing data
   ```powershell
   $env:ORTHO_DEBUG = '1'
   # Re-run routing
   Remove-Item Env:ORTHO_DEBUG
   ```

2. **Compare GPU kernel modes** — Check if persistent kernel is actually being used:
   ```python
   # In logs, search for:
   # "[GPU] Using PERSISTENT kernel" vs "[GPU] Using MULTI-LAUNCH kernel"
   ```

3. **Disable debug screenshots** — Test without visual debugging overhead:
   ```powershell
   Remove-Item Env:ORTHO_SCREENSHOT_FREQ, Env:ORTHO_SCREENSHOT_SCALE -ErrorAction SilentlyContinue
   ```

4. **Git diff analysis** — Compare code changes between April 5 and April 8:
   ```bash
   git log --oneline --since="2026-04-05" --until="2026-04-08"
   git diff <apr5-commit> <apr8-commit> -- orthoroute/algorithms/manhattan/
   ```

### Investigation Priorities

| Priority | Investigation | Expected impact |
|----------|--------------|-----------------|
| 🔴 **#1** | Verify GPU kernel mode (persistent vs multi-launch) | Could explain entire 43% regression |
| 🟡 **#2** | Measure debug screenshot overhead | ~5-10% overhead possible |
| 🟡 **#3** | Profile per-iteration breakdown with `ORTHO_DEBUG=1` | Identify specific bottleneck |
| 🟢 **#4** | Parameter tuning (pres_fac_mult, hist_gain) | Optimize iteration count |

---

## Regression Test Thresholds

**Current thresholds** (from April 5 golden metrics):
```json
{
  "total_time_s_max": 900,        // Apr 5: 717.5s × 1.25
  "iter_avg_time_s_max": 15.0,    // Apr 5: 11.0s × 1.36
  "iterations_max": 80,            // Apr 5: 65 × 1.23
  "overuse_final_max": 0,
  "converged": true
}
```

**April 8 performance vs. thresholds:**
- ✅ `total_time_s`: 1,049.8s < 900s **FAIL** (exceeds threshold by 17%)
- ✅ `iter_avg_time_s`: 15.7s > 15.0s **FAIL** (exceeds threshold by 5%)
- ✅ `iterations`: 67 < 80 **PASS**
- ✅ `overuse_final`: 0 = 0 **PASS**
- ✅ `converged`: true **PASS**

**Action required**: Investigate regression or update thresholds if performance change is intentional.

---

## Conclusions

**Status**: ⚠️ **Performance regression detected**

**Summary:**
- ✅ Routing quality: Excellent (100% success, zero overuse, fewer barrel conflicts)
- ⚠️ Routing speed: Regressed by 46% vs. best baseline (April 5)
- ❓ Root cause: Unknown — requires profiling to isolate

**Next steps:**
1. Run with `ORTHO_DEBUG=1` profiling enabled
2. Verify GPU persistent kernel is active
3. Compare against April 5 codebase
4. Measure debug screenshot overhead
5. Update baseline if regression is intentional/acceptable, or fix if unintentional

---

## Appendix: Raw Log Excerpts

### Initialization
```
2026-04-08 18:23:12,697 - root - ERROR - [LOG] File: DEBUG | logs/latest.log + logs/run_20260408_182312.log
2026-04-08 18:23:15,002 - orthoroute.algorithms.manhattan.unified_pathfinder - WARNING - PathFinder (GPU=True, Portals=True)
2026-04-08 18:23:16,645 - orthoroute.infrastructure.kicad.rich_kicad_interface - INFO - Got layer count from BoardStackup.material_name: 18 copper layers
2026-04-08 18:23:50,253 - orthoroute.algorithms.manhattan.parameter_derivation - INFO - STRATEGY: SPARSE (fast convergence)
```

### Convergence
```
2026-04-08 18:41:20,396 - orthoroute.algorithms.manhattan.unified_pathfinder - WARNING - [ITER  67] nets=512/512  ✓ CONVERGED  edges=0  via_overuse=0%  barrel=359  iter=3.9s  total=1049.8s
2026-04-08 18:41:20,396 - orthoroute.algorithms.manhattan.unified_pathfinder - WARNING - [CLEAN] All nets routed with zero overuse
2026-04-08 18:41:20,396 - orthoroute.algorithms.manhattan.unified_pathfinder - WARNING - ROUTING COMPLETE: All 512 nets routed successfully with zero overuse!
```

### Final Geometry
```
2026-04-08 18:41:20,557 - orthoroute.algorithms.manhattan.unified_pathfinder - INFO - [ESCAPE-MERGE] escapes=2048 + routed=2176 → total=4224 tracks after dedup
2026-04-08 18:41:20,557 - orthoroute.algorithms.manhattan.unified_pathfinder - INFO - [ESCAPE-MERGE] escape_vias=0 + routed_vias=2688 → total=2688 vias after dedup
```
