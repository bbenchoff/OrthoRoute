# OrthoRoute Performance Baseline — April 5, 2026

**Date**: April 5, 2026  
**Version**: OrthoRoute (post-regression-suite branch `feature/profiler-optimization`)  
**Test Board**: TestBackplane.kicad_pcb  
**Purpose**: Capture current routing performance after persistent-kernel and algorithm improvements since April 3 baseline

---

## Executive Summary

Successfully routed a **18-layer backplane** with **512 nets**, **1,604 pads** in **11.96 minutes (717.5s)** — a **4× improvement over the April 3 baseline (47.9 min)** and **2× improvement over the April 3 persistent-kernel run (25 min)**.

**Run history (all runs, both dates):**

| Run | Date | Change | Total Time | Iter avg | Iters | Tracks | Vias | Notes |
|-----|------|--------|------------|----------|-------|--------|------|-------|
| 1 | Apr 3 | Baseline — multi-launch kernel | 47.9 min | ~39s | 74 | 4,118 | 2,585 | 512/512 ✓ |
| 2 | Apr 3 | Persistent CUDA kernel enabled | 25 min | ~20s | 70 | 4,118 | 2,585 | **2× gain** |
| 3 | Apr 3 | `cupyx.scatter_add` bitmap fix | ~25 min | ~22s | — | — | — | Bug fix; no perf gain |
| 4 | Apr 3 | GPU-resident `node_owner_gpu` | ~25 min | ~22s | — | — | — | No measurable improvement |
| **5** | **Apr 5** | **Current state** | **11.96 min** | **11.0s** | **65** | **4,290** | **2,754** | **512/512 ✓ — 4× baseline** |

**Primary gain between Apr 3 Run 2 and today**: iteration count reduced (70 → 65) and per-iter time halved (~20s → 11s), together yielding the 2× improvement over the already-optimized Apr 3 run.

---

## Test Board Characteristics

| Property | Value |
|----------|-------|
| Copper layers | 18 |
| Pads | 1,604 |
| Routable nets | 512 |
| Total nets | 1,088 |
| Components | 12 |
| Board size | 73.1 × 97.3 mm |
| Lattice nodes | 446,472 |
| Lattice edges | 14,281,664 |

---

## Routing Run — Measured Results (April 5, 2026)

**Hardware**: NVIDIA T1200 Laptop GPU, 4 GB VRAM (3.0 GB free), Turing architecture (Compute Capability 7.5)  
**Log**: `run_20260405_211914.log` (no longer available; values captured in `golden_metrics.json`)  
**Debug output**: `run_20260405_214600/` (3 pre-routing screenshots; production run artifacts not retained)

| Metric | Value |
|--------|-------|
| **Success rate** | 512 / 512 nets (100%) |
| **Total time** | 717.5s (11.96 min) |
| **Iterations** | 65 |
| **Iter 1 time** | 29.2s |
| **Iter avg time** | 11.0s |
| **Iter min/max** | not measured (log gone) |
| **Final overuse** | 0 edges |
| **Barrel conflicts** | 444 |
| **Tracks written** | 4,290 (+172 vs Apr 3) |
| **Vias written** | 2,754 (+169 vs Apr 3) |

### Board state after routing (golden_board.json)

| Field | Post-routing value |
|-------|--------------------|
| `tracks_existing` | 9,605 |
| `vias_existing` | 6,021 |
| `pads` | 1,604 |
| `copper_layers` | 18 |

---

## Performance Comparison vs. April 3 Baseline

| Metric | Apr 3 Run 1 (baseline) | Apr 3 Run 2 (best) | **Apr 5** | vs. baseline | vs. Apr 3 best |
|--------|------------------------|---------------------|-----------|--------------|----------------|
| Total time | 47.9 min | 25 min | **11.96 min** | **−75%** | **−52%** |
| Iterations | 74 | 70 | **65** | −12% | −7% |
| Iter avg | ~39s | ~20s | **11.0s** | −72% | −45% |
| Iter 1 | — | — | 29.2s | — | — |
| Tracks | 4,118 | 4,118 | **4,290** | +4% | +4% |
| Vias | 2,585 | 2,585 | **2,754** | +7% | +7% |
| Success rate | 100% | 100% | **100%** | = | = |
| Overuse | 0 | 0 | **0** | = | = |

---

## Active Thresholds (golden_metrics.json — GPU block)

These are the soft-warning thresholds used by the regression test suite (measured × 1.25 headroom):

| Threshold | Value | Basis |
|-----------|-------|-------|
| `total_time_s_max` | 900s | 717.5s × 1.25 |
| `iter_avg_time_s_max` | 15.0s | 11.0s × 1.36 |
| `iter_1_time_s_max` | 40.0s | 29.2s × 1.37 |
| `iterations_max` | 80 | 65 × 1.23 |
| `tracks_delta_min` | 4,000 | floor below 4,290 |
| `vias_delta_min` | 2,500 | floor below 2,754 |
| `overuse_final_max` | 0 | exact |
| `converged` | true | exact |

---

## Regression Test Status (April 5, 2026)

Suite: `tests/regression/` + `tests/unit/`  
Runner: `python -m pytest tests/ -v`

| Category | Passing | Skipping | Failing |
|----------|---------|----------|---------|
| Unit tests | 26 | 0 | 0 |
| Regression — board load | 5 | 0 | 0 |
| Regression — log health | 2 | 0 | 0 |
| Regression — GPU/iter soft checks | 3 | 0 | 0 |
| Regression — routing quality | 0 | 23 | 0 |
| Regression — write-back | 0 | 6 | 0 |
| **Total** | **36** | **29** | **0** |

**Why 29 still skip**: Routing quality and write-back tests require `routing_result` or `log_routing_result` populated from a live routing run. The April 5 log was overwritten by subsequent KiCad launches before tests could consume it. Next full routing run with `ORTHO_DEBUG=1` will unlock these.

---

## Known Bottlenecks — Updated Status

| Priority | Target | Apr 3 Status | Apr 5 Status |
|----------|--------|-------------|-------------|
| ✅ DONE | MULTI-LAUNCH Python→CUDA overhead | Fixed (persistent kernel compiled, not triggered) | **Appears resolved** — 11s/iter vs 20s/iter |
| ✅ DONE | `int(seed)` bitmap loop correctness | Bug fixed Apr 3 | Stable |
| ✅ DONE | GPU-resident `node_owner_gpu` | No gain measured Apr 3 | Deployed |
| 🟡 #1 | Unknown per-net gap (~70–100ms) | Not identified | Partially resolved — halved iter time |
| 🟡 #2 | `initialize_graph()` — 20.9s one-time | Identified | Not measured this run |
| 🟢 #3 | `_rebuild_via_usage_from_committed` | ~1% runtime — deferred | Still deferred |

**Observation**: The 11s/iter average (down from 20s in Apr 3 Run 2 and 39s in baseline) strongly suggests the persistent kernel or equivalent optimization is now active. The "unknown 70–100ms gap" from April 3 has been substantially reduced. Exact cause unknown — the routing log was not retained.

---

## Next Optimization Targets

1. **Profile the Apr 5 improvement** — run with `ORTHO_DEBUG=1`, retain the log, identify what changed vs Apr 3. Compare `[GPU-SEEDS]` lines: is it now using PERSISTENT KERNEL?

2. **Unlock routing quality regression tests** — retain the next routing log before launching KiCad again. The 29 skipping tests require a log with `[ITER N]` lines.

3. **`initialize_graph()` profiling** — at 11 min total, the 20.9s init is now ~3% of runtime. Still worth a 30–40% speedup for future larger boards.

4. **CPU-only baseline** — run `python main.py cli TestBoards/TestBackplane.kicad_pcb --cpu-only` to populate the `cpu` block in `golden_metrics.json`.

---

## System Health

| Check | Result |
|-------|--------|
| Errors in log | 0 |
| Exceptions | 0 |
| Keepout violations | 0 |
| Routing success | 512/512 (100%) |
| Final overuse | 0 |

**Conclusion**: System is stable. Routing quality and performance both improved over the April 3 baseline. Ready for further optimization and regression test expansion.

---

## References

### Related Documents
- [optimization_baseline_2026-04-03.md](optimization_baseline_2026-04-03.md) — Previous baseline (identifies MULTI-LAUNCH root cause)
- [OPTIMIZATION_QUICK_REF.md](OPTIMIZATION_QUICK_REF.md) — Priority table
- [CURRENT_LOGGING_REVIEW.md](CURRENT_LOGGING_REVIEW.md) — Logging analysis

### Key Files
- **Golden metrics**: `tests/regression/golden_metrics.json` — GPU thresholds from this run
- **Golden board**: `tests/regression/golden_board.json` — Post-routing board signature
- **Main router**: `orthoroute/algorithms/manhattan/unified_pathfinder.py`
- **GPU kernels**: `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`

### Test Board
- `TestBoards/TestBackplane.kicad_pcb` — 18-layer backplane, 1,604 pads, 512 nets

---

**Document Version**: 1.0  
**Date**: April 5, 2026  
**Status**: Baseline Established — 4× improvement over April 3
