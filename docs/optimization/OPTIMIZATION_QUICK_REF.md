# OrthoRoute Optimization Quick Reference

**Baseline**: April 3, 2026 | **Test**: TestBackplane (512 nets, 32 layers) | **Time**: 44.23 min  
**Live-run data updated**: April 3, 2026 (40 iters, 3,311 GPU paths profiled)

---

## 🎯 Top Optimization Targets (Revised — live run data)

| Priority | Target | Time | % of Total | Est. Savings |
|----------|--------|------|------------|--------------|
| 🔴 **#1** | GPU Python overhead (296ms/path × 3,311 paths) | ~980s/run | **~95% of routing** | **10-20× with persistent kernel** |
| 🟡 **#2** | `initialize_graph()` | 20.9s | one-time | **6-8s** |
| 🟢 **#3** | `_rebuild_via_usage_from_committed()` | ~11s/40 iters | **~1%** | minimal — wrong target |

> ⚠️ **Baseline Priority #1 was wrong.** Earlier analysis assumed via rebuild was the main cost.
> Live profiling showed it is **1% of runtime**. The real bottleneck is the Python/CUDA
> launch overhead of the MULTI-LAUNCH kernel — 296ms per path vs 16ms of actual GPU compute.

---

## 🔑 Key GPU Finding (April 3, 2026)

From 40 iterations, 3,311 paths routed on RTX Turing (compute 75, 4.3 GB VRAM):

```
Per-net timing breakdown:
  GPU Dijkstra kernel (MULTI-LAUNCH):  ~16ms   (5% of per-net time)
  Python ↔ CUDA launch overhead:      ~296ms  (95% of per-net time)
  ─────────────────────────────────────────────
  Total per net:                       ~312ms
  Total routing (40 iters, 3,311 paths): 1,032s (~17 min so far)
```

**Root cause**: MULTI-LAUNCH uses a Python `for` loop (~150 iterations) to drive the CUDA
wavefront, with each iteration being a separate kernel launch + Python→CUDA sync. The
PERSISTENT KERNEL is already compiled (`[CUDA] Compiled PERSISTENT KERNEL`) but is
not being activated — device-side queues would eliminate this loop entirely.

**Where to look**: `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`
- Search `MULTI-LAUNCH` and `PERSISTENT KERNEL` to find the routing selection logic
- The persistent kernel takes over the wavefront loop on-device; Python only launches once

---

## 📊 Current Baseline

```
Total Time: 44.23 minutes (2,654 seconds)
├─ Initialization: ~25s
│  ├─ initialize_graph: 20.9s
│  ├─ finalize: 3.95s
│  └─ keepout_obstacles: 0.15ms ✅ optimized
│
├─ Routing (64 iterations): ~39 min
│  ├─ GPU overhead (Python→CUDA launch loop): ~95% ← REAL BOTTLENECK
│  │  └─ MULTI-LAUNCH: 150 kernel launches per net × 512 nets × N iters
│  ├─ Via rebuilds (64x): ~11s (275ms avg, 1% of total) ← not the target
│  └─ Via pooling (64x): ~1.6s (25ms avg)
│
└─ Visualization/Output: ~4 min
```

---

## 🔧 Before Optimization: Setup

### 1. Add @profile_time Decorators
```python
# In unified_pathfinder.py
from orthoroute.shared.utils.performance_utils import profile_time

@profile_time
def route_net(self, net): ...

@profile_time
def _expand_wavefront(self, ...): ...
```

### 2. GPU Kernel Selection
```python
# In cuda_dijkstra.py — find where MULTI-LAUNCH is selected vs PERSISTENT
# Look for: "Using MULTI-LAUNCH kernel" log line and the condition above it
# Goal: activate the already-compiled PERSISTENT KERNEL for ROI > threshold
```

### 3. Baseline Test Command
```powershell
# Always test with same board for comparison
python main.py cli TestBoards/TestBackplane.kicad_pcb
```

---

## 📈 Performance Analysis Commands

```python
# Analyze a completed run log (Python — avoids PowerShell multi-line issues)
import re, os
log = r"<plugin_dir>/logs/latest.log"
lines = open(log, encoding="utf-8", errors="ignore").readlines()

# Iteration count
iters = [l for l in lines if re.search(r"WARNING.*\[ITER\s+\d+\]", l)]
print(f"Iters: {len(iters)}")

# Via rebuild stats
vals = [float(re.search(r": ([\d.]+)ms", l).group(1)) for l in lines if "[PROFILE]" in l and re.search(r": ([\d.]+)ms", l)]
print(f"Rebuild: {sum(vals)/1000:.1f}s total, {sum(vals)/len(vals):.0f}ms avg over {len(vals)} calls")

# GPU solve (kernel only) vs total per-net time
gpu = [float(re.search(r"\(([\d.]+)ms\)", l).group(1)) for l in lines if "Path found in" in l and "ms)" in l]
tot = [float(re.search(r"in ([\d.]+)s", l).group(1)) for l in lines if "SUCCESS! Path found in" in l and re.search(r"in ([\d.]+)s", l)]
print(f"GPU kernel: {sum(gpu)/len(gpu):.1f}ms avg,  Total/net: {sum(tot)/len(tot)*1000:.0f}ms avg,  Overhead: {(sum(tot)/len(tot))*1000 - sum(gpu)/len(gpu):.0f}ms")

# Run timestamps
ts = [re.search(r"(\d{2}:\d{2}:\d{2})", l) for l in lines]
ts = [m.group(1) for m in ts if m]
print(f"Run: {ts[0]} → {ts[-1]}")
```

---

## ✅ Success Criteria

**Must Maintain**:
- ✅ 512/512 nets routed (100% success)
- ✅ Zero final overuse
- ✅ All keepout constraints enforced
- ✅ Correct barrel conflict detection

**Performance Targets** (revised after live profiling):
- **Realistic** (MULTI-LAUNCH → PERSISTENT KERNEL): 44 min → 3-5 min (10-20× faster)
- **Fallback** (minor fixes only): 44 min → 40 min (< 10% gains)

---

## 🚀 GPU Architecture Notes

### Compiled Kernels (all available, from logs)
| Kernel | Purpose | Status |
|--------|---------|--------|
| ACTIVE-LIST | Sparse wavefront (446K× memory saving) | ✅ Used |
| MULTI-LAUNCH | Python loop drives wavefront (~150 iters/net) | ✅ Used (bottleneck) |
| PERSISTENT KERNEL | Device-side queues, 1 launch per net | ✅ Compiled, **not activated** |
| COMPACTION | GPU-side frontier compaction | ✅ Compiled |
| ACCOUNTANT | GPU-side history/present/cost updates | ✅ Compiled |
| GPU BACKTRACE | Eliminates 256 MB parent/dist CPU transfer | ✅ Compiled |
| RR-WAVEFRONT | Round-robin layer preference | ✅ Used |
| DELTA-STEPPING | Bucket assignment | ✅ Compiled |

### GPU Hardware (RTX Turing, compute 75)
- VRAM: 4.3 GB total / 3.0 GB free at start
- Graph: 446,472 nodes, 14,281,664 edges
- Sort: 14M edges/sec (1.0s one-time GPU radix sort)

### Key Log Patterns
```
[GPU-SEEDS] Using MULTI-LAUNCH kernel (Python loop)   ← bottleneck: search for this
[GPU-SEEDS] Path found in 151 iterations (21.31ms)    ← 21ms GPU, ~280ms Python overhead
[GPU-SEEDS] SUCCESS! Path found in 0.534s (148 nodes) ← 534ms total per net
```

---

## 📚 Key Files

| File | Purpose |
|------|---------|
| `orthoroute/algorithms/manhattan/unified_pathfinder.py` | Main router (~5,967 lines) |
| `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py` | GPU kernel selection logic |
| `orthoroute/shared/utils/performance_utils.py` | @profile_time decorator |
| `TestBoards/TestBackplane.kicad_pcb` | Standard test board |
| `docs/optimization/optimization_baseline_2026-04-03.md` | Full baseline report |

---

## 🎬 Next Steps

1. ✅ `@profile_time` decorator created (`shared/utils/performance_utils.py`)
2. ✅ Logging reclassified — milestones on console, detail in file
3. ⬜ Apply `@profile_time` to algorithm functions in `unified_pathfinder.py` (`route_net`, `_expand_wavefront`, `_update_costs`, `_commit_route`, `_rebuild_via_usage_from_committed`)
4. ⬜ Re-run baseline to confirm log size reduction + capture profile data
5. ⬜ Optimize via rebuild operation first (22.2s target)
