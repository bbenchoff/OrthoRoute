# OrthoRoute Optimization Quick Reference

**Baseline**: April 3, 2026 | **Test**: TestBackplane (512 nets, 18 layers) | **Time**: 44.23 min → **25 min**  
**Last updated**: April 3, 2026 — persistent kernel enabled, bitmap attempts did not improve further

---

## ✅ Completed Optimizations

| Date | Change | Before | After | Gain |
|------|--------|--------|-------|------|
| Apr 3 | Persistent CUDA kernel (bitmap fix) | 47.9 min / 39s/iter | **25 min / ~20s/iter** | **~2×** |
| Apr 3 | cuda_dijkstra.py log reclassification | 861 MB logs | ~10-20 MB logs | log size |
| Apr 3 | Vectorize `int(seed)` loops → `cupyx.scatter_add` | correctness bug | fixed | **bug fix only** |
| Apr 3 | GPU-resident `node_owner_gpu` bitmap (zero upload) | 56KB/net upload | 0 upload | **no perf gain** |

---

## ❌ Attempted — No Improvement Found

| Attempt | Hypothesis | Result |
|---------|------------|--------|
| Vectorize bitmap init loops (`cp.scatter_add`) | ~100 GPU→CPU syncs costing ~50ms/net | Fixed correctness bug (`cp.scatter_add` → `cupyx.scatter_add` + bitmap OR corruption); **iteration time unchanged** |
| GPU-resident `node_owner_gpu` + on-GPU bitmap build | 56KB `cp.asarray()` upload per net was bottleneck | Built and deployed; per-iter time: **~22s same as before**; upload was not the bottleneck |

**Conclusion**: The ~70–100ms per-net wall overhead beyond the GPU kernel is **not** in bitmap construction or bitmap upload. Root cause still unidentified.

---

## 🎯 Next Optimization Targets

| Priority | Target | Est. Cost | % of Total | Est. Savings |
|----------|--------|-----------|------------|--------------|
| 🔴 **#1 (UNKNOWN)** | Mystery ~70–100ms gap per net | unknown | **~80% of per-net time** | **3–4× if found** |
| 🟡 **#2** | `initialize_graph()` | 20.9s once | one-time | 6-8s |
| 🟢 **#3** | `_rebuild_via_usage_from_committed()` | ~11s/run | **~1%** | minimal |

> **Current per-net breakdown** (measured April 3, 2026 after all bitmap optimizations):
> - GPU kernel time: **4–22ms** (persistent kernel, runs to convergence on-device)
> - Total wall time per net: **~90–130ms** 
> - **Unaccounted gap: ~70–100ms** — location unknown, profiling needed
> - Per iteration (~512 nets): **~22s**
> - Total run (70 iterations): **~25 min**

---

## 🔍 Profiling Plan for Mystery Gap

The ~70–100ms gap per net is not in:
- GPU kernel execution (measured: 4–22ms)
- Bitmap construction (eliminated via GPU path)
- Bitmap upload (eliminated via GPU-resident `node_owner_gpu`)
- Frontier/dist/parent init (vectorized)

**Remaining candidates to profile with `@profile_time`**:
1. `_path_to_edges(path)` — converts node list to edge indices
2. `commit_path(edge_indices)` — updates accounting arrays
3. `_mark_via_barrel_ownership_for_path()` — marks node_owner
4. `_build_routing_seeds()` / `_get_portal_seeds()` — portal lookup
5. `node_owner_gpu` scatter write in `_mark_via_barrel_ownership_for_path`
6. Python overhead between `find_path_fullgraph_gpu_seeds()` return and `continue`

**How to profile**: Add `@profile_time` decorator (from `shared/utils/performance_utils.py`) to each candidate and rerun.

---

## 📊 Run History

| Run | Change | Total Time | Iter avg | Notes |
|-----|--------|------------|----------|-------|
| 1 | Multi-launch kernel (baseline) | 47.9 min | ~39s | 74 iters |
| 2 | Persistent CUDA kernel | **25 min** | ~20s | 70 iters — **2× gain** |
| 3 | Vectorize bitmap loops (bug fix) | ~25 min | ~22s | Correctness fix, no perf change |
| 4 | GPU-resident bitmap (zero upload) | ~25 min | ~22s | No improvement — upload wasn't bottleneck |
  Python ↔ CUDA launch overhead:      ~296ms  (95% of per-net time)
  ─────────────────────────────────────────────
  Total per net:                       ~312ms
  Total routing (40 iters, 3,311 paths): 1,032s (~17 min so far)
```

**Root cause**: MULTI-LAUNCH uses a Python `for` loop (~150 iterations) to drive the CUDA
wavefront, with each iteration being a separate kernel launch + Python→CUDA sync. The
PERSISTENT KERNEL is already compiled (`[CUDA] Compiled PERSISTENT KERNEL`) but was
not activating — device-side queues eliminate this loop entirely.

**RESOLVED April 3, 2026**: Added `allowed_bitmap` param to persistent kernel PTX.
See `orthoroute/algorithms/manhattan/pathfinder/persistent_kernel.py` — `use_bitmap` flag.

---

## 📊 Current Baseline (post persistent kernel)

```
Total Time: 25 minutes (1,500 seconds)     ← was 47.9 min
├─ Initialization: ~25s (unchanged)
│  ├─ initialize_graph: 20.9s
│  ├─ finalize: 3.95s
│  └─ keepout_obstacles: 0.15ms ✅
│
├─ Routing (70 iterations): ~24 min
│  ├─ Python bitmap construction (NEW #1): ~50ms/net × 88 hotset × 70 iters ← BOTTLENECK
│  │  └─ int(seed) GPU→CPU syncs in bitmap loop: ~100 round-trips/net
│  ├─ GPU persistent kernel: 3–24ms/net ✅ fast
│  ├─ GPU→CPU convergence check: ~10ms/net (every 10 iters)
│  ├─ Via rebuilds (70x): ~11s (~1%)
│  └─ Via pooling (70x): ~1.6s
│
└─ Visualization/Output: ~1 min
```

---

## 🔧 Next Fix: Vectorize Bitmap Construction

**File**: `cuda_dijkstra.py` → `find_path_fullgraph_gpu_seeds()` ~line 5567

**Current (slow)**:
```python
for seed in seed_nodes:           # O(100) GPU→CPU syncs
    seed_int = int(seed)
    word_idx = seed_int // 32
    roi_bitmaps[0, word_idx] = roi_bitmaps[0, word_idx] | (cp.uint32(1) << bit_idx)
```

**Fix (vectorized)**:
```python
seed_words = seed_nodes // 32
seed_bits  = (seed_nodes & 31).astype(cp.uint32)
masks = cp.uint32(1) << seed_bits
# scatter-or into bitmap (one kernel call, zero PCIe round-trips)
cp.scatter_add(roi_bitmaps[0], seed_words, masks)
```

Expected gain: reduces per-net Python overhead from ~50ms to ~5ms → total ~3-4× speedup
(25 min → ~8-10 min).

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
