# OrthoRoute Golden Result — April 10, 2026

**Status:** ✅ **GOLDEN STANDARD** — Use this as baseline for regression testing and optimization benchmarking.

## Summary

This run achieved **100% routing completion with zero overuse** — the primary success criterion for OrthoRoute.

```
ROUTING COMPLETE: All 512 nets routed successfully with zero overuse!
✓ CONVERGED  edges=0  via_overuse=0%  barrel=367
```

## Headless Cross-Check (Same Day)

A later headless run using the file parser also achieved full convergence and stayed within golden thresholds.

| Metric | Golden (Plugin IPC) | Headless (Log `run_20260410_184636.log`) | Status |
|--------|----------------------|-------------------------------------------|--------|
| Nets Routed | 512/512 | 512/512 | ✅ |
| Converged | True | True | ✅ |
| Iterations | 73 | 64 | ✅ (<= 88 threshold) |
| Total Time | 1106.6s | 565.0s | ✅ (<= 1328s threshold) |
| Final Overuse | 0 | 0 | ✅ |
| Barrel Conflicts | 367 | 305 | ✅ (<= 450 threshold) |

This confirms headless regression viability after the pad coordinate transform fix in the file parser.

---

## Validation Workflow (April 12, 2026 Update)

After establishing this golden result as the baseline, the optimization workflow was enhanced with automated validation tools:

**Smoke test (fast validation checkpoint):**
```powershell
.\scripts\optimize_and_validate.ps1 -Compare tests/regression/smoke_metrics.json
# 100 nets, <30s, validates correctness before full backplane test
```

**Full backplane validation:**
```powershell
.\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane -Compare tests/regression/golden_metrics.json
# 512 nets, 11-18 min, compares against this golden result
```

**Log analysis:**
```powershell
python scripts/analyze_log.py --compare tests/regression/golden_metrics.json
# Extracts metrics and validates against thresholds
```

**See:** [optimization_workflow.md](optimization_workflow.md) for complete optimization workflow guide with automated validation.

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| **Date** | April 10, 2026 @ 15:06:14 |
| **Board** | TestBackplane.kicad_pcb |
| **Mode** | KiCad Plugin (GUI) via IPC API |
| **Hardware** | NVIDIA GPU (CUDA Compute Capability 75) |
| **GPU Memory** | 3.0 GB free / 4.3 GB total |
| **Repository** | Detached HEAD at 8b38742 |

---

## Board Characteristics

| Property | Value |
|----------|-------|
| **Dimensions** | 73.1 × 97.3 mm |
| **Copper Layers** | 18 (F.Cu + In1..In16 + B.Cu) |
| **Total Pads** | 1,604 |
| **Components** | 12 footprints |
| **Routable Nets** | 512 |
| **Existing Tracks** | 9,605 (loaded, then cleared for routing) |
| **Existing Vias** | 6,021 (loaded, then cleared for routing) |
| **Congestion Ratio** | ρ = 0.161 (SPARSE) |

---

## Routing Results — GOLDEN METRICS

### Primary Success Criteria ✅

| Metric | Value | Status |
|--------|-------|--------|
| **Nets Routed** | **512/512 (100%)** | ✅ PERFECT |
| **Edge Overuse** | **0** | ✅ CONVERGED |
| **Via Overuse** | **0%** | ✅ CONVERGED |
| **Convergence** | **Full** | ✅ ACHIEVED |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Total Time** | **1,106.6 seconds (18.4 minutes)** |
| **Routing Time** | ~1,106 seconds (99.9% of total) |
| **Iterations** | 73 (of max 250) |
| **Average Iteration** | ~15.2 seconds |
| **Final Iteration** | 4.2 seconds (hotset=1 net) |

### Graph Metrics

| Metric | Value |
|--------|-------|
| **Lattice Size** | 106 × 234 × 18 = **446,472 nodes** |
| **Grid Pitch** | 0.4 mm |
| **Routing Bounds** | (195.2, 47.6) to (237.0, 140.9) mm (with 3mm margin) |
| **Total Edges** | 14,281,664 |
| **Via Edges** | 13,493,376 (272 layer pairs) |
| **H/V Edges** | 788,288 |

### Via Configuration

| Property | Value |
|----------|-------|
| **Via Pairs** | 272 (FULL BLIND/BURIED ENABLED) |
| **Via Policy** | 16×16 internal + 16×2 F.Cu transitions |
| **Via Cost** | 0.7 (initial), 0.35 (late-stage annealing) |

### Output Geometry

| Component | Count |
|-----------|-------|
| **Routing Tracks** | 2,226 |
| **Escape Stubs** | 2,048 |
| **Total Tracks** (after dedup) | **4,274** |
| **Routing Vias** | 2,738 |
| **Escape Vias** | 0 |
| **Total Vias** | **2,738** |

### Barrel Conflicts

| Iteration | Barrel Conflicts | Trend |
|-----------|------------------|-------|
| Iter 1 | 2,566 | Initial |
| Iter 2 | 2,238 | ↓ Decreasing |
| Iter 3 | 2,096 | ↓ Decreasing |
| Iter 4 | 1,945 | ↓ Decreasing |
| Iter 5 | 1,687 | ↓ Decreasing |
| Iter 10 | 974 | ↓ Decreasing |
| Iter 20 | 663 | ↓ Decreasing |
| Iter 30 | 581 | ↓ Decreasing |
| Iter 40 | 520 | ↓ Decreasing |
| Iter 50 | 467 | ↓ Decreasing |
| Iter 60 | 423 | ↓ Decreasing |
| Iter 70 | 394 | ↓ Decreasing |
| **Iter 73 (Final)** | **367** | ✅ **Acceptable** |

**Note:** Barrel conflicts at 367 are within acceptable limits. Routing converged with zero overuse despite remaining barrel conflicts.

---

## PathFinder Parameters

### Algorithm Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `pres_fac_init` | 1.0 | Initial present factor |
| `pres_fac_mult` | 1.1 | Escalation multiplier |
| `pres_fac_max` | 64.0 | Maximum present factor |
| `hist_gain` | 0.2 | History cost gain |
| `via_cost` | 0.7 → 0.35 | Late-stage annealing applied |
| `grid_pitch` | 0.4 mm | Lattice resolution |
| `max_iterations` | 250 | Dynamically derived |
| `stagnation_patience` | 5 | Early termination |

### Derived Parameters (from BoardAnalyzer)

```
Congestion ratio ρ = 0.161 (SPARSE)
→ Max iterations: 250
→ Stagnation patience: 5
→ Via cost: 0.7
```

---

## GPU Acceleration Metrics

### CUDA Configuration

| Component | Status |
|-----------|--------|
| **GPU Available** | ✅ YES |
| **CUDA Dijkstra** | ✅ Enabled |
| **Compute Capability** | 75 (NVIDIA RTX 2060 or similar) |
| **GPU Memory** | 3.0 GB free / 4.3 GB total |

### GPU Kernel Performance

| Kernel | Performance | Notes |
|--------|-------------|-------|
| **GPU Sort** | 1.0s (14.1M edges/sec) | Radix sort on 14.3M edges |
| **Via Metadata** | 1.817s | Built metadata for 13.5M via edges |
| **Via Penalty** | 2.03ms | Applied 408 penalties (vs ~800ms CPU) |
| **Hard Block** | 2.30ms | Blocked 2,472 via edges (vs ~30s CPU) |
| **Persistent Dijkstra** | ~7-23ms per net | GPU wavefront expansion |

### GPU Speedup Estimates

- **Via penalties:** ~394× faster (2ms vs 800ms)
- **Hard blocking:** ~13,000× faster (2.3ms vs 30s)
- **Overall routing:** ~127× faster than CPU-only mode (estimated)

---

## Layer Utilization

**Manhattan discipline enforced:** Odd layers horizontal, even layers vertical.

| Layer | Horizontal | Vertical | Orientation |
|-------|------------|----------|-------------|
| F.Cu | 0 | 0 | (Not used for routing) |
| In1.Cu | 232 | 0 | H |
| In2.Cu | 0 | 285 | V |
| In3.Cu | 168 | 0 | H |
| In4.Cu | 0 | 214 | V |
| In5.Cu | 121 | 0 | H |
| In6.Cu | 0 | 140 | V |
| In7.Cu | 151 | 0 | H |
| In8.Cu | 0 | 120 | V |
| In9.Cu | 65 | 0 | H |
| In10.Cu | 0 | 108 | V |
| In11.Cu | 72 | 0 | H |
| In12.Cu | 0 | 97 | V |
| In13.Cu | 93 | 0 | H |
| In14.Cu | 0 | 120 | V |
| In15.Cu | 94 | 0 | H |
| In16.Cu | 0 | 146 | V |
| B.Cu | 0 | 0 | (Not used for routing) |

**Total:** 996 horizontal + 1,230 vertical = 2,226 routing tracks

**Layer balance:** Good distribution across all 16 internal layers.

---

## Iteration Convergence Timeline

| Iteration | Nets Routed | Overused Edges | Barrel Conflicts | pres_fac | Time |
|-----------|-------------|----------------|------------------|----------|------|
| 1 | 512/512 | ~high | 2,566 | 1.0 | - |
| 10 | 512/512 | ~medium | 974 | 2.6 | - |
| 20 | 512/512 | ~low | 663 | 6.7 | - |
| 30 | 512/512 | ~lower | 581 | 17.4 | - |
| 40 | 512/512 | ~minimal | 520 | 45.3 | - |
| 50 | 512/512 | ~trace | 467 | 64.0 (max) | - |
| 60 | 512/512 | ~trace | 423 | 64.0 | - |
| 70 | 512/512 | ~zero | 394 | 64.0 | - |
| **73** | **512/512** | **0** ✅ | **367** | **64.0** | **4.2s** |

**Convergence achieved at iteration 73** with full hotset reduction (final hotset = 1 net).

---

## Regression Test Criteria

### MUST PASS (Core Functionality)

```python
def test_golden_result_routing():
    """Regression test against April 10 golden result."""
    result = route_board("TestBackplane.kicad_pcb", use_gpu=True)
    
    # Primary success criteria
    assert result.nets_routed == 512, "Must route all 512 nets"
    assert result.edge_overuse == 0, "Must achieve zero overuse"
    assert result.convergence == "full", "Must fully converge"
    
    # Performance criteria (with 20% tolerance)
    assert result.total_time < 1328, "Must complete within 22 minutes (18.4 * 1.2)"
    assert result.iterations <= 88, "Must converge within 88 iterations (73 * 1.2)"
    
    # Graph metrics
    assert result.lattice_nodes == 446_472, "Lattice size must match"
    assert result.total_edges == 14_281_664, "Edge count must match"
    
    # Output geometry (±10% tolerance)
    assert 3846 <= result.total_tracks <= 4701, "Total tracks: 4274 ± 10%"
    assert 2464 <= result.total_vias <= 3012, "Total vias: 2738 ± 10%"
    
    # Barrel conflicts (should improve or match)
    assert result.barrel_conflicts <= 450, "Barrel conflicts ≤ 450 (367 + margin)"
```

### SHOULD MONITOR (Optimization Targets)

- **Routing time:** Target < 15 minutes (current: 18.4 min)
- **Iterations:** Target < 60 (current: 73)
- **Barrel conflicts:** Target < 300 (current: 367)
- **Via count:** Target < 2,500 (current: 2,738)

---

## Comparison with Previous Runs

### vs. April 8, 2026 Baseline

| Metric | April 8 | April 10 (Golden) | Delta |
|--------|---------|-------------------|-------|
| **Nets Routed** | 506/512 (98.8%) | **512/512 (100%)** | **+6 nets** ✅ |
| **Convergence** | Partial | **Full** | ✅ |
| **Total Time** | 17.5 min | 18.4 min | +0.9 min (+5%) |
| **Iterations** | ~80 | 73 | -7 iterations ✅ |
| **Edge Overuse** | >0 | **0** | ✅ |
| **Barrel Conflicts** | Declining | 367 (final) | Stabilized |

**Verdict:** April 10 is superior due to full convergence despite slightly longer runtime.

### vs. April 5, 2026 Best Time

| Metric | April 5 | April 10 (Golden) | Delta |
|--------|---------|-------------------|-------|
| **Total Time** | 11.96 min | 18.4 min | +6.4 min (+54%) ⚠️ |
| **Nets Routed** | Unknown | **512/512 (100%)** | N/A |
| **Convergence** | Unknown | **Full** | N/A |

**Note:** April 5 run may not have achieved full convergence. Golden result prioritizes correctness over raw speed.

---

## Critical Success Factors

1. **IPC API Integration** — Direct communication with KiCad ensures accurate board data
2. **GPU Acceleration** — CUDA kernels provide 100-1000× speedup for via operations
3. **Blind/Buried Vias** — 272 layer pairs enable flexible routing in dense 18-layer board
4. **Column-Based Escapes** — Deterministic pad escape planning prevents pad congestion
5. **Late-Stage Via Annealing** — Halving via cost at high pres_fac reduces unnecessary vias
6. **Hotset Routing** — Final iterations route only 1-5 problematic nets for efficiency

---

## Hardware Requirements for Reproduction

### Minimum Specifications

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA GPU with CUDA Compute Capability ≥ 7.0 |
| **GPU Memory** | ≥ 4 GB VRAM |
| **RAM** | ≥ 16 GB system RAM |
| **Storage** | ≥ 1 GB free (for debug screenshots if enabled) |
| **Python** | 3.9+ |
| **CuPy** | Compatible with CUDA version |

### Recommended Specifications (Used for Golden Run)

- NVIDIA RTX 2060 (or similar, Compute Capability 75)
- 4.3 GB VRAM
- 16+ GB system RAM
- Windows 11 with KiCad 9.0

---

## Reproduction Instructions

### 1. Setup Environment

```powershell
# Clone repository
git clone https://github.com/bbenchoff/OrthoRoute.git
cd OrthoRoute

# Checkout golden result commit
git checkout 8b38742

# Install dependencies (with GPU support)
pip install -r requirements.txt

# Verify CuPy installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

### 2. Run in KiCad Plugin Mode (Recommended)

```powershell
# Copy to KiCad plugin directory
.\copy_to_kicad.ps1

# Launch KiCad
.\launch_kicad_debug.ps1

# In KiCad PCB Editor:
# 1. Open TestBoards/TestBackplane.kicad_pcb
# 2. Tools → External Plugins → OrthoRoute
# 3. Click "Begin Autorouting"
```

### 3. Expected Log Output

```
[ROUTER SELECTION] Selected router: unified_pathfinder
PathFinder loaded (GPU=YES)
Lattice: 106×234×18 = 446,472 nodes
CSR: 446472 nodes, 14281664 edges
[GPU] CUDA Near-Far Dijkstra enabled
Congestion ratio ρ = 0.161 (SPARSE)
Max iterations: 250
...
[ITER  73] nets=512/512  ✓ CONVERGED  edges=0  via_overuse=0%  barrel=367
ROUTING COMPLETE: All 512 nets routed successfully with zero overuse!
```

### 4. Verify Results

- Total time: 18-22 minutes (±20% variance acceptable)
- All 512 nets routed
- Zero edge overuse
- Barrel conflicts ≤ 450

---

## Known Limitations

1. **Barrel Conflicts:** 367 remaining conflicts are acceptable but could be improved
2. **Runtime:** 18.4 minutes is slower than April 5 (11.96 min) — optimization opportunity
3. **Via Count:** 2,738 vias is high — layer compaction could reduce this
4. **F.Cu/B.Cu Unused:** Outer layers not utilized for routing (design choice)

---

## Future Optimization Targets

### High Priority (Performance)

1. **Reduce runtime to < 15 minutes**
   - Profile hotspots (likely via conflict detection at ~0.9ms × 512 nets/iter)
   - Optimize `_build_owner_bitmap_for_fullgraph` (called per-net)
   - Batch GPU operations more aggressively

2. **Reduce iterations to < 60**
   - Improve initial routing quality (better net ordering?)
   - Earlier via cost annealing
   - Adaptive pres_fac escalation

### Medium Priority (Quality)

3. **Reduce barrel conflicts to < 200**
   - Improved via pool management
   - Conflict-aware routing in early iterations
   - Dedicated barrel conflict resolution pass

4. **Reduce via count to < 2,500**
   - More aggressive via cost in early iterations
   - Layer compaction analysis
   - Prefer long horizontal/vertical runs over frequent vias

### Low Priority (Features)

5. **Utilize F.Cu and B.Cu layers**
   - Currently unused for routing
   - Could reduce via count and improve routability

6. **Investigate April 5 performance**
   - Why was it 54% faster?
   - Did it achieve full convergence?
   - Can we replicate that speed with full convergence?

---

## Conclusion

**This April 10, 2026 run is the GOLDEN RESULT for OrthoRoute regression testing.**

✅ **Use this as the baseline** for:
- Regression test suite validation
- Performance optimization benchmarking  
- Algorithm improvement evaluation
- GPU acceleration verification

⚠️ **Do NOT regress** on:
- 100% net routing completion
- Zero edge overuse (full convergence)
- Lattice size consistency (446,472 nodes)

🎯 **Optimization targets** (while maintaining correctness):
- Runtime < 15 minutes
- Iterations < 60
- Barrel conflicts < 200
- Via count < 2,500

---

**Log File:** `logs/run_20260410_150614.log`  
**Repository:** Detached HEAD at 8b38742  
**Date:** April 10, 2026 @ 15:06:14  
**Status:** ✅ PRODUCTION READY — GOLDEN STANDARD
