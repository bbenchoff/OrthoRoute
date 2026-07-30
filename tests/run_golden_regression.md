# Running Golden Result Regression Test

This document describes how to run the full regression test against the **Golden Result (April 10, 2026)** baseline.

## Quick Start

### Option 1: Automated via KiCad Plugin (Recommended)

```powershell
# 1. Launch KiCad with TestBackplane
.\launch_kicad_debug.ps1

# 2. In KiCad: Tools → External Plugins → OrthoRoute → Begin Autorouting

# 3. Wait for completion (~18-22 minutes with GPU)

# 4. Run regression tests against the generated log
cd tests
pytest regression/test_backplane.py -v
```

### Option 2: Headless Test Suite (Limited)

```powershell
# Full headless routing + golden comparison (GPU recommended)
python tests/regression/run_headless_routing.py

# Validate an existing log only (no re-route)
python tests/regression/run_headless_routing.py --log-only

# Validate a specific log file
python tests/regression/run_headless_routing.py --log-only --log-file logs/run_20260410_184636.log

# Quick smoke test (20 nets, 3 iterations, ~30 seconds)
pytest tests/regression/test_backplane.py::TestHeadlessRouting -v
```

Headless timeout policy:
- Base timeout is `gpu.total_time_s_max` from `tests/regression/golden_metrics.json`.
- The runner applies a +10% buffer.
- The run is accepted only if the log contains `ROUTING COMPLETE` and log-derived completion time is within timeout.

---

## Golden Result Baseline (April 10, 2026)

### Board Configuration
- **File:** TestBoards/TestBackplane.kicad_pcb
- **Size:** 73.1 × 97.3 mm
- **Layers:** 18 copper layers
- **Pads:** 1,604
- **Nets:** 512 routable (1,088 total)
- **Hardware:** NVIDIA GPU (Compute Capability 75, 4.3 GB VRAM)

### Expected Results (100% Pass Criteria)

| Metric | Golden Value | Test Tolerance | Status |
|--------|--------------|----------------|--------|
| **Nets Routed** | 512/512 | Must be 100% | **HARD FAIL** |
| **Convergence** | Zero overuse | Must converge | **HARD FAIL** |
| **Iterations** | 73 | ≤ 88 (+20%) | SOFT WARN |
| **Total Time** | 1,106.6s (18.4 min) | ≤ 1,328s (+20%) | SOFT WARN |
| **Total Tracks** | 4,274 | 3,846-4,701 (±10%) | SOFT WARN |
| **Total Vias** | 2,738 | 2,464-3,012 (±10%) | SOFT WARN |
| **Barrel Conflicts** | 367 | ≤ 450 | SOFT WARN |
| **Lattice Nodes** | 446,472 | Must match exactly | **HARD FAIL** |
| **Lattice Layers** | 18 | Must match exactly | **HARD FAIL** |

---

## Latest Headless Validation (April 10, 2026 @ 18:46)

Source log: `logs/run_20260410_184636.log`

| Metric | Golden Baseline | Latest Headless Run | Result |
|--------|------------------|---------------------|--------|
| Nets Routed | 512/512 | 512/512 | PASS |
| Converged | True | True | PASS |
| Iterations | 73 (golden), threshold <= 88 | 64 | PASS |
| Total Time | 1106.6s (golden), threshold <= 1328s | 565.0s | PASS |
| Final Overuse | 0 | 0 | PASS |
| Barrel Conflicts | 367 (golden), threshold <= 450 | 305 | PASS |

Notes:
- This confirms the file-parser coordinate fix restored full headless convergence.
- The latest headless run completed faster than the golden plugin run while meeting all thresholds.

---

## Test Categories

### Group A: Log Health (Always Runs)
- ✅ `test_no_errors` — No ERROR lines in log
- ✅ `test_no_criticals` — No CRITICAL lines in log
- ✅ `test_ipc_adapter_in_log` — Confirms IPC API used (not SWIG/file fallback)
- ✅ `test_gpu_mode_detected` — Confirms GPU acceleration active

### Group A2: Board Load (Parses .kicad_pcb Directly)
- ✅ `test_pad_count` — 1,604 pads
- ✅ `test_copper_layers` — 18 layers
- ✅ `test_existing_tracks` — 9,605 tracks (±10%)
- ✅ `test_existing_vias` — 6,021 vias (±10%)

### Group A3: Lattice Size
- ✅ `test_lattice_nodes` — 446,472 nodes (**HARD FAIL** if wrong)
- ✅ `test_lattice_layers` — 18 layers (**HARD FAIL** if wrong)

### Group B: Routing Quality
- ✅ `test_all_nets_routed` — 512/512 nets (**HARD FAIL**)
- ✅ `test_convergence` — Zero overuse (**HARD FAIL**)
- ⚠️ `test_iteration_budget` — ≤ 88 iterations (SOFT WARN)
- ⚠️ `test_total_time` — ≤ 1,328 seconds (SOFT WARN)
- ⚠️ `test_barrel_conflicts` — ≤ 450 conflicts (SOFT WARN)

### Group C: Library Availability (Setup Validation)
- ✅ `test_numpy_available` — NumPy functional
- ✅ `test_scipy_available` — SciPy importable
- ✅ `test_orthoroute_importable` — OrthoRoute package importable
- ✅ `test_unified_pathfinder_importable` — Core router importable
- ⚠️ `test_cupy_installed` — CuPy available (SOFT WARN)
- ⚠️ `test_cuda_device_info` — CUDA memory ≥ 2 GB (SOFT WARN)

---

## Running Tests

### Full Test Suite
```powershell
cd tests
pytest regression/test_backplane.py -v
```

### Specific Test Groups
```powershell
# Log health only
pytest regression/test_backplane.py::TestLogHealth -v

# Board load verification
pytest regression/test_backplane.py::TestBoardLoad -v

# Routing quality (requires log from full run)
pytest regression/test_backplane.py::TestRoutingQuality -v

# GPU availability
pytest regression/test_backplane.py::TestGPUMode -v

# Headless routing (quick smoke test)
pytest regression/test_backplane.py::TestHeadlessRouting -v
```

### Test Output Example

```
tests/regression/test_backplane.py::TestRoutingQuality::test_all_nets_routed PASSED
tests/regression/test_backplane.py::TestRoutingQuality::test_convergence PASSED
tests/regression/test_backplane.py::TestRoutingQuality::test_iteration_budget PASSED
tests/regression/test_backplane.py::TestRoutingQuality::test_total_time PASSED

============================== 68 passed, 3 warnings in 2.45s ==============================
```

---

## Updating Golden Baselines

When you deliberately improve the algorithm and achieve better results:

### 1. Update `tests/regression/golden_board.json`

```json
{
  "_source": "Measured YYYY-MM-DD after successful GPU routing run",
  "lattice_nodes": 446472,
  "tracks_existing": 9605,
  "vias_existing": 6021,
  ...
}
```

### 2. Update `tests/regression/golden_metrics.json`

```json
{
  "gpu": {
    "_source": "NVIDIA GPU, OrthoRoute YYYY-MM-DD",
    "_note": "Actual run: N iterations, Xs total. Thresholds = measured × 1.20 headroom.",
    "iterations_max": N,
    "total_time_s_max": X,
    ...
  }
}
```

### 3. Document the Change

Update `docs/optimization/golden_result_YYYY-MM-DD.md` with:
- New performance metrics
- What changed in the algorithm
- Comparison with previous baseline
- Reproduction instructions

---

## Troubleshooting

### No log files found
**Cause:** Regression tests look for log files in:
- `logs/` (GitHub repo)
- `<KiCad plugins>/com_github_bbenchoff_orthoroute/logs/` (plugin folder)

**Solution:** Run OrthoRoute via KiCad plugin first to generate logs.

### File parser unavailable
**Cause:** KiCadFileParser is incomplete (loads 0 pads/nets).

**Solution:** Use KiCad plugin mode instead of headless CLI mode.

### GPU tests skipped
**Cause:** CuPy not installed or CUDA unavailable.

**Solution:** Install CuPy matching your CUDA version:
```powershell
pip install cupy-cuda12x  # For CUDA 12.x
```

### Lattice size mismatch
**Cause:** Board file changed or algorithm modified grid generation.

**Solution:** Update `golden_board.json` if intentional, otherwise investigate regression.

### Performance regression (SOFT WARN)
**Cause:** Routing takes longer than baseline +20% headroom.

**Solution:** 
1. Check GPU acceleration is active (`GPU=YES` in log)
2. Profile hotspots (see `docs/optimization/`)
3. Update baseline if algorithm intentionally changed

---

## Continuous Integration (Future)

### GitHub Actions Workflow (Planned)

```yaml
name: Regression Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run fast headless tests
        run: pytest tests/regression/test_backplane.py::TestHeadlessRouting -v
      - name: Validate board signature
        run: pytest tests/regression/test_backplane.py::TestBoardLoad -v
```

**Note:** Full GPU routing tests require:
- NVIDIA GPU on CI runner
- KiCad installed
- CUDA toolkit
- ~20 minute execution time

Consider running full tests nightly instead of per-commit.

---

## References

- **Golden Result Documentation:** [docs/optimization/golden_result_2026-04-10.md](../docs/optimization/golden_result_2026-04-10.md)
- **Test Implementation:** [tests/regression/test_backplane.py](test_backplane.py)
- **Golden Board Signature:** [tests/regression/golden_board.json](golden_board.json)
- **Golden Metrics:** [tests/regression/golden_metrics.json](golden_metrics.json)
- **Test Fixtures:** [tests/conftest.py](../conftest.py)
