# OrthoRoute Test Suite

All test files live in this directory. No test-related documentation exists outside it.

```
tests/
├── conftest.py                   # Shared fixtures (board, router, routing_result, log)
├── unit/                         # Fast headless tests (~2 s, no KiCad required)
│   ├── test_domain_models.py
│   ├── test_logging_utils.py
│   ├── test_performance_utils.py
│   └── test_routing_engine.py
└── regression/                   # Full-pipeline tests (require KiCad run first)
    ├── run_headless_routing.py    # Full headless backplane routing + golden comparison
    ├── test_backplane.py
    ├── golden_board.json          # Board signature — update if .kicad_pcb changes
    └── golden_metrics.json        # Routing baselines — update after algorithm improvements
```

---

## Quick start

```powershell
# Unit tests only (fast, no KiCad required)
python -m pytest tests/unit/ -v

# Unit tests with coverage report
python -m pytest tests/unit/ -v --cov=orthoroute --cov-report=term-missing

# Full regression suite (needs logs/latest.log from a KiCad run)
python -m pytest tests/regression/ -v

# Full headless backplane route + timeout/log completion check + golden compare
python tests/regression/run_headless_routing.py

# Log-only validation against timeout/completion (no reroute)
python tests/regression/run_headless_routing.py --log-only

# Everything
python -m pytest tests/ -v
```

---

## Unit tests — current coverage

**Result:** 26 passed — no failures, no skips  
**Last verified:** 2026-04-05

| File under test | Test file | Tests | Notes |
|---|---|---|---|
| `domain/models/board.py` | `test_domain_models.py` | 12 | `Coordinate`, `Bounds`, `Pad`, `Net`, `Board` |
| `shared/utils/logging_utils.py` | `test_logging_utils.py` | 3 | `init_logging()` — console/file handler levels |
| `shared/utils/performance_utils.py` | `test_performance_utils.py` | 3 | `@profile_time` — return value, no-op, debug emit |
| `algorithms/manhattan/unified_pathfinder.py` | `test_routing_engine.py` | 8 | `PathFinderConfig`, `UnifiedPathFinder` init, GPU, pipeline API |

### test_domain_models.py — 12 tests

| Class | Test | What it checks |
|---|---|---|
| `TestCoordinate` | `test_creation` | `Coordinate(x, y)` constructs |
| `TestCoordinate` | `test_immutable` | Frozen dataclass rejects mutation |
| `TestCoordinate` | `test_equality` | Value equality on identical coords |
| `TestBounds` | `test_width_height` | `Bounds.width` / `.height` computed properties |
| `TestPad` | `test_minimal_pad` | `Pad(id, component_id, net_id, position, size)` |
| `TestPad` | `test_pad_defaults` | Optional fields are None by default |
| `TestNet` | `test_net_has_pads` | Pads attached via `Net.pads` list |
| `TestNet` | `test_empty_net_not_routable` | `Net.is_routable` → `False` with < 2 pads |
| `TestNet` | `test_net_with_two_pads_is_routable` | `Net.is_routable` → `True` with ≥ 2 pads |
| `TestBoard` | `test_board_holds_nets` | `Board.nets` list populated |
| `TestBoard` | `test_board_net_count` | `len(board.nets)` correct |
| `TestBoard` | `test_board_add_net` | Nets can be appended post-construction |

### test_logging_utils.py — 3 tests

| Class | Test | What it checks |
|---|---|---|
| `TestInitLogging` | `test_normal_mode_console_level` | Console handler level is `ERROR` (not WARNING) |
| `TestInitLogging` | `test_debug_mode_file_level` | `ORTHO_DEBUG=1` sets file handler to `DEBUG` |
| `TestInitLogging` | `test_no_debug_console_filters_warnings` | Normal mode console suppresses WARNING |

### test_performance_utils.py — 3 tests

| Class | Test | What it checks |
|---|---|---|
| `TestProfileTime` | `test_returns_correct_value` | Decorated function return value preserved |
| `TestProfileTime` | `test_noop_without_debug` | No log emitted when `ORTHO_DEBUG` unset |
| `TestProfileTime` | `test_emits_profile_log_with_debug` | `[PROFILE]` WARNING log emitted when `ORTHO_DEBUG=1` |

### test_routing_engine.py — 8 tests

| Class | Test | Marker | What it checks |
|---|---|---|---|
| `TestPathFinderConfig` | `test_default_config_instantiation` | — | `PathFinderConfig()` creates without error |
| `TestPathFinderConfig` | `test_config_has_hotset_cap` | — | `hotset_cap` present and > 0 |
| `TestUnifiedPathFinderInit` | `test_instantiation_cpu_mode` | — | CPU-mode router constructs |
| `TestUnifiedPathFinderInit` | `test_instantiation_gpu_mode` | `requires_gpu` | GPU-mode router constructs when CUDA available |
| `TestUnifiedPathFinderInit` | `test_cuda_dijkstra_importable` | `requires_gpu` | `CUDADijkstra` kernel class importable |
| `TestUnifiedPathFinderInit` | `test_has_required_pipeline_methods` | — | `initialize_graph`, `map_all_pads`, `route_multiple_nets` callable |
| `TestRouteMultipleNetsReturnShape` | `test_empty_request_returns_minimal_dict` | — | Empty input → returns `{}` (no crash) |
| `TestRouteMultipleNetsReturnShape` | `test_required_keys_are_defined` | — | Non-empty result includes all 15 required keys |

### What is NOT covered by unit tests

These areas belong in `tests/regression/` or require a running KiCad process:

| Area | Why excluded |
|---|---|
| `infrastructure/kicad/rich_kicad_interface.py` | Requires KiCad IPC process |
| `infrastructure/kicad/file_parser.py` | Covered by dedicated parser tests and headless regression runner |
| `presentation/pipeline.py` — full pipeline run | Requires a board with pads |
| `algorithms/manhattan/unified_pathfinder.py` — `_pathfinder_negotiation` | ~6 k-line monolith; full run too slow for unit suite |
| `presentation/gui/main_window.py` | Requires PyQt6 display |

### GPU test behaviour

Two tests are gated with `@requires_gpu` (auto-skipped when no CUDA device is present):

```python
requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="No CUDA GPU available")
```

Current platform: **NVIDIA T1200 Laptop GPU, 4 GB, CuPy 14.0.1** — both GPU tests run and pass.

### Adding new unit tests

1. Create `tests/unit/test_<module>.py`.
2. One test class per public API — no full routing runs.
3. Guard GPU/KiCad imports with `pytest.importorskip` or a `requires_x` marker.

---

## Regression tests — overview

Board under test: **TestBackplane.kicad_pcb** — 18 copper layers, 1,604 pads, 512 routable nets.

### Test groups

| Group | Class(es) | Trigger | Current status |
|---|---|---|---|
| **A** — Log health | `TestLogHealth`, `TestIterationMetrics` | `log_content` fixture (needs `logs/latest.log`) | Runs when a routing log exists |
| **A2** — Board load | `TestBoardLoad` | `log_content` fixture | Skipped (no log file headlessly) |
| **B** — Routing quality | `TestRoutingQuality` | `routing_result` fixture (board must load with ≥ 10 pads) | Runs with KiCad or headless parser load |
| **C** — Write-back | `TestWriteBack` | `routing_result` + KiCad IPC | Skipped (KiCad IPC not available headlessly) |

Groups B and C activate automatically when run inside a KiCad session (IPC adapter) or when the file parser successfully loads the board.

#### Why Groups A/A2 skip headlessly

The `log_path` fixture calls `pytest.skip()` when `logs/latest.log` doesn't exist.
Run OrthoRoute via KiCad with `ORTHO_DEBUG=1` first — then re-run the regression suite.

#### Why Group C still skips headlessly

Write-back tests require an active KiCad IPC session. Headless routing validates
algorithmic convergence but does not apply tracks/vias to a live KiCad board.

### Pass/fail policy

**Hard fail** → `AssertionError` → build broken, must fix before merging.  
**Soft warn** → `warnings.warn(UserWarning)` → visible in output, never blocks CI.

| Test | Hard fail? | What it checks |
|---|---|---|
| `test_gpu_mode_detected` | soft warn | GPU preferred; warns if CPU used |
| `test_gpu_mode_matches_available_hardware` | soft warn | GPU=YES ↔ CUDA kernels compiled |
| `test_all_nets_routed` | **YES** | `nets_routed == total_nets` |
| `test_convergence` | **YES** | `converged == True` |
| `test_result_has_required_key[*]` | **YES** | All 15 return-dict keys present |
| `test_iteration_budget` | soft warn | Iterations ≤ `active_metrics.iterations_max` |
| `test_total_time` | soft warn | Total time ≤ mode-specific threshold |
| `test_overuse_final` | soft warn | `overuse_final == 0` |
| `test_iter_stability` | soft warn | No iteration >3× mode avg limit |
| `test_no_barrel_conflicts` | soft warn | `barrel_conflicts == 0` |
| `test_no_excluded_nets` | soft warn | `excluded_nets == 0` |

### Golden files

| File | Purpose | When to update |
|---|---|---|
| `regression/golden_board.json` | Board signature (pads, layers, lattice size, …) | When `TestBackplane.kicad_pcb` changes |
| `regression/golden_metrics.json` | Routing baseline thresholds | After an intentional algorithm improvement |

### How to run a full regression

```powershell
# 1. Route the board via KiCad with debug logging
$env:ORTHO_DEBUG = '1'
# launch KiCad, open TestBoards/TestBackplane.kicad_pcb, run the plugin

# 2. Run all regression groups
python -m pytest tests/regression/ -v

# 3. Headless CI — only unit tests run without KiCad
python -m pytest tests/unit/ -v
```

---

## Regression test metrics

Track routing quality results over time.  
`iteration_metrics` values come from the `[ITER N]` log lines written by `_pathfinder_negotiation()`.

### Baselines (`golden_metrics.json`)

The file contains separate `"gpu"` and `"cpu"` blocks. The `active_metrics` conftest fixture auto-selects the right block by reading `PathFinder loaded (GPU=YES/NO)` from the log.

#### GPU baseline — NVIDIA T1200 Laptop (4 GB)

| Metric | Threshold | Type |
|---|---|---|
| `nets_routed` | 512 / 512 | HARD FAIL if < 100% |
| `converged` | `true` | HARD FAIL if false |
| `iterations_max` | 60 | SOFT WARN if exceeded |
| `total_time_s_max` | 900 s | SOFT WARN if exceeded |
| `iter_avg_time_s_max` | 25 s/iter | SOFT WARN if exceeded |
| `iter_1_time_s_max` | 120 s | SOFT WARN (first iter = 512 nets × ~100 ms/net) |
| `overuse_final_max` | 0 | SOFT WARN |

#### CPU-only baseline (--cpu-only flag)

| Metric | Threshold | Type |
|---|---|---|
| `nets_routed` | 512 / 512 | HARD FAIL if < 100% |
| `converged` | `true` | HARD FAIL if false |
| `iterations_max` | 120 | SOFT WARN if exceeded |
| `total_time_s_max` | 7200 s | SOFT WARN if exceeded |
| `iter_avg_time_s_max` | 120 s/iter | SOFT WARN if exceeded |
| `overuse_final_max` | 0 | SOFT WARN |

> CPU baseline values are estimates — update `golden_metrics.json["cpu"]` after first CPU run:
> ```powershell
> python main.py cli TestBoards/TestBackplane.kicad_pcb --cpu-only
> ```

Board signature: 1,604 pads · 512/1,088 routable nets · 446,472 lattice nodes · 14,281,664 edges · 5,315 existing tracks · 3,267 existing vias.

### Progress log

Add a row after each meaningful algorithm change or full KiCad run.

#### 2026-04-05 — First live KiCad run (IPC connected, GPU)

| Metric | Value | vs. baseline |
|---|---|---|
| Compute mode | GPU (T1200, 4 GB) | ✅ |
| IPC adapter | kipy IPC (pynng) | ✅ |
| Nets routed | in progress | — |
| Converged | in progress | — |
| Iterations | in progress | — |
| Total time | in progress | — |
| Avg iter time | ~0.085 s/net × 512 = ~43 s/iter | ✅ |
| Overuse (final) | in progress | — |
| Existing tracks | 5,315 | ✅ (golden) |
| Existing vias | 3,267 | ✅ (golden) |

> First successful IPC connection + board load + GPU routing run.
> Board: 73×97 mm, ρ=0.161 (SPARSE), 446,472 nodes, 14,281,664 edges.
> GPU sort: 14.3M edges in 1.2 s (11.9M edges/sec).
> Update table when routing completes.

#### 2026-04-03 — Baseline run (RTX Turing GPU)

| Metric | Value | vs. baseline |
|---|---|---|
| Nets routed | 512 / 512 | ✅ PASS |
| Converged | yes | ✅ PASS |
| Iterations | — | — |
| Total time | — | — |
| Avg iter time | — | — |
| Overuse (final) | 0 | ✅ |
| Existing tracks | 5,315 | ✅ (golden) |
| Existing vias | 3,267 | ✅ (golden) |

> Source: `debug_output/run_20260403_184510/` log.
> This is the reference run used to populate `golden_board.json` and `golden_metrics.json`.

<!-- TEMPLATE — copy for each new run
#### YYYY-MM-DD — [Description of change]

| Metric | Value | vs. baseline |
|---|---|---|
| Nets routed | ? / 512 | ✅ / ⚠️ / ❌ |
| Converged | yes / no | ✅ / ❌ |
| Iterations | ? | ✅ / ⚠️ |
| Total time | ? s | ✅ / ⚠️ |
| Avg iter time | ? s/iter | ✅ / ⚠️ |
| Overuse (final) | ? | ✅ / ⚠️ |
| Tracks delta | +? | ✅ / ⚠️ |
| Vias delta | +? | ✅ / ⚠️ |

> Notes: what changed, GPU / CPU, any skipped groups.
-->

### Updating baselines

```powershell
# Edit tests/regression/golden_metrics.json with new thresholds
# Edit tests/regression/golden_board.json if the board file changed
# Add a row to the Progress log above
```

`*_max` fields are soft-warn upper bounds.
`nets_routed` and `converged` are hard-fail values — do not lower them without explicit approval.

#### Original tolerance rationale

| Metric | Tolerance | Rationale |
|---|---|---|
| `nets_routed` | exact match | Must not regress routing success |
| `overuse_edges` | must be 0 | Must not regress convergence |
| `iterations` | ±10% | PathFinder is non-deterministic |
| `total_time_s` | ≤ 110% of golden | Must not be more than 10% slower |

> The current implementation uses `*_max` thresholds in `golden_metrics.json` rather than percentage tolerances. The table above is the original design intent — useful when deciding how tight to set thresholds.

#### Future: `--export-metrics` CLI flag

```powershell
python main.py cli TestBoards/TestBackplane.kicad_pcb --export-metrics tests/regression/golden_metrics.json
```
Would emit routing result as JSON directly from the routing run. **Not yet implemented.**

---

## TODO — unit tests still needed

Priority order: top items cover logic most likely to silently regress.

### 🔴 HIGH — routing algorithm core

| Module | Class / function | Test ideas |
|---|---|---|
| `unified_pathfinder.py` | `Lattice3D` / `_build_lattice` | `node_count == Nx × Ny × Nz`; no diagonal adjacency; H/V layer discipline alternates |
| `unified_pathfinder.py` | CSR construction | `indptr[-1] == len(indices)`; no self-loops; every node degree ≥ 2 |
| `unified_pathfinder.py` | `EdgeAccountant.commit_path` / `clear_path` | Usage increments on commit; decrements on clear; `verify_present_matches_canonical` passes |
| `unified_pathfinder.py` | via pooling — `via_col_use` | Column usage increments correctly; capacity limit enforced; idempotent at cap |

**Skeleton — lattice:**
```python
# tests/unit/test_lattice.py
def test_lattice_node_count():
    lat = Lattice3D(bounds=(0, 0, 10, 10), pitch=1.0, layers=6)
    assert lat.num_nodes == lat.x_steps * lat.y_steps * 6

def test_layer_directions_alternate():
    lat = Lattice3D(bounds=(0, 0, 10, 10), pitch=1.0, layers=4)
    assert lat.get_legal_axis(0) == 'v'  # F.Cu = vertical
    assert lat.get_legal_axis(1) == 'h'  # In1  = horizontal
```

**Skeleton — via pooling:**
```python
# tests/unit/test_via_pooling.py
def test_via_column_accounting():
    pf = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
    pf._increment_via_column_use(x=5, y=10)
    assert pf.via_col_use[5, 10] == 1
```

### 🟡 MEDIUM — infrastructure adapters

| Module | What to test |
|---|---|
| `infrastructure/kicad/file_parser.py` | `load_board()` on a small synthetic `.kicad_pcb` returns correct pad/net counts |
| `infrastructure/kicad/rich_kicad_interface.py` | nm→mm coordinate conversion; layer name normalisation (`BL_F_Cu` → `F.Cu`); keepout dict schema keys |
| `algorithms/manhattan/pad_escape_planner.py` | Escape via within board bounds; DRC clearance from neighbouring pads; retry on conflict |

### 🟡 MEDIUM — domain services

| Module | What to test |
|---|---|
| `domain/services/drc_checker.py` (if exists) | Clearance violation detected; keepout region blocks routing |
| `algorithms/manhattan/parameter_derivation.py` | Derived pitch / clearance values in valid range for known inputs |

### 🟢 LOW — CLI integration

| What to test |
|---|
| `main.py cli TestBoards/TestBackplane.kicad_pcb` exits 0 and emits routing summary (subprocess smoke test) |
| Future `--export-metrics <path>` flag writes valid JSON |

---

## TODO — regression improvements still needed

### 🔴 HIGH — make Groups B/C run headlessly

| Item | Detail |
|---|---|
| Replace `KiCadFileParser` with a fixture-level mock board | Build a `Board` domain object directly from the known golden values (1,604 pads, 512 nets) so Groups B/C run without KiCad IPC or a real file parse |
| Add `--export-metrics` flag to `main.py` | Emit routing result as JSON; allows CI to ingest metrics without parsing log files |

**Mock board skeleton:**
```python
# tests/conftest.py (alternative board_object path)
def _make_synthetic_board():
    """Build a Board from golden_board.json values — no file I/O needed."""
    from orthoroute.domain.models.board import Board, Net, Pad, Coordinate
    board = Board(id="golden", name="TestBackplane")
    # populate 512 routable nets with 2 pads each from stored positions
    # (positions can be loaded from a compact fixture CSV)
    return board
```

### 🟡 MEDIUM — expand Group A log checks

| Item | Detail |
|---|---|
| Assert `[STEP5]` deterministic lattice line present | Detects silent fallback to wrong lattice size |
| Assert IPC adapter chosen (not SWIG or file fallback) | `test_ipc_adapter_used` is currently a soft warn; make HARD FAIL once IPC is stable |
| Assert `[PREFLIGHT]` and `[LATTICE]` lines both appear | Confirms pipeline steps 2 and 4 ran; catches short-circuits |
| Assert per-iteration `nets_routed` is monotonically non-decreasing | Detects net loss between iterations |

### 🟡 MEDIUM — expand Group B routing quality checks

| Item | Detail |
|---|---|
| Track/via delta exact counts (not just `> 0`) | Record `tracks_after − tracks_before`; warn if < `tracks_delta_min` |
| Per-net failure list in result | `failed_nets` should be empty; log net names for diagnosis if not |
| `iteration_metrics` list non-empty | Confirms `_pathfinder_negotiation` populated the per-iter dict |
| `iteration_metrics[-1].nets_routed == total_nets` | Final iteration must have all nets placed |

### 🟢 LOW — CI integration

| Item | Detail |
|---|---|
| GitHub Actions workflow running `tests/unit/` on every push | No KiCad required; fast domain/utils regression feedback |
| Scheduled weekly run against KiCad via self-hosted runner | Requires KiCad 9+ and `ORTHO_DEBUG=1`; posts metrics to PR comment |
| Threshold auto-tightening script | After 3 consecutive runs within X% of golden, auto-reduce `*_max` values by 5% |
