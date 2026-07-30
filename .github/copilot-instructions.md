# OrthoRoute — Agent Workspace Instructions

GPU-accelerated PCB autorouter for KiCad (PathFinder / Manhattan lattice, up to 32 layers, 3,200 pads). Research code transitioning to production. See [README](../README.md) and the [build log](https://bbenchoff.github.io/pages/OrthoRoute.html) for background.

---

## Agent Workflow — REQUIRED

**For every code or config change:**

1. **Edit** source files in the repo
2. **Sync** — run `.\copy_to_kicad.ps1` immediately after every change
3. **Launch KiCad** — run `.\launch_kicad_debug.ps1` (debug mode) or ask user to start KiCad
4. **Wait** — do nothing until the user reports back from KiCad
5. **Commit** only after explicit approval ("commit", "looks good", etc.)
6. **Never push** to remote without explicit instruction

```powershell
.\copy_to_kicad.ps1        # sync after every change
.\launch_kicad_debug.ps1   # launch KiCad with ORTHO_DEBUG=1 for testing
```

---

## Build & Run

```powershell
pip install -r requirements.txt                              # Install deps
python build.py                                              # Build: build/OrthoRoute-1.0.0.zip
python build.py --deploy                                     # Build + install to KiCad plugin folder
.\copy_to_kicad.ps1                                         # Fast dev sync (Python + config only)
python main.py plugin                                        # Run with GUI (KiCad must be open)
python main.py --test-manhattan                              # Built-in GUI acceptance test
pytest tests/                                                # Run test suite (unit + regression)
```

**Debug mode:** `.\launch_kicad_debug.ps1` or `$env:ORTHO_DEBUG = '1'` → full DEBUG log + screenshots in `debug_output/`.
Tear down: `Remove-Item Env:ORTHO_DEBUG, Env:ORTHO_SCREENSHOT_FREQ, Env:ORTHO_SCREENSHOT_SCALE -ErrorAction SilentlyContinue`

**Testing:** 167 unit tests + 63 regression tests. See [tests/README.md](../tests/README.md) and [tests/run_golden_regression.md](../tests/run_golden_regression.md).

---

## Architecture

Four-layer clean architecture — **strict dependency rule: flow inward only**:

```
presentation/  →  application/  →  domain/  →  infrastructure/
```

`domain/` has **zero** infrastructure imports — enforced by `.github/instructions/domain-layer.instructions.md`. Use DI via `application/interfaces/`.

Key patterns: Repository, Strategy (`RoutingEngine`), CQRS, `EventBus`.

### Naming Conventions

- **Interfaces**: ABCs in `application/interfaces/` (e.g., `BoardRepository`, `EventPublisher`)
- **Commands**: `<Action><Entity>Command` (e.g., `RouteNetCommand`)
- **Events**: `<Entity><Action>` (e.g., `NetRouted`, `RoutingStarted`)
- **Repositories**: `Memory<Entity>Repository` for in-memory implementations
- **Models**: One class per file in `domain/models/`; `@dataclass(frozen=True)` for value objects

---

## Key Files

| File | Purpose |
|------|---------|
| [main.py](../main.py) | Entry point (CLI, plugin, tests) |
| [orthoroute.json](../orthoroute.json) | Default config (included in built package via `build.py`) |
| [orthoroute/presentation/pipeline.py](../orthoroute/presentation/pipeline.py) | Shared execution pipeline (CLI + GUI) |
| [orthoroute/presentation/plugin/kicad_plugin.py](../orthoroute/presentation/plugin/kicad_plugin.py) | KiCad plugin entry point |
| [orthoroute/algorithms/manhattan/unified_pathfinder.py](../orthoroute/algorithms/manhattan/unified_pathfinder.py) | Main routing engine (~5,967 lines — do NOT refactor without tests) |
| [orthoroute/infrastructure/kicad/rich_kicad_interface.py](../orthoroute/infrastructure/kicad/rich_kicad_interface.py) | IPC board data extraction (pads, tracks, vias, zones, keepouts) |
| [orthoroute/presentation/gui/main_window.py](../orthoroute/presentation/gui/main_window.py) | PCB viewer (rendering + display controls) |
| [orthoroute/shared/utils/performance_utils.py](../orthoroute/shared/utils/performance_utils.py) | `@profile_time` → logs `[PROFILE] func: Xms` at WARNING — **only when `ORTHO_DEBUG=1`** |
| [orthoroute/shared/utils/logging_utils.py](../orthoroute/shared/utils/logging_utils.py) | `init_logging()` — active entry point; console ERROR+, file ERROR (normal) or DEBUG (`ORTHO_DEBUG=1`) |

**Test board:** `TestBoards/TestBackplane.kicad_pcb` — 18-layer backplane (73.1×97.3mm), 1,604 pads, 512 nets. **Golden result:** 18.4 min, 512/512 nets routed, zero overuse (Apr 10, 2026).

---

## KiCad Integration

**Plugin Installation:**
- Plugin folder: `Documents/KiCad/9.0/3rdparty/plugins/com_github_bbenchoff_orthoroute`
- Name must use underscores (not dots) for Python import compatibility
- Manual install required due to [KiCad bug #19465](https://gitlab.com/kicad/code/kicad/-/issues/19465)
- Icons (`icon-24.png`, `icon-64.png`) + metadata (`plugin.json`, `metadata.json`) required for toolbar button
- See [docs/plugin_manager_integration.md](../docs/plugin_manager_integration.md)

**Board Data Adapters (tried in order):**
1. **IPC API** (KiCad 9.0+, preferred) — via `rich_kicad_interface.py`
2. **SWIG** (pcbnew module) — legacy fallback
3. **File Parser** — ⚠️ currently broken, returns 0 pads/nets

**Logs:**
- Plugin mode: `<plugin_dir>/logs/latest.log`, `<plugin_dir>/logs/run_<timestamp>.log`
- Repo mode: `<repo>/logs/latest.log`

**GPU:** 127× speedup requires NVIDIA + CuPy. Fallback: `--cpu-only`. See [docs/cloud_gpu_setup.md](../docs/cloud_gpu_setup.md).

---

## Critical Pitfalls

1. **Logging regressions** — Console must show ERROR+ only. Per-paint-event calls in `_draw_tracks`, `_draw_vias`, `_draw_zones` must stay at DEBUG. If WARNING/INFO appears on the console after a change, revert the logger level.

2. **`UnifiedPathFinder` is a 5,967-line monolith** — Do NOT refactor without tests. Follow `.github/instructions/refactor-pathfinder.instructions.md`: one extraction at a time, test before and after.

3. **Dependency violations** — `domain/` importing from `infrastructure/` is always a bug. See `.github/instructions/domain-layer.instructions.md`.

4. **Plugin not appearing** — Check `Preferences → Plugins → Enable Python API` and restart KiCad. Check `<plugin_dir>/logs/latest.log`.

5. **`_build_owner_bitmap_for_fullgraph`** called per-net (~0.9ms × 512 = ~460ms/iter) — known optimization candidate; do not add more per-net calls in this pattern.

---

## Docs Reference

- [docs/contributing.md](../docs/contributing.md) — project status, test gaps, contribution priorities
- [docs/tuning_guide.md](../docs/tuning_guide.md) — PathFinder parameter tuning
- [docs/congestion_ratio.md](../docs/congestion_ratio.md) — convergence metrics
- [docs/barrel_conflicts_explained.md](../docs/barrel_conflicts_explained.md) — via conflict resolution
- [docs/ORP_ORS_file_formats.md](../docs/ORP_ORS_file_formats.md) — headless cloud routing formats
- [docs/layer_compaction.md](../docs/layer_compaction.md) — layer reduction strategies
- [docs/optimization/](../docs/optimization/) — profiling baselines and logging review
- [docs/optimization/golden_result_2026-04-10.md](../docs/optimization/golden_result_2026-04-10.md) — **GOLDEN STANDARD**: regression baseline, 512/512 nets, zero overuse, GPU benchmarks
