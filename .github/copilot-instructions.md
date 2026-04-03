# OrthoRoute — Agent Workspace Instructions

GPU-accelerated PCB autorouter for KiCad (PathFinder / Manhattan lattice, up to 32 layers, 3 200 pads). Research code transitioning to production. See [README](../README.md) and the [build log](https://bbenchoff.github.io/pages/OrthoRoute.html) for full background.

---

## Agent Workflow — REQUIRED

**Always follow this sequence when making any code or config change:**

1. **Make the change** — edit source files in the repo
2. **Sync to KiCad** — run `.\copy_to_kicad.ps1` immediately after every change
3. **Wait for user confirmation** — user tests in KiCad and confirms it works
4. **Commit locally** — only after explicit user approval ("commit", "looks good", etc.)
5. **Never push** to remote git without explicit user instruction

```powershell
# Step 2 — always run this after changes, before asking the user to test
.\copy_to_kicad.ps1
```

> Do NOT commit before the user has tested. Do NOT push to remote unless asked.

---

## Quick Start

### Build & Run Commands

```powershell
# Install dependencies
pip install -r requirements.txt

# Build KiCad plugin package
python build.py
# Output: build/OrthoRoute-1.0.0.zip  (also deploys package dir)
# NOTE: orthoroute.json is now included in the built package

# Run plugin with GUI (requires KiCad running with board open)
python main.py
python main.py plugin

# Run without GUI
python main.py plugin --no-gui

# CLI mode (direct board file)
python main.py cli TestBoards/TestBackplane.kicad_pcb

# Headless cloud routing
python main.py --headless board.orp --output solution.ors --max-iterations 200

# Built-in tests (no formal test suite exists)
python main.py --test-manhattan   # GUI test
python main.py --test-headless    # Headless test
python main.py --test-via         # Via pathfinding test
```

### Testing

⚠️ **Critical Gap**: No unit tests exist. This is the biggest blocker to refactoring. See [docs/contributing.md](../docs/contributing.md) for test contribution guidance.

```powershell
# Pytest (when tests exist)
pytest
pytest --cov=orthoroute

# Built-in acceptance tests
python main.py --test-manhattan  # Requires KiCad running
```

#### Environment Variables — Set Before Running Tests

| Variable | Default | Purpose |
|---|---|---|
| `ORTHO_DEBUG` | `0` | Master debug switch. `1` = full DEBUG log file + iteration screenshots |
| `ORTHO_SCREENSHOT_FREQ` | `1` | Save screenshot every N iterations (only when `ORTHO_DEBUG=1`) |
| `ORTHO_SCREENSHOT_SCALE` | `8` | PNG resolution multiplier for screenshots |

**Normal (production) run — no env vars needed:**
```powershell
# Log file captures WARNING+ (~66 milestone lines including [ROUTING START] and per-iteration
# timing: iter=Xs  total=Xs); no screenshots generated
python main.py plugin
```

**Debug / profiling run:**
```powershell
$env:ORTHO_DEBUG = '1'              # full DEBUG log + screenshots
$env:ORTHO_SCREENSHOT_FREQ = '5'   # every 5 iterations (reduce I/O)
python main.py plugin
# Analyse profiling output:
$log = "<plugin_dir>\logs\latest.log"
Get-Content $log | Select-String "\[PROFILE\]" | ...
```

**Tear down after testing:**
```powershell
Remove-Item Env:ORTHO_DEBUG, Env:ORTHO_SCREENSHOT_FREQ, Env:ORTHO_SCREENSHOT_SCALE -ErrorAction SilentlyContinue
```

---

## Architecture

### Clean Architecture / Domain-Driven Design

Four-layer architecture with **strict dependency rule** (dependencies flow inward):

```
presentation/     → CLI, GUI, KiCad plugin entry points
    ↓
application/      → Use cases, command/query handlers, orchestration
    ↓
domain/           → Pure business logic (Board, Net, RoutingEngine interface)
    ↓
infrastructure/   → KiCad integration, GPU providers, persistence
```

**Key Patterns:**
- **Repository Pattern**: Abstract data access behind interfaces (`BoardRepository`, `RoutingRepository`)
- **Strategy Pattern**: `RoutingEngine` abstract base with multiple implementations (`UnifiedPathFinder`, `ManhattanRRGRoutingEngine`)
- **CQRS**: Separate command/query handlers in `application/`
- **Event-Driven**: Domain events (`NetRouted`, `RoutingStarted`) via `EventBus`

### Layer Responsibilities

| Layer            | Purpose                                    | Example Files                                  |
|------------------|--------------------------------------------|------------------------------------------------|
| **domain/**      | Business models, abstract services         | `models/board.py`, `services/routing_engine.py`|
| **application/** | Orchestrate domain logic, port abstractions| `services/routing_orchestrator.py`, `interfaces/`|
| **infrastructure/**| External integrations, concrete adapters | `kicad/ipc_adapter.py`, `gpu/cuda_provider.py` |
| **presentation/**| User interaction, plugin entry             | `plugin/kicad_plugin.py`, `gui/`, `pipeline.py`|

### Routing Algorithm Integration

**Main Implementations:**
- **`UnifiedPathFinder`** ([unified_pathfinder.py](../orthoroute/algorithms/manhattan/unified_pathfinder.py)): Primary PathFinder negotiated congestion router (3,936 lines—needs refactoring)
- **`ManhattanRRGRoutingEngine`** ([manhattan_router_rrg.py](../orthoroute/algorithms/manhattan/manhattan_router_rrg.py)): Legacy routing resource graph router (disabled by default)

Both implement `RoutingEngine` interface from [domain/services/routing_engine.py](../orthoroute/domain/services/routing_engine.py).

**Keepout enforcement** is applied inside `initialize_graph()` via `_apply_keepout_obstacles(board)`, which runs a vectorised NumPy ray-casting PIP test over the full lattice and sets `base_cost = 1e9` on edges inside keepout polygons (planar for `keepout_tracks=True`, via edges for `keepout_vias=True`).

**Pipeline Execution** ([presentation/pipeline.py](../orthoroute/presentation/pipeline.py)):
1. Build lattice & CSR graph
2. Preflight validation
3. Map pads to lattice
4. Route nets (batch)

---

## Development Conventions

### Naming Patterns

- **Interfaces**: Abstract base classes in `application/interfaces/` (e.g., `BoardRepository`, `EventPublisher`)
- **Commands**: `<Action><Entity>Command` (e.g., `RouteNetCommand`, `LoadBoardCommand`)
- **Events**: `<Entity><Action>` (e.g., `NetRouted`, `RoutingStarted`)
- **Repositories**: `Memory<Entity>Repository` for in-memory implementations
- **Providers**: `<Technology>Provider` (e.g., `CUDAProvider`, `CPUFallbackProvider`)

### Code Organization

- **One class per file** in `domain/models/`
- **Immutable value objects**: Use `@dataclass(frozen=True)` for domain models
- **Command/query handlers**: Grouped in `application/commands/` and `application/queries/`
- **Infrastructure mirrors application**: Concrete implementations mirror application interface structure

### Dependency Direction

```python
# ✅ Correct: Infrastructure imports from application/domain
from orthoroute.domain.models.board import Board
from orthoroute.application.interfaces.board_repository import BoardRepository

# ❌ Wrong: Domain importing infrastructure
from orthoroute.infrastructure.kicad.ipc_adapter import IPCAdapter  # Never in domain/
```

---

## KiCad Integration

### Connection Strategy (Multi-Adapter Fallback)

OrthoRoute tries 3 connection methods in priority order:

1. **IPC API** (`infrastructure/kicad/ipc_adapter.py`): KiCad 9.0+ HTTP/socket API (preferred)
2. **SWIG** (`infrastructure/kicad/swig_adapter.py`): Python bindings (legacy)
3. **File Parser** (`infrastructure/kicad/file_parser.py`): Direct `.kicad_pcb` parsing (always-available fallback)

**Data extracted via IPC API** (`rich_kicad_interface.py`):
- Pads (nm→mm, layer name normalisation `BL_F_Cu` → `F.Cu`)
- Tracks and vias with round-cap geometry
- Copper zones with per-layer `filled_polygons`
- Keepout rule areas (`ZT_RULE_AREA`) with all five constraint flags from `rule_area_settings` proto

**Setup Requirements:**
- KiCad 9.0+ installed
- IPC API enabled: `Preferences → Plugins → Enable Python API`
- Restart KiCad after enabling

### Plugin Installation

⚠️ **Manual installation required** due to [KiCad bug #19465](https://gitlab.com/kicad/code/kicad/-/issues/19465).

```powershell
# 1. Build package (includes orthoroute.json)
python build.py

# 2. Deploy directly to local KiCad (Windows OneDrive-aware)
python build.py --deploy

# — OR — extract ZIP manually:
# Windows: C:\Users\<username>\Documents\KiCad\9.0\plugins\com_github_bbenchoff_orthoroute\
# Linux:   ~/.local/share/KiCad/9.0/plugins/com_github_bbenchoff_orthoroute/
# macOS:   ~/Documents/KiCad/9.0/plugins/com_github_bbenchoff_orthoroute/

# 3. Restart KiCad → Plugin button appears in PCB Editor toolbar
```

**Dev sync (faster than rebuild):**
```powershell
# copy_to_kicad.ps1 — syncs all .py sources + orthoroute.json directly to the
# local KiCad plugin folder. Resolves the destination via
# [Environment]::GetFolderPath('MyDocuments') so it works on any machine
# regardless of OneDrive / corporate folder redirection.
.\copy_to_kicad.ps1          # quiet — shows summary only
.\copy_to_kicad.ps1 -Verbose # shows each file copied
```

**Plugin logs** (written every run):
```
<plugin_dir>/logs/latest.log
<plugin_dir>/logs/run_<timestamp>.log
```

---

## GPU Acceleration

### GPU vs CPU Fallback

**GPU Provider** (`infrastructure/gpu/cuda_provider.py`):
- Requires NVIDIA GPU + CUDA Toolkit + CuPy (`pip install cupy>=10.0.0`)
- 127× speedup on PathFinder routing
- Automatic fallback to CPU if unavailable

**CPU Fallback** (`infrastructure/gpu/cpu_fallback.py`):
- Pure NumPy implementation
- Always available
- Use `--cpu-only` flag to force CPU mode

### Configuration

```python
# In plugin initialization
from orthoroute.algorithms.manhattan.unified_pathfinder import UnifiedPathFinder

router = UnifiedPathFinder(
    config=config,
    use_gpu=True  # Auto-detects GPU availability
)
```

**Memory Issues:**
- Large boards may exceed GPU VRAM
- Use `--cpu-only` or reduce board complexity
- See "Will it work with my GPU?" in [README](../README.md)

---

## Configuration & Settings

### Configuration Files

- **`orthoroute.json`**: Default routing parameters, display settings, Manhattan layer directions
- **`orthoroute/shared/configuration/`**: Configuration management system

### Key Configuration Areas

```json
{
  "routing": {
    "algorithm": "manhattan",
    "use_gpu": true,
    "max_iterations": 50000,
    "timeout_per_net": 30.0,
    "manhattan_grid_resolution": 0.05,
    "manhattan_layer_directions": {
      "In1.Cu": "horizontal",
      "In2.Cu": "vertical"
    }
  }
}
```

**Tuning:** See [docs/tuning_guide.md](../docs/tuning_guide.md) for parameter optimization.

---

## Dependencies

### Required

- **Python 3.8+** (3.10+ for plugin, 3.11 recommended)
- **NumPy ≥1.20.0**: Array operations, routing algorithms
- **psutil ≥5.8.0**: System utilities

### Optional

```toml
[gui]
PyQt6>=6.0.0                  # Graphical interface

[gpu]
cupy>=10.0.0                  # CUDA acceleration (127× speedup)

[kicad]
kicad-python>=0.5.0           # KiCad integration

[dev]
pytest>=6.0.0                 # Testing framework
black>=21.0.0                 # Code formatting
flake8>=3.8.0                 # Linting
mypy>=0.800                   # Type checking
```

**Install with extras:**
```powershell
pip install -e ".[dev,gpu,gui]"
```

---

## Key Files & Entry Points

### Critical Files

| File                                          | Purpose                                        |
|-----------------------------------------------|------------------------------------------------|
| [main.py](../main.py)                         | Main entry point (CLI, plugin, tests)          |
| [build.py](../build.py)                       | Plugin package builder (includes orthoroute.json) |
| [orthoroute.json](../orthoroute.json)         | Default configuration (copied into built package) |
| [presentation/pipeline.py](../orthoroute/presentation/pipeline.py) | Shared execution pipeline (CLI + GUI) |
| [presentation/plugin/kicad_plugin.py](../orthoroute/presentation/plugin/kicad_plugin.py) | KiCad plugin entry point |
| [algorithms/manhattan/unified_pathfinder.py](../orthoroute/algorithms/manhattan/unified_pathfinder.py) | Main routing engine (~5,967 lines) |
| [infrastructure/kicad/rich_kicad_interface.py](../orthoroute/infrastructure/kicad/rich_kicad_interface.py) | IPC board data extraction (pads, tracks, vias, zones, keepouts) |
| [presentation/gui/main_window.py](../orthoroute/presentation/gui/main_window.py) | PCB viewer — rendering + display controls |
| [domain/models/board.py](../orthoroute/domain/models/board.py) | Board aggregate root (nets, layers, keepouts) |
| [shared/utils/performance_utils.py](../orthoroute/shared/utils/performance_utils.py) | `@profile_time` decorator — logs `[PROFILE] func: Xms` at WARNING |
| [shared/utils/logging_utils.py](../orthoroute/shared/utils/logging_utils.py) | `setup_logging()` — console WARNING+, file DEBUG+ (rotating) |

### Test Boards

- **TestBoards/**: Example boards for testing
  - `TestBackplane.kicad_pcb`: Complex 32-layer backplane (3,200 pads, 870 via pairs)

---

## Documentation Reference

**Link to existing docs, don't duplicate:**

- **[docs/contributing.md](../docs/contributing.md)**: Contribution guide, project status, test gaps
- **[docs/tuning_guide.md](../docs/tuning_guide.md)**: PathFinder parameter optimization
- **[docs/ORP_ORS_file_formats.md](../docs/ORP_ORS_file_formats.md)**: Headless cloud routing formats
- **[docs/cloud_gpu_setup.md](../docs/cloud_gpu_setup.md)**: GPU rental setup (AWS, GCP, Azure)
- **[docs/layer_compaction.md](../docs/layer_compaction.md)**: Layer reduction strategies
- **[docs/congestion_ratio.md](../docs/congestion_ratio.md)**: Convergence metrics
- **[docs/barrel_conflicts_explained.md](../docs/barrel_conflicts_explained.md)**: Via conflict resolution
- **[docs/plugin_manager_integration.md](../docs/plugin_manager_integration.md)**: KiCad plugin details

---

## Common Pitfalls

### Development Environment Issues

1. **"Plugin button doesn't appear in KiCad"**
   - Verify IPC API enabled: `Preferences → Plugins → Enable Python API`
   - Check folder name: Must be exactly `com_github_bbenchoff_orthoroute` (underscores)
   - Check plugin logs: `<plugin_dir>/logs/latest.log`

2. **"No KiCad process found"**
   - KiCad must be running with a board open
   - IPC API must be enabled in preferences
   - Verify environment variables: `KICAD_API_SOCKET`, `KICAD_API_TOKEN`

3. **"Out of Memory" (GPU)**
   - Board exceeds GPU VRAM capacity
   - Use `--cpu-only` flag
   - Reduce board complexity or grid resolution

4. **Import errors / Module not found**
   - Ensure running from OrthoRoute root directory
   - `main.py` automatically adds package dir to `sys.path`
   - Check PYTHONPATH includes project root

5. **Log file flooded with identical lines during panning/zooming**
   - All per-paint-event messages in `_draw_tracks`, `_draw_vias`, `_draw_zones` are at DEBUG level
   - If you see them at INFO it means a regression — check those three methods

6. **Console output too noisy during routing**
   - Console handler is set to WARNING+ — only milestones, errors, and `[PROFILE]` lines should appear
   - If INFO messages flood the console, a logger call was accidentally promoted — check recent changes to `unified_pathfinder.py`
   - Full detail is only in the log file when `ORTHO_DEBUG=1`; default file level is `CRITICAL`

7. **Log file is empty / only has startup lines**
   - Default file log level is `WARNING` — you should see ~66 milestone lines including `[ROUTING START]` and `[ITER N]` with timing
   - Set `$env:ORTHO_DEBUG = '1'` before launching KiCad to get full DEBUG output

8. **Screenshots not being saved**
   - Screenshots are disabled in normal mode
   - Set `$env:ORTHO_DEBUG = '1'` to enable; files appear in `debug_output/run_<timestamp>/`
   - Control frequency with `ORTHO_SCREENSHOT_FREQ` and resolution with `ORTHO_SCREENSHOT_SCALE`

### Architectural Pitfalls

5. **Violating dependency rule**
   - Domain layer must have zero infrastructure dependencies
   - Use dependency injection via application interfaces
   - See "Dependency Direction" section above

6. **Modifying UnifiedPathFinder**
   - 3,936-line monolith—refactoring needed
   - Extract small classes first
   - Add tests before refactoring
   - See [docs/contributing.md](../docs/contributing.md) for guidance

---

## Known Issues

- **No unit tests**: Biggest blocker to refactoring (contributions welcome!)
- **Large files**: `unified_pathfinder.py` is 3,936 lines (needs extraction)
- **Configuration scattered**: Multiple config locations (consolidation needed)
- **KiCad plugin manager bug**: Manual installation required ([#19465](https://gitlab.com/kicad/code/kicad/-/issues/19465))

---

## PCB Viewer — Display Features

The `PCBViewer` widget (`main_window.py`) supports these display toggles (checkboxes in sidebar):

| Checkbox | What it draws |
|----------|---------------|
| Components | Footprint outlines and pad shapes |
| Tracks | Routed track segments with round caps |
| Vias | Via drill circles per layer |
| Pads | Individual pad shapes |
| Airwires | Unrouted connection lines |
| Zones | Copper fill zones (semi-transparent, layer-colored) |
| Keepouts | KiCad rule areas as dashed red semi-transparent polygons |

**Right-click on a keepout area** shows a context menu with the keepout name, affected layers, and per-constraint status (No Tracks / No Vias / No Copper Fills / No Pads / No Footprints).

**Layer visibility** checkboxes hide/show individual copper layers across all draw methods.

---

## Keepout Rule Areas

KiCad rule areas (`ZT_RULE_AREA`) are extracted in `rich_kicad_interface._extract_zones()` and stored as `board_data['keepouts']` — a list of dicts:

```python
{
    'name': str,              # Zone name (may be empty)
    'layers': List[str],      # e.g. ['In1.Cu', 'In2.Cu']
    'outline': [[x, y], ...], # Polygon vertices in mm
    'keepout_tracks': bool,   # Block routing tracks
    'keepout_vias': bool,     # Block vias
    'keepout_copper': bool,   # Block copper fills (zone pours)
    'keepout_pads': bool,     # Block pads
    'keepout_footprints': bool,
}
```

The router enforces `keepout_tracks` and `keepout_vias` via `_apply_keepout_obstacles(board)` in `UnifiedPathFinder.initialize_graph()`. `keepout_copper` affects zone fills only (KiCad DRC enforces this, not OrthoRoute).

---

## Project Status

**Working:**
- ✅ GPU-accelerated PathFinder routing (127× speedup)
- ✅ Novel portal escape architecture (16% → 80%+ routing success)
- ✅ Complex multi-layer boards (32 layers, 3,200 pads tested)
- ✅ Blind/buried via support (870 via pairs)
- ✅ Track and via extraction from KiCad IPC API
- ✅ Copper zone extraction and rendering
- ✅ Keepout rule area extraction, visualization, router enforcement, right-click inspection
- ✅ PCB viewer layer visibility controls
- ✅ `build.py` correctly packages `orthoroute.json` and validates it
- ✅ Logging reclassified in `unified_pathfinder.py` — console shows ~66 WARNING milestones per run
- ✅ Log file defaults to `CRITICAL` (normal mode); `ORTHO_DEBUG=1` enables full DEBUG detail
- ✅ Iteration screenshots disabled by default; opt-in via `ORTHO_DEBUG=1` (PNG write offloaded to background thread — zero routing stall)
- ✅ `@profile_time` decorator available in `shared/utils/performance_utils.py`
- ✅ `copy_to_kicad.ps1` — portable dev sync script (resolves KiCad path via `MyDocuments`)
- ✅ `_path_to_edges` vectorized (180× speedup); `commit_path` vectorized (9× speedup) — iter time 22s → 11s

**Needs Work:**
- ⚠️ No unit tests
- ⚠️ Large classes need refactoring (`unified_pathfinder.py` ~5,967 lines)
- ⚠️ `_build_owner_bitmap_for_fullgraph` still called per-net (~0.9ms × 512 = ~460ms/iter) — candidate for once-per-iter caching
- ⚠️ Configuration consolidation

See [docs/contributing.md](../docs/contributing.md) for detailed contribution guidance.

---

## Contact & Contribution

> "I swear to fucking god there is never going to be a discord or slack for this shit. WE'RE ALREADY ON A MESSAGING PLATFORM IT'S CALLED GITHUB YOU MAY CONTACT ME VIA PULL REQUESTS AND ISSUES"
> — [docs/contributing.md](../docs/contributing.md)

**Contribution Priority:**
1. Add unit tests (critical)
2. Extract classes from `UnifiedPathFinder`
3. Document coordinate system
4. Add type hints
5. Write usage examples

Start small—don't try to refactor the entire 3,936-line UnifiedPathFinder on your first PR. See [docs/contributing.md](../docs/contributing.md) for "Good first contributions."
