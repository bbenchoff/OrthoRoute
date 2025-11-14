# Cloud Routing Quick Start - Implementation Checklist

**For the next Claude session implementing cloud routing from scratch**

---

## What Cloud Routing Does

Allows routing large boards on rented cloud GPUs:
1. Export board geometry to `.ORP` file (no KiCad needed)
2. Upload to cloud, run headless: `python main.py headless board.ORP`
3. Download `.ORS` solution file
4. Import back into KiCad

**Cost:** $5-10 per large board vs. buying expensive hardware

---

## Files to Create

### 1. Serialization Module (4 files)

**`orthoroute/infrastructure/serialization/__init__.py`**
```python
from .orp_exporter import export_board_to_orp, import_board_from_orp, derive_orp_filename
from .ors_exporter import export_solution_to_ors, import_solution_from_ors, derive_ors_filename, get_solution_summary

# Aliases
export_pcb_to_orp = export_board_to_orp
import_pcb_from_orp = import_board_from_orp
```

**`orthoroute/infrastructure/serialization/orp_exporter.py`**
- Implement `export_board_to_orp(board_data, filepath, compress=True)`
- Implement `import_board_from_orp(filepath)`
- Implement `derive_orp_filename(kicad_filename)`
- Format: JSON with gzip compression
- Contains: board metadata, pads, nets, DRC rules, grid parameters

**`orthoroute/infrastructure/serialization/ors_exporter.py`**
- Implement `export_solution_to_ors(solution_data, filepath, compress=True)`
- Implement `import_solution_from_ors(filepath)`
- Implement `derive_ors_filename(orp_filename)`
- Implement `get_solution_summary(solution_data)` - returns formatted string
- Format: JSON with gzip compression
- Contains: per-net traces/vias, iteration metrics, final metadata

**`docs/cloud_routing_workflow.md`**
- Copy from current directory (already written)
- Documents complete user workflow

### 2. Modify main.py (2 additions)

**Add headless subparser:**
```python
headless_parser = subparsers.add_parser('headless')
headless_parser.add_argument('orp_file')
headless_parser.add_argument('-o', '--output')
headless_parser.add_argument('--max-iterations', type=int, default=200)
```

**Add run_headless() function:**
- Import board from .ORP
- Create UnifiedPathFinder
- Run routing
- Export solution to .ORS
- ~150 lines total

### 3. Modify main_window.py (2 methods + menu items)

**Add to File menu:**
```python
export_pcb_action = QAction("Export PCB...", self)
export_pcb_action.setShortcut("Ctrl+E")
export_pcb_action.triggered.connect(self.export_pcb)

import_solution_action = QAction("Import Solution...", self)
import_solution_action.setShortcut("Ctrl+I")
import_solution_action.triggered.connect(self.import_solution)
```

**Add methods:**
- `export_pcb()` - Shows save dialog, calls export_pcb_to_orp()
- `import_solution()` - Shows open dialog, calls import_solution_from_ors()

---

## Key Implementation Details

### ORP Format (Board Export)

```json
{
  "format_version": "1.0",
  "board_metadata": {
    "filename": "board.kicad_pcb",
    "bounds": {"x_min": 100, "y_min": 50, "x_max": 200, "y_max": 150},
    "layer_count": 6
  },
  "pads": [
    {"position": [150.5, 100.2], "net": "VCC", "drill": 0.8, "layer_mask": 4227858432}
  ],
  "nets": {
    "VCC": [[150.5, 100.2], [175.3, 120.5]],
    "GND": [[...]]
  },
  "drc_rules": {
    "clearance": 0.15,
    "track_width": 0.25,
    "via_diameter": 0.6,
    "via_drill": 0.3,
    "min_drill": 0.2
  },
  "grid_parameters": {"pitch": 0.4}
}
```

**Save with gzip:**
```python
import gzip, json
with gzip.open(filepath, 'wt', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
```

### ORS Format (Solution Export)

```json
{
  "format_version": "1.0",
  "nets": {
    "VCC": {
      "traces": [
        {"layer": "In1.Cu", "start": [150.5, 100.2], "end": [151.5, 100.2], "width": 0.25}
      ],
      "vias": [
        {"position": [150.5, 100.2], "layer_from": "F.Cu", "layer_to": "In1.Cu", "diameter": 0.6, "drill": 0.3}
      ]
    }
  },
  "iteration_metrics": [
    {"iteration": 1, "overuse": 42000, "nets_routed": 512, "runtime_seconds": 45.2}
  ],
  "final_metadata": {
    "total_iterations": 12,
    "converged": true,
    "total_time_seconds": 540,
    "timestamp": "2025-11-12T10:30:00"
  }
}
```

### Headless Mode Core Logic

```python
def run_headless(orp_file, output_file=None, max_iterations=200):
    # 1. Import board
    board_data = import_board_from_orp(orp_file)

    # 2. Create Board object
    board = Board(name=board_data['board_metadata']['filename'],
                  layer_count=board_data['board_metadata']['layer_count'])

    # 3. Initialize PathFinder
    config = PathFinderConfig(max_iterations=max_iterations)
    pf = UnifiedPathFinder(config=config)
    pf.initialize_graph(board, board_data)

    # 4. Route
    result = pf.route_multiple_nets(board.nets)

    # 5. Extract solution
    solution_data = {
        "format_version": "1.0",
        "nets": {},  # Extract from pf.net_paths
        "final_metadata": {
            "total_iterations": result['iterations'],
            "converged": result['converged']
        }
    }

    # 6. Export
    if output_file is None:
        output_file = derive_ors_filename(orp_file)
    export_solution_to_ors(solution_data, output_file)
```

---

## Implementation Priority

**Priority 1 (Core Functionality):**
1. ORP export (board → file)
2. ORP import (file → board_data)
3. Headless mode (basic routing)

**Priority 2 (User Experience):**
4. GUI export menu (Ctrl+E)
5. Filename derivation helpers
6. Basic error handling

**Priority 3 (Polish):**
7. ORS export (solution → file)
8. ORS import (file → solution)
9. GUI import menu (Ctrl+I)
10. Solution preview/metrics

**Priority 4 (Documentation):**
11. User workflow guide
12. Cloud provider setup instructions

---

## Testing Commands

```bash
# Test serialization imports
python -c "from orthoroute.infrastructure.serialization import export_pcb_to_orp"

# Test headless mode help
python main.py headless --help

# Test export (requires board loaded in GUI)
# File → Export PCB → test.ORP

# Test headless routing
python main.py headless test.ORP

# Test import (requires GUI)
# File → Import Solution → test.ORS
```

---

## Estimated Implementation Time

- Serialization module: 2-3 hours
- Headless mode in main.py: 1 hour
- GUI export/import: 1 hour
- Testing and debugging: 2 hours
- **Total:** 6-7 hours for complete implementation

---

## Success Criteria

✓ Can export .ORP file from GUI
✓ Can run headless routing: `python main.py headless board.ORP`
✓ Produces .ORS solution file
✓ Can import .ORS back into GUI
✓ Round-trip works: Export → Route → Import

---

**This document + CLOUD_ROUTING_IMPLEMENTATION_GUIDE.md contain everything needed to re-implement cloud routing in a fresh codebase.**
