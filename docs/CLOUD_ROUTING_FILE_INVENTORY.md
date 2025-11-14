# Cloud Routing File Inventory

**Created:** November 11-12, 2025
**Purpose:** Complete list of files added/modified for cloud routing functionality

---

## NEW FILES (Created for Cloud Routing)

### Serialization Module
**Location:** `orthoroute/infrastructure/serialization/`

| File | Size | Purpose |
|------|------|---------|
| `__init__.py` | 5.7 KB | Public API exports |
| `orp_exporter.py` | 35 KB | ORP format export/import |
| `ors_exporter.py` | 15 KB | ORS format export/import |
| `README.md` | 5.4 KB | Module documentation |
| `MIGRATION.md` | 5.8 KB | API migration guide |

**Legacy file (kept for compatibility):**
| File | Size | Purpose |
|------|------|---------|
| `orthoroute/infrastructure/serialization.py` | 18 KB | Standalone legacy module |

### Documentation
| File | Size | Purpose |
|------|------|---------|
| `docs/cloud_routing_workflow.md` | 8.7 KB | User workflow guide |
| `CLOUD_ROUTING_IMPLEMENTATION_GUIDE.md` | ~20 KB | Technical implementation guide |
| `CLOUD_ROUTING_QUICK_START.md` | ~10 KB | Quick reference for re-implementation |
| `CLOUD_ROUTING_FILE_INVENTORY.md` | This file | Complete file inventory |

---

## MODIFIED FILES (Changes for Cloud Routing)

### main.py
**Changes:**
1. Added import for serialization functions (line ~15-20)
2. Added `run_headless()` function (line ~287-472)
3. Added headless subparser to argparse (line ~650-670)
4. Added headless mode dispatch in `main()` (line ~760-770)

**Lines added:** ~200 lines
**Location of changes:** Search for "headless" or "run_headless"

### orthoroute/presentation/gui/main_window.py
**Changes:**
1. Added import for serialization functions (line ~46-50)
2. Added Export PCB menu action in `_create_menus()` (line ~1440-1450)
3. Added Import Solution menu action in `_create_menus()` (line ~1451-1460)
4. Added `export_pcb()` method (line ~2944-2997)
5. Added `import_solution()` method (line ~2999-3110)
6. Removed `IterationMetricsLogger` initialization (line ~1855-1867) - causes import error in 22eb7db

**Lines added:** ~170 lines
**Location of changes:** Search for "export_pcb" or "import_solution"

### build.py
**Changes:**
1. Simplified package configuration (removed 'production', 'lite', 'development' variants)
2. Changed to single 'default' package
3. Minor metadata improvements

**Lines changed:** ~40 lines
**Impact:** Build process simplified, single output package

---

## PRESERVED FILES (DO NOT MODIFY - Algorithm Code)

These files should NEVER be modified when implementing cloud routing:

```
orthoroute/algorithms/manhattan/*.py (all algorithm files)
orthoroute/algorithms/manhattan/pathfinder/*.py (all pathfinder files)
```

**Reason:** Cloud routing uses the existing algorithm AS-IS. No algorithm changes needed.

---

## Dependencies

### Existing (Already in codebase)
- Python 3.8+
- NumPy
- SciPy
- PyQt6 (for GUI)

### New (For cloud routing)
- gzip (built-in Python module)
- json (built-in Python module)
- pathlib (built-in Python module)

**No new external dependencies required!**

---

## File Size Summary

**Total cloud routing additions:**
- Serialization module: ~85 KB (6 files)
- Documentation: ~39 KB (4 files)
- Code changes: ~370 lines in main.py and main_window.py

**Total impact:** ~125 KB of new code, ~370 lines of modifications

---

## Integration Points

### Where Cloud Routing Touches Algorithm Code

**1. Board Data Format**
- Cloud routing uses same `board_data` dictionary as GUI
- No special format conversion needed
- UnifiedPathFinder accepts board_data directly

**2. PathFinder Configuration**
- Uses PathFinderConfig from `orthoroute.algorithms.manhattan.unified_pathfinder`
- Default parameters used in headless mode
- Same config object as GUI mode

**3. Routing Results**
- Reads from `pf.net_paths` (dictionary of net_name → path)
- Converts paths (node indices) to geometry (traces/vias)
- Uses `pf.lattice.idx_to_coord()` for coordinate conversion

**4. No Algorithm Modifications**
- Cloud routing is pure infrastructure
- Algorithm code unchanged
- Guaranteed consistency between GUI and headless

---

## Testing

### Minimal Test (Export/Import)

```python
# Test ORP export
from orthoroute.infrastructure.serialization import export_board_to_orp
board_data = {...}  # Minimal board
export_board_to_orp(board_data, "test.ORP")

# Test ORP import
from orthoroute.infrastructure.serialization import import_board_from_orp
loaded = import_board_from_orp("test.ORP")
assert loaded['board_metadata']['filename'] == board_data['board_metadata']['filename']
```

### Full Workflow Test

```bash
# 1. Load board in GUI
# 2. Press Ctrl+E → Save test.ORP
# 3. Run headless
python main.py headless test.ORP
# 4. Verify test.ORS created
# 5. Press Ctrl+I → Load test.ORS
# 6. Verify solution displays
```

---

## Common Implementation Mistakes to Avoid

### 1. Coordinate System Inconsistency
❌ **Wrong:** Different coordinate origins in export vs import
✓ **Right:** Always use board lower-left as (0,0), PCB millimeters

### 2. Layer ID Confusion
❌ **Wrong:** Using layer index (0-17) vs. KiCad layer ID (0, 1, 31)
✓ **Right:** Use KiCad layer IDs consistently

### 3. Missing Compression
❌ **Wrong:** Saving plain JSON files (1.3 MB for small board)
✓ **Right:** Use gzip compression (~50 KB for same board)

### 4. Algorithm Coupling
❌ **Wrong:** Modifying pathfinder code for cloud routing
✓ **Right:** Cloud routing is pure infrastructure, algorithm unchanged

### 5. Parameter Serialization
❌ **Wrong:** Storing pres_fac, hist_gain in .ORP file
✓ **Right:** Parameters NOT serialized, use defaults in headless mode

---

## Verification Checklist

Before declaring cloud routing complete:

- [ ] `export_pcb_to_orp()` creates valid .ORP file
- [ ] .ORP file is gzip compressed (file size ~50KB for test board)
- [ ] `import_board_from_orp()` loads .ORP correctly
- [ ] `python main.py headless test.ORP` runs without errors
- [ ] Headless mode produces .ORS file
- [ ] .ORS file contains nets, traces, vias, metadata
- [ ] `import_solution_from_ors()` loads .ORS correctly
- [ ] GUI Export menu (Ctrl+E) works
- [ ] GUI Import menu (Ctrl+I) works
- [ ] Round-trip preserves board geometry
- [ ] Documentation is complete and accurate

---

## Known Issues (Accepted Limitations)

### 1. Existing Traces Not Exported
**Issue:** .ORP export assumes clean board (no existing traces)
**Impact:** Cannot export partially routed boards
**Workaround:** Only export clean boards
**Future:** Add trace export support

### 2. Keepout Zones Ignored
**Issue:** Keepout zones not serialized in .ORP
**Impact:** Routing may violate keepouts
**Workaround:** Manually avoid keepout areas
**Future:** Add keepout zone export

### 3. Parameter Override Not Supported
**Issue:** Cannot override PathFinder parameters via CLI
**Impact:** Must use default parameters in headless mode
**Workaround:** Modify config.py if custom parameters needed
**Future:** Add --config flag for parameter JSON

### 4. No Real-Time Progress Streaming
**Issue:** Cannot see cloud routing progress in real-time
**Impact:** Must SSH to cloud and tail logs
**Workaround:** Use `tail -f logs/run_*.log` on cloud
**Future:** Implement WebSocket streaming

---

## Additional Resources

### Cloud Provider Setup

**Vast.ai (Recommended - Cheapest)**
```bash
# Search for RTX 5090
# Filter: CUDA 12.x, 32GB+ VRAM
# Rent instance (~$0.37/hr)
# SSH into instance

# Install dependencies
pip install cupy-cuda12x numpy scipy

# Upload ORP file
# scp board.ORP user@instance:/workspace/

# Run routing
cd /workspace
python main.py headless board.ORP

# Monitor progress
tail -f logs/run_*.log

# Download solution
# scp user@instance:/workspace/board.ORS ./
```

**RunPod**
- Similar process, ~$0.40/hr for RTX 4090
- Better UI, easier setup
- Slightly more expensive

**Lambda Labs**
- ~$1.10/hr for A100
- Most expensive but reliable
- Good for critical production runs

---

## File Format Specifications (Detailed)

### ORP Format - Complete Field Reference

```json
{
  // Required fields
  "format_version": "1.0",

  "board_metadata": {
    "filename": "string",           // Original .kicad_pcb filename
    "bounds": {
      "x_min": float,               // Board bounding box in mm
      "y_min": float,
      "x_max": float,
      "y_max": float
    },
    "layer_count": int              // Total number of layers
  },

  "pads": [
    {
      "position": [float, float],   // [x, y] in mm
      "net": "string",              // Net name
      "drill": float,               // Drill diameter in mm
      "layer_mask": int             // KiCad layer bitmask
    }
  ],

  "nets": {
    "NET_NAME": [
      [float, float],               // Terminal position [x, y] in mm
      [float, float],               // Another terminal
      ...
    ]
  },

  "drc_rules": {
    "clearance": float,             // Min clearance in mm
    "track_width": float,           // Min track width in mm
    "via_diameter": float,          // Via diameter in mm
    "via_drill": float,             // Via drill diameter in mm
    "min_drill": float              // Minimum drill size in mm
  },

  "grid_parameters": {
    "pitch": float                  // Grid discretization in mm (usually 0.4)
  },

  // Optional fields
  "components": [                   // Component reference (optional)
    {
      "reference": "U1",
      "position": [float, float],
      "rotation": float
    }
  ]
}
```

### ORS Format - Complete Field Reference

```json
{
  // Required fields
  "format_version": "1.0",

  "nets": {
    "NET_NAME": {
      "traces": [
        {
          "layer": "string",        // Layer name (e.g. "In1.Cu")
          "start": [float, float],  // Start position [x, y] in mm
          "end": [float, float],    // End position [x, y] in mm
          "width": float            // Track width in mm
        }
      ],
      "vias": [
        {
          "position": [float, float],     // Via position [x, y] in mm
          "layer_from": "string",         // Start layer (e.g. "F.Cu")
          "layer_to": "string",           // End layer (e.g. "In5.Cu")
          "diameter": float,              // Via diameter in mm
          "drill": float                  // Drill diameter in mm
        }
      ]
    }
  },

  // Optional but recommended
  "iteration_metrics": [
    {
      "iteration": int,
      "overuse_count": int,
      "nets_routed": int,
      "nets_failed": int,
      "total_overflow": float,
      "wirelength": float,
      "via_count": int,
      "runtime_seconds": float
    }
  ],

  "final_metadata": {
    "total_iterations": int,
    "converged": bool,
    "total_time_seconds": float,
    "final_wirelength": float,
    "final_via_count": int,
    "final_overflow": float,
    "timestamp": "string",          // ISO format: "2025-11-12T10:30:00"
    "orthoroute_version": "string"  // "1.0.0"
  }
}
```

---

## Implementation Notes for Next Session

### Starting Point
You will receive a codebase that has:
- ✓ Working PathFinder algorithm
- ✓ GUI with KiCad integration
- ✗ NO cloud routing functionality

### Implementation Order

**Phase 1: Core Serialization (2-3 hours)**
1. Create `orthoroute/infrastructure/serialization/` directory
2. Implement `orp_exporter.py` (ORP export/import functions)
3. Implement `ors_exporter.py` (ORS export/import functions)
4. Create `__init__.py` with public API
5. Test imports work

**Phase 2: Headless Mode (1 hour)**
1. Add `run_headless()` function to main.py
2. Add headless argparse subcommand
3. Add mode dispatch in main()
4. Test: `python main.py headless test.ORP`

**Phase 3: GUI Integration (1 hour)**
1. Add export_pcb() method to main_window.py
2. Add import_solution() method to main_window.py
3. Add menu items with Ctrl+E and Ctrl+I shortcuts
4. Test in GUI

**Phase 4: Documentation (30 minutes)**
1. Copy/create docs/cloud_routing_workflow.md
2. Update user guide

**Total Time: 4-5 hours**

### Code to Copy Directly

The following can be copied verbatim from the implementation guide:
- Entire serialization module architecture
- `run_headless()` function skeleton
- Menu item creation code
- Export/import method implementations

### Code to Adapt

The following needs adaptation to existing codebase:
- Coordinate conversion (depends on Lattice3D implementation)
- Path-to-geometry conversion (depends on graph structure)
- Board object construction (depends on Board class definition)

### Testing Strategy

**After each phase:**
1. Run `python build.py` - must succeed
2. Test imports - must work
3. Run applicable workflow test
4. Fix any errors before proceeding

**Final test:**
1. Load TestBackplane in GUI
2. Export with Ctrl+E
3. Run headless mode
4. Import with Ctrl+I
5. Verify routing matches

---

## Questions for User (Before Starting)

When re-implementing, ask the user:

1. **Which git commit should be the base?** (e.g., 22eb7db, 8b2ceea, or other)
2. **Should checkpoint support be included?** (adds complexity)
3. **Should iteration metrics be exported to ORS?** (useful for convergence analysis)
4. **What level of error handling?** (basic vs comprehensive)
5. **Should parameter override be supported?** (--config flag)

---

## Success Indicators

### When Implementation is Complete

✓ User can press Ctrl+E in GUI and get .ORP file
✓ User can run `python main.py headless board.ORP` on cloud
✓ Headless mode produces .ORS file with routing solution
✓ User can press Ctrl+I in GUI and load .ORS file
✓ Imported solution displays correctly
✓ Round-trip works: KiCad → ORP → Cloud → ORS → KiCad
✓ Documentation explains complete workflow
✓ Build succeeds, no import errors
✓ Algorithm code untouched (no coupling)

---

**End of File Inventory**

**Next Claude Session:** Use this file + CLOUD_ROUTING_IMPLEMENTATION_GUIDE.md + CLOUD_ROUTING_QUICK_START.md to implement cloud routing from scratch.
