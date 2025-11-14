# OrthoRoute Serialization Module

This module provides serialization and deserialization functionality for cloud routing workflows. It defines two JSON-based formats that are completely independent of KiCad:

## File Formats

### .ORP (OrthoRoute PCB)
Board representation format containing all data needed for routing:

- **Board metadata**: filename, bounds, layer count
- **Pads**: position, net assignment, drill size, layers
- **Nets**: terminal positions, connectivity
- **DRC rules**: clearance, track width, via parameters
- **Grid parameters**: discretization settings
- **Format version**: "1.0"

All coordinates are in PCB space (millimeters).

### .ORS (OrthoRoute Solution)
Routing results format containing geometry and metrics:

- **Per-net geometry**: traces and vias organized by net
- **Per-iteration metrics**: convergence data for each iteration
- **Final metadata**: total iterations, convergence status, timing
- **Statistics**: wirelength, via count, overflow
- **Format version**: "1.0"

## Usage

### Exporting Board to ORP

```python
from orthoroute.domain.models.board import Board
from orthoroute.infrastructure.serialization import export_board_to_orp

# Load or create a Board object
board = load_board_from_kicad("design.kicad_pcb")

# Export to ORP format
export_board_to_orp(board, "design.orp")
```

### Importing Board from ORP

```python
from orthoroute.infrastructure.serialization import import_board_from_orp

# Import ORP data
orp_data = import_board_from_orp("design.orp")

# Access board data
print(f"Board has {len(orp_data['nets'])} nets")
print(f"Board bounds: {orp_data['board']['bounds']}")
```

### Exporting Routing Solution to ORS

```python
from orthoroute.algorithms.manhattan import UnifiedPathFinder
from orthoroute.infrastructure.serialization import export_solution_to_ors

# Run routing
router = UnifiedPathFinder(...)
router.route_all_nets()

# Get geometry and metrics
geometry = router.get_geometry_payload()
iteration_metrics = []  # Collect during routing

# Export metadata
metadata = {
    "total_iterations": router.iteration,
    "converged": router.converged,
    "total_time": router.total_time,
    "final_wirelength": geometry.total_wirelength(),
    "final_via_count": len(geometry.vias),
    "final_overflow": 0,
    "board_name": "MyBoard",
}

# Export solution
export_solution_to_ors(geometry, iteration_metrics, metadata, "solution.ors")
```

### Importing Routing Solution from ORS

```python
from orthoroute.infrastructure.serialization import (
    import_solution_from_ors,
    convert_ors_to_geometry_payload,
    get_solution_summary
)

# Import solution
geometry_data, metadata = import_solution_from_ors("solution.ors")

# Convert to geometry payload for visualization
geometry = convert_ors_to_geometry_payload(geometry_data)

# Generate human-readable summary
summary = get_solution_summary({"geometry": geometry_data, **metadata})
print(summary)

# Access specific net geometry
for net_id, net_geom in geometry_data['by_net'].items():
    print(f"Net {net_id}: {len(net_geom['tracks'])} tracks, {len(net_geom['vias'])} vias")
```

### Utility Functions

```python
from orthoroute.infrastructure.serialization import (
    derive_orp_filename,
    derive_ors_filename
)

# Derive filenames
orp_file = derive_orp_filename("design.kicad_pcb")  # -> "design.ORP"
ors_file = derive_ors_filename(orp_file)             # -> "design.ORS"
```

## Design Decisions

### 1. KiCad Independence
The serialization formats do not depend on KiCad APIs or data structures. They use the domain models from `orthoroute.domain.models`, making them suitable for:
- Cloud routing workflows
- Integration with other PCB tools
- Long-term data archival
- Cross-platform compatibility

### 2. JSON Format
JSON was chosen over binary formats for:
- Human readability for debugging
- Easy parsing in multiple languages
- Version control friendly (text diffs)
- Wide tool support

### 3. Coordinate System
All coordinates are in PCB space (millimeters) with explicit documentation. The grid discretization parameters are stored separately to allow re-routing with different settings.

### 4. Geometry Organization
The ORS format provides geometry in two ways:
- **by_net**: Organized by net ID for per-net analysis
- **all_tracks/all_vias**: Flat lists for bulk processing

This dual representation optimizes for different access patterns.

### 5. Backward Compatibility
The module provides compatibility aliases (`export_pcb_to_orp`) for existing code while using more accurate names (`export_board_to_orp`) for new code.

## File Structure

```
orthoroute/infrastructure/serialization/
├── __init__.py           # Public API and utilities
├── orp_exporter.py       # Board (PCB) serialization
├── ors_exporter.py       # Solution serialization
└── README.md            # This file
```

## Format Versioning

Both formats include a `format_version` field. Current version is "1.0". Future versions will maintain backward compatibility or provide migration tools.

## Error Handling

All functions include proper error handling:
- `ValueError`: Invalid data or format version mismatch
- `FileNotFoundError`: File doesn't exist
- `IOError`: File I/O errors

Errors are logged using Python's logging module for debugging.
