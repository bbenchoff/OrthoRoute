---
description: "Use when reading or writing KiCad IPC adapter code: pads, tracks, vias, zones, keepouts, layer names, coordinate units, protobuf fields. Covers rich_kicad_interface.py, ipc_adapter.py, and anything under infrastructure/kicad/."
applyTo: "orthoroute/infrastructure/kicad/**"
---

# KiCad IPC Adapter Conventions

These rules apply to all code under `orthoroute/infrastructure/kicad/`.

## Adapter Priority

Three adapters are tried in order — never skip the fallback chain:

1. **IPC API** (`ipc_adapter.py`) — KiCad 9.0+ socket/HTTP API. Preferred.
2. **SWIG** (`swig_adapter.py`) — Python bindings. Legacy fallback.
3. **File Parser** (`file_parser.py`) — Direct `.kicad_pcb` parsing. Always available.

Changes to one adapter must not break the others. Each implements the same interface from `application/interfaces/board_repository.py`.

## Coordinate Units

- KiCad IPC returns coordinates in **nanometres (nm)** as integers.
- All domain models use **millimetres (mm)** as floats.
- Convert at the adapter boundary — **never pass raw nm into domain objects**:

```python
# ✅ Correct
pad_x_mm = pad.position.x / 1_000_000  # nm → mm

# ❌ Wrong — nm leaks into domain
pad.position.x  # raw nm value
```

## Layer Name Normalisation

IPC API returns layer names with a bus-prefix artifact. Strip it before storing:

```python
# IPC returns:  "BL_F_Cu", "BL_In1_Cu", "BL_B_Cu"
# Domain needs: "F.Cu",    "In1.Cu",    "B.Cu"

layer_name = raw_name.removeprefix("BL_").replace("_", ".", 1)
```

Always normalize before passing layer names to domain models or the router.

## Protobuf Field Names

Key fields used in `rich_kicad_interface.py` — use these exact names, do not guess:

| Object | Proto field | Domain meaning |
|--------|-------------|----------------|
| Zone | `ZT_RULE_AREA` | Keepout / rule area (not copper fill) |
| Zone | `rule_area_settings` | Contains all five keepout constraint flags |
| Keepout flags | `keepout_tracks` | Blocks routing tracks |
| Keepout flags | `keepout_vias` | Blocks vias |
| Keepout flags | `keepout_copper` | Blocks copper fill (KiCad DRC only, router ignores) |
| Keepout flags | `keepout_pads` | Blocks pads |
| Keepout flags | `keepout_footprints` | Blocks footprints |
| Via | `drill_diameter` | Via drill size (nm) |
| Pad | `orientation` | Pad rotation in degrees |

## Keepout Extraction Pattern

Keepouts are extracted in `_extract_zones()` and returned as dicts — preserve this schema exactly:

```python
{
    'name': str,               # Zone name (may be empty string)
    'layers': List[str],       # Normalised layer names, e.g. ['In1.Cu', 'In2.Cu']
    'outline': [[x, y], ...],  # Polygon vertices in mm (already converted)
    'keepout_tracks': bool,
    'keepout_vias': bool,
    'keepout_copper': bool,    # Router does NOT enforce this flag
    'keepout_pads': bool,
    'keepout_footprints': bool,
}
```

## Router Enforcement Boundary

The router (`UnifiedPathFinder.initialize_graph`) enforces only `keepout_tracks` and `keepout_vias`.
`keepout_copper` is for KiCad DRC zone fills — do not add router enforcement for it.

## Error Handling

- IPC connection errors must be caught and trigger fallback to SWIG, not crash.
- Log failures at WARNING with the adapter name: `logger.warning("IPC adapter failed: %s", e)`.
- Never swallow errors silently in the fallback chain — always log before falling through.

## Tests

Mock the IPC socket at the adapter boundary; do not launch KiCad in unit tests.
Use `tests/infrastructure/kicad/` for adapter tests. See [docs/contributing.md](../../../docs/contributing.md).
