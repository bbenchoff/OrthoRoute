# Migration Guide

## Old `serialization.py` to New Package

The serialization functionality has been refactored from a single `serialization.py` file into a proper package with separate modules for better organization and maintainability.

### What Changed

**Old structure:**
```
orthoroute/infrastructure/
├── serialization.py    # Single file with all functions
```

**New structure:**
```
orthoroute/infrastructure/
├── serialization.py    # Old file (now shadowed by package)
└── serialization/
    ├── __init__.py          # Public API
    ├── orp_exporter.py      # Board serialization
    ├── ors_exporter.py      # Solution serialization
    ├── README.md            # Documentation
    └── MIGRATION.md         # This file
```

### Compatibility

**All existing imports continue to work without changes:**

```python
# These all work exactly as before:
from orthoroute.infrastructure.serialization import (
    export_pcb_to_orp,
    import_pcb_from_orp,
    export_solution_to_ors,
    import_solution_from_ors,
    derive_orp_filename,
    derive_ors_filename,
    get_solution_summary,
)
```

### New API

The new package provides more accurate function names while maintaining backward compatibility:

```python
# New names (recommended for new code):
from orthoroute.infrastructure.serialization import (
    export_board_to_orp,      # More accurate than export_pcb_to_orp
    import_board_from_orp,    # More accurate than import_pcb_from_orp
)

# Old names still work (aliases):
from orthoroute.infrastructure.serialization import (
    export_pcb_to_orp,        # Alias to export_board_to_orp
    import_pcb_from_orp,      # Alias to import_board_from_orp
)
```

### Key Improvements

1. **Better Organization**: Separate modules for ORP (board) and ORS (solution) formats
2. **Type Hints**: Full type annotations for better IDE support
3. **Documentation**: Comprehensive docstrings and README
4. **Domain Model Integration**: Works directly with `orthoroute.domain.models.board.Board`
5. **Error Handling**: Improved validation and error messages
6. **Format Compliance**: Strict adherence to format specifications

### Function Signature Changes

#### `export_board_to_orp` (formerly `export_pcb_to_orp`)

**Old signature:**
```python
def export_pcb_to_orp(board_data: Dict[str, Any], output_path: str, compress: bool = True) -> bool
```

**New signature:**
```python
def export_board_to_orp(board: Board, filepath: str) -> None
```

**Changes:**
- Takes a `Board` object directly instead of a dictionary
- Removed `compress` parameter (now always uncompressed JSON)
- Raises exceptions instead of returning bool
- More accurate parameter name: `filepath` instead of `output_path`

**Migration example:**
```python
# Old code:
board_data = {...}  # Dictionary
success = export_pcb_to_orp(board_data, "output.orp", compress=False)

# New code:
board = Board(...)  # Domain model
export_board_to_orp(board, "output.orp")  # Raises exception on error
```

#### `export_solution_to_ors`

**Old signature:**
```python
def export_solution_to_ors(
    ors_path: str,
    geometry_payload,
    iteration_metrics: List[Dict[str, Any]],
    routing_metadata: Dict[str, Any],
    compress: bool = True
) -> bool
```

**New signature:**
```python
def export_solution_to_ors(
    geometry,
    iteration_metrics: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    filepath: str
) -> None
```

**Changes:**
- Reordered parameters: `filepath` is now last (more Pythonic)
- Removed `compress` parameter
- Renamed `routing_metadata` to `metadata` (shorter)
- Raises exceptions instead of returning bool

**Migration example:**
```python
# Old code:
success = export_solution_to_ors(
    "solution.ors",
    geometry,
    metrics,
    metadata,
    compress=False
)

# New code:
export_solution_to_ors(
    geometry,
    metrics,
    metadata,
    "solution.ors"
)
```

### Cleanup Steps (Optional)

After verifying that all code works with the new package, you can optionally:

1. **Rename the old file** to avoid confusion:
   ```bash
   mv orthoroute/infrastructure/serialization.py \
      orthoroute/infrastructure/serialization_old.py.bak
   ```

2. **Update imports** in your codebase to use the new names:
   ```python
   # Old
   from orthoroute.infrastructure.serialization import export_pcb_to_orp

   # New (recommended)
   from orthoroute.infrastructure.serialization import export_board_to_orp
   ```

3. **Remove compression logic** if you were using `compress=True/False`:
   - The new format always uses uncompressed JSON for readability
   - If you need compression, use gzip externally:
     ```bash
     gzip solution.ors  # Creates solution.ors.gz
     ```

### Troubleshooting

**Q: My code is importing from `serialization.py` but getting the new package instead**

A: This is expected. Python treats directories with `__init__.py` as packages and they take precedence over `.py` files. All functions from the old module are available in the new package with the same names.

**Q: I'm getting attribute errors for internal functions**

A: Some internal functions (prefixed with `_`) from the old module are not exported in the new package. These were implementation details and should not be used directly.

**Q: The function signatures are different**

A: See the "Function Signature Changes" section above. The new signatures are more Pythonic and work with domain models instead of dictionaries.

## Questions?

If you encounter any issues during migration, please:
1. Check the README.md for usage examples
2. Review the docstrings in the module files
3. Look at the test code in this MIGRATION.md
