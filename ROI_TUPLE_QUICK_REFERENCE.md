# ROI Tuple Format - Quick Reference

## Standard 13-Element Format

All ROI tuples must use this format:

```python
roi_tuple = (
    roi_source,    # 0: Local source node index (int)
    roi_sink,      # 1: Local sink node index (int)
    roi_indptr,    # 2: CSR indptr array (CuPy/NumPy array)
    roi_indices,   # 3: CSR indices array (CuPy/NumPy array)
    roi_weights,   # 4: CSR weights array (CuPy/NumPy array)
    roi_size,      # 5: Number of nodes in ROI (int)
    roi_bitmap,    # 6: Bitmap mask for filtering (array or None)
    bbox_minx,     # 7: Bounding box min X (int)
    bbox_maxx,     # 8: Bounding box max X (int)
    bbox_miny,     # 9: Bounding box min Y (int)
    bbox_maxy,     # 10: Bounding box max Y (int)
    bbox_minz,     # 11: Bounding box min Z/layer (int)
    bbox_maxz      # 12: Bounding box max Z/layer (int)
)
```

## Creating ROI Tuples

### Example 1: With Bitmap Filtering
```python
roi_tuple = (
    0,              # source
    10,             # sink
    roi_indptr,     # CSR indptr
    roi_indices,    # CSR indices
    roi_weights,    # CSR weights
    100,            # roi_size
    bitmap_mask,    # bitmap (for L-corridor filtering)
    50,             # bbox_minx
    150,            # bbox_maxx
    30,             # bbox_miny
    130,            # bbox_maxy
    0,              # bbox_minz (layer 0)
    3               # bbox_maxz (layer 3)
)
```

### Example 2: Without Bitmap Filtering
```python
roi_tuple = (
    0,              # source
    10,             # sink
    roi_indptr,     # CSR indptr
    roi_indices,    # CSR indices
    roi_weights,    # CSR weights
    100,            # roi_size
    None,           # No bitmap filtering
    0,              # bbox_minx (use default)
    999999,         # bbox_maxx (use large value)
    0,              # bbox_miny (use default)
    999999,         # bbox_maxy (use large value)
    0,              # bbox_minz (use default)
    999999          # bbox_maxz (use large value)
)
```

## Validation Functions

### Manual Validation
```python
from orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra import _validate_roi_tuple

try:
    _validate_roi_tuple(my_tuple, expected_len=13)
    print("Tuple is valid")
except ValueError as e:
    print(f"Validation error: {e}")
```

### Normalization (for backward compatibility)
```python
from orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra import _normalize_roi_tuple

# Automatically converts old formats to 13-element format
normalized_tuple = _normalize_roi_tuple(old_tuple)
```

## Common Patterns

### Extracting Data from 13-Element Tuple
```python
# Get just the first 6 elements (most common case)
roi_source, roi_sink, roi_indptr, roi_indices, roi_weights, roi_size = roi_tuple[:6]

# Get specific elements
roi_bitmap = roi_tuple[6]
bbox_minx, bbox_maxx = roi_tuple[7], roi_tuple[8]
bbox_miny, bbox_maxy = roi_tuple[9], roi_tuple[10]
bbox_minz, bbox_maxz = roi_tuple[11], roi_tuple[12]
```

### Processing Batches
```python
from orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra import _normalize_roi_tuple

# Normalize all tuples in batch
roi_batch = [_normalize_roi_tuple(t) for t in raw_batch]

# Process normalized batch
for roi_tuple in roi_batch:
    src, dst = roi_tuple[0], roi_tuple[1]
    # ... process ROI
```

## Error Messages

### Validation Errors
```
ValueError: ROI tuple length mismatch: got 6, expected 13
```
**Fix**: Use 13-element format or let normalization handle it

### Normalization Errors
```
ValueError: Cannot normalize ROI tuple of length 5
```
**Fix**: Check tuple creation - only 6, 11, or 13 element formats supported

### Creation Errors
```
ValueError: ROI tuple length mismatch: 8 != 13 (expected 13-element format)
```
**Fix**: Add missing elements to match 13-element format

## Warnings

When old formats are automatically normalized, you'll see warnings:

```
[ROI-TUPLE] Normalized 6-element tuple to 13 elements
[ROI-TUPLE] Normalized 11-element tuple to 13 elements
```

**Action**: Update code to use 13-element format directly to remove warnings

## Migration Guide

### Updating Old 6-Element Code

**Before:**
```python
roi_tuple = (roi_source, roi_sink, roi_indptr, roi_indices, roi_weights, roi_size)
```

**After:**
```python
roi_tuple = (
    roi_source, roi_sink, roi_indptr, roi_indices, roi_weights, roi_size,
    None,      # roi_bitmap
    0,         # bbox_minx
    999999,    # bbox_maxx
    0,         # bbox_miny
    999999,    # bbox_maxy
    0,         # bbox_minz
    999999     # bbox_maxz
)
```

### Updating Old 11-Element Code

**Before:**
```python
roi_tuple = (roi_nodes, g2r, bbox, src, dst, via_mask, plane, xs, ys, zmin, zmax)
```

**After:**
```python
roi_tuple = (
    roi_nodes, g2r, bbox, zmin, zmax,  # Entry/exit layers
    src, dst, via_mask, plane, xs, ys, zmin, zmax
)
```

## Best Practices

1. **Always use 13-element format** in new code
2. **Validate at creation** to catch errors early
3. **Use None for unused fields** rather than dummy values where appropriate
4. **Document tuple structure** in function docstrings
5. **Test with validation functions** before deployment

## Quick Checklist

When creating ROI tuples:
- [ ] 13 elements total
- [ ] Elements 0-5: source, sink, indptr, indices, weights, size
- [ ] Element 6: bitmap (or None)
- [ ] Elements 7-12: bbox coordinates (6 values)
- [ ] Validated with `_validate_roi_tuple()` if uncertain
- [ ] Documented in function docstring

## Support

For questions or issues:
1. Check `ROI_VALIDATION_SUMMARY.md` for detailed implementation notes
2. Run `test_roi_validation.py` to verify your understanding
3. Review git diff for complete changes
