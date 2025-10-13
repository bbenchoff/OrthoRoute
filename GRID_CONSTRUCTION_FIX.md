# Grid Construction Fixes - 2025-10-11 Evening

## Issues Identified (from Debug Images)

User identified two critical grid construction bugs:

1. **Grid Bounds Too Large**: Routing grid covered entire board instead of just the routing area (wasting 4.1M nodes for mostly empty space)
2. **F.Cu (Layer 0) Had Routing Edges**: Layer 0 should be EMPTY (escape traces only), but it had vertical routing edges

## Fixes Applied

### Fix 1: Tighter Grid Bounds (`_calc_bounds`)

**File**: `orthoroute/algorithms/manhattan/unified_pathfinder.py` line 1789-1822

**Before**:
- Used all pads or full board bounds
- 3mm margin
- No filtering of unconnected pads

**After**:
- Only includes pads with nets (excludes unconnected pads)
- 5mm margin (reasonable for escape + routing space)
- Logs routing grid coverage

**Impact**: Smaller grid → less memory → faster routing

### Fix 2: F.Cu Layer Empty (`_assign_directions`)

**File**: `orthoroute/algorithms/manhattan/unified_pathfinder.py` line 1041-1051

**Before**:
```python
Layer 0 (F.Cu): 'v' (vertical routing)
Layer 1: 'h' (horizontal)
Layer 2: 'v' (vertical)
...
```

**After**:
```python
Layer 0 (F.Cu): 'empty' (NO routing, only escape traces)
Layer 1: 'h' (horizontal routing, 0.4mm pitch)
Layer 2: 'v' (vertical routing, 0.4mm pitch)
Layer 3: 'h' (horizontal)
Layer 4: 'v' (vertical)
...
```

**Implementation**:
1. `_assign_directions()` marks layer 0 as 'empty'
2. `build_graph()` skips lateral edge creation for 'empty' layers (line 1102-1104)
3. Edge count calculation skips 'empty' layers (line 1081-1082)

**Impact**: F.Cu preserved for escape traces, routing uses alternating H/V grid on inner layers

### Fix 3: Handle Missing `net` Attribute

**File**: `orthoroute/algorithms/manhattan/unified_pathfinder.py` line 1802-1803

**Before**: Checked `pad.net` (crashes if attribute doesn't exist)

**After**: Checks both `pad.net` and `pad.net_name` (different board formats)

---

## Grid Construction Rules (Now Implemented)

1. ✅ **Grid Bounds**: Bounding box of connected pads + 5mm margin
2. ✅ **Layer 0 (F.Cu)**: EMPTY - no routing edges, only escape traces
3. ✅ **Layer 1**: Horizontal traces, 0.4mm apart
4. ✅ **Layer 2**: Vertical traces, 0.4mm apart
5. ✅ **Layers 3+**: Alternating H/V pattern (H/V/H/V...)
6. ✅ **Vias**: At grid intersections (x,y) for layer transfers

---

## Expected Results After Fix

**Grid Size Reduction**:
- Before: 611×567×12 = 4,157,244 nodes (full board)
- After: Should be smaller (tight bounding box around routing area)

**Edge Count Reduction**:
- Before: ~54M edges (includes F.Cu routing edges)
- After: ~52M edges (F.Cu lateral edges removed)

**Layer 0 Behavior**:
- No horizontal or vertical routing edges on F.Cu
- Vias still allowed from F.Cu to inner layers (for escapes)
- Escape traces use F.Cu freely without competing with routing

**Routing Quality**:
- Less memory pressure → larger batch sizes possible
- Cleaner layer separation → less cross-layer conflicts
- F.Cu escapes don't interfere with pathfinding

---

## Files Modified

1. **unified_pathfinder.py**:
   - Line 1041-1051: `_assign_directions()` - mark F.Cu as 'empty'
   - Line 1081-1086: Edge count - skip 'empty' layers
   - Line 1102-1118: Lateral edge building - skip 'empty' layers
   - Line 1789-1822: `_calc_bounds()` - tighter bounding box from connected pads
   - Line 1802-1803: Handle both `net` and `net_name` attributes

---

## Testing

Run `python main.py --test-manhattan` and check logs for:

```
[INFO] Routing grid bounds: (X0, Y0) to (X1, Y1) mm
[INFO] Grid covers N connected pads (excluded unconnected pads)
[INFO] Lattice: X×Y×12 = N nodes
[INFO] Pre-allocating for M edges
```

The grid should be smaller and edge count should be reduced compared to before (was 4.1M nodes, 54M edges).

---

## Next Steps

After this fix, PathFinder should:
1. Use less memory (smaller grid)
2. Route faster (fewer nodes to explore)
3. Produce cleaner layer 0 (F.Cu) with only escape traces
4. Have proper H/V alternating grid on routing layers

The PathFinder negotiation bug (expand kernels reading base weights) was already fixed in the previous commit. Combined with these grid fixes, routing success rate should improve significantly.
