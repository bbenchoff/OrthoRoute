# Grid Bounds Fix - Continuation Guide

## Problem Statement

**Bug**: Routing traces appear outside the intended bounding box. Traces should be confined to:
- Bounding box of all F.Cu vias (created by pad escape planner) + 3mm margin

**Current Behavior**:
- Lattice is 611×567×12 = 4.1M nodes (entire board bounds)
- Only 1% of nets routing successfully (88/8192)
- Traces appearing outside intended routing area

**Expected Behavior**:
- Lattice should be constrained to portal via bounding box + 3mm
- Much smaller grid (probably ~250×200 = ~600k nodes)
- High routing success rate (>90%)

## Root Cause Identified

The `_calc_bounds()` method in `unified_pathfinder.py` is called during `initialize_graph()` BEFORE pad data is available. It tries to read `board.components` which is empty, then falls back to full `board._kicad_bounds` (even with the +3mm fix, this is still the ENTIRE board, not the portal area).

**File**: `orthoroute/algorithms/manhattan/unified_pathfinder.py`
**Method**: `_calc_bounds()` (approximately lines 1791-1826)

## What Has Been Fixed Already

✅ **lattice_builder_mixin.py** - Fixed bounds calculation with proper margin (but this code isn't being used)
✅ **Portal-based routing bounds** - Correctly calculated AFTER escape planning (logged as `[ROUTING-BOUNDS]`)
⚠️ **unified_pathfinder.py line 1812** - Added +3mm to KiCad bounds fallback (but still using full board, not portal area)

## The Core Issue

**Timing Problem**:
1. `initialize_graph()` calls `_calc_bounds()`
2. `_calc_bounds()` tries to read `board.components` → EMPTY
3. Falls back to `board._kicad_bounds` → FULL BOARD (34-278mm × 29-256mm)
4. Lattice built at full board size (4.1M nodes)
5. THEN escape planning runs, creating portals (96-257mm × 34-237mm) ← THE BOUNDS WE ACTUALLY WANT

**The Fix**: Make `_calc_bounds()` read from `board.nets` instead, which IS available during initialization.

## Exact Code Change Needed

### File: `orthoroute/algorithms/manhattan/unified_pathfinder.py`

**Replace the `_calc_bounds()` method (lines ~1791-1826) with:**

```python
def _calc_bounds(self, board: Board) -> Tuple[float, float, float, float]:
    """
    Compute routing grid bounds from pads extracted via board.nets.

    This is called during initialize_graph() BEFORE escape planning,
    so we extract pads from board.nets (which IS available) rather than
    board.components (which may be incomplete).
    """
    pads_with_nets = []
    ROUTING_MARGIN = 3.0  # mm

    # Extract pads from board.nets (reliable during initialization)
    try:
        if hasattr(board, 'nets') and board.nets:
            for net in board.nets:
                # Only consider nets with 2+ pads (routable nets)
                if hasattr(net, 'pads') and len(net.pads) >= 2:
                    for pad in net.pads:
                        if hasattr(pad, 'position') and pad.position is not None:
                            pads_with_nets.append(pad)

            if pads_with_nets:
                xs = [p.position.x for p in pads_with_nets]
                ys = [p.position.y for p in pads_with_nets]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                logger.info(f"[BOUNDS] Extracted {len(pads_with_nets)} pads from {len(board.nets)} nets")
                logger.info(f"[BOUNDS] Pad area: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")

                # Add routing margin
                bounds = (
                    min_x - ROUTING_MARGIN,
                    min_y - ROUTING_MARGIN,
                    max_x + ROUTING_MARGIN,
                    max_y + ROUTING_MARGIN
                )
                logger.info(f"[BOUNDS] Final with {ROUTING_MARGIN}mm margin: ({bounds[0]:.1f}, {bounds[1]:.1f}) to ({bounds[2]:.1f}, {bounds[3]:.1f})")
                return bounds

    except Exception as e:
        logger.warning(f"[BOUNDS] Failed to extract pads from board.nets: {e}")

    # Fallback: Use full board bounds + margin (suboptimal but safe)
    logger.warning(f"[BOUNDS] No pads found via board.nets, falling back to board._kicad_bounds + {ROUTING_MARGIN}mm")
    if hasattr(board, "_kicad_bounds"):
        b = board._kicad_bounds
        return (b[0] - ROUTING_MARGIN, b[1] - ROUTING_MARGIN,
                b[2] + ROUTING_MARGIN, b[3] + ROUTING_MARGIN)

    # Ultimate fallback
    logger.error("[BOUNDS] No bounds available, using default 100x100mm")
    return (0, 0, 100, 100)
```

## Implementation Steps

1. **Read the current file**:
   ```bash
   Read orthoroute/algorithms/manhattan/unified_pathfinder.py
   ```

2. **Locate `_calc_bounds()` method** (search for "def _calc_bounds")

3. **Replace the entire method** with the code above using the Edit tool

4. **Test the fix**:
   ```bash
   python main.py --test-manhattan
   ```

5. **Verify in logs**:
   - Look for `[BOUNDS]` log lines showing pad extraction working
   - Check lattice size - should be MUCH smaller (e.g., 400×300×12 = 1.4M nodes instead of 4.1M)
   - Check routing success rate - should be >50% instead of 1%

## Expected Test Output

**Before Fix**:
```
[BOUNDS] No pads with nets found, using board bounds
Lattice: 611×567×12 = 4,157,244 nodes
Routing: 88/8192 nets succeeded (1%)
```

**After Fix**:
```
[BOUNDS] Extracted 16384 pads from 8192 nets
[BOUNDS] Pad area: (99.2, 36.8) to (254.4, 233.6)
[BOUNDS] Final with 3.0mm margin: (96.2, 33.8) to (257.4, 236.6)
Lattice: 403×500×12 = 2,418,000 nodes  (or similar - much smaller!)
Routing: 6500/8192 nets succeeded (79%)  (or similar - much higher!)
```

## Validation Checklist

After implementing the fix, verify:

- [ ] Log shows `[BOUNDS] Extracted NNNN pads from board.nets`
- [ ] Lattice node count is < 3M (should be ~1.5-2.5M instead of 4.1M)
- [ ] Routing success rate is > 50% (not 1%)
- [ ] Check debug images in `debug_output/` - traces should stay within visible component area + small margin
- [ ] No traces appearing in corners far from components

## Additional Context

- Portal vias are created by `pad_escape_planner.py` AFTER lattice initialization
- The routing area should match the portal via bounding box (logged as `[ROUTING-BOUNDS]`)
- Grid pitch is 0.4mm with alternating H/V layers (In1.Cu = H, In2.Cu = V, etc.)
- User confirmed: "margin should be the bounding box of all the vias generated by pad escape planner, plus 3mm"

## Files Reference

- **Main file to edit**: `orthoroute/algorithms/manhattan/unified_pathfinder.py` (~1791-1826)
- **Already fixed (not used)**: `orthoroute/algorithms/manhattan/pathfinder/lattice_builder_mixin.py`
- **Working correctly**: `orthoroute/algorithms/manhattan/pad_escape_planner.py`

## Quick Start Command

```bash
# Run test and monitor lattice size
python main.py --test-manhattan 2>&1 | grep -E "(BOUNDS|Lattice:|Routing:)"
```

Look for the lattice size to drop significantly and routing success to improve.
