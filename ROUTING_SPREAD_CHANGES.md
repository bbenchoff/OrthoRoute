# Routing Spread Solutions - Detailed Changes

## File Modified
`C:\Users\Benchoff\Documents\GitHub\OrthoRoute\orthoroute\algorithms\manhattan\unified_pathfinder.py`

---

## SOLUTION 2: Spatial Cost Smoothing

### Location 1: Lines 870-893 (NEW METHOD)
**Added method `_spatial_blur()` to apply spatial smoothing**

```python
def _spatial_blur(self, overuse_array, radius=3):
    """
    Apply spatial smoothing to create pressure zones around congestion.
    Solution 2: Spatial Cost Smoothing for routing spread.

    Args:
        overuse_array: Array of overuse values per edge
        radius: Smoothing radius (default 3)

    Returns:
        Smoothed overuse array
    """
    if radius <= 0:
        return overuse_array

    # Simple box filter for spatial smoothing
    kernel_size = 2 * radius + 1
    kernel = self.xp.ones(kernel_size) / kernel_size

    # Pad edges to handle boundaries
    padded = self.xp.pad(overuse_array, radius, mode='edge')
    smoothed = self.xp.convolve(padded, kernel, mode='valid')

    return smoothed
```

### Location 2: Lines 945-950 (ADDED TO EXISTING METHOD)
**Added spatial penalty to `update_costs()` method**

```python
# Solution 2: Add spatial smoothing penalty to spread routes away from congested areas
if hasattr(self, '_spatial_blur'):
    over_raw = xp.maximum(0, self.present - self.capacity)
    smoothed_overuse = self._spatial_blur(over_raw, radius=3)
    spatial_penalty = smoothed_overuse * 0.5 * pres_fac
    self.total_cost += spatial_penalty
```

**Context**: Added after line computing `self.total_cost`, before jitter application.

---

## SOLUTION 5: Initial Route Randomization

### Location: Lines 2907-2917 (ADDED BEFORE COST UPDATE)
**Added jitter to break symmetry in iteration 1**

```python
# Solution 5: Initial Route Randomization - add jitter to first iteration
if it == 1:
    # Add random jitter to break symmetry in first iteration
    import numpy as np
    np.random.seed(42)  # Deterministic but random
    xp = self.accounting.xp
    jitter = xp.random.uniform(-0.05, 0.05, size=self.graph.base_costs.shape)
    adjusted_base = self.graph.base_costs * (1.0 + jitter)
    logger.info("[SOLUTION 5] Applied initial route randomization jitter to break symmetry")
else:
    adjusted_base = self.graph.base_costs
```

**Context**: Added just before "CRITICAL: Cost update ONCE per iteration" comment.

**Related Changes**: Updated all `self.graph.base_costs` references in cost update calls (lines 2927, 2939, 2950) to use `adjusted_base` instead.

---

## SOLUTION 7: Wider ROI Inflation

### Location 1: Line 3629 (MODIFIED)
**Changed minimum ROI radius from 40 to 80**

**Before:**
```python
adaptive_radius = max(40, min(int(manhattan_dist * 1.2) + 10, 800))
```

**After:**
```python
# Solution 7: Wider ROI Inflation - increased minimum from 40 to 80 for better spreading
adaptive_radius = max(80, min(int(manhattan_dist * 1.2) + 10, 800))
```

### Location 2: Line 3741 (MODIFIED)
**Changed minimum ROI radius from 40 to 80 (fallback path)**

**Before:**
```python
adaptive_radius = max(40, int(manhattan_dist * 1.5) + 10)
```

**After:**
```python
# Solution 7: Wider ROI Inflation - increased minimum from 40 to 80 for better spreading
adaptive_radius = max(80, int(manhattan_dist * 1.5) + 10)
```

---

## SOLUTION 4: Outside-In Hotset Selection

### Location: Lines 4237-4282 (MODIFIED SCORING SECTION)
**Added distance-based scoring to prioritize outer nets**

**Before:**
```python
# Score offenders by total overuse they contribute
scores = []
for net_id in offenders:
    if net_id in self._net_to_edges:
        impact = sum(float(over[ei]) for ei in self._net_to_edges[net_id] if ei in over_idx)
        scores.append((impact, net_id))

# Sort by impact (highest first)
scores.sort(reverse=True)
```

**After:**
```python
# Solution 4: Outside-In Hotset Selection - compute board center for distance-based scoring
if not hasattr(self, '_board_center'):
    all_nodes = list(range(self.lattice.x_steps * self.lattice.y_steps * self.lattice.layers))
    if all_nodes:
        mid_node = all_nodes[len(all_nodes) // 2]
        cx, cy, _ = self.lattice.idx_to_coord(mid_node)
        self._board_center = (cx, cy)
    else:
        self._board_center = (self.lattice.x_steps // 2, self.lattice.y_steps // 2)

center_x, center_y = self._board_center

# Score offenders by total overuse they contribute + distance from center
scores = []
for net_id in offenders:
    if net_id in self._net_to_edges:
        impact = sum(float(over[ei]) for ei in self._net_to_edges[net_id] if ei in over_idx)

        # Solution 4: Add distance-based component to boost outer nets
        if net_id in tasks:
            src, dst = tasks[net_id]
            src_x, src_y, _ = self.lattice.idx_to_coord(src)
            dst_x, dst_y, _ = self.lattice.idx_to_coord(dst)
            # Net center (average of endpoints)
            net_cx = (src_x + dst_x) / 2.0
            net_cy = (src_y + dst_y) / 2.0
            # Distance from board center
            dist_from_center = ((net_cx - center_x)**2 + (net_cy - center_y)**2)**0.5
            # Normalize distance (0-1 range based on max possible distance)
            max_dist = ((self.lattice.x_steps/2)**2 + (self.lattice.y_steps/2)**2)**0.5
            norm_dist = dist_from_center / max(max_dist, 1.0)
            # Boost score for nets far from center (outer nets routed first)
            distance_bonus = norm_dist * 50.0  # Scale factor to make distance significant
            total_score = impact + distance_bonus
        else:
            total_score = impact

        scores.append((total_score, net_id))

# Sort by score (highest first) - outer nets with overuse will route first
scores.sort(reverse=True)
```

---

## SOLUTION 8: Regional Penalty System

### Location: Lines 895-911 (NEW METHOD - PLACEHOLDER)
**Added skeleton for column-based smoothing**

```python
def _smooth_column_costs(self, cost_array, grid_size=5):
    """
    Solution 8: Regional Penalty System - smooth costs by column to penalize congested vertical bands.
    TODO: Full implementation requires graph structure mapping.
    This is a placeholder for future enhancement.

    Args:
        cost_array: Array of costs per edge
        grid_size: Size of smoothing window

    Returns:
        Smoothed cost array
    """
    # Placeholder: would need to map edges to spatial grid positions
    # For now, just return the original array
    # Full implementation would group edges by X-coordinate and apply smoothing
    return cost_array
```

**Note**: This is a placeholder. Full implementation requires:
1. Mapping edges to 2D grid coordinates
2. Grouping edges by X-coordinate (columns)
3. Applying smoothing kernel across columns
4. Integrating into `update_costs()` pipeline

---

## Summary of Changes

### Total Lines Added: ~110
- Solution 2: ~30 lines (method + integration)
- Solution 5: ~13 lines (jitter logic)
- Solution 7: ~2 lines (comment updates for modified values)
- Solution 4: ~50 lines (distance calculation and scoring)
- Solution 8: ~15 lines (placeholder method)

### Total Lines Modified: ~6
- Solution 5: 3 lines (adjusted_base substitutions)
- Solution 7: 2 lines (40 -> 80 changes)

### No Lines Deleted
All changes are additive or modifications to existing code.

### Backward Compatibility
- All changes are backward compatible
- No breaking changes to APIs
- No changes to configuration files
- Existing tests should pass

---

## Testing Checklist

- [x] Syntax validation (Python compilation)
- [x] Import test (module loads)
- [x] Code presence verification (all solutions found)
- [ ] Full routing test on benchmark design
- [ ] Performance validation (overhead < 2%)
- [ ] Convergence analysis (iterations to completion)
- [ ] Overuse distribution heatmap comparison
- [ ] Wirelength regression check
- [ ] Layer balance validation

---

## Configuration Values Used

| Parameter | Value | Location | Tunable? |
|-----------|-------|----------|----------|
| Spatial blur radius | 3 | Line 948 | Yes (hardcoded) |
| Spatial blur weight | 0.5 | Line 949 | Yes (hardcoded) |
| Initial jitter range | ±0.05 | Line 2913 | Yes (hardcoded) |
| Initial jitter seed | 42 | Line 2911 | Yes (hardcoded) |
| ROI minimum radius | 80 | Lines 3629, 3741 | Yes (hardcoded) |
| Distance bonus scale | 50.0 | Line 4269 | Yes (hardcoded) |

All values are currently hardcoded but could be moved to configuration file for easier tuning.
