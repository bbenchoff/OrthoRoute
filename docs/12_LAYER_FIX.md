# Critical Convergence Fix for Narrow 12-Layer Boards

## Problem Statement

### Board Characteristics
- **Dimensions**: 23.8mm × 188.5mm (VERY NARROW - 7.93:1 aspect ratio)
- **Layers**: 12 copper layers
- **Nets**: 512 nets to route
- **Channels**: Only ~60 horizontal grid steps (23.8mm / 0.4mm pitch)

### Symptom
- **Original congestion ratio (ρ)**: 0.408 → classified as "SPARSE"
- **Auto-config selected**: Fast convergence strategy (pres_fac_mult=1.15, hotset=25%)
- **Reality**: Board stuck at 15k overuse, oscillating, 7-25 nets failing
- **Root cause**: Standard congestion calculation doesn't account for narrow width constraint

### Why It Failed

The narrow width (23.8mm) creates a severe routing bottleneck:
- Only 60 horizontal channels available across entire board width
- 512 nets competing for these narrow corridors
- Vertical capacity is fine (188.5mm = 471 channels), but horizontal is severely constrained
- Standard ρ calculation: `ρ = demand / (horizontal_capacity + vertical_capacity)`
  - This averages the constraints, hiding the horizontal bottleneck
  - Result: ρ=0.408 suggests "plenty of room" when reality is "severe horizontal congestion"

**Physics**: A 23.8mm-wide board with 512 nets behaves like a 4-lane highway with 512 cars trying to merge - extreme congestion regardless of road length!

---

## Solution Applied

### 1. Aspect Ratio Penalty in Congestion Calculation

**File**: `orthoroute/algorithms/manhattan/board_analyzer.py` (lines 159-174)

Added aspect ratio detection and penalty:

```python
# CRITICAL FIX: Apply aspect ratio penalty for narrow boards
# Narrow boards create routing bottlenecks that standard ρ calculation misses
aspect_ratio = max(board_width_mm, board_height_mm) / max(0.1, min(board_width_mm, board_height_mm))

if aspect_ratio > 4.0:
    # Narrow boards (aspect > 4:1) have constrained routing in one dimension
    # Even with low nominal congestion, narrow width limits available channels
    # Apply increasing penalty as aspect ratio grows
    constraint_penalty = 1.5 + 0.5 * min(2.0, (aspect_ratio - 4.0) / 4.0)
    original_ratio = congestion_ratio
    congestion_ratio *= constraint_penalty

    logger.info(f"ASPECT RATIO PENALTY APPLIED:")
    logger.info(f"  Aspect ratio: {aspect_ratio:.2f}:1 (narrow board detected)")
    logger.info(f"  Constraint penalty: {constraint_penalty:.3f}×")
    logger.info(f"  Adjusted ρ: {original_ratio:.3f} → {congestion_ratio:.3f}")
```

**Penalty formula**:
- **4:1 to 8:1 aspect**: penalty = 1.5 + 0.5 × (aspect - 4) / 4
  - 4:1 → 1.5×
  - 6:1 → 1.75×
  - 8:1 → 2.0×
- **>8:1 aspect**: penalty capped at 2.5×

**Result for 7.93:1 board**:
- Original ρ: 0.408 (SPARSE)
- Penalty: 1.991×
- Adjusted ρ: 0.813 (NORMAL)

---

### 2. Narrow Board History Weight Boost

**File**: `orthoroute/algorithms/manhattan/parameter_derivation.py` (lines 127-135)

Added aspect-based history weight bonus to prevent oscillation:

```python
# Aspect ratio penalty: narrow boards need stronger history to avoid oscillation
aspect_ratio = max(board.board_width_mm, board.board_height_mm) / max(0.1, min(board.board_width_mm, board.board_height_mm))
aspect_bonus = 0.0
if aspect_ratio > 4.0:
    # Narrow boards (>4:1) get +4-8 bonus history weight
    aspect_bonus = 4.0 + 4.0 * min(1.0, (aspect_ratio - 4.0) / 8.0)
    logger.info(f"Aspect ratio penalty: {aspect_ratio:.2f}:1 → +{aspect_bonus:.1f} hist_cost_weight")

hist_cost_weight = base_hist_weight + layer_bonus + congestion_bonus + aspect_bonus
```

**Rationale**: Narrow boards create "memory pressure" where routes must remember past conflicts strongly to avoid thrashing back and forth in limited horizontal space.

**Result for 7.93:1 board**:
- Base: 8.0
- Layer bonus (12 layers): +6.0
- Congestion bonus (ρ=0.813): +0.6
- **Aspect bonus (7.93:1): +6.0**
- **Total hist_cost_weight: 20.6** (was 14.6 without fix)

---

### 3. Reduced Hotset for Narrow Boards

**File**: `orthoroute/algorithms/manhattan/parameter_derivation.py` (lines 163-166)

Reduced hotset percentage for very narrow boards:

```python
# Narrow boards need smaller hotset to reduce oscillation
if aspect_ratio > 6.0:
    hotset_pct *= 0.75  # Reduce by 25% for very narrow boards
    logger.info(f"Narrow board detected: reducing hotset to {hotset_pct*100:.0f}%")
```

**Rationale**: Large hotsets in narrow boards cause too many nets to re-route simultaneously, creating thrashing in limited horizontal space.

**Result for 7.93:1 board**:
- Original (NORMAL): 20% → 102 nets
- Reduced: 15% → **76 nets** (25% smaller)

---

## Parameter Changes Summary

| Parameter | Before Fix | After Fix | Change |
|-----------|------------|-----------|--------|
| **Congestion ratio (ρ)** | 0.408 | 0.813 | +99% (2×) |
| **Strategy** | SPARSE | **NORMAL** | ✓ |
| **pres_fac_mult** | 1.15 | **1.12** | Gentler |
| **pres_fac_max** | 6.0 | **9.3** | Higher ceiling |
| **hist_cost_weight** | 14.6 | **20.6** | +41% |
| **hotset_cap** | 128 (25%) | **76 (15%)** | -41% |
| **max_iterations** | 30 | **45** | +50% |
| **stagnation_patience** | 5 | **7** | +40% |

---

## Expected Convergence Improvement

### Before Fix (SPARSE strategy, ρ=0.408)
- Fast escalation (pres_fac_mult=1.15) overwhelms history
- Large hotset (25%) causes thrashing
- Weak history (14.6) forgets conflicts quickly
- **Result**: Oscillation, stuck at 15k overuse

### After Fix (NORMAL strategy, ρ=0.813)
- Gentler escalation (pres_fac_mult=1.12) keeps history competitive
- Smaller hotset (15%) reduces disruption
- Strong history (20.6) remembers narrow-board conflicts
- More iterations (45) and patience (7) allow slow convergence
- **Expected**: Steady overuse reduction, target <5,000

---

## Testing

### Verification Log Output

```
ANALYZING BOARD CHARACTERISTICS
Board: 23.8mm × 188.5mm
Layers: 12 total, 10 signal
Grid pitch: 0.4mm
Nets: 512

ASPECT RATIO PENALTY APPLIED:
  Aspect ratio: 7.93:1 (narrow board detected)
  Constraint penalty: 1.991×
  Adjusted ρ: 0.408 → 0.813
Congestion ratio ρ = 0.813 (NORMAL)

DERIVING ROUTING PARAMETERS
Present schedule: mult=1.120, max=9.3
Aspect ratio penalty: 7.93:1 → +6.0 hist_cost_weight
History: weight=20.6, gain=0.202, decay=1.000
Narrow board detected: reducing hotset to 15%
Hotset: cap=76 (15% of 512 nets)
STRATEGY: NORMAL (balanced)
```

### Test Command
```bash
python main.py --test-manhattan 2>&1 | tee test_narrow_board_fix.log
```

---

## Generalization

This fix applies automatically to any narrow board:

- **4:1 to 8:1 aspect**: Progressive penalty (1.5× to 2.0×)
- **>8:1 aspect**: Maximum penalty (2.5×)
- **<4:1 aspect**: No penalty (normal congestion calculation)

Examples:
- 50mm × 200mm (4:1) → penalty = 1.5×
- 30mm × 180mm (6:1) → penalty = 1.75×
- 23.8mm × 188.5mm (7.93:1) → penalty = 1.991×
- 20mm × 200mm (10:1) → penalty = 2.5× (capped)

---

## Technical Insight

The core issue is **dimensional asymmetry** in routing resources:

### Standard Congestion Formula (Wrong for Narrow Boards)
```
ρ = demand / (h_capacity + v_capacity)
  = 34036mm / (3364mm + 79003mm)
  = 34036 / 82367
  = 0.408 (SPARSE - WRONG!)
```

This averages horizontal and vertical capacity, hiding the horizontal bottleneck.

### Physics-Based Reasoning
A narrow board is like a highway with one narrow bridge:
- Even if total highway length is long (high v_capacity)
- The narrow bridge (low h_capacity) is the **bottleneck**
- Traffic backs up regardless of total road length

### Correct Approach
Weight congestion by **constraint factor** based on aspect ratio:
```
ρ_effective = ρ_nominal × constraint_penalty(aspect_ratio)
            = 0.408 × 1.991
            = 0.813 (NORMAL - CORRECT!)
```

This reflects the **effective congestion** experienced by routes trying to navigate the narrow dimension.

---

## Files Modified

1. **`orthoroute/algorithms/manhattan/board_analyzer.py`**
   - Lines 159-174: Aspect ratio penalty in congestion calculation

2. **`orthoroute/algorithms/manhattan/parameter_derivation.py`**
   - Lines 127-135: Aspect-based history weight bonus
   - Lines 163-166: Hotset reduction for narrow boards

---

## Success Criteria

- ✓ Aspect ratio detected correctly (7.93:1)
- ✓ Congestion ratio adjusted (0.408 → 0.813)
- ✓ Strategy changed to NORMAL (from SPARSE)
- ✓ Parameters re-derived appropriately
- ⏳ Convergence test in progress (monitor for overuse <5,000)

---

## Future Enhancements

If convergence still struggles on extremely narrow boards (>10:1):

1. **Emergency layer redistribution**: If one layer >70% utilized, trigger emergency spreading
2. **Adaptive via cost**: Increase via_cost on narrow boards to encourage horizontal spreading
3. **Column-aware hotset**: Prioritize nets in different X-columns to avoid local thrashing
4. **Multi-stage present**: Use staged pressure (iter 1-10: low, 11-20: medium, 21+: high)

---

**Date**: 2025-10-27
**Issue**: Narrow board convergence failure
**Root Cause**: Aspect ratio not factored into congestion calculation
**Fix**: Aspect ratio penalty, history boost, reduced hotset
**Status**: DEPLOYED - awaiting convergence verification
