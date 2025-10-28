# CUDA Kernel Compilation Fix: nx/ny/nz Variable Scope Issue

## Problem Summary

The CUDA kernel in `cuda_dijkstra.py` was failing to compile with the following errors:
```
error: identifier "nx" is undefined (line 208)
error: identifier "ny" is undefined (line 209)
```

## Root Cause

In the `wavefront_expand_all` kernel (lines 194-433), the variables `nx`, `ny`, and `nz` were declared inside an `if (use_astar)` block (lines 341-357), but were being used outside that scope in the LEGACY PATH validation code (lines 384-429).

### Original Code Structure (BROKEN)

```c++
// Lines 338-357 (ORIGINAL)
// P0-3: A* HEURISTIC with procedural coordinate decoding
float f_new = g_new;  // Default: Dijkstra (no heuristic)

if (use_astar) {
    // Decode neighbor coordinates from node index (no memory loads!)
    const int plane_size = Nx * Ny;
    const int nz = neighbor / plane_size;           // DECLARED HERE (local scope)
    const int remainder = neighbor - (nz * plane_size);
    const int ny = remainder / Nx;                   // DECLARED HERE (local scope)
    const int nx = remainder - (ny * Nx);            // DECLARED HERE (local scope)

    // ... A* heuristic calculation ...
}

// Lines 379-429
} else {
    // LEGACY PATH: Separate dist/parent updates
    const float old = atomicMinFloat(&dist[nidx], g_new);
    if (g_new + 1e-8f < old) {
        // MANHATTAN VALIDATION: Verify parent->child is adjacent
        // ...

        if (z_node != nz) {                          // ERROR: nz undefined here!
            if (nx == x_node && ny == y_node) {      // ERROR: nx, ny undefined here!
                valid_parent = true;
            }
        } else {
            const int dx = abs(nx - x_node);         // ERROR: nx undefined here!
            const int dy = abs(ny - y_node);         // ERROR: ny undefined here!
            // ...
        }
    }
}
```

The problem: `nx`, `ny`, `nz` were declared with `const int` inside the `if (use_astar)` block, making them local to that block. When `use_astar` is false, the code enters the `else` branch and tries to use these variables in the Manhattan validation code, but they're out of scope.

## Solution

Move the coordinate decoding **before** the A* conditional, so `nx`, `ny`, `nz` are available in all subsequent code paths.

### Fixed Code Structure

```c++
// Lines 338-358 (FIXED)
// P0-3: A* HEURISTIC with procedural coordinate decoding
// CRITICAL FIX: Declare neighbor coordinates at broader scope
// (needed for both A* heuristic AND Manhattan validation in LEGACY PATH)
const int plane_size = Nx * Ny;
const int nz = neighbor / plane_size;           // NOW DECLARED AT BROADER SCOPE
const int remainder = neighbor - (nz * plane_size);
const int ny = remainder / Nx;                   // NOW DECLARED AT BROADER SCOPE
const int nx = remainder - (ny * Nx);            // NOW DECLARED AT BROADER SCOPE

float f_new = g_new;  // Default: Dijkstra (no heuristic)

if (use_astar) {
    // Load goal coordinates (just 3 ints per ROI)
    const int gx = goal_coords[roi_idx * 3 + 0];
    const int gy = goal_coords[roi_idx * 3 + 1];
    const int gz = goal_coords[roi_idx * 3 + 2];

    // Manhattan distance heuristic (admissible)
    const float h = (abs(gx - nx) + abs(gy - ny)) * 0.4f + abs(gz - nz) * 1.5f;
    f_new = g_new + h;
}

// Now nx, ny, nz are available in the LEGACY PATH below
```

## Changes Made

**File**: `C:\Users\Benchoff\Documents\GitHub\OrthoRoute\orthoroute\algorithms\manhattan\pathfinder\cuda_dijkstra.py`

**Lines Modified**: 338-358 (in the `wavefront_expand_all` kernel)

**What Changed**:
1. Moved coordinate decoding (`const int nz = ...`, `const int ny = ...`, `const int nx = ...`) from inside the `if (use_astar)` block to before it
2. Kept the A* heuristic calculation inside the `if (use_astar)` block (only goal coordinates and heuristic calculation)
3. Added explanatory comment about why the broader scope is needed

## Verification

### 1. Kernel Compilation Test
```bash
$ python test_kernel_compile.py
CuPy imported successfully
CUDADijkstra imported successfully

Attempting to compile CUDA kernels...
SUCCESS! All kernels compiled without errors

Compiled kernels:
  - relax_edges_parallel
  - wavefront_expand_all
  - active_list_relax_kernel
```

### 2. Basic Routing Test
```bash
$ python test_simple_route.py
Creating CUDADijkstra instance...
Creating simple test graph...
Running wavefront expansion (this tests nx/ny/nz usage)...
SUCCESS! Kernel executed without errors
```

## Other Occurrences Checked

The codebase has three locations where similar Manhattan validation using `nx`, `ny`, `nz` occurs:

1. **Lines 338-429** (wavefront_expand_all kernel) - **FIXED** ✓
2. **Lines 767-839** (RELAX_NEIGHBOR macro) - **NO FIX NEEDED** - `nx`, `ny`, `nz` are macro parameters (properly scoped)
3. **Lines 1059-1119** (active_list_relax_kernel) - **ALREADY FIXED** - Has comment "CRITICAL FIX: Declare neighbor coordinates at function scope"

## Performance Impact

**None**. The coordinate decoding is performed exactly once per neighbor regardless of whether A* is enabled. The only change is when it happens (before the A* check instead of inside it). There is no additional computation.

## Related Issues

This fix ensures that Manhattan validation in the LEGACY PATH (used when `use_atomic_parent_keys=False`) works correctly. The LEGACY PATH is typically used in the first routing iteration (`iter==1`) and validates parent-child adjacency to enforce Manhattan geometry constraints.

## Testing Recommendations

1. Run existing routing tests to ensure no regression
2. Test with both `use_astar=True` and `use_astar=False`
3. Test with both `use_atomic_parent_keys=True` (ATOMIC PATH) and `False` (LEGACY PATH)
4. Verify Manhattan geometry constraints are still enforced correctly

## Date
October 27, 2025
