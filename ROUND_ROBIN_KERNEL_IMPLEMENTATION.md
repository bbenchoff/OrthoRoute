# Round-Robin Layer Bias - Kernel-Side Implementation Guide

## Problem Statement

Current round-robin implementation takes 7-8 seconds per net due to CPU fallbacks and separate kernel launches. The solution is to integrate the bias **directly into the existing CUDA relax kernel** as suggested by the routing expert.

## Performance Comparison

| Approach | Per-Net Time | Why |
|----------|-------------|-----|
| Current (disabled) | 7-8 seconds | CPU fallback, Python loops over millions of edges |
| Column multipliers (attempted) | 7-8 seconds | Still hits CPU fallback, RawKernel recompilation |
| **Kernel-side bias (target)** | **<10ms** | One-time ~10 integer ops in existing hot path |

## Implementation Steps

### Step 1: Add Helper Function (Python)

```python
# In negotiation_mixin.py or unified_pathfinder.py

def _pick_preferred_even_layer(self, net_id: str) -> int:
    """Hash net_id to preferred even layer for round-robin load balancing."""
    layer_count = getattr(self.config, 'layer_count', 18)
    even_layers = [z for z in range(layer_count) if (z & 1) == 0]
    if not even_layers:
        return 0

    # Deterministic hash with good distribution
    h = (hash(net_id) ^ 0x9E3779B9) & 0xffffffff
    return even_layers[h % len(even_layers)]
```

### Step 2: Prepare Metadata Per Batch (Python)

```python
# Before launching CUDA solver, compute once per batch:

# For each net/ROI in batch
pref_layers = []
src_x_coords = []

for net_id, (source_idx, sink_idx) in batch:
    # Get preferred layer
    pref_z = self._pick_preferred_even_layer(net_id)
    pref_layers.append(pref_z)

    # Get source x-coordinate
    x_steps = self.lattice.x_steps
    y_steps = self.lattice.y_steps
    plane_size = x_steps * y_steps
    src_x = (source_idx % plane_size) % x_steps
    src_x_coords.append(src_x)

    logger.info(f"[ROUNDROBIN] net={net_id} preferred_layer=L{pref_z} src_x={src_x}")

# Upload to GPU (tiny arrays)
pref_z_gpu = cp.asarray(pref_layers, dtype=cp.int32)  # [K] ROIs
src_x_gpu = cp.asarray(src_x_coords, dtype=cp.int32)  # [K] ROIs

# Parameters
alpha = self.config.first_vertical_roundrobin_alpha if self.current_iteration <= 3 else 0.0
window_cols = int(8.0 / self.config.grid_pitch)  # 8mm window
```

### Step 3: Modify CUDA Relax Kernel

**Location**: `cuda_dijkstra.py` line 71-120

**Current signature**:
```cpp
void relax_edges_parallel(
    const int K,              // Number of ROIs
    const int max_roi_size,   // Max nodes per ROI
    const int max_edges,      // Max edges per ROI
    const bool* active,       // (K,) active mask
    const int* min_nodes,     // (K,) current min node per ROI
    const int* indptr,        // (K, max_roi_size+1) CSR indptr
    const int* indices,       // (K, max_edges) CSR indices
    const float* weights,     // (K, max_edges) CSR weights
    float* dist,              // (K, max_roi_size) distances
    int* parent               // (K, max_roi_size) parents
)
```

**New signature** (add these parameters):
```cpp
void relax_edges_parallel(
    const int K,
    const int max_roi_size,
    const int max_edges,
    const bool* active,
    const int* min_nodes,
    const int* indptr,
    const int* indices,
    const float* weights,
    float* dist,
    int* parent,
    // NEW PARAMETERS FOR ROUND-ROBIN BIAS:
    const int* pref_layer,    // (K,) preferred even layer per ROI
    const int* src_x,         // (K,) source x-coordinate per ROI
    const int x_steps,        // Grid dimensions
    const int plane_size,     // x_steps * y_steps
    const int window_cols,    // Bias window size (columns)
    const float alpha         // Bias strength (0.12 typical)
)
```

**Kernel body modification** (insert after line 101, before line 102):

```cpp
// Line 99-101: Current code
for (int edge_idx = start; edge_idx < end; edge_idx++) {
    int v = indices[roi_idx * max_edges + edge_idx];
    float cost = weights[roi_idx * max_edges + edge_idx];

    // === NEW: ROUND-ROBIN BIAS ===
    // Apply bias only if alpha > 0 (early iterations only)
    if (alpha > 0.0f) {
        // Decode node coordinates (cheap arithmetic)
        int z_u = u / plane_size;           // Layer of source node u
        int x_u = (u % plane_size) % x_steps;  // X-coordinate of u
        int z_v = v / plane_size;           // Layer of destination node v

        // Check if this is a vertical (inter-layer) edge
        bool is_vertical = (z_v != z_u);

        // Only bias vertical edges on even (routing) layers within window
        if (is_vertical && (z_u & 1) == 0) {  // z_u is even
            // Check if within window of source
            int dx = x_u - src_x[roi_idx];
            if (dx < 0) dx = -dx;  // abs(dx)

            if (dx <= window_cols) {
                // Get preferred layer for this ROI
                int pref_z = pref_layer[roi_idx];

                // Apply bias: discount preferred layer, penalize others
                float m = (z_u == pref_z) ? (1.0f - alpha) : (1.0f + alpha);
                cost *= m;  // Modify edge cost
            }
        }
    }
    // === END ROUND-ROBIN BIAS ===

    float new_dist = u_dist + cost;
    // ... rest of relaxation logic unchanged
```

### Step 4: Update Python Launch Code

**Location**: Find where `relax_kernel` is launched in `cuda_dijkstra.py`

**Current launch** (approximate):
```python
self.relax_kernel(
    (blocks,), (threads,),
    (K, max_roi_size, max_edges, active, min_nodes,
     indptr, indices, weights, dist, parent)
)
```

**New launch**:
```python
# Compute parameters
alpha = config.first_vertical_roundrobin_alpha if iteration <= 3 else 0.0
window_cols = int(8.0 / config.grid_pitch)

self.relax_kernel(
    (blocks,), (threads,),
    (K, max_roi_size, max_edges, active, min_nodes,
     indptr, indices, weights, dist, parent,
     # New round-robin parameters:
     pref_z_gpu,           # (K,) int32 array
     src_x_gpu,            # (K,) int32 array
     cp.int32(x_steps),    # Scalar
     cp.int32(plane_size), # Scalar
     cp.int32(window_cols),# Scalar
     cp.float32(alpha))    # Scalar
)
```

### Step 5: Verify and Test

1. **Syntax check**: `python -c "import orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra"`
2. **First run**: Watch for kernel compilation messages (should be one-time)
3. **Performance**: Per-net time should drop from 7-8s to <1s
4. **Correctness**: Check `[ROUNDROBIN]` logs show different nets get different preferred layers
5. **Balance**: Monitor `[COLUMN-BALANCE]` Gini coefficients - should improve over iterations

## Expected Results

### Log Output
```
[ROUNDROBIN] net=B04B06_013 preferred_layer=L4 src_x=231
[ROUNDROBIN] net=B09B13_006 preferred_layer=L2 src_x=229
[ROUNDROBIN] net=B01B05_006 preferred_layer=L6 src_x=234
[ITER-1-BBOX] Net B04B06_013: BBOX-ONLY mode (no bitmap fence)  # <1s later
[ITER-1-BBOX] Net B09B13_006: BBOX-ONLY mode (no bitmap fence)  # <1s later
```

### Performance Metrics
- **Per-net time**: 7-8s → <1s (7-8× speedup)
- **Iteration 1**: 1+ hour → 5-10 minutes (6-12× speedup)
- **Gini coefficient**: 0.6-0.7 → <0.4 (better column balance)

## Fallback: CPU Path (Optional)

If GPU is unavailable, use vectorized NumPy (not Python loops):

```python
def _apply_roundrobin_bias_cpu_vectorized(self, edge_costs, adj_indptr, adj_indices,
                                           net_id, source_idx, plane_size, x_steps):
    """Vectorized CPU fallback - still fast (<100ms per net)."""
    import numpy as np

    pref_z = self._pick_preferred_even_layer(net_id)
    src_x = (source_idx % plane_size) % x_steps
    alpha = self.config.first_vertical_roundrobin_alpha
    window_cols = int(8.0 / self.config.grid_pitch)

    # Vectorized edge processing (no loops!)
    num_nodes = len(adj_indptr) - 1
    row_degs = np.diff(adj_indptr)
    src_nodes = np.repeat(np.arange(num_nodes), row_degs)
    dst_nodes = adj_indices

    # Decode coordinates
    z_u = src_nodes // plane_size
    x_u = src_nodes % x_steps
    z_v = dst_nodes // plane_size

    # Compute mask
    is_vertical = (z_v != z_u)
    is_even = (z_u & 1) == 0
    in_window = (np.abs(x_u - src_x) <= window_cols)
    mask = is_vertical & is_even & in_window

    # Apply bias
    mult = np.where(z_u == pref_z, 1.0 - alpha, 1.0 + alpha)
    edge_costs[mask] *= mult[mask]
```

## Configuration Parameters

Add to `config.py` (already present):
```python
first_vertical_roundrobin_alpha: float = 0.12   # Bias strength
first_vertical_roundrobin_window_cols: int = 20 # Computed from 8mm / grid_pitch
first_vertical_roundrobin_max_iter: int = 3     # Active iterations
```

## Why This Approach Works

1. **Zero extra passes**: Bias computed inline during existing relaxation
2. **Minimal overhead**: ~10 integer ops per vertical edge, negligible cost
3. **GPU-native**: No CPU↔GPU transfers, stays on device
4. **Deterministic**: Same net always gets same preferred layer
5. **Adaptive**: Only active in early iterations when symmetry matters

## Testing Checklist

- [ ] Kernel compiles without errors
- [ ] No performance regression on non-round-robin iterations
- [ ] Per-net time drops from 7-8s to <1s
- [ ] Different nets get different preferred layers (check logs)
- [ ] Column balance improves (Gini < 0.4 on even layers)
- [ ] Results are deterministic (same routing with same seed)

## Troubleshooting

**Issue**: Still slow (7-8s per net)
- **Check**: Is `alpha > 0` in kernel? Log it.
- **Check**: Are `pref_z_gpu` and `src_x_gpu` on device? Use `hasattr(x, 'get')`
- **Check**: Is kernel being recompiled per call? Should see one compilation message only.

**Issue**: Kernel compilation fails
- **Check**: Are all new parameters properly typed in signature?
- **Check**: Is the conditional logic valid C++ (use `&&` not `and`, etc.)?

**Issue**: Wrong results / crashes
- **Check**: Array dimensions match (K elements for pref_z, src_x)
- **Check**: `plane_size` and `x_steps` are correct values
- **Check**: Node indexing math matches lattice structure

## Estimated Implementation Time

- **Kernel modification**: 30 minutes (copy-paste + test compile)
- **Python launcher update**: 20 minutes (prepare arrays, update call)
- **Testing & debugging**: 1-2 hours (verify performance, correctness)
- **Total**: 2-3 hours for a skilled CUDA developer

## Status

- [x] Implementation plan documented
- [ ] Helper function added
- [ ] Kernel modified
- [ ] Launch code updated
- [ ] Tested and verified

## References

- Expert guidance: See conversation thread for detailed rationale
- Config parameters: `orthoroute/algorithms/manhattan/pathfinder/config.py`
- Kernel code: `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py` lines 71-120
