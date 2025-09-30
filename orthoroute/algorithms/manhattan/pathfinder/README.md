# PathFinder Module - Developer Guide

## Quick Start

```python
from orthoroute.algorithms.manhattan.unified_pathfinder_refactored import UnifiedPathFinder

# Create router
router = UnifiedPathFinder(use_gpu=True)

# Initialize graph from board
router.initialize_graph(board)

# Route nets
route_requests = [("VCC", "pad1", "pad2"), ("GND", "pad3", "pad4")]
paths = router.route_multiple_nets(route_requests)

# Generate geometry
success_count, total_count = router.emit_geometry(board)
```

## Module Structure

### Core Modules

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `config.py` | Configuration constants | All tunable parameters |
| `data_structures.py` | Core data types | Portal, EdgeRec, Geometry, PathFinderConfig |
| `spatial_hash.py` | Spatial indexing | SpatialHash class |
| `kicad_geometry.py` | KiCad integration | KiCadGeometry class |

### Mixin Modules

| Mixin | Methods | Main Responsibilities |
|-------|---------|----------------------|
| `lattice_builder_mixin.py` | 16 | Build 3D routing lattice, escape routing |
| `graph_builder_mixin.py` | 10 | Construct CSR matrices, GPU buffers |
| `negotiation_mixin.py` | 34 | PathFinder algorithm, congestion handling |
| `pathfinding_mixin.py` | 35 | Dijkstra, A*, delta-stepping algorithms |
| `roi_extractor_mixin.py` | 34 | ROI extraction, multi-ROI batching |
| `geometry_mixin.py` | 36 | Geometry generation, DRC validation |
| `diagnostics_mixin.py` | 16 | Profiling, instrumentation, debugging |

## Key Methods by Use Case

### Initialization

```python
# Basic initialization
router = UnifiedPathFinder()

# With custom config
from orthoroute.algorithms.manhattan.pathfinder import PathFinderConfig

config = PathFinderConfig(
    max_iterations=30,
    batch_size=32,
    mode="multi_roi",
    enable_instrumentation=True
)
router = UnifiedPathFinder(config=config)

# Initialize routing graph
router.initialize_graph(board)
```

### Routing

```python
# Route multiple nets
route_requests = [
    ("net_name", "source_pad_id", "sink_pad_id"),
    # ...
]
paths = router.route_multiple_nets(route_requests, progress_cb=callback)

# Single net routing (internal)
path = router._route_single_net_cpu(net_id, source_idx, sink_idx)
```

### Geometry Export

```python
# Emit all geometry
success, total = router.emit_geometry(board)

# Get geometry payload
geometry = router.get_geometry_payload()
# Returns: Geometry(tracks=[], vias=[])

# Visualization data
viz_data = router.get_route_visualization_data(paths)
```

### Diagnostics

```python
# Get performance summary
summary = router.get_instrumentation_summary()

# Export metrics to CSV
router._export_instrumentation_csv()

# Check routing result
result = router.get_routing_result()
if result.success:
    print(f"Routed {result.successful_nets}/{result.total_nets} nets")
else:
    print(f"Failure: {router.get_last_failure_message()}")
```

## Configuration Options

### Routing Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `"delta_stepping"` | Parallel delta-stepping SSSP | Large boards, many nets |
| `"near_far"` | Near-far worklist algorithm | Medium boards |
| `"multi_roi"` | Batched multi-ROI routing | GPU acceleration |
| `"multi_roi_bidirectional"` | Bidirectional A* with ROI | Complex routing |

### Key Parameters

```python
config = PathFinderConfig(
    # Grid
    grid_pitch=0.4,              # Grid spacing in mm
    layer_count=6,               # Number of copper layers

    # Algorithm
    max_iterations=30,           # PathFinder negotiation iterations
    max_search_nodes=50000,      # Max nodes per search
    batch_size=32,               # Nets per batch

    # Costs
    pres_fac_init=1.0,          # Initial present factor
    pres_fac_mult=1.6,          # Present factor multiplier
    pres_fac_max=1000.0,        # Max present factor

    # Performance
    mode="multi_roi",            # Routing algorithm
    roi_parallel=True,           # Enable parallel ROI
    per_net_budget_s=0.5,       # Time budget per net

    # Quality
    strict_capacity=True,        # Enforce capacity limits
    reroute_only_offenders=True # Only reroute congested nets
)
```

## Mixin Method Reference

### LatticeBuilderMixin

**Public Methods:**
- `build_routing_lattice(board)` - Main lattice construction

**Key Internal Methods:**
- `_build_3d_lattice(bounds, layers)` - Generate 3D graph
- `_connect_pads_optimized(pads)` - Create escape routing
- `_create_escape_stub(pad, pad_idx)` - Generate via stubs
- `_validate_spatial_integrity()` - Verify lattice consistency

### GraphBuilderMixin

**Key Methods:**
- `_build_gpu_matrices()` - Construct CSR matrices
- `_populate_cpu_csr()` - CPU graph building
- `_sync_edge_arrays_to_live_csr()` - Synchronize arrays
- `_precompute_edge_penalties()` - Compute edge costs

### NegotiationMixin

**Public Methods:**
- `route_multiple_nets(requests, progress_cb)` - Main routing API
- `rip_up_net(net_id)` - Remove net from routing
- `commit_net_path(net_id, path)` - Commit routed path
- `update_congestion_costs(pres_fac_mult)` - Update edge costs

**Key Internal Methods:**
- `_pathfinder_negotiation(nets, progress_cb)` - Negotiation loop
- `_build_ripup_queue(nets)` - Select nets to reroute
- `_select_offenders_for_ripup(queue)` - Choose offenders

### PathfindingMixin

**Key Algorithms:**
- `_gpu_dijkstra_roi_csr()` - GPU Dijkstra on ROI
- `_gpu_dijkstra_astar_csr()` - GPU A* with heuristic
- `_gpu_dijkstra_bidirectional_astar()` - Bidirectional A*
- `_gpu_delta_stepping_sssp()` - Delta-stepping SSSP
- `_cpu_dijkstra_fallback()` - CPU Dijkstra fallback

### ROIExtractorMixin

**Key Methods:**
- `_extract_roi_subgraph_gpu(min_x, max_x, min_y, max_y)` - Extract ROI
- `_route_multi_roi_batch(batch)` - Batch routing
- `_pack_multi_roi_buffers(roi_list)` - Memory packing
- `_calculate_optimal_k(roi_sizes)` - Optimize batch size

### GeometryMixin

**Public Methods:**
- `emit_geometry(board)` - Generate all geometry
- `get_geometry_payload()` - Get geometry data
- `prepare_routing_runtime()` - Prepare for routing

**Key Internal Methods:**
- `_path_to_geometry(net_id, path)` - Convert path to tracks
- `_generate_pad_stubs(net_id, path)` - Create pad stubs
- `_check_clearance_violations_rtree(intents)` - DRC validation

### DiagnosticsMixin

**Public Methods:**
- `get_instrumentation_summary()` - Performance metrics

**Key Internal Methods:**
- `_export_instrumentation_csv()` - Export metrics
- `_analyze_warp_divergence(metrics, data)` - GPU profiling
- `_estimate_layer_shortfall(overuse)` - Capacity analysis
- `_log_build_sanity_checks(layers)` - Build validation

## Advanced Usage

### Custom Pathfinding Algorithm

To add a new algorithm, create methods in PathfindingMixin:

```python
def _my_custom_algorithm(self, source, sink, roi_data):
    """Custom pathfinding algorithm"""
    # Your implementation
    return path

# Use in negotiation by setting config.mode = "custom"
```

### Performance Tuning

```python
# GPU memory optimization
router._gpu_memory_pool_optimization()
router._enable_zero_copy_optimizations()

# Adaptive tuning
router._adaptive_delta_tuning(success_rate, routing_time)
router._auto_tune_k()  # Auto-tune batch size

# ROI caching for stable regions
router._roi_cache = {}  # Already initialized
# Cache is automatically managed
```

### Debugging

```python
# Enable detailed logging
import logging
logging.getLogger('orthoroute.algorithms.manhattan').setLevel(logging.DEBUG)

# Enable instrumentation
config = PathFinderConfig(enable_instrumentation=True)
router = UnifiedPathFinder(config=config)

# After routing, get metrics
summary = router.get_instrumentation_summary()
print(f"Total routing time: {summary['total_time_s']:.2f}s")
print(f"GPU utilization: {summary['gpu_utilization']:.1f}%")

# Export detailed CSV
router._export_instrumentation_csv()
# Creates: instrumentation_TIMESTAMP.csv
```

## Testing

### Unit Testing Individual Mixins

```python
import unittest
from orthoroute.algorithms.manhattan.pathfinder import LatticeBuilderMixin

class TestLatticeBuilder(unittest.TestCase):
    def setUp(self):
        # Create mock object with required attributes
        self.mixin = type('MockRouter', (LatticeBuilderMixin,), {
            'config': PathFinderConfig(),
            'use_gpu': False,
            'nodes': {},
            'edges': []
        })()

    def test_bounds_calculation(self):
        bounds = self.mixin._calculate_bounds_fast(mock_board)
        self.assertEqual(len(bounds), 4)
```

### Integration Testing

```python
def test_full_routing():
    router = UnifiedPathFinder(use_gpu=True)
    router.initialize_graph(board)

    requests = [("VCC", "U1.1", "U2.1")]
    paths = router.route_multiple_nets(requests)

    assert len(paths) == 1
    assert paths["VCC"] is not None

    success, total = router.emit_geometry(board)
    assert success == total
```

## Common Issues

### Issue: GPU Out of Memory

```python
# Solution 1: Reduce batch size
config = PathFinderConfig(batch_size=16)

# Solution 2: Reduce ROI size
config = PathFinderConfig(max_roi_nodes=10000)

# Solution 3: Use CPU fallback
router = UnifiedPathFinder(use_gpu=False)
```

### Issue: Routing Fails to Converge

```python
# Increase iterations
config = PathFinderConfig(max_iterations=50)

# Adjust present factor
config = PathFinderConfig(
    pres_fac_mult=2.0,  # More aggressive
    pres_fac_max=2000.0
)

# Enable diagnostics
config = PathFinderConfig(enable_instrumentation=True)
router = UnifiedPathFinder(config=config)
# Check: router.get_instrumentation_summary()
```

### Issue: Slow Performance

```python
# Enable GPU
router = UnifiedPathFinder(use_gpu=True)

# Use multi-ROI mode
config = PathFinderConfig(mode="multi_roi", roi_parallel=True)

# Increase batch size
config = PathFinderConfig(batch_size=64)
```

## Contributing

When adding new functionality:

1. **Choose the right mixin**: Place methods in the appropriate module
2. **Add docstrings**: Document parameters and return values
3. **Update __init__.py**: Export new public APIs
4. **Add tests**: Create unit tests for new methods
5. **Update this README**: Document new features

## References

- **Original Paper**: "PathFinder: A Negotiation-Based Performance-Driven Router for FPGAs"
- **Algorithm**: Congestion-driven iterative improvement
- **GPU Acceleration**: CUDA-based parallel pathfinding
- **ROI Extraction**: Spatial partitioning for efficiency

## License

See repository LICENSE file for details.