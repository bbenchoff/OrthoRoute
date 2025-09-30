"""
PathFinder Module - Modular High-Performance PCB Routing Engine

This module provides a refactored, maintainable implementation of the
UnifiedPathFinder routing algorithm split into logical components.

Architecture:
=============

The PathFinder algorithm is implemented using a mixin-based architecture
where functionality is organized into specialized modules:

Core Modules:
-------------
- config.py: Configuration constants and PathFinderConfig dataclass
- data_structures.py: Core data structures (Portal, EdgeRec, Geometry)
- spatial_hash.py: Spatial hashing for efficient collision detection
- kicad_geometry.py: KiCad-specific geometry handling and layer management

Mixin Modules:
--------------
- lattice_builder_mixin.py: 3D routing lattice construction and escape routing
- graph_builder_mixin.py: CSR/GPU graph matrix construction and management
- negotiation_mixin.py: PathFinder negotiation algorithm and congestion handling
- pathfinding_mixin.py: Multiple pathfinding algorithms (Dijkstra, A*, Delta-stepping)
- roi_extractor_mixin.py: Region-of-interest extraction and multi-ROI batching
- geometry_mixin.py: PCB geometry generation and DRC validation
- diagnostics_mixin.py: Instrumentation, profiling, and debugging tools

Main Class:
-----------
The UnifiedPathFinder class (from unified_pathfinder_refactored.py) composes
all mixins using multiple inheritance to provide a complete routing engine.

Usage:
======

Basic usage:

    from orthoroute.algorithms.manhattan.pathfinder import UnifiedPathFinder

    # Create router instance
    router = UnifiedPathFinder(use_gpu=True)

    # Initialize routing graph from board
    router.initialize_graph(board)

    # Route all nets
    route_requests = [("net1", "pad1", "pad2"), ...]
    paths = router.route_multiple_nets(route_requests)

    # Emit geometry
    success_count, total_count = router.emit_geometry(board)

Advanced configuration:

    from orthoroute.algorithms.manhattan.pathfinder import (
        UnifiedPathFinder,
        PathFinderConfig
    )

    config = PathFinderConfig(
        max_iterations=30,
        batch_size=32,
        mode="multi_roi",  # Use multi-ROI parallel routing
        use_gpu=True
    )

    router = UnifiedPathFinder(config=config)

Module Exports:
===============
"""

# Core configuration and constants
from .config import (
    # Grid parameters
    GRID_PITCH,
    # LAYER_COUNT removed - use board.layer_count instead

    # Algorithm parameters
    BATCH_SIZE,
    MAX_ITERATIONS,
    MAX_SEARCH_NODES,
    PER_NET_BUDGET_S,
    MAX_ROI_NODES,

    # Cost parameters
    PRES_FAC_INIT,
    PRES_FAC_MULT,
    PRES_FAC_MAX,
    HIST_ACCUM_GAIN,
    OVERUSE_EPS,

    # Tuning parameters
    DELTA_MULTIPLIER,
    ADAPTIVE_DELTA,
    STRICT_CAPACITY,
    REROUTE_ONLY_OFFENDERS,

    # Via parameters
    VIA_COST,
    VIA_CAPACITY_PER_NET,

    # Performance parameters
    ROI_SAFETY_CAP,
    NET_LIMIT,
    DISABLE_EARLY_STOP,
    CAPACITY_END_MODE,

    # Quality parameters
    MIN_STUB_LENGTH_MM,
    PAD_CLEARANCE_MM,
    BASE_ROI_MARGIN_MM,
    BOTTLENECK_RADIUS_FACTOR,
    HISTORICAL_ACCUMULATION,

    # Negotiation parameters
    STAGNATION_PATIENCE,
    STRICT_OVERUSE_BLOCK,
    HIST_COST_WEIGHT,

    # Debugging
    ROUTING_SEED,
    ENABLE_PROFILING,
    ENABLE_INSTRUMENTATION
)

# Data structures
from .data_structures import (
    Portal,
    EdgeRec,
    Geometry,
    canonical_edge_key
)

# Configuration
from .config import PathFinderConfig

# Geometry helpers
from .spatial_hash import SpatialHash
from .kicad_geometry import KiCadGeometry

# Mixin classes (typically not used directly, but available for inspection)
from .lattice_builder_mixin import LatticeBuilderMixin
from .graph_builder_mixin import GraphBuilderMixin
from .negotiation_mixin import NegotiationMixin
from .pathfinding_mixin import PathfindingMixin
from .roi_extractor_mixin import RoiExtractorMixin
from .geometry_mixin import GeometryMixin
from .diagnostics_mixin import DiagnosticsMixin

# Note: UnifiedPathFinder is imported from parent module to avoid circular imports
# Use: from orthoroute.algorithms.manhattan import UnifiedPathFinder

__all__ = [
    # Configuration
    'PathFinderConfig',
    'GRID_PITCH',
    # 'LAYER_COUNT' removed - use board.layer_count instead
    'BATCH_SIZE',
    'MAX_ITERATIONS',
    'MAX_SEARCH_NODES',
    'PER_NET_BUDGET_S',
    'MAX_ROI_NODES',
    'PRES_FAC_INIT',
    'PRES_FAC_MULT',
    'PRES_FAC_MAX',
    'HIST_ACCUM_GAIN',
    'OVERUSE_EPS',
    'DELTA_MULTIPLIER',
    'ADAPTIVE_DELTA',
    'STRICT_CAPACITY',
    'REROUTE_ONLY_OFFENDERS',
    'VIA_COST',
    'VIA_CAPACITY_PER_NET',
    'ROI_SAFETY_CAP',
    'NET_LIMIT',
    'DISABLE_EARLY_STOP',
    'CAPACITY_END_MODE',
    'MIN_STUB_LENGTH_MM',
    'PAD_CLEARANCE_MM',
    'BASE_ROI_MARGIN_MM',
    'BOTTLENECK_RADIUS_FACTOR',
    'HISTORICAL_ACCUMULATION',
    'STAGNATION_PATIENCE',
    'STRICT_OVERUSE_BLOCK',
    'HIST_COST_WEIGHT',
    'ROUTING_SEED',
    'ENABLE_PROFILING',
    'ENABLE_INSTRUMENTATION',

    # Data structures
    'Portal',
    'EdgeRec',
    'Geometry',
    'canonical_edge_key',

    # Geometry helpers
    'SpatialHash',
    'KiCadGeometry',

    # Mixin classes
    'LatticeBuilderMixin',
    'GraphBuilderMixin',
    'NegotiationMixin',
    'PathfindingMixin',
    'RoiExtractorMixin',
    'GeometryMixin',
    'DiagnosticsMixin',
]

__version__ = "1.0.0-refactored"
__author__ = "OrthoRoute Development Team"
__doc__ = """
PathFinder PCB Routing Engine - Modular Architecture

This package provides a high-performance, GPU-accelerated PCB routing engine
based on the PathFinder negotiated congestion algorithm with extensive
optimizations for modern PCB design workflows.

Key Features:
- GPU-accelerated pathfinding with CPU fallback
- Multiple routing algorithms (Dijkstra, A*, Delta-stepping, Bidirectional A*)
- Region-of-interest extraction for efficient large board routing
- Multi-ROI parallel batching for GPU efficiency
- Real-time DRC validation with spatial indexing
- Comprehensive diagnostics and profiling
- KiCad native layer management and geometry generation

The modular architecture using mixins allows for:
- Clear separation of concerns
- Easy testing and debugging of individual components
- Flexible configuration and extensibility
- Better code maintainability and readability

For detailed documentation, see the individual module docstrings.
"""