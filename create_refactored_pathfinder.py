#!/usr/bin/env python3
"""
Script to create unified_pathfinder_refactored.py that composes all extracted mixins.
"""

import os
from pathlib import Path

# Read the __init__ and remaining core methods from original file
def extract_core_methods():
    """Extract __init__ and utility methods that form the core class"""

    core_methods = [
        '__init__',
        '_uid_component',
        '_uid_pad',
        '_uid_pad_label',
        '_choose_two_pads_for_net',
        '_get_all_pads',
        '_get_pad_net_name',
        '_get_pad_coordinates',
        'set_gui_status_callback',
        '_ensure_delta',
        '_is_empty_path',
        '_normalize_path_to_edge_ids',
        '_owner_add',
        '_owner_remove',
        '_normalize_owner_types',
        '_ensure_store_arrays',
        '_reset_present_usage',
        '_commit_present_usage_to_store',
        '_compute_overuse_stats_present',
        '_batched',
        '_edge_indices_from_node_path',
        '_refresh_present_usage_from_store',
        '_compute_overuse_from_present',
        '_compute_overuse_from_store',
        '_update_edge_history_from_present',
        '_path_nodes_to_csr_edges',
        '_coerce_store_key_to_csr_idx',
        '_debug_store_miss',
        '_store_add_usage',
        '_accumulate_edge_usage_present',
        '_on_live_size_changed',
        '_accumulate_edge_usage_gpu',
        '_build_reverse_edge_index_gpu',
        '_live_edge_count',
        '_pf_should_stop',
        '_as_py_float',
        '_as_py_int',
        '_map_layer_for_gui',
        'on_live_size_changed',
        '_commit_batch_store',
        '_pf_cost_for_edge',
        '_can_commit_edge',
        '_commit_edge_to_net',
        '_rip_edge_from_net',
        '_compute_overuse_from_csr',
        '_build_gpu_spatial_index',
        '_initialize_coordinate_array',
        '_assert_coordinate_consistency',
        '_rip_up_route',
        '_add_route_congestion',
        '_path_to_edge_indices',
        '_update_congestion_history',
        'initialize_graph',
        'map_all_pads',
        '_on_grid',
        '_mark_overlaps_as_overuse',
        '_rebuild_present_from_store',
        '_initialize_layer_rtrees',
        '_apply_csr_masks',
        '_apply_via_in_pad_masks',
        '_get_standard_layer_names',
        '_make_hv_polarity',
        '_derive_allowed_layer_pairs',
        'add_vertical_edge',
        '_init_occupancy_grids',
        '_inflate_width_clearance',
        'commit_segment',
        '_is_segment_legal',
        '_spacing_penalty',
        '_init_pathfinder_edge_tracking',
        '_ekey',
        '_inc_edge_usage',
        '_dec_edge_usage',
        '_calculate_airwire_bounds',
        '_get_pad_layer',
        '_apply_direction_masks',
        '_track_layer_usage'
    ]

    return core_methods

def main():
    """Create the refactored unified_pathfinder.py"""

    output_file = Path('orthoroute/algorithms/manhattan/unified_pathfinder_refactored.py')

    # Create header
    header = '''"""
Unified High-Performance PathFinder - Refactored with Mixins

This is a refactored version of unified_pathfinder.py that composes
functionality from extracted mixin modules for better maintainability.

Architecture:
- LatticeBuilderMixin: Lattice construction and escape routing
- GraphBuilderMixin: CSR/GPU matrix construction
- NegotiationMixin: PathFinder negotiation algorithm
- PathfindingMixin: Dijkstra/A*/Delta-stepping implementations
- ROIExtractorMixin: Region-of-interest extraction
- GeometryMixin: Geometry generation and validation
- DiagnosticsMixin: Debug and instrumentation

The UnifiedPathFinder class uses multiple inheritance to compose all mixins
into a single routing engine with all capabilities.
"""

import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Optional dependencies
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    from rtree import index as rtree_index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False

# Sparse matrix backend selection
try:
    if CUPY_AVAILABLE:
        from cupyx.scipy import sparse as sp
        XP = cp
        GPU_BACKEND_AVAILABLE = True
    else:
        raise ImportError("CuPy not available")
except ImportError:
    from scipy import sparse as sp
    XP = np
    GPU_BACKEND_AVAILABLE = False

try:
    from scipy.sparse import csr_matrix as cpu_csr_matrix
except ImportError:
    cpu_csr_matrix = None

# Local imports
from ...domain.models.board import Board, Pad

# Import configuration and constants
from .pathfinder.config import *
from .pathfinder.data_structures import Portal, EdgeRec, Geometry, PathFinderConfig, canonical_edge_key
from .pathfinder.spatial_hash import SpatialHash
from .pathfinder.kicad_geometry import KiCadGeometry

# Import all mixins
from .pathfinder.lattice_builder_mixin import LatticeBuilderMixin
from .pathfinder.graph_builder_mixin import GraphBuilderMixin
from .pathfinder.negotiation_mixin import NegotiationMixin
from .pathfinder.pathfinding_mixin import PathfindingMixin
from .pathfinder.roi_extractor_mixin import RoiExtractorMixin
from .pathfinder.geometry_mixin import GeometryMixin
from .pathfinder.diagnostics_mixin import DiagnosticsMixin

logger = logging.getLogger(__name__)

# Module metadata
VERSION_TAG = "UPF-2025-09-29-refactored"
logger.info("UnifiedPathFinder (Refactored): %s (%s)", VERSION_TAG, __file__)


class UnifiedPathFinder(
    LatticeBuilderMixin,
    GraphBuilderMixin,
    NegotiationMixin,
    PathfindingMixin,
    RoiExtractorMixin,
    GeometryMixin,
    DiagnosticsMixin
):
    """
    Unified High-Performance PathFinder with Mixin Architecture

    This class composes all PathFinder functionality using multiple inheritance
    from specialized mixin classes. The mixins provide:

    - Lattice building and escape routing
    - GPU/CPU graph construction
    - PathFinder negotiation algorithm
    - Multiple pathfinding algorithms (Dijkstra, A*, Delta-stepping)
    - Region-of-interest extraction and caching
    - Geometry generation and DRC validation
    - Comprehensive diagnostics and profiling

    All mixins share state through self and expect certain attributes to exist.
    The __init__ method sets up the core data structures that mixins depend on.
    """

'''

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)

        # Read the __init__ method from original file
        print("Extracting core __init__ and utility methods...")

        with open('orthoroute/algorithms/manhattan/unified_pathfinder.py', 'r', encoding='utf-8') as source:
            lines = source.readlines()

        # Find and extract __init__ method
        core_methods = extract_core_methods()

        def find_method_range(lines, method_name):
            """Find start and end lines for a method"""
            start_line = None
            end_line = None

            # Find start
            for i, line in enumerate(lines):
                if f'    def {method_name}(' in line:
                    start_line = i
                    break

            if start_line is None:
                return None, None

            # Find end (next method at same indentation)
            indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())

            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if line.strip() == '':
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent == indent_level and line.strip().startswith('def '):
                    end_line = i
                    break

            if end_line is None:
                end_line = len(lines)

            return start_line, end_line

        # Extract each core method
        extracted_count = 0
        for method_name in core_methods:
            start, end = find_method_range(lines, method_name)
            if start is not None:
                method_code = ''.join(lines[start:end])
                f.write(method_code)
                f.write('\n')
                extracted_count += 1
                print(f"  Extracted {method_name}")
            else:
                print(f"  WARNING: {method_name} not found")

        print(f"\nExtracted {extracted_count}/{len(core_methods)} core methods")
        print(f"Created: {output_file}")

        # Print file size
        size_kb = output_file.stat().st_size / 1024
        print(f"File size: {size_kb:.1f} KB")

if __name__ == '__main__':
    main()