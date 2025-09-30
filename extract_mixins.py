#!/usr/bin/env python3
"""
Script to extract UnifiedPathFinder methods into logical mixin modules.
This refactors the massive 11,577-line unified_pathfinder.py into maintainable components.
"""

import os
import re
from pathlib import Path

# Define method groups for each mixin
MIXIN_METHODS = {
    'lattice_builder_mixin': [
        'build_routing_lattice', '_validate_spatial_integrity', '_calculate_bounds_fast',
        '_build_3d_lattice', '_verify_lattice_correctness_geometry', '_verify_lattice_correctness',
        '_spot_check_layer_neighbors', '_get_node_neighbors', '_connect_pads_optimized',
        '_create_escape_stub', '_connect_via_to_lattice', '_find_local_rails_at_position',
        '_find_nearest_rail_fast', '_refresh_edge_dependent_arrays', '_refresh_edge_arrays_after_portal_bind'
    ],
    'graph_builder_mixin': [
        '_init_gpu_buffers_once', '_reset_gpu_buffers', '_ensure_gpu_edge_buffers',
        '_populate_cpu_csr', '_assert_live_sizes', '_build_gpu_matrices',
        '_precompute_edge_penalties', '_build_edge_lookup_from_csr', '_sync_edge_arrays_to_live_csr'
    ],
    'negotiation_mixin': [
        'route_multiple_nets', '_pathfinder_negotiation', '_parse_nets_fast',
        '_route_all_nets_cpu_in_batches_with_metrics', '_route_batch_cpu_with_metrics',
        '_route_single_net_cpu', '_update_edge_total_costs', '_compute_overuse_stats',
        '_update_edge_history_gpu', 'rip_up_net', 'commit_net_path',
        'update_congestion_costs', '_build_ripup_queue', '_select_offenders_for_ripup',
        '_prepare_net_for_reroute', '_restore_net_after_failed_reroute',
        '_emit_capacity_analysis', '_identify_most_congested_nets', '_analyze_layer_capacity',
        '_dump_repro_bundle', '_calculate_iteration_metrics', '_route_batch_gpu_with_metrics',
        '_log_multi_roi_performance', '_refresh_present_usage_from_accounting',
        '_apply_capacity_limit_after_negotiation', '_route_batch_gpu',
        '_force_top_k_offenders', '_compute_overuse_from_edge_store', '_bump_history',
        '_get_adaptive_roi_margin', '_update_net_failure_count', '_assert_terminals_reachable'
    ],
    'pathfinding_mixin': [
        '_gpu_delta_stepping_sssp', '_gpu_delta_stepping_sssp_with_metrics',
        '_gpu_roi_near_far_sssp_with_metrics', '_gpu_near_far_worklist_sssp',
        '_cpu_dijkstra_roi_heap', '_gpu_dijkstra_roi_csr', '_gpu_dijkstra_multi_roi_csr',
        '_compute_manhattan_heuristic', '_gpu_dijkstra_astar_csr', '_gpu_dijkstra_multi_roi_astar',
        '_gpu_dijkstra_bidirectional_astar', '_gpu_dijkstra_multi_roi_bidirectional_astar',
        '_get_min_f_node', '_get_min_f_node_roi', '_expand_bidirectional_neighbors',
        '_expand_bidirectional_neighbors_roi', '_build_reverse_graph',
        '_reconstruct_bidirectional_path', '_reconstruct_bidirectional_path_roi',
        '_gpu_dijkstra_delta_stepping_csr', '_process_delta_bucket_gpu',
        '_gpu_dijkstra_multi_roi_delta_stepping', '_process_multi_roi_delta_bucket',
        '_relax_edges_near_far_gpu', '_reconstruct_path_gpu', '_push_to_bucket_gpu',
        '_relax_edges_delta_stepping_gpu', '_cpu_dijkstra_fallback',
        '_calculate_adaptive_roi_margin', '_cpu_astar_fallback_with_roi',
        '_deadline_passed', '_convert_coo_to_csr_gpu'
    ],
    'roi_extractor_mixin': [
        '_extract_roi_subgraph_cpu', '_extract_roi_subgraph_gpu',
        '_extract_roi_subgraph_gpu_with_nodes', '_extract_roi_edges_gpu',
        '_extract_roi_edges_gpu_device_only', '_initialize_multi_roi_gpu',
        '_estimate_roi_memory_bytes', '_calculate_optimal_k', '_validate_roi_connectivity',
        '_pack_multi_roi_buffers', '_get_multi_roi_kernel', '_launch_multi_roi_kernel',
        '_extract_path_from_parents', '_route_multi_roi_batch', '_extract_single_roi_data',
        '_is_roi_dirty', '_extract_single_roi_data_async', '_process_roi_chunk',
        '_route_batch_sequential_fallback', '_update_multi_roi_stats', '_auto_tune_k',
        '_get_gpu_memory_usage_mb', '_gpu_device_only_dijkstra_astar',
        '_gpu_manhattan_heuristic_device_only', '_gpu_reconstruct_paths_device_only',
        '_gpu_memory_pool_optimization', '_gpu_coalesced_memory_layout',
        '_enable_zero_copy_optimizations', '_gpu_multi_roi_astar_parallel',
        '_reconstruct_path_from_parent', '_enable_production_multi_roi_mode',
        '_extract_roi_subgraph'
    ],
    'geometry_mixin': [
        'emit_geometry', 'get_geometry_payload', 'prepare_routing_runtime',
        '_build_pad_keepouts', '_snap_all_pads_to_lattice', '_coords_to_node_index',
        '_idx_to_coord', '_find_portal_for_pad', '_register_portal_stub',
        '_build_geometry_intents', '_emit_pad_stubs', '_snap_mm', '_path_to_geometry',
        '_generate_pad_stubs', '_node_index_to_coords', '_validate_geometry_intents',
        '_convert_intents_to_view', 'get_last_failure_message', 'get_routing_result',
        '_check_clearance_violations_rtree', '_calculate_track_clearance',
        '_point_to_segment_distance', '_add_edge_to_rtree', '_remove_edge_from_rtree',
        'get_route_visualization_data'
    ],
    'diagnostics_mixin': [
        '_csr_smoke_check', '_check_overuse_invariant', '_log_build_sanity_checks',
        '_log_first_routed_nets_debug', '_log_first_illegal_expansion',
        '_adaptive_delta_tuning', '_analyze_warp_divergence', '_suggest_warp_optimization',
        '_export_instrumentation_csv', 'get_instrumentation_summary',
        '_finalize_insufficient_layers', '_estimate_layer_shortfall', '_finalize_success',
        '_count_failed_nets_last_iter', '_log_top_congested_nets'
    ]
}

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

    # Find end (next method at same indentation or class end)
    indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())

    for i in range(start_line + 1, len(lines)):
        line = lines[i]
        if line.strip() == '':
            continue
        current_indent = len(line) - len(line.lstrip())
        # Check if this is a new method at the same level
        if current_indent == indent_level and (line.strip().startswith('def ') or line.strip().startswith('class ')):
            end_line = i
            break

    if end_line is None:
        end_line = len(lines)

    return start_line, end_line

def extract_methods(lines, method_list):
    """Extract specified methods from the source"""
    extracted = []
    missing = []

    for method_name in method_list:
        start, end = find_method_range(lines, method_name)
        if start is not None:
            method_lines = lines[start:end]
            extracted.append('\n'.join(method_lines))
            print(f"    {method_name}: lines {start+1}-{end}")
        else:
            missing.append(method_name)
            print(f"    WARNING: {method_name} not found")

    return extracted, missing

def generate_mixin_header(mixin_name):
    """Generate standard header for mixin module"""
    class_name = ''.join(word.capitalize() for word in mixin_name.replace('_mixin', '').split('_')) + 'Mixin'

    header = f'''"""
{class_name.replace('Mixin', ' Mixin')} - Extracted from UnifiedPathFinder

This module contains {mixin_name.replace('_', ' ')} functionality.
Part of the PathFinder routing algorithm refactoring.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from types import SimpleNamespace
from ....domain.models.board import Board, Pad

logger = logging.getLogger(__name__)


class {class_name}:
    """
    {class_name.replace('Mixin', '')} functionality for UnifiedPathFinder.

    This mixin is designed to be used with multiple inheritance.
    It expects the following attributes to be available on self:
    - config: PathFinderConfig instance
    - use_gpu: bool indicating GPU availability
    - node_count: int number of nodes
    - edges: list of edges
    - nodes: dict of node data
    """

'''
    return header

def main():
    """Extract all mixins from unified_pathfinder.py"""

    source_file = Path('orthoroute/algorithms/manhattan/unified_pathfinder.py')
    output_dir = Path('orthoroute/algorithms/manhattan/pathfinder')

    print(f"Reading source file: {source_file}")
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    print(f"Source file has {len(lines)} lines\n")

    all_missing = {}

    for mixin_name, methods in MIXIN_METHODS.items():
        print(f"\nExtracting {mixin_name}...")
        print(f"  Methods to extract: {len(methods)}")

        extracted_methods, missing = extract_methods(lines, methods)

        if missing:
            all_missing[mixin_name] = missing

        # Generate mixin file
        output_file = output_dir / f"{mixin_name}.py"
        print(f"  Writing to: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(generate_mixin_header(mixin_name))

            # Write methods
            for method_code in extracted_methods:
                f.write(method_code)
                f.write('\n\n')

        print(f"  [OK] Extracted {len(extracted_methods)} methods ({len(missing)} missing)")

    # Summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)

    total_extracted = sum(len(methods) - len(all_missing.get(name, [])) for name, methods in MIXIN_METHODS.items())
    total_methods = sum(len(methods) for methods in MIXIN_METHODS.values())

    print(f"\nTotal methods extracted: {total_extracted}/{total_methods}")

    if all_missing:
        print("\nMissing methods:")
        for mixin_name, missing in all_missing.items():
            print(f"  {mixin_name}: {', '.join(missing)}")
    else:
        print("\n[OK] All methods extracted successfully!")

    print("\nGenerated files:")
    for mixin_name in MIXIN_METHODS.keys():
        output_file = output_dir / f"{mixin_name}.py"
        if output_file.exists():
            size = output_file.stat().st_size / 1024
            print(f"  {output_file.name}: {size:.1f} KB")

if __name__ == '__main__':
    main()