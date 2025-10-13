#!/usr/bin/env python3
"""
Diagnostic script to analyze portal setup and routing failures.
"""

import sys
import logging
from orthoroute.infrastructure.kicad.kicad_file_parser import KiCadFileParser
from orthoroute.algorithms.manhattan.unified_pathfinder import UnifiedPathFinder

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_portals():
    """Analyze portal configuration and check if nodes are valid."""

    # Load the board
    logger.info("Loading board...")
    parser = KiCadFileParser()
    board_path = r"C:\Users\Benchoff\Documents\GitHub\OrthoRoute\examples\breakout.kicad_pcb"
    board = parser.load_board(board_path)

    # Create pathfinder (this will build lattice and run escape planner)
    logger.info("Creating pathfinder...")
    pathfinder = UnifiedPathFinder(board, use_gpu=False)

    # Check lattice configuration
    lattice = pathfinder.lattice
    logger.info(f"\nLattice configuration:")
    logger.info(f"  Dimensions: {lattice.x_steps} x {lattice.y_steps} x {lattice.layers}")
    logger.info(f"  Total nodes: {lattice.num_nodes:,}")
    logger.info(f"  Layer directions: {lattice.layer_dir}")

    # Check escape planner portals
    escape_planner = pathfinder.escape_planner
    if not escape_planner or not escape_planner.portals:
        logger.error("No portals found!")
        return

    logger.info(f"\nPortal analysis:")
    logger.info(f"  Total portals: {len(escape_planner.portals)}")

    # Analyze first 10 portals
    portal_samples = list(escape_planner.portals.items())[:10]

    for pad_id, portal in portal_samples:
        logger.info(f"\n  Pad: {pad_id}")
        logger.info(f"    Portal position: ({portal.x_idx}, {portal.y_idx})")
        logger.info(f"    Entry layer: {portal.entry_layer}")
        logger.info(f"    Direction: {portal.direction}, Delta: {portal.delta_steps}")

        # Check if node exists
        node_idx = lattice.node_idx(portal.x_idx, portal.y_idx, portal.entry_layer)
        logger.info(f"    Node index: {node_idx}")

        # Check if within bounds
        if node_idx >= lattice.num_nodes:
            logger.error(f"    ERROR: Node index {node_idx} exceeds lattice size {lattice.num_nodes}!")
        else:
            logger.info(f"    ✓ Node within bounds")

        # Check layer direction
        layer_dir = lattice.layer_dir[portal.entry_layer]
        logger.info(f"    Entry layer direction: {layer_dir}")

        # Check if Y-coordinate is valid (escape goes in Y direction)
        if portal.y_idx >= lattice.y_steps:
            logger.error(f"    ERROR: Y index {portal.y_idx} exceeds Y steps {lattice.y_steps}!")
        elif portal.y_idx < 0:
            logger.error(f"    ERROR: Y index {portal.y_idx} is negative!")
        else:
            logger.info(f"    ✓ Y coordinate valid")

    # Check graph edges for portal nodes
    logger.info("\nBuilding graph to check edges...")
    graph = lattice.build_graph(via_cost=3.0, use_gpu=False)

    logger.info(f"\nGraph statistics:")
    logger.info(f"  Total edges: {len(graph.indptr) - 1:,}")

    # Check edge counts for portal nodes
    logger.info("\nEdge counts for portal nodes:")
    for i, (pad_id, portal) in enumerate(portal_samples):
        node_idx = lattice.node_idx(portal.x_idx, portal.y_idx, portal.entry_layer)
        if node_idx < len(graph.indptr) - 1:
            edge_start = graph.indptr[node_idx]
            edge_end = graph.indptr[node_idx + 1]
            num_edges = edge_end - edge_start
            logger.info(f"  {pad_id[:30]}: node {node_idx}, {num_edges} edges")

            if num_edges == 0:
                logger.error(f"    ERROR: Portal node has NO EDGES!")
        else:
            logger.error(f"  {pad_id}: ERROR: Node index out of range!")

    # Check layer 0 (F.Cu) edges
    logger.info("\nChecking F.Cu (layer 0) edge counts...")
    f_cu_nodes_with_edges = 0
    f_cu_total_edges = 0
    sample_coords = [(50, 50), (100, 100), (150, 150)]

    for x, y in sample_coords:
        if x < lattice.x_steps and y < lattice.y_steps:
            node_idx = lattice.node_idx(x, y, 0)
            if node_idx < len(graph.indptr) - 1:
                edge_start = graph.indptr[node_idx]
                edge_end = graph.indptr[node_idx + 1]
                num_edges = edge_end - edge_start
                logger.info(f"  F.Cu node ({x}, {y}, 0): {num_edges} edges")
                if num_edges > 0:
                    f_cu_nodes_with_edges += 1
                f_cu_total_edges += num_edges

    if f_cu_total_edges == 0:
        logger.warning("  WARNING: Sample F.Cu nodes have NO edges!")
    else:
        logger.info(f"  ✓ F.Cu nodes have edges")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*60)
    logger.info(f"Portals created: {len(escape_planner.portals)}")
    logger.info(f"Entry layers used: {set(p.entry_layer for p in escape_planner.portals.values())}")
    logger.info(f"Layer directions: {lattice.layer_dir[:6]}...")

    # Check for the bug mentioned in docs
    logger.info("\nChecking for common issues:")

    # Issue 1: Are portals on correct layers?
    portal_layers = [p.entry_layer for p in escape_planner.portals.values()]
    vertical_layers = [i for i, d in enumerate(lattice.layer_dir) if d == 'v']
    horizontal_layers = [i for i, d in enumerate(lattice.layer_dir) if d == 'h']

    portals_on_vertical = sum(1 for l in portal_layers if l in vertical_layers)
    portals_on_horizontal = sum(1 for l in portal_layers if l in horizontal_layers)

    logger.info(f"  Portals on vertical layers: {portals_on_vertical}")
    logger.info(f"  Portals on horizontal layers: {portals_on_horizontal}")
    logger.info(f"  Vertical layers: {vertical_layers[:6]}...")
    logger.info(f"  Horizontal layers: {horizontal_layers[:6]}...")

    if portals_on_horizontal > 0:
        logger.warning(f"  ⚠ {portals_on_horizontal} portals on horizontal layers (may be unreachable via Y-escape!)")

    # Issue 2: F.Cu layer edges?
    if f_cu_total_edges == 0:
        logger.error("  ✗ F.Cu layer has NO edges - this may be the problem!")
    else:
        logger.info("  ✓ F.Cu layer has edges")

if __name__ == '__main__':
    diagnose_portals()
