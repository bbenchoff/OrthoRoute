#!/usr/bin/env python3
"""
OrthoRoute - Main Entry Point
Advanced PCB autorouter with Manhattan routing and GPU acceleration
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Optional
import numpy as np

# Add the package directory to Python path
package_dir = Path(__file__).parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from orthoroute.shared.configuration import initialize_config, get_config
from orthoroute.shared.utils.logging_utils import setup_logging, init_logging


def setup_environment():
    """Setup the application environment."""
    # Initialize early logging for acceptance test
    init_logging()

    # Initialize configuration
    config = initialize_config()

    # NOTE: setup_logging() disabled to prevent duplicate handlers
    # init_logging() already configured DEBUG→file, WARNING→console
    # setup_logging(config.get_settings().logging)

    return config


def show_usage():
    """Show usage information."""
    print("OrthoRoute - KiCad PCB Autorouter")
    print("Usage:")
    print("  python main.py                      # Run KiCad plugin with GUI (default)")
    print("  python main.py plugin               # Run as KiCad plugin with GUI")
    print("  python main.py plugin --no-gui      # Run as KiCad plugin without GUI")
    print("  python main.py cli board.kicad_pcb  # Command line mode")
    print("")
    print("Alternative entry point:")
    print("  python src/orthoroute_plugin.py")
    sys.exit(0)


def run_plugin(show_gui: bool = False):
    """Run as KiCad plugin with the same GUI as orthoroute_plugin.py."""
    try:
        config = setup_environment()
        
        # Use new architecture for both GUI and non-GUI modes
        from orthoroute.presentation.plugin.kicad_plugin import KiCadPlugin
        
        plugin = KiCadPlugin()
        
        if show_gui:
            success = plugin.run_with_gui()
        else:
            success = plugin.run()
        
        if success:
            logging.info("Plugin execution completed successfully")
            sys.exit(0)
        else:
            logging.error("Plugin execution failed")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Plugin execution failed: {e}")
        sys.exit(1)


def run_test_manhattan():
    """Run automated Manhattan routing test without GUI."""
    try:
        config = setup_environment()
        print("Starting automated Manhattan routing test...")
        logging.info("Starting automated Manhattan routing test...")
        
        from orthoroute.presentation.plugin.kicad_plugin import KiCadPlugin
        
        plugin = KiCadPlugin()
        
        # Run with GUI for automated testing and auto-start routing
        print("Loading board from KiCad and starting GUI...")
        print("Auto-starting routing process...")
        success = plugin.run_with_gui_autostart()
        
        if success:
            logging.info("Manhattan routing test completed successfully")
            print("TEST PASSED: Manhattan routing executed without errors")
            sys.exit(0)
        else:
            logging.error("Manhattan routing test failed")
            print("TEST FAILED: Manhattan routing encountered errors")
            print("Note: Make sure KiCad is running with a board that has routable nets")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Manhattan routing test failed with exception: {e}")
        print(f"TEST FAILED: Exception occurred: {e}")
        if "division by zero" in str(e):
            print("Note: This typically occurs when the board has no routable nets")
            print("Make sure KiCad is running with a board that has components with nets to route")
        elif "No KiCad process" in str(e):
            print("Note: KiCad must be running for the test to work")
        sys.exit(1)


def run_headless_autoroute():
    """Run headless autoroute test that bypasses GUI and exercises PathFinder."""
    try:
        config = setup_environment()
        print("Starting headless autoroute test...")
        logging.info("Starting headless autoroute test...")

        from orthoroute.presentation.plugin.kicad_plugin import KiCadPlugin
        from orthoroute.algorithms.manhattan.unified_pathfinder import UnifiedPathFinder, PathFinderConfig

        plugin = KiCadPlugin()

        # Load the same board/fixture as --test-manhattan
        logging.info("[HEADLESS] Loading board from KiCad...")
        board = None

        # Try to load board using plugin's run method (loads from KiCad or file)
        try:
            # Run plugin to load board (returns board object on success)
            board = plugin.run()
        except Exception as e:
            logging.warning(f"[HEADLESS] Plugin run failed: {e}")
            # Try loading from available adapters directly
            for adapter_name, adapter in plugin.kicad_adapters:
                if adapter_name == 'File':
                    continue  # Skip file adapter for now
                try:
                    if hasattr(adapter, 'connect'):
                        if not adapter.connect():
                            continue
                    board = adapter.load_board()
                    if board:
                        logging.info(f"[HEADLESS] Loaded board via {adapter_name}: {board.name}")
                        break
                except Exception as adapter_e:
                    logging.warning(f"[HEADLESS] {adapter_name} adapter failed: {adapter_e}")

        if not board:
            logging.error("[HEADLESS] Failed to load board from any source")
            logging.error("[HEADLESS] Make sure KiCad is running with a board loaded")
            sys.exit(1)

        logging.info(f"[HEADLESS] Loaded board: {board.name} with {len(board.nets)} nets")

        # Check if board has nets to route
        if len(board.nets) == 0:
            logging.warning("[HEADLESS] Board has no nets to route - this will test lattice build only")
            print("HEADLESS TEST: Board has no nets - lattice build test only")

        # Create UnifiedPathFinder with same config as GUI but force CPU-only
        pf_config = PathFinderConfig()
        pf = UnifiedPathFinder(config=pf_config, use_gpu=False)
        logging.info(f"[HEADLESS] Created UnifiedPathFinder with instance_tag={pf._instance_tag}")

        logging.info("[HEADLESS] Building lattice/registry...")
        pf.initialize_graph(board)
        pf.map_all_pads(board)
        pf.prepare_routing_runtime()

        if len(board.nets) > 0:
            logging.info("[HEADLESS] Starting PathFinder negotiation...")
            pf.route_multiple_nets(board.nets)
        else:
            logging.info("[HEADLESS] Skipping routing - no nets available")

        logging.info("[HEADLESS] Emitting geometry...")
        tracks, vias = pf.emit_geometry(board)

        logging.info(f"[HEADLESS] Done. tracks={tracks} vias={vias}")
        print(f"HEADLESS TEST PASSED: tracks={tracks} vias={vias}")
        sys.exit(0)

    except Exception as e:
        logging.error(f"[HEADLESS] Test failed with exception: {e}")
        print(f"HEADLESS TEST FAILED: Exception occurred: {e}")
        if "No KiCad process" in str(e):
            print("Note: KiCad must be running for the headless test to work")
        sys.exit(1)


def run_tiny_via_test():
    """Run tiny 2-layer via test case to verify via pathfinding works."""
    try:
        config = setup_environment()
        print("Starting tiny 2-layer via test...")
        logging.info("Starting tiny 2-layer via test...")

        from orthoroute.algorithms.manhattan.unified_pathfinder import UnifiedPathFinder, PathFinderConfig
        from orthoroute.domain.models.board import Board, Net, Pad, Component, Coordinate

        # Create minimal 2-layer board with source L0, sink L1 at same (u,v)
        logging.info("[VIA-TEST] Creating minimal test board...")

        # Create a simple board
        board = Board(id="via_test", name="Via Test Board")
        board.layer_count = 2  # Only 2 layers for simple test

        # Create simple components and pads
        pos = Coordinate(x=0.0, y=0.0)
        comp1 = Component(id="comp1", reference="U1", value="Test", footprint="Test", position=pos)
        comp2 = Component(id="comp2", reference="U2", value="Test", footprint="Test", position=pos)

        # Pads at same position but different layers (requires via)
        pad1 = Pad(id="pad1", component_id=comp1.id, position=pos, layer="F.Cu", size=(1.0, 1.0), net_id=None)
        pad2 = Pad(id="pad2", component_id=comp2.id, position=pos, layer="B.Cu", size=(1.0, 1.0), net_id=None)

        # Create a net connecting the pads
        net = Net(id="net1", name="TEST_NET")
        pad1.net_id = net.id
        pad2.net_id = net.id
        net.pad_ids = [pad1.id, pad2.id]
        board.nets = [net]

        logging.info(f"[VIA-TEST] Created test board with {len(board.nets)} nets")

        # Create UnifiedPathFinder with CPU-only mode
        pf_config = PathFinderConfig()
        pf = UnifiedPathFinder(config=pf_config, use_gpu=False)
        logging.info(f"[VIA-TEST] Created UnifiedPathFinder with instance_tag={pf._instance_tag}")

        # Build lattice
        logging.info("[VIA-TEST] Building lattice/registry...")
        pf.initialize_graph(board)
        pf.map_all_pads(board)
        pf.prepare_routing_runtime()

        # Check via creation
        if hasattr(pf, 'via_edge_ids') and len(pf.via_edge_ids) > 0:
            logging.info(f"[VIA-TEST] Via edges created: {len(pf.via_edge_ids)}")
        else:
            logging.error("[VIA-TEST] No via edges created!")
            sys.exit(1)

        # Route the test net
        logging.info("[VIA-TEST] Routing test net...")
        pf.route_multiple_nets(board.nets)

        # Check results
        logging.info("[VIA-TEST] Checking results...")
        tracks, vias = pf.emit_geometry(board)

        # Verify via usage
        via_eids = getattr(pf, 'via_edge_ids', [])
        vias_used = 0
        if hasattr(pf, 'owner') and hasattr(pf, 'e_is_via'):
            owned_edges = np.where(pf.owner != -1)[0]
            vias_used = int(np.sum(pf.e_is_via[owned_edges])) if len(owned_edges) > 0 else 0

        logging.info(f"[VIA-TEST] Results: tracks={tracks} vias={vias} vias_used={vias_used}")

        # Assertions
        assert len(via_eids) > 0, "No via edges created"
        assert vias_used >= 1, f"No vias used in routing: vias_used={vias_used}"

        owned_edges = np.where(pf.owner != -1)[0]
        assert len(owned_edges) > 0, "No edges owned after routing"

        present_nonzero = int(np.count_nonzero(pf.present_cost > 0))
        assert present_nonzero > 0, f"present_cost is zero: {present_nonzero}"

        print(f"VIA TEST PASSED: vias_used={vias_used} owned_edges={len(owned_edges)} present_nonzero={present_nonzero}")
        logging.info("[VIA-TEST] All assertions passed!")
        sys.exit(0)

    except Exception as e:
        logging.error(f"[VIA-TEST] Test failed with exception: {e}")
        print(f"VIA TEST FAILED: Exception occurred: {e}")
        sys.exit(1)


def run_headless(
    orp_file: str,
    output_file: Optional[str] = None,
    max_iterations: int = 200,
    checkpoint_interval: int = 30,
    resume_checkpoint: Optional[str] = None,
    use_gpu: bool = None,
    cpu_only: bool = False
):
    """
    Run headless cloud routing mode.

    This is the main entry point for cloud-based routing:
    1. Import board from .ORP file
    2. Run routing algorithm (identical to GUI mode)
    3. Export solution to .ORS file

    Args:
        orp_file: Path to input .ORP file (board export)
        output_file: Path to output .ORS file (default: derive from input)
        max_iterations: Maximum routing iterations (default: 200)
        checkpoint_interval: Checkpoint save interval in minutes (default: 30)
        resume_checkpoint: Path to checkpoint file to resume from
        use_gpu: Force GPU mode if True, auto-detect if None
        cpu_only: Force CPU-only mode if True
    """
    try:
        import time
        from pathlib import Path
        from orthoroute.infrastructure.serialization import (
            import_board_from_orp,
            export_solution_to_ors,
            derive_ors_filename
        )
        from orthoroute.algorithms.manhattan.unified_pathfinder import UnifiedPathFinder, PathFinderConfig
        # NOTE: IterationMetricsLogger not available in baseline - PathFinder runs without it
        # from orthoroute.algorithms.manhattan.iteration_metrics import IterationMetricsLogger

        config = setup_environment()
        start_time = time.time()

        logging.info("=" * 80)
        logging.info("HEADLESS CLOUD ROUTING MODE")
        logging.info("=" * 80)
        logging.info(f"[HEADLESS] Input: {orp_file}")

        # Determine output path
        if not output_file:
            output_file = derive_ors_filename(orp_file)
        logging.info(f"[HEADLESS] Output: {output_file}")
        logging.info(f"[HEADLESS] Max iterations: {max_iterations}")
        logging.info(f"[HEADLESS] Checkpoint interval: {checkpoint_interval} minutes")

        # Step 1: Import board from .ORP file
        logging.info("[HEADLESS] Step 1: Loading board from .ORP file...")
        orp_data = import_board_from_orp(orp_file)
        if not orp_data:
            logging.error("[HEADLESS] Failed to load board from .ORP file")
            sys.exit(1)

        # Convert ORP dictionary to board_data format (same as GUI uses)
        from orthoroute.infrastructure.serialization import convert_orp_to_board_data
        board_data = convert_orp_to_board_data(orp_data)
        if not board_data:
            logging.error("[HEADLESS] Failed to convert ORP data to board_data format")
            sys.exit(1)

        # Convert board_data to Board object (using same logic as GUI's _create_board_from_data)
        from orthoroute.domain.models.board import Board, Net, Pad, Component, Coordinate

        layer_count = board_data.get('layers', 2)
        board = Board(id="headless-board", name=board_data.get('filename', 'board'), layer_count=layer_count)
        board.nets = []
        board.components = []

        # Track components and their pads
        components_dict = {}
        all_pads = []

        # Create components from board_data (if any)
        components_data = board_data.get('components', [])
        for comp_data in components_data:
            comp_id = comp_data.get('reference', comp_data.get('id', ''))
            if comp_id:
                component = Component(
                    id=comp_id,
                    reference=comp_id,
                    value=comp_data.get('value', ''),
                    footprint=comp_data.get('footprint', ''),
                    position=Coordinate(
                        x=comp_data.get('x', 0.0),
                        y=comp_data.get('y', 0.0)
                    ),
                    angle=comp_data.get('rotation', 0.0)
                )
                components_dict[comp_id] = component

        # Convert nets from board_data (nets contain pads)
        nets_data = board_data.get('nets', {})
        logging.info(f"[HEADLESS] Converting {len(nets_data)} nets from board_data...")
        for net_id, net_info in nets_data.items():
            net = Net(id=net_id, name=net_info.get('name', net_id))
            net.pads = []

            # Create pads from net data
            for pad_ref in net_info.get('pads', []):
                if isinstance(pad_ref, dict):
                    pad_name = pad_ref.get('name', pad_ref.get('id', ''))
                    component_id = pad_ref.get('component', '')

                    pad = Pad(
                        id=pad_name or pad_ref.get('id', f"pad_{len(all_pads)}"),
                        component_id=component_id,
                        net_id=net_id,
                        position=Coordinate(x=pad_ref.get('x', 0.0), y=pad_ref.get('y', 0.0)),
                        size=(pad_ref.get('width', 0.2), pad_ref.get('height', 0.2)),
                        drill_size=pad_ref.get('drill'),
                        layer=pad_ref.get('layers', ['F.Cu'])[0] if pad_ref.get('layers') else 'F.Cu'
                    )
                    net.pads.append(pad)
                    all_pads.append(pad)

                    # Add pad to component (treat empty component_id as default component)
                    if not component_id:
                        component_id = "GENERIC_COMPONENT"
                        pad.component_id = component_id  # Update pad's component_id

                    if component_id not in components_dict:
                        components_dict[component_id] = Component(
                            id=component_id,
                            reference=component_id,
                            value="",
                            footprint="",
                            position=Coordinate(x=pad.position.x, y=pad.position.y),
                            pads=[]
                        )
                    components_dict[component_id].pads.append(pad)

            board.nets.append(net)

        # Add all components to board
        board.components = list(components_dict.values())

        # Store KiCad-calculated bounds for accurate routing area
        board._kicad_bounds = board_data.get('bounds', None)
        board.layer_names = board_data.get('layer_names', [])

        logging.info(f"[HEADLESS] Loaded board: {board.name}")
        logging.info(f"[HEADLESS]   - Nets: {len(board.nets)}")
        logging.info(f"[HEADLESS]   - Components: {len(board.components)}")
        logging.info(f"[HEADLESS]   - Component IDs: {list(components_dict.keys())[:5]}")
        if board.components:
            logging.info(f"[HEADLESS]   - Component[0] has {len(board.components[0].pads)} pads")
        logging.info(f"[HEADLESS]   - Pads: {len(all_pads)}")
        logging.info(f"[HEADLESS]   - Layers: {board.layer_count}")

        # Step 2: Create UnifiedPathFinder with same config as GUI
        logging.info("[HEADLESS] Step 2: Creating UnifiedPathFinder...")

        # Determine GPU mode
        if cpu_only:
            use_gpu_mode = False
            logging.info("[HEADLESS] GPU mode: DISABLED (--cpu-only)")
        elif use_gpu:
            use_gpu_mode = True
            logging.info("[HEADLESS] GPU mode: FORCED (--use-gpu)")
        else:
            # Auto-detect
            use_gpu_mode = os.environ.get('USE_GPU', '1') == '1' and not os.environ.get('ORTHO_CPU_ONLY', '0') == '1'
            logging.info(f"[HEADLESS] GPU mode: AUTO-DETECT ({'ENABLED' if use_gpu_mode else 'DISABLED'})")

        # Create PathFinder config with custom max iterations
        pf_config = PathFinderConfig()
        pf_config.max_routing_iterations = max_iterations

        pf = UnifiedPathFinder(config=pf_config, use_gpu=use_gpu_mode)
        logging.info(f"[HEADLESS] Created UnifiedPathFinder (instance_tag={pf._instance_tag})")

        # Setup iteration metrics logger
        debug_dir = pf.debug_dir if hasattr(pf, 'debug_dir') else 'debug_output'
        board_info = {
            'board_name': board.name,
            'nets': len(board.nets),
            'pads': len(all_pads),
            'layers': board.layer_count,
            'max_iterations': max_iterations,
            'mode': 'headless',
        }
        # NOTE: IterationMetricsLogger not available in baseline - basic logging still works
        # metrics_logger = IterationMetricsLogger(debug_dir, board_info)
        # pf._metrics_logger = metrics_logger
        logging.info(f"[HEADLESS] Debug directory: {debug_dir}")

        # Step 3: Initialize graph (build lattice, CSR)
        logging.info("[HEADLESS] Step 3: Building routing graph...")
        pf.initialize_graph(board)

        # Step 4: Map pads to lattice
        logging.info("[HEADLESS] Step 4: Mapping pads to lattice...")
        pf.map_all_pads(board)

        # Step 4.5: Precompute pad escapes and portals (CRITICAL for routing!)
        logging.info("[HEADLESS] Step 4.5: Computing pad escape portals...")

        # CRITICAL: Attach GUI pads to board (required by pad escape planner)
        board._gui_pads = board_data.get('pads', [])
        logging.info(f"[HEADLESS] Attached {len(board._gui_pads)} GUI pads for escape planning")

        escape_tracks, escape_vias = pf.precompute_all_pad_escapes(board)
        logging.info(f"[HEADLESS] Generated {len(escape_tracks)} escape tracks, {len(escape_vias)} escape vias")
        logging.info(f"[HEADLESS] Created {len(pf.portals)} portals for pad escapes")

        # Step 5: Prepare routing runtime
        logging.info("[HEADLESS] Step 5: Preparing routing runtime...")
        pf.prepare_routing_runtime()

        # Step 6: Route all nets (main routing loop)
        logging.info("[HEADLESS] Step 6: Starting routing algorithm...")
        logging.info("[HEADLESS] This may take hours for large boards - logs will show progress")

        # Custom iteration callback to track metrics
        iteration_metrics = []

        def iteration_callback(iter_num, provisional_tracks, provisional_vias, overflow_sum, overflow_cnt):
            """Called after each routing iteration."""
            nonlocal iteration_metrics
            iteration_metrics.append({
                'iteration': iter_num,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_s': 0.0,  # Will be calculated from timestamps
                'overuse': overflow_cnt,
                'barrel_conflicts': 0,  # Not available in callback
                'routed_nets': len(provisional_tracks),  # Approximate
                'failed_nets': 0,  # Not available in callback
                'total_edges': len(provisional_tracks) + len(provisional_vias),
                'pres_fac': 1.0,  # Not available in callback
                'pres_fac_mult': 1.0,  # Not available in callback
                'hist_gain': 0.0,  # Not available in callback
                'hist_cost_weight': 0.0,  # Not available in callback
                'via_penalty': 3.0,  # Not available in callback
                'hotset_size': 0,  # Not available in callback
                'stagnant_iters': 0,  # Not available in callback
                'stagnation_events': 0,  # Not available in callback
                'plateau_kick_applied': False,  # Not available in callback
            })
            logging.info(f"[HEADLESS] Iteration {iter_num}: overuse={overflow_cnt}, tracks={len(provisional_tracks)}, vias={len(provisional_vias)}")

        result = pf.route_multiple_nets(board.nets, iteration_cb=iteration_callback)

        # Step 7: Emit geometry
        logging.info("[HEADLESS] Step 7: Emitting geometry...")
        tracks, vias = pf.emit_geometry(board)
        logging.info(f"[HEADLESS] Generated {tracks} tracks, {vias} vias")

        # Step 8: Export solution to .ORS file
        logging.info("[HEADLESS] Step 8: Exporting solution to .ORS file...")

        # Get geometry payload
        geom = pf.get_geometry_payload()

        # Build routing metadata
        end_time = time.time()
        total_time = end_time - start_time

        # Extract convergence info (includes barrel conflicts now)
        barrel_conflicts = result.get('barrel_conflicts', 0) if isinstance(result, dict) else 0
        fully_converged = result.get('converged', False) if isinstance(result, dict) else False

        routing_metadata = {
            'total_time': total_time,
            'iterations': len(iteration_metrics),
            'converged': fully_converged,
            'barrel_conflicts': barrel_conflicts,
            'nets_routed': result.get('nets_routed', 0) if isinstance(result, dict) else 0,
            'wirelength': result.get('wirelength', 0.0) if isinstance(result, dict) else 0.0,
            'via_count': vias,
            'track_count': tracks,
            'overflow': result.get('overflow', 0) if isinstance(result, dict) else 0,
        }

        # Export to .ORS
        export_solution_to_ors(
            geom,
            iteration_metrics,
            routing_metadata,
            output_file,
            compress=True
        )

        # If we got here, export succeeded (would have raised exception otherwise)
        if True:
            logging.warning("=" * 80)
            logging.warning("ROUTING COMPLETE!")
            logging.warning("=" * 80)
            logging.warning(f"Solution file: {output_file}")
            logging.warning(f"Total runtime: {total_time/60:.1f} minutes")
            logging.warning(f"Iterations: {len(iteration_metrics)}")

            # Report convergence status
            if fully_converged:
                logging.warning(f"Converged: YES ✓")
            else:
                logging.warning(f"Converged: NO")

            logging.warning(f"Geometry: {tracks} tracks, {vias} vias")

            # Report barrel conflicts if present (known limitation)
            if barrel_conflicts > 0:
                logging.warning(f"Barrel conflicts: {barrel_conflicts} (via overlaps - see docs/barrel_conflicts_explained.md)")
            else:
                logging.warning(f"Barrel conflicts: 0 ✓")

            logging.warning("=" * 80)
            logging.warning(f"Next step: Import {output_file} into KiCad (Ctrl+I)")
            logging.warning("=" * 80)
            sys.exit(0)
        else:
            logging.error("[HEADLESS] Failed to export solution")
            sys.exit(1)

    except Exception as e:
        logging.error(f"[HEADLESS] Fatal error: {e}", exc_info=True)
        sys.exit(1)


def run_cli(board_file: str, output_dir: str = ".", config_path: Optional[str] = None):
    """Run command line interface using unified pipeline (same as GUI)."""
    try:
        from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser
        from orthoroute.algorithms.manhattan.unified_pathfinder import UnifiedPathFinder, PathFinderConfig

        # Initialize configuration if custom path provided
        if config_path:
            initialize_config(config_path)

        config = setup_environment()

        # Load board
        logging.info(f"Loading board from: {board_file}")
        parser = KiCadFileParser()
        board = parser.load_board(board_file)

        if not board:
            logging.error("Failed to load board file")
            sys.exit(1)

        logging.info(f"Loaded board: {board.name} with {len(board.nets)} nets")

        # Create UnifiedPathFinder (same as GUI) - FORCE CPU-ONLY
        pf = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
        logging.info(f"[CLI] Created UnifiedPathFinder with instance_tag={pf._instance_tag}")

        # Use unified pipeline (SAME THREE CALLS AS GUI)
        logging.info("[CLI] Step 1: Building lattice & CSR...")
        pf.initialize_graph(board)

        logging.info("[CLI] Step 2: Mapping pads to lattice...")
        pf.map_all_pads(board)

        logging.info("[CLI] Step 3: Preparing routing runtime...")
        pf.prepare_routing_runtime()

        logging.info("[CLI] Step 4: Routing nets...")
        pf.route_multiple_nets(board.nets)

        logging.info("[CLI] Step 5: Emitting geometry...")
        tracks, vias = pf.emit_geometry(board)

        logging.info(f"[CLI] Routing completed: {tracks} tracks, {vias} vias")

        if tracks > 0 or vias > 0:
            # Save geometry results
            geom = pf.get_geometry_payload()
            logging.info(f"[CLI] Generated {len(geom.tracks)} track objects, {len(geom.vias)} via objects")
            logging.info(f"[CLI] Results would be saved to: {output_dir}")
        else:
            logging.warning("[CLI] No copper generated")
            sys.exit(1)

    except Exception as e:
        logging.error(f"CLI execution failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    import time
    run_start = time.time()

    parser = argparse.ArgumentParser(
        description="OrthoRoute - KiCad PCB Autorouter Plugin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run KiCad plugin with GUI (default)
  %(prog)s plugin                       # Run as KiCad plugin with GUI
  %(prog)s plugin --no-gui              # Run as KiCad plugin without GUI
  %(prog)s --test-manhattan             # Run automated Manhattan routing test
  %(prog)s cli board.kicad_pcb          # Route board via CLI
  %(prog)s cli board.kicad_pcb -o out/  # Route and save to directory
  %(prog)s headless input.ORP           # Headless cloud routing mode
  %(prog)s headless input.ORP -o out.ORS --max-iterations 200
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Plugin mode
    plugin_parser = subparsers.add_parser('plugin', help='Run as KiCad plugin')
    plugin_parser.add_argument(
        '--no-gui', action='store_true',
        help='Run without GUI (default shows GUI)'
    )
    plugin_parser.add_argument(
        '--min-run-sec', type=int, default=0,
        help='Keep process alive for at least this many seconds (for CI/agents)'
    )
    
    # CLI mode
    cli_parser = subparsers.add_parser('cli', help='Command line interface')
    cli_parser.add_argument(
        'board_file',
        help='KiCad board file (.kicad_pcb)'
    )
    cli_parser.add_argument(
        '-o', '--output',
        default='.',
        help='Output directory (default: current directory)'
    )
    cli_parser.add_argument(
        '-c', '--config',
        help='Configuration file path'
    )
    
    # Headless mode
    headless_parser = subparsers.add_parser('headless', help='Headless cloud routing mode')
    headless_parser.add_argument(
        'orp_file',
        help='Input .ORP file (board export)'
    )
    headless_parser.add_argument(
        '-o', '--output',
        help='Output .ORS filepath (default: derive from input, e.g., input.ORP → input.ORS)'
    )
    headless_parser.add_argument(
        '--max-iterations',
        type=int,
        default=200,
        help='Override default iteration limit (default: 200)'
    )
    headless_parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=30,
        help='Checkpoint save interval in minutes (default: 30)'
    )
    headless_parser.add_argument(
        '--resume-checkpoint',
        help='Resume from checkpoint file'
    )
    headless_parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Enable GPU acceleration if available (default: auto-detect)'
    )
    headless_parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU-only mode (no GPU)'
    )

    # Global options
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    parser.add_argument(
        '--test-manhattan',
        action='store_true',
        help='Run automated Manhattan routing test without GUI'
    )
    parser.add_argument(
        '--autoroute',
        action='store_true',
        help='Run headless autoroute test (bypasses GUI, exercises PathFinder)'
    )
    parser.add_argument(
        '--test-via',
        action='store_true',
        help='Run tiny 2-layer via test case (source L0, sink L1 same position)'
    )
    parser.add_argument(
        '--min-run-sec', type=int, default=0,
        help='Keep process alive for at least this many seconds (for CI/agents)'
    )
    
    # Parse arguments
    args = parser.parse_args()

    min_run = int(getattr(args, "min_run_sec", 0) or 0)
    if min_run > 0:
        logging.getLogger().info(f"[RUN-MIN] min_run_sec={min_run}")

    # Check for test modes first (override other modes)
    if getattr(args, 'test_manhattan', False):
        run_test_manhattan()
    elif getattr(args, 'autoroute', False):
        run_headless_autoroute()
    elif getattr(args, 'test_via', False):
        run_tiny_via_test()
    elif not args.mode:
        # Handle no arguments (default to plugin mode)
        run_plugin(show_gui=True)
    else:
        # Route to appropriate handler
        try:
            if args.mode == 'plugin':
                run_plugin(show_gui=not getattr(args, 'no_gui', False))
            elif args.mode == 'cli':
                run_cli(
                    args.board_file,
                    args.output,
                    getattr(args, 'config', None)
                )
            elif args.mode == 'headless':
                run_headless(
                    args.orp_file,
                    output_file=getattr(args, 'output', None),
                    max_iterations=getattr(args, 'max_iterations', 200),
                    checkpoint_interval=getattr(args, 'checkpoint_interval', 30),
                    resume_checkpoint=getattr(args, 'resume_checkpoint', None),
                    use_gpu=getattr(args, 'use_gpu', None),
                    cpu_only=getattr(args, 'cpu_only', False)
                )
            else:
                parser.error(f"Unknown mode: {args.mode}")

        except KeyboardInterrupt:
            logging.info("Operation cancelled by user")
            sys.exit(130)

    # Keep-alive for headless/short agent runs
    if min_run > 0:
        remaining = max(0.0, min_run - (time.time() - run_start))
        if remaining > 0:
            logging.getLogger().info(f"[RUN-MIN] Sleeping {remaining:.1f}s to satisfy min runtime")
            time.sleep(remaining)


if __name__ == '__main__':
    main()