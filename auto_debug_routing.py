#!/usr/bin/env python3
"""
Automated Routing Debug Harness

Runs routing with detailed diagnostics, analyzes failures, and suggests fixes.
"""

import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_routing_test():
    """Run routing with detailed diagnostics."""
    logger.info("="*80)
    logger.info("AUTOMATED ROUTING DEBUG SESSION")
    logger.info("="*80)

    try:
        # Import after logging setup
        from orthoroute.algorithms.manhattan.unified_pathfinder import UnifiedPathFinder
        from orthoroute.algorithms.manhattan.pathfinder import PathFinderConfig
        from orthoroute.domain.models.board import Board, Component, Pad, Net

        logger.info("✓ Imports successful")

        # Create minimal test board
        logger.info("Creating test board with 12 layers...")
        board = Board(id="test", name="Test Board", layer_count=12)

        # Add test component with 4 pads (2 nets)
        comp = Component(id="U1", reference="U1", x=200.0, y=100.0)

        # Create 4 pads in a grid pattern
        pads = [
            Pad(id="1", name="1", net_name="TEST_NET_1", x=200.0, y=100.0, width=1.0, height=1.0),
            Pad(id="2", name="2", net_name="TEST_NET_1", x=210.0, y=100.0, width=1.0, height=1.0),
            Pad(id="3", name="3", net_name="TEST_NET_2", x=200.0, y=110.0, width=1.0, height=1.0),
            Pad(id="4", name="4", net_name="TEST_NET_2", x=210.0, y=110.0, width=1.0, height=1.0),
        ]

        for pad in pads:
            comp.add_pad(pad)
        board.add_component(comp)

        # Create nets
        net1 = Net(id="TEST_NET_1", name="TEST_NET_1")
        net1.add_pad(pads[0])
        net1.add_pad(pads[1])
        board.add_net(net1)

        net2 = Net(id="TEST_NET_2", name="TEST_NET_2")
        net2.add_pad(pads[2])
        net2.add_pad(pads[3])
        board.add_net(net2)

        logger.info(f"✓ Test board created: {len(board.components)} components, {len(board.nets)} nets, {len(pads)} pads")

        # Create PathFinder with debug config
        config = PathFinderConfig()
        config.max_iterations = 5
        config.max_search_nodes = 100000

        logger.info(f"Creating PathFinder: max_search={config.max_search_nodes}, max_iters={config.max_iterations}")
        pf = UnifiedPathFinder(config=config, use_gpu=False)

        logger.info("Initializing graph...")
        pf.initialize_graph(board)
        logger.info("✓ Graph initialized")

        logger.info("Mapping pads...")
        pf.map_all_pads(board)
        logger.info("✓ Pads mapped")

        logger.info("Starting routing...")
        start_time = time.time()

        # Route with detailed logging
        result = pf.route_multiple_nets(board.nets)

        elapsed = time.time() - start_time
        logger.info(f"Routing completed in {elapsed:.2f}s")
        logger.info(f"Result: {len(result)} nets routed")

        # Analyze results
        if len(result) == len(board.nets):
            logger.info("✓✓✓ SUCCESS: All nets routed!")
            return True
        else:
            logger.error(f"✗ FAILURE: {len(result)}/{len(board.nets)} nets routed")

            # Check for detailed failure info
            if hasattr(pf, '_routing_result'):
                logger.error(f"Routing result: {pf._routing_result}")

            return False

    except Exception as e:
        logger.error(f"✗✗✗ FATAL ERROR: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting automated debug session...")
    success = run_routing_test()

    if success:
        logger.info("="*80)
        logger.info("AUTOMATED TEST: PASSED ✓")
        logger.info("="*80)
        sys.exit(0)
    else:
        logger.info("="*80)
        logger.info("AUTOMATED TEST: FAILED ✗")
        logger.info("="*80)
        logger.info("\nCheck auto_debug.log for details")
        sys.exit(1)