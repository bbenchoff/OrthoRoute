"""Keepout rule-area enforcement (lifted from PR #17 by RolandWa).

A keepout polygon must price its interior edges out of the graph and the
router must route around it - verified end to end on a synthetic board.
"""

import numpy as np
import pytest

from benchmarks.synthetic_boards import make_backplane


@pytest.fixture(scope="module")
def routed_with_keepout():
    from orthoroute.algorithms.manhattan.unified_pathfinder import (
        PathFinderConfig, UnifiedPathFinder,
    )
    board = make_backplane(connectors=2, pins_per=8, layers=4, pattern="neighbor")
    # Partial strip in the middle (x 12..14mm, y up to 8mm): blocks the
    # direct corridor for the lower nets but leaves a detour below/above.
    # (A full-height strip would bisect the board -> genuinely unroutable.)
    board.keepouts = [{
        "name": "TEST_KEEPOUT",
        "layers": [],  # empty -> all layers
        "outline": [[12.0, 0.0], [14.0, 0.0], [14.0, 8.0], [12.0, 8.0]],
        "keepout_tracks": True,
        "keepout_vias": True,
    }]
    config = PathFinderConfig()
    config.portal_x_snap_max = 0.75
    config.max_iterations = 150
    pf = UnifiedPathFinder(config=config, use_gpu=False)
    pf.initialize_graph(board)
    pf.map_all_pads(board)
    pf.precompute_all_pad_escapes(board)
    pf.prepare_routing_runtime()
    pf.route_multiple_nets(board.nets)
    return pf, board


def test_keepout_edges_blocked(routed_with_keepout):
    pf, _ = routed_with_keepout
    costs = np.asarray(pf.graph.base_costs)
    assert (costs >= 1e9).sum() > 0, "no edges were blocked by the keepout"


def test_routes_avoid_keepout(routed_with_keepout):
    pf, _ = routed_with_keepout
    plane = pf.lattice.x_steps * pf.lattice.y_steps
    b = pf.lattice.bounds
    pitch = pf.lattice.pitch
    for name, path in pf.net_paths.items():
        for node in path:
            rem = node % plane
            x_mm = b[0] + (rem % pf.lattice.x_steps) * pitch
            y_mm = b[1] + (rem // pf.lattice.x_steps) * pitch
            # No path node may sit strictly inside the keepout polygon.
            assert not (12.0 < x_mm < 14.0 and 0.0 < y_mm < 8.0), \
                f"{name} routed through keepout at ({x_mm:.1f}, {y_mm:.1f})mm"


def test_all_nets_still_route_around_keepout(routed_with_keepout):
    pf, board = routed_with_keepout
    routed = sum(1 for p in pf.net_paths.values() if len(p) >= 2)
    assert routed == len(board.nets), "keepout made board unroutable"


def test_zero_overuse_around_keepout(routed_with_keepout):
    pf, _ = routed_with_keepout
    total, _ = pf.accounting.compute_overuse(pf)
    assert total == 0
