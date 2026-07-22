"""End-to-end CPU-only engine smoke test (pytest twin of --test-via).

Routes one two-pad net on a synthetic 4-layer board through the full live
call sequence. This is the KiCad-free gate every backend change must pass.
"""

import numpy as np
import pytest

from orthoroute.algorithms.manhattan.unified_pathfinder import (
    PathFinderConfig,
    UnifiedPathFinder,
)

from conftest import make_two_pad_board


@pytest.fixture(scope="module")
def routed():
    """Route the synthetic board once; tests inspect the result."""
    board = make_two_pad_board(layer_count=4)
    config = PathFinderConfig()
    # 3.0mm ROUTING_MARGIN = 7.5 grid steps: pads sit half a pitch off-grid,
    # so allow the escape planner to snap them (see --test-via).
    config.portal_x_snap_max = 0.75
    pf = UnifiedPathFinder(config=config, use_gpu=False)

    pf.initialize_graph(board)
    pf.map_all_pads(board)
    pf.precompute_all_pad_escapes(board)
    pf.prepare_routing_runtime()
    pf.route_multiple_nets(board.nets)
    tracks, vias = pf.emit_geometry(board)
    return pf, board, tracks, vias


def test_net_routed(routed):
    pf, board, _, _ = routed
    path = pf.net_paths.get("TEST_NET", [])
    assert len(path) >= 2


def test_path_uses_via(routed):
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    plane = pf.lattice.x_steps * pf.lattice.y_steps
    layer_changes = sum(1 for a, b in zip(path, path[1:]) if a // plane != b // plane)
    assert layer_changes >= 1


def test_path_respects_hv_discipline(routed):
    """Every lateral step must follow its layer's legal axis."""
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    for a, b in zip(path, path[1:]):
        ax, ay, az = pf.lattice.idx_to_coord(a)
        bx, by, bz = pf.lattice.idx_to_coord(b)
        if az == bz:  # lateral move
            assert pf.lattice.is_legal_planar_edge(ax, ay, az, bx, by, bz) or \
                   pf.lattice.is_legal_planar_edge(bx, by, bz, ax, ay, az)
        else:  # via move: same (x, y)
            assert (ax, ay) == (bx, by)


def test_converged_no_overuse(routed):
    pf, _, _, _ = routed
    total, count = pf.accounting.compute_overuse(pf)
    assert (total, count) == (0, 0)


def test_geometry_emitted(routed):
    _, _, tracks, vias = routed
    assert tracks > 0
    assert vias >= 1  # at least the escape/path vias


def test_present_matches_canonical(routed):
    pf, _, _, _ = routed
    assert pf.accounting.verify_present_matches_canonical()


def test_deterministic_across_runs(routed):
    """Seeded RNG + stable sorts: a second identical run yields the same path."""
    pf, _, _, _ = routed
    first_path = list(pf.net_paths["TEST_NET"])

    board2 = make_two_pad_board(layer_count=4)
    config2 = PathFinderConfig()
    config2.portal_x_snap_max = 0.75
    pf2 = UnifiedPathFinder(config=config2, use_gpu=False)
    pf2.initialize_graph(board2)
    pf2.map_all_pads(board2)
    pf2.precompute_all_pad_escapes(board2)
    pf2.prepare_routing_runtime()
    pf2.route_multiple_nets(board2.nets)

    assert list(pf2.net_paths["TEST_NET"]) == first_path
