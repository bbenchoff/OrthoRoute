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


def test_portal_entry_is_negotiated_without_outer_layer_routing(routed):
    """Portal stubs must via in place before any lateral routing."""
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    coords = [pf.lattice.idx_to_coord(node) for node in path]

    assert coords[0][2] == 0
    assert coords[-1][2] == 0
    entry_layer, exit_layer = pf.net_portal_layers["TEST_NET"]
    assert 0 < entry_layer < pf.lattice.layers - 1
    assert 0 < exit_layer < pf.lattice.layers - 1

    for (ax, ay, az), (bx, by, bz) in zip(coords, coords[1:]):
        if az == bz:
            assert az not in (0, pf.lattice.layers - 1), (
                "path moved laterally on an outer copper layer"
            )


def test_portal_vias_attach_from_either_outer_layer(routed):
    """Back-side pads need the same in-place via chain as front-side pads."""
    from dataclasses import replace

    pf, _, _, _ = routed
    src_pad, dst_pad = pf.net_pad_ids["TEST_NET"]
    src = replace(pf.portals[src_pad], pad_layer=pf.lattice.layers - 1)
    dst = replace(pf.portals[dst_pad], pad_layer=pf.lattice.layers - 1)
    inner_path = [
        pf.lattice.node_idx(src.x_idx, src.y_idx, 1),
        pf.lattice.node_idx(src.x_idx, src.y_idx, 2),
        pf.lattice.node_idx(dst.x_idx, dst.y_idx, 2),
    ]

    attached = pf._attach_portal_vias(inner_path, src, dst)
    layers = [pf.lattice.idx_to_coord(node)[2] for node in attached]
    assert layers[:3] == [3, 2, 1]
    assert layers[-2:] == [2, 3]


def test_escape_planner_collects_distinct_portal_candidates(routed):
    pf, _, _, _ = routed
    all_cells = []
    for pad_id in pf.net_pad_ids["TEST_NET"]:
        candidates = pf.escape_planner.portal_candidates[pad_id]
        assert candidates[0] is pf.portals[pad_id]
        assert len(candidates) >= 2
        cells = {(portal.x_idx, portal.y_idx) for portal in candidates}
        assert len(cells) == len(candidates)
        all_cells.extend(cells)
    assert len(all_cells) == len(set(all_cells))


def test_selected_portals_match_path_and_emitted_stubs(routed):
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    selected = pf.net_selected_portals["TEST_NET"]
    first = pf.lattice.idx_to_coord(path[0])
    last = pf.lattice.idx_to_coord(path[-1])

    assert first[:2] == (selected[0].x_idx, selected[0].y_idx)
    assert last[:2] == (selected[1].x_idx, selected[1].y_idx)

    portal_points = {
        tuple(
            round(value, 6)
            for value in pf.lattice.geom.lattice_to_world(
                portal.x_idx, portal.y_idx
            )
        )
        for portal in selected
    }
    stub_points = {
        (round(track["x2"], 6), round(track["y2"], 6))
        for track in pf._escape_tracks
        if track["net"] == "TEST_NET"
    }
    assert portal_points <= stub_points


def test_via_ownership_is_reversible_and_tracks_collisions(routed):
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    via_nodes = pf._via_nodes_for_path(path)
    original_id = pf._get_net_id("TEST_NET")
    other_id = pf._get_net_id("OTHER_NET")

    pf._rebuild_node_owner()
    assert all(pf.node_owner[node] == original_id for node in via_nodes)

    pf._mark_via_barrel_ownership_for_path("OTHER_NET", path)
    assert all(pf.node_owner[node] == -2 for node in via_nodes)
    _, conflicts = pf._detect_barrel_conflicts()
    assert conflicts > 0
    assert {"TEST_NET", "OTHER_NET"} <= pf._barrel_conflict_nets

    pf._clear_via_barrel_ownership_for_path("OTHER_NET", path)
    assert all(pf.node_owner[node] == original_id for node in via_nodes)
    assert pf._detect_barrel_conflicts()[1] == 0
    assert all(
        pf._node_owner_members[node] == {original_id}
        for node in via_nodes
    )
    assert other_id not in {
        owner
        for members in pf._node_owner_members.values()
        for owner in members
    }


def test_adjacent_via_chain_is_one_physical_column(routed):
    """Adjacent graph hops in one barrel count as one physical via."""
    pf, _, _, _ = routed
    source_portal = pf.net_selected_portals["TEST_NET"][0]
    x_idx, y_idx = source_portal.x_idx, source_portal.y_idx
    via_chain = [
        pf.lattice.node_idx(x_idx, y_idx, layer)
        for layer in (0, 1, 2)
    ]

    pf.via_col_use.fill(0)
    pf.via_seg_use.fill(0)
    pf._accumulate_via_usage_for_path(via_chain)

    assert pf.via_col_use[
        source_portal.x_idx, source_portal.y_idx
    ] == 1
    assert pf.via_seg_use[x_idx, y_idx].sum() == 1

    # Geometry for a committed net must not add the same barrel again.
    x_mm, y_mm = pf.lattice.geom.lattice_to_world(x_idx, y_idx)
    original_escape_vias = pf._escape_vias
    pf._escape_vias = [{
        "net": "TEST_NET",
        "x": x_mm,
        "y": y_mm,
        "from_layer": "F.Cu",
        "to_layer": "In1.Cu",
    }]
    pf._rebuild_via_usage_from_committed()
    assert pf.via_col_use[x_idx, y_idx] == 1

    # Restore the module-scoped fixture for the remaining assertions.
    pf._escape_vias = original_escape_vias


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


def test_two_layer_route_end_to_end():
    """Regression for #18/#13: a 2-layer board must route, not crash.

    Previously initialize_graph raised ValueError("No edges") because no
    inner layers exist on a 2-layer board and lateral edges were only
    emitted for inner layers.
    """
    board = make_two_pad_board(layer_count=2)
    config = PathFinderConfig()
    config.portal_x_snap_max = 0.75
    pf = UnifiedPathFinder(config=config, use_gpu=False)

    pf.initialize_graph(board)
    pf.map_all_pads(board)
    pf.precompute_all_pad_escapes(board)
    pf.prepare_routing_runtime()
    pf.route_multiple_nets(board.nets)
    tracks, vias = pf.emit_geometry(board)

    path = pf.net_paths.get("TEST_NET", [])
    assert len(path) >= 2, "2-layer net was not routed"
    assert tracks > 0
    total, count = pf.accounting.compute_overuse(pf)
    assert (total, count) == (0, 0)
