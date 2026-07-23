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


def test_selected_escape_geometry_is_physically_conflict_free(routed):
    pf, _, _, _ = routed
    pf._rebuild_escape_occupancy()
    conflicts, _, _ = pf._detect_escape_conflicts()
    assert conflicts == set()
    assert set(pf._escape_preferred_portals) == {
        pad_id
        for pad_ids in pf.net_pad_ids.values()
        for pad_id in pad_ids
    }
    assert {
        record["pad"]
        for record in pf._escape_reserved_records.values()
    } == set(pf._escape_preferred_portals)


def test_escape_distance_detects_crossing_segments():
    distance = UnifiedPathFinder._segment_distance(
        ((0.0, 0.0), (1.0, 1.0)),
        ((0.0, 1.0), (1.0, 0.0)),
    )
    assert distance == 0.0


def test_horizontal_escape_uses_short_orthogonal_dogleg(routed):
    pf, _, _, _ = routed
    segments = pf.escape_planner._escape_segments(
        0.0, 0.0, 4.0, 0.2
    )

    assert segments == [
        ((0.0, 0.0), (3.8, 0.0)),
        ((3.8, 0.0), (4.0, 0.2)),
    ]


def test_escape_conflicts_identify_both_portals_for_history(routed):
    pf, _, _, _ = routed
    portal = pf.net_selected_portals["TEST_NET"][0]
    pf._escape_records.clear()
    pf._escape_spatial.clear()
    for net_id, pad_id in (("FIRST", "PAD_A"), ("SECOND", "PAD_B")):
        pf._insert_escape_record(
            pf._escape_record(net_id, pad_id, portal)
        )

    conflicts, _, _ = pf._detect_escape_conflicts()

    assert len(conflicts) == 1
    assert pf._escape_conflict_portal_keys(conflicts) == {
        ("PAD_A", portal.x_idx, portal.y_idx),
        ("PAD_B", portal.x_idx, portal.y_idx),
    }
    pf._rebuild_escape_occupancy()


def test_committed_escape_conflicts_never_eliminate_all_seeds(routed):
    pf, _, _, _ = routed
    pad_id = pf.net_pad_ids["TEST_NET"][0]
    candidates = pf.portal_candidates[pad_id]
    old_strict = pf._escape_reservations_strict
    pf._escape_records.clear()
    pf._escape_spatial.clear()
    for index, portal in enumerate(candidates):
        pf._insert_escape_record(
            pf._escape_record(
                "BLOCKER",
                f"BLOCKER_PAD_{index}",
                portal,
            )
        )
    pf._escape_reservations_strict = False

    seeds, seed_portals = pf._get_pad_portal_seeds(
        pad_id, "TEST_NET"
    )

    assert seeds
    assert seed_portals
    pf._escape_reservations_strict = old_strict
    pf._rebuild_escape_occupancy()


def test_stagnation_rip_clears_all_geometry_ownership():
    board = make_two_pad_board(layer_count=4)
    config = PathFinderConfig()
    config.portal_x_snap_max = 0.75
    pf = UnifiedPathFinder(config=config, use_gpu=False)
    pf.initialize_graph(board)
    pf.precompute_all_pad_escapes(board)
    pf.route_multiple_nets(board.nets)

    path = list(pf.net_paths["TEST_NET"])
    path_nodes = pf._unique_path_nodes(path)
    via_nodes = pf._via_nodes_for_path(path)
    net_id = pf._get_net_id("TEST_NET")
    pf.locked_nets.clear()
    edge = pf._net_to_edges["TEST_NET"][0]
    pf.accounting.present[edge] = pf.accounting.capacity[edge] + 1

    victims = pf._rip_top_k_offenders(k=1)

    assert victims == {"TEST_NET"}
    assert not pf.net_paths["TEST_NET"]
    assert "TEST_NET" not in pf.net_selected_portals
    assert np.all(pf.path_node_use[path_nodes] == 0)
    assert all(
        net_id not in pf._node_owner_members.get(node, ())
        for node in via_nodes
    )
    assert all(
        record["net"] != "TEST_NET"
        for record in pf._escape_records.values()
    )


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


def test_path_node_use_prices_tracks_for_later_vias(routed):
    """Planar copper must be visible before a later net chooses a via."""
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    path_nodes = pf._unique_path_nodes(path)
    via_nodes = set(pf._via_nodes_for_path(path))
    planar_node = next(node for node in path_nodes if node not in via_nodes)

    pf._rebuild_path_node_use()
    assert np.all(pf.path_node_use[path_nodes] == 1)

    old_pres_fac = getattr(pf, "_pres_fac_now", 1.0)
    pf._pres_fac_now = 2.0
    penalty = pf._build_owner_penalty(None, "OTHER_NET")
    expected = pf.config.path_node_penalty_base * pf._pres_fac_now
    assert penalty[planar_node] == pytest.approx(expected)
    pf._pres_fac_now = old_pres_fac

    pf._clear_path_node_use(path)
    assert np.all(pf.path_node_use[path_nodes] == 0)
    pf._mark_path_node_use(path)
    assert np.all(pf.path_node_use[path_nodes] == 1)


def test_barrel_owner_routes_before_crossing_track(routed):
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    tasks = {
        "BARREL_OWNER": (path[0], path[-1]),
        "TRACK_VICTIM": (path[0], path[-1]),
    }
    pf._barrel_owner_nets = {"BARREL_OWNER"}
    pf._barrel_victim_nets = {"TRACK_VICTIM"}

    ordered = pf._order_nets_by_difficulty(tasks)

    assert ordered.index("BARREL_OWNER") < ordered.index("TRACK_VICTIM")


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


def test_via_pool_overuse_selects_its_nets(routed):
    """A density-only violation must not produce an empty hotset."""
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    via_hop = next(
        (u, v)
        for u, v in zip(path, path[1:])
        if pf.lattice.idx_to_coord(u)[:2]
        == pf.lattice.idx_to_coord(v)[:2]
        and pf.lattice.idx_to_coord(u)[2]
        != pf.lattice.idx_to_coord(v)[2]
    )
    x_idx, y_idx, _ = pf.lattice.idx_to_coord(via_hop[0])

    pf._rebuild_via_usage_from_committed()
    old_capacity = int(pf.via_col_cap[x_idx, y_idx])
    pf.via_col_cap[x_idx, y_idx] = 0
    try:
        assert "TEST_NET" in pf._find_via_pool_offenders()
        pf.accounting.history.fill(0)
        hotset = pf._build_hotset({
            "TEST_NET": (path[0], path[-1]),
        })
        assert "TEST_NET" in hotset
    finally:
        pf.via_col_cap[x_idx, y_idx] = old_capacity


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
