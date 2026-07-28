"""End-to-end CPU-only engine smoke test (pytest twin of --test-via).

Routes one two-pad net on a synthetic 4-layer board through the full live
call sequence. This is the KiCad-free gate every backend change must pass.
"""

import numpy as np
import pytest

from orthoroute.algorithms.manhattan.unified_pathfinder import (
    EdgeAccountant,
    PathFinderConfig,
    PathFinderRouter,
    UnifiedPathFinder,
)

from conftest import make_two_pad_board


def test_cost_balance_ignores_legal_capacity_occupancy():
    accounting = EdgeAccountant(4)
    accounting.capacity[:] = 1
    accounting.present_ema[:] = [0, 1, 2, 4]
    accounting.history[:] = [5, 5, 5, 5]

    # Only excess uses [0, 0, 1, 3] contribute to present cost:
    # history = 2 * 20, present = 10 * 4.
    assert accounting.cost_balance_ratio(2, 10) == pytest.approx(1.0)


def _make_columnar_connector_board(columns=8):
    """Two regular columnar SMD arrays joined pad-for-pad."""
    from orthoroute.domain.models.board import (
        Board, Component, Coordinate, Net, Pad,
    )

    board = Board(id="columnar", name="Columnar connectors")
    board.layer_count = 6
    components = []
    pad_sets = []
    row_offsets = (-1.7, 1.6, 3.2, 6.5)
    for reference, base_y in (("J1", 10.0), ("J2", 30.0)):
        component = Component(
            id=reference,
            reference=reference,
            value="Dense",
            footprint="Dense",
            position=Coordinate(5.15, base_y),
        )
        pads = []
        for column in range(columns):
            for row, y_offset in enumerate(row_offsets):
                pad = Pad(
                    id=f"{reference}_{column}_{row}",
                    component_id=reference,
                    net_id=None,
                    position=Coordinate(
                        5.15 + 0.4 * column,
                        base_y + y_offset,
                    ),
                    size=(0.2, 1.15),
                    layer="F.Cu",
                    shape="rect",
                )
                component.pads.append(pad)
                pads.append(pad)
        board.add_component(component)
        components.append(component)
        pad_sets.append(pads)

    for index, (first, second) in enumerate(zip(*pad_sets)):
        board.add_net(Net(
            id=f"N{index}",
            name=f"N{index}",
            pads=[first, second],
        ))
    return board


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


def test_routing_state_snapshot_is_independent_and_restorable(routed):
    """Long negotiation must preserve an earlier, better export state."""
    pf, _, _, _ = routed
    net_id = "TEST_NET"
    live_path = list(pf.net_paths[net_id])
    live_portal_y = pf.net_selected_portals[net_id][0].y_idx

    mutated_snapshot = pf._capture_routing_state()
    mutated_snapshot["paths"][net_id][0] = -1
    mutated_snapshot["selected_portals"][net_id][0].y_idx += 1
    assert pf.net_paths[net_id] == live_path
    assert pf.net_selected_portals[net_id][0].y_idx == live_portal_y

    clean_snapshot = pf._capture_routing_state()
    expected_edges = pf._path_to_edges(
        pf._path_without_dynamic_escape_chains(net_id, live_path)
    )
    pf.net_paths.clear()
    pf.net_selected_portals.clear()
    pf.net_portal_layers.clear()
    pf.accounting.canonical.clear()
    pf.accounting.present.fill(0)
    pf._net_to_edges.clear()
    pf._edge_to_nets.clear()

    pf._restore_routing_state(clean_snapshot)

    assert pf.net_paths[net_id] == live_path
    assert pf.net_selected_portals[net_id][0].y_idx == live_portal_y
    assert pf._net_to_edges[net_id] == expected_edges
    assert pf.accounting.verify_present_matches_canonical()
    for edge_idx in expected_edges:
        assert pf.accounting.canonical[edge_idx] >= 1


def test_restored_route_can_be_ripped_as_a_new_recovery_branch(routed):
    """A stagnation retry must branch from restored canonical occupancy."""
    pf, _, _, _ = routed
    state = pf._capture_routing_state()
    net_id = "TEST_NET"
    expected_edges = list(pf._net_to_edges[net_id])

    pf.net_paths[net_id] = []
    pf.net_selected_portals.pop(net_id, None)
    pf.accounting.canonical.clear()
    pf.accounting.present.fill(0)
    pf._net_to_edges.clear()
    pf._edge_to_nets.clear()
    pf._restore_routing_state(state)

    pf.locked_nets.discard(net_id)
    victims = pf._rip_top_k_offenders(k=1)

    # The one-net fixture is congestion-free, so force the same bookkeeping
    # path directly when no natural offender exists.
    if not victims:
        pf.accounting.present[expected_edges[0]] = (
            pf.accounting.capacity[expected_edges[0]] + 1
        )
        victims = pf._rip_top_k_offenders(k=1)
    assert victims == {net_id}
    assert pf.net_paths[net_id] == []
    assert net_id not in pf._net_to_edges
    for edge_idx in expected_edges:
        assert pf.accounting.canonical.get(edge_idx, 0) == 0

    # Preserve the module-scoped fixture for the remaining tests.
    pf._restore_routing_state(state)


def test_stagnation_recovery_ranks_unique_edges_and_node_conflicts():
    class RecoveryFixture:
        pass

    pf = RecoveryFixture()
    pf.net_paths = {"EDGE_NET": [1], "NODE_NET": [2]}
    pf.locked_nets = set()
    pf._net_to_edges = {
        "EDGE_NET": [0, 1],
        "NODE_NET": [],
    }
    pf._edge_to_nets = {
        0: {"EDGE_NET"},
        1: {"EDGE_NET"},
    }
    pf._canonical_edge_resource_mask = lambda: np.asarray(
        [True, False, True],
        dtype=bool,
    )
    pf._detect_path_node_conflicts = lambda: (
        set(),
        {2, 3, 4},
        {"NODE_NET": 3},
    )

    scores = UnifiedPathFinder._rank_stagnation_offenders(
        pf,
        np.asarray([2, 2, 0], dtype=np.float32),
    )

    # The mirrored reverse arc is not counted, while node-only offenders
    # participate in the same recovery ranking.
    assert scores == [(3.0, "NODE_NET"), (2.0, "EDGE_NET")]
    assert pf._path_node_conflict_scores == {"NODE_NET": 3}


def test_stagnation_recovery_rotates_victims_within_one_best_basin():
    class RecoveryFixture:
        pass

    pf = RecoveryFixture()
    pf._stagnation_victim_history = set()
    scores = [(3.0, "NODE_NET"), (2.0, "EDGE_NET")]

    assert UnifiedPathFinder._select_stagnation_victims(
        pf, scores, 1
    ) == {"NODE_NET"}

    pf._stagnation_victim_history = set()
    wider_scores = scores + [(1.0, "THIRD_NET")]
    assert UnifiedPathFinder._select_stagnation_victims(
        pf, wider_scores, 2
    ) == {"NODE_NET", "EDGE_NET"}
    # Finish the one-net tail before wrapping to the top of a new cycle.
    assert UnifiedPathFinder._select_stagnation_victims(
        pf, wider_scores, 2
    ) == {"THIRD_NET", "NODE_NET"}
    assert UnifiedPathFinder._select_stagnation_victims(
        pf, scores, 1
    ) == {"EDGE_NET"}
    assert UnifiedPathFinder._select_stagnation_victims(
        pf, scores, 1
    ) == {"NODE_NET"}


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


def test_columnar_connectors_use_zero_conflict_dynamic_entries():
    board = _make_columnar_connector_board()
    config = PathFinderConfig()
    config.track_width = 0.1016
    config.clearance = 0.1016
    config.via_diameter = 0.4
    config.via_drill = 0.15
    # Keep this portal-seed test focused on multiple H entry depths. Demand-
    # aware defaults are covered separately in test_board_analyzer.py.
    config.preferred_layer_directions = [
        "v", "h", "v", "h", "v", "h",
    ]
    pf = UnifiedPathFinder(config=config, use_gpu=False)

    pf.initialize_graph(board)
    pf.precompute_all_pad_escapes(board)
    tasks = pf._parse_requests(board.nets)
    pf._plan_escape_assignment()

    assert len(tasks) == 32
    assert len(pf.portals) == 64
    assert all(portal.dynamic_entry for portal in pf.portals.values())
    assert all(
        len(candidates) >= 2
        for candidates in pf.portal_candidates.values()
    )
    assert pf._escape_reservations_strict
    assert pf._escape_assignment_conflicts == set()

    pad_id = pf.net_pad_ids["N0"][0]
    portal = pf.portals[pad_id]
    assert {
        candidate.delta_steps
        for candidate in pf.portal_candidates[pad_id]
    } >= {3, 4}
    portal_seeds = pf._get_portal_seeds(portal)
    assert {
        pf.lattice.idx_to_coord(node)[2]
        for node, _ in portal_seeds
    } == {1, 3}
    seed_costs = {
        pf.lattice.idx_to_coord(node)[2]: cost
        for node, cost in portal_seeds
    }
    assert seed_costs[1] == pytest.approx(
        config.via_cost * config.portal_via_discount
    )
    assert seed_costs[3] == pytest.approx(3.0 * seed_costs[1])
    entry_node = pf.lattice.node_idx(
        portal.x_idx, portal.y_idx, 1
    )
    baseline_costs = dict(pf._get_pad_portal_seeds(
        pad_id, current_net="N0"
    )[0])
    barrel_node = pf.lattice.node_idx(
        portal.x_idx, portal.y_idx, portal.pad_layer
    )
    pf.node_owner[barrel_node] = pf._get_net_id("FOREIGN")
    pf.path_node_use[barrel_node] = 1
    occupied_costs = dict(pf._get_pad_portal_seeds(
        pad_id, current_net="N0"
    )[0])
    assert occupied_costs[entry_node] - baseline_costs[entry_node] == (
        pytest.approx(
            config.owner_penalty_base
            + config.path_node_penalty_base
        )
    )
    pf.node_owner[barrel_node] = -1
    pf.path_node_use[barrel_node] = 0
    layer_one_node = pf.lattice.node_idx(
        portal.x_idx, portal.y_idx, 1
    )
    layer_three_node = pf.lattice.node_idx(
        portal.x_idx, portal.y_idx, 3
    )
    history_key = (
        pad_id, portal.x_idx, portal.y_idx, 1
    )
    pf._portal_barrel_history[history_key] = 1.0
    depth_costs = dict(pf._get_pad_portal_seeds(
        pad_id, current_net="N0"
    )[0])
    assert (
        depth_costs[layer_one_node] - baseline_costs[layer_one_node]
        == pytest.approx(config.portal_barrel_history_penalty)
    )
    assert depth_costs[layer_three_node] == pytest.approx(
        baseline_costs[layer_three_node]
    )
    del pf._portal_barrel_history[history_key]
    entry_layer = portal_seeds[0][0]
    entry_layer = pf.lattice.idx_to_coord(entry_layer)[2]
    geometry = pf.escape_planner._emit_portal_escape_geometry(
        "N0",
        pad_id,
        portal,
        entry_layer,
        include_via=True,
    )
    tracks = [item for item in geometry if "x1" in item]
    vias = [item for item in geometry if "x" in item]

    assert len(tracks) == 2
    assert tracks[0]["layer"] == "F.Cu"
    assert tracks[0]["x1"] == pytest.approx(tracks[0]["x2"])
    assert tracks[1]["layer"] == pf.config.layer_names[entry_layer]
    assert tracks[1]["y1"] == pytest.approx(tracks[1]["y2"])
    assert len(vias) == 1
    assert vias[0]["x"] == pytest.approx(portal.pad_x)
    assert vias[0]["dynamic_entry"]

    pf.route_multiple_nets([board.nets[0]])
    pf.emit_geometry(board)
    path = pf.net_paths["N0"]
    assert pf.lattice.idx_to_coord(path[0])[2] == 0
    assert pf.lattice.idx_to_coord(path[-1])[2] == 0
    inner_path = pf._path_without_dynamic_escape_chains("N0", path)
    assert pf.lattice.idx_to_coord(inner_path[0])[2] in {1, 3}
    assert pf.lattice.idx_to_coord(inner_path[-1])[2] in {1, 3}
    assert sum(
        via.get("dynamic_entry", False)
        for via in pf.get_geometry_payload().vias
        if via["net"] == "N0"
    ) == 2


def test_full_component_geometry_recovers_partial_dynamic_columns():
    board = _make_columnar_connector_board(columns=10)
    selected_nets = board.nets[:32] + [board.nets[32], board.nets[36]]
    config = PathFinderConfig()
    config.track_width = 0.1016
    config.clearance = 0.1016
    config.via_diameter = 0.4
    config.via_drill = 0.15
    pf = UnifiedPathFinder(config=config, use_gpu=False)

    pf.initialize_graph(board)
    pf.precompute_all_pad_escapes(board, selected_nets)

    regular_pad_ids = {
        pf.escape_planner._pad_key(pad)
        for net in board.nets[:32]
        for pad in net.pads
    }
    partial_pad_ids = {
        pf.escape_planner._pad_key(pad)
        for net in selected_nets[32:]
        for pad in net.pads
    }
    assert all(
        pf.portals[pad_id].dynamic_entry
        for pad_id in regular_pad_ids
    )
    assert all(
        pf.portals[pad_id].dynamic_entry
        for pad_id in partial_pad_ids
    )


def test_dynamic_candidates_include_short_non_via_in_pad_escape():
    board = _make_columnar_connector_board()
    config = PathFinderConfig()
    config.track_width = 0.1016
    config.clearance = 0.1016
    config.via_diameter = 0.3024
    config.via_drill = 0.15
    pf = UnifiedPathFinder(config=config, use_gpu=False)

    pf.initialize_graph(board)
    pf.precompute_all_pad_escapes(board)
    pf._parse_requests(board.nets)
    pad_id = pf.net_pad_ids["N1"][0]
    short = next(
        portal
        for portal in pf.portal_candidates[pad_id]
        if portal.delta_steps == 2
    )
    via_x, via_y = pf.escape_planner._portal_world(short)
    dx = max(
        0.0,
        abs(via_x - short.pad_x) - 0.5 * 0.2,
    )
    dy = max(
        0.0,
        abs(via_y - short.pad_y) - 0.5 * 1.15,
    )

    assert (dx * dx + dy * dy) ** 0.5 >= (
        0.5 * config.via_diameter
    )


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


def test_cleanup_prices_live_escape_conflicts_above_reservations(routed):
    pf, _, _, _ = routed
    old_freeze = getattr(pf, "_freeze_selected_portals", False)
    try:
        pf._freeze_selected_portals = False
        ordinary = pf._escape_candidate_congestion_penalty(1, 1)
        assert ordinary == 2 * pf.config.escape_reservation_penalty

        pf._freeze_selected_portals = True
        cleanup = pf._escape_candidate_congestion_penalty(1, 1)
        assert cleanup == (
            pf.config.portal_cleanup_escape_penalty
            + pf.config.escape_reservation_penalty
        )
    finally:
        pf._freeze_selected_portals = old_freeze


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
    path = pf._path_without_dynamic_escape_chains(
        "TEST_NET", pf.net_paths["TEST_NET"]
    )
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
    assert ("OTHER_NET", "TEST_NET") in pf._exact_barrel_pairs

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


def test_explicit_portal_via_detects_nearby_graph_track(routed):
    """Off-grid terminal copper must participate in convergence."""
    from dataclasses import replace

    pf, _, _, _ = routed
    victim = "TEST_NET"
    edge = next(
        edge
        for edge in pf._net_to_edges[victim]
        if (
            pf.lattice.idx_to_coord(int(pf._edge_src[edge]))[2]
            == pf.lattice.idx_to_coord(int(pf.solver.indices[edge]))[2]
        )
    )
    source = int(pf._edge_src[edge])
    target = int(pf.solver.indices[edge])
    x0, y0, layer = pf.lattice.idx_to_coord(source)
    x1, y1, _ = pf.lattice.idx_to_coord(target)
    start = pf.lattice.geom.lattice_to_world(x0, y0)
    end = pf.lattice.geom.lattice_to_world(x1, y1)
    base = pf.net_selected_portals[victim][0]
    portal = replace(
        base,
        x_idx=x0,
        y_idx=y0,
        via_x=0.5 * (start[0] + end[0]),
        via_y=0.5 * (start[1] + end[1]),
        dynamic_entry=True,
    )
    owner = "PORTAL_OWNER"
    old_selected = pf.net_selected_portals.get(owner)
    old_layers = pf.net_portal_layers.get(owner)
    old_pad_ids = pf.net_pad_ids.get(owner)
    old_path = pf.net_paths.get(owner)
    try:
        pf.net_selected_portals[owner] = (portal,)
        pf.net_portal_layers[owner] = (layer,)
        pf.net_pad_ids[owner] = ("PORTAL_OWNER_PAD",)
        pf.net_paths[owner] = [source, target]

        pairs, owners, victims, keys, nodes = (
            pf._detect_portal_grid_conflicts()
        )

        assert any(
            pair[0][0] == owner
            and pair[1] == victim
            and pair[2] == "track"
            for pair in pairs
        )
        assert owner in owners
        assert victim in victims
        assert ("PORTAL_OWNER_PAD", x0, y0, layer) in keys
        assert {source, target} <= nodes
    finally:
        if old_selected is None:
            pf.net_selected_portals.pop(owner, None)
        else:
            pf.net_selected_portals[owner] = old_selected
        if old_layers is None:
            pf.net_portal_layers.pop(owner, None)
        else:
            pf.net_portal_layers[owner] = old_layers
        if old_pad_ids is None:
            pf.net_pad_ids.pop(owner, None)
        else:
            pf.net_pad_ids[owner] = old_pad_ids
        if old_path is None:
            pf.net_paths.pop(owner, None)
        else:
            pf.net_paths[owner] = old_path


def test_committed_portal_clearance_is_a_live_foreign_cost(routed):
    pf, _, _, _ = routed
    net_name = "TEST_NET"
    portal = pf.net_selected_portals[net_name][0]
    entry_layer = pf.net_portal_layers[net_name][0]
    nodes = pf._portal_clearance_nodes(portal, entry_layer)
    net_numeric_id = pf._get_net_id(net_name)

    assert nodes.size
    assert np.all(
        pf.portal_clearance_owner[nodes] == net_numeric_id
    )

    old_pres_fac = getattr(pf, "_pres_fac_now", 1.0)
    pf._pres_fac_now = 2.0
    penalty = pf._build_owner_penalty(None, "OTHER_NET")
    expected = pf.config.owner_penalty_base * pf._pres_fac_now
    assert np.any(penalty[nodes] >= expected)
    pf._pres_fac_now = old_pres_fac


def test_portal_cleanup_freezes_position_and_entry_depth(routed):
    pf, _, _, _ = routed
    net_name = "TEST_NET"
    pad_id = pf.net_pad_ids[net_name][0]
    selected = pf.net_selected_portals[net_name][0]
    selected_layer = pf.net_portal_layers[net_name][0]
    old_freeze = getattr(pf, "_freeze_selected_portals", False)
    old_movable = getattr(
        pf, "_portal_cleanup_movable_nets", set()
    )
    try:
        pf._freeze_selected_portals = True
        pf._portal_cleanup_movable_nets = set()
        seeds, portals = pf._get_pad_portal_seeds(
            pad_id, current_net=net_name
        )
        assert len(seeds) == 1
        node, _ = seeds[0]
        assert pf.lattice.idx_to_coord(node)[2] == selected_layer
        assert portals[node] is selected

        pf._portal_cleanup_movable_nets = {net_name}
        movable_seeds, _ = pf._get_pad_portal_seeds(
            pad_id, current_net=net_name
        )
        assert len(movable_seeds) > 1
    finally:
        pf._freeze_selected_portals = old_freeze
        pf._portal_cleanup_movable_nets = old_movable


def test_portal_cleanup_moves_nonconflicting_high_impact_peers():
    assert PathFinderRouter._should_run_one_sided_cleanup(
        physical_conflicts=5,
        overused_edges=3,
        already_active=False,
        edge_threshold=3,
    )
    assert not PathFinderRouter._should_run_one_sided_cleanup(
        physical_conflicts=5,
        overused_edges=3,
        already_active=True,
    )
    assert not PathFinderRouter._should_run_one_sided_cleanup(
        physical_conflicts=5,
        overused_edges=4,
        already_active=False,
        edge_threshold=3,
    )

    pairs = {
        (("A", "PAD-A", 1, 2), "B", "via"),
        (("B", "PAD-B", 3, 4), "C", "track"),
        (("Y", "PAD-Y", 5, 6), "X", "via"),
    }

    router = UnifiedPathFinder(
        config=PathFinderConfig(),
        use_gpu=False,
    )
    movable = router._portal_cleanup_movable_components(
        pairs,
        {
            (("C", "PAD-C"), ("D", "PAD-D")),
            (("Q", "PAD-Q"), ("P", "PAD-P")),
        },
        {
            ("D", "E"),
            ("R", "S"),
        },
    )

    assert movable == {"B", "D", "P", "R", "X"}

    all_pairs = {
        (identity[0], victim)
        for identity, victim, _kind in pairs
    }
    all_pairs.update(
        (first[0], second[0])
        for first, second in {
            (("C", "PAD-C"), ("D", "PAD-D")),
            (("Q", "PAD-Q"), ("P", "PAD-P")),
        }
    )
    all_pairs.update({("D", "E"), ("R", "S")})
    assert not any(
        first in movable and second in movable
        for first, second in all_pairs
    )

    next_movable = router._portal_cleanup_movable_components(
        pairs,
        {
            (("C", "PAD-C"), ("D", "PAD-D")),
            (("Q", "PAD-Q"), ("P", "PAD-P")),
        },
        {
            ("D", "E"),
            ("R", "S"),
        },
    )
    assert next_movable == {"A", "C", "E", "Q", "S", "Y"}
    assert not any(
        first in next_movable and second in next_movable
        for first, second in all_pairs
    )


def test_physical_hotset_is_bounded_and_severity_ranked():
    router = object.__new__(PathFinderRouter)
    router.config = PathFinderConfig()
    router.config.physical_hotset_cap = 3
    router._barrel_conflict_nets = {"A", "B", "C", "D", "E"}
    router._physical_conflict_scores = {
        "A": 2,
        "B": 9,
        "C": 4,
        "D": 9,
        "E": 1,
    }

    assert router._select_physical_hotset() == {"B", "C", "D"}


def test_spatial_via_overuse_counts_columns_and_segments():
    router = object.__new__(PathFinderRouter)
    router.via_col_use = np.array([[1, 5], [3, 2]])
    router.via_col_cap = np.full((2, 2), 3)
    router.via_seg_use = np.array([[[1, 4], [3, 2]]])
    router.via_seg_cap = np.full((1, 2, 2), 2)

    assert router._spatial_via_overuse_total() == 5


def test_portal_cleanup_prices_exact_foreign_edges(routed):
    pf, _, _, _ = routed
    net_name = "TEST_NET"
    portal = pf.net_selected_portals[net_name][0]
    entry_layer = pf.net_portal_layers[net_name][0]
    portal_edges = set(map(
        int,
        pf._portal_conflicting_graph_edges(
            portal, entry_layer
        ),
    ))
    assert portal_edges
    via_edges = [
        edge
        for edge in portal_edges
        if (
            pf.lattice.idx_to_coord(
                int(pf._edge_src[edge])
            )[2]
            != pf.lattice.idx_to_coord(
                int(pf.solver.indices[edge])
            )[2]
        )
    ]
    assert via_edges
    for edge in via_edges:
        source = int(pf._edge_src[edge])
        target = int(pf.solver.indices[edge])
        assert pf._edge_index_for_hop(target, source) in portal_edges

    pf._rebuild_portal_cleanup_edge_owners()
    foreign = set(map(
        int, pf._portal_cleanup_foreign_edges("OTHER_NET")
    ))
    own = set(map(
        int, pf._portal_cleanup_foreign_edges(net_name)
    ))

    assert portal_edges <= foreign
    assert portal_edges.isdisjoint(own)


def test_portal_cleanup_makes_foreign_barrels_prohibitive(routed):
    pf, _, _, _ = routed
    net_name = "TEST_NET"
    portal = pf.net_selected_portals[net_name][0]
    entry_layer = pf.net_portal_layers[net_name][0]
    nodes = pf._portal_clearance_nodes(portal, entry_layer)
    old_freeze = getattr(pf, "_freeze_selected_portals", False)
    try:
        pf._freeze_selected_portals = True
        penalty = pf._build_owner_penalty(None, "OTHER_NET")
        assert np.any(
            penalty[nodes]
            >= pf.config.portal_cleanup_node_penalty
        )
    finally:
        pf._freeze_selected_portals = old_freeze


def test_layer_depth_bias_is_monotonic_without_congestion():
    config = PathFinderConfig()
    config.layer_depth_bias = 0.1
    router = UnifiedPathFinder(config=config, use_gpu=False)
    accounting = type("Accounting", (), {
        "xp": np,
        "use_gpu": False,
        "present": np.zeros(4, dtype=np.float32),
        "present_ema": np.zeros(4, dtype=np.float32),
        "capacity": np.ones(4, dtype=np.float32),
    })()
    graph = type("Graph", (), {
        "edge_layer": np.arange(4, dtype=np.int32),
    })()

    bias = router._compute_layer_bias(
        accounting,
        graph,
        num_layers=4,
        alpha=0.0,
        max_boost=1.8,
    )

    assert bias == pytest.approx([1.0, 1.1, 1.2, 1.3])


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
    assert (
        pf.config.path_node_penalty_base
        == pf.config.owner_penalty_base
    )
    pf._pres_fac_now = old_pres_fac

    pf._clear_path_node_use(path)
    assert np.all(pf.path_node_use[path_nodes] == 0)
    pf._mark_path_node_use(path)
    assert np.all(pf.path_node_use[path_nodes] == 1)


def test_shared_path_nodes_are_physical_conflicts(routed):
    """Perpendicular guided tracks may not cross through one graph node."""
    pf, _, _, _ = routed
    original_paths = dict(pf.net_paths)
    try:
        shared = pf.lattice.node_idx(2, 2, 1)
        pf.net_paths.clear()
        pf.net_paths.update({
            "H_NET": [
                pf.lattice.node_idx(1, 2, 1),
                shared,
                pf.lattice.node_idx(3, 2, 1),
            ],
            "V_NET": [
                pf.lattice.node_idx(2, 1, 1),
                shared,
                pf.lattice.node_idx(2, 3, 1),
            ],
        })
        pf._rebuild_path_node_use()

        pairs, nodes, scores = pf._detect_path_node_conflicts()

        assert pairs == {("H_NET", "V_NET")}
        assert nodes == {shared}
        assert scores == {"H_NET": 1, "V_NET": 1}
    finally:
        pf.net_paths.clear()
        pf.net_paths.update(original_paths)
        pf._rebuild_path_node_use()


def test_path_nodes_are_capacity_one_negotiated_resources():
    router = object.__new__(PathFinderRouter)
    router.path_node_use = np.array([0, 1, 2, 4], dtype=np.int16)

    assert router._compute_path_node_overuse() == (4, 2)


def test_path_node_metrics_expose_layer_localized_congestion():
    router = object.__new__(PathFinderRouter)
    router.lattice = type("Lattice", (), {
        "x_steps": 2,
        "y_steps": 2,
        "layers": 2,
    })()
    router.path_node_use = np.array(
        [0, 1, 2, 4, 1, 1, 0, 0],
        dtype=np.int16,
    )

    assert router._path_node_layer_metrics() == [
        {
            "layer": 0,
            "capacity_nodes": 4,
            "occupied_nodes": 3,
            "conflict_nodes": 2,
            "excess_uses": 4,
            "max_use": 4,
        },
        {
            "layer": 1,
            "capacity_nodes": 4,
            "occupied_nodes": 2,
            "conflict_nodes": 0,
            "excess_uses": 0,
            "max_use": 1,
        },
    ]


def test_best_route_score_combines_edge_and_node_overuse():
    score = PathFinderRouter._negotiated_route_score

    edge_better_but_resource_worse = score(0, 90, 20, 0)
    combined_resource_better = score(0, 100, 0, 50)

    assert combined_resource_better < edge_better_but_resource_worse


def test_node_conflict_history_persists_once_per_iteration(routed):
    pf, _, _, _ = routed
    node = int(np.flatnonzero(
        (pf.node_owner == -1) & (pf.path_node_use == 0)
    )[0])
    old_value = pf.node_conflict_history[node]
    old_iteration = pf.iteration
    had_marker = hasattr(pf, "_node_history_iteration")
    old_marker = getattr(pf, "_node_history_iteration", None)
    had_nodes = hasattr(pf, "_node_history_nodes")
    old_nodes = getattr(pf, "_node_history_nodes", None)
    try:
        pf.iteration = old_iteration + 1000
        pf._accumulate_node_conflict_history([node, node])
        pf._accumulate_node_conflict_history([node])
        assert pf.node_conflict_history[node] == pytest.approx(
            old_value + pf.config.node_history_increment
        )

        pf.iteration += 1
        pf._accumulate_node_conflict_history([node])
        expected_history = (
            old_value + 2 * pf.config.node_history_increment
        )
        assert pf.node_conflict_history[node] == pytest.approx(
            expected_history
        )
        penalty = pf._build_owner_penalty(None, "HISTORY_TEST")
        assert penalty[node] == pytest.approx(
            expected_history * pf.config.node_history_penalty
        )
    finally:
        pf.node_conflict_history[node] = old_value
        pf.iteration = old_iteration
        if had_marker:
            pf._node_history_iteration = old_marker
        else:
            del pf._node_history_iteration
        if had_nodes:
            pf._node_history_nodes = old_nodes
        else:
            del pf._node_history_nodes


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


def test_physical_offenders_bypass_edge_hotset_cap(routed):
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    physical = {f"PHYSICAL_{index}" for index in range(150)}
    tasks = {
        net_id: (path[0], path[-1])
        for net_id in physical
    }
    edge = pf._net_to_edges["TEST_NET"][0]
    old_present = pf.accounting.present[edge]
    old_cap = pf.config.hotset_cap
    old_physical = getattr(pf, "_barrel_conflict_nets", set())
    old_clean = dict(getattr(pf, "_net_clean_iters", {}))
    old_reroute = dict(getattr(pf, "_last_reroute_iter", {}))
    old_hotset = set(getattr(pf, "_prev_hotset", set()))
    try:
        pf.accounting.present[edge] = (
            pf.accounting.capacity[edge] + 1
        )
        pf.config.hotset_cap = 1
        pf._barrel_conflict_nets = physical

        hotset = pf._build_hotset(tasks)

        assert physical <= hotset
    finally:
        pf.accounting.present[edge] = old_present
        pf.config.hotset_cap = old_cap
        pf._barrel_conflict_nets = old_physical
        pf._net_clean_iters = old_clean
        pf._last_reroute_iter = old_reroute
        pf._prev_hotset = old_hotset


def test_physical_offenders_wait_for_graph_cleanup(routed):
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    physical = {f"PHYSICAL_{index}" for index in range(10)}
    tasks = {
        net_id: (path[0], path[-1])
        for net_id in physical
    }
    edge = pf._net_to_edges["TEST_NET"][0]
    old_present = pf.accounting.present[edge]
    old_threshold = pf.config.portal_cleanup_edge_threshold
    old_physical = getattr(pf, "_barrel_conflict_nets", set())
    old_clean = dict(getattr(pf, "_net_clean_iters", {}))
    old_reroute = dict(getattr(pf, "_last_reroute_iter", {}))
    old_hotset = set(getattr(pf, "_prev_hotset", set()))
    old_paths = {
        net_id: pf.net_paths.get(net_id)
        for net_id in physical
    }
    try:
        pf.accounting.present[edge] = (
            pf.accounting.capacity[edge] + 4
        )
        pf.config.portal_cleanup_edge_threshold = 3
        pf._barrel_conflict_nets = physical
        for net_id in physical:
            pf.net_paths[net_id] = path

        hotset = pf._build_hotset(tasks)

        assert physical.isdisjoint(hotset)
    finally:
        pf.accounting.present[edge] = old_present
        pf.config.portal_cleanup_edge_threshold = old_threshold
        pf._barrel_conflict_nets = old_physical
        pf._net_clean_iters = old_clean
        pf._last_reroute_iter = old_reroute
        pf._prev_hotset = old_hotset
        for net_id, old_path in old_paths.items():
            if old_path is None:
                pf.net_paths.pop(net_id, None)
            else:
                pf.net_paths[net_id] = old_path


def test_physical_cleanup_pauses_when_graph_reopens():
    should_cleanup = PathFinderRouter._should_run_one_sided_cleanup

    assert should_cleanup(1, 3, False, 3)
    assert should_cleanup(1, 3, True, 3)
    assert not should_cleanup(1, 1, False, 3, overuse_total=4)
    assert not should_cleanup(1, 4, True, 3)
    assert not should_cleanup(0, 0, True, 3)


def test_history_hotset_cap_scales_with_live_overuse():
    cap = PathFinderRouter._history_hotset_cap

    assert cap(0) == 16
    assert cap(8) == 16
    assert cap(9) == 32
    assert cap(32) == 32
    assert cap(33) == 64
    assert cap(128) == 64
    assert cap(129) == 100
    assert cap(2_048) == 100
    assert cap(2_049) == 180
    assert cap(16_384) == 180
    assert cap(16_385) == 256
    assert cap(100_000) == 256


def test_hotset_exploration_shrinks_during_severe_congestion():
    fraction = PathFinderRouter._hotset_exploration_fraction

    assert fraction(128) == pytest.approx(0.40)
    assert fraction(2_048) == pytest.approx(0.40)
    assert fraction(2_049) == pytest.approx(0.25)
    assert fraction(16_384) == pytest.approx(0.25)
    assert fraction(16_385) == pytest.approx(0.15)


def test_rolling_progress_detects_tiny_new_minima():
    insufficient = PathFinderRouter._rolling_progress_insufficient

    slow, fraction = insufficient(
        [75_977, 75_900, 75_850, 75_800, 75_780, 75_733],
        window=5,
        minimum_fraction=0.025,
        minimum_overuse=16_384,
    )
    assert slow
    assert fraction == pytest.approx(244 / 75_977)

    slow, fraction = insufficient(
        [100_000, 98_000, 96_000, 94_000, 92_000, 90_000],
        window=5,
        minimum_fraction=0.025,
        minimum_overuse=16_384,
    )
    assert not slow
    assert fraction == pytest.approx(0.10)

    assert insufficient(
        [100, 99, 99, 99, 99, 99],
        window=5,
        minimum_fraction=0.025,
        minimum_overuse=16_384,
    ) == (False, None)

    slow, fraction = insufficient(
        [79_630, 79_662, 79_670, 78_909, 78_436, 78_133],
        window=5,
        minimum_fraction=0.025,
        minimum_overuse=16_384,
    )
    assert slow
    assert fraction == pytest.approx(1_497 / 79_630)


def test_rate_plateau_temporarily_expands_severe_hotset():
    router = object.__new__(PathFinderRouter)
    router.config = PathFinderConfig()
    router.iteration = 20
    router._hotset_rate_boost_until = 25

    assert router._effective_history_hotset_cap(75_000) == 512
    assert router._effective_history_hotset_cap(2_000) == 100

    router.iteration = 26
    assert router._effective_history_hotset_cap(75_000) == 256


def test_pressure_schedule_scales_with_bounded_reroute_work():
    scale = PathFinderRouter._pressure_work_scale

    assert scale(0) == pytest.approx(1.0)
    assert scale(50) == pytest.approx(1.0)
    assert scale(100) == pytest.approx(1.0)
    assert scale(180) == pytest.approx(1.8)
    assert scale(256) == pytest.approx(2.0)
    assert scale(512) == pytest.approx(2.0)
    assert scale(
        256,
        reference_hotset=128,
        maximum_scale=3.0,
    ) == pytest.approx(2.0)


def test_conflict_aware_hotset_covers_conflicts_before_exploration():
    import random

    ranked = ["A", "B", "C", "D", "E", "F", "G", "H"]
    conflict_pairs = {
        ("A", "B"),
        ("A", "C"),
        ("D", "E"),
    }

    selected = PathFinderRouter._select_conflict_aware_hotset(
        ranked,
        conflict_pairs,
        cap=4,
        exploration_fraction=0.25,
        rng=random.Random(42),
    )

    assert len(selected) == 4
    assert {"A", "D"}.issubset(selected)
    assert all(
        first in selected or second in selected
        for first, second in conflict_pairs
    )


def test_conflict_aware_hotset_keeps_edge_only_offenders_eligible():
    import random

    selected = PathFinderRouter._select_conflict_aware_hotset(
        ["A", "B", "EDGE_1", "EDGE_2"],
        {("A", "B")},
        cap=3,
        exploration_fraction=0.0,
        rng=random.Random(42),
    )

    assert selected == ["A", "EDGE_1", "EDGE_2"]


def test_conflict_aware_hotset_fills_budget_on_dense_component():
    import random

    ranked = ["A", "B", "C", "D", "EDGE"]
    conflict_pairs = {
        (first, second)
        for index, first in enumerate(ranked[:4])
        for second in ranked[index + 1:4]
    }

    selected = PathFinderRouter._select_conflict_aware_hotset(
        ranked,
        conflict_pairs,
        cap=5,
        exploration_fraction=0.20,
        rng=random.Random(42),
    )

    assert len(selected) == 5
    assert set(selected) == set(ranked)


def test_initial_net_order_is_reproducible_across_global_rng_state():
    """The greedy pass must be a controlled experiment, not process RNG."""
    import random

    router = object.__new__(PathFinderRouter)
    router.iteration = 1
    router.lattice = type("Lattice", (), {
        "idx_to_coord": lambda self, node: (node, 0, 0),
    })()
    router.accounting = type("Accounting", (), {
        "use_gpu": False,
        "present": np.zeros(1, dtype=np.float32),
        "capacity": np.ones(1, dtype=np.float32),
    })()
    router._net_to_edges = {}
    router._barrel_owner_nets = set()
    router._barrel_victim_nets = set()
    tasks = {
        f"N{index:02d}": (index, 40 - index)
        for index in range(20)
    }

    random.seed(1)
    first = router._order_nets_by_difficulty(tasks)
    random.seed(999)
    second = router._order_nets_by_difficulty(tasks)

    assert first == second


def test_historical_only_edges_do_not_make_clean_nets_offenders():
    """Retained history prices routes but must not define live rip-up."""
    router = object.__new__(PathFinderRouter)
    router.config = PathFinderConfig()
    router.iteration = 2
    router.net_paths = {
        "LIVE": [0, 1],
        "STALE": [2, 3],
    }
    router._net_to_edges = {
        "LIVE": [0],
        "STALE": [1],
    }
    router._edge_to_nets = {
        0: {"LIVE"},
        1: {"STALE"},
    }
    router.accounting = type("Accounting", (), {
        "use_gpu": False,
        "present": np.array([2.0, 1.0], dtype=np.float32),
        "capacity": np.ones(2, dtype=np.float32),
        "history": np.array([0.0, 100.0], dtype=np.float32),
        "compute_overuse": lambda self, router_instance=None: (1, 1),
    })()

    hotset = router._build_hotset({
        "LIVE": (0, 1),
        "STALE": (2, 3),
    })

    assert hotset == {"LIVE"}


def test_path_node_offenders_negotiate_while_edges_are_overused():
    """Capacity-one nodes must not wait for edge cleanup to enter hotsets."""
    router = object.__new__(PathFinderRouter)
    router.config = PathFinderConfig()
    router.iteration = 2
    router.net_paths = {
        "EDGE": [0, 1],
        "NODE_ONLY": [2, 3],
    }
    router._net_to_edges = {
        "EDGE": [0],
        "NODE_ONLY": [1],
    }
    router._edge_to_nets = {
        0: {"EDGE"},
        1: {"NODE_ONLY"},
    }
    router._path_node_conflict_scores = {"NODE_ONLY": 7}
    router.accounting = type("Accounting", (), {
        "use_gpu": False,
        "present": np.array([2.0, 1.0], dtype=np.float32),
        "capacity": np.ones(2, dtype=np.float32),
        "compute_overuse": lambda self, router_instance=None: (1, 1),
    })()

    hotset = router._build_hotset({
        "EDGE": (0, 1),
        "NODE_ONLY": (2, 3),
    })

    assert hotset == {"EDGE", "NODE_ONLY"}


def test_stagnation_rip_waits_for_spatial_via_tail():
    should_rip = PathFinderRouter._should_rip_for_stagnation

    assert should_rip(0)
    assert should_rip(8)
    assert not should_rip(9)
    assert not should_rip(359)
    assert should_rip(0, tail_threshold=-1)
    assert not should_rip(
        0,
        physical_cleanup_started=True,
    )
    assert not should_rip(
        8,
        physical_cleanup_started=True,
    )


def test_large_physical_drop_marks_cleanup_stage():
    detected = PathFinderRouter._physical_cleanup_drop_detected

    assert not detected(None, 75)
    assert not detected(0, 0)
    assert not detected(100, 76)
    assert detected(100, 75)
    assert detected(100, 25)
    assert not detected(100, 101)


def test_physical_hotset_limit_scales_with_remaining_conflicts():
    limit = PathFinderRouter._physical_hotset_limit

    assert limit(0) == 64
    assert limit(3200) == 64
    assert limit(3201) == 65
    assert limit(26844) == 537
    assert limit(49566) == 992
    assert limit(52696) == 1024
    assert limit(100000) == 1024
    assert limit(100000, max_cap=512) == 512
    assert limit(0, max_cap=32, min_cap=64) == 32


def test_physical_offenders_wait_for_via_pool_cleanup(routed):
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    physical = {f"PHYSICAL_{index}" for index in range(4)}
    tasks = {
        net_id: (path[0], path[-1])
        for net_id in physical
    }
    old_physical = getattr(pf, "_barrel_conflict_nets", set())
    old_paths = {
        net_id: pf.net_paths.get(net_id)
        for net_id in physical
    }
    old_use = int(pf.via_col_use[0, 0])
    old_capacity = int(pf.via_col_cap[0, 0])
    try:
        pf._barrel_conflict_nets = physical
        for net_id in physical:
            pf.net_paths[net_id] = [path[0], path[0]]
        pf.via_col_use[0, 0] = old_capacity + 4

        hotset = pf._build_hotset(tasks)

        assert physical.isdisjoint(hotset)
    finally:
        pf.via_col_use[0, 0] = old_use
        pf._barrel_conflict_nets = old_physical
        for net_id, old_path in old_paths.items():
            if old_path is None:
                pf.net_paths.pop(net_id, None)
            else:
                pf.net_paths[net_id] = old_path


def test_physical_offenders_stay_hot_when_edges_are_clean(routed):
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    tasks = {
        "PHYSICAL_OWNER": (path[0], path[-1]),
        "PHYSICAL_VICTIM": (path[0], path[-1]),
    }
    old_physical = getattr(pf, "_barrel_conflict_nets", set())
    try:
        pf._barrel_conflict_nets = set(tasks)
        overuse, _ = pf.accounting.compute_overuse(pf)
        assert overuse == 0
        assert set(tasks) <= pf._build_hotset(tasks)
    finally:
        pf._barrel_conflict_nets = old_physical


def test_unrouted_nets_bypass_hotset_cap_and_cooldown(routed):
    pf, _, _, _ = routed
    path = pf.net_paths["TEST_NET"]
    tasks = {
        f"UNROUTED_{index}": (path[0], path[-1])
        for index in range(150)
    }
    edge = pf._net_to_edges["TEST_NET"][0]
    old_present = pf.accounting.present[edge]
    old_cap = pf.config.hotset_cap
    old_physical = getattr(pf, "_barrel_conflict_nets", set())
    old_clean = dict(getattr(pf, "_net_clean_iters", {}))
    old_reroute = dict(getattr(pf, "_last_reroute_iter", {}))
    old_hotset = set(getattr(pf, "_prev_hotset", set()))
    try:
        pf.accounting.present[edge] = (
            pf.accounting.capacity[edge] + 1
        )
        pf.config.hotset_cap = 1
        pf._barrel_conflict_nets = set()
        pf._last_reroute_iter = {
            net_id: pf.iteration
            for net_id in tasks
        }

        assert set(tasks) <= pf._build_hotset(tasks)
    finally:
        pf.accounting.present[edge] = old_present
        pf.config.hotset_cap = old_cap
        pf._barrel_conflict_nets = old_physical
        pf._net_clean_iters = old_clean
        pf._last_reroute_iter = old_reroute
        pf._prev_hotset = old_hotset


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


def test_via_pool_reroutes_only_one_stable_peer(routed):
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
    old_capacity = int(pf.via_col_cap[x_idx, y_idx])
    old_peer_path = pf.net_paths.get("PEER_NET")
    old_keepers = dict(getattr(pf, "_via_pool_keepers", {}))
    old_member_state = dict(getattr(
        pf, "_via_pool_member_state", {}
    ))
    old_stagnation = dict(getattr(
        pf, "_via_pool_keeper_stagnation", {}
    ))
    had_rotation_threshold = hasattr(
        pf.config, "via_keeper_rotation_overuse_threshold"
    )
    old_rotation_threshold = getattr(
        pf.config, "via_keeper_rotation_overuse_threshold", 8
    )
    try:
        pf.net_paths["PEER_NET"] = list(path)
        pf.via_col_cap[x_idx, y_idx] = 1
        pf._rebuild_via_usage_from_committed()

        first = pf._find_via_pool_offenders()
        second = pf._find_via_pool_offenders()
        rotated = pf._find_via_pool_offenders()

        peers = {"TEST_NET", "PEER_NET"}
        assert len(first & peers) == 1
        assert second & peers == first & peers
        assert rotated & peers == peers - (first & peers)

        pf._via_pool_keepers = {}
        pf._via_pool_member_state = {}
        pf._via_pool_keeper_stagnation = {}
        pf.config.via_keeper_rotation_overuse_threshold = 0
        broad_first = pf._find_via_pool_offenders()
        pf._find_via_pool_offenders()
        broad_third = pf._find_via_pool_offenders()
        assert broad_third & peers == broad_first & peers
    finally:
        pf.via_col_cap[x_idx, y_idx] = old_capacity
        if old_peer_path is None:
            pf.net_paths.pop("PEER_NET", None)
        else:
            pf.net_paths["PEER_NET"] = old_peer_path
        pf._via_pool_keepers = old_keepers
        pf._via_pool_member_state = old_member_state
        pf._via_pool_keeper_stagnation = old_stagnation
        if had_rotation_threshold:
            pf.config.via_keeper_rotation_overuse_threshold = (
                old_rotation_threshold
            )
        else:
            del pf.config.via_keeper_rotation_overuse_threshold
        pf._rebuild_via_usage_from_committed()


def test_via_keeper_rotation_tail_scales_with_route_size():
    scale = PathFinderRouter._scaled_via_keeper_rotation_threshold

    assert scale(8, 80) == 8
    assert scale(8, 1024) == 8
    assert scale(8, 1025) == 16
    assert scale(8, 2048) == 16
    assert scale(8, 8192) == 64
    assert scale(0, 8192) == 0


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
