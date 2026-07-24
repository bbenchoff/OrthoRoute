"""End-to-end CPU-only engine smoke test (pytest twin of --test-via).

Routes one two-pad net on a synthetic 4-layer board through the full live
call sequence. This is the KiCad-free gate every backend change must pass.
"""

import numpy as np
import pytest

from orthoroute.algorithms.manhattan.unified_pathfinder import (
    PathFinderConfig,
    PathFinderRouter,
    UnifiedPathFinder,
)

from conftest import make_two_pad_board


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
