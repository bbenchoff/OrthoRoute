"""JLCPCB k+N+k core-span stack: topology, barrel extent, and export."""

import hashlib
import json

import numpy as np
import pytest

from orthoroute.algorithms.manhattan.hdi_stack import (
    MANUFACTURING_PROFILES,
    HDIStack,
    jlcpcb_adjacent_stack,
    jlcpcb_hdi_stack,
    pcbway_mechanical_stack,
    stack_for_profile,
    stack_vendor,
    vendor_project_rules,
)
from orthoroute.algorithms.manhattan.unified_pathfinder import (
    Lattice3D,
    PathFinderConfig,
    UnifiedPathFinder,
)
from orthoroute.infrastructure.kicad.text_board_exporter import (
    export_geometry_to_board,
)


def _layer_names(count):
    return (
        ["F.Cu"]
        + [f"In{index}.Cu" for index in range(1, count - 1)]
        + ["B.Cu"]
    )


def test_jlcpcb_28_layer_stack_is_3_22_3():
    stack = jlcpcb_hdi_stack(28)

    assert stack.name == "jlcpcb_hdi_3+22+3"
    assert stack.notation == "3+22+3"
    assert stack.vendor == "jlcpcb"
    assert stack.core_span == (3, 24)
    assert stack.core_pair == (3, 24)
    assert stack.core_hub_layer == 3
    laser = {(0, 1), (1, 2), (2, 3), (24, 25), (25, 26), (26, 27)}
    hub = {(3, z) for z in range(4, 25)}
    assert stack.allowed_via_spans == frozenset(laser | hub)
    assert stack.process_for_span(0, 1).name == "laser_microvia"
    assert stack.process_for_span(0, 1).drill_mm == pytest.approx(0.1)
    assert stack.process_for_span(0, 1).diameter_mm == pytest.approx(0.25)
    assert stack.process_for_span(3, 10).name == "mechanical_buried_core"
    assert stack.process_for_span(3, 10).drill_mm == pytest.approx(0.15)
    assert stack.process_for_span(3, 10).diameter_mm == pytest.approx(0.30)


@pytest.mark.parametrize(
    ("build_ups", "core_span", "notation"),
    [(3, (3, 24), "3+22+3"), (2, (2, 25), "2+24+2"), (1, (1, 26), "1+26+1")],
)
def test_jlcpcb_orders(build_ups, core_span, notation):
    stack = jlcpcb_hdi_stack(28, build_up_steps=build_ups)

    assert stack.core_span == core_span
    assert stack.notation == notation
    assert stack.build_up_steps == build_ups


def test_pcbway_six_build_up_core_span_stack():
    from orthoroute.algorithms.manhattan.hdi_stack import pcbway_hdi_stack

    stack = pcbway_hdi_stack(32)
    assert stack.name == "pcbway_hdi_6+20+6"
    assert stack.vendor == "pcbway"
    assert stack.core_span == (6, 25)
    assert stack.core_hub_layer == 6
    assert stack.process_for_span(0, 1).diameter_mm == pytest.approx(0.2524)
    assert stack.process_for_span(6, 10).drill_mm == pytest.approx(0.2)
    assert stack.physical_spans(0, 10) == (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 25)
    )
    assert stack.hole_edge_spacing_mm == pytest.approx(0.2794)
    with pytest.raises(ValueError, match="1..6"):
        pcbway_hdi_stack(32, build_up_steps=7)
    assert stack_for_profile("pcbway_6n6", 28).notation == "6+16+6"
    assert stack_for_profile("pcbway_4n4", 28).notation == "4+20+4"


def test_jlcpcb_layer_ceiling_and_minimum():
    with pytest.raises(ValueError, match="28-layer ceiling"):
        jlcpcb_hdi_stack(30)
    with pytest.raises(ValueError, match="at least 8"):
        jlcpcb_hdi_stack(6)
    with pytest.raises(ValueError, match="1..3 sequential build-ups"):
        jlcpcb_hdi_stack(28, build_up_steps=4)


def test_physical_spans_collapse_core_hops_into_one_hole():
    stack = jlcpcb_hdi_stack(28)

    assert stack.physical_spans(0, 10) == ((0, 1), (1, 2), (2, 3), (3, 24))
    assert stack.physical_spans(10, 15) == ((3, 24),)
    assert stack.physical_spans(3, 4) == ((3, 24),)
    assert stack.physical_spans(3, 25) == ((3, 24), (24, 25))
    assert stack.physical_spans(24, 26) == ((24, 25), (25, 26))
    assert stack.physical_spans(2, 3) == ((2, 3),)
    assert stack.physical_spans(7, 7) == ()

    assert stack.expand_span(15, 10) == ((24, 3),)
    assert stack.expand_span(0, 5) == ((0, 1), (1, 2), (2, 3), (3, 24))
    assert stack.expand_span(5, 0) == ((24, 3), (3, 2), (2, 1), (1, 0))

    assert stack.physical_extent(10, 15) == (3, 24)
    assert stack.physical_extent(0, 10) == (0, 24)
    assert stack.physical_extent(0, 1) == (0, 1)
    assert stack.physical_extent(24, 26) == (24, 26)
    assert stack.physical_extent(5, 5) == (5, 5)


def test_is_physical_span_accepts_only_drilled_holes():
    stack = jlcpcb_hdi_stack(28)

    assert stack.is_physical_span(0, 1)
    assert stack.is_physical_span(26, 27)
    assert stack.is_physical_span(3, 24)
    assert stack.is_physical_span(24, 3)
    assert not stack.is_physical_span(3, 10)   # hub hop, not a hole
    assert not stack.is_physical_span(10, 15)
    assert not stack.is_physical_span(0, 10)
    assert not stack.is_physical_span(5, 5)
    assert not stack.is_physical_span(0, 99)

    adjacent = pcbway_mechanical_stack(28)
    assert adjacent.is_physical_span(4, 5)
    assert not adjacent.is_physical_span(4, 6)


def test_process_for_non_graph_hop_reports_realising_hole():
    stack = jlcpcb_hdi_stack(28)

    assert stack.process_for_span(10, 15).name == "mechanical_buried_core"
    assert stack.process_for_span(0, 10).name == "mechanical_buried_core"
    assert stack.process_for_span(1, 2).name == "laser_microvia"
    # Adjacent-span stacks keep rejecting non-adjacent hops.
    with pytest.raises(ValueError, match="not legal"):
        pcbway_mechanical_stack(28).process_for_span(0, 4)


def test_adjacent_stacks_keep_identity_extents():
    stack = pcbway_mechanical_stack(32)

    assert stack.core_span is None
    assert stack.physical_extent(4, 1) == (1, 4)
    assert stack.expand_span(4, 1) == ((4, 3), (3, 2), (2, 1))
    assert set(stack.via_cost_scale_by_span(4.0).values()) == {4.0}


def test_core_hub_edges_cost_a_share_of_one_hole():
    stack = jlcpcb_hdi_stack(28, core_via_cost_scale=2.5)
    scales = stack.via_cost_scale_by_span(4.0)

    assert scales[(0, 1)] == pytest.approx(4.0)
    assert scales[(26, 27)] == pytest.approx(4.0)
    assert scales[(3, 10)] == pytest.approx(10.0)
    assert scales[(3, 24)] == pytest.approx(10.0)


def test_core_via_sites_use_every_lattice_point_at_0p4_pitch():
    stack = jlcpcb_hdi_stack(28)
    spacing = stack.center_spacing_by_span()

    assert spacing[(0, 1)] == pytest.approx(0.35)
    assert spacing[(3, 10)] == pytest.approx(0.40)
    assert Lattice3D._via_site_policy(spacing[(3, 10)], 0.4) == ("all", 1)
    # PCBWay 0.4294 mm still forces the checkerboard.
    pcbway = pcbway_mechanical_stack(28).center_spacing_by_span()
    assert Lattice3D._via_site_policy(pcbway[(3, 4)], 0.4) == (
        "checkerboard", 1
    )


def _has_edge(graph, source, target):
    start = int(graph.indptr[source])
    end = int(graph.indptr[source + 1])
    return bool(np.any(graph.indices[start:end] == target))


def _edge_cost(graph, source, target):
    start = int(graph.indptr[source])
    end = int(graph.indptr[source + 1])
    offsets = np.flatnonzero(graph.indices[start:end] == target)
    assert offsets.size == 1
    return float(graph.base_costs[start + int(offsets[0])])


def test_graph_builds_hub_edges_on_odd_parity_sites():
    stack = jlcpcb_hdi_stack(8)  # 3+2+3, core (3, 4), hub pair (3, 4)
    lattice = Lattice3D((0.0, 0.0, 0.8, 0.8), 0.4, layers=8)
    graph = lattice.build_graph(
        via_cost=0.7,
        allowed_via_spans=set(stack.allowed_via_spans),
        min_via_center_spacing=0.4,
        via_pair_center_spacing=stack.center_spacing_by_span(),
        via_pair_cost_scale=stack.via_cost_scale_by_span(4.0),
    )
    x, y = 1, 0  # odd parity: excluded by a checkerboard policy
    assert _has_edge(
        graph, lattice.node_idx(x, y, 3), lattice.node_idx(x, y, 4)
    )
    assert _has_edge(
        graph, lattice.node_idx(x, y, 0), lattice.node_idx(x, y, 1)
    )
    # Hub edges carry the core share; laser hops the adjacent scale.
    assert _edge_cost(
        graph, lattice.node_idx(x, y, 3), lattice.node_idx(x, y, 4)
    ) == pytest.approx(0.7 * 4.0 * 2.0)
    assert _edge_cost(
        graph, lattice.node_idx(x, y, 0), lattice.node_idx(x, y, 1)
    ) == pytest.approx(0.7 * 4.0)


def test_core_site_stride_pins_hub_edges_to_a_via_grid():
    stack = jlcpcb_hdi_stack(8, core_site_stride=2)  # 3+2+3, hub pair (3, 4)
    assert stack.site_policy_by_span() == {(3, 4): ("stride", 2)}
    assert jlcpcb_hdi_stack(8).site_policy_by_span() == {}

    lattice = Lattice3D((0.0, 0.0, 0.8, 0.8), 0.4, layers=8)
    graph = lattice.build_graph(
        via_cost=0.7,
        allowed_via_spans=set(stack.allowed_via_spans),
        min_via_center_spacing=0.4,
        via_pair_center_spacing=stack.center_spacing_by_span(),
        via_pair_cost_scale=stack.via_cost_scale_by_span(4.0),
        via_pair_site_policy=stack.site_policy_by_span(),
    )
    hub = lambda x, y: (lattice.node_idx(x, y, 3), lattice.node_idx(x, y, 4))
    laser = lambda x, y: (lattice.node_idx(x, y, 0), lattice.node_idx(x, y, 1))
    assert _has_edge(graph, *hub(0, 0))
    assert _has_edge(graph, *hub(2, 2))
    assert not _has_edge(graph, *hub(1, 0))
    assert not _has_edge(graph, *hub(0, 1))
    assert not _has_edge(graph, *hub(1, 1))
    # Laser microvias stay available on every site.
    for x in range(3):
        for y in range(3):
            assert _has_edge(graph, *laser(x, y))
    # The override only ever tightens: stride 2 beats the checkerboard that
    # PCBWay spacing would demand, and never loosens it.
    assert Lattice3D._stricter_site_policy(
        ("checkerboard", 1), ("stride", 2)
    ) == ("stride", 2)
    assert Lattice3D._stricter_site_policy(
        ("stride", 2), ("all", 1)
    ) == ("stride", 2)


def test_portal_seeds_stop_at_hub_when_core_grid_is_active():
    from types import SimpleNamespace

    layers = 28
    grid = _router(jlcpcb_hdi_stack(layers, core_site_stride=2), layers)
    free = _router(jlcpcb_hdi_stack(layers), layers)
    portal = SimpleNamespace(x_idx=0, y_idx=0, pad_layer=0, dynamic_entry=False)

    def entry_layers(router):
        return sorted(
            router.lattice.idx_to_coord(node)[2]
            for node, _ in router._get_portal_seeds(portal)
        )

    assert entry_layers(grid) == [1, 2, 3]
    assert entry_layers(free) == list(range(1, layers - 1))

    # Planned escapes drop into the core at the portal, priced as one
    # laser column plus one core hole regardless of the entry depth.
    planned = _router(jlcpcb_hdi_stack(layers, core_site_stride=2), layers)
    planned.config.planned_core_escapes = True
    seeds = {
        planned.lattice.idx_to_coord(node)[2]: cost
        for node, cost in planned._get_portal_seeds(portal)
    }
    assert sorted(seeds) == list(range(1, layers - 1))
    unit = 0.7 * 0.15
    assert seeds[1] == pytest.approx(unit * 1)
    assert seeds[3] == pytest.approx(unit * 3)
    assert seeds[4] == pytest.approx(unit * 4)     # three lasers + core hole
    assert seeds[10] == pytest.approx(unit * 4)
    assert seeds[24] == pytest.approx(unit * 4)
    assert seeds[25] == pytest.approx(unit * 5)    # plus one bottom laser


def _router(stack, layers):
    config = PathFinderConfig()
    config.layer_count = layers
    config.layer_names = _layer_names(layers)
    config.hdi_stack = stack
    router = UnifiedPathFinder(config=config, use_gpu=False)
    router._build_via_extent_tables(stack)
    router.lattice = Lattice3D((0.0, 0.0, 0.4, 0.4), 0.4, layers=layers)
    return router


def test_router_barrel_extent_covers_whole_core():
    router = _router(jlcpcb_hdi_stack(28), 28)

    assert router._via_physical_extent(10, 15) == (3, 24)
    assert router._via_physical_extent(15, 10) == (3, 24)
    assert router._via_physical_extent(0, 10) == (0, 24)
    assert router._via_physical_extent(0, 1) == (0, 1)
    assert router._via_physical_extent(24, 26) == (24, 26)

    u = router.lattice.node_idx(0, 0, 10)
    v = router.lattice.node_idx(0, 0, 15)
    nodes = router._via_nodes_for_hop(u, v)
    assert len(nodes) == 22
    assert [router.lattice.idx_to_coord(n)[2] for n in nodes] == list(
        range(3, 25)
    )
    # Laser hops still occupy only their own two layers.
    u = router.lattice.node_idx(0, 0, 0)
    v = router.lattice.node_idx(0, 0, 1)
    assert [
        router.lattice.idx_to_coord(n)[2]
        for n in router._via_nodes_for_hop(u, v)
    ] == [0, 1]


def test_hole_track_pressure_charges_each_segment_with_its_layer():
    config = PathFinderConfig()
    config.hole_track_pressure_weight = 2.0
    router = UnifiedPathFinder(config=config, use_gpu=False)
    nx, ny, nz = 3, 2, 6
    router._Nx, router._Ny, router._Nz = nx, ny, nz
    router._segZ = nz - 2
    router.path_node_use = np.zeros(nx * ny * nz, dtype=np.int32)
    router.via_seg_use = np.zeros((nx, ny, nz - 2), dtype=np.int16)
    plane = nx * ny
    router.path_node_use[2 * plane + 0 * nx + 1] = 1   # (x=1, y=0, z=2)
    router.path_node_use[4 * plane + 1 * nx + 2] = 3   # (x=2, y=1, z=4)
    router.path_node_use[0 * plane + 0 * nx + 0] = 1   # F.Cu: not a segment
    router.path_node_use[5 * plane + 0 * nx + 0] = 1   # B.Cu: not a segment

    pressure = router._hole_track_pressure()

    assert pressure.shape == (nx, ny, nz - 2)
    assert pressure[1, 0, 1] == pytest.approx(2.0)   # layer 2 -> segment 1
    assert pressure[2, 1, 3] == pytest.approx(6.0)   # layer 4 -> segment 3
    assert float(pressure.sum()) == pytest.approx(8.0)

    config.hole_track_pressure_weight = 0.0
    assert router._hole_track_pressure() is None


def test_hub_hops_at_one_site_charge_the_core_hole_once():
    layers = 28
    router = _router(jlcpcb_hdi_stack(layers), layers)
    lattice = Lattice3D((0.0, 0.0, 0.8, 0.8), 0.4, layers=layers)
    router.lattice = lattice
    nx, ny = lattice.x_steps, lattice.y_steps
    router._Nx, router._Ny, router._Nz = nx, ny, layers
    router._segZ = layers - 2
    router.via_col_use = np.zeros((nx, ny), dtype=np.int16)
    router.via_seg_use = np.zeros((nx, ny, layers - 2), dtype=np.int16)
    router._via_keepouts_map = {}
    node = lattice.node_idx
    # Enter the core at (1,1) on layer 10, turn via the hub to layer 15,
    # leave laterally, then another hole at (2,1) back up to layer 2.
    path = [
        node(1, 1, 10), node(1, 1, 3), node(1, 1, 15),
        node(2, 1, 15), node(2, 1, 3), node(2, 1, 2),
    ]
    col = np.zeros((nx, ny), dtype=np.int16)
    seg = np.zeros((nx, ny, layers - 2), dtype=np.int16)

    router._accumulate_via_usage_for_path(
        path, net_id="N1", col_use=col, seg_use=seg
    )

    assert col[1, 1] == 1 and col[2, 1] == 1
    # Segments 2..22 (layers 3..23) charged exactly once at (1,1).
    assert seg[1, 1, 2:23].tolist() == [1] * 21
    assert seg[1, 1, :2].sum() == 0 and seg[1, 1, 23:].sum() == 0
    # (2,1): core hole once plus laser 2->3 (segment 1).
    assert seg[2, 1, 2:23].tolist() == [1] * 21
    assert seg[2, 1, 1] == 1
    # Keepouts cover every core layer at both sites.
    assert all((z, 1, 1) in router._via_keepouts_map for z in range(3, 25))
    assert (2, 2, 1) in router._via_keepouts_map

    # The incremental path must agree with the full rebuild.
    router.via_col_use[:] = 0
    router.via_seg_use[:] = 0
    router._via_keepouts_map = {}
    router.accounting_via._add_via_contribution("N1", path) if hasattr(
        router, "accounting_via"
    ) else router._add_via_contribution("N1", path)
    assert np.array_equal(router.via_col_use, col)
    assert np.array_equal(router.via_seg_use, seg)


def test_physical_cleanup_gate_can_be_widened():
    gate = UnifiedPathFinder._should_run_one_sided_cleanup

    # Ordinary tail gate: three overused edges at most.
    assert gate(1000, 2, False, 3, overuse_total=3)
    assert not gate(1000, 50, False, 3, overuse_total=4873)
    # Widened for barrel-dominated regimes.
    assert gate(1000, 50, False, 3, overuse_total=4873, total_threshold=8192)
    assert not gate(1000, 50, False, 3, overuse_total=9000, total_threshold=8192)
    # Never runs without physical conflicts.
    assert not gate(0, 0, False, 3, overuse_total=0, total_threshold=8192)


def test_hotset_cap_floor_only_applies_while_overuse_is_severe():
    config = PathFinderConfig()
    router = UnifiedPathFinder(config=config, use_gpu=False)

    severe = router._bounded_history_hotset_cap(20_000)
    moderate = router._bounded_history_hotset_cap(5_000)
    mild = router._bounded_history_hotset_cap(1_000)
    tail = router._bounded_history_hotset_cap(8)
    assert severe < 1024 and moderate < 1024

    config.hotset_cap_floor = 1024
    assert router._bounded_history_hotset_cap(20_000) == 1024
    assert router._bounded_history_hotset_cap(5_000) == 1024
    # Near convergence the staircase still protects the route from churn.
    assert router._bounded_history_hotset_cap(1_000) == mild
    assert router._bounded_history_hotset_cap(8) == tail

    config.hotset_cap_floor = 0
    assert router._bounded_history_hotset_cap(20_000) == severe


def test_router_without_core_span_keeps_hop_extent():
    router = _router(pcbway_mechanical_stack(28), 28)
    assert router._via_physical_extent(10, 15) == (10, 15)

    plain = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
    assert plain._via_physical_extent(7, 2) == (2, 7)


def test_router_expands_escape_into_laser_column_and_core_hole():
    router = _router(jlcpcb_hdi_stack(28), 28)

    expanded = router._expand_hdi_vias([{
        "net": "N1",
        "x": 1.0,
        "y": 2.0,
        "from_layer": "F.Cu",
        "to_layer": "In10.Cu",
        "diameter": 9.0,
        "drill": 9.0,
        "escape": True,
    }])

    assert [via["via_process"] for via in expanded] == [
        "laser_microvia",
        "laser_microvia",
        "laser_microvia",
        "mechanical_buried_core",
    ]
    assert (expanded[-1]["from_layer"], expanded[-1]["to_layer"]) == (
        "In3.Cu", "In24.Cu"
    )
    assert expanded[-1]["drill"] == pytest.approx(0.15)
    assert expanded[-1]["diameter"] == pytest.approx(0.30)
    assert expanded[0]["diameter"] == pytest.approx(0.25)

    hub_hop = router._expand_hdi_vias([{
        "net": "N1",
        "x": 1.0,
        "y": 2.0,
        "from_layer": "In10.Cu",
        "to_layer": "In3.Cu",
        "diameter": 9.0,
        "drill": 9.0,
    }])
    assert len(hub_hop) == 1
    assert (hub_hop[0]["from_layer"], hub_hop[0]["to_layer"]) == (
        "In3.Cu", "In24.Cu"
    )


def test_manufacturing_profiles_resolve():
    ids = [profile_id for profile_id, _ in MANUFACTURING_PROFILES]
    assert ids[0] == "board_default"
    assert "jlcpcb_3n3" in ids and "jlcpcb_adjacent" in ids
    for profile_id in ids:
        stack = stack_for_profile(profile_id, 28)
        if profile_id == "board_default":
            assert stack is None
        else:
            assert isinstance(stack, HDIStack)
            assert stack.layer_count == 28
    assert stack_for_profile("jlcpcb-hdi", 28).name == "jlcpcb_hdi_3+22+3"
    assert stack_for_profile(
        "jlcpcb_hdi", 28, core_via_cost_scale=3.0
    ).core_via_cost_scale == pytest.approx(3.0)
    comparison = jlcpcb_adjacent_stack(28)
    assert comparison.core_span is None
    assert comparison.name.endswith("comparison_only")
    assert comparison.center_spacing_by_span()[(0, 1)] == pytest.approx(0.4)


def test_vendor_helpers():
    assert stack_vendor("jlcpcb_hdi_3+22+3") == "jlcpcb"
    assert stack_vendor("pcbway_mechanical_adjacent_32L") == "pcbway"
    assert stack_vendor(None) == "pcbway"
    assert vendor_project_rules("jlcpcb")["min_hole_to_hole"] == (
        pytest.approx(0.24)
    )
    assert vendor_project_rules("pcbway")["min_hole_to_hole"] == (
        pytest.approx(0.2794)
    )


def _board_text():
    names = _layer_names(16)
    copper = ['\t\t(0 "F.Cu" signal)\n'] + [
        f'\t\t({2 * index + 2} "In{index}.Cu" signal)\n'
        for index in range(1, 15)
    ] + ['\t\t(2 "B.Cu" signal)\n']
    stack_layers = "".join(
        f'\t\t\t(layer "{name}" (type "copper") (thickness 0.035))\n'
        for name in names
    )
    return (
        "(kicad_pcb\n"
        "\t(version 20260206)\n"
        "\t(general (thickness 1.6))\n"
        "\t(layers\n"
        + "".join(copper)
        + '\t\t(1 "F.Mask" user)\n'
        "\t)\n"
        "\t(setup\n"
        "\t\t(stackup\n"
        + stack_layers
        + "\t\t)\n"
        "\t)\n"
        '\t(footprint "X" (layer "F.Cu"))\n'
        ")\n"
    )


def test_exporter_emits_jlcpcb_core_hole_as_buried_via(tmp_path):
    source = tmp_path / "source.kicad_pcb"
    source.write_text(_board_text(), encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({
        "source_sha256": source_sha,
        "tracks": [{
            "net": "N1", "layer": "In5.Cu",
            "x1": 1, "y1": 2, "x2": 3, "y2": 2, "width": 0.1,
        }],
        "vias": [
            {
                "net": "N1", "x": 1, "y": 2,
                "from_layer": "F.Cu", "to_layer": "In1.Cu",
                "diameter": 0.25, "drill": 0.1,
                "via_process": "laser_microvia", "via_kind": "microvia",
                "hdi_stack": "jlcpcb_hdi_3+10+3",
            },
            {
                "net": "N1", "x": 1, "y": 2,
                "from_layer": "In3.Cu", "to_layer": "In12.Cu",
                "diameter": 0.30, "drill": 0.15,
                "via_process": "mechanical_buried_core",
                "via_kind": "buried",
                "hdi_stack": "jlcpcb_hdi_3+10+3",
            },
        ],
    }), encoding="utf-8")
    output = tmp_path / "routed.kicad_pcb"

    result = export_geometry_to_board(
        geometry, source, output, layer_count=16
    )
    text = output.read_text(encoding="utf-8")

    assert result["vias"] == 2
    assert '(via micro' in text
    assert '(layers "F.Cu" "In1.Cu")' in text
    assert '(via blind' in text
    assert '(layers "In3.Cu" "In12.Cu")' in text
    manifest = json.loads(
        (tmp_path / "routed-fabrication.json").read_text(encoding="utf-8")
    )
    assert manifest["fabricator"] == "jlcpcb"
    assert manifest["fabrication_profile"] == (
        "jlcpcb_hdi_laser_buildup_mechanical_core"
    )
    assert manifest["qualification"] == (
        "preliminary_requires_jlcpcb_dfm_approval"
    )
    core_rows = [
        row for row in manifest["via_span_schedule"]
        if row["via_process"] == "mechanical_buried_core"
    ]
    assert core_rows == [{
        "from_layer": "In3.Cu",
        "to_layer": "In12.Cu",
        "from_index": 3,
        "to_index": 12,
        "copper_layers_spanned": 10,
        "dielectric_gaps_spanned": 9,
        "via_type": "buried",
        "via_process": "mechanical_buried_core",
        "via_kind": "buried",
        "diameter_mm": 0.3,
        "drill_mm": 0.15,
        "count": 1,
    }]
    markdown = (tmp_path / "routed-fabrication.md").read_text(
        encoding="utf-8"
    )
    assert "Preliminary JLCPCB via-span schedule" in markdown
    assert "requires JLCPCB stackup/DFM approval" in markdown
