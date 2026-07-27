"""Explicit HDI stack topology and geometry regressions."""

import numpy as np
import pytest

from orthoroute.algorithms.manhattan.hdi_stack import (
    pcbway_elic_stack,
    pcbway_mechanical_stack,
)
from orthoroute.algorithms.manhattan.unified_pathfinder import (
    Lattice3D,
    PathFinderConfig,
    UnifiedPathFinder,
)


@pytest.mark.parametrize(
    ("layers", "notation", "core_pair"),
    [
        (8, "3+2+3", (3, 4)),
        (10, "4+2+4", (4, 5)),
        (12, "5+2+5", (5, 6)),
        (14, "6+2+6", (6, 7)),
    ],
)
def test_pcbway_elic_family(layers, notation, core_pair):
    stack = pcbway_elic_stack(layers)

    assert stack.notation == notation
    assert stack.core_pair == core_pair
    assert stack.allowed_via_spans == frozenset(
        (layer, layer + 1) for layer in range(layers - 1)
    )
    assert stack.process_for_span(*core_pair).name == "mechanical_core"
    assert stack.process_for_span(0, 1).name == "laser_microvia"
    assert stack.process_for_span(0, 1).drill_mm == pytest.approx(0.1)
    assert stack.process_for_span(*core_pair).drill_mm == pytest.approx(0.2)


def test_hdi_span_expands_to_adjacent_physical_vias():
    stack = pcbway_elic_stack(8)

    assert stack.expand_span(0, 4) == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    )
    assert stack.expand_span(4, 1) == (
        (4, 3),
        (3, 2),
        (2, 1),
    )


def test_pcbway_mechanical_stack_uses_cnc_geometry_everywhere():
    stack = pcbway_mechanical_stack(14)

    assert stack.name == "pcbway_mechanical_adjacent_14L"
    assert stack.allowed_via_spans == frozenset(
        (layer, layer + 1) for layer in range(13)
    )
    assert {
        process.name for _, process in stack.via_processes
    } == {"mechanical_blind_buried"}
    assert {
        process.kind for _, process in stack.via_processes
    } == {"blind_buried"}
    assert stack.process_for_span(0, 1).drill_mm == pytest.approx(0.15)
    assert stack.process_for_span(6, 7).diameter_mm == pytest.approx(
        0.3024
    )
    assert stack.center_spacing_by_span()[(0, 1)] == pytest.approx(
        0.4294
    )


def test_graph_applies_spacing_per_via_process():
    lattice = Lattice3D((0.0, 0.0, 0.4, 0.4), 0.2, layers=4)
    spans = {(0, 1), (1, 2), (2, 3)}
    graph = lattice.build_graph(
        via_cost=0.7,
        allowed_via_spans=spans,
        min_via_center_spacing=0.19,
        via_pair_center_spacing={
            (0, 1): 0.19,
            (1, 2): 0.25,
            (2, 3): 0.19,
        },
    )

    def has_edge(source, target):
        start = int(graph.indptr[source])
        end = int(graph.indptr[source + 1])
        return bool(np.any(graph.indices[start:end] == target))

    # Odd-parity site: laser microvias are legal on every 0.2 mm site,
    # while the larger core drill uses a checkerboard sublattice.
    x, y = 1, 0
    assert has_edge(
        lattice.node_idx(x, y, 0),
        lattice.node_idx(x, y, 1),
    )
    assert not has_edge(
        lattice.node_idx(x, y, 1),
        lattice.node_idx(x, y, 2),
    )
    assert has_edge(
        lattice.node_idx(x, y, 2),
        lattice.node_idx(x, y, 3),
    )


def test_router_expands_escape_via_with_span_specific_geometry():
    config = PathFinderConfig()
    config.layer_count = 8
    config.layer_names = (
        ["F.Cu"]
        + [f"In{layer}.Cu" for layer in range(1, 7)]
        + ["B.Cu"]
    )
    config.hdi_stack = pcbway_elic_stack(8)
    router = UnifiedPathFinder(config=config, use_gpu=False)

    expanded = router._expand_hdi_vias([{
        "net": "N1",
        "x": 1.0,
        "y": 2.0,
        "from_layer": "F.Cu",
        "to_layer": "In4.Cu",
        "diameter": 9.0,
        "drill": 9.0,
        "escape": True,
    }])

    assert len(expanded) == 4
    assert all(
        router._layer_name_to_index(via["to_layer"])
        - router._layer_name_to_index(via["from_layer"])
        == 1
        for via in expanded
    )
    assert [via["via_process"] for via in expanded] == [
        "laser_microvia",
        "laser_microvia",
        "laser_microvia",
        "mechanical_core",
    ]
    assert expanded[0]["drill"] == pytest.approx(0.1)
    assert expanded[-1]["drill"] == pytest.approx(0.2)
