"""Tests for Lattice3D construction and the CSR routing graph.

Pins the engine's core geometric invariants: flat node indexing, H/V layer
discipline, legal via pairs, and the (documented) fact that 2-layer boards
produce an empty routing graph because lateral edges exist only on inner
layers.
"""

import numpy as np
import pytest

from orthoroute.algorithms.manhattan.unified_pathfinder import (
    EdgeAccountant,
    Lattice3D,
    PathFinderConfig,
    UnifiedPathFinder,
)

BOUNDS = (0.0, 0.0, 4.0, 4.0)  # 4x4 mm
PITCH = 0.4


@pytest.fixture(scope="module")
def lattice4():
    return Lattice3D(BOUNDS, PITCH, layers=4)


class TestIndexing:
    def test_node_index_roundtrip(self, lattice4):
        for (x, y, z) in [(0, 0, 0), (3, 7, 1), (10, 10, 3)]:
            idx = lattice4.node_idx(x, y, z)
            assert lattice4.idx_to_coord(idx) == (x, y, z)

    def test_flat_index_formula(self, lattice4):
        # flat = layer*(x_steps*y_steps) + y*x_steps + x (CLAUDE.md invariant)
        plane = lattice4.x_steps * lattice4.y_steps
        assert lattice4.node_idx(2, 3, 1) == plane + 3 * lattice4.x_steps + 2

    def test_num_nodes(self, lattice4):
        assert lattice4.num_nodes == lattice4.x_steps * lattice4.y_steps * 4


class TestLayerDiscipline:
    def test_layer_directions(self, lattice4):
        # F.Cu vertical (escape stubs), then layers alternate H/V by parity.
        # B.Cu's entry is irrelevant: outer layers get no lateral edges.
        assert lattice4.layer_dir == ["v", "h", "v", "h"]

    def test_legal_planar_edges(self, lattice4):
        # In1.Cu (z=1) is horizontal: +X moves legal, +Y moves not.
        assert lattice4.is_legal_planar_edge(0, 0, 1, 1, 0, 1)
        assert not lattice4.is_legal_planar_edge(0, 0, 1, 0, 1, 1)
        # In2.Cu (z=2) is vertical: the opposite.
        assert lattice4.is_legal_planar_edge(0, 0, 2, 0, 1, 2)
        assert not lattice4.is_legal_planar_edge(0, 0, 2, 1, 0, 2)

    def test_custom_preferred_directions(self):
        lattice = Lattice3D(
            BOUNDS,
            PITCH,
            layers=4,
            preferred_layer_directions=["v", "v", "h", "h"],
        )
        assert lattice.layer_dir == ["v", "v", "h", "h"]
        assert lattice.get_allowed_axes(1) == ("v",)
        assert lattice.is_legal_planar_edge(0, 0, 1, 0, 1, 1)
        assert not lattice.is_legal_planar_edge(0, 0, 1, 1, 0, 1)

    def test_guided_layers_allow_both_axes(self):
        lattice = Lattice3D(
            BOUNDS,
            PITCH,
            layers=4,
            wrong_way_cost_multiplier=2.5,
        )
        assert lattice.get_allowed_axes(1) == ("h", "v")
        assert lattice.is_legal_planar_edge(0, 0, 1, 1, 0, 1)
        assert lattice.is_legal_planar_edge(0, 0, 1, 0, 1, 1)
        assert lattice.planar_cost_multiplier(1, "h") == 1.0
        assert lattice.planar_cost_multiplier(1, "v") == 2.5

    @pytest.mark.parametrize("multiplier", [0.5, float("nan")])
    def test_invalid_wrong_way_multiplier(self, multiplier):
        with pytest.raises(ValueError, match="wrong_way_cost_multiplier"):
            Lattice3D(
                BOUNDS,
                PITCH,
                layers=4,
                wrong_way_cost_multiplier=multiplier,
            )


class TestViaPairs:
    def test_four_layer_pairs(self, lattice4):
        pairs = lattice4.get_legal_via_pairs(4)
        # Adjacent pairs include F.Cu but exclude B.Cu.
        assert pairs == {(0, 1), (1, 2)}

    def test_four_layer_full_span_pairs(self, lattice4):
        pairs = lattice4.get_legal_via_pairs(
            4, allow_any_layer_via=True
        )
        assert pairs == {(1, 2), (0, 1), (0, 2)}

    def test_two_layer_pairs_through_via(self, lattice4):
        # 2-layer boards have no inner layers: F.Cu/B.Cu route directly and
        # the only legal via is the (0,1) through-hole (regression: #18).
        assert lattice4.get_legal_via_pairs(2) == {(0, 1)}


class TestGraphBuild:
    def test_four_layer_graph(self, lattice4):
        graph = lattice4.build_graph(via_cost=0.7)
        E = int(graph.indptr[-1])
        assert E > 0
        assert len(graph.indptr) == lattice4.num_nodes + 1
        assert len(graph.indices) == E
        assert len(graph.base_costs) == E

        # edge_kind: 1 for via edges, 0 for lateral; both kinds must exist.
        via_edges = int(np.sum(graph.edge_kind))
        assert 0 < via_edges < E

        # Expected lateral edge count: 2 directed edges per adjacent pair on
        # each inner layer, along that layer's legal axis only.
        xs, ys = lattice4.x_steps, lattice4.y_steps
        expected_lateral = 2 * ys * (xs - 1) + 2 * xs * (ys - 1)  # z=1 (h) + z=2 (v)
        assert E - via_edges == expected_lateral
        assert via_edges == 2 * xs * ys * 2

    def test_two_layer_graph_has_edges(self):
        # Regression for #18/#13: 2-layer boards used to produce an empty
        # graph (ValueError("No edges")) because lateral edges were emitted
        # only for inner layers. F.Cu/B.Cu must carry lateral routing when
        # no inner layers exist.
        lattice = Lattice3D(BOUNDS, PITCH, layers=2)
        graph = lattice.build_graph(via_cost=0.7)
        E = int(graph.indptr[-1])
        assert E > 0

        # Lateral edges on BOTH layers (F.Cu 'v', B.Cu 'h') plus through vias.
        via_edges = int(np.sum(graph.edge_kind))
        xs, ys = lattice.x_steps, lattice.y_steps
        expected_lateral = 2 * xs * (ys - 1) + 2 * ys * (xs - 1)  # z=0 (v) + z=1 (h)
        assert E - via_edges == expected_lateral
        # One physical pair, materialized as its two directed graph edges.
        assert via_edges == 2 * xs * ys

    def test_guided_graph_prices_wrong_way_edges(self):
        lattice = Lattice3D(
            BOUNDS,
            PITCH,
            layers=4,
            wrong_way_cost_multiplier=2.5,
        )
        graph = lattice.build_graph(via_cost=0.7)
        xs, ys = lattice.x_steps, lattice.y_steps
        via_edges = int(np.sum(graph.edge_kind))
        expected_lateral = 2 * (
            ys * (xs - 1) + xs * (ys - 1)
        ) * 2
        assert len(graph.indices) - via_edges == expected_lateral

        source = lattice.node_idx(2, 2, 1)
        targets = {
            lattice.node_idx(3, 2, 1): PITCH,
            lattice.node_idx(2, 3, 1): PITCH * 2.5,
        }
        start = int(graph.indptr[source])
        end = int(graph.indptr[source + 1])
        costs = {
            int(target): float(cost)
            for target, cost in zip(
                graph.indices[start:end],
                graph.base_costs[start:end],
            )
        }
        for target, expected in targets.items():
            assert costs[target] == pytest.approx(expected)

    def test_edge_costs_positive(self, lattice4):
        graph = lattice4.build_graph(via_cost=0.7)
        costs = np.asarray(graph.base_costs)
        assert np.all(costs > 0)

    def test_checkerboard_via_sites_meet_sub_pitch_clearance(self, lattice4):
        graph = lattice4.build_graph(
            via_cost=0.7,
            min_via_center_spacing=0.404,
        )
        plane = lattice4.x_steps * lattice4.y_steps
        via_sources = []
        for source in range(lattice4.num_nodes):
            start = int(graph.indptr[source])
            end = int(graph.indptr[source + 1])
            if np.any(graph.edge_kind[start:end] == 1):
                via_sources.append(source)

        via_xy = {
            (
                source % plane % lattice4.x_steps,
                source % plane // lattice4.x_steps,
            )
            for source in via_sources
        }
        assert via_xy
        assert all((x + y) % 2 == 0 for x, y in via_xy)
        assert len(via_xy) == sum(
            (x + y) % 2 == 0
            for x in range(lattice4.x_steps)
            for y in range(lattice4.y_steps)
        )

    def test_adjacent_via_hops_emit_as_one_span(self, lattice4):
        router = UnifiedPathFinder(PathFinderConfig(), use_gpu=False)
        router.lattice = lattice4
        path = [
            lattice4.node_idx(2, 3, 0),
            lattice4.node_idx(2, 3, 1),
            lattice4.node_idx(2, 3, 2),
            lattice4.node_idx(3, 3, 2),
        ]
        assert router._coalesce_vertical_runs(path) == [
            lattice4.node_idx(2, 3, 0),
            lattice4.node_idx(2, 3, 2),
            lattice4.node_idx(3, 3, 2),
        ]

    def test_path_to_edges_rejects_non_edge_hop(self, lattice4):
        router = UnifiedPathFinder(PathFinderConfig(), use_gpu=False)
        router.lattice = lattice4
        router.graph = lattice4.build_graph(via_cost=0.7)
        router._indptr_cpu = None
        router._indices_cpu = None
        valid_path = [
            lattice4.node_idx(2, 3, 1),
            lattice4.node_idx(3, 3, 1),
        ]
        invalid_path = [
            lattice4.node_idx(2, 3, 1),
            lattice4.node_idx(4, 3, 1),
        ]

        directed = router._path_to_directed_edges(valid_path)
        resources = router._path_to_edges(valid_path)

        assert len(directed) == 1
        assert len(resources) == 2
        assert set(resources) == set(
            router._path_to_edges(list(reversed(valid_path)))
        )
        with pytest.raises(ValueError, match="not a graph edge"):
            router._path_to_edges(invalid_path)

    def test_opposite_traversals_share_physical_edge_capacity(
        self, lattice4
    ):
        router = UnifiedPathFinder(PathFinderConfig(), use_gpu=False)
        router.lattice = lattice4
        router.graph = lattice4.build_graph(via_cost=0.7)
        router._indptr_cpu = None
        router._indices_cpu = None
        a = lattice4.node_idx(2, 3, 1)
        b = lattice4.node_idx(3, 3, 1)

        forward = router._path_to_edges([a, b])
        backward = router._path_to_edges([b, a])
        assert set(forward) == set(backward)

        router.accounting = EdgeAccountant(
            len(router.graph.indices), use_gpu=False
        )
        router.accounting.commit_path(forward)
        router.accounting.commit_path(backward)
        assert router.accounting.compute_overuse() == (2, 2)
        assert router.accounting.compute_overuse(router) == (1, 1)
