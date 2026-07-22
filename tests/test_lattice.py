"""Tests for Lattice3D construction and the CSR routing graph.

Pins the engine's core geometric invariants: flat node indexing, H/V layer
discipline, legal via pairs, and the (documented) fact that 2-layer boards
produce an empty routing graph because lateral edges exist only on inner
layers.
"""

import numpy as np
import pytest

from orthoroute.algorithms.manhattan.unified_pathfinder import Lattice3D

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


class TestViaPairs:
    def test_four_layer_pairs(self, lattice4):
        pairs = lattice4.get_legal_via_pairs(4)
        # Inner layers {1,2} full blind/buried + F.Cu transitions, no B.Cu.
        assert pairs == {(1, 2), (2, 1), (0, 1), (1, 0), (0, 2), (2, 0)}

    def test_two_layer_pairs_empty(self, lattice4):
        assert lattice4.get_legal_via_pairs(2) == set()


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

    def test_two_layer_graph_is_empty(self):
        # Documented limitation: no inner layers -> no lateral edges, no via
        # pairs -> build fails loudly rather than routing nothing.
        lattice = Lattice3D(BOUNDS, PITCH, layers=2)
        with pytest.raises(ValueError):
            lattice.build_graph(via_cost=0.7)

    def test_edge_costs_positive(self, lattice4):
        graph = lattice4.build_graph(via_cost=0.7)
        costs = np.asarray(graph.base_costs)
        assert np.all(costs > 0)
