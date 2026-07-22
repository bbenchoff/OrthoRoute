"""Tests for the synthetic backplane generator and metrics harness.

Also the convergence scoreboard: the 2x4 neighbor board MUST fully route
with zero overuse (it does so in 1 iteration since the ownership-as-cost /
escape-planner fixes; before them it stranded a net and hit the iteration
cap). If the completeness/overuse assertions here regress, a negotiation
change broke convergence.
"""

import pytest

from benchmarks.metrics import collect_route_metrics
from benchmarks.synthetic_boards import make_backplane


class TestGenerator:
    def test_pairs_pattern(self):
        board = make_backplane(connectors=2, pins_per=16, layers=4, pattern="pairs")
        assert board.layer_count == 4
        assert len(board.components) == 2
        pads = [p for n in board.nets for p in n.pads]
        assert len(board.nets) == 16       # 32 pads / 2 per net
        assert len(pads) == 32
        assert all(len(n.pads) == 2 for n in board.nets)

    def test_neighbor_pattern(self):
        board = make_backplane(connectors=2, pins_per=8, layers=4, pattern="neighbor")
        assert len(board.nets) == 8
        for net in board.nets:
            a, b = net.pads
            # Straight across: same pin slot, different connector
            assert a.component_id != b.component_id
            assert a.position.y == b.position.y

    def test_bus_pattern(self):
        board = make_backplane(connectors=4, pins_per=8, layers=6, pattern="bus")
        assert len(board.nets) == 8
        assert all(len(n.pads) == 4 for n in board.nets)

    def test_every_pad_on_exactly_one_net(self):
        board = make_backplane(connectors=3, pins_per=10, layers=4,
                               pattern="pairs", rows=2)
        seen = set()
        for net in board.nets:
            for pad in net.pads:
                assert pad.id not in seen, f"pad {pad.id} on two nets"
                assert pad.net_id == net.id
                seen.add(pad.id)
        assert len(seen) == 30

    def test_pairs_deterministic(self):
        b1 = make_backplane(connectors=2, pins_per=12, layers=4, seed=7)
        b2 = make_backplane(connectors=2, pins_per=12, layers=4, seed=7)
        pairing1 = [[p.id for p in n.pads] for n in b1.nets]
        pairing2 = [[p.id for p in n.pads] for n in b2.nets]
        assert pairing1 == pairing2

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            make_backplane(pins_per=7, rows=2)  # doesn't divide
        with pytest.raises(ValueError):
            make_backplane(pattern="starburst")
        with pytest.raises(ValueError):
            make_backplane(connectors=3, pattern="neighbor")  # odd


@pytest.fixture(scope="module")
def routed():
    from orthoroute.algorithms.manhattan.unified_pathfinder import (
        PathFinderConfig, UnifiedPathFinder,
    )
    board = make_backplane(connectors=2, pins_per=4, layers=4,
                           pattern="neighbor")
    config = PathFinderConfig()
    config.portal_x_snap_max = 0.75
    config.max_iterations = 12  # bound runtime; convergence not asserted
    pf = UnifiedPathFinder(config=config, use_gpu=False)
    pf.initialize_graph(board)
    pf.map_all_pads(board)
    pf.precompute_all_pad_escapes(board)
    pf.prepare_routing_runtime()
    pf.route_multiple_nets(board.nets)
    pf.emit_geometry(board)
    return pf, board


class TestMetricsHarness:

    def test_metrics_structure(self, routed):
        pf, board = routed
        m = collect_route_metrics(pf, board, timings={"route": 1.0})

        assert m["board"]["nets"] == 4
        assert m["board"]["pads"] == 8
        assert m["lattice"]["layers"] == 4
        assert m["completion"]["total_nets"] == 4
        assert m["completion"]["routed_nets"] == 4, "trivial board must fully route"
        assert m["completion"]["complete"] is True
        assert m["convergence"]["overuse_total"] == 0, "trivial board must converge"
        assert m["convergence"]["iterations"] is not None
        assert m["copper"]["wirelength_mm"] > 0
        assert m["copper"]["layers_used"], "no layer carried copper"
        assert all(int(z) < 4 for z in m["copper"]["lateral_steps_per_layer"])
        assert m["timings_s"]["route"] == 1.0

    def test_metrics_json_serializable(self, routed):
        import json
        pf, board = routed
        json.dumps(collect_route_metrics(pf, board))
