"""Tests for benchmark result accounting."""

from types import SimpleNamespace

from benchmarks.metrics import collect_route_metrics


class _Accounting:
    def compute_overuse(self, _router):
        return 0, 0


def _net(name, pad_count):
    return SimpleNamespace(
        name=name,
        id=name,
        pads=[object() for _ in range(pad_count)],
    )


def test_metrics_exclude_single_pin_nets_from_completion():
    board = SimpleNamespace(
        name="parsed-backplane",
        layer_count=2,
        nets=[
            _net("ROUTED", 2),
            _net("TRIVIAL", 2),
            _net("NO_CONNECT", 1),
        ],
    )
    router = SimpleNamespace(
        lattice=SimpleNamespace(
            x_steps=2, y_steps=2, layers=2, num_nodes=8, pitch=0.4,
        ),
        net_paths={"ROUTED": [0, 1, 5], "TRIVIAL": [2]},
        accounting=_Accounting(),
        iteration=3,
        _excluded_nets=set(),
    )

    metrics = collect_route_metrics(router, board)

    assert metrics["board"]["nets"] == 3
    assert metrics["board"]["routable_nets"] == 2
    assert metrics["board"]["singleton_nets"] == 1
    assert metrics["completion"] == {
        "routed_nets": 1,
        "trivial_nets": 1,
        "completed_nets": 2,
        "total_nets": 2,
        "excluded_nets": 0,
        "excluded_net_ids": [],
        "unrouted_net_ids": [],
        "complete": True,
    }
    assert metrics["copper"]["wirelength_mm"] == 0.4
    assert metrics["copper"]["via_transitions"] == 1
    assert metrics["copper"]["via_layer_steps"] == 1
    assert metrics["copper"]["layers_used"] == [0]


def test_metrics_coalesce_adjacent_layer_hops_into_one_via():
    board = SimpleNamespace(
        name="multi-hop-via",
        layer_count=4,
        nets=[_net("ROUTED", 2)],
    )
    # plane=4: 0→4→8 is one z run at xy=0, then a lateral step on z=2.
    router = SimpleNamespace(
        lattice=SimpleNamespace(
            x_steps=2, y_steps=2, layers=4, num_nodes=16, pitch=0.4,
        ),
        net_paths={"ROUTED": [0, 4, 8, 9]},
        accounting=_Accounting(),
        iteration=1,
        _excluded_nets=set(),
    )

    copper = collect_route_metrics(router, board)["copper"]

    assert copper["via_transitions"] == 1
    assert copper["via_layer_steps"] == 2


def test_metrics_report_excluded_and_unrouted_ids():
    board = SimpleNamespace(
        name="partial",
        layer_count=2,
        nets=[_net("DONE", 2), _net("DROPPED", 2)],
    )
    router = SimpleNamespace(
        lattice=SimpleNamespace(
            x_steps=2, y_steps=2, layers=2, num_nodes=8, pitch=0.4,
        ),
        net_paths={"DONE": [0, 1], "DROPPED": []},
        accounting=_Accounting(),
        iteration=5,
        _excluded_nets={"DROPPED"},
    )

    completion = collect_route_metrics(router, board)["completion"]

    assert completion["complete"] is False
    assert completion["excluded_net_ids"] == ["DROPPED"]
    assert completion["unrouted_net_ids"] == ["DROPPED"]
