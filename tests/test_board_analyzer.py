from orthoroute.algorithms.manhattan.board_analyzer import (
    assign_layer_axes_by_demand,
    preferred_layer_directions_for_board,
)
from orthoroute.domain.models.board import (
    Board,
    Coordinate,
    Net,
    Pad,
)


def _two_pad_net(name, first, second):
    return Net(
        id=name,
        name=name,
        pads=[
            Pad(
                id=f"{name}-1",
                component_id="A",
                net_id=None,
                position=Coordinate(*first),
                size=(0.2, 0.2),
            ),
            Pad(
                id=f"{name}-2",
                component_id="B",
                net_id=None,
                position=Coordinate(*second),
                size=(0.2, 0.2),
            ),
        ],
    )


def test_sixteen_layers_allocate_eight_h_for_sixty_percent_demand():
    h_layers, v_layers = assign_layer_axes_by_demand(
        list(range(1, 15)),
        0.604,
    )

    assert len(h_layers) == 8
    assert len(v_layers) == 6
    assert h_layers | v_layers == set(range(1, 15))
    assert not h_layers & v_layers


def test_board_demand_is_applied_before_graph_construction():
    board = Board(id="demand", name="Demand", layer_count=16)
    board.add_net(_two_pad_net("H", (0.0, 0.0), (60.4, 0.0)))
    board.add_net(_two_pad_net("V", (0.0, 0.0), (0.0, 39.6)))

    directions, h_layers, v_layers, h_fraction = (
        preferred_layer_directions_for_board(board, 16)
    )

    assert h_fraction == 0.604
    assert len(directions) == 16
    assert directions[0] == "v"
    assert len(h_layers) == 8
    assert len(v_layers) == 6
    assert sum(
        directions[layer] == "h" for layer in range(1, 15)
    ) == 8
