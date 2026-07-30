"""Unit tests for domain models: Board, Net, Pad, Coordinate, Bounds."""
import pytest
from orthoroute.domain.models.board import (
    Board,
    Bounds,
    Component,
    Coordinate,
    Layer,
    Net,
    Pad,
)


class TestCoordinate:
    def test_creation(self):
        c = Coordinate(x=1.0, y=2.0)
        assert c.x == 1.0
        assert c.y == 2.0

    def test_immutable(self):
        c = Coordinate(x=1.0, y=2.0)
        with pytest.raises((AttributeError, TypeError)):
            c.x = 99.0  # frozen dataclass must raise

    def test_equality(self):
        assert Coordinate(1.0, 2.0) == Coordinate(1.0, 2.0)
        assert Coordinate(1.0, 2.0) != Coordinate(1.0, 3.0)


class TestBounds:
    def test_width_height(self):
        b = Bounds(min_x=0.0, min_y=0.0, max_x=10.0, max_y=5.0)
        assert b.width == pytest.approx(10.0)
        assert b.height == pytest.approx(5.0)


class TestPad:
    def test_minimal_pad(self):
        p = Pad(
            id="pad1",
            component_id="U1",
            net_id="net_vcc",
            position=Coordinate(0.0, 0.0),
            size=(0.5, 0.5),
            layer="F.Cu",
        )
        assert p.net_id == "net_vcc"
        assert p.layer == "F.Cu"

    def test_pad_defaults(self):
        p = Pad(
            id="pad2",
            component_id="R1",
            net_id=None,
            position=Coordinate(1.0, 2.0),
            size=(0.3, 0.3),
        )
        assert p.shape == "circle"
        assert p.angle == 0.0


class TestNet:
    def test_net_has_pads(self):
        pad = Pad(id="p1", component_id="U1", net_id="n1",
                  position=Coordinate(0.0, 0.0), size=(0.5, 0.5))
        net = Net(id="n1", name="CLK", pads=[pad])
        assert net.name == "CLK"
        assert len(net.pads) == 1

    def test_empty_net_not_routable(self):
        net = Net(id="n2", name="UNCONNECTED", pads=[])
        assert not net.is_routable

    def test_net_with_two_pads_is_routable(self):
        pads = [
            Pad(id=f"p{i}", component_id="U1", net_id="n3",
                position=Coordinate(float(i), 0.0), size=(0.5, 0.5))
            for i in range(2)
        ]
        net = Net(id="n3", name="SIG", pads=pads)
        assert net.is_routable


class TestBoard:
    def test_board_holds_nets(self):
        board = Board(id="b1", name="test", nets=[], components=[], layers=[])
        assert board.nets == []

    def test_board_net_count(self):
        nets = [Net(id=f"n{i}", name=f"NET{i}") for i in range(5)]
        board = Board(id="b2", name="test2", nets=nets, components=[], layers=[])
        assert len(board.nets) == 5

    def test_board_add_net(self):
        board = Board(id="b3", name="test3")
        net = Net(id="n_new", name="ADDED")
        board.add_net(net)
        assert len(board.nets) == 1
        assert board.nets[0].name == "ADDED"
