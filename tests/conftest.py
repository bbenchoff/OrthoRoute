"""Shared pytest fixtures for the OrthoRoute test suite.

All tests here are KiCad-free and CPU-only: they exercise the engine's
data structures and the file-format code directly, with no IPC connection
and no GPU requirement.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def make_two_pad_board(layer_count: int = 4):
    """Minimal synthetic Board: two F.Cu SMD pads on one net, diagonal offset.

    Mirrors the --test-via smoke-test board; importable by any test that
    needs a routable board object.
    """
    from orthoroute.domain.models.board import (
        Board, Component, Coordinate, Net, Pad,
    )

    board = Board(id="test_board", name="Test Board")
    board.layer_count = layer_count

    pos1 = Coordinate(x=1.0, y=1.0)
    pos2 = Coordinate(x=5.0, y=5.0)
    comp1 = Component(id="comp1", reference="U1", value="T", footprint="T", position=pos1)
    comp2 = Component(id="comp2", reference="U2", value="T", footprint="T", position=pos2)
    pad1 = Pad(id="pad1", component_id="comp1", position=pos1, layer="F.Cu",
               size=(1.0, 1.0), net_id=None)
    pad2 = Pad(id="pad2", component_id="comp2", position=pos2, layer="F.Cu",
               size=(1.0, 1.0), net_id=None)
    comp1.pads.append(pad1)
    comp2.pads.append(pad2)
    net = Net(id="net1", name="TEST_NET", pads=[pad1, pad2])
    board.add_component(comp1)
    board.add_component(comp2)
    board.add_net(net)
    return board
