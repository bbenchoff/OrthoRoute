"""Regression test for issue #13's track-extraction bug.

_extract_tracks called .get() on kipy Track proto objects (which have no
.get), so every existing track raised, warned, and was silently dropped.
Uses real kipy proto objects, no running KiCad needed.
"""

import pytest

kipy = pytest.importorskip("kipy")

from kipy.board_types import BoardLayer, Net, Track
from kipy.geometry import Vector2

from orthoroute.infrastructure.kicad.rich_kicad_interface import RichKiCadInterface


class FakeBoard:
    def __init__(self, tracks):
        self._tracks = tracks

    def get_tracks(self):
        return self._tracks


def make_track(x1_mm, y1_mm, x2_mm, y2_mm, width_mm, layer, net_name):
    t = Track()
    t.start = Vector2.from_xy_mm(x1_mm, y1_mm)
    t.end = Vector2.from_xy_mm(x2_mm, y2_mm)
    t.width = int(width_mm * 1_000_000)
    t.layer = layer
    net = Net()
    net.name = net_name
    t.net = net  # .net returns a copy; must assign a whole Net object
    return t


def test_tracks_extracted_from_proto_objects():
    iface = RichKiCadInterface.__new__(RichKiCadInterface)  # skip IPC connect
    board = FakeBoard([
        make_track(1.0, 2.0, 3.0, 2.0, 0.25, BoardLayer.BL_F_Cu, "GND"),
        make_track(0.0, 0.0, 0.0, 5.0, 0.2, BoardLayer.BL_In1_Cu, "VCC"),
    ])

    tracks = iface._extract_tracks(board)

    assert len(tracks) == 2, "proto tracks must not be silently dropped"
    first = tracks[0]
    assert first['start_x'] == pytest.approx(1.0)
    assert first['end_x'] == pytest.approx(3.0)
    assert first['width'] == pytest.approx(0.25)
    assert first['layer'] == 'F.Cu'
    assert first['net_name'] == 'GND'
    assert tracks[1]['layer'] == 'In1.Cu'
