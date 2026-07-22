"""Tests for the KiCad-capable .kicad_pcb file parser.

The parser is exercised on small, self-contained synthetic boards (no
third-party board files) that pin both dialects and the footprint
transform:

- legacy (KiCad <= 7): numbered net table, ``(module ...)`` footprints,
  ``(fp_text reference ...)``.
- modern (KiCad 8-10, format 20260206): no net table (nets synthesized
  from pad references, including the name-only ``(net "NAME")`` form),
  ``(footprint ...)``, ``(property "Reference" ...)``.

Footprint rotation is CCW-positive in KiCad's Y-down board frame, and
back-side pads carry offsets already in the flipped frame (no extra
mirror); the expected pad positions below are computed from that rule
(the same transform is verified pad-exact against pcbnew 10.0.4 on real
boards).
"""

import pytest

from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser


LEGACY = """(kicad_pcb (version 20171130) (host pcbnew 5.0)
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
  (net 0 "")
  (net 1 "GND")
  (net 2 "VCC")
  (module R_0805 (layer F.Cu)
    (at 10.0 20.0 90)
    (fp_text reference "R1" (at 0 0))
    (fp_text value "10k" (at 0 0))
    (pad "1" smd rect (at -1.0 0) (size 1.0 1.3) (layers "F.Cu") (net 1 "GND"))
    (pad "2" smd rect (at 1.0 0) (size 1.0 1.3) (layers "F.Cu") (net 2 "VCC"))
  )
)"""


# KiCad 10 (format 20260206): no top-level net table, (footprint ...),
# (property "Reference" ...). A back-side footprint and a name-only
# (net "SIG") pad exercise the modern-dialect code paths.
MODERN = """(kicad_pcb (version 20260206) (generator pcbnew)
  (layers
    (0 "F.Cu" signal)
    (1 "In1.Cu" signal)
    (2 "In2.Cu" signal)
    (31 "B.Cu" signal)
    (44 "Edge.Cuts" user)
  )
  (footprint "R_0805" (layer "F.Cu")
    (at 10.0 20.0 90)
    (property "Reference" "R1" (at 0 0 0))
    (property "Value" "10k" (at 0 0 0))
    (pad "1" smd rect (at -1.0 0) (size 1.0 1.3) (layers "F.Cu") (net 1 "GND"))
    (pad "2" smd rect (at 1.0 0) (size 1.0 1.3) (layers "F.Cu") (net 2 "VCC"))
  )
  (footprint "LED_0603" (layer "B.Cu")
    (at 30.0 40.0 0)
    (property "Reference" "D1" (at 0 0 0))
    (property "Value" "RED" (at 0 0 0))
    (pad "1" smd rect (at -0.8 0) (size 0.8 0.9) (layers "B.Cu") (net 1 "GND"))
    (pad "2" smd rect (at 0.8 0) (size 0.8 0.9) (layers "B.Cu") (net "SIG"))
  )
)"""


def _load(tmp_path_factory, name, text):
    path = tmp_path_factory.mktemp(name) / (name + ".kicad_pcb")
    path.write_text(text, encoding="utf-8")
    return KiCadFileParser().load_board(str(path))


@pytest.fixture(scope="module")
def parsed(tmp_path_factory):
    return _load(tmp_path_factory, "legacy", LEGACY)


@pytest.fixture(scope="module")
def modern(tmp_path_factory):
    return _load(tmp_path_factory, "modern", MODERN)


def _by_ref(board, ref):
    return next(c for c in board.components if c.reference == ref)


class TestLegacyDialect:
    """KiCad <= 7 files: numbered net table, (module ...), fp_text reference."""

    def test_module_footprints_supported(self, parsed):
        assert len(parsed.components) == 1
        assert parsed.components[0].reference == "R1"
        assert parsed.components[0].value == "10k"

    def test_numbered_nets_resolved_to_names(self, parsed):
        assert {n.name for n in parsed.nets} == {"GND", "VCC"}
        pads = parsed.components[0].pads
        assert pads[0].net_id == "GND"
        assert pads[1].net_id == "VCC"

    def test_rotated_pad_world_position(self, parsed):
        # Footprint at (10, 20) rot 90 CCW in Y-down frame:
        # pad 1 local (-1, 0) -> world (10, 21).
        p1 = parsed.components[0].pads[0]
        assert p1.position.x == pytest.approx(10.0, abs=1e-6)
        assert p1.position.y == pytest.approx(21.0, abs=1e-6)

    def test_copper_count_excludes_edge_cuts(self, parsed):
        assert parsed.layer_count == 2


class TestKiCad10Dialect:
    """KiCad 8-10 files: no net table, (footprint ...), property reference."""

    def test_footprint_and_property_reference(self, modern):
        assert {c.reference for c in modern.components} == {"R1", "D1"}
        assert _by_ref(modern, "R1").value == "10k"

    def test_nets_synthesized_without_net_table(self, modern):
        # No top-level (net ...) table: names come from pad references,
        # including the name-only (net "SIG") form.
        assert {n.name for n in modern.nets} == {"GND", "VCC", "SIG"}

    def test_copper_count_excludes_edge_cuts(self, modern):
        assert modern.layer_count == 4  # F, In1, In2, B -- not Edge.Cuts

    def test_back_side_footprint_pad_layer(self, modern):
        assert _by_ref(modern, "D1").pads[0].layer == "B.Cu"

    def test_back_side_pad_position_no_mirror(self, modern):
        # D1 at (30, 40) rot 0 on B.Cu; pad 1 local (-0.8, 0) carries the
        # offset already in the flipped frame -> world (29.2, 40), no mirror.
        p1 = _by_ref(modern, "D1").pads[0]
        assert p1.position.x == pytest.approx(29.2, abs=1e-6)
        assert p1.position.y == pytest.approx(40.0, abs=1e-6)
