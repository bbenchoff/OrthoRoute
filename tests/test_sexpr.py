"""Unit tests for the KiCad s-expression parser and node stripper."""

import pytest

from orthoroute.infrastructure.kicad.sexpr import (
    SExprError,
    atoms,
    child,
    children,
    find_top_level_spans,
    first_atom,
    node_name,
    parse,
    strip_top_level_nodes,
)


class TestParse:
    def test_simple_nested(self):
        result = parse('(kicad_pcb (version 20260206) (net "GND"))')
        assert result == [["kicad_pcb", ["version", "20260206"], ["net", "GND"]]]

    def test_quoted_string_escapes(self):
        result = parse(r'(property "Ref\"quoted\"" "line\nbreak")')
        assert result == [["property", 'Ref"quoted"', "line\nbreak"]]

    def test_atoms_with_special_chars(self):
        result = parse("(at 12.0523 -9.6333 90)")
        assert result == [["at", "12.0523", "-9.6333", "90"]]

    def test_empty_string_atom(self):
        assert parse('(net 0 "")') == [["net", "0", ""]]

    def test_unbalanced_open_raises(self):
        with pytest.raises(SExprError):
            parse("(kicad_pcb (net")

    def test_unbalanced_close_raises(self):
        with pytest.raises(SExprError):
            parse("(net))")

    def test_unterminated_string_raises(self):
        with pytest.raises(SExprError):
            parse('(net "GND')

    def test_multiline_whitespace(self):
        result = parse('(segment\n\t(start 1 2)\n\t(end 3 4)\n)')
        assert result == [["segment", ["start", "1", "2"], ["end", "3", "4"]]]


class TestHelpers:
    NODE = ["footprint", "R_0402",
            ["property", "Reference", "R1"],
            ["pad", "1", "smd", ["net", "GND"]],
            ["pad", "2", "smd", ["net", "VCC"]]]

    def test_node_name(self):
        assert node_name(self.NODE) == "footprint"
        assert node_name("atom") is None
        assert node_name([]) is None

    def test_children(self):
        pads = children(self.NODE, "pad")
        assert len(pads) == 2
        assert pads[0][1] == "1"

    def test_child_first_match(self):
        assert child(self.NODE, "property") == ["property", "Reference", "R1"]
        assert child(self.NODE, "missing") is None

    def test_atoms_and_first_atom(self):
        assert atoms(self.NODE) == ["R_0402"]
        assert first_atom(child(self.NODE, "pad")) == "1"
        assert first_atom(None) is None


class TestStripper:
    BOARD = (
        '(kicad_pcb\n'
        '\t(version 20260206)\n'
        '\t(segment\n\t\t(start 1 2)\n\t\t(net "GND")\n\t)\n'
        '\t(zone\n\t\t(net "GND")\n\t)\n'
        '\t(via\n\t\t(at 3 4)\n\t)\n'
        ')\n'
    )

    def test_strip_removes_only_named_nodes(self):
        stripped, removed = strip_top_level_nodes(self.BOARD, ("segment", "via"))
        assert removed == 2
        assert "segment" not in stripped
        assert "(via" not in stripped
        assert "zone" in stripped
        assert "(version 20260206)" in stripped

    def test_strip_is_idempotent(self):
        once, n1 = strip_top_level_nodes(self.BOARD, ("segment", "via"))
        twice, n2 = strip_top_level_nodes(once, ("segment", "via"))
        assert n1 == 2 and n2 == 0
        assert once == twice

    def test_strip_leaves_no_blank_lines(self):
        stripped, _ = strip_top_level_nodes(self.BOARD, ("segment", "via"))
        assert "\n\n" not in stripped

    def test_stripped_output_still_parses(self):
        stripped, _ = strip_top_level_nodes(self.BOARD, ("segment", "via"))
        result = parse(stripped)
        names = [node_name(c) for c in result[0][1:] if isinstance(c, list)]
        assert names == ["version", "zone"]

    def test_deep_nodes_not_stripped(self):
        # A (via ...) nested inside a footprint (e.g. padstack) must survive.
        text = '(kicad_pcb\n\t(footprint "X"\n\t\t(via (at 1 1))\n\t)\n)\n'
        stripped, removed = strip_top_level_nodes(text, ("via",))
        assert removed == 0
        assert stripped == text

    def test_paren_inside_string_ignored(self):
        text = '(kicad_pcb\n\t(net ")(")\n\t(segment (net ")("))\n)\n'
        spans = find_top_level_spans(text, ("segment",))
        assert len(spans) == 1
        stripped, removed = strip_top_level_nodes(text, ("segment",))
        assert removed == 1
        assert '(net ")(")' in stripped
