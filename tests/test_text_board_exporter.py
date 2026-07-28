"""Tests for deterministic reduced-layer KiCad text export."""

import hashlib
import json

import pytest

from orthoroute.infrastructure.kicad.text_board_exporter import (
    export_geometry_to_board,
    export_project_for_geometry,
)


def _board_text():
    copper = (
        ['\t\t(0 "F.Cu" signal)\n']
        + [
            f'\t\t({2 * index + 2} "In{index}.Cu" signal)\n'
            for index in range(1, 15)
        ]
        + ['\t\t(2 "B.Cu" signal)\n']
    )
    stack_layers = "".join(
        f'\t\t\t(layer "{name}" (type "copper") (thickness 0.035))\n'
        for name in (
            ["F.Cu"]
            + [f"In{index}.Cu" for index in range(1, 15)]
            + ["B.Cu"]
        )
    )
    return (
        "(kicad_pcb\n"
        "\t(version 20260206)\n"
        "\t(general (thickness 1.6))\n"
        "\t(layers\n"
        + "".join(copper)
        + '\t\t(1 "F.Mask" user)\n'
        "\t)\n"
        "\t(setup\n"
        "\t\t(stackup\n"
        + stack_layers
        + "\t\t)\n"
        "\t)\n"
        '\t(footprint "X" (layer "F.Cu"))\n'
        ")\n"
    )


def _write_inputs(tmp_path):
    source = tmp_path / "source.kicad_pcb"
    source.write_text(_board_text(), encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({
        "source_sha256": source_sha,
        "tracks": [{
            "net": "N1", "layer": "In12.Cu",
            "x1": 1, "y1": 2, "x2": 3, "y2": 2, "width": 0.1,
        }],
        "vias": [
            {
                "net": "N1", "x": 1, "y": 2,
                "from_layer": "F.Cu", "to_layer": "In1.Cu",
                "diameter": 0.2524, "drill": 0.1,
                "via_process": "laser_microvia",
            },
            {
                "net": "N1", "x": 3, "y": 2,
                "from_layer": "In6.Cu", "to_layer": "In7.Cu",
                "diameter": 0.3024, "drill": 0.15,
                "via_process": "mechanical_blind_buried",
            },
        ],
    }), encoding="utf-8")
    return source, geometry


def test_export_reduces_layers_and_preserves_via_types(tmp_path):
    source, geometry = _write_inputs(tmp_path)
    output = tmp_path / "routed.kicad_pcb"

    result = export_geometry_to_board(
        geometry, source, output, layer_count=14
    )
    text = output.read_text(encoding="utf-8")

    assert result["tracks"] == 1
    assert result["vias"] == 2
    assert '"In12.Cu"' in text
    assert '"In13.Cu"' not in text
    assert "(via micro" in text
    assert "(via blind" in text
    assert "(thickness 1.6)" in text
    assert text.count('(type "copper")') == 14


def test_export_refuses_source_hash_mismatch(tmp_path):
    source, geometry = _write_inputs(tmp_path)
    payload = json.loads(geometry.read_text(encoding="utf-8"))
    payload["source_sha256"] = "0" * 64
    geometry.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source hash mismatch"):
        export_geometry_to_board(
            geometry, source, tmp_path / "out.kicad_pcb", layer_count=14
        )


def test_export_refuses_to_overwrite_source(tmp_path):
    source, geometry = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="refusing to overwrite"):
        export_geometry_to_board(
            geometry, source, source, layer_count=14
        )


def test_project_export_sets_geometry_constraints(tmp_path):
    _, geometry = _write_inputs(tmp_path)
    source_project = tmp_path / "source.kicad_pro"
    output_project = tmp_path / "routed.kicad_pro"
    source_project.write_text(json.dumps({
        "board": {"design_settings": {"rules": {}}},
        "net_settings": {"classes": [{
            "name": "Default",
            "clearance": 0.2,
            "track_width": 0.2,
        }]},
    }), encoding="utf-8")

    export_project_for_geometry(
        geometry, source_project, output_project
    )
    project = json.loads(output_project.read_text(encoding="utf-8"))
    rules = project["board"]["design_settings"]["rules"]
    default_class = project["net_settings"]["classes"][0]

    assert rules["min_track_width"] == pytest.approx(0.1)
    assert rules["min_via_annular_width"] == pytest.approx(0.0762)
    assert rules["min_microvia_drill"] == pytest.approx(0.1)
    assert rules["min_through_hole_diameter"] == pytest.approx(0.15)
    assert default_class["clearance"] == pytest.approx(0.1)
    assert default_class["via_diameter"] == pytest.approx(0.3024)


def test_export_consolidates_touching_mechanical_via_stack(tmp_path):
    source, geometry = _write_inputs(tmp_path)
    payload = json.loads(geometry.read_text(encoding="utf-8"))
    payload["vias"] = [
        {
            "net": "N1", "x": 3, "y": 2,
            "from_layer": "F.Cu", "to_layer": "In1.Cu",
            "diameter": 0.3024, "drill": 0.15,
            "via_process": "mechanical_blind_buried",
        },
        {
            "net": "N1", "x": 3, "y": 2,
            "from_layer": "In1.Cu", "to_layer": "In2.Cu",
            "diameter": 0.3024, "drill": 0.15,
            "via_process": "mechanical_blind_buried",
        },
    ]
    geometry.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "routed.kicad_pcb"

    result = export_geometry_to_board(
        geometry, source, output, layer_count=14
    )
    text = output.read_text(encoding="utf-8")

    assert result["vias"] == 1
    assert '(layers "F.Cu" "In2.Cu")' in text
    manifest = json.loads(
        (tmp_path / "routed-fabrication.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["fabrication_profile"] == (
        "pcbway_advanced_hdi_mechanical"
    )
    assert manifest["fabricator_approval_required"] is True
    assert manifest["via_span_schedule"] == [{
        "from_layer": "F.Cu",
        "to_layer": "In2.Cu",
        "from_index": 0,
        "to_index": 2,
        "copper_layers_spanned": 3,
        "dielectric_gaps_spanned": 2,
        "via_type": "blind",
        "via_process": "mechanical_blind_buried",
        "via_kind": "",
        "diameter_mm": 0.3024,
        "drill_mm": 0.15,
        "count": 1,
    }]
    assert (tmp_path / "routed-fabrication.csv").exists()
    assert (tmp_path / "routed-fabrication.md").exists()
