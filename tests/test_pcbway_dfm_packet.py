import json

from benchmarks.prepare_pcbway_dfm_packet import (
    prepare_packet,
    provisional_stackup,
    span_review,
)


def _manifest():
    return {
        "board": "accepted.kicad_pcb",
        "layer_count": 6,
        "copper_layers": [
            "F.Cu",
            "In1.Cu",
            "In2.Cu",
            "In3.Cu",
            "In4.Cu",
            "B.Cu",
        ],
        "board_thickness_mm": 1.6,
        "via_span_schedule": [
            {
                "from_layer": "F.Cu",
                "to_layer": "In1.Cu",
                "from_index": 0,
                "to_index": 1,
                "copper_layers_spanned": 2,
                "dielectric_gaps_spanned": 1,
                "via_type": "blind",
                "via_process": "mechanical_blind_buried",
                "via_kind": "blind_buried",
                "diameter_mm": 0.3024,
                "drill_mm": 0.15,
                "count": 10,
            },
            {
                "from_layer": "F.Cu",
                "to_layer": "In4.Cu",
                "from_index": 0,
                "to_index": 5,
                "copper_layers_spanned": 6,
                "dielectric_gaps_spanned": 5,
                "via_type": "blind",
                "via_process": "mechanical_blind_buried",
                "via_kind": "blind_buried",
                "diameter_mm": 0.3024,
                "drill_mm": 0.15,
                "count": 2,
            },
        ],
    }


def test_span_review_flags_deep_nonadjacent_mechanical_spans():
    rows = span_review(_manifest())

    assert rows[0]["review_status"] == "requires_confirmation"
    assert rows[1]["review_status"] == "not_clearly_covered"
    assert "deeper_than_published_typical" in rows[1]["review_reasons"]


def test_stackup_audit_exposes_sub_2mil_dielectrics():
    manifest = _manifest()
    manifest["layer_count"] = 32
    manifest["copper_layers"] = (
        ["F.Cu"]
        + [f"In{index}.Cu" for index in range(1, 31)]
        + ["B.Cu"]
    )

    rows = provisional_stackup(manifest)

    assert sum(
        row["review"] == "below_published_2mil_minimum"
        for row in rows
    ) == 30


def test_prepare_packet_writes_review_and_zip(tmp_path):
    manifest_path = tmp_path / "board-fabrication.json"
    manifest_path.write_text(
        json.dumps(_manifest()), encoding="utf-8"
    )
    drc_summary = tmp_path / "board-drc-summary.md"
    drc_summary.write_text("zero errors\n", encoding="utf-8")

    result = prepare_packet(
        manifest_path,
        drc_summary,
        tmp_path / "PCBWay-DFM-6L",
    )

    assert result["total_drill_events"] == 12
    assert result["not_clearly_covered_span_classes"] == 1
    assert (tmp_path / "PCBWay-DFM-6L" / "README.md").exists()
    assert (tmp_path / "PCBWay-DFM-6L.zip").exists()
