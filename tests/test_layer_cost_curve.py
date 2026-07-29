import json

from benchmarks.summarize_layer_cost_curve import write_layer_cost_curve


def test_layer_cost_curve_uses_only_accepted_qualified_boards(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({
        "copper": {
            "wirelength_mm": 1234.5,
            "via_layer_steps": 77,
        },
    }), encoding="utf-8")
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({
        "iteration": 42,
        "elapsed_seconds": 99.5,
        "metrics": str(metrics),
        "warm_start": {"displaced_net_count": 2048},
    }), encoding="utf-8")
    manifest = tmp_path / "fabrication.json"
    manifest.write_text(json.dumps({
        "via_span_schedule": [
            {
                "from_layer": "F.Cu",
                "to_layer": "In1.Cu",
                "via_type": "blind",
                "dielectric_gaps_spanned": 1,
                "count": 10,
            },
            {
                "from_layer": "In1.Cu",
                "to_layer": "In3.Cu",
                "via_type": "buried",
                "dielectric_gaps_spanned": 2,
                "count": 5,
            },
        ],
    }), encoding="utf-8")
    state = {
        "runs": [
            {
                "layers": 30,
                "accepted": True,
                "reason": "symmetric_layer_pair_peel",
                "progress": str(progress),
                "qualification": {
                    "deliverable": {
                        "board": "accepted.kicad_pcb",
                        "drc": "accepted-drc.json",
                        "drc_warnings": 7,
                        "reported_errors": 2,
                        "fabrication_manifest_json": str(manifest),
                    },
                },
            },
            {
                "layers": 28,
                "accepted": False,
                "progress": "not-read.json",
            },
        ],
    }

    result = write_layer_cost_curve(state, tmp_path / "layer-cost-curve")

    assert result["accepted_board_count"] == 1
    assert result["layers"] == [30]
    csv_text = (tmp_path / "layer-cost-curve.csv").read_text(
        encoding="utf-8"
    )
    assert "1234.5" in csv_text
    assert ",15,77,7,2," in csv_text
    assert (tmp_path / "layer-cost-curve.md").exists()
    svg = (tmp_path / "layer-cost-curve.svg").read_text(encoding="utf-8")
    assert 'width="1200"' in svg
    assert "30L" in svg
