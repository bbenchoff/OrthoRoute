import json

from benchmarks.summarize_kicad_drc import summarize, write_summary


def test_summarize_groups_rule_and_connectivity_items():
    rows = summarize({
        "violations": [
            {"severity": "warning", "type": "hole_to_hole"},
            {"severity": "error", "type": "clearance"},
            {"severity": "warning", "type": "hole_to_hole"},
        ],
        "unconnected_items": [
            {"severity": "error", "type": "unconnected_items"},
        ],
    })

    assert rows == [
        {
            "section": "unconnected",
            "severity": "error",
            "type": "unconnected_items",
            "count": 1,
        },
        {
            "section": "violation",
            "severity": "error",
            "type": "clearance",
            "count": 1,
        },
        {
            "section": "violation",
            "severity": "warning",
            "type": "hole_to_hole",
            "count": 2,
        },
    ]


def test_write_summary_reports_combined_electrical_errors(tmp_path):
    report_path = tmp_path / "board-drc.json"
    report_path.write_text(json.dumps({
        "violations": [
            {"severity": "error", "type": "clearance"},
            {"severity": "warning", "type": "silk_over_copper"},
        ],
        "unconnected_items": [
            {"severity": "error", "type": "unconnected_items"},
            {"severity": "error", "type": "unconnected_items"},
        ],
    }), encoding="utf-8")

    result = write_summary(report_path, tmp_path / "summary")

    assert result["rule_errors"] == 1
    assert result["unconnected_items"] == 2
    assert result["reported_errors"] == 3
    assert result["warnings"] == 1
    assert (tmp_path / "summary.csv").exists()
    assert "Reported electrical errors: 3" in (
        tmp_path / "summary.md"
    ).read_text(encoding="utf-8")
