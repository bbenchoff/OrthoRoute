import json

from benchmarks.summarize_kicad_drc import (
    defect_site_census,
    summarize,
    write_summary,
)


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
    assert result["defect_site_census"]["defect_site_count"] == 1
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary-defect-sites.csv").exists()
    assert "Reported electrical errors: 3" in (
        tmp_path / "summary.md"
    ).read_text(encoding="utf-8")


def test_defect_site_census_deduplicates_colocated_electrical_reports():
    report = {
        "violations": [
            {
                "severity": "error",
                "type": "clearance",
                "items": [{"pos": {"x": 10.0, "y": 20.0}}],
            },
            {
                "severity": "error",
                "type": "shorting_items",
                "items": [{"pos": {"x": 10.2, "y": 20.0}}],
            },
            {
                "severity": "error",
                "type": "hole_clearance",
                "items": [{"pos": {"x": 10.4, "y": 20.0}}],
            },
            {
                "severity": "warning",
                "type": "hole_to_hole",
                "items": [{"pos": {"x": 10.0, "y": 20.0}}],
            },
        ]
    }

    rows = defect_site_census(report)

    assert len(rows) == 1
    assert rows[0]["report_count"] == 3
    assert rows[0]["report_types"] == (
        "clearance:1;hole_clearance:1;shorting_items:1"
    )
