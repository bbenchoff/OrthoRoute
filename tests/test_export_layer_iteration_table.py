import csv
import json
from pathlib import Path

from benchmarks.export_layer_iteration_table import export_tables


def test_export_layer_iteration_table_includes_metrics_and_hotspots(
    tmp_path: Path,
):
    results = tmp_path / "results"
    results.mkdir()
    journal_path = results / "Backplane-30L-test-progress.json"
    journal_path.write_text(
        json.dumps(
            {
                "status": "routing",
                "run_name": (
                    "Backplane-30L-FAB-pcbway_advanced_hdi-"
                    "HDI-pcbway_mechanical-20260728_000000"
                ),
                "started": "2026-07-28T00:00:00",
                "git_sha": "abc123",
                "source_sha256": "source",
                "experiment_config": {
                    "layer_count": 30,
                    "net_limit": 0,
                    "grid_pitch": 0.4,
                    "fabrication_profile": "pcbway_advanced_hdi",
                    "hdi_stack": "pcbway_mechanical",
                    "direction_mode": "guided",
                    "layer_depth_bias": 0.0,
                },
                "warm_start": {
                    "source_layers": 32,
                    "target_layers": 30,
                },
                "iterations": [
                    {
                        "iteration": 1,
                        "elapsed_seconds": 10.5,
                        "routed_nets": 8192,
                        "negotiated_overuse_total": 100,
                        "overuse_total": 20,
                        "edge_overuse": 3,
                        "via_column_overuse": 17,
                        "via_segment_overuse": 0,
                        "path_node_overuse_total": 80,
                        "exact_barrel_conflicts": 90,
                        "portal_grid_conflicts": 12,
                        "escape_conflicts": 2,
                        "pres_fac": 8.0,
                        "pres_fac_max": 128.0,
                        "hotset_size": 180,
                        "slow_progress_events": 0,
                        "slow_progress_fraction": 0.1,
                        "path_node_layers": [
                            {
                                "layer": 1,
                                "capacity_nodes": 100,
                                "occupied_nodes": 80,
                                "excess_uses": 70,
                            },
                            {
                                "layer": 2,
                                "capacity_nodes": 100,
                                "occupied_nodes": 90,
                                "excess_uses": 10,
                            },
                        ],
                    },
                    {
                        "iteration": 2,
                        "elapsed_seconds": 20,
                        "routed_nets": 8192,
                        "negotiated_overuse_total": 75,
                        "overuse_total": 15,
                        "path_node_overuse_total": 60,
                        "exact_barrel_conflicts": 70,
                        "portal_grid_conflicts": 9,
                        "pres_fac": 10,
                        "pres_fac_max": 128,
                        "hotset_size": 180,
                        "slow_progress_events": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "LAYER_ITERATION_TABLE.csv"
    markdown_path = tmp_path / "LAYER_ITERATION_TABLE.md"

    rows = export_tables(results, csv_path, markdown_path)

    assert len(rows) == 2
    assert rows[0]["total_layers"] == 30
    assert rows[0]["run_kind"] == "peel"
    assert rows[0]["hottest_node_layer"] == 1
    assert rows[0]["hottest_node_layer_excess"] == 70
    assert rows[0]["most_occupied_layer"] == 2
    assert rows[0]["peak_node_occupancy_pct"] == 90.0
    assert rows[1]["delta_complete_excess"] == -25
    assert rows[1]["running_best_excess"] == 75

    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[1]["complete_excess"] == "75"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## 30 total copper layers" in markdown
    assert "complete excess" in markdown
    assert "| 2 | 8192 | 75 | -25 | 75 |" in markdown


def test_preview_scope_can_be_recovered_from_legacy_run_name(
    tmp_path: Path,
):
    results = tmp_path / "results"
    results.mkdir()
    journal_path = results / "Backplane-20L-preview-progress.json"
    journal_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "run_name": (
                    "Backplane-20L-1024N-FAB-pcbway_advanced_hdi-"
                    "HDI-pcbway_mechanical-20260728_000000"
                ),
                "started": "2026-07-28T00:00:00",
                "experiment_config": {
                    "layer_count": 20,
                    "grid_pitch": 0.4,
                    "fabrication_profile": "pcbway_advanced_hdi",
                    "hdi_stack": "pcbway_mechanical",
                },
                "iterations": [
                    {
                        "iteration": 1,
                        "routed_nets": 1024,
                        "negotiated_overuse_total": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = export_tables(
        results,
        tmp_path / "table.csv",
        tmp_path / "table.md",
    )

    assert rows[0]["scope"] == "1024-net-preview"
    assert rows[0]["run_kind"] == "preview-fresh"
