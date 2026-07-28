from pathlib import Path

from benchmarks.summarize_route_progress import (
    _is_candidate_name,
    _label,
    _latest_hotset_policy_snapshot,
    _latest_layer_node_snapshot,
    _linear_layer_fit,
    _normalize_iteration,
    _physical_edge_overuse,
    _row,
    _stall_row,
    _write_layer_node_markdown,
    _write_layer_node_svg,
    _write_hotset_policy_markdown,
    _write_hotset_policy_svg,
)


def test_candidate_filter_includes_future_full_layer_sweeps():
    for layers in (14, 16, 18, 20, 24):
        assert _is_candidate_name(
            f"Backplane-{layers}L-{layers}L-O10-P10-"
            "HDI-pcbway_mechanical-20260727-progress.json"
        )


def test_candidate_filter_keeps_reduced_gates_and_rejects_noise():
    assert _is_candidate_name(
        "Backplane-8L-1024N-8L-HDI-pcbway_elic-progress.json"
    )
    assert _is_candidate_name(
        "Backplane-14L-1024N-14L-HDI-pcbway_mechanical-progress.json"
    )
    assert not _is_candidate_name(
        "Backplane-18L-16L-HDI-pcbway_mechanical-progress.json"
    )
    assert not _is_candidate_name(
        "Backplane-18L-18L-HDI-unrelated-progress.json"
    )


def test_labels_distinguish_same_layer_runs_by_revision_and_time():
    journal = {
        "run_name": (
            "Backplane-16L-16L-HDI-pcbway_mechanical-"
            "20260727_151423"
        ),
        "git_sha": "e19d9fd123456789",
    }

    assert _label(Path("ignored.json"), journal) == (
        "16L 8,192 nets mechanical e19d9fd 15:14"
    )


def test_comparison_row_records_where_each_minimum_occurred():
    run = {
        "label": "16L 8,192 nets mechanical",
        "path": Path("progress.json"),
        "journal": {"status": "routing"},
        "iterations": [
            {
                "iteration": 1,
                "routed_nets": 8192,
                "overuse_total": 20,
                "negotiated_overuse_total": 60,
                "barrel_conflicts": 50,
                "path_node_conflicts": 40,
                "pres_fac": 1.0,
            },
            {
                "iteration": 2,
                "routed_nets": 8192,
                "overuse_total": 10,
                "negotiated_overuse_total": 40,
                "barrel_conflicts": 60,
                "path_node_conflicts": 30,
                "pres_fac": 2.0,
            },
        ],
    }

    row = _row(run)

    assert row["best_overuse"] == 10
    assert row["best_overuse_iteration"] == 2
    assert row["initial_negotiated_overuse"] == 60
    assert row["best_negotiated_overuse"] == 40
    assert row["best_negotiated_overuse_iteration"] == 2
    assert row["final_negotiated_overuse"] == 40
    assert row["best_physical"] == 50
    assert row["best_physical_iteration"] == 1
    assert row["best_path_nodes"] == 30
    assert row["best_path_nodes_iteration"] == 2
    assert row["final_pres_fac"] == 2.0


def test_legacy_directed_arcs_normalize_to_one_physical_edge():
    item = {
        "edge_overuse": 20,
        "via_column_overuse": 3,
        "via_segment_overuse": 2,
        "path_node_overuse_total": 40,
    }

    assert _physical_edge_overuse(
        item, edge_accounting_mode="paired arcs normalized"
    ) == 15
    normalized = _normalize_iteration(
        item, edge_accounting_mode="paired arcs normalized"
    )
    assert normalized["_physical_edge_overuse"] == 15
    assert normalized["_physical_negotiated_overuse"] == 55


def test_stall_row_requires_eight_passes_without_a_new_minimum():
    iterations = [
        {
            "iteration": index,
            "_physical_negotiated_overuse": (
                100 - index if index <= 3 else 97 + index
            ),
        }
        for index in range(1, 12)
    ]
    run = {
        "label": "16L 8,192 nets mechanical current",
        "path": Path("progress.json"),
        "journal": {
            "status": "failed",
            "run_name": "Backplane-16L-16L-current",
            "git_sha": "abc",
        },
        "iterations": iterations,
        "edge_accounting_mode": "unique physical",
    }

    row = _stall_row(run)

    assert row["observation"] == "stalled"
    assert row["best_iteration"] == 3
    assert row["overuse_at_stall"] == 97
    assert row["stall_iteration"] == 11
    assert row["eligible_for_fit"] is True


def test_layer_fit_is_withheld_until_two_distinct_stalls():
    one = [{
        "layers": 16,
        "overuse_at_stall": 290_000,
        "eligible_for_fit": True,
        "edge_accounting": "unique physical",
    }]
    assert _linear_layer_fit(one) is None

    two = one + [{
        "layers": 20,
        "overuse_at_stall": 90_000,
        "eligible_for_fit": True,
        "edge_accounting": "unique physical",
    }]
    fit = _linear_layer_fit(two)
    assert fit is not None
    assert fit["slope"] == -50_000
    assert fit["zero_layer"] == 21.8


def test_zero_convergence_is_fit_eligible_without_a_plateau():
    run = {
        "label": "20L 8,192 nets mechanical current",
        "path": Path("progress.json"),
        "journal": {
            "status": "complete",
            "run_name": "Backplane-20L-20L-current",
            "git_sha": "def",
            "completion": {"complete": True},
        },
        "iterations": [
            {
                "iteration": 1,
                "_physical_negotiated_overuse": 10,
            },
            {
                "iteration": 2,
                "_physical_negotiated_overuse": 0,
            },
        ],
        "edge_accounting_mode": "unique physical",
    }

    row = _stall_row(run)

    assert row["observation"] == "converged"
    assert row["overuse_at_stall"] == 0
    assert row["stall_iteration"] == 2
    assert row["eligible_for_fit"] is True


def test_latest_layer_node_snapshot_selects_newest_instrumented_iteration(
    tmp_path,
):
    old = {
        "label": "old",
        "iterations": [{
            "iteration": 2,
            "path_node_layers": [{
                "layer": 0,
                "capacity_nodes": 4,
                "occupied_nodes": 2,
                "conflict_nodes": 0,
                "excess_uses": 0,
                "max_use": 1,
            }],
        }],
    }
    current = {
        "label": "20L current",
        "iterations": [
            {"iteration": 1},
            {
                "iteration": 3,
                "path_node_layers": [
                    {
                        "layer": 0,
                        "capacity_nodes": 4,
                        "occupied_nodes": 3,
                        "conflict_nodes": 1,
                        "excess_uses": 2,
                        "max_use": 3,
                    },
                    {
                        "layer": 1,
                        "capacity_nodes": 4,
                        "occupied_nodes": 2,
                        "conflict_nodes": 0,
                        "excess_uses": 0,
                        "max_use": 1,
                    },
                ],
            },
        ],
    }

    snapshot = _latest_layer_node_snapshot([old, current])

    assert snapshot["run"] == "20L current"
    assert snapshot["iteration"] == 3
    assert snapshot["rows"][0]["occupied_pct"] == 75.0
    assert snapshot["rows"][0]["conflict_pct"] == 25.0
    assert snapshot["rows"][0]["role"] == "outer"
    markdown = tmp_path / "layers.md"
    svg = tmp_path / "layers.svg"
    _write_layer_node_markdown(snapshot, markdown)
    _write_layer_node_svg(snapshot, svg)
    assert "20L current" in markdown.read_text(encoding="utf-8")
    rendered = svg.read_text(encoding="utf-8")
    assert 'width="1200" height="820"' in rendered
    assert "L0: 2 excess uses" in rendered


def test_hotset_policy_snapshot_measures_contiguous_wave_efficiency(
    tmp_path,
):
    run = {
        "label": "20L current",
        "iterations": [
            {
                "iteration": 1,
                "elapsed_seconds": 100.0,
                "_physical_edge_overuse": 20,
                "_physical_negotiated_overuse": 120,
                "path_node_overuse_total": 100,
                "escape_conflicts": 2,
                "portal_grid_conflicts": 30,
                "exact_barrel_conflicts": 50,
            },
            {
                "iteration": 2,
                "elapsed_seconds": 110.0,
                "_physical_edge_overuse": 16,
                "_physical_negotiated_overuse": 100,
                "path_node_overuse_total": 84,
                "hotset_size": 256,
                "hotset_cap": 256,
                "pres_fac": 2.0,
                "pres_fac_max": 64.0,
                "escape_conflicts": 3,
                "portal_grid_conflicts": 28,
                "exact_barrel_conflicts": 45,
            },
            {
                "iteration": 3,
                "elapsed_seconds": 130.0,
                "_physical_edge_overuse": 10,
                "_physical_negotiated_overuse": 60,
                "path_node_overuse_total": 50,
                "hotset_size": 512,
                "hotset_cap": 512,
                "pres_fac": 4.0,
                "pres_fac_max": 64.0,
                "slow_progress_events": 1,
                "slow_progress_fraction": 0.02,
                "hotset_rate_boost_until": 7,
                "escape_conflicts": 5,
                "portal_grid_conflicts": 24,
                "exact_barrel_conflicts": 35,
            },
        ],
    }

    snapshot = _latest_hotset_policy_snapshot([run])

    assert snapshot["rows"][0]["drop_per_second"] == 2.0
    assert snapshot["rows"][1]["hotset_size"] == 512
    assert snapshot["rows"][1]["drop_per_second"] == 2.0
    assert snapshot["rows"][1]["slow_progress_fraction"] == 2.0
    assert len(snapshot["phases"]) == 2
    assert snapshot["phases"][0]["escape_delta"] == 1
    assert snapshot["phases"][1]["portal_delta"] == -4
    assert snapshot["phases"][1]["exact_barrel_delta"] == -10
    assert snapshot["phases"][1]["matched_prior_iterations"] == "2-2"
    assert snapshot["phases"][1]["efficiency_ratio"] == 1.0
    assert snapshot["phases"][1]["iteration_efficiency_ratio"] == 2.0
    markdown = tmp_path / "hotset.md"
    svg = tmp_path / "hotset.svg"
    _write_hotset_policy_markdown(snapshot, markdown)
    _write_hotset_policy_svg(snapshot, svg)
    assert "256" in markdown.read_text(encoding="utf-8")
    rendered = svg.read_text(encoding="utf-8")
    assert 'width="1200" height="820"' in rendered
    assert "512-net hotset" in rendered
