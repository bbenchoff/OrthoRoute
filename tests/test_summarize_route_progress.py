from pathlib import Path

from benchmarks.summarize_route_progress import (
    _is_candidate_name,
    _label,
    _latest_hotset_policy_snapshot,
    _latest_layer_balance_history,
    _latest_layer_node_snapshot,
    _linear_layer_fit,
    _normalize_iteration,
    _physical_edge_overuse,
    _row,
    _stall_row,
    _write_layer_node_markdown,
    _write_layer_node_svg,
    _write_layer_balance_markdown,
    _write_layer_balance_svg,
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


def test_layer_balance_history_tracks_relative_fabric_thirds(tmp_path):
    def layers(excess):
        return [
            {
                "layer": index,
                "capacity_nodes": 100,
                "occupied_nodes": 50,
                "conflict_nodes": value,
                "excess_uses": value,
                "max_use": 2,
            }
            for index, value in enumerate([0, *excess, 0])
        ]

    run = {
        "label": "8L balance test",
        "iterations": [
            {
                "iteration": 1,
                "_physical_negotiated_overuse": 80,
                "path_node_layers": layers([10, 10, 8, 8, 6, 6]),
            },
            {
                "iteration": 2,
                "_physical_negotiated_overuse": 70,
                "path_node_layers": layers([8, 8, 8, 8, 8, 8]),
            },
        ],
    }

    snapshot = _latest_layer_balance_history([run])

    assert snapshot["ranges"] == ["L1-L2", "L3-L4", "L5-L6"]
    assert snapshot["rows"][0]["internal_node_excess"] == 48
    assert snapshot["rows"][0]["shallow_pct"] == 41.667
    assert snapshot["rows"][1]["deep_pct"] == 33.333
    markdown = tmp_path / "balance.md"
    svg = tmp_path / "balance.svg"
    _write_layer_balance_markdown(snapshot, markdown)
    _write_layer_balance_svg(snapshot, svg)
    assert "Relative fabric thirds" in markdown.read_text(encoding="utf-8")
    rendered = svg.read_text(encoding="utf-8")
    assert 'width="1200" height="820"' in rendered
    assert "total internal excess" in rendered
    assert "shallow L1-L2" in rendered


def test_hotset_policy_snapshot_measures_contiguous_wave_efficiency(
    tmp_path,
):
    run = {
        "label": "20L current",
        "journal": {"max_iterations": 10},
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
                "hotset_conflict_pair_count": 20,
                "hotset_conflict_pairs_covered": 15,
                "hotset_conflict_pair_coverage_fraction": 0.75,
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
    assert (
        snapshot["rows"][1]["required_drop_per_remaining_pass"]
        == 8.5714
    )
    assert snapshot["rows"][1]["slow_progress_fraction"] == 2.0
    assert snapshot["rows"][1]["conflict_pair_coverage_pct"] == 75.0
    assert len(snapshot["phases"]) == 2
    assert snapshot["phases"][0]["escape_delta"] == 1
    assert snapshot["phases"][1]["portal_delta"] == -4
    assert snapshot["phases"][1]["exact_barrel_delta"] == -10
    assert snapshot["phases"][1]["matched_prior_iterations"] == "2-2"
    assert snapshot["phases"][1]["efficiency_ratio"] == 1.0
    assert snapshot["phases"][1]["iteration_efficiency_ratio"] == 2.0
    assert snapshot["phases"][1]["linear_pace_ratio"] == 4.6667
    assert snapshot["phases"][1]["projected_zero_iteration"] == 4.5
    assert (
        snapshot["phases"][1]["conflict_pair_coverage_pct_avg"]
        == 75.0
    )
    markdown = tmp_path / "hotset.md"
    svg = tmp_path / "hotset.svg"
    _write_hotset_policy_markdown(snapshot, markdown)
    _write_hotset_policy_svg(snapshot, svg)
    assert "256" in markdown.read_text(encoding="utf-8")
    rendered = svg.read_text(encoding="utf-8")
    assert 'width="1200" height="820"' in rendered
    assert "512-net hotset" in rendered
    assert "event 1" in rendered


def test_hotset_policy_comparison_does_not_cross_phase_boundary():
    hotsets = [None, 512, 512, 512, 256, 256, 1024, 1024, 1024]
    objectives = [1000, 900, 820, 760, 740, 720, 680, 650, 620]
    iterations = []
    for index, (hotset, objective) in enumerate(
        zip(hotsets, objectives),
        start=1,
    ):
        row = {
            "iteration": index,
            "elapsed_seconds": float(index * 10),
            "_physical_edge_overuse": objective // 10,
            "_physical_negotiated_overuse": objective,
            "path_node_overuse_total": objective - objective // 10,
        }
        if hotset is not None:
            row["hotset_size"] = hotset
        iterations.append(row)

    snapshot = _latest_hotset_policy_snapshot([{
        "label": "boundary test",
        "iterations": iterations,
    }])

    phase_1024 = snapshot["phases"][-1]
    assert phase_1024["iterations"] == "7-9"
    assert phase_1024["matched_prior_iterations"] == "5-6"
    assert phase_1024["matched_prior_drop_per_pass"] == 20.0
    assert phase_1024["drop_per_pass"] == 33.3333


def test_hotset_policy_splits_equal_hotsets_at_pressure_tiers():
    run = {
        "label": "20L pressure ladder",
        "journal": {"max_iterations": 20},
        "iterations": [],
    }
    for iteration, objective, pressure in (
        (1, 1000, 64.0),
        (2, 900, 64.0),
        (3, 850, 128.0),
        (4, 825, 232.32),
        (5, 800, 256.0),
        (6, 790, 464.64),
    ):
        row = {
            "iteration": iteration,
            "elapsed_seconds": float(iteration * 10),
            "_physical_edge_overuse": objective // 10,
            "_physical_negotiated_overuse": objective,
            "path_node_overuse_total": objective - objective // 10,
        }
        if iteration > 1:
            row.update({
                "hotset_size": 1024,
                "hotset_cap": 1024,
                "pres_fac": pressure,
                "adaptive_pressure_limit": 512.0,
                "pressure_trial_reference_ceiling": 256.0,
                "pressure_trial_reference_fraction": 0.01,
                "pressure_trial_underperform_count": 1,
                "pressure_backoff_count": 0,
                "pressure_rejected_ceiling": None,
            })
        run["iterations"].append(row)

    snapshot = _latest_hotset_policy_snapshot([run])

    assert [
        (phase["iterations"], phase["pressure_tier"])
        for phase in snapshot["phases"]
    ] == [
        ("2-2", 64),
        ("3-3", 128),
        ("4-5", 256),
        ("6-6", 512),
    ]
    last = snapshot["rows"][-1]
    assert last["adaptive_pressure_limit"] == 512.0
    assert last["pressure_trial_reference_ceiling"] == 256.0
    assert last["pressure_trial_reference_fraction"] == 1.0
    assert last["pressure_trial_underperform_count"] == 1
