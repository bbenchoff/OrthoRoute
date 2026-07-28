from pathlib import Path

from benchmarks.summarize_route_progress import (
    _is_candidate_name,
    _label,
    _normalize_iteration,
    _physical_edge_overuse,
    _row,
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
