import json

import pytest

from benchmarks.mark_route_superseded import mark_superseded


def test_mark_superseded_preserves_snapshots_and_records_reason(tmp_path):
    progress = tmp_path / "route-progress.json"
    progress.write_text(
        json.dumps({
            "status": "routing",
            "iterations": [{"iteration": 1}, {"iteration": 2}],
            "run_name": "example",
        }),
        encoding="utf-8",
    )

    marked = mark_superseded(
        progress,
        "measured_plateau",
        minimum_iteration=2,
    )
    persisted = json.loads(progress.read_text(encoding="utf-8"))

    assert marked["status"] == "failed"
    assert persisted["iterations"] == [
        {"iteration": 1},
        {"iteration": 2},
    ]
    assert persisted["previous_status"] == "routing"
    assert persisted["termination_reason"] == "measured_plateau"
    assert persisted["superseded"]["snapshots_preserved"] is True
    assert persisted["superseded"]["iteration"] == 2


def test_mark_superseded_refuses_terminal_or_early_journal(tmp_path):
    progress = tmp_path / "route-progress.json"
    progress.write_text(
        json.dumps({
            "status": "routing",
            "iterations": [{"iteration": 1}],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected at least 2"):
        mark_superseded(
            progress,
            "too_early",
            minimum_iteration=2,
        )

    journal = json.loads(progress.read_text(encoding="utf-8"))
    journal["status"] = "complete"
    progress.write_text(json.dumps(journal), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal status"):
        mark_superseded(progress, "do_not_overwrite")
