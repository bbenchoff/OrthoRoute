"""ORTHO_STRICT=1 turns the present==canonical accounting mismatch into a
RuntimeError; the default remains a warning (historical behavior)."""
import logging

import pytest


@pytest.fixture
def corrupted_accountant():
    """EdgeAccountant whose present array disagrees with canonical."""
    from orthoroute.algorithms.manhattan.unified_pathfinder import EdgeAccountant
    acc = EdgeAccountant(num_edges=10, use_gpu=False)
    acc.commit_path([0, 1, 2])
    acc.present[5] = 3.0  # corrupt present directly; canonical knows nothing of edge 5
    assert not acc.verify_present_matches_canonical()
    return acc


def test_default_mode_warns_and_continues(corrupted_accountant, caplog, monkeypatch):
    from orthoroute.algorithms.manhattan.unified_pathfinder import (
        enforce_present_matches_canonical,
    )
    monkeypatch.delenv("ORTHO_STRICT", raising=False)
    with caplog.at_level(logging.WARNING):
        enforce_present_matches_canonical(corrupted_accountant, iteration=7)
    assert any("Accounting mismatch" in r.message for r in caplog.records)


def test_strict_mode_raises_with_iteration(corrupted_accountant, monkeypatch):
    from orthoroute.algorithms.manhattan.unified_pathfinder import (
        enforce_present_matches_canonical,
    )
    monkeypatch.setenv("ORTHO_STRICT", "1")
    with pytest.raises(RuntimeError, match=r"ITER 7.*Accounting mismatch"):
        enforce_present_matches_canonical(corrupted_accountant, iteration=7)


def test_clean_accountant_passes_in_both_modes(monkeypatch):
    from orthoroute.algorithms.manhattan.unified_pathfinder import (
        EdgeAccountant,
        enforce_present_matches_canonical,
    )
    acc = EdgeAccountant(num_edges=10, use_gpu=False)
    acc.commit_path([0, 1, 2])
    assert acc.verify_present_matches_canonical()
    monkeypatch.setenv("ORTHO_STRICT", "1")
    enforce_present_matches_canonical(acc, iteration=1)  # must not raise
