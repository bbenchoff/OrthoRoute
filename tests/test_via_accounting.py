"""Tests for EdgeAccountant: present/canonical bookkeeping and overuse.

The PathFinder convergence invariant depends on `present` being rebuilt
from committed nets every iteration and staying in sync with the canonical
store; these tests pin that contract.
"""

import pytest

from orthoroute.algorithms.manhattan.unified_pathfinder import EdgeAccountant


@pytest.fixture
def acct():
    return EdgeAccountant(num_edges=10, use_gpu=False)


class TestCommitClear:
    def test_commit_updates_present_and_canonical(self, acct):
        acct.commit_path([1, 2, 3])
        assert acct.canonical == {1: 1, 2: 1, 3: 1}
        assert list(acct.present[[1, 2, 3]]) == [1.0, 1.0, 1.0]
        assert acct.verify_present_matches_canonical()

    def test_double_commit_stacks(self, acct):
        acct.commit_path([2, 3])
        acct.commit_path([3, 4])
        assert acct.canonical[3] == 2
        assert acct.present[3] == 2.0
        assert acct.verify_present_matches_canonical()

    def test_clear_reverses_commit(self, acct):
        acct.commit_path([1, 2])
        acct.clear_path([1, 2])
        assert acct.canonical == {}
        assert float(acct.present.sum()) == 0.0
        assert acct.verify_present_matches_canonical()

    def test_clear_never_goes_negative(self, acct):
        acct.clear_path([5])
        assert 5 not in acct.canonical
        assert acct.present[5] == 0.0

    def test_refresh_from_canonical_rebuilds(self, acct):
        acct.commit_path([1, 2, 2])
        acct.present.fill(99.0)  # corrupt
        acct.refresh_from_canonical()
        assert acct.present[2] == 2.0
        assert acct.present[0] == 0.0
        assert acct.verify_present_matches_canonical()


class TestOveruse:
    def test_no_overuse_at_capacity(self, acct):
        acct.commit_path([1, 2, 3])  # capacity is 1 everywhere
        total, count = acct.compute_overuse()
        assert (total, count) == (0, 0)

    def test_overuse_counts_excess(self, acct):
        acct.commit_path([1, 2])
        acct.commit_path([2, 3])
        acct.commit_path([2])
        total, count = acct.compute_overuse()
        assert total == 2  # edge 2 used 3x, capacity 1
        assert count == 1

    def test_mismatch_detected(self, acct):
        acct.commit_path([1])
        acct.present[1] = 5.0  # desync
        assert not acct.verify_present_matches_canonical()
