import numpy as np

from benchmarks.analyze_layer_capacity import (
    crossing_profile,
    minimum_internal_path_nodes,
)


def test_crossing_profile_counts_every_required_cut():
    profile = crossing_profile(
        [(0, 3), (1, 4), (3, 1), (2, 2)],
        cut_count=4,
    )

    assert np.array_equal(profile, np.array([1, 3, 3, 1]))


def test_crossing_profile_clamps_intervals_to_grid():
    profile = crossing_profile(
        [(-2, 2), (1, 8)],
        cut_count=4,
    )

    assert np.array_equal(profile, np.array([1, 2, 1, 1]))


def test_internal_node_bound_accounts_for_both_portal_offsets():
    # Portal pairs may shorten each distance by at most 2 * 12 steps.
    # Remaining planar hops are [6, 0, 1], and each path needs one node.
    assert minimum_internal_path_nodes(
        [30, 20, 25],
        portal_max_offset_steps=12,
    ) == 10
