import numpy as np

from benchmarks.analyze_layer_capacity import crossing_profile


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
