from benchmarks.summarize_route_progress import _is_candidate_name


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
