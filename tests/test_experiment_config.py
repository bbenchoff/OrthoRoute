from types import SimpleNamespace

from benchmarks.experiment_config import effective_pathfinder_config


def test_effective_config_records_post_derivation_values():
    config = SimpleNamespace(
        max_iterations=200,
        pres_fac_init=1.0,
        pres_fac_mult=1.08,
        pres_fac_max=10.0,
        hist_cost_weight=16.0,
        hist_gain=0.35,
        history_decay=1.0,
        hotset_cap=1638,
        stagnation_patience=8,
        slow_progress_hotset_cap=512,
        slow_progress_hotset_cap_max=1024,
        slow_progress_pressure_after=2,
        slow_progress_pres_fac_max=1024.0,
        slow_progress_window=5,
        slow_progress_min_fraction=0.025,
    )
    router = SimpleNamespace(
        _initial_pres_fac_max=64.0,
        _pres_fac_max_now=256.0,
    )

    snapshot = effective_pathfinder_config(config, router)

    assert snapshot["max_iterations"] == 200
    assert snapshot["ordinary_hotset_cap"] == 1638
    assert snapshot["derived_pres_fac_max"] == 10.0
    assert snapshot["initial_pressure_ceiling"] == 64.0
    assert snapshot["current_pressure_ceiling"] == 256.0


def test_effective_config_without_router_marks_live_ceilings_unknown():
    config = SimpleNamespace(
        max_iterations=250,
        pres_fac_init=1.0,
        pres_fac_mult=1.1,
        pres_fac_max=8.0,
        hist_cost_weight=10.0,
        hist_gain=0.2,
        history_decay=1.0,
        hotset_cap=100,
        stagnation_patience=6,
        slow_progress_hotset_cap=512,
        slow_progress_hotset_cap_max=1024,
        slow_progress_pressure_after=2,
        slow_progress_pres_fac_max=1024.0,
        slow_progress_window=5,
        slow_progress_min_fraction=0.025,
    )

    snapshot = effective_pathfinder_config(config)

    assert snapshot["initial_pressure_ceiling"] is None
    assert snapshot["current_pressure_ceiling"] is None
