"""Reproducible snapshots of the effective PathFinder configuration."""

from typing import Any, Dict, Optional


def effective_pathfinder_config(
    config: Any,
    router: Optional[Any] = None,
) -> Dict[str, Any]:
    """Capture settings after board-derived tuning and environment overrides.

    The monster runner constructs its requested configuration before routing,
    but PathFinder applies board-derived values at negotiation start.  A
    callback-time snapshot is therefore the first authoritative view of the
    settings actually used by the route.
    """
    return {
        "max_iterations": int(config.max_iterations),
        "pres_fac_init": float(config.pres_fac_init),
        "pres_fac_mult": float(config.pres_fac_mult),
        "derived_pres_fac_max": float(config.pres_fac_max),
        "initial_pressure_ceiling": (
            None
            if router is None
            else float(getattr(router, "_initial_pres_fac_max", 0.0))
        ),
        "current_pressure_ceiling": (
            None
            if router is None
            else float(getattr(router, "_pres_fac_max_now", 0.0))
        ),
        "hist_cost_weight": float(config.hist_cost_weight),
        "hist_gain": float(config.hist_gain),
        "history_decay": float(config.history_decay),
        "ordinary_hotset_cap": int(config.hotset_cap),
        "stagnation_patience": int(config.stagnation_patience),
        "slow_progress_hotset_cap": int(
            config.slow_progress_hotset_cap
        ),
        "slow_progress_hotset_cap_max": int(
            config.slow_progress_hotset_cap_max
        ),
        "slow_progress_pressure_after": int(
            config.slow_progress_pressure_after
        ),
        "slow_progress_pres_fac_max": float(
            config.slow_progress_pres_fac_max
        ),
        "slow_progress_window": int(config.slow_progress_window),
        "slow_progress_min_fraction": float(
            config.slow_progress_min_fraction
        ),
    }
