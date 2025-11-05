"""
Parameter Derivation Module

Derives optimal routing parameters from board characteristics.
Eliminates manual tuning - parameters scale with board complexity.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import logging

from .board_analyzer import BoardCharacteristics

logger = logging.getLogger(__name__)


@dataclass
class DerivedRoutingParameters:
    """Auto-tuned parameters for specific board"""

    # Present cost schedule
    pres_fac_init: float
    pres_fac_mult: float
    pres_fac_max: float

    # History cost
    hist_cost_weight: float
    hist_gain: float
    history_decay: float

    # Hotset configuration
    hotset_cap: int
    hotset_percentage: float

    # Layer bias controller
    layer_bias_alpha: float
    layer_bias_min: float
    layer_bias_max: float

    # Via configuration
    via_cost_base: float

    # Convergence
    stagnation_patience: int
    max_iterations: int

    # Strategy description
    strategy: str


def derive_routing_parameters(board: BoardCharacteristics) -> DerivedRoutingParameters:
    """
    Derive optimal routing parameters from board characteristics.

    Key scaling rules:
    - Congestion ratio ρ drives aggressiveness
    - Layer count L drives balancing strength
    - Net count N drives hotset sizing
    """
    logger.info("=" * 80)
    logger.info("DERIVING ROUTING PARAMETERS")
    logger.info("=" * 80)

    ρ = board.congestion_ratio
    L = board.layer_count
    N = board.net_count

    # ========================================================================
    # PRESENT COST SCHEDULE
    # ========================================================================
    # Principle: Sparse boards can escalate fast, dense boards need gentle ramp

    pres_fac_init = 1.0

    if ρ < 0.6:
        # SPARSE: Fast convergence - AGGRESSIVE growth to break plateau early
        pres_fac_mult = 1.30  # Increased from 1.15 to reach critical threshold ~3x faster
        pres_fac_max = 12.0   # Increased ceiling to allow full escalation
        strategy = "SPARSE (aggressive convergence)"
    elif ρ < 0.9:
        # NORMAL: Balanced - moderate acceleration
        pres_fac_mult = 1.20  # Increased from 1.12
        pres_fac_max = 10.0   # Increased ceiling
        strategy = "NORMAL (accelerated)"
    elif ρ < 1.2:
        # TIGHT: EXTREMELY AGGRESSIVE escalation for large boards
        pres_fac_mult = 2.5  # Extreme - hits max pressure by iteration 8
        pres_fac_max = 256.0   # Very high ceiling for difficult boards
        strategy = "TIGHT (extreme aggression for fast convergence)"
    else:
        # DENSE: Very gentle
        pres_fac_mult = 1.08
        pres_fac_max = 10.0
        strategy = "DENSE (conservative)"

    # Adjust max by layer count
    if L <= 12:
        pres_fac_max *= 0.75  # Lower ceiling for few layers
    elif L >= 25:
        pres_fac_max *= 1.25  # Higher ceiling for many layers

    logger.info(f"Present schedule: mult={pres_fac_mult:.3f}, max={pres_fac_max:.1f}")

    # ========================================================================
    # HISTORY COST
    # ========================================================================
    # Principle: History needs more weight when:
    # - Fewer layers (less flexibility, need strong memory)
    # - Higher congestion (more conflicts to remember)

    base_hist_weight = 8.0

    # Layer penalty: fewer layers = stronger history
    if L <= 10:
        layer_bonus = 6.0
    elif L <= 15:
        layer_bonus = 4.0
    elif L <= 25:
        layer_bonus = 2.0
    else:
        layer_bonus = 0.0

    # Congestion penalty: tighter board = stronger history
    congestion_bonus = 6.0 * max(0.0, min(1.0, ρ - 0.7))

    hist_cost_weight = base_hist_weight + layer_bonus + congestion_bonus

    # History gain: Balanced for tight boards
    # Not too high (fights present) or too low (gets drowned out)
    hist_gain = 0.15  # Moderate value for ρ ~ 0.9

    # No decay (full memory)
    history_decay = 1.0

    logger.info(f"History: weight={hist_cost_weight:.1f}, gain={hist_gain:.3f}, decay={history_decay:.3f}")

    # ========================================================================
    # HOTSET SIZING
    # ========================================================================
    # Principle: Hotset size scales with net count and inversely with congestion
    # - Dense boards: smaller hotset (less disruption)
    # - Sparse boards: larger hotset (faster convergence)

    # Base percentage
    if ρ < 0.7:
        hotset_pct = 0.25  # 25% for sparse boards
    elif ρ < 1.0:
        hotset_pct = 0.20  # 20% for normal boards
    elif ρ < 1.3:
        hotset_pct = 0.15  # 15% for tight boards
    else:
        hotset_pct = 0.12  # 12% for dense boards

    # Absolute limits
    hotset_cap = max(32, min(int(N * hotset_pct), int(N * 0.30)))

    logger.info(f"Hotset: cap={hotset_cap} ({hotset_pct*100:.0f}% of {N} nets)")

    # ========================================================================
    # LAYER BIAS CONTROLLER
    # ========================================================================
    # Principle: Fewer layers need stronger load balancing
    # - 6-12 layers: Aggressive balancing (α=0.20, wide range)
    # - 13-20 layers: Moderate balancing (α=0.12)
    # - 20+ layers: Gentle balancing (α=0.08)

    if L <= 12:
        layer_bias_alpha = 0.20
        layer_bias_min = 0.60
        layer_bias_max = 1.80
    elif L <= 20:
        layer_bias_alpha = 0.12
        layer_bias_min = 0.75
        layer_bias_max = 1.50
    else:
        layer_bias_alpha = 0.08
        layer_bias_min = 0.85
        layer_bias_max = 1.20

    logger.info(f"Layer bias: α={layer_bias_alpha:.3f}, range=[{layer_bias_min:.2f}, {layer_bias_max:.2f}]")

    # ========================================================================
    # VIA COSTS
    # ========================================================================
    # Principle: More via restrictions = higher via cost
    via_flexibility = board.via_flexibility

    if via_flexibility > 0.9:
        via_cost_base = 0.5  # Unrestricted vias
    elif via_flexibility > 0.7:
        via_cost_base = 0.7  # Some restrictions
    else:
        via_cost_base = 1.0  # Heavily restricted

    logger.info(f"Via cost: {via_cost_base:.2f} (flexibility={via_flexibility*100:.0f}%)")

    # ========================================================================
    # CONVERGENCE PARAMETERS
    # ========================================================================
    # Principle: Tighter boards need more iterations and patience

    if ρ < 0.8:
        stagnation_patience = 5
        max_iterations = 200  # Extended for full convergence
    elif ρ < 1.1:
        stagnation_patience = 6
        max_iterations = 200  # Extended for full convergence
    else:
        stagnation_patience = 8
        max_iterations = 200  # Extended for full convergence

    # Fewer layers also need more iterations (less flexibility)
    if L <= 12:
        max_iterations = max(max_iterations, 200)  # Extended
        stagnation_patience = max(stagnation_patience, 7)

    logger.info(f"Convergence: max_iters={max_iterations}, patience={stagnation_patience}")

    logger.info("=" * 80)
    logger.info(f"STRATEGY: {strategy}")
    logger.info("=" * 80)

    return DerivedRoutingParameters(
        pres_fac_init=pres_fac_init,
        pres_fac_mult=pres_fac_mult,
        pres_fac_max=pres_fac_max,
        hist_cost_weight=hist_cost_weight,
        hist_gain=hist_gain,
        history_decay=history_decay,
        hotset_cap=hotset_cap,
        hotset_percentage=hotset_pct,
        layer_bias_alpha=layer_bias_alpha,
        layer_bias_min=layer_bias_min,
        layer_bias_max=layer_bias_max,
        via_cost_base=via_cost_base,
        stagnation_patience=stagnation_patience,
        max_iterations=max_iterations,
        strategy=strategy,
    )


def apply_derived_parameters(config, derived: DerivedRoutingParameters):
    """
    Apply derived parameters to PathFinderConfig.
    Modifies config in-place.
    """
    logger.info("Applying derived parameters to config...")

    config.pres_fac_init = derived.pres_fac_init
    config.pres_fac_mult = derived.pres_fac_mult
    config.pres_fac_max = derived.pres_fac_max

    config.hist_cost_weight = derived.hist_cost_weight
    config.hist_gain = derived.hist_gain
    config.history_decay = derived.history_decay

    config.hotset_cap = derived.hotset_cap

    config.via_cost = derived.via_cost_base

    config.stagnation_patience = derived.stagnation_patience
    config.max_iterations = derived.max_iterations

    logger.info("Parameters applied successfully")
