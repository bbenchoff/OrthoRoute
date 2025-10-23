"""
PathFinder Configuration Module

Centralized configuration constants and dataclass for PathFinder routing algorithm.
All tunable parameters are defined here to avoid scattered magic numbers.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple

# ============================================================================
# PATHFINDER CONFIGURATION - ALL PARAMETERS IN ONE PLACE
# ============================================================================

# Grid and Geometry Parameters
GRID_PITCH = 0.4                    # Grid pitch in mm for routing lattice
# Note: LAYER_COUNT removed - use board.layer_count from KiCad API instead
# See: board.get_copper_layer_count() or board.layer_count property

# PathFinder Algorithm Parameters
BATCH_SIZE = 32                     # Number of nets processed per batch
MAX_ITERATIONS = 30                 # Maximum PathFinder negotiation iterations
MAX_SEARCH_NODES = 50000            # Maximum nodes explored per net
PER_NET_BUDGET_S = 0.5             # Time budget per net in seconds
MAX_ROI_NODES = 20000              # Maximum nodes in Region of Interest

# PathFinder Cost Parameters
PRES_FAC_INIT = 1.0                # Initial present factor for congestion
PRES_FAC_MULT = 2.0                # Present factor multiplier per iteration (faster growth breaks deadlocks)
PRES_FAC_MAX = 512.0               # Maximum present factor cap
HIST_ACCUM_GAIN = 1.0              # Historical cost accumulation gain
OVERUSE_EPS = 1e-6                 # Epsilon for overuse calculations

# Algorithm Tuning Parameters
DELTA_MULTIPLIER = 4.0             # Delta-stepping bucket size multiplier
ADAPTIVE_DELTA = True              # Enable adaptive delta tuning
STRICT_CAPACITY = True             # Enforce strict capacity constraints
REROUTE_ONLY_OFFENDERS = True      # Reroute only offending nets in incremental mode

# Via and Routing Parameters
VIA_COST = 1.0  # Cheap vias encourage spreading into empty vertical channels (was 6.0)
VIA_CAPACITY_PER_NET = 999         # Via capacity limit per net
STORE_REMAP_ON_RESIZE = 0          # Edge store remapping behavior

# Performance and Safety Parameters
ROI_SAFETY_CAP = MAX_ROI_NODES     # ROI node safety limit
NET_LIMIT = 0                      # Net processing limit (0 = no limit)
DISABLE_EARLY_STOP = False         # Disable early stopping optimization
CAPACITY_END_MODE = True           # End routing when capacity exhausted
EMERGENCY_CPU_ONLY = False         # Force CPU-only mode for debugging
SMART_FALLBACK = False             # Enable smart GPU->CPU fallback
DISABLE_GPU_ROI = False            # Disable GPU ROI extraction
DUMP_REPRO_BUNDLE = False          # Dump debug reproduction data

# Routing Quality Parameters
MIN_STUB_LENGTH_MM = 0.25          # Minimum visible stub length in mm
PAD_CLEARANCE_MM = 0.15            # Default pad clearance in mm
BASE_ROI_MARGIN_MM = 4.0           # Base ROI margin in mm
BOTTLENECK_RADIUS_FACTOR = 0.1     # Bottleneck radius as fraction of board width
HISTORICAL_ACCUMULATION = 0.1      # Historical cost accumulation factor

# ROI Widening Ladder Parameters
ROI_WIDEN_LEVELS = 4               # Number of widening levels to try (0=narrow, 1=wider, 2=generous, 3=full)
ROI_WIDEN_FACTOR = 2.0             # Margin multiplier per level

# Fixed Seed for Deterministic Routing
ROUTING_SEED = 42                  # Fixed seed for reproducible results

# Debugging and Profiling Parameters
ENABLE_PROFILING = False           # Enable performance profiling
ENABLE_INSTRUMENTATION = False     # Enable detailed instrumentation

# Negotiation Parameters
STAGNATION_PATIENCE = 5            # Iterations without improvement before stopping
STRICT_OVERUSE_BLOCK = True        # Block overused edges with infinite cost
HIST_COST_WEIGHT = 1.0             # Weight for historical cost component

# Legacy compatibility - keep original constant names for existing code
DEFAULT_BATCH_SIZE = BATCH_SIZE
DEFAULT_MAX_ITERS = MAX_ITERATIONS
DEFAULT_PRES_FAC_INIT = PRES_FAC_INIT
DEFAULT_PRES_FAC_MULT = PRES_FAC_MULT
DEFAULT_PRES_FAC_MAX = PRES_FAC_MAX
DEFAULT_HIST_ACCUM_GAIN = HIST_ACCUM_GAIN
DEFAULT_OVERUSE_EPS = OVERUSE_EPS
DEFAULT_STORE_REMAP_ON_RESIZE = STORE_REMAP_ON_RESIZE
DEFAULT_GRID_PITCH = GRID_PITCH

# Module metadata
VERSION_TAG = "UPF-2025-09-26-ripup_a1"


@dataclass
class PathFinderConfig:
    """Configuration for PathFinder routing algorithm - uses centralized constants."""
    batch_size: int = BATCH_SIZE
    max_iters: int = MAX_ITERATIONS
    max_iterations: int = MAX_ITERATIONS  # Alias for compatibility
    max_search_nodes: int = MAX_SEARCH_NODES
    pres_fac_init: float = PRES_FAC_INIT
    pres_fac_mult: float = PRES_FAC_MULT
    pres_fac_max: float = PRES_FAC_MAX
    hist_accum_gain: float = HIST_ACCUM_GAIN
    hist_gain: float = 2.5  # Alias for historical cost gain (unified_pathfinder uses this name)
    overuse_eps: float = OVERUSE_EPS
    mode: str = "near_far"  # Use fast GPU ROI pathfinding with actual CUDA kernels
    roi_parallel: bool = False
    per_net_budget_s: float = PER_NET_BUDGET_S
    max_roi_nodes: int = MAX_ROI_NODES
    delta_multiplier: float = DELTA_MULTIPLIER
    grid_pitch: float = GRID_PITCH
    # Centralized via policy (used everywhere; avoid module-level lookups)
    via_cost: float = VIA_COST
    via_capacity_per_net: int = VIA_CAPACITY_PER_NET
    # Blind/buried via policy
    allow_any_layer_via: bool = True
    enable_buried_via_keepouts: bool = True   # block intermediate layers at the (x,y) column
    keepout_weight: float = 1e9                # effectively "INF" for track edges touching a blocked node
    via_span_alpha: float = 0.08               # Small penalty for long via spans (reduces shaft congestion)
    # Column spreading parameters (to prevent "elevator shaft" congestion)
    column_spread_alpha: float = 0.25          # Fraction of overuse that leaks sideways (expert tuning: wider, gentler)
    column_spread_radius: int = 3              # Columns ±N get diffused history cost (expert tuning: wider spread)
    first_vertical_roundrobin_alpha: float = 0.12  # Nudge for round-robin layer selection (optimized)
    column_present_beta: float = 0.12          # Soft-cap slope for column occupancy (expert tuning: stronger pressure)
    corridor_widen_delta_cols: int = 2         # Columns to add when widening ROI
    corridor_widen_fail_threshold: int = 2     # Consecutive failures before widening
    column_jitter_eps: float = 1e-3            # Tiny deterministic jitter per column
    adaptive_delta: bool = ADAPTIVE_DELTA
    strict_capacity: bool = STRICT_CAPACITY
    reroute_only_offenders: bool = REROUTE_ONLY_OFFENDERS
    layer_count: int = 0  # Will be set from board.layer_count; 0 = uninitialized
    # Layer shortfall estimation tuning
    layer_shortfall_percentile: float = 95.0  # Percentile for congested channel estimation
    layer_shortfall_cap: int = 16            # Maximum layers to suggest
    enable_profiling: bool = ENABLE_PROFILING
    enable_instrumentation: bool = ENABLE_INSTRUMENTATION
    stagnation_patience: int = STAGNATION_PATIENCE
    strict_overuse_block: bool = STRICT_OVERUSE_BLOCK
    hist_cost_weight: float = HIST_COST_WEIGHT
    iter1_always_connect: bool = False  # DISABLED: Testing showed it reduces success from 33% to 20%
    iter1_relax_hv_discipline: bool = True  # ENABLED: Allow H/V violations in Iter-1 (soft penalty, not hard reject)
    iter1_disable_bitmap: bool = True  # ENABLED: Disable bitmap fencing in Iter-1 for max connectivity

    # Atomic parent keys (cycle prevention)
    use_atomic_parent_keys: bool = True  # Enable for iter>=2 to eliminate cycles
    atomic_parent_min_iter: int = 2  # Only use in iter>=2 (iter-1 uses standard parent updates)

    # Diagnostics toggles
    log_iteration_details: bool = False
    # Cost weights
    acc_fac: float = 0.0
    # Phase control parameters
    phase_block_after: int = 2
    congestion_multiplier: float = 1.0
    max_search_nodes: int = 2000000  # Maximum nodes explored per net
    # Portal escape configuration
    portal_enabled: bool = True
    portal_discount: float = 0.4  # 60% discount on first escape via from terminals
    portal_delta_min: int = 3      # Min vertical offset (1.2mm @ 0.4mm pitch)
    portal_delta_max: int = 12     # Max vertical offset (4.8mm)
    portal_delta_pref: int = 6     # Preferred offset (2.4mm)
    portal_x_snap_max: float = 0.5  # Max x-snap in steps (½ pitch)
    portal_via_discount: float = 0.15  # Escape via multiplier (85% discount)
    portal_retarget_patience: int = 3  # Iters before retargeting
    span_alpha: float = 0.15  # Span penalty: cost *= (1 + alpha*(span-1))
    hotset_cap: int = 150  # Guardrail to prevent mass decommits
    strict_drc: bool = False  # Legacy compatibility
    # Layer and via configuration
    layer_names: Optional[List[str]] = field(default_factory=lambda: ['F.Cu', 'In1.Cu', 'In2.Cu', 'In3.Cu', 'In4.Cu', 'B.Cu'])
    allowed_via_spans: Optional[Set[Tuple[int, int]]] = None  # None = all layer pairs allowed (blind/buried)
    # ROI widening ladder parameters
    roi_widen_levels: int = 4              # Number of widening levels to try
    roi_widen_factor: float = 2.0          # Margin multiplier per level
    BASE_ROI_MARGIN_MM: float = 4.0        # Base ROI margin for level 0