"""
═══════════════════════════════════════════════════════════════════════════════
UNIFIED HIGH-PERFORMANCE PATHFINDER - PCB ROUTING ENGINE WITH PORTAL ESCAPES
═══════════════════════════════════════════════════════════════════════════════

ALGORITHM OVERVIEW:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This implements the PathFinder negotiated congestion routing algorithm for
multi-layer PCB routing with full blind/buried via support and portal-based
pad escapes. PathFinder is an iterative refinement algorithm that resolves
resource conflicts through economic pressure.

CORE PATHFINDER LOOP:
───────────────────────────────────────────────────────────────────────────────
1. Initialize: Build 3D lattice graph with H/V layer constraints + portal escapes
2. For iteration = 1 to MAX_ITERATIONS:
     a) REFRESH: Rebuild usage from committed net paths (clean accounting)
     b) UPDATE_COSTS: Apply congestion penalties (present + historical)
     c) HOTSET: Select only nets touching overused edges (adaptive cap)
     d) ROUTE: Route hotset nets using heap-based Dijkstra on ROI subgraphs
     e) COMMIT: Update edge usage and tracking structures
     f) CHECK: If no overuse → SUCCESS, exit
     g) ESCALATE: Increase present_factor pressure for next iteration
3. If max iterations reached: Run detail refinement pass on conflict zones

KEY INSIGHT: PathFinder uses economics - overused edges get expensive, forcing
nets to find alternatives. Historical cost prevents oscillation. Portal escapes
provide cheap entry to inner layers to spread routing across all 18 layers.

═══════════════════════════════════════════════════════════════════════════════
PRECOMPUTED PAD ESCAPE ARCHITECTURE (THE BREAKTHROUGH)
═══════════════════════════════════════════════════════════════════════════════

THE PROBLEM:
───────────────────────────────────────────────────────────────────────────────
SMD pads are physically on F.Cu (layer 0). Without precomputed escapes:
• All nets start on F.Cu → massive congestion on top layer
• Via cost (3.0) discourages layer changes → router fights on F.Cu/In1.Cu
• 16 inner layers sit idle while top layers are saturated
• Routing completion: 16% (only 73/464 nets route successfully)

THE SOLUTION: PRECOMPUTED DRC-CLEAN PAD ESCAPES
───────────────────────────────────────────────────────────────────────────────
Before routing begins, we precompute escape routing for EVERY pad attached to
a net. This completely eliminates F.Cu as a bottleneck by distributing traffic
across 9 horizontal routing layers (In1, In3, In5, In7, In9, In11, In13, In15, B.Cu).

The routing problem transforms from "route 3200 pads on F.Cu" to "route between
portal landing points on horizontal layers" - a pure grid routing problem that
PathFinder excels at.

PRECOMPUTATION PIPELINE:
───────────────────────────────────────────────────────────────────────────────
1. PORTAL PLANNING (per pad):
   • X-alignment: Snap to nearest lattice column (±½ pitch = 0.2mm tolerance)
   • Length: Random 1.2mm - 5mm (3-12 grid steps @ 0.4mm pitch)
   • Direction: Random ±Y (up or down from pad)
   • Bounds checking: Flip direction if out of board bounds
   • DRC Pass 1: Check 0.15mm clearance from ALL OTHER PADS
     - Via position must maintain clearance
     - Stub path (sampled at 25%, 50%, 75%) must maintain clearance
     - Up to 10 random attempts per pad
   • Result: Portal landing point (x_grid, y_grid ± delta_steps)

2. ESCAPE GEOMETRY GENERATION (first pass):
   • Vertical stub: F.Cu track from pad center to portal landing point
   • Portal via: F.Cu → random horizontal layer (odd index: 1, 3, 5, ...)
     - Via spec: 0.15mm hole, 0.25mm diameter (0.05mm annular ring)
   • Entry layer: Randomly selected from In1, In3, In5, In7, In9, In11, In13, In15, B.Cu
   • All 3200 escapes generated in parallel

3. DRC CONFLICT RESOLUTION (second pass - up to 3 iterations):
   • Via-to-via checking: Ensure escapes maintain 0.4mm clearance from each other
   • Track-to-via checking: Ensure escape stubs don't violate via clearances
   • Conflict detection: Point-to-segment distance calculations
   • Retry logic: Regenerate conflicting escapes with new random parameters
   • Result: DRC-clean escape geometry for all routable pads

4. ROUTING STARTS FROM PORTAL LANDINGS:
   • Source terminal: (x_src_portal, y_src_portal, L_horizontal_i)
   • Dest terminal: (x_dst_portal, y_dst_portal, L_horizontal_j)
   • PathFinder routes between portal landing points on horizontal layers
   • No F.Cu congestion - traffic is pre-distributed across 9 layers
   • Pure Manhattan grid routing with alternating H/V discipline

5. FINAL GEOMETRY EMISSION:
   • Precomputed escape stubs (vertical F.Cu tracks)
   • Precomputed portal vias (F.Cu → entry layer)
   • Routed paths (between portal landing points)
   • Regular routing vias (layer transitions during pathfinding)

KEY ADVANTAGES:
───────────────────────────────────────────────────────────────────────────────
✓ F.Cu bottleneck eliminated: Traffic distributed before routing starts
✓ Deterministic layer spreading: Random layer selection ensures even distribution
✓ DRC-clean from the start: No escape geometry violates clearance rules
✓ Parallel precomputation: All 3200 escapes generated simultaneously
✓ Pure grid routing problem: PathFinder works on its optimal problem class
✓ Minimal via count: Only one via per pad (escape via), rest is grid routing
✓ Retry resilience: Conflicts automatically resolved through regeneration

RESULTS:
───────────────────────────────────────────────────────────────────────────────
• Before: 16% completion (73/464 nets), F.Cu saturated, inner layers idle
• After: Expected 80-90%+ completion, even layer utilization, clean geometry

═══════════════════════════════════════════════════════════════════════════════
GRAPH REPRESENTATION & DATA STRUCTURES
═══════════════════════════════════════════════════════════════════════════════

3D LATTICE:
───────────────────────────────────────────────────────────────────────────────
• Grid: (x_steps × y_steps × layers) nodes
  - x_steps, y_steps: Board dimensions ÷ grid_pitch (default 0.4mm)
  - layers: Copper layer count (6-18 typical, supports up to 32)

• Node indexing: flat_idx = layer × (x_steps × y_steps) + y × x_steps + x
  - Fast arithmetic: layer = idx ÷ plane_size
  - Enables O(1) coordinate lookups without function calls

LAYER DISCIPLINE (H/V Manhattan Routing):
───────────────────────────────────────────────────────────────────────────────
• F.Cu (L0): Vertical routing only (for portal escapes)
• Inner layers: Alternating H/V polarity
  - L1 (In1.Cu): Horizontal
  - L2 (In2.Cu): Vertical
  - L3 (In3.Cu): Horizontal
  - ... continues alternating
• B.Cu (L17): Opposite polarity of F.Cu

CSR GRAPH (Compressed Sparse Row):
───────────────────────────────────────────────────────────────────────────────
• Format: indptr[N+1], indices[E], base_costs[E]
  - indptr[i] to indptr[i+1]: edge index range for node i
  - indices[j]: destination node for edge j
  - base_costs[j]: base cost for edge j (before congestion)

• Construction (memory-efficient for 30M edges):
  - Pre-allocate numpy structured array with edge count
  - Fill array directly (no Python list intermediate)
  - Sort by source node in-place
  - Extract indices/costs components
  - Immediately free temporary arrays

EDGE TYPES:
───────────────────────────────────────────────────────────────────────────────
1. Lateral edges (H/V movement):
   • Cost: grid_pitch (0.4mm base unit)
   • Enforces Manhattan discipline per layer
   • Count: ~3M edges for 183×482×18 lattice

2. Via edges (layer transitions):
   • Constraint: Same (x,y), different layers
   • Full blind/buried: ALL layer pairs allowed
   • Cost: via_cost × (1 + span_alpha × (span-1))
   • Count: ~27M edges for full blind/buried
   • Storage: Boolean numpy array (~30MB, not Python set)

3. Portal escape edges:
   • Special via edges at terminal nodes
   • Cost: via_cost × portal_via_discount × span_cost
   • Applied only to first hop from pad terminals

EDGE ACCOUNTING (EdgeAccountant):
───────────────────────────────────────────────────────────────────────────────
• canonical: Dict[edge_idx → usage_count] - persistent ground truth
• present: Array[E] - current iteration usage (REBUILT each iteration)
• history: Array[E] - accumulated historical congestion
• total_cost: Array[E] - final cost for routing

FORMULA: total_cost[e] = base[e] + pres_fac × overuse[e] + hist_weight × history[e]

COST EVOLUTION:
Iteration 1: pres_fac=1.0   → Light penalties, natural shortest paths
Iteration 2: pres_fac=1.8   → Moderate penalties on overused edges
Iteration 7: pres_fac=34.0  → Strong penalties, forced alternatives
Iteration 11: pres_fac=357  → Extreme penalties, via annealing kicks in
Iteration 16+: pres_fac=1000 (capped) → Near-infinite cost on overuse

═══════════════════════════════════════════════════════════════════════════════
HOTSET MECHANISM (PREVENTS THRASHING)
═══════════════════════════════════════════════════════════════════════════════

PROBLEM (without hotsets):
• Re-routing ALL 464 nets every iteration takes minutes
• 90% of nets are clean, re-routing them wastes time and risks new conflicts

SOLUTION (adaptive hotsets):
• Iteration 1: Route all nets (initial solution)
• Iteration 2+: Only re-route nets that touch overused edges

HOTSET BUILDING (O(1) via edge-to-nets tracking):
───────────────────────────────────────────────────────────────────────────────
1. Find overused edges: over_idx = {e | present[e] > capacity[e]}
2. Find offending nets: offenders = ⋃(edge_to_nets[e] for e in over_idx)
3. Score by impact: impact[net] = Σ(overuse[e] for e in net_to_edges[net] ∩ over_idx)
4. Adaptive cap: min(hotset_cap, max(64, 3 × |over_idx|))
   • 26 overused edges → hotset ~78 nets (not 418)
   • 500 overused edges → hotset capped at 150

NET-TO-EDGE TRACKING:
• _net_to_edges: Dict[net_id → [edge_indices]] - cached when paths committed
• _edge_to_nets: Dict[edge_idx → {net_ids}] - reverse mapping
• Updated on: commit, clear, rip operations
• Enables O(1) hotset building instead of O(N×E) path scanning

TYPICAL EVOLUTION:
• Iter 1: Route 464 nets → 81 succeed, 514 overused edges
• Iter 2: Hotset 150 nets → 81 succeed, 275 overused edges
• Iter 7: Hotset 150 nets → 81 succeed, 143 overused edges
• Iter 12: Hotset 96 nets → 61 succeed, 29 overused edges (rip event)
• Iter 27: Hotset 64 nets → 73 succeed, 22 overused edges
• Detail pass: Hotset 8 nets, 6 iters → 0 overuse (SUCCESS)

═══════════════════════════════════════════════════════════════════════════════
PATHFINDER NEGOTIATION - ITERATION DETAIL
═══════════════════════════════════════════════════════════════════════════════

STEP 0: CLEAN ACCOUNTING (iter 2+)
  • _rebuild_usage_from_committed_nets()
  • Clear canonical and present arrays
  • Rebuild from all currently routed nets using net_to_edges cache
  • Prevents ghost usage from rip/re-route cycles

STEP 1: UPDATE COSTS (once per iteration, not per net)
  • Check via annealing policy:
    - If pres_fac ≥ 200 and via_overuse > 70%: via_cost × 1.5 (penalize vias)
    - Else if pres_fac ≥ 200: via_cost × 0.5 (encourage layer hopping)
  • Compute: total_cost[e] = base[e] + pres_fac × overuse[e] + hist_weight × history[e]
  • Costs reused for all nets in this iteration (major speedup)

STEP 2: BUILD HOTSET
  • Find overused edges using edge_to_nets
  • Adaptive cap prevents thrashing
  • Log: overuse_edges, offenders, unrouted, cap, hotset_size

STEP 3: ROUTE NETS IN HOTSET
  • For each net:
    a) Clear old path from accounting (if exists)
    b) Extract ROI: Typically 5K-50K nodes from 1.6M total
    c) Run heap-based Dijkstra on ROI: O(E_roi × log V_roi)
    d) Fallback to larger ROI if needed (max 5 per iteration)
    e) Commit path: Update canonical, present, net_to_edges, edge_to_nets

STEP 4: COMPUTE OVERUSE & METRICS
  • overuse_sum, overused_edge_count
  • via_overuse percentage (for annealing policy)
  • Every 3 iterations: Log top-10 overused channels with coordinates

STEP 5: UPDATE HISTORY
  • history[e] += hist_gain × overuse[e]
  • Prevents oscillation

STEP 6: TERMINATION & STAGNATION
  • SUCCESS: If overuse == 0 → exit
  • STAGNATION: If no improvement for 5 iterations:
    - Rip top-K offenders (k=13-20)
    - Hold pres_fac for 2 iterations
    - Grow ROI margin (+0.6mm)
  • CONTINUE: pres_fac × 1.8, next iteration

STEP 7: DETAIL REFINEMENT (after 30 iters if overuse remains)
  • Extract conflict zone (nets touching overused edges)
  • Run focused negotiation with pres_fac=500-1000
  • 10 iteration limit
  • Often achieves zero overuse on final 8-20 nets

═══════════════════════════════════════════════════════════════════════════════
ROI EXTRACTION & SHORTEST PATH SOLVING
═══════════════════════════════════════════════════════════════════════════════

ROI EXTRACTION (Region of Interest):
───────────────────────────────────────────────────────────────────────────────
• Problem: Full graph is 1.6M nodes, 30M edges - too large for per-net Dijkstra
• Solution: Extract subgraph containing only nodes near src/dst
• Method: BFS expansion from src and dst simultaneously
• Result: ROI typically 5K-50K nodes (100-1000× smaller than full graph)

ADAPTIVE ROI SIZING:
• initial_radius: 24 steps (~10mm @ 0.4mm pitch)
• Stagnation bonus: +0.6mm per stagnation event (grows when stuck)
• Fallback: If ROI fails, retry with radius=60 (limit 5 fallbacks/iteration)

SimpleDijkstra: HEAP-BASED O(E log V) SSSP
───────────────────────────────────────────────────────────────────────────────
• Priority queue: Python heapq with (distance, node) tuples
• Operates on ROI subgraph (not full graph)
• Early termination when destination reached
• Visited tracking prevents re-expansion
• Typical performance: 0.1-0.5s per net on 18-layer board

MULTI-SOURCE/MULTI-SINK (for portal routing - TO BE IMPLEMENTED):
• Initialize heap with multiple (distance, node) entries for all portal layers
• Terminate when ANY destination portal layer reached
• Choose best entry/exit layers dynamically per net

GPU SUPPORT (currently disabled):
───────────────────────────────────────────────────────────────────────────────
• config.use_gpu defaults to False
• GPU arrays available but SimpleDijkstra runs on CPU
• Avoids host↔device copy overhead without GPU SSSP kernel
• Future: GPU near-far/delta-stepping when fully vectorized

═══════════════════════════════════════════════════════════════════════════════
BLIND/BURIED VIA SUPPORT
═══════════════════════════════════════════════════════════════════════════════

VIA POLICY: ALL LAYER PAIRS ALLOWED
───────────────────────────────────────────────────────────────────────────────
• Any layer can connect to any other layer at same (x,y)
• Examples:
  - F.Cu ↔ In1.Cu (microvia)
  - In5.Cu ↔ In12.Cu (buried via)
  - F.Cu ↔ B.Cu (through via)
  - F.Cu ↔ In10.Cu (blind via)

VIA COSTING (encourages short spans but allows long):
───────────────────────────────────────────────────────────────────────────────
• Base cost: via_cost = 3.0
• Span penalty: cost = via_cost × (1 + 0.15 × (span - 1))
  - span=1 (adjacent): 3.0
  - span=5: 4.8
  - span=10: 7.05
  - span=17 (through): 10.2

• Portal discount (applied after graph build):
  - First hop from pad terminals: cost × 0.4
  - Escape via F.Cu → In1.Cu: 3.0 × 0.4 = 1.2 (cheap)
  - Makes entering grid economical, encourages immediate layer spreading

VIA EDGE REPRESENTATION:
───────────────────────────────────────────────────────────────────────────────
• Count: C(18,2) × x_steps × y_steps = 153 via pairs/cell × 88,206 cells = 27M edges
• Storage: Boolean numpy array (30MB) marks which edges are vias
• Used for: via-specific overuse tracking and annealing policy

COORDINATE SYSTEMS:
───────────────────────────────────────────────────────────────────────────────
• World: (x_mm, y_mm, layer) - Physical PCB coordinates in millimeters
• Lattice: (x_idx, y_idx, layer) - Grid indices (0..x_steps, 0..y_steps)
• Node: flat_index - Single integer for CSR: layer×(x_steps×y_steps) + y×x_steps + x

CONVERSIONS:
• world_to_lattice(): (x_mm, y_mm) → (x_idx, y_idx) via floor + clamp
• lattice_to_world(): (x_idx, y_idx) → (x_mm, y_mm) via pitch×idx + offset
• node_idx(): (x_idx, y_idx, layer) → flat_index for CSR indexing
• Arithmetic layer lookup: layer = flat_idx ÷ (x_steps × y_steps)

═══════════════════════════════════════════════════════════════════════════════
CRITICAL INVARIANTS
═══════════════════════════════════════════════════════════════════════════════

INVARIANT 1: Edge capacity = 1 per edge
• No edge sharing allowed
• Multiple nets on same edge = overuse = must resolve

INVARIANT 2: Present usage rebuilt from committed nets each iteration
• Never carry stale present_usage between iterations
• Prevents ghost usage accumulation

INVARIANT 3: Hotset contains ONLY nets touching overused edges
• Plus unrouted nets + explicitly ripped nets
• Prevents thrashing (re-routing clean nets wastes time)

INVARIANT 4: Costs updated once per iteration, before routing
• All nets in iteration see same cost landscape
• Enables fair negotiation

INVARIANT 5: Portal escape stubs are private (no congestion)
• Not in global routing graph
• Emitted directly to geometry

═══════════════════════════════════════════════════════════════════════════════
COMMON FAILURE MODES & FIXES
═══════════════════════════════════════════════════════════════════════════════

"Stuck at 81/464 routed for many iterations"
• CAUSE: All pads on F.Cu, via cost too high, router fights on top layers
• FIX: Portal escapes with discounted vias (IMPLEMENTED)

"Hotset contains 400+ nets when only 26 edges overused"
• CAUSE: Hotset not capped adaptively
• FIX: adaptive_cap = min(150, max(64, 3 × overused_edges)) (FIXED)

"Overuse jumps: 193 → 265 → 318"
• CAUSE: Ghost usage from dirty accounting
• FIX: Rebuild present from scratch each iteration (FIXED)

"MemoryError during graph construction"
• CAUSE: Python list of 30M tuples exhausts memory
• FIX: Pre-allocate numpy structured array (FIXED)

"Only F.Cu and In1.Cu show overuse, 16 layers idle"
• CAUSE: Portal escapes not implemented yet
• FIX: Portal discounts + multi-layer seeding (TO BE IMPLEMENTED)

"48 nets unmapped (dropped during parsing)"
• CAUSE: Pad key mismatch between mapping and lookup
• FIX: Consistent key generation with coordinates for orphaned pads (FIXED)

═══════════════════════════════════════════════════════════════════════════════
PERFORMANCE OPTIMIZATIONS
═══════════════════════════════════════════════════════════════════════════════

1. NO EDGE LOOKUP DICT:
   • OLD: 30M-entry Python dict (u,v) → edge_idx (~several GB)
   • NEW: On-the-fly CSR scan (degree ~4-6 in Manhattan lattice)
   • Saves: ~3GB memory + ~10s startup time

2. NUMPY VIA TRACKING:
   • OLD: Python set with 27M edge indices (~750MB)
   • NEW: Boolean array (~30MB)
   • 25× memory reduction

3. BINARY SEARCH IN LOGGING:
   • OLD: O(N) linear scan to find source node for edge
   • NEW: np.searchsorted(indptr, edge_idx) → O(log N)

4. ARITHMETIC VIA DETECTION:
   • OLD: idx_to_coord() calls for each edge
   • NEW: layer = idx ÷ plane_size (arithmetic)
   • Millions of function calls eliminated

5. HEAP-BASED DIJKSTRA:
   • OLD: O(V²) np.argmin() scan per iteration
   • NEW: O(E log V) priority queue
   • 10-100× speedup on ROI pathfinding

6. COST COMPUTED ONCE PER ITERATION:
   • OLD: ~464 full-graph cost sweeps per iteration
   • NEW: 1 cost sweep per iteration
   • Eliminated 14 billion operations per iteration

TYPICAL PERFORMANCE (18-layer backplane, 512 nets, 3200 pads):
───────────────────────────────────────────────────────────────────────────────
• Graph build: ~5-10s (with optimizations)
• Portal planning: ~1s (to be implemented)
• Iter 1 (route all 464 nets): ~2-3 minutes
• Iter 2+ (hotset 64-150 nets): ~30-60s each
• Detail pass (8 nets): ~5-10s
• Expected convergence: 15-25 iterations

MEMORY USAGE:
• CSR graph: ~360MB (30M edges × 12 bytes)
• Via tracking: ~30MB (boolean array)
• Edge accounting: ~120MB (3 float32 arrays)
• Net tracking: ~50MB (dicts)
• Total: ~600MB for 18-layer board

═══════════════════════════════════════════════════════════════════════════════
FILE ORGANIZATION
═══════════════════════════════════════════════════════════════════════════════

CLASSES (in order):
1. PathFinderConfig (line ~380): Configuration dataclass
2. CSRGraph (line ~430): Compressed sparse row graph with memory-efficient construction
3. EdgeAccountant (line ~490): Edge usage/cost accounting
4. Lattice3D (line ~550): 3D grid geometry with H/V discipline
5. ROIExtractor (line ~720): Region-of-interest extraction
6. SimpleDijkstra (line ~780): Heap-based O(E log V) shortest path solver
7. PathFinderRouter (line ~860): Main routing engine

KEY METHODS:
• initialize_graph(): Build lattice, graph, accounting structures
• route_multiple_nets(): Main entry, calls negotiation
• _pathfinder_negotiation(): Core PathFinder (30 iteration limit)
• _route_all(): Route hotset nets with ROI-based Dijkstra
• _build_hotset(): Identify nets touching overused edges (adaptive)
• _rebuild_usage_from_committed_nets(): Clean accounting
• _apply_portal_discount(): Reduce via cost at terminals

PORTAL METHODS (TO BE IMPLEMENTED):
• _plan_portal_for_pad(): Choose escape point 1.2-5mm from pad
• _get_portal_seeds(): Multi-layer entry points with discounted costs
• _route_with_portals(): Multi-source/multi-sink Dijkstra
• _emit_portal_geometry(): Vertical escape stubs + trimmed via stacks
• _retarget_failed_portals(): Adjust portals when nets fail repeatedly
• _gpu_roi_near_far_sssp_with_metrics(): GPU shortest path solver
• emit_geometry(): Converts paths to KiCad tracks/vias

═══════════════════════════════════════════════════════════════════════════════
"""

# Standard library
import logging
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

# Third-party
import numpy as np

# Local config
from .pathfinder.config import PAD_CLEARANCE_MM
from .pad_escape_planner import PadEscapePlanner, Portal

# Optional GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

# Local imports
from ...domain.models.board import Board
from .pathfinder.kicad_geometry import KiCadGeometry

# GPU pathfinding
try:
    from .pathfinder.cuda_dijkstra import CUDADijkstra
    CUDA_DIJKSTRA_AVAILABLE = True
except ImportError:
    CUDADijkstra = None
    CUDA_DIJKSTRA_AVAILABLE = False

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class GeometryPayload:
    """Wrapper for geometry with attribute access"""
    def __init__(self, tracks, vias):
        self.tracks = tracks
        self.vias = vias

@dataclass
class PathFinderConfig:
    """PathFinder algorithm parameters - LOCKED CONFIG FOR STABILITY"""
    max_iterations: int = 10  # Reduced for speed testing
    # LOCKED CONFIGURATION (DO NOT MODIFY):
    pres_fac_init: float = 1.0
    pres_fac_mult: float = 1.8
    pres_fac_max: float = 1000.0
    hist_gain: float = 2.5
    grid_pitch: float = 0.4
    via_cost: float = 3.0  # Base via cost for routing
    portal_discount: float = 0.4  # 60% discount on first escape via from terminals
    span_alpha: float = 0.15  # Span penalty: cost *= (1 + alpha*(span-1))

    # Portal escape configuration
    portal_enabled: bool = True
    portal_delta_min: int = 3      # Min vertical offset (1.2mm @ 0.4mm pitch)
    portal_delta_max: int = 12     # Max vertical offset (4.8mm)
    portal_delta_pref: int = 6     # Preferred offset (2.4mm)
    portal_x_snap_max: float = 0.5  # Max x-snap in steps (½ pitch)
    portal_via_discount: float = 0.15  # Escape via multiplier (85% discount)
    portal_retarget_patience: int = 3  # Iters before retargeting

    stagnation_patience: int = 5
    use_gpu: bool = True  # GPU algorithm fixed, validation will catch ROI construction issues
    batch_size: int = 32
    layer_count: int = 6
    strict_drc: bool = False  # Legacy compatibility
    mode: str = "near_far"
    roi_parallel: bool = False
    per_net_budget_s: float = 5.0
    max_roi_nodes: int = 100000
    delta_multiplier: float = 4.0
    adaptive_delta: bool = True
    strict_capacity: bool = True
    reroute_only_offenders: bool = True
    layer_shortfall_percentile: float = 95.0
    layer_shortfall_cap: int = 16
    enable_profiling: bool = False
    enable_instrumentation: bool = False
    strict_overuse_block: bool = False
    hist_cost_weight: float = 2.0  # Make chronic chokepoints more expensive
    log_iteration_details: bool = False
    acc_fac: float = 0.0
    phase_block_after: int = 2
    congestion_multiplier: float = 1.0
    max_search_nodes: int = 2000000
    layer_names: List[str] = field(default_factory=lambda: ['F.Cu', 'In1.Cu', 'In2.Cu', 'In3.Cu', 'In4.Cu', 'B.Cu'])
    hotset_cap: int = 150  # Guardrail to prevent mass decommits
    allowed_via_spans: Optional[Set[Tuple[int, int]]] = None  # None = all layer pairs allowed (blind/buried)


# Legacy constants
DEFAULT_GRID_PITCH = 0.4
GRID_PITCH = 0.4
LAYER_COUNT = 6


# ═══════════════════════════════════════════════════════════════════════════════
# CSR GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

class CSRGraph:
    """Compressed Sparse Row graph"""

    def __init__(self, use_gpu=False, edge_capacity=None):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.indptr = None
        self.indices = None
        self.base_costs = None

        # Pre-allocate numpy array if capacity known (memory efficient)
        if edge_capacity:
            self._edge_array = np.zeros(edge_capacity, dtype=[('src', 'i4'), ('dst', 'i4'), ('cost', 'f4')])
            self._edge_idx = 0
            self._use_array = True
        else:
            self._edges = []
            self._use_array = False

    def add_edge(self, u: int, v: int, cost: float):
        """Add directed edge"""
        if self._use_array:
            self._edge_array[self._edge_idx] = (u, v, cost)
            self._edge_idx += 1
        else:
            self._edges.append((u, v, cost))

    def finalize(self, num_nodes: int):
        """Build CSR from edge list (memory-efficient)"""
        if self._use_array:
            # Already in numpy array (pre-allocated)
            E = self._edge_idx
            edge_array = self._edge_array[:E]  # Trim to actual size
        else:
            if not self._edges:
                raise ValueError("No edges")
            E = len(self._edges)

            # Convert to numpy array for memory-efficient sorting
            edge_array = np.array(self._edges, dtype=[('src', 'i4'), ('dst', 'i4'), ('cost', 'f4')])
            # Free memory immediately
            self._edges.clear()

        # Sort by source node (in-place, memory efficient)
        edge_array.sort(order='src')

        # Extract components
        indices = edge_array['dst'].astype(np.int32)
        costs = edge_array['cost'].astype(np.float32)
        indptr = np.zeros(num_nodes + 1, dtype=np.int32)

        # Free edge array memory
        if self._use_array:
            del self._edge_array

        # Build indptr
        curr_src = -1
        for i, u in enumerate(edge_array['src']):
            while curr_src < u:
                curr_src += 1
                indptr[curr_src] = i

        while curr_src < num_nodes:
            curr_src += 1
            indptr[curr_src] = E

        if self.use_gpu:
            self.indptr = cp.asarray(indptr)
            self.indices = cp.asarray(indices)
            self.base_costs = cp.asarray(costs)
        else:
            self.indptr = indptr
            self.indices = indices
            self.base_costs = costs

        self._edges = []
        logger.info(f"CSR: {num_nodes} nodes, {E} edges")


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE ACCOUNTING
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeAccountant:
    """Edge usage tracking"""

    def __init__(self, num_edges: int, use_gpu=False):
        self.E = num_edges
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        self.canonical: Dict[int, int] = {}
        self.present = self.xp.zeros(num_edges, dtype=self.xp.float32)
        self.history = self.xp.zeros(num_edges, dtype=self.xp.float32)
        self.capacity = self.xp.ones(num_edges, dtype=self.xp.float32)
        self.total_cost = None

    def refresh_from_canonical(self):
        """Rebuild present"""
        self.present.fill(0)
        for idx, count in self.canonical.items():
            if 0 <= idx < self.E:
                self.present[idx] = float(count)

    def commit_path(self, edge_indices: List[int]):
        """Add path and keep present in sync"""
        for idx in edge_indices:
            self.canonical[idx] = self.canonical.get(idx, 0) + 1
            # Keep present in sync during iteration
            self.present[idx] = self.present[idx] + 1

    def clear_path(self, edge_indices: List[int]):
        """Remove path and keep present in sync"""
        for idx in edge_indices:
            if idx in self.canonical:
                self.canonical[idx] -= 1
                if self.canonical[idx] <= 0:
                    del self.canonical[idx]
            # Reflect in present
            self.present[idx] = self.xp.maximum(0, self.present[idx] - 1)

    def compute_overuse(self) -> Tuple[int, int]:
        """(overuse_sum, overuse_count)"""
        usage = self.present.get() if self.use_gpu else self.present
        cap = self.capacity.get() if self.use_gpu else self.capacity
        over = np.maximum(0, usage - cap)
        return int(over.sum()), int((over > 0).sum())

    def verify_present_matches_canonical(self) -> bool:
        """Sanity check: verify present usage matches canonical store"""
        recomputed = self.xp.zeros(self.E, dtype=self.xp.float32)
        for idx, count in self.canonical.items():
            if 0 <= idx < self.E:
                recomputed[idx] = float(count)

        if self.use_gpu:
            present_cpu = self.present.get()
            recomputed_cpu = recomputed.get()
        else:
            present_cpu = self.present
            recomputed_cpu = recomputed

        mismatch = np.sum(np.abs(present_cpu - recomputed_cpu))
        if mismatch > 0.01:
            logger.error(f"[ACCOUNTING] Present/canonical mismatch: {mismatch:.2f}")
            return False
        return True

    def update_history(self, gain: float, base_costs=None, history_cap_multiplier=10.0, decay_factor=0.98):
        """
        Update history with:
        - Gentle decay: history *= 0.98 before adding increment
        - Clamping: increment capped at history_cap = 10 * base_cost
        """
        # Apply gentle decay before adding new history
        self.history *= decay_factor

        over = self.xp.maximum(0, self.present - self.capacity)
        increment = gain * over

        # Clamp per-edge history increment
        if base_costs is not None:
            history_cap = history_cap_multiplier * base_costs
            increment = self.xp.minimum(increment, history_cap)

        self.history += increment

    def update_costs(self, base_costs, pres_fac: float, hist_weight: float = 1.0, add_jitter: bool = True, via_cost_multiplier: float = 1.0):
        """
        total = base * via_multiplier + pres_fac*overuse + hist_weight*history + epsilon_jitter
        Jitter breaks ties and prevents oscillation in equal-cost paths.
        Via cost multiplier enables late-stage via annealing.
        """
        over = self.xp.maximum(0, self.present - self.capacity)

        # Apply via cost multiplier to base costs (allows annealing)
        adjusted_base = base_costs * via_cost_multiplier
        self.total_cost = adjusted_base + pres_fac * over + hist_weight * self.history

        # Add per-edge epsilon jitter to break ties (stable across iterations)
        if add_jitter:
            E = len(self.total_cost)
            # Use edge index modulo prime for deterministic jitter
            jitter = self.xp.arange(E, dtype=self.xp.float32) % 9973
            jitter = jitter * 1e-6  # tiny epsilon
            self.total_cost += jitter


# ═══════════════════════════════════════════════════════════════════════════════
# 3D LATTICE
# ═══════════════════════════════════════════════════════════════════════════════

class Lattice3D:
    """3D routing lattice with H/V discipline"""

    def __init__(self, bounds: Tuple[float, float, float, float], pitch: float, layers: int):
        self.bounds = bounds
        self.pitch = pitch
        self.layers = layers

        self.geom = KiCadGeometry(bounds, pitch, layer_count=layers)
        self.x_steps = self.geom.x_steps
        self.y_steps = self.geom.y_steps
        self.num_nodes = self.x_steps * self.y_steps * layers

        self.layer_dir = self._assign_directions()
        logger.info(f"Lattice: {self.x_steps}×{self.y_steps}×{layers} = {self.num_nodes:,} nodes")

    def _assign_directions(self) -> List[str]:
        """F.Cu=V, alternating"""
        dirs = []
        for z in range(self.layers):
            if z == 0:
                dirs.append('v')
            else:
                dirs.append('h' if z % 2 == 1 else 'v')
        return dirs

    def node_idx(self, x: int, y: int, z: int) -> int:
        """(x,y,z) → flat"""
        return self.geom.node_index(x, y, z)

    def idx_to_coord(self, idx: int) -> Tuple[int, int, int]:
        """flat → (x,y,z)"""
        return self.geom.index_to_coords(idx)

    def world_to_lattice(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """mm → lattice"""
        return self.geom.world_to_lattice(x_mm, y_mm)

    def build_graph(self, via_cost: float, allowed_via_spans: Optional[Set[Tuple[int, int]]] = None, use_gpu=False) -> CSRGraph:
        """
        Build graph with H/V constraints and flexible via spans.

        Args:
            via_cost: Base cost for via transitions
            allowed_via_spans: Set of (from_layer, to_layer) pairs allowed for vias.
                              If None, all layer pairs are allowed (full blind/buried support).
                              Layers are indexed 0..N-1.
            use_gpu: Enable GPU acceleration
        """
        # Count edges to pre-allocate array (avoids MemoryError with 30M edges)
        edge_count = 0

        # Count H/V edges
        for z in range(self.layers):
            if self.layer_dir[z] == 'h':
                edge_count += 2 * self.y_steps * (self.x_steps - 1)
            else:
                edge_count += 2 * self.x_steps * (self.y_steps - 1)

        # Count via edges
        if allowed_via_spans is None:
            # Full blind/buried: all layer pairs
            via_pairs_per_xy = self.layers * (self.layers - 1) // 2
        else:
            via_pairs_per_xy = len(allowed_via_spans)

        edge_count += 2 * self.x_steps * self.y_steps * via_pairs_per_xy

        logger.info(f"Pre-allocating for {edge_count:,} edges")
        graph = CSRGraph(use_gpu, edge_capacity=edge_count)

        # Build lateral edges (H/V discipline)
        for z in range(self.layers):
            direction = self.layer_dir[z]

            if direction == 'h':
                for y in range(self.y_steps):
                    for x in range(self.x_steps - 1):
                        u = self.node_idx(x, y, z)
                        v = self.node_idx(x+1, y, z)
                        graph.add_edge(u, v, self.pitch)
                        graph.add_edge(v, u, self.pitch)
            else:
                for x in range(self.x_steps):
                    for y in range(self.y_steps - 1):
                        u = self.node_idx(x, y, z)
                        v = self.node_idx(x, y+1, z)
                        graph.add_edge(u, v, self.pitch)
                        graph.add_edge(v, u, self.pitch)

        # Build via edges with flexible layer spans
        via_count = 0
        for x in range(self.x_steps):
            for y in range(self.y_steps):
                for z_from in range(self.layers):
                    for z_to in range(z_from + 1, self.layers):
                        # Check if this via span is allowed
                        if allowed_via_spans is not None:
                            # Explicit whitelist: check both directions
                            if (z_from, z_to) not in allowed_via_spans and (z_to, z_from) not in allowed_via_spans:
                                continue

                        # Calculate via cost with span penalty: cost = base * (1 + alpha*(span-1))
                        # This allows span-1 to cost 'via_cost', but longer spans cost progressively more
                        span = z_to - z_from
                        span_alpha = 0.15  # Mild penalty for long spans
                        cost = via_cost * (1.0 + span_alpha * (span - 1))

                        u = self.node_idx(x, y, z_from)
                        v = self.node_idx(x, y, z_to)
                        graph.add_edge(u, v, cost)
                        graph.add_edge(v, u, cost)
                        via_count += 2

        logger.info(f"Vias: {via_count:,} edges ({via_count // 2} bidirectional pairs)")
        if allowed_via_spans is not None:
            logger.info(f"Via policy: {len(allowed_via_spans)} allowed layer pairs")
        else:
            logger.info(f"Via policy: all layer pairs allowed (full blind/buried)")
        return graph


# ═══════════════════════════════════════════════════════════════════════════════
# ROI EXTRACTION (GPU-Accelerated BFS)
# ═══════════════════════════════════════════════════════════════════════════════

class ROIExtractor:
    """Extract Region of Interest subgraph using GPU-vectorized BFS"""

    def __init__(self, graph: CSRGraph, use_gpu: bool = False):
        self.graph = graph
        self.xp = graph.xp
        self.N = len(graph.indptr) - 1

    def extract_roi(self, src: int, dst: int, initial_radius: int = 40, stagnation_bonus: float = 0.0) -> tuple:
        """
        Bidirectional BFS ROI extraction - expands until both src and dst are covered.
        Returns: (roi_nodes, global_to_roi)
        """
        import numpy as np
        indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
        indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices

        N = self.N
        seen = np.zeros(N, dtype=np.uint8)   # 0=unseen, 1=src-wave, 2=dst-wave, 3=both
        q_src = [src]
        q_dst = [dst]
        seen[src] = 1
        seen[dst] = 2

        depth = 0
        # Apply stagnation bonus: +0.6mm per stagnation mark (grid_pitch=0.4mm → ~1.5 steps)
        max_depth = int(initial_radius + stagnation_bonus * 1.5)
        met = False

        while depth < max_depth and (q_src or q_dst) and not met:
            def step(queue, mark):
                next_q = []
                met_flag = False
                for u in queue:
                    s, e = int(indptr[u]), int(indptr[u+1])
                    for ei in range(s, e):
                        v = int(indices[ei])
                        if seen[v] == 0:
                            seen[v] = mark
                            next_q.append(v)
                        elif seen[v] != mark:
                            # Visited by the other wave → mark as both
                            seen[v] = 3
                            met_flag = True
                return next_q, met_flag

            q_src, met_src = step(q_src, 1)
            if met_src:
                met = True
            if not met:
                q_dst, met_dst = step(q_dst, 2)
                if met_dst:
                    met = True
            depth += 1

        roi_mask = seen > 0
        roi_nodes = np.where(roi_mask)[0]

        # Cap ROI size if enormous
        max_nodes = getattr(self, "max_roi_nodes", 100_000)
        if roi_nodes.size > max_nodes:
            logger.debug(f"ROI {roi_nodes.size} > {max_nodes}, truncating")
            roi_nodes = roi_nodes[:max_nodes]

        global_to_roi = np.full(N, -1, dtype=np.int32)
        global_to_roi[roi_nodes] = np.arange(len(roi_nodes), dtype=np.int32)

        return roi_nodes, global_to_roi


# ═══════════════════════════════════════════════════════════════════════════════
# DIJKSTRA WITH ROI
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleDijkstra:
    """Dijkstra SSSP with ROI support (CPU only; copies from GPU if needed)"""

    def __init__(self, graph: CSRGraph, lattice=None):
        # Copy CSR to CPU if they live on GPU
        self.indptr = graph.indptr.get() if hasattr(graph.indptr, "get") else graph.indptr
        self.indices = graph.indices.get() if hasattr(graph.indices, "get") else graph.indices
        self.N = len(self.indptr) - 1
        # Store plane_size for layer calculation
        self.plane_size = lattice.x_steps * lattice.y_steps if lattice else None

    def find_path_roi(self, src: int, dst: int, costs, roi_nodes, global_to_roi) -> Optional[List[int]]:
        """Find shortest path within ROI subgraph using heap-based Dijkstra (O(E log V))"""
        import numpy as np
        import heapq

        # Use GPU if ROI is large enough and GPU solver available
        roi_size = len(roi_nodes) if hasattr(roi_nodes, '__len__') else roi_nodes.shape[0]
        use_gpu = hasattr(self, 'gpu_solver') and self.gpu_solver and roi_size > 5000

        if use_gpu:
            try:
                path = self.gpu_solver.find_path_roi_gpu(src, dst, costs, roi_nodes, global_to_roi)
                if path:
                    return path
                # If GPU returned None, fall through to CPU
            except Exception as e:
                logger.warning(f"[GPU] Pathfinding failed: {e}, using CPU")
                # Fall through to CPU

        # Ensure arrays are CPU NumPy
        costs = costs.get() if hasattr(costs, "get") else costs
        roi_nodes = roi_nodes.get() if hasattr(roi_nodes, "get") else roi_nodes
        global_to_roi = global_to_roi.get() if hasattr(global_to_roi, "get") else global_to_roi

        # Map src/dst to ROI space
        roi_src = int(global_to_roi[src])
        roi_dst = int(global_to_roi[dst])

        if roi_src < 0 or roi_dst < 0:
            logger.warning("src or dst not in ROI")
            return None

        roi_size = len(roi_nodes)
        dist = np.full(roi_size, np.inf, dtype=np.float32)
        parent = np.full(roi_size, -1, dtype=np.int32)
        visited = np.zeros(roi_size, dtype=bool)
        dist[roi_src] = 0.0

        # Heap-based Dijkstra: O(E log V) instead of O(V²)
        heap = [(0.0, roi_src)]

        while heap:
            du, u_roi = heapq.heappop(heap)

            # Skip if already visited (stale heap entry)
            if visited[u_roi]:
                continue

            visited[u_roi] = True

            # Early exit if we reached destination
            if u_roi == roi_dst:
                break

            u_global = int(roi_nodes[u_roi])

            s, e = int(self.indptr[u_global]), int(self.indptr[u_global + 1])
            for ei in range(s, e):
                v_global = int(self.indices[ei])
                v_roi = int(global_to_roi[v_global])

                if v_roi < 0 or visited[v_roi]:
                    continue

                alt = du + float(costs[ei])
                if alt < dist[v_roi]:
                    dist[v_roi] = alt
                    parent[v_roi] = u_roi
                    heapq.heappush(heap, (alt, v_roi))

        if not np.isfinite(dist[roi_dst]):
            return None

        # Reconstruct path in global coordinates
        path, cur = [], roi_dst
        while cur != -1:
            path.append(int(roi_nodes[cur]))
            cur = int(parent[cur])
        path.reverse()

        return path if len(path) > 1 else None

    def find_path_multisource_multisink(self, src_seeds: List[Tuple[int, float]],
                                        dst_targets: List[Tuple[int, float]],
                                        costs, roi_nodes, global_to_roi) -> Optional[Tuple[List[int], int, int]]:
        """
        Find shortest path from any source to any destination with portal entry costs.

        Returns: (path, entry_layer, exit_layer) or None
        """
        import numpy as np
        import heapq

        # Ensure arrays are CPU NumPy
        costs = costs.get() if hasattr(costs, "get") else costs
        roi_nodes = roi_nodes.get() if hasattr(roi_nodes, "get") else roi_nodes
        global_to_roi = global_to_roi.get() if hasattr(global_to_roi, "get") else global_to_roi

        roi_size = len(roi_nodes)
        dist = np.full(roi_size, np.inf, dtype=np.float32)
        parent = np.full(roi_size, -1, dtype=np.int32)
        visited = np.zeros(roi_size, dtype=bool)

        # Initialize heap with all source seeds
        heap = []
        src_roi_nodes = set()
        for global_node, initial_cost in src_seeds:
            roi_idx = int(global_to_roi[global_node])
            if roi_idx >= 0:
                dist[roi_idx] = initial_cost
                heapq.heappush(heap, (initial_cost, roi_idx))
                src_roi_nodes.add(roi_idx)

        # Build target set
        dst_roi_nodes = {}  # roi_idx -> (global_node, initial_cost)
        for global_node, initial_cost in dst_targets:
            roi_idx = int(global_to_roi[global_node])
            if roi_idx >= 0:
                dst_roi_nodes[roi_idx] = (global_node, initial_cost)

        if not heap or not dst_roi_nodes:
            return None

        # Multi-source Dijkstra
        reached_target = None
        final_dist = np.inf

        while heap:
            du, u_roi = heapq.heappop(heap)

            if visited[u_roi]:
                continue

            visited[u_roi] = True

            # Check if we reached any target
            if u_roi in dst_roi_nodes:
                target_global, target_cost = dst_roi_nodes[u_roi]
                total_dist = du + target_cost
                if total_dist < final_dist:
                    final_dist = total_dist
                    reached_target = u_roi
                    # Don't break - might find better target
                continue

            u_global = int(roi_nodes[u_roi])

            s, e = int(self.indptr[u_global]), int(self.indptr[u_global + 1])
            for ei in range(s, e):
                v_global = int(self.indices[ei])
                v_roi = int(global_to_roi[v_global])

                if v_roi < 0 or visited[v_roi]:
                    continue

                alt = du + float(costs[ei])
                if alt < dist[v_roi]:
                    dist[v_roi] = alt
                    parent[v_roi] = u_roi
                    heapq.heappush(heap, (alt, v_roi))

        if reached_target is None:
            return None

        # Reconstruct path
        path, cur = [], reached_target
        while cur != -1:
            path.append(int(roi_nodes[cur]))
            cur = int(parent[cur])
        path.reverse()

        if len(path) <= 1:
            return None

        # Determine entry and exit layers
        if self.plane_size:
            entry_layer = path[0] // self.plane_size
            exit_layer = path[-1] // self.plane_size
        else:
            # Fallback if plane_size not set
            entry_layer = exit_layer = 0

        return (path, entry_layer, exit_layer)


# ═══════════════════════════════════════════════════════════════════════════════
# PATHFINDER ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

class PathFinderRouter:
    """PathFinder negotiated congestion routing"""

    def __init__(self, config: PathFinderConfig = None, use_gpu: bool = None):
        self.config = config or PathFinderConfig()

        # Legacy API: accept use_gpu as kwarg
        if use_gpu is not None:
            self.config.use_gpu = use_gpu

        self.lattice: Optional[Lattice3D] = None
        self.graph: Optional[CSRGraph] = None
        self.accounting: Optional[EdgeAccountant] = None
        self.solver: Optional[SimpleDijkstra] = None
        self.roi_extractor: Optional[ROIExtractor] = None

        self.pad_to_node: Dict[str, int] = {}
        self.net_paths: Dict[str, List[int]] = {}
        self.iteration = 0
        self._negotiation_ran = False
        self._geometry_payload = GeometryPayload([], [])
        self._provisional_geometry = GeometryPayload([], [])  # For GUI feedback during routing

        # Hotset management: locked nets and clean streak tracking
        self.locked_nets: Set[str] = set()
        self.net_clean_streak: Dict[str, int] = defaultdict(int)  # iterations since last overuse
        self.locked_freeze_threshold: int = 3  # Lock after K clean iterations
        self.clean_nets_count: int = 0  # Track clean nets for sanity checking

        # Edge-to-nets tracking for efficient hotset building
        self._net_to_edges: Dict[str, List[int]] = {}  # net_id -> [edge_indices]
        self._edge_to_nets: Dict[int, Set[str]] = defaultdict(set)  # edge_idx -> {net_ids}

        # Portal escape tracking
        self.portals: Dict[str, Portal] = {}  # pad_id -> Portal
        self.net_portal_failures: Dict[str, int] = defaultdict(int)  # net_id -> failure count
        self.net_pad_ids: Dict[str, Tuple[str, str]] = {}  # net_id -> (src_pad_id, dst_pad_id)
        self.net_portal_layers: Dict[str, Tuple[int, int]] = {}  # net_id -> (entry_layer, exit_layer)

        # Pad escape planner (initialized after lattice is created)
        self.escape_planner: Optional[PadEscapePlanner] = None

        # ROI policy: track stagnation and fallback usage
        self.stagnation_counter: int = 0  # increments each stagnation event
        self.full_graph_fallback_count: int = 0  # limit to 5 per iteration

        # Rip tracking and pres_fac freezing (Fix 5)
        self._last_ripped: Set[str] = set()
        self._freeze_pres_fac_until: int = 0

        # Legacy attributes for compatibility
        self._instance_tag = f"PF-{int(time.time() * 1000) % 100000}"

        logger.info(f"PathFinder (GPU={self.config.use_gpu and GPU_AVAILABLE}, Portals={self.config.portal_enabled})")

    def initialize_graph(self, board: Board) -> bool:
        """Build routing graph"""
        logger.info("=" * 80)
        logger.info("PATHFINDER NEGOTIATED CONGESTION ROUTER - RUNTIME CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"[CONFIG] pres_fac_init    = {self.config.pres_fac_init}")
        logger.info(f"[CONFIG] pres_fac_mult    = {self.config.pres_fac_mult}")
        logger.info(f"[CONFIG] pres_fac_max     = {self.config.pres_fac_max}")
        logger.info(f"[CONFIG] hist_gain        = {self.config.hist_gain}")
        logger.info(f"[CONFIG] via_cost         = {self.config.via_cost}")
        logger.info(f"[CONFIG] grid_pitch       = {self.config.grid_pitch} mm")
        logger.info(f"[CONFIG] max_iterations   = {self.config.max_iterations}")
        logger.info(f"[CONFIG] stagnation_patience = {self.config.stagnation_patience}")
        logger.info("=" * 80)

        bounds = self._calc_bounds(board)

        # Use board's real layer count (critical for dense boards)
        layers_from_board = getattr(board, "layer_count", None) or len(getattr(board, "layers", [])) or self.config.layer_count
        self.config.layer_count = int(layers_from_board)

        # Ensure we have enough layer names
        if not getattr(self.config, "layer_names", None) or len(self.config.layer_names) < self.config.layer_count:
            self.config.layer_names = (
                getattr(board, "layers", None)
                or (["F.Cu"] + [f"In{i}.Cu" for i in range(1, self.config.layer_count-1)] + ["B.Cu"])
            )

        logger.info(f"Using {self.config.layer_count} layers from board")

        self.lattice = Lattice3D(bounds, self.config.grid_pitch, self.config.layer_count)

        self.graph = self.lattice.build_graph(
            self.config.via_cost,
            allowed_via_spans=self.config.allowed_via_spans,
            use_gpu=self.config.use_gpu and GPU_AVAILABLE
        )
        self.graph.finalize(self.lattice.num_nodes)

        E = len(self.graph.indices)
        self.accounting = EdgeAccountant(E, use_gpu=self.config.use_gpu and GPU_AVAILABLE)

        self.solver = SimpleDijkstra(self.graph, self.lattice)

        # Add GPU solver if available
        use_gpu_solver = self.config.use_gpu and GPU_AVAILABLE and CUDA_DIJKSTRA_AVAILABLE
        if use_gpu_solver:
            try:
                self.solver.gpu_solver = CUDADijkstra(self.graph)
                logger.info("[GPU] CUDA Near-Far Dijkstra enabled (ROI > 5K nodes)")
            except Exception as e:
                logger.warning(f"[GPU] Failed to initialize CUDA Dijkstra: {e}")
                self.solver.gpu_solver = None
        else:
            self.solver.gpu_solver = None
        self.roi_extractor = ROIExtractor(self.graph, use_gpu=self.config.use_gpu and GPU_AVAILABLE)

        # Identify via edges for via-specific accounting
        self._identify_via_edges()

        self._map_pads(board)

        # Initialize escape planner after pads are mapped
        self.escape_planner = PadEscapePlanner(self.lattice, self.config, self.pad_to_node)

        # Plan portal escape points for each pad
        self._plan_portals(board)

        # Note: Portal discounts are applied at seed level in _get_portal_seeds()
        # No need for graph-level discount modification

        logger.info("=== Init complete ===")
        return True

    def _calc_bounds(self, board: Board) -> Tuple[float, float, float, float]:
        """Board bbox - prefer KiCad-provided bounds"""
        if hasattr(board, "_kicad_bounds"):
            x0, y0, x1, y1 = board._kicad_bounds
            return (x0, y0, x1, y1)

        pads = []
        for comp in board.components:
            pads.extend(comp.pads)

        if not pads:
            return (0, 0, 100, 100)

        # Pads have position.x and position.y
        xs = [p.position.x for p in pads]
        ys = [p.position.y for p in pads]

        margin = 3.0
        return (min(xs)-margin, min(ys)-margin, max(xs)+margin, max(ys)+margin)

    def _pad_key(self, pad, comp=None):
        """Generate unique pad key with coordinates for orphaned pads"""
        comp_id = getattr(pad, "component_id", None) or (getattr(comp, "id", None) if comp else None) or "GENERIC_COMPONENT"

        # For orphaned pads (all in GENERIC_COMPONENT), include coordinates to ensure uniqueness
        # since pad IDs like "1", "2", "3" will collide across multiple components
        if comp_id == "GENERIC_COMPONENT" and hasattr(pad, 'position'):
            xq = int(round(pad.position.x * 1000))
            yq = int(round(pad.position.y * 1000))
            return f"{comp_id}_{pad.id}@{xq},{yq}"

        return f"{comp_id}_{pad.id}"

    def _get_pad_layer(self, pad) -> int:
        """Get the layer index for a pad with fallback handling"""
        # Check if pad has explicit layer information
        if hasattr(pad, 'layer') and pad.layer:
            layer_name = str(pad.layer)
            if layer_name in self.config.layer_names:
                return self.config.layer_names.index(layer_name)
            logger.debug(f"Pad layer '{layer_name}' not in layer_names, using fallback")

        # Check if pad has layers list (multi-layer pads)
        if hasattr(pad, 'layers') and pad.layers:
            # Use first layer in the list
            layer_name = str(pad.layers[0])
            if layer_name in self.config.layer_names:
                return self.config.layer_names.index(layer_name)
            logger.debug(f"Pad layers[0] '{layer_name}' not in layer_names, using fallback")

        # Check drill attribute to determine if through-hole
        drill = getattr(pad, 'drill', 0.0)
        if drill > 0:
            # Through-hole pad - default to F.Cu (layer 0)
            return 0  # F.Cu

        # Default to F.Cu for SMD pads
        return 0

    def _map_pads(self, board: Board):
        """Map every pad to a lattice node with unique keys."""
        count_components = 0
        count_board_level = 0
        sample_ids = []
        oob_count = 0
        layer_fallback_count = 0

        def _snap_to_node(x_mm, y_mm, layer=0):
            x_idx, y_idx = self.lattice.world_to_lattice(x_mm, y_mm)
            # Clamp to valid range (prevents OOB)
            x_idx = max(0, min(x_idx, self.lattice.x_steps - 1))
            y_idx = max(0, min(y_idx, self.lattice.y_steps - 1))
            return self.lattice.node_idx(x_idx, y_idx, layer)

        # Pads that come via components - keep on physical layers
        for comp in getattr(board, "components", []):
            for pad in getattr(comp, "pads", []):
                pad_id = self._pad_key(pad, comp)
                layer = self._get_pad_layer(pad)
                node = _snap_to_node(pad.position.x, pad.position.y, layer)
                self.pad_to_node[pad_id] = node
                count_components += 1
                if len(sample_ids) < 5:
                    sample_ids.append(pad_id)

        # Pads that might live at board level (GUI created "generic component")
        for pad in getattr(board, "pads", []):
            pad_id = self._pad_key(pad, comp=None)
            if pad_id not in self.pad_to_node:
                layer = self._get_pad_layer(pad)
                node = _snap_to_node(pad.position.x, pad.position.y, layer)
                self.pad_to_node[pad_id] = node
                count_board_level += 1

        logger.info(f"Mapped {len(self.pad_to_node)} pads (from ~{count_components + count_board_level})")
        logger.info(f"[VERIFY] Sample pad IDs: {sample_ids[:5]}")

    def _plan_portals(self, board: Board):
        """Plan portal escape points for all pads"""
        if not self.config.portal_enabled:
            logger.info("Portal escapes disabled")
            return

        portal_count = 0
        tht_skipped = 0

        # Plan portals for component pads
        for comp in getattr(board, "components", []):
            for pad in getattr(comp, "pads", []):
                # Skip through-hole pads (they already span all layers)
                drill = getattr(pad, 'drill', 0.0)
                if drill > 0:
                    tht_skipped += 1
                    continue

                pad_id = self._pad_key(pad, comp)
                if pad_id in self.pad_to_node:
                    portal = self._plan_portal_for_pad(pad, pad_id)
                    if portal:
                        self.portals[pad_id] = portal
                        portal_count += 1

        # Plan portals for board-level pads
        for pad in getattr(board, "pads", []):
            drill = getattr(pad, 'drill', 0.0)
            if drill > 0:
                tht_skipped += 1
                continue

            pad_id = self._pad_key(pad, comp=None)
            if pad_id in self.pad_to_node and pad_id not in self.portals:
                portal = self._plan_portal_for_pad(pad, pad_id)
                if portal:
                    self.portals[pad_id] = portal
                    portal_count += 1

        logger.info(f"Planned {portal_count} portals (skipped {tht_skipped} THT pads)")

    def _plan_portal_for_pad(self, pad, pad_id: str) -> Optional[Portal]:
        """Plan portal escape point for a single pad"""
        # Get pad position and layer
        pad_x, pad_y = pad.position.x, pad.position.y
        pad_layer = self._get_pad_layer(pad)

        # Snap pad x to nearest lattice column (within ½ pitch)
        x_idx_nearest, _ = self.lattice.world_to_lattice(pad_x, pad_y)
        x_idx_nearest = max(0, min(x_idx_nearest, self.lattice.x_steps - 1))

        # Check if snap is within tolerance
        x_mm_snapped, _ = self.lattice.geom.lattice_to_world(x_idx_nearest, 0)
        x_snap_dist_steps = abs(pad_x - x_mm_snapped) / self.config.grid_pitch

        if x_snap_dist_steps > self.config.portal_x_snap_max:
            logger.debug(f"Pad {pad_id}: x-snap {x_snap_dist_steps:.2f} exceeds max {self.config.portal_x_snap_max}")
            return None

        x_idx = x_idx_nearest

        # Get pad y index
        _, y_idx_pad = self.lattice.world_to_lattice(pad_x, pad_y)
        y_idx_pad = max(0, min(y_idx_pad, self.lattice.y_steps - 1))

        # Score all candidate portal offsets
        candidates = []
        cfg = self.config

        for delta_steps in range(cfg.portal_delta_min, cfg.portal_delta_max + 1):
            for direction in [+1, -1]:
                y_idx_portal = y_idx_pad + direction * delta_steps

                # Check bounds
                if y_idx_portal < 0 or y_idx_portal >= self.lattice.y_steps:
                    continue

                # Score this candidate
                # Component 1: Delta preference (prefer portal_delta_pref)
                delta_penalty = abs(delta_steps - cfg.portal_delta_pref)

                # Component 2: X-snap penalty
                x_snap_penalty = x_snap_dist_steps * 2.0  # Weight x-snap errors

                # Component 3: Congestion avoidance (sample history at portal location)
                # (Skip for now - history not populated yet at init time)
                congestion_penalty = 0.0

                total_score = delta_penalty + x_snap_penalty + congestion_penalty

                candidates.append((total_score, x_idx, y_idx_portal, delta_steps, direction))

        if not candidates:
            return None

        # Pick best candidate (lowest score)
        score, x_idx, y_idx, delta, direction = min(candidates)

        return Portal(
            x_idx=x_idx,
            y_idx=y_idx,
            pad_layer=pad_layer,
            delta_steps=delta,
            direction=direction,
            pad_x=pad_x,
            pad_y=pad_y,
            score=score,
            retarget_count=0
        )

    def _get_portal_seeds(self, portal: Portal) -> List[Tuple[int, float]]:
        """Get multi-layer entry points at portal with discounted costs"""
        seeds = []
        cfg = self.config

        for layer in range(self.lattice.layers):
            # Get node index at portal for this layer
            node_idx = self.lattice.node_idx(portal.x_idx, portal.y_idx, layer)

            # Calculate via cost from entry_layer (where escape via connects)
            if layer == portal.entry_layer:
                # Same as entry layer - no via needed, minimal cost
                cost = 0.1
            else:
                # Via cost from entry_layer to this layer
                span = abs(layer - portal.entry_layer)
                base_via_cost = cfg.via_cost * (1.0 + cfg.span_alpha * (span - 1))
                # Portal discount makes entry-layer vias cheap
                cost = base_via_cost * cfg.portal_via_discount

            seeds.append((node_idx, cost))

        return seeds

    def route_multiple_nets(self, requests: List, progress_cb=None, iteration_cb=None) -> Dict:
        """
        Main entry for routing multiple nets.

        Args:
            requests: List of net routing requests
            progress_cb: Callback(done, total, eta) called after each net routed
            iteration_cb: Callback(iteration, tracks, vias) called after each pathfinder iteration
                         Used for screenshot capture and progress visualization

        Returns:
            Dict of routing results
        """
        logger.info(f"=== Route {len(requests)} nets ===")

        tasks = self._parse_requests(requests)

        if not tasks:
            self._negotiation_ran = True
            return {}

        result = self._pathfinder_negotiation(tasks, progress_cb, iteration_cb)

        return result

    def _parse_requests(self, requests: List) -> Dict[str, Tuple[int, int]]:
        """Parse to (net: (src, dst))"""
        tasks = {}

        # Track why nets are dropped
        unmapped_pads = 0
        same_cell_trivial = 0
        kept = 0

        for req in requests:
            if hasattr(req, 'name') and hasattr(req, 'pads'):
                net_name = req.name
                pads = req.pads

                if len(pads) < 2:
                    continue

                p1, p2 = pads[0], pads[1]

                # Use same key scheme as mapping
                p1_id = self._pad_key(p1)
                p2_id = self._pad_key(p2)

                if p1_id in self.pad_to_node and p2_id in self.pad_to_node:
                    # Route from PORTAL positions (escape vias), not from pads
                    # Portals are pre-computed and stored in self.portals
                    if p1_id in self.portals and p2_id in self.portals:
                        p1_portal = self.portals[p1_id]
                        p2_portal = self.portals[p2_id]

                        # Use the entry_layer stored in each portal (chosen during escape planning)
                        # This ensures routing starts from the actual escape via layer
                        entry_layer = p1_portal.entry_layer
                        exit_layer = p2_portal.entry_layer

                        # Convert portal positions to node indices
                        src = self.lattice.node_idx(p1_portal.x_idx, p1_portal.y_idx, entry_layer)
                        dst = self.lattice.node_idx(p2_portal.x_idx, p2_portal.y_idx, exit_layer)

                        if src != dst:
                            tasks[net_name] = (src, dst)
                            self.net_pad_ids[net_name] = (p1_id, p2_id)  # Track pad IDs
                            self.net_portal_layers[net_name] = (entry_layer, exit_layer)  # Track layers
                            kept += 1
                        else:
                            same_cell_trivial += 1
                            self.net_paths[net_name] = [src]
                            logger.debug(f"Net {net_name}: trivial route (portals at same position)")
                    else:
                        # Portals not found - skip this net
                        if p1_id not in self.portals:
                            logger.debug(f"Net {net_name}: no portal for {p1_id}")
                        if p2_id not in self.portals:
                            logger.debug(f"Net {net_name}: no portal for {p2_id}")
                        unmapped_pads += 1
                else:
                    unmapped_pads += 1
                    if unmapped_pads <= 3:  # Log first 3 examples
                        logger.warning(f"Net {net_name}: pads {p1_id}, {p2_id} not in pad_to_node")
                        logger.warning(f"  Available keys sample: {list(self.pad_to_node.keys())[:5]}")
                        logger.warning(f"  p1 attrs: {dir(p1)[:10] if hasattr(p1, '__dir__') else 'N/A'}")
                    else:
                        logger.debug(f"Net {net_name}: pads {p1_id}, {p2_id} not in pad_to_node")

            elif isinstance(req, tuple) and len(req) == 3:
                net_name, src, dst = req
                if isinstance(src, int) and isinstance(dst, int):
                    if src != dst:
                        tasks[net_name] = (src, dst)
                        kept += 1
                    else:
                        same_cell_trivial += 1
                        # Mark as trivially routed
                        self.net_paths[net_name] = [src]

        routed_trivial = same_cell_trivial
        dropped = unmapped_pads
        logger.info(f"[VERIFY] Parsed {len(tasks)} tasks from {len(requests)} requests")
        logger.info(f"[VERIFY]   routable={kept}, trivial={routed_trivial}, unmapped={unmapped_pads}, total_handled={kept+routed_trivial}")

        return tasks

    def _pathfinder_negotiation(self, tasks: Dict[str, Tuple[int, int]], progress_cb=None, iteration_cb=None) -> Dict:
        """CORE PATHFINDER ALGORITHM"""
        cfg = self.config
        pres_fac = cfg.pres_fac_init
        best_overuse = float('inf')
        stagnant = 0

        self._negotiation_ran = True

        logger.info(f"[NEGOTIATE] {len(tasks)} nets, {cfg.max_iterations} iters")

        for it in range(1, cfg.max_iterations + 1):
            self.iteration = it
            logger.info(f"[ITER {it}] pres_fac={pres_fac:.2f}")

            # STEP 0: Clean accounting rebuild (iter 2+)
            if it > 1 and cfg.reroute_only_offenders:
                # Rebuild usage from all currently routed nets before building hotset
                committed_nets = {nid for nid, path in self.net_paths.items() if path}
                self._rebuild_usage_from_committed_nets(committed_nets)
            else:
                # STEP 1: Refresh (iter 1 only, or if not using hotsets)
                self.accounting.refresh_from_canonical()

            # STEP 2: Update costs (with history weight and via annealing)
            # Late-stage via policy: anneal via cost when pres_fac >= 200
            via_cost_mult = 1.0
            if pres_fac >= 200:
                # Check if >70% of overuse is on vias
                present = self.accounting.present.get() if self.accounting.use_gpu else self.accounting.present
                cap = self.accounting.capacity.get() if self.accounting.use_gpu else self.accounting.capacity
                over = np.maximum(0, present - cap)

                # Use numpy boolean indexing for efficient via overuse calculation
                via_overuse = float(over[self._via_edges[:len(over)]].sum())
                total_overuse = float(over.sum())

                if total_overuse > 0:
                    via_ratio = via_overuse / total_overuse
                    if via_ratio > 0.7:
                        # Most overuse is on vias: increase via cost to widen horizontal corridors
                        via_cost_mult = 1.5
                        logger.info(f"[VIA POLICY] {via_ratio*100:.1f}% via overuse → increasing via cost by 1.5x")
                    else:
                        # Normal case: reduce via cost to enable layer hopping
                        via_cost_mult = 0.5
                        logger.info(f"[VIA POLICY] Late-stage annealing: via_cost *= 0.5")

            self.accounting.update_costs(
                self.graph.base_costs, pres_fac, cfg.hist_cost_weight,
                via_cost_multiplier=via_cost_mult
            )

            # STEP 3: Route (hotset incremental after iter 1)
            if cfg.reroute_only_offenders and it > 1:
                # Pass ripped set to _build_hotset (Fix 2)
                offenders = self._build_hotset(tasks, ripped=getattr(self, "_last_ripped", set()))
                sub_tasks = {k: v for k, v in tasks.items() if k in offenders}
                logger.info(f"  Hotset: {len(offenders)}/{len(tasks)} nets")
                # Clear _last_ripped after use
                self._last_ripped = set()
            else:
                sub_tasks = tasks

            routed, failed = self._route_all(sub_tasks, all_tasks=tasks, pres_fac=pres_fac)

            # CRITICAL: Refresh present to reflect committed paths
            self.accounting.refresh_from_canonical()

            # ACCOUNTING SANITY CHECK: Verify present matches canonical
            if not self.accounting.verify_present_matches_canonical():
                logger.warning(f"[ITER {it}] Accounting mismatch detected - potential bug")

            # STEP 4: Overuse
            over_sum, over_cnt = self.accounting.compute_overuse()

            # Instrumentation: via overuse ratio
            present = self.accounting.present.get() if self.accounting.use_gpu else self.accounting.present
            cap = self.accounting.capacity.get() if self.accounting.use_gpu else self.accounting.capacity
            over = np.maximum(0, present - cap)
            # Use numpy boolean indexing for efficient via overuse calculation
            via_overuse = float(over[self._via_edges[:len(over)]].sum())
            via_ratio = (via_overuse / over_sum * 100) if over_sum > 0 else 0.0

            logger.info(f"[ITER {it}] routed={routed} failed={failed} overuse={over_sum} edges={over_cnt} via_overuse={via_ratio:.1f}%")

            # Instrumentation: Top-10 overused channels
            if over_sum > 0 and it % 3 == 0:  # Every 3 iterations
                self._log_top_overused_channels(over, top_k=10)

            # Clean-phase: if overuse==0, freeze good nets and finish stragglers
            if over_sum == 0:
                unrouted = {nid for nid in tasks.keys() if not self.net_paths.get(nid)}
                if not unrouted:
                    logger.info("[CLEAN] All nets routed with zero overuse")
                    break
                # Freeze placed nets and lower pressure for stragglers
                placed = {nid for nid in tasks.keys() if self.net_paths.get(nid)}
                pres_fac = min(pres_fac, 10.0)
                logger.info(f"[CLEAN] Overuse=0, {len(unrouted)} unrouted left → freeze {len(placed)} nets, pres_fac={pres_fac:.2f}")

            # STEP 5: History (with decay and clamping)
            self.accounting.update_history(
                cfg.hist_gain,
                base_costs=self.graph.base_costs,
                history_cap_multiplier=10.0,
                decay_factor=0.98
            )

            # Progress callback
            if progress_cb:
                try:
                    progress_cb(it, cfg.max_iterations, f"Iteration {it}")
                except:
                    pass

            # Iteration callback for screenshots (generate provisional geometry)
            if iteration_cb:
                try:
                    # Generate current routing state for visualization
                    provisional_tracks, provisional_vias = self._generate_geometry_from_paths()
                    iteration_cb(it, provisional_tracks, provisional_vias)
                except Exception as e:
                    logger.warning(f"[ITER {it}] Iteration callback failed: {e}")

            # STEP 6: Terminate?
            if failed == 0 and over_sum == 0:
                logger.info("[SUCCESS] Zero overuse")
                return {'success': True, 'paths': self.net_paths}

            if over_sum < best_overuse:
                best_overuse = over_sum
                stagnant = 0
            else:
                stagnant += 1

            if stagnant >= cfg.stagnation_patience:
                self.stagnation_counter += 1  # Track cumulative stagnation events
                victims = self._rip_top_k_offenders(k=20)  # Only rip 16-24 worst nets
                self._last_ripped = victims  # Store for next hotset build (Fix 2)
                # Freeze pres_fac for next 2 iterations to let smaller hotset settle (Fix 4)
                self._freeze_pres_fac_until = it + 2
                logger.warning(f"[STAGNATION {self.stagnation_counter}] Ripped {len(victims)} nets, "
                              f"holding pres_fac for 2 iters, ROI margin now +{self.stagnation_counter*0.6:.1f}mm")
                stagnant = 0
                continue

            # STEP 7: Escalate (but respect pres_fac freeze after rip - Fix 4)
            if it <= getattr(self, "_freeze_pres_fac_until", 0):
                logger.debug(f"[ITER {it}] Holding pres_fac={pres_fac:.2f} post-rip")
            else:
                pres_fac = min(pres_fac * cfg.pres_fac_mult, cfg.pres_fac_max)

        # If we exited with low overuse (<100), run detail pass
        if 0 < over_sum <= 100:
            logger.info(f"[DETAIL PASS] Overuse={over_sum} at max_iters, running detail refinement")
            detail_result = self._detail_pass(tasks, over_sum, over_cnt)
            if detail_result['success']:
                return detail_result

        return {
            'success': False,
            'error_code': 'ROUTING-FAILED',
            'message': f'{failed} unrouted, {over_cnt} overused',
            'overuse_sum': over_sum,
            'overuse_edges': over_cnt,
            'failed_nets': failed
        }

    def _detail_pass(self, tasks: Dict[str, Tuple[int, int]], initial_overuse: int, initial_edges: int) -> Dict:
        """
        Detail pass: extract conflict subgraph and route only affected nets
        with fine ROI, lower via cost, higher history gain, and max 60 hotset.
        """
        logger.info("[DETAIL] Extracting conflict subgraph...")

        present = self.accounting.present.get() if self.accounting.use_gpu else self.accounting.present
        cap = self.accounting.capacity.get() if self.accounting.use_gpu else self.accounting.capacity
        over = np.maximum(0, present - cap)

        # Find overused edges and their neighborhoods (radius ~5-10 edges)
        conflict_edges = set(int(ei) for ei in range(len(over)) if over[ei] > 0)

        # Collect nets that use any conflict edge
        conflict_nets = set()
        for net_id, path in self.net_paths.items():
            if path and any(ei in conflict_edges for ei in self._path_to_edges(path)):
                conflict_nets.add(net_id)

        logger.info(f"[DETAIL] Found {len(conflict_nets)} nets in conflict zone")

        if not conflict_nets:
            return {'success': False, 'error_code': 'NO-CONFLICT-NETS'}

        # Build subset of tasks for conflict nets
        conflict_tasks = {nid: tasks[nid] for nid in conflict_nets if nid in tasks}

        # Detail loop: max 10 iterations with aggressive settings
        cfg = self.config
        pres_fac = cfg.pres_fac_max * 0.5  # Start high
        best_overuse = initial_overuse

        for detail_it in range(1, 11):
            logger.info(f"[DETAIL {detail_it}/10] pres_fac={pres_fac:.1f}")

            self.accounting.refresh_from_canonical()

            # Update costs with lower via cost and higher history gain
            via_cost_mult = 0.3  # Much lower via cost for detail pass
            self.accounting.update_costs(
                self.graph.base_costs, pres_fac, cfg.hist_cost_weight * 1.5,
                via_cost_multiplier=via_cost_mult
            )

            # Build hotset (capped at 60 for detail pass)
            detail_hotset = self._build_hotset(conflict_tasks)
            detail_hotset = set(list(detail_hotset)[:60])  # Hard cap at 60

            if not detail_hotset:
                detail_hotset = conflict_nets  # Route all if hotset empty

            detail_sub_tasks = {k: v for k, v in conflict_tasks.items() if k in detail_hotset}

            # Route with wider ROI (stagnation bonus for fine search)
            old_stagnation = self.stagnation_counter
            self.stagnation_counter += 3  # Temporarily increase for wider ROI
            routed, failed = self._route_all(detail_sub_tasks, all_tasks=tasks, pres_fac=pres_fac)
            self.stagnation_counter = old_stagnation

            self.accounting.refresh_from_canonical()
            over_sum, over_cnt = self.accounting.compute_overuse()

            logger.info(f"[DETAIL {detail_it}/10] overuse={over_sum} edges={over_cnt}")

            if over_sum == 0:
                logger.info("[DETAIL] SUCCESS: Zero overuse achieved")
                return {'success': True, 'paths': self.net_paths}

            if over_sum < best_overuse:
                best_overuse = over_sum
            else:
                # No improvement: escalate and continue
                pass

            # Update history for conflict edges only
            self.accounting.update_history(
                cfg.hist_gain * 2.0,  # Double history gain in detail pass
                base_costs=self.graph.base_costs,
                history_cap_multiplier=15.0,
                decay_factor=1.0  # No decay in detail pass
            )

            pres_fac = min(pres_fac * 1.5, cfg.pres_fac_max)

        # Detail pass exhausted
        logger.warning(f"[DETAIL] Failed to reach zero: final overuse={best_overuse}")
        return {'success': False, 'error_code': 'DETAIL-INCOMPLETE', 'overuse_sum': best_overuse}

    def _order_nets_by_difficulty(self, tasks: Dict[str, Tuple[int, int]]) -> List[str]:
        """
        Order nets by difficulty score = distance * (pin_degree + 1) * (congestion + 1).
        Route hardest first. Apply slight shuffle each iteration.
        """
        import random
        present = self.accounting.present.get() if self.accounting.use_gpu else self.accounting.present
        cap = self.accounting.capacity.get() if self.accounting.use_gpu else self.accounting.capacity
        over = np.maximum(0, present - cap)

        scores = []
        for net_id, (src, dst) in tasks.items():
            # Distance estimate (Manhattan in lattice space)
            sx, sy, sz = self.lattice.idx_to_coord(src)
            dx, dy, dz = self.lattice.idx_to_coord(dst)
            distance = abs(dx - sx) + abs(dy - sy) + abs(dz - sz)

            # Pin degree (for point-to-point, degree=2; could be extended for multi-pin)
            pin_degree = 2

            # Congestion: average overuse along prior path (if exists)
            congestion = 0.0
            if net_id in self._net_to_edges:
                # Use cached edge mapping (O(1) lookup) instead of recomputing (O(M) scan)
                edges = self._net_to_edges[net_id]
                congestion = sum(float(over[ei]) for ei in edges) / max(1, len(edges))

            difficulty = distance * (pin_degree + 1) * (congestion + 1)
            scores.append((difficulty, net_id))

        scores.sort(reverse=True)  # hardest first

        # Apply slight shuffle: rotate order by small random amount
        rotation = random.randint(0, min(5, len(scores) // 10))
        ordered = [nid for _, nid in scores]
        if rotation > 0:
            ordered = ordered[rotation:] + ordered[:rotation]

        return ordered

    def _route_all(self, tasks: Dict[str, Tuple[int, int]], all_tasks: Dict[str, Tuple[int, int]] = None, pres_fac: float = 1.0) -> Tuple[int, int]:
        """Route nets with adaptive ROI extraction and intra-iteration cost updates"""
        if all_tasks is None:
            all_tasks = tasks

        routed_this_pass = 0
        failed_this_pass = 0
        total = len(tasks)
        cfg = self.config

        # Reset full-graph fallback counter at start of iteration
        self.full_graph_fallback_count = 0

        # ROI margin grows with stagnation: +0.6mm per stagnation mark
        roi_margin_bonus = self.stagnation_counter * 0.6

        # Order nets by difficulty: hardest first, with slight shuffle per iteration
        ordered_nets = self._order_nets_by_difficulty(tasks)

        # Compute costs once per iteration (not per net) - major performance win
        # Note: via_cost_multiplier is already baked into base costs at iteration start
        self.accounting.update_costs(self.graph.base_costs, pres_fac, cfg.hist_cost_weight)
        costs = self.accounting.total_cost.get() if self.accounting.use_gpu else self.accounting.total_cost

        # GPU Batching: Route multiple nets in parallel if GPU available
        use_gpu_batching = (hasattr(self.solver, 'gpu_solver') and
                           self.solver.gpu_solver is not None and
                           total > 8)  # Only batch if enough nets

        if use_gpu_batching:
            logger.info(f"[GPU-BATCH] Routing {total} nets with batch_size={cfg.batch_size}")
            return self._route_all_batched_gpu(ordered_nets, tasks, all_tasks, costs, pres_fac, roi_margin_bonus)

        # Fallback: Sequential routing (CPU or small batches)
        for idx, net_id in enumerate(ordered_nets):
            src, dst = tasks[net_id]
            if idx % 50 == 0 and total > 50:
                logger.info(f"  Routing net {idx+1}/{total}")

            # Only clear if we're actually re-routing this net
            if net_id in self.net_paths and self.net_paths[net_id]:
                # Use cached edges if available, otherwise compute
                if net_id in self._net_to_edges:
                    old_edges = self._net_to_edges[net_id]
                else:
                    old_edges = self._path_to_edges(self.net_paths[net_id])
                self.accounting.clear_path(old_edges)
                # Clear old tracking before re-routing
                self._clear_net_edge_tracking(net_id)

            # Calculate adaptive ROI radius based on net length
            src_x, src_y, src_z = self.lattice.idx_to_coord(src)
            dst_x, dst_y, dst_z = self.lattice.idx_to_coord(dst)
            manhattan_dist = abs(dst_x - src_x) + abs(dst_y - src_y)

            # Adaptive radius: 120% of Manhattan distance
            adaptive_radius = max(30, min(int(manhattan_dist * 1.2), 150))

            # Check if we have portals for this net
            use_portals = cfg.portal_enabled and net_id in self.net_pad_ids
            src_seeds = []
            dst_targets = []

            if use_portals:
                src_pad_id, dst_pad_id = self.net_pad_ids[net_id]
                src_portal = self.portals.get(src_pad_id)
                dst_portal = self.portals.get(dst_pad_id)

                if src_portal and dst_portal:
                    # Get multi-layer seeds at both portals
                    src_seeds = self._get_portal_seeds(src_portal)
                    dst_targets = self._get_portal_seeds(dst_portal)

                    # Extract ROI centered on PORTAL nodes on their entry layers (not pad layer!)
                    src_portal_node = self.lattice.node_idx(src_portal.x_idx, src_portal.y_idx, src_portal.entry_layer)
                    dst_portal_node = self.lattice.node_idx(dst_portal.x_idx, dst_portal.y_idx, dst_portal.entry_layer)

                    roi_nodes, global_to_roi = self.roi_extractor.extract_roi(
                        src_portal_node, dst_portal_node, initial_radius=adaptive_radius, stagnation_bonus=roi_margin_bonus
                    )
                else:
                    use_portals = False

            # If not using portals, extract ROI around pad nodes with adaptive sizing
            if not use_portals:
                roi_nodes, global_to_roi = self.roi_extractor.extract_roi(
                    src, dst, initial_radius=adaptive_radius, stagnation_bonus=roi_margin_bonus
                )

            # Log ROI sizes periodically
            if idx % 50 == 0:
                logger.debug(f"  ROI size={len(roi_nodes)} for net {net_id} (margin_bonus={roi_margin_bonus:.1f}mm)")

            # Debug first net
            if idx == 0:
                logger.info(f"[DEBUG] First net: portal_enabled={cfg.portal_enabled}, net_id={net_id}, use_portals={use_portals}")
                if use_portals:
                    logger.info(f"[DEBUG]   src_seeds count={len(src_seeds)}, dst_targets count={len(dst_targets)}")
                    logger.info(f"[DEBUG]   ROI size={len(roi_nodes)}")

            path = None
            entry_layer = exit_layer = None

            if use_portals:
                # Route with multi-source/multi-sink using portal seeds
                result = self.solver.find_path_multisource_multisink(
                    src_seeds, dst_targets, costs, roi_nodes, global_to_roi
                )
                if result:
                    path, entry_layer, exit_layer = result

            # Fallback to normal routing if portals not available or failed
            if not use_portals or not path:
                if idx == 0 and use_portals:
                    logger.info(f"[DEBUG] Portal routing failed, falling back to normal routing")
                path = self.solver.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)

            # If ROI fails and we haven't exhausted fallback quota, try larger ROI
            if (not path or len(path) <= 1) and self.full_graph_fallback_count < 5:
                # Fallback: 1.5× adaptive radius (capped at 200)
                fallback_radius = min(int(adaptive_radius * 1.5), 200)
                logger.debug(f"  ROI failed for {net_id}, trying larger ROI radius={fallback_radius} (fallback {self.full_graph_fallback_count+1}/5)")

                if use_portals:
                    # Use portal nodes for larger ROI
                    src_pad_id, dst_pad_id = self.net_pad_ids[net_id]
                    src_portal = self.portals.get(src_pad_id)
                    dst_portal = self.portals.get(dst_pad_id)
                    if src_portal and dst_portal:
                        src_portal_node = self.lattice.node_idx(src_portal.x_idx, src_portal.y_idx, src_portal.pad_layer)
                        dst_portal_node = self.lattice.node_idx(dst_portal.x_idx, dst_portal.y_idx, dst_portal.pad_layer)
                        roi_nodes, global_to_roi = self.roi_extractor.extract_roi(
                            src_portal_node, dst_portal_node, initial_radius=fallback_radius, stagnation_bonus=roi_margin_bonus * 2
                        )
                        result = self.solver.find_path_multisource_multisink(
                            src_seeds, dst_targets, costs, roi_nodes, global_to_roi
                        )
                        if result:
                            path, entry_layer, exit_layer = result
                else:
                    # Use pad nodes for larger ROI
                    roi_nodes, global_to_roi = self.roi_extractor.extract_roi(
                        src, dst, initial_radius=fallback_radius, stagnation_bonus=roi_margin_bonus * 2
                    )
                    path = self.solver.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)

                self.full_graph_fallback_count += 1

            if path and len(path) > 1:
                edge_indices = self._path_to_edges(path)
                self.accounting.commit_path(edge_indices)  # bumps present for next nets
                self.net_paths[net_id] = path
                # Store portal entry/exit layers if using portals
                if use_portals and entry_layer is not None and exit_layer is not None:
                    self.net_portal_layers[net_id] = (entry_layer, exit_layer)
                # Update edge-to-nets tracking
                self._update_net_edge_tracking(net_id, edge_indices)
                routed_this_pass += 1
            else:
                failed_this_pass += 1
                self.net_paths[net_id] = []
                # Clear tracking for failed nets
                self._clear_net_edge_tracking(net_id)

                # Track portal failures and retarget if needed
                if cfg.portal_enabled and net_id in self.net_pad_ids:
                    self.net_portal_failures[net_id] += 1
                    if self.net_portal_failures[net_id] >= cfg.portal_retarget_patience:
                        # Retarget portals for this net
                        self._retarget_portals_for_net(net_id)
                        self.net_portal_failures[net_id] = 0  # Reset counter

        # Count total routed/failed across all nets
        total_routed = sum(1 for path in self.net_paths.values() if path)
        total_failed = len(all_tasks) - total_routed

        return total_routed, total_failed

    def _route_all_batched_gpu(self, ordered_nets: List[str], tasks: Dict, all_tasks: Dict,
                                costs, pres_fac: float, roi_margin_bonus: float) -> Tuple[int, int]:
        """Route nets in parallel batches using GPU"""
        import numpy as np

        cfg = self.config
        batch_size = min(cfg.batch_size, 64)  # Increased for speed
        total = len(ordered_nets)

        routed_this_pass = 0
        failed_this_pass = 0

        # Process nets in batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_nets = ordered_nets[batch_start:batch_end]

            if batch_start % 50 == 0:
                logger.info(f"  Routing nets {batch_start+1}-{batch_end}/{total} (batch of {len(batch_nets)})")

            # Prepare ROI batch
            roi_batch = []
            batch_metadata = []  # Track (net_id, use_portals, src, dst)

            for net_id in batch_nets:
                logger.info(f"[DEBUG] Extracting ROI for net {net_id}")
                src, dst = tasks[net_id]

                # Clear old path
                if net_id in self.net_paths and self.net_paths[net_id]:
                    if net_id in self._net_to_edges:
                        old_edges = self._net_to_edges[net_id]
                    else:
                        old_edges = self._path_to_edges(self.net_paths[net_id])
                    self.accounting.clear_path(old_edges)
                    self._clear_net_edge_tracking(net_id)

                # Calculate adaptive ROI
                src_x, src_y, src_z = self.lattice.idx_to_coord(src)
                dst_x, dst_y, dst_z = self.lattice.idx_to_coord(dst)
                manhattan_dist = abs(dst_x - src_x) + abs(dst_y - src_y)
                adaptive_radius = max(30, min(int(manhattan_dist * 1.2), 150))

                # Extract ROI (portal or normal)
                use_portals = cfg.portal_enabled and net_id in self.net_pad_ids

                if use_portals:
                    src_pad_id, dst_pad_id = self.net_pad_ids[net_id]
                    src_portal = self.portals.get(src_pad_id)
                    dst_portal = self.portals.get(dst_pad_id)

                    if src_portal and dst_portal:
                        # Use entry_layer (routing layer) not pad_layer (F.Cu)!
                        src_portal_node = self.lattice.node_idx(src_portal.x_idx, src_portal.y_idx, src_portal.entry_layer)
                        dst_portal_node = self.lattice.node_idx(dst_portal.x_idx, dst_portal.y_idx, dst_portal.entry_layer)
                        roi_nodes, global_to_roi = self.roi_extractor.extract_roi(
                            src_portal_node, dst_portal_node, initial_radius=adaptive_radius, stagnation_bonus=roi_margin_bonus
                        )
                    else:
                        use_portals = False
                        roi_nodes, global_to_roi = self.roi_extractor.extract_roi(
                            src, dst, initial_radius=adaptive_radius, stagnation_bonus=roi_margin_bonus
                        )
                else:
                    roi_nodes, global_to_roi = self.roi_extractor.extract_roi(
                        src, dst, initial_radius=adaptive_radius, stagnation_bonus=roi_margin_bonus
                    )

                # Build ROI CSR for this net
                roi_size = len(roi_nodes)
                logger.info(f"[DEBUG] ROI extracted for net {net_id}: {roi_size} nodes")
                roi_src = int(global_to_roi[src])
                roi_dst = int(global_to_roi[dst])

                if roi_src < 0 or roi_dst < 0:
                    batch_metadata.append((net_id, False, None, None, None))
                    continue

                # Extract ROI CSR subgraph
                roi_indptr, roi_indices, roi_weights = self.solver.gpu_solver._extract_roi_csr(
                    roi_nodes.get() if hasattr(roi_nodes, 'get') else roi_nodes,
                    global_to_roi.get() if hasattr(global_to_roi, 'get') else global_to_roi,
                    costs
                )

                roi_batch.append((roi_src, roi_dst, roi_indptr, roi_indices, roi_weights, roi_size))
                batch_metadata.append((net_id, use_portals, roi_nodes, src, dst))

            # Route entire batch on GPU in parallel
            if roi_batch:
                logger.info(f"[DEBUG] Starting GPU solve for batch of {len(roi_batch)} nets")
                paths = self.solver.gpu_solver.find_paths_on_rois(roi_batch)
                logger.info(f"[DEBUG] GPU solve complete, processing {len(paths)} results")

                # Process results
                for i, (net_id, use_portals, roi_nodes_arr, src, dst) in enumerate(batch_metadata):
                    if roi_nodes_arr is None or i >= len(paths):
                        logger.info(f"[DEBUG] Net {net_id}: FAILED (no ROI or path)")
                        failed_this_pass += 1
                        self.net_paths[net_id] = []
                        continue

                    local_path = paths[i]
                    if local_path and len(local_path) > 1:
                        # Convert local ROI path to global
                        roi_nodes_cpu = roi_nodes_arr.get() if hasattr(roi_nodes_arr, 'get') else roi_nodes_arr
                        global_path = [int(roi_nodes_cpu[node_idx]) for node_idx in local_path]

                        # Commit path
                        edge_indices = self._path_to_edges(global_path)
                        self.accounting.commit_path(edge_indices)
                        self.net_paths[net_id] = global_path
                        self._update_net_edge_tracking(net_id, edge_indices)
                        logger.info(f"[DEBUG] Net {net_id}: path length {len(global_path)}")
                        routed_this_pass += 1
                    else:
                        logger.info(f"[DEBUG] Net {net_id}: FAILED (no path found)")
                        failed_this_pass += 1
                        self.net_paths[net_id] = []
                        self._clear_net_edge_tracking(net_id)

        # Count total
        total_routed = sum(1 for path in self.net_paths.values() if path)
        total_failed = len(all_tasks) - total_routed

        return total_routed, total_failed

    def _retarget_portals_for_net(self, net_id: str):
        """Retarget portals when a net fails repeatedly"""
        if net_id not in self.net_pad_ids:
            return

        src_pad_id, dst_pad_id = self.net_pad_ids[net_id]

        # Retarget source portal
        if src_pad_id in self.portals:
            portal = self.portals[src_pad_id]
            # Try flipping direction first
            if portal.retarget_count == 0:
                portal.direction = -portal.direction
                portal.y_idx = portal.y_idx - 2 * portal.direction * portal.delta_steps  # Flip to other side
                portal.retarget_count += 1
                logger.debug(f"Retargeted portal for {src_pad_id}: flipped direction")
            # Then try different delta
            elif portal.retarget_count == 1:
                # Try delta closer to preferred
                new_delta = self.config.portal_delta_pref
                if new_delta != portal.delta_steps:
                    delta_change = new_delta - portal.delta_steps
                    portal.y_idx += portal.direction * delta_change
                    portal.delta_steps = new_delta
                    portal.retarget_count += 1
                    logger.debug(f"Retargeted portal for {src_pad_id}: adjusted delta to {new_delta}")

        # Retarget destination portal (same logic)
        if dst_pad_id in self.portals:
            portal = self.portals[dst_pad_id]
            if portal.retarget_count == 0:
                portal.direction = -portal.direction
                portal.y_idx = portal.y_idx - 2 * portal.direction * portal.delta_steps
                portal.retarget_count += 1
            elif portal.retarget_count == 1:
                new_delta = self.config.portal_delta_pref
                if new_delta != portal.delta_steps:
                    delta_change = new_delta - portal.delta_steps
                    portal.y_idx += portal.direction * delta_change
                    portal.delta_steps = new_delta
                    portal.retarget_count += 1

    def _rebuild_usage_from_committed_nets(self, keep_net_ids: Set[str]):
        """
        Rebuild present usage from scratch based on committed nets.
        Prevents ghost usage accumulation.
        """
        # Clear accounting
        self.accounting.canonical.clear()
        self.accounting.present.fill(0)

        # Rebuild from nets we're keeping
        for net_id in keep_net_ids:
            if net_id in self._net_to_edges:
                for ei in self._net_to_edges[net_id]:
                    self.accounting.canonical[ei] = self.accounting.canonical.get(ei, 0) + 1
                    self.accounting.present[ei] += 1

        logger.debug(f"[USAGE] Rebuilt from {len(keep_net_ids)} committed nets")

    def _update_net_edge_tracking(self, net_id: str, edge_indices: List[int]):
        """Update edge-to-nets tracking when a net is routed"""
        # Clear old tracking for this net
        self._clear_net_edge_tracking(net_id)

        # Store new edges for this net
        self._net_to_edges[net_id] = edge_indices

        # Update reverse mapping
        for ei in edge_indices:
            self._edge_to_nets[ei].add(net_id)

    def _clear_net_edge_tracking(self, net_id: str):
        """Clear edge-to-nets tracking for a net"""
        if net_id in self._net_to_edges:
            # Remove this net from all edge mappings
            for ei in self._net_to_edges[net_id]:
                self._edge_to_nets[ei].discard(net_id)
            del self._net_to_edges[net_id]

    def _build_hotset(self, tasks: Dict[str, Tuple[int, int]], ripped: Optional[Set[str]] = None) -> Set[str]:
        """
        Build hotset: ONLY nets touching overused edges, with adaptive capping.
        Prevents thrashing by limiting hotset size based on actual overuse.
        """
        if ripped is None:
            ripped = set()

        present = self.accounting.present.get() if self.accounting.use_gpu else self.accounting.present
        cap = self.accounting.capacity.get() if self.accounting.use_gpu else self.accounting.capacity
        over = np.maximum(0, present - cap)
        over_idx = set(map(int, np.where(over > 0)[0]))

        # NO OVERUSE CASE: only route unrouted nets
        if len(over_idx) == 0:
            unrouted = {nid for nid in tasks.keys() if not self.net_paths.get(nid)}
            hotset = unrouted | ripped
            logger.info(f"[HOTSET] no-overuse; unrouted={len(unrouted)} ripped={len(ripped)} → hotset={len(hotset)}")
            return hotset

        # OVERUSE EXISTS: collect nets touching overused edges using fast lookup
        offenders = set()
        for ei in over_idx:
            offenders.update(self._edge_to_nets.get(ei, set()))

        # Add ripped nets
        offenders |= ripped

        # Add unrouted nets (small priority, will be at end after sorting)
        unrouted = {nid for nid in tasks.keys() if not self.net_paths.get(nid)}

        # Score offenders by total overuse they contribute
        scores = []
        for net_id in offenders:
            if net_id in self._net_to_edges:
                impact = sum(float(over[ei]) for ei in self._net_to_edges[net_id] if ei in over_idx)
                scores.append((impact, net_id))

        # Add unrouted with low priority
        for net_id in unrouted:
            if net_id not in offenders:
                scores.append((0.0, net_id))

        # Sort by impact (highest first)
        scores.sort(reverse=True)

        # ADAPTIVE CAP: scale with number of overused edges, not fixed cap
        # Rule: allow ~2-4 nets per overused edge, but cap at config.hotset_cap
        adaptive_cap = min(self.config.hotset_cap, max(64, 3 * len(over_idx)))
        hotset = {nid for _, nid in scores[:adaptive_cap]}

        logger.info(f"[HOTSET] overuse_edges={len(over_idx)}, offenders={len(offenders)}, "
                    f"unrouted={len(unrouted)}, cap={adaptive_cap} → hotset={len(hotset)}/{len(tasks)}")

        return hotset

    def _log_top_overused_channels(self, over: np.ndarray, top_k: int = 10):
        """Log top-K overused channels with spatial info"""
        # Find top-K overused edges
        overused_edges = [(ei, float(over[ei])) for ei in range(len(over)) if over[ei] > 0]
        if not overused_edges:
            return

        overused_edges.sort(key=lambda x: x[1], reverse=True)
        top_edges = overused_edges[:top_k]

        logger.info(f"[INSTRUMENTATION] Top-{len(top_edges)} overused channels:")

        indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
        indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices

        for rank, (ei, overuse_val) in enumerate(top_edges, 1):
            # Find source node for this edge using binary search: O(log N) instead of O(N)
            u = int(np.searchsorted(indptr, ei, side='right') - 1)

            if 0 <= u < len(indptr) - 1 and indptr[u] <= ei < indptr[u + 1]:
                v = int(indices[ei])
                ux, uy, uz = self.lattice.idx_to_coord(u)
                vx, vy, vz = self.lattice.idx_to_coord(v)

                # Convert to mm for spatial context
                ux_mm, uy_mm = self.lattice.geom.lattice_to_world(ux, uy)
                vx_mm, vy_mm = self.lattice.geom.lattice_to_world(vx, vy)

                edge_type = "VIA" if uz != vz else "TRACK"
                layer_info = f"L{uz}" if uz == vz else f"L{uz}→L{vz}"

                # Count nets using this edge (use cached reverse lookup)
                nets_on_edge = len(self._edge_to_nets.get(ei, set()))

                logger.info(f"  {rank:2d}. {edge_type:5s} {layer_info:6s} "
                           f"({ux_mm:6.2f},{uy_mm:6.2f})→({vx_mm:6.2f},{vy_mm:6.2f}) "
                           f"overuse={overuse_val:.1f} nets={nets_on_edge}")

    def _rip_top_k_offenders(self, k=20) -> Set[str]:
        """
        Rip only the worst 16-24 nets to break stagnation (not the world).
        Respect locked nets - don't rip unless they touch new overuse.
        Returns the set of ripped net IDs.
        """
        present = self.accounting.present.get() if self.accounting.use_gpu else self.accounting.present
        cap = self.accounting.capacity.get() if self.accounting.use_gpu else self.accounting.capacity
        over = np.maximum(0, present - cap)
        over_idx = set(map(int, np.where(over > 0)[0]))

        # Score nets by impact on worst edges (use fast lookup)
        scores = []
        for net_id, path in self.net_paths.items():
            if not path or net_id in self.locked_nets:
                continue
            if net_id in self._net_to_edges:
                impact = sum(float(over[ei]) for ei in self._net_to_edges[net_id] if ei in over_idx)
                if impact > 0:
                    scores.append((impact, net_id))

        scores.sort(reverse=True)
        victims = {nid for _, nid in scores[:k]}

        for net_id in victims:
            if self.net_paths.get(net_id) and net_id in self._net_to_edges:
                # Use cached edges for efficiency
                self.accounting.clear_path(self._net_to_edges[net_id])
                self.net_paths[net_id] = []
                # Clear edge tracking for ripped nets
                self._clear_net_edge_tracking(net_id)
                # Reset clean streak so they can't immediately lock again
                self.net_clean_streak[net_id] = 0

        logger.info(f"[STAGNATION] Ripped {len(victims)} nets (locked={len(self.locked_nets)} preserved)")
        return victims

    def _apply_portal_discount(self):
        """Apply portal discount to span-1 vias adjacent to terminals"""
        if self.config.portal_discount >= 1.0:
            return  # No discount

        indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
        indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices
        base_costs = self.graph.base_costs.get() if hasattr(self.graph.base_costs, 'get') else self.graph.base_costs

        # Get terminal nodes
        terminal_nodes = set(self.pad_to_node.values())
        plane_size = self.lattice.x_steps * self.lattice.y_steps

        discount_count = 0
        for terminal in terminal_nodes:
            tz = terminal // plane_size
            # Find via edges from this terminal
            for ei in range(int(indptr[terminal]), int(indptr[terminal+1])):
                v = int(indices[ei])
                vz = v // plane_size
                span = abs(vz - tz)

                # Apply discount only to span-1 vias (adjacent layers)
                if span == 1 and self._via_edges[ei]:
                    base_costs[ei] *= self.config.portal_discount
                    discount_count += 1

        logger.info(f"Applied portal discount ({self.config.portal_discount}x) to {discount_count} escape vias")

    def _identify_via_edges(self):
        """Mark which edges are vias (vertical transitions between layers)"""
        indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
        indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices

        # Use numpy boolean array instead of Python set for memory efficiency
        # With 27M edges, this uses ~30MB instead of ~750MB
        num_edges = int(indptr[-1])
        self._via_edges = np.zeros(num_edges, dtype=bool)

        # Use arithmetic instead of idx_to_coord for speed
        plane_size = self.lattice.x_steps * self.lattice.y_steps

        for u in range(len(indptr) - 1):
            uz = u // plane_size  # Fast arithmetic instead of idx_to_coord
            for ei in range(int(indptr[u]), int(indptr[u+1])):
                v = int(indices[ei])
                vz = v // plane_size
                # Via edge: different layer (same x,y is implicit in Manhattan CSR construction)
                self._via_edges[ei] = (uz != vz)

        logger.info(f"Identified {int(self._via_edges.sum())} via edges")

    def _path_to_edges(self, node_path: List[int]) -> List[int]:
        """Nodes → edge indices via on-the-fly CSR scan (no dict needed)"""
        out = []
        indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
        indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices

        for u, v in zip(node_path, node_path[1:]):
            s, e = int(indptr[u]), int(indptr[u+1])
            # Linear scan over neighbors (small degree in Manhattan lattice, so fast)
            for ei in range(s, e):
                if int(indices[ei]) == v:
                    out.append(ei)
                    break
        return out

    def map_all_pads(self, board: Board) -> None:
        """Legacy API: pad mapping (already done in initialize_graph)"""
        logger.info(f"map_all_pads: Already mapped {len(self.pad_to_node)} pads")

    def prepare_routing_runtime(self):
        """Legacy API: prepare for routing (no-op, already ready)"""
        logger.info("prepare_routing_runtime: Ready")

    def _segment_world(self, a_idx: int, b_idx: int, layer: int, net: str):
        ax, ay, _ = self.lattice.idx_to_coord(a_idx)
        bx, by, _ = self.lattice.idx_to_coord(b_idx)
        (ax_mm, ay_mm) = self.lattice.geom.lattice_to_world(ax, ay)
        (bx_mm, by_mm) = self.lattice.geom.lattice_to_world(bx, by)
        return {
            'net': net,
            'layer': self.config.layer_names[layer] if layer < len(self.config.layer_names) else f"L{layer}",
            'x1': ax_mm, 'y1': ay_mm, 'x2': bx_mm, 'y2': by_mm,
            'width': self.config.grid_pitch * 0.6,
        }

    def precompute_all_pad_escapes(self, board: Board, nets_to_route: List = None) -> Tuple[List, List]:
        """
        Delegate to PadEscapePlanner for precomputing all pad escapes.

        Returns (tracks, vias) for visualization.
        """
        if not self.escape_planner:
            logger.error("Escape planner not initialized! Call initialize_graph first.")
            return ([], [])

        result = self.escape_planner.precompute_all_pad_escapes(board, nets_to_route)

        # CRITICAL: Copy portals from escape_planner to self.portals
        # The pathfinder needs these to route from portal positions, not pad positions
        self.portals = self.escape_planner.portals.copy()
        logger.info(f"Copied {len(self.portals)} portals from escape planner to pathfinder")

        return result

    def _via_world(self, at_idx: int, net: str, from_layer: int, to_layer: int):
        x, y, _ = self.lattice.idx_to_coord(at_idx)
        (x_mm, y_mm) = self.lattice.geom.lattice_to_world(x, y)
        return {
            'net': net,
            'x': x_mm, 'y': y_mm,
            'from_layer': self.config.layer_names[from_layer] if from_layer < len(self.config.layer_names) else f"L{from_layer}",
            'to_layer': self.config.layer_names[to_layer] if to_layer < len(self.config.layer_names) else f"L{to_layer}",
            'diameter': 0.25,  # hole (0.15) + 2×annular (0.05) = 0.25mm
            'drill': 0.15,     # hole diameter
        }

    def emit_geometry(self, board: Board) -> Tuple[int, int]:
        """
        Convert routed node paths into drawable segments and vias.
        - Clean geometry (for KiCad export): only if overuse == 0
        - Provisional geometry (for GUI feedback): always generated
        """
        # Check for overuse
        over_sum, over_cnt = self.accounting.compute_overuse()

        # Generate provisional geometry for GUI feedback (always)
        provisional_tracks, provisional_vias = self._generate_geometry_from_paths()
        self._provisional_geometry = GeometryPayload(provisional_tracks, provisional_vias)

        if over_sum > 0:
            logger.warning(f"[EMIT] Overuse={over_sum}: emitting provisional geometry only (not exported)")
            self._geometry_payload = GeometryPayload([], [])  # No clean geometry
            return (0, 0)

        # No overuse: emit clean geometry for export
        logger.info("[EMIT] Converting to clean geometry")
        self._geometry_payload = GeometryPayload(provisional_tracks, provisional_vias)
        return (len(provisional_tracks), len(provisional_vias))

    def _generate_geometry_from_paths(self) -> Tuple[List, List]:
        """Generate tracks and vias from net_paths"""
        tracks, vias = [], []

        for net_id, path in self.net_paths.items():
            if not path:
                continue

            # NOTE: Escape geometry is pre-computed by PadEscapePlanner and already
            # added to board_data in main_window.py. No need to emit it here.

            # Generate tracks/vias from main path
            run_start = path[0]
            prev = path[0]
            prev_dir = None
            prev_layer = self.lattice.idx_to_coord(prev)[2]

            for node in path[1:]:
                x0, y0, z0 = self.lattice.idx_to_coord(prev)
                x1, y1, z1 = self.lattice.idx_to_coord(node)

                if z1 != z0:
                    # flush any pending straight run before via
                    if prev != run_start:
                        tracks.append(self._segment_world(run_start, prev, prev_layer, net_id))
                    vias.append(self._via_world(prev, net_id, z0, z1))
                    run_start = node
                    prev_dir = None
                else:
                    dir_vec = (np.sign(x1 - x0), np.sign(y1 - y0))
                    if prev_dir is None or dir_vec == prev_dir:
                        # keep extending run
                        pass
                    else:
                        # direction changed: flush previous run
                        tracks.append(self._segment_world(run_start, prev, prev_layer, net_id))
                        run_start = prev
                    prev_dir = dir_vec

                prev = node
                prev_layer = z1

            # flush final run
            if prev != run_start:
                tracks.append(self._segment_world(run_start, prev, prev_layer, net_id))

        return (tracks, vias)

    def get_geometry_payload(self):
        """Get clean geometry (only if no overuse)"""
        return self._geometry_payload

    def get_provisional_geometry(self):
        """Get provisional geometry for GUI feedback (always available)"""
        return self._provisional_geometry


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY API
# ═══════════════════════════════════════════════════════════════════════════════

UnifiedPathFinder = PathFinderRouter

logger.info(f"PathFinder loaded (GPU={'YES' if GPU_AVAILABLE else 'NO'})")
